import os
from typing import Optional

import lightning as L
from diffusers.pipelines import FluxFillPipeline, FluxPriorReduxPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict

import prodigyopt
from PIL import Image
from .harmonizer import MultiStageHarmonizer
from .transformer import tranformer_forward
from .pipeline_tools import encode_images, prepare_text_input, Flux_fill_encode_masks_images
from .image_project import image_output

class InsertAnything(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        flux_redux_id: str, 
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe: FluxFillPipeline = (
            FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)
        )

        self.flux_redux: FluxPriorReduxPipeline = (
            FluxPriorReduxPipeline.from_pretrained(flux_redux_id).to(dtype=dtype).to(device)
        )

        self.flux_redux.image_embedder.requires_grad_(False).eval()
        self.flux_redux.image_encoder.requires_grad_(False).eval()


        self.transformer = self.flux_fill_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_fill_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_fill_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_fill_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        # ── Scene-Aware Reference Harmonizer ─────────────────────────────────
        # Instantiated AFTER init_lora so PEFT's add_adapter() cannot
        # accidentally wrap any of the harmonizer's linear layers.
        # Instantiated BEFORE self.to(device).to(dtype) so it is moved
        # automatically along with the rest of the LightningModule.
        #
        # When use_scene_harmonizer is False the attribute is set to None and
        # every downstream call site is a zero-cost no-op.
        if self.model_config.get("use_scene_harmonizer", False):
            self.harmonizer: Optional[MultiStageHarmonizer] = MultiStageHarmonizer(
                hidden_dim=3072,
                num_heads=24,
                head_dim=128,
            )
        else:
            self.harmonizer: Optional[MultiStageHarmonizer] = None

        # ── Harmonizer trainability report (runs once at init) ────────────────
        # Printed before self.to(dtype) so values reflect float32 init state.
        # requires_grad is unaffected by dtype conversion.
        if self.harmonizer is not None:
            total_params = sum(p.numel() for p in self.harmonizer.parameters())
            print("[Harmonizer V3 Init]")
            print(f"  inject_at={self.harmonizer.inject_at}")
            print(f"  params={total_params}")
            print(f"  alpha_0={self.harmonizer.alpha_0.item():.1f}")
            print(f"  alpha_1={self.harmonizer.alpha_1.item():.1f}")
            print(f"  alpha_2={self.harmonizer.alpha_2.item():.1f}")
            print(f"  diag_gamma_norm={self.harmonizer.diag_gamma.norm().item():.1f}")
            print(f"  diag_beta_norm={self.harmonizer.diag_beta.norm().item():.1f}")
            for name, param in self.harmonizer.named_parameters():
                print(f"  {name}.requires_grad={param.requires_grad}")

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def save_harmonizer(self, path: str) -> None:
        """Save MultiStageHarmonizer weights to <path>/harmonizer.pt.

        Safe to call when use_scene_harmonizer is False — becomes a no-op.
        Does not affect LoRA weights or the FluxFillPipeline state dict.

        Args:
            path: Directory where harmonizer.pt will be written.
                  Created automatically if it does not exist.
        """
        if self.harmonizer is None:
            return
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "harmonizer.pt")
        torch.save(self.harmonizer.state_dict(), save_path)

    def load_harmonizer(self, path: str) -> None:
        """Load MultiStageHarmonizer weights from <path>/harmonizer.pt.

        Must be called after InsertAnything is constructed with
        use_scene_harmonizer: true.  Raises RuntimeError if the harmonizer
        was not initialized (i.e., config flag is False).

        Args:
            path: Directory that contains harmonizer.pt.
        """
        if self.harmonizer is None:
            raise RuntimeError(
                "Cannot load harmonizer weights: harmonizer is not initialized. "
                "Set model.use_scene_harmonizer: true in the config before "
                "constructing InsertAnything."
            )
        ckpt_path = os.path.join(path, "harmonizer.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"harmonizer.pt not found at '{ckpt_path}'. "
                "Verify that the checkpoint directory contains a harmonizer.pt file."
            )
        state = torch.load(ckpt_path, map_location=self.device)
        # strict=False: V2 checkpoints (no diag_gamma/diag_beta) load cleanly.
        self.harmonizer.load_state_dict(state, strict=False)
        self.harmonizer.to(device=self.device, dtype=self.dtype)

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # LoRA parameters (re-enabled below).
        self.trainable_params = self.lora_layers

        # Harmonizer parameters are full (non-LoRA) trainable weights.
        # They live outside self.transformer so requires_grad_(False) above
        # did not touch them.  We still add them explicitly so the optimizer
        # receives them.
        if self.harmonizer is not None:
            self.trainable_params = (
                self.trainable_params + list(self.harmonizer.parameters())
            )

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        

        imgs = batch["result"]

        src = batch["src"]
        mask = batch["mask"]

        ref = batch["ref"]

        prompt_embeds = []
        pooled_prompt_embeds = []

        for i in range(ref.shape[0]):

            image_tensor = ref[i].cpu()

            image_tensor = image_tensor.permute(1, 2, 0)

            image_numpy = image_tensor.numpy()

            pil_image = Image.fromarray((image_numpy * 255).astype('uint8'))

            prompt_embed, pooled_prompt_embed = image_output(self.flux_redux, pil_image, self.device)


            prompt_embeds.append(prompt_embed.squeeze(1))
        
            pooled_prompt_embeds.append(pooled_prompt_embed.squeeze(1))     


        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0)

        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_fill_pipe, prompt_embeds=prompt_embeds.to(self.device), pooled_prompt_embeds=pooled_prompt_embeds.to(self.device)
        )

        # Prepare inputs
        with torch.no_grad():


            # Prepare image input
            x_0, img_ids = encode_images(self.flux_fill_pipe, imgs)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare conditions
            src_latents, mask_latents = Flux_fill_encode_masks_images(self.flux_fill_pipe, src, mask)

            condition_latents = torch.cat((src_latents, mask_latents), dim=-1)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            model_config=self.model_config,
            # Pass harmonizer (None when use_scene_harmonizer: false → no-op).
            harmonizer=self.harmonizer,
            hidden_states=torch.cat((x_t, condition_latents), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss






