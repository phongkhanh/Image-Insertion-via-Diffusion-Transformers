"""
Scene-Aware Reference Harmonization module — Version 3.

Applies repeated cross-attention (reference tokens as Q, scene tokens as K/V)
at three points inside the dual-block stack: after blocks 6, 12, and 18.

All three injection points share one set of Q/K/V/out projection weights,
one LayerNorm, and one pair of channel-wise FiLM parameters (diag_gamma,
diag_beta).  Each point has its own learnable scalar gate (alpha_0 … alpha_2).
All learnable parameters are initialized to zero → exact identity at step 0.

Architecture per stage:
    ref_normed   = LayerNorm(encoder_hidden_states)
    delta        = CrossAttention(Q=ref_normed, K=hidden_states, V=hidden_states)

    # Channel-wise FiLM: collapse token dim → per-channel modulation
    delta_pooled = delta.mean(dim=1)                      # (B, D)
    gamma        = diag_gamma * delta_pooled              # (B, D)
    beta         = diag_beta  * delta_pooled              # (B, D)

    output = encoder_hidden_states + alpha_i * (
                 delta
                 + gamma.unsqueeze(1) * encoder_hidden_states
                 + beta.unsqueeze(1)
             )

    When diag_gamma = diag_beta = 0  →  output = ref + alpha_i * delta  (V2)
    When alpha_i = 0                 →  output = ref                     (identity)

New parameters vs V2:
    diag_gamma : Parameter(hidden_dim,)  init=0   3 072 scalars
    diag_beta  : Parameter(hidden_dim,)  init=0   3 072 scalars
    net increase: 6 144 parameters

V2 checkpoint compatibility: load with strict=False; diag_gamma/diag_beta
default to zero → first forward pass is identical to V2.

Tensor conventions (Flux Fill configuration):
    B          : batch size
    N_ref      : number of reference tokens  (512 + N_redux ≈ 1241)
    N_img      : number of image tokens      (48 × 96 = 4608, diptych 1536×768)
    hidden_dim : 3072  (= num_heads × head_dim = 24 × 128)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageHarmonizer(nn.Module):
    """
    Repeated cross-attention from reference tokens (Q) to scene tokens (K, V).

    Injected after dual blocks 6, 12, and 18.  Projections are shared across
    all three stages; each stage has its own learnable gate.

    Args:
        hidden_dim: Transformer hidden dimension (num_heads × head_dim).
        num_heads:  Number of attention heads.
        head_dim:   Dimension per head.
    """

    inject_at: list[int] = [6, 12, 18]

    def __init__(
        self,
        hidden_dim: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
    ) -> None:
        super().__init__()

        if hidden_dim != num_heads * head_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal "
                f"num_heads ({num_heads}) × head_dim ({head_dim}). "
                f"Got product {num_heads * head_dim}."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Shared pre-norm on the query (reference) stream.
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)

        # Shared Q / K / V / out projections, no bias — consistent with Flux.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Per-stage learnable gates, all zero-initialized → identity at step 0.
        self.alpha_0 = nn.Parameter(torch.zeros(1))
        self.alpha_1 = nn.Parameter(torch.zeros(1))
        self.alpha_2 = nn.Parameter(torch.zeros(1))

        # Channel-wise FiLM parameters (V3), shared across all three stages.
        # Both zero-initialized:
        #   gamma = diag_gamma * delta_pooled  →  starts at 0  →  no FiLM effect
        #   beta  = diag_beta  * delta_pooled  →  starts at 0  →  no FiLM effect
        # When diag_gamma = diag_beta = 0 the forward pass reduces exactly to V2.
        self.diag_gamma = nn.Parameter(torch.zeros(hidden_dim))
        self.diag_beta  = nn.Parameter(torch.zeros(hidden_dim))

        # Per-stage delta magnitudes written by forward(), read by the callback.
        # Plain Python list — not a parameter, not a buffer, not in state_dict.
        self._delta_mag: list = [None, None, None]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        """
        Harmonize reference features by attending over scene features.

        Args:
            encoder_hidden_states: Reference token features.
                Shape: (B, N_ref, hidden_dim)
            hidden_states: Image/scene token features.
                Shape: (B, N_img, hidden_dim)
            stage: Which injection point (0, 1, or 2).

        Returns:
            Harmonized reference features.
                Shape: (B, N_ref, hidden_dim)
        """
        alpha = (self.alpha_0, self.alpha_1, self.alpha_2)[stage]

        B, N_ref, D = encoder_hidden_states.shape
        _, N_img, _ = hidden_states.shape

        ref_normed = self.norm(encoder_hidden_states)

        q = self.q_proj(ref_normed)       # (B, N_ref, D)
        k = self.k_proj(hidden_states)    # (B, N_img, D)
        v = self.v_proj(hidden_states)    # (B, N_img, D)

        q = q.view(B, N_ref, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_img, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_img, self.num_heads, self.head_dim).transpose(1, 2)

        # No RoPE: reference and image tokens live in different coordinate
        # spaces, so a shared positional bias would be semantically incorrect.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )  # (B, heads, N_ref, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(B, N_ref, D)
        delta = self.out_proj(attn_out)

        # Record magnitude for logging. .detach().item() leaves the autograd
        # graph untouched and produces a plain Python float — zero overhead.
        self._delta_mag[stage] = delta.detach().abs().mean().item()

        # ── Channel-wise FiLM correction (V3) ────────────────────────────────
        # Collapse token dimension → global scene descriptor (B, D).
        # Per-channel scale/shift derived from that descriptor.
        # Both diag_gamma and diag_beta are zero at init → film_correction = 0
        # → this branch adds nothing until the parameters learn.
        delta_pooled    = delta.mean(dim=1)                             # (B, D)
        gamma           = self.diag_gamma * delta_pooled                # (B, D)
        beta            = self.diag_beta  * delta_pooled                # (B, D)
        film_correction = (
            gamma.unsqueeze(1) * encoder_hidden_states                  # (B, N_ref, D)
            + beta.unsqueeze(1)                                         # (B, N_ref, D)
        )

        # delta is retained in the gated sum so alpha_i receives a non-zero
        # gradient from step 1 (via delta, as in V2).  diag_gamma and
        # diag_beta start receiving gradients once alpha_i is non-zero.
        return encoder_hidden_states + alpha * (delta + film_correction)

    # ------------------------------------------------------------------
    # Inference loading
    # ------------------------------------------------------------------

    @classmethod
    def load_for_inference(
        cls,
        path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "MultiStageHarmonizer":
        """Load a trained MultiStageHarmonizer for inference.

        Args:
            path:   Full path to harmonizer.pt saved by save_harmonizer().
            device: Target device, e.g. torch.device("cuda").
            dtype:  Target dtype, e.g. torch.bfloat16.

        Returns:
            MultiStageHarmonizer ready for inference (no gradients needed).
        """
        instance = cls(hidden_dim=3072, num_heads=24, head_dim=128)
        state = torch.load(path, map_location=device, weights_only=True)
        # strict=False: V2 checkpoints (no diag_gamma/diag_beta keys) load
        # cleanly; missing keys default to zero-init → exact V2 behavior.
        instance.load_state_dict(state, strict=False)
        instance.to(device=device, dtype=dtype)
        instance.eval()
        return instance
