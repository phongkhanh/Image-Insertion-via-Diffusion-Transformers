import os
import sys
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageFilter, ImageDraw
from huggingface_hub import snapshot_download
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
import math
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask
from src.models.harmonizer import MultiStageHarmonizer
from src.models.transformer import build_inference_forward


dtype = torch.bfloat16
size = (768, 768)

LORA_PATH    = "runs/20260604-030052/ckpt/4000"
BASELINE_PATH = "runs/20260603-125049/ckpt/final"
AVAILABLE_MODELS = {"My Model": LORA_PATH, "Baseline": BASELINE_PATH}

# ── My Model on cuda:0 ────────────────────────────────────────────────────────
print("[Model] Loading My Model on cuda:0 ...")
pipe_my = FluxFillPipeline.from_pretrained("checkpoint/FLUX.1-Fill-dev", torch_dtype=dtype).to("cuda:0")
pipe_my.load_lora_weights(LORA_PATH)
_harmonizer_path = os.path.join(LORA_PATH, "harmonizer.pt")
if os.path.isfile(_harmonizer_path):
    _harmonizer = MultiStageHarmonizer.load_for_inference(_harmonizer_path, "cuda:0", dtype)
    pipe_my.transformer.forward = build_inference_forward(pipe_my.transformer, _harmonizer)
    print(f"[Harmonizer] Loaded")
else:
    print(f"[Harmonizer] Not found — baseline mode")
redux_my = FluxPriorReduxPipeline.from_pretrained("checkpoint/FLUX.1-Redux-dev").to(dtype=dtype).to("cuda:0")

# ── Baseline on cuda:1 ────────────────────────────────────────────────────────
print("[Model] Loading Baseline on cuda:2 ...")
pipe_bl = FluxFillPipeline.from_pretrained("checkpoint/FLUX.1-Fill-dev", torch_dtype=dtype).to("cuda:2")
pipe_bl.load_lora_weights(BASELINE_PATH)
redux_bl = FluxPriorReduxPipeline.from_pretrained("checkpoint/FLUX.1-Redux-dev").to(dtype=dtype).to("cuda:2")

print("[Model] Both models ready.")

_current_model_label = "My Model"


def load_model(model_label):
    global _current_model_label
    if model_label not in AVAILABLE_MODELS:
        raise gr.Error(f"Model not found: {model_label}")
    _current_model_label = model_label
    print(f"[Model] Active: {model_label}")
    return f"Active: {model_label}"


MODEL_SEEDS = {"My Model": 42, "Baseline": 42}

def _get_pipe_redux_device():
    if _current_model_label == "My Model":
        return pipe_my, redux_my, "cuda:0"
    return pipe_bl, redux_bl, "cuda:2"

# ── SAM ──────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "/data2/data_fusion/work/journal/module_3/VLM/segment_anything_src/sam_vit_h_4b8939.pth"
sam_predictor = None
try:
    from segment_anything import sam_model_registry, SamPredictor
    if os.path.isfile(SAM_CHECKPOINT):
        _sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        _sam.to(device="cuda:0")
        sam_predictor = SamPredictor(_sam)
        print(f"[SAM] Loaded from {SAM_CHECKPOINT}")
    else:
        print(f"[SAM] Checkpoint not found at {SAM_CHECKPOINT} — SAM BBox disabled")
except ImportError:
    print("[SAM] segment_anything not installed — SAM BBox disabled")
# ─────────────────────────────────────────────────────────────────────────────


###   example  #####
ref_dir='./examples/ref_image'
ref_mask_dir='./examples/ref_mask'
image_dir='./examples/source_image'
image_mask_dir='./examples/source_mask'

ref_list=[os.path.join(ref_dir,file) for file in os.listdir(ref_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file ]
ref_list.sort()

ref_mask_list=[os.path.join(ref_mask_dir,file) for file in os.listdir(ref_mask_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file]
ref_mask_list.sort()

image_list=[os.path.join(image_dir,file) for file in os.listdir(image_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file ]
image_list.sort()

image_mask_list=[os.path.join(image_mask_dir,file) for file in os.listdir(image_mask_dir) if '.jpg' in file or '.png' in file or '.jpeg' in file]
image_mask_list.sort()
###   example  #####




def _empty_bbox():
    return {"clicks": 0, "x1": 0, "y1": 0, "x2": 0, "y2": 0}


def _draw_bbox_viz(image_pil, bbox_state):
    """Draw current bbox state on image for visualization."""
    if image_pil is None:
        return None
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(5, min(img.width, img.height) // 80)
    clicks = bbox_state.get("clicks", 0)
    if clicks >= 1:
        x1, y1 = bbox_state["x1"], bbox_state["y1"]
        draw.ellipse([x1-r, y1-r, x1+r, y1+r], fill=(0, 255, 0), outline=(0, 180, 0))
    if clicks >= 2:
        bx1 = min(bbox_state["x1"], bbox_state["x2"])
        by1 = min(bbox_state["y1"], bbox_state["y2"])
        bx2 = max(bbox_state["x1"], bbox_state["x2"])
        by2 = max(bbox_state["y1"], bbox_state["y2"])
        draw.rectangle([bx1, by1, bx2, by2], outline=(0, 255, 0), width=3)
    return img


def on_bbox_click(original_image_pil, bbox_state, evt: gr.SelectData):
    """Handle 2-click bbox selection on SAM display image."""
    x, y = evt.index
    if bbox_state.get("clicks", 0) % 2 == 0:
        new_state = {"clicks": 1, "x1": x, "y1": y, "x2": x, "y2": y}
    else:
        new_state = {**bbox_state, "clicks": 2, "x2": x, "y2": y}
    return _draw_bbox_viz(original_image_pil, new_state), new_state


def sam_predict_from_bbox(image_pil, bbox):
    if sam_predictor is None:
        raise gr.Error("SAM is not available. Install segment-anything and download sam_vit_h_4b8939.pth.")
    image_np = np.array(image_pil.convert("RGB"))
    sam_predictor.set_image(image_np)
    masks, scores, _ = sam_predictor.predict(box=bbox[None, :], multimask_output=True)
    best = masks[np.argmax(scores)]
    return Image.fromarray((best * 255).astype(np.uint8))


def generate_sam_mask(original_image_pil, bbox_state):
    """Generate SAM mask from 2-click bbox state. Returns (preview, mask_for_state)."""
    if original_image_pil is None:
        raise gr.Error("Please upload an image first.")
    if bbox_state.get("clicks", 0) < 2:
        raise gr.Error("Please click 2 points on the image to define the bounding box.")
    x1 = min(bbox_state["x1"], bbox_state["x2"])
    y1 = min(bbox_state["y1"], bbox_state["y2"])
    x2 = max(bbox_state["x1"], bbox_state["x2"])
    y2 = max(bbox_state["y1"], bbox_state["y2"])
    mask_pil = sam_predict_from_bbox(original_image_pil, np.array([x1, y1, x2, y2]))
    return mask_pil, mask_pil


def extract_image_from_editor(editor_data):
    """Extract background PIL image from ImageEditor data."""
    if editor_data is None:
        return None, _empty_bbox()
    return editor_data.get("background"), _empty_bbox()


def run_local(base_image, base_mask, reference_image, ref_mask, base_mask_option, ref_mask_option, bg_sam_mask, ref_sam_mask):

    tar_image = base_image["background"] if base_image else None
    ref_image = reference_image["background"] if reference_image else None

    if tar_image is None:
        raise gr.Error("Please upload a Background Image.")
    if ref_image is None:
        raise gr.Error("Please upload a Reference Image.")

    if base_mask_option == "Draw Mask":
        tar_mask = base_image["layers"][0]
    elif base_mask_option == "SAM BBox":
        if bg_sam_mask is None:
            raise gr.Error("Please click 'Generate SAM Mask' for the Background Image first.")
        tar_mask = bg_sam_mask
    else:
        tar_mask = base_mask["background"] if base_mask else None

    if ref_mask_option == "Draw Mask":
        ref_mask = reference_image["layers"][0]
    elif ref_mask_option == "SAM BBox":
        if ref_sam_mask is None:
            raise gr.Error("Please click 'Generate SAM Mask' for the Reference Image first.")
        ref_mask = ref_sam_mask
    else:
        ref_mask = ref_mask["background"] if ref_mask else None

    if tar_mask is None:
        raise gr.Error("Please upload a Background Mask.")
    if ref_mask is None:
        raise gr.Error("Please upload a Reference Mask.")

    tar_image = tar_image.convert("RGB")
    tar_mask = tar_mask.convert("L")
    ref_image = ref_image.convert("RGB")
    ref_mask = ref_mask.convert("L")

    tar_image = np.asarray(tar_image)
    tar_mask = np.asarray(tar_mask)
    tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    if tar_mask.sum() == 0:
        raise gr.Error('No mask for the background image.Please check mask button!')

    if ref_mask.sum() == 0:
        raise gr.Error('No mask for the reference image.Please check mask button!')

    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2] 
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)


    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 

    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

    # zome in
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=2)    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop


    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2,x1:x2,:]
    tar_mask = tar_mask[y1:y2,x1:x2]

    H1, W1 = tar_image.shape[0], tar_image.shape[1]
    # zome in


    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)

    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    _pipe, _redux, _device = _get_pipe_redux_device()
    pipe_prior_output = _redux(Image.fromarray(masked_ref_image))


    tar_image = pad_to_square(tar_image, pad_value=255)

    H2, W2 = tar_image.shape[0], tar_image.shape[1]

    tar_image = cv2.resize(tar_image, size)
    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)


    tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)
    mask_black = np.ones_like(tar_image) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)


    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)



    generator = torch.Generator(_device).manual_seed(MODEL_SEEDS[_current_model_label])
    edited_image = _pipe(
        image=diptych_ref_tar,
        mask_image=mask_diptych,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        max_sequence_length=512,
        num_inference_steps=50,
        generator=generator,
        **pipe_prior_output,
    ).images[0]



    width, height = edited_image.size
    left = width // 2
    right = width
    top = 0
    bottom = height
    edited_image = edited_image.crop((left, top, right, bottom))


    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
    edited_image = Image.fromarray(edited_image)


    return [edited_image]

with gr.Blocks() as demo:

    gr.Markdown("# Insert-Anything")
    gr.Markdown("### Select mask input method for each image. For **SAM BBox**: upload image → click 2 points → Generate SAM Mask → Run.")

    # ── States ────────────────────────────────────────────────────────────────
    bg_original_state  = gr.State(None)
    bg_bbox_state      = gr.State(_empty_bbox())
    bg_sam_mask_state  = gr.State(None)
    ref_original_state = gr.State(None)
    ref_bbox_state     = gr.State(_empty_bbox())
    ref_sam_mask_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):

            # ── Background ────────────────────────────────────────────────────
            with gr.Row():
                base_image = gr.ImageEditor(
                    label="Background Image", sources="upload", type="pil",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=30, color_mode="fixed"),
                    layers=False, interactive=True)
                base_mask = gr.ImageEditor(
                    label="Background Mask", sources="upload", type="pil",
                    layers=False, brush=False, eraser=False)

            # SAM BBox components for background (hidden by default)
            with gr.Row():
                bg_sam_display = gr.Image(
                    label="Background — Click point 1 then point 2 to draw BBox",
                    interactive=True, visible=False)
                bg_mask_preview = gr.Image(
                    label="SAM Mask Preview", interactive=False, visible=False)
            with gr.Row():
                bg_generate_btn = gr.Button("Generate SAM Mask", visible=False)

            with gr.Row():
                base_mask_option = gr.Radio(
                    ["Draw Mask", "Upload with Mask", "SAM BBox"],
                    label="Background Mask Option", value="Upload with Mask")

            # ── Reference ─────────────────────────────────────────────────────
            with gr.Row():
                ref_image = gr.ImageEditor(
                    label="Reference Image", sources="upload", type="pil",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=30, color_mode="fixed"),
                    layers=False, interactive=True)
                ref_mask = gr.ImageEditor(
                    label="Reference Mask", sources="upload", type="pil",
                    layers=False, brush=False, eraser=False)

            # SAM BBox components for reference (hidden by default)
            with gr.Row():
                ref_sam_display = gr.Image(
                    label="Reference — Click point 1 then point 2 to draw BBox",
                    interactive=True, visible=False)
                ref_mask_preview = gr.Image(
                    label="SAM Mask Preview", interactive=False, visible=False)
            with gr.Row():
                ref_generate_btn = gr.Button("Generate SAM Mask", visible=False)

            with gr.Row():
                ref_mask_option = gr.Radio(
                    ["Draw Mask", "Upload with Mask", "SAM BBox"],
                    label="Reference Mask Option", value="Upload with Mask")

        with gr.Column(scale=1):
            baseline_gallery = gr.Gallery(
                label='Output', show_label=True, elem_id="gallery",
                height=701, columns=1, object_fit="contain")
            with gr.Accordion("Advanced Option", open=True):
                gr.Markdown("---")
                gr.Markdown("### Model")
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=_current_model_label,
                    label="LoRA Weights",
                    interactive=True)
                load_model_btn = gr.Button("Load Model")
                model_status = gr.Textbox(
                    value=f"Loaded: {_current_model_label}",
                    label="Model Status", interactive=False, max_lines=1)

    run_local_button = gr.Button(value="Run")

    # #### example #####
    num_examples = len(image_list)
    for i in range(num_examples):
        with gr.Row():
            if i == 0:
                gr.Examples([image_list[i]], inputs=[base_image], label="Examples - Background Image", examples_per_page=1)
                gr.Examples([image_mask_list[i]], inputs=[base_mask], label="Examples - Background Mask", examples_per_page=1)
                gr.Examples([ref_list[i]], inputs=[ref_image], label="Examples - Reference Object", examples_per_page=1)
                gr.Examples([ref_mask_list[i]], inputs=[ref_mask], label="Examples - Reference Mask", examples_per_page=1)
            else:
                gr.Examples([image_list[i]], inputs=[base_image], examples_per_page=1, label="")
                gr.Examples([image_mask_list[i]], inputs=[base_mask], examples_per_page=1, label="")
                gr.Examples([ref_list[i]], inputs=[ref_image], examples_per_page=1, label="")
                gr.Examples([ref_mask_list[i]], inputs=[ref_mask], examples_per_page=1, label="")
        if i < num_examples - 1:
            gr.HTML("<hr>")
    # #### example #####

    # ── Event wiring ──────────────────────────────────────────────────────────

    def _bg_mode_change(option):
        show_mask   = option in ("Draw Mask", "Upload with Mask")
        show_sam    = option == "SAM BBox"
        return (gr.update(visible=show_mask),   # base_mask
                gr.update(visible=show_sam),    # bg_sam_display
                gr.update(visible=show_sam),    # bg_mask_preview
                gr.update(visible=show_sam))    # bg_generate_btn

    def _ref_mode_change(option):
        show_mask   = option in ("Draw Mask", "Upload with Mask")
        show_sam    = option == "SAM BBox"
        return (gr.update(visible=show_mask),
                gr.update(visible=show_sam),
                gr.update(visible=show_sam),
                gr.update(visible=show_sam))

    base_mask_option.change(
        fn=_bg_mode_change, inputs=[base_mask_option],
        outputs=[base_mask, bg_sam_display, bg_mask_preview, bg_generate_btn])

    ref_mask_option.change(
        fn=_ref_mode_change, inputs=[ref_mask_option],
        outputs=[ref_mask, ref_sam_display, ref_mask_preview, ref_generate_btn])

    # When image is uploaded → copy to SAM display and reset bbox
    base_image.change(
        fn=extract_image_from_editor, inputs=[base_image],
        outputs=[bg_original_state, bg_bbox_state])
    base_image.change(
        fn=lambda d: d.get("background") if d else None, inputs=[base_image],
        outputs=[bg_sam_display])

    ref_image.change(
        fn=extract_image_from_editor, inputs=[ref_image],
        outputs=[ref_original_state, ref_bbox_state])
    ref_image.change(
        fn=lambda d: d.get("background") if d else None, inputs=[ref_image],
        outputs=[ref_sam_display])

    # 2-click bbox on SAM display
    bg_sam_display.select(
        fn=on_bbox_click, inputs=[bg_original_state, bg_bbox_state],
        outputs=[bg_sam_display, bg_bbox_state])

    ref_sam_display.select(
        fn=on_bbox_click, inputs=[ref_original_state, ref_bbox_state],
        outputs=[ref_sam_display, ref_bbox_state])

    # Generate SAM mask buttons
    bg_generate_btn.click(
        fn=generate_sam_mask, inputs=[bg_original_state, bg_bbox_state],
        outputs=[bg_mask_preview, bg_sam_mask_state])

    ref_generate_btn.click(
        fn=generate_sam_mask, inputs=[ref_original_state, ref_bbox_state],
        outputs=[ref_mask_preview, ref_sam_mask_state])

    # Load model
    load_model_btn.click(
        fn=load_model, inputs=[model_dropdown], outputs=[model_status])

    # Run inference
    run_local_button.click(
        fn=run_local,
        inputs=[base_image, base_mask, ref_image, ref_mask,
                base_mask_option, ref_mask_option,
                bg_sam_mask_state, ref_sam_mask_state],
        outputs=[baseline_gallery])

demo.queue()
demo.launch(share=True)