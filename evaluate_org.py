import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from skimage.metrics import structural_similarity as ssim
import lpips
import timm
from cleanfid import fid
from transformers import CLIPModel, CLIPProcessor
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask

# ── Config ───────────────────────────────────────────────────────────────────
LORA_PATH           = "checkpoint/20250321_steps5000_pytorch_lora_weights.safetensors"
TEST_ROOT           = "/data1/stage/navsim_workspace/AnyInsertion/data_training_mask_prompt/test"
TEST_CLASSES        = ["person", "object", "garment"]
SAVE_DIR            = "result/eval"
DEVICE              = torch.device("cuda:0")
DTYPE               = torch.bfloat16
SIZE                = (768, 768)
SEED                = 42
NUM_INFERENCE_STEPS = 8
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading generation models...")
pipe = FluxFillPipeline.from_pretrained("checkpoint/FLUX.1-Fill-dev", torch_dtype=DTYPE).to(DEVICE)
pipe.load_lora_weights(LORA_PATH)
redux = FluxPriorReduxPipeline.from_pretrained("checkpoint/FLUX.1-Redux-dev").to(dtype=DTYPE).to(DEVICE)

print("Loading metric models...")
loss_fn_lpips = lpips.LPIPS(net="alex").to(DEVICE)
dino          = timm.create_model("vit_small_patch8_224.dino", pretrained=True, num_classes=0).to(DEVICE).eval()
clip_model    = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
clip_proc     = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# ── Inference ────────────────────────────────────────────────────────────────
def run_inference(ref_image, ref_mask, tar_image, tar_mask):
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3   = np.stack([ref_mask] * 3, -1)
    masked_ref   = ref_image * ref_mask_3 + 255 * (1 - ref_mask_3)
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref = masked_ref[y1:y2, x1:x2]
    ref_mask   = ref_mask[y1:y2, x1:x2]
    masked_ref, ref_mask = expand_image_mask(masked_ref, ref_mask, ratio=1.3)
    masked_ref = pad_to_square(masked_ref, pad_value=255)

    kernel   = np.ones((7, 7), np.uint8)
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=2)

    tar_box      = get_bbox_from_mask(tar_mask)
    tar_box      = expand_bbox(tar_mask, tar_box, ratio=1.2)
    tar_box_crop = expand_bbox(tar_image, tar_box, ratio=2)
    tar_box_crop = box2squre(tar_image, tar_box_crop)
    y1, y2, x1, x2 = tar_box_crop

    old_tar   = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2]
    tar_mask  = tar_mask[y1:y2, x1:x2]
    H1, W1    = tar_image.shape[:2]

    tar_mask   = pad_to_square(tar_mask, pad_value=0)
    tar_mask   = cv2.resize(tar_mask, SIZE)
    masked_ref = cv2.resize(masked_ref.astype(np.uint8), SIZE)
    pipe_prior = redux(Image.fromarray(masked_ref))

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2    = tar_image.shape[:2]
    tar_image = cv2.resize(tar_image, SIZE)

    diptych  = np.concatenate([masked_ref, tar_image], axis=1)
    mask_dip = np.concatenate([np.zeros_like(tar_image), np.stack([tar_mask] * 3, -1)], axis=1)

    diptych_pil      = Image.fromarray(diptych)
    mask_dip[mask_dip == 1] = 255
    mask_dip_pil     = Image.fromarray(mask_dip)

    gen = torch.Generator(DEVICE).manual_seed(SEED)
    out = pipe(
        image=diptych_pil, mask_image=mask_dip_pil,
        height=mask_dip_pil.size[1], width=mask_dip_pil.size[0],
        max_sequence_length=512, generator=gen,
        num_inference_steps=NUM_INFERENCE_STEPS,
        **pipe_prior
    ).images[0]

    w, h = out.size
    out  = out.crop((w // 2, 0, w, h))
    out  = np.array(out)
    out  = crop_back(out, old_tar, np.array([H1, W1, H2, W2]), np.array(tar_box_crop))
    return out, masked_ref  # return masked_ref for CLIP-I


# ── Feature extraction helpers ───────────────────────────────────────────────
def crop_fg(img_np, mask):
    """Crop image to foreground bounding box."""
    if mask.sum() > 100:
        ys, xs = np.where(mask > 0)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        return img_np[y1:y2+1, x1:x2+1]
    return img_np


def to_dino_tensor(img_np):
    img = cv2.resize(img_np, (224, 224))
    img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    img  = (img - mean[:, None, None]) / std[:, None, None]
    return img.unsqueeze(0).to(DEVICE)


def to_lpips_tensor(img_np):
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 127.5 - 1
    return t.unsqueeze(0).to(DEVICE)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred, gt, tar_mask, ref_image_orig):
    pred_r = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    mask_r = cv2.resize(tar_mask.astype(np.uint8), (gt.shape[1], gt.shape[0]))
    bg_mask = (mask_r == 0)

    # SSIM — background only
    pred_bg = pred_r.copy(); pred_bg[~bg_mask] = 0
    gt_bg   = gt.copy();     gt_bg[~bg_mask]   = 0
    ssim_bg = ssim(gt_bg, pred_bg, channel_axis=2, data_range=255)

    # LPIPS — full image
    lpips_full = loss_fn_lpips(to_lpips_tensor(pred_r), to_lpips_tensor(gt)).item()

    # LPIPS — foreground only
    pred_fg_img = crop_fg(pred_r, mask_r)
    gt_fg_img   = crop_fg(gt,     mask_r)
    pred_fg_img = cv2.resize(pred_fg_img, (gt_fg_img.shape[1], gt_fg_img.shape[0]))
    lpips_fg = loss_fn_lpips(to_lpips_tensor(pred_fg_img), to_lpips_tensor(gt_fg_img)).item()

    # DINO Score — foreground of pred vs foreground of gt
    with torch.no_grad():
        feat_pred = dino(to_dino_tensor(crop_fg(pred_r, mask_r)))
        feat_gt   = dino(to_dino_tensor(crop_fg(gt,     mask_r)))
    dino_score = F.cosine_similarity(feat_pred, feat_gt).item()

    # CLIP-I Score — foreground of pred vs ref_image (appearance preservation)
    pred_fg_pil = Image.fromarray(cv2.resize(crop_fg(pred_r, mask_r), (224, 224)))
    ref_pil     = Image.fromarray(cv2.resize(ref_image_orig, (224, 224)))
    inputs = clip_proc(images=[pred_fg_pil, ref_pil], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
    clip_i = F.cosine_similarity(feats[0:1], feats[1:2]).item()

    return ssim_bg, lpips_full, lpips_fg, dino_score, clip_i


# ── Run evaluation ────────────────────────────────────────────────────────────
all_ssim, all_lpips_full, all_lpips_fg, all_dino, all_clip_i = [], [], [], [], []

print("\n========== EVALUATION SUMMARY ==========")
for cls in TEST_CLASSES:
    n = len([f for f in os.listdir(os.path.join(TEST_ROOT, cls, "ref_image")) if f.endswith(".png")])
    print(f"  {cls:10}: {n} samples")
total = sum(
    len([f for f in os.listdir(os.path.join(TEST_ROOT, cls, "ref_image")) if f.endswith(".png")])
    for cls in TEST_CLASSES
)
print(f"  {'TOTAL':10}: {total} samples")
print(f"  Inference steps: {NUM_INFERENCE_STEPS}")
print("=========================================\n")

for cls in TEST_CLASSES:
    test_dir = os.path.join(TEST_ROOT, cls)
    save_cls = os.path.join(SAVE_DIR, cls)
    gt_cls   = os.path.join(SAVE_DIR, cls + "_gt")
    os.makedirs(save_cls, exist_ok=True)
    os.makedirs(gt_cls,   exist_ok=True)

    files = sorted([f for f in os.listdir(os.path.join(test_dir, "ref_image")) if f.endswith(".png")])
    ssim_list, lpips_full_list, lpips_fg_list, dino_list, clip_i_list = [], [], [], [], []

    print(f"\n{'='*65}")
    print(f"Class: {cls}  ({len(files)} samples)")

    for fname in files:
        ref_image = cv2.cvtColor(cv2.imread(os.path.join(test_dir, "ref_image", fname)), cv2.COLOR_BGR2RGB)
        ref_mask  = (cv2.imread(os.path.join(test_dir, "ref_mask",  fname)) > 128).astype(np.uint8)[:, :, 0]
        tar_image = cv2.cvtColor(cv2.imread(os.path.join(test_dir, "tar_image", fname)), cv2.COLOR_BGR2RGB)
        tar_mask  = (cv2.imread(os.path.join(test_dir, "tar_mask",  fname)) > 128).astype(np.uint8)[:, :, 0]
        tar_mask  = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))
        gt        = tar_image.copy()

        try:
            pred, _ = run_inference(ref_image, ref_mask, tar_image, tar_mask)
            s, lp_full, lp_fg, dino_s, clip_s = compute_metrics(pred, gt, tar_mask, ref_image)
            ssim_list.append(s); lpips_full_list.append(lp_full)
            lpips_fg_list.append(lp_fg); dino_list.append(dino_s); clip_i_list.append(clip_s)
            cv2.imwrite(os.path.join(save_cls, fname), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            shutil.copy(os.path.join(test_dir, "tar_image", fname), os.path.join(gt_cls, fname))
            print(f"  {fname}: SSIM={s:.4f} LPIPS={lp_full:.4f} LPIPS_fg={lp_fg:.4f} DINO={dino_s:.4f} CLIP-I={clip_s:.4f}")
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    fid_score = fid.compute_fid(save_cls, gt_cls, device=DEVICE)
    all_ssim       += ssim_list
    all_lpips_full += lpips_full_list
    all_lpips_fg   += lpips_fg_list
    all_dino       += dino_list
    all_clip_i     += clip_i_list

    print(f"\n  [{cls}]")
    print(f"    SSIM_bg  (↑): {np.mean(ssim_list):.4f}")
    print(f"    LPIPS    (↓): {np.mean(lpips_full_list):.4f}")
    print(f"    LPIPS_fg (↓): {np.mean(lpips_fg_list):.4f}")
    print(f"    DINO     (↑): {np.mean(dino_list):.4f}")
    print(f"    CLIP-I   (↑): {np.mean(clip_i_list):.4f}")
    print(f"    FID      (↓): {fid_score:.2f}  ({len(ssim_list)}/{len(files)} ok)")

# ── Overall ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("OVERALL RESULTS")
print(f"  CLIP-I   (↑): {np.mean(all_clip_i):.4f}   ← fg appearance vs reference")
print(f"  DINO     (↑): {np.mean(all_dino):.4f}   ← fg semantic identity")
print(f"  LPIPS_fg (↓): {np.mean(all_lpips_fg):.4f}   ← fg perceptual quality")
print(f"  SSIM_bg  (↑): {np.mean(all_ssim):.4f}   ← background preservation")
print(f"  LPIPS    (↓): {np.mean(all_lpips_full):.4f}   ← overall perceptual quality")
print(f"  FID      (↓): per class (see above)")
print(f"  Total    : {len(all_ssim)} samples")
