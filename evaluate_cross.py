import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import linear_sum_assignment
import timm
from cleanfid import fid
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask

# ── Args ──────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--lora_path", type=str, default="runs/20260604-030052/ckpt/4000")
_parser.add_argument("--run_name",  type=str, default=None)
_parser.add_argument("--save_dir",  type=str, default="result/cross_eval")
_parser.add_argument("--test_root", type=str, default="/data1/stage/navsim_workspace/AnyInsertion/data_training_mask_prompt/test")
_parser.add_argument("--steps",     type=int, default=50)
_args = _parser.parse_args()

# ── Config ───────────────────────────────────────────────────────────────────
LORA_PATH           = _args.lora_path
TEST_ROOT           = _args.test_root
TEST_CLASSES        = ["person"]
SAVE_DIR            = _args.save_dir
DEVICE              = torch.device("cuda:0")
DTYPE               = torch.bfloat16
SIZE                = (768, 768)
SEED                = 42
NUM_INFERENCE_STEPS = _args.steps
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading generation models...")
pipe  = FluxFillPipeline.from_pretrained("checkpoint/FLUX.1-Fill-dev", torch_dtype=DTYPE).to(DEVICE)
pipe.load_lora_weights(LORA_PATH)
redux = FluxPriorReduxPipeline.from_pretrained("checkpoint/FLUX.1-Redux-dev").to(dtype=DTYPE).to(DEVICE)

from src.models.harmonizer import MultiStageHarmonizer
from src.models.transformer import build_inference_forward

_harmonizer_path = os.path.join(LORA_PATH, "harmonizer.pt")
if os.path.isfile(_harmonizer_path):
    _harmonizer = MultiStageHarmonizer.load_for_inference(_harmonizer_path, DEVICE, DTYPE)
    pipe.transformer.forward = build_inference_forward(pipe.transformer, _harmonizer)
    print(f"[Harmonizer] Loaded from {_harmonizer_path}")
else:
    print(f"[Harmonizer] Not found — baseline mode")

print("Loading metric models...")
dino       = timm.create_model("vit_small_patch8_224.dino", pretrained=True, num_classes=0).to(DEVICE).eval()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# ── Smart pair selection ──────────────────────────────────────────────────────
def _fg_mean_L(img_bgr, mask_gray):
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    fg   = (mask_gray > 128)
    return lab[:, :, 0][fg].mean() if fg.sum() > 50 else 128.0

def _bg_mean_L(img_bgr, mask_gray):
    resized = cv2.resize(mask_gray, (img_bgr.shape[1], img_bgr.shape[0]))
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    bg   = (resized < 128)
    return lab[:, :, 0][bg].mean() if bg.sum() > 50 else 128.0

def _dino_feat(img_rgb, mask, dino_model, device):
    ys, xs = np.where(mask > 128)
    if len(ys) < 50:
        img_crop = img_rgb
    else:
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        img_crop = img_rgb[y1:y2+1, x1:x2+1]
    img  = cv2.resize(img_crop, (224, 224))
    t    = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    t    = (t - mean[:, None, None]) / std[:, None, None]
    with torch.no_grad():
        feat = dino_model(t.unsqueeze(0).to(device))
    return F.normalize(feat, dim=-1)

def smart_pairs(files, test_dir, dino_model, device):
    """
    Pair each ref with the tar that maximises:
      lighting_gap (|ref_fg_L - tar_bg_L|, normalised to 0-1)
    + pose_diff   (1 - cosine_sim(DINO_ref_fg, DINO_tar_fg), 0-1)
    Uses linear_sum_assignment for optimal one-to-one matching.
    """
    N = len(files)
    ref_L, tar_L, feats = [], [], []

    print(f"  Pre-computing features for {N} images...")
    for fname in files:
        ref_img  = cv2.imread(os.path.join(test_dir, "ref_image", fname))
        ref_mask = cv2.imread(os.path.join(test_dir, "ref_mask",  fname), cv2.IMREAD_GRAYSCALE)
        tar_img  = cv2.imread(os.path.join(test_dir, "tar_image", fname))
        tar_mask = cv2.imread(os.path.join(test_dir, "tar_mask",  fname), cv2.IMREAD_GRAYSCALE)

        ref_L.append(_fg_mean_L(ref_img, ref_mask))
        tar_L.append(_bg_mean_L(tar_img, tar_mask))
        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        feats.append(_dino_feat(ref_rgb, ref_mask, dino_model, device))

    # build N×N score matrix
    scores = np.full((N, N), -1e9)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            light = abs(ref_L[i] - tar_L[j]) / 100.0
            pose  = max(0.0, min(1.0 - F.cosine_similarity(feats[i], feats[j]).item(), 1.0))
            scores[i, j] = max(light, pose)  # good if EITHER lighting OR pose differs

    # optimal one-to-one assignment (maximise total score)
    row_ind, col_ind = linear_sum_assignment(-scores)
    pairs = [(files[r], files[c]) for r, c in zip(row_ind, col_ind)]

    print(f"  {'ref':>15}  →  {'tar':>15}   light_gap  pose_diff   reason")
    for r, c in zip(row_ind, col_ind):
        light = abs(ref_L[r] - tar_L[c]) / 100.0
        pose  = max(0.0, min(1.0 - F.cosine_similarity(feats[r], feats[c]).item(), 1.0))
        reason = "LIGHT" if light >= pose else "POSE"
        print(f"  {files[r]:>15}  →  {files[c]:>15}   {light:.3f}      {pose:.3f}       {reason}")

    return pairs


# ── Inference ─────────────────────────────────────────────────────────────────
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

    diptych_pil  = Image.fromarray(diptych)
    mask_dip[mask_dip == 1] = 255
    mask_dip_pil = Image.fromarray(mask_dip)

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
    return out


# ── Composite ─────────────────────────────────────────────────────────────────
def make_composite(ref_img, tar_img, pred_img, h=512):
    def _resize_h(img, height):
        ratio = height / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * ratio)), height))
    panels = [_resize_h(img, h) for img in (ref_img, tar_img, pred_img)]
    sep    = np.full((h, 4, 3), 200, dtype=np.uint8)
    return np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)


# ── Feature helpers ───────────────────────────────────────────────────────────
def crop_fg(img_np, mask):
    if mask.sum() > 100:
        ys, xs = np.where(mask > 0)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        return img_np[y1:y2+1, x1:x2+1]
    return img_np


def to_dino_tensor(img_np):
    img  = cv2.resize(img_np, (224, 224))
    img  = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    img  = (img - mean[:, None, None]) / std[:, None, None]
    return img.unsqueeze(0).to(DEVICE)


# ── Metrics (no GT) ───────────────────────────────────────────────────────────
def compute_metrics(pred, tar_image, tar_mask, ref_image):
    pred_r  = cv2.resize(pred, (tar_image.shape[1], tar_image.shape[0]))
    mask_r  = cv2.resize(tar_mask.astype(np.uint8), (tar_image.shape[1], tar_image.shape[0]))
    bg_mask = (mask_r == 0)

    # SSIM_bg — background of pred vs background of tar_image (scene preservation)
    pred_bg = pred_r.copy(); pred_bg[~bg_mask] = 0
    tar_bg  = tar_image.copy(); tar_bg[~bg_mask] = 0
    ssim_bg = ssim(tar_bg, pred_bg, channel_axis=2, data_range=255)

    # DINO — pred_fg vs ref_image (identity preservation vs reference)
    pred_fg = crop_fg(pred_r, mask_r)
    ref_fg  = crop_fg(ref_image, np.ones(ref_image.shape[:2], dtype=np.uint8))
    with torch.no_grad():
        feat_pred = dino(to_dino_tensor(pred_fg))
        feat_ref  = dino(to_dino_tensor(ref_fg))
    dino_score = F.cosine_similarity(feat_pred, feat_ref).item()

    # CLIP-I — pred_fg vs ref_image (appearance preservation vs reference)
    pred_fg_pil = Image.fromarray(cv2.resize(pred_fg, (224, 224)))
    ref_pil     = Image.fromarray(cv2.resize(ref_image, (224, 224)))
    inputs = clip_proc(images=[pred_fg_pil, ref_pil], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
    clip_i = F.cosine_similarity(feats[0:1], feats[1:2]).item()

    return ssim_bg, dino_score, clip_i


# ── Run evaluation ────────────────────────────────────────────────────────────
all_ssim, all_dino, all_clip_i = [], [], []
all_rows = []

_auto_name = "_".join(LORA_PATH.replace("\\", "/").rstrip("/").split("/")[-2:])
_run_name  = _args.run_name if _args.run_name else _auto_name

print("\n========== CROSS-IMAGE EVALUATION ==========")
for cls in TEST_CLASSES:
    n = len([f for f in os.listdir(os.path.join(TEST_ROOT, cls, "ref_image")) if f.endswith(".png")])
    print(f"  {cls:10}: {n} samples  →  {n} cross pairs")
print(f"  Pair seed       : {SEED}")
print(f"  Inference steps : {NUM_INFERENCE_STEPS}")
print("=============================================\n")

for cls in TEST_CLASSES:
    test_dir      = os.path.join(TEST_ROOT, cls)
    save_cls      = os.path.join(SAVE_DIR, cls)
    composite_cls = os.path.join(SAVE_DIR, cls + "_composite")
    fid_real_cls  = os.path.join(SAVE_DIR, cls + "_fid_real")
    os.makedirs(save_cls,      exist_ok=True)
    os.makedirs(composite_cls, exist_ok=True)
    os.makedirs(fid_real_cls,  exist_ok=True)

    files = sorted([f for f in os.listdir(os.path.join(test_dir, "ref_image")) if f.endswith(".png")])
    pairs = smart_pairs(files, test_dir, dino, DEVICE)

    ssim_list, dino_list, clip_i_list = [], [], []

    print(f"\n{'='*65}")
    print(f"Class: {cls}  ({len(pairs)} pairs)")

    for ref_fname, tar_fname in pairs:
        ref_image = cv2.cvtColor(cv2.imread(os.path.join(test_dir, "ref_image", ref_fname)), cv2.COLOR_BGR2RGB)
        ref_mask  = (cv2.imread(os.path.join(test_dir, "ref_mask",  ref_fname)) > 128).astype(np.uint8)[:, :, 0]
        tar_image = cv2.cvtColor(cv2.imread(os.path.join(test_dir, "tar_image", tar_fname)), cv2.COLOR_BGR2RGB)
        tar_mask  = (cv2.imread(os.path.join(test_dir, "tar_mask",  tar_fname)) > 128).astype(np.uint8)[:, :, 0]
        tar_mask  = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

        ref_stem = os.path.splitext(ref_fname)[0]
        tar_stem = os.path.splitext(tar_fname)[0]
        out_fname = f"{ref_stem}__into__{tar_stem}.png"

        try:
            pred = run_inference(ref_image, ref_mask, tar_image, tar_mask)
            s, dino_s, clip_s = compute_metrics(pred, tar_image, tar_mask, ref_image)
            ssim_list.append(s); dino_list.append(dino_s); clip_i_list.append(clip_s)

            cv2.imwrite(os.path.join(save_cls, out_fname),      cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(fid_real_cls, out_fname),  cv2.cvtColor(tar_image, cv2.COLOR_RGB2BGR))
            composite = make_composite(ref_image, tar_image, pred)
            cv2.imwrite(os.path.join(composite_cls, out_fname), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            print(f"  {ref_stem} → {tar_stem}: SSIM_bg={s:.4f} DINO={dino_s:.4f} CLIP-I={clip_s:.4f}")
            all_rows.append({
                "run":       _run_name,
                "class":     cls,
                "ref_file":  ref_fname,
                "tar_file":  tar_fname,
                "out_file":  out_fname,
                "SSIM_bg":   round(s,      4),
                "DINO_ref":  round(dino_s,  4),
                "CLIP_I":    round(clip_s,  4),
                "composite_path": os.path.join(composite_cls, out_fname),
            })
        except Exception as e:
            print(f"  [ERROR] {ref_stem} → {tar_stem}: {e}")
            all_rows.append({
                "run": _run_name, "class": cls,
                "ref_file": ref_fname, "tar_file": tar_fname, "out_file": out_fname,
                "SSIM_bg": None, "DINO_ref": None, "CLIP_I": None,
                "composite_path": "ERROR",
            })

    fid_score = fid.compute_fid(save_cls, fid_real_cls, device=DEVICE)
    all_ssim   += ssim_list
    all_dino   += dino_list
    all_clip_i += clip_i_list

    print(f"\n  [{cls}]")
    print(f"    SSIM_bg  (↑): {np.mean(ssim_list):.4f}  ← background not disturbed")
    print(f"    DINO_ref (↑): {np.mean(dino_list):.4f}  ← inserted obj matches ref identity")
    print(f"    CLIP-I   (↑): {np.mean(clip_i_list):.4f}  ← inserted obj matches ref appearance")
    print(f"    FID      (↓): {fid_score:.2f}  ({len(ssim_list)}/{len(pairs)} ok)")

# ── Overall ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("OVERALL RESULTS (cross-image)")
print(f"  CLIP-I   (↑): {np.mean(all_clip_i):.4f}   ← appearance vs reference")
print(f"  DINO_ref (↑): {np.mean(all_dino):.4f}   ← identity vs reference")
print(f"  SSIM_bg  (↑): {np.mean(all_ssim):.4f}   ← background preservation")
print(f"  FID      (↓): per class (see above)")
print(f"  Total    : {len(all_ssim)} pairs")

# ── Excel export ──────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)

summary_rows = []
for cls in TEST_CLASSES:
    sub = df[df["class"] == cls].dropna(subset=["SSIM_bg"])
    if len(sub) == 0:
        continue
    summary_rows.append({
        "run":      _run_name,
        "class":    cls,
        "n_ok":     len(sub),
        "SSIM_bg":  round(sub["SSIM_bg"].mean(),  4),
        "DINO_ref": round(sub["DINO_ref"].mean(), 4),
        "CLIP_I":   round(sub["CLIP_I"].mean(),   4),
    })
ok = df.dropna(subset=["SSIM_bg"])
summary_rows.append({
    "run":      _run_name,
    "class":    "ALL",
    "n_ok":     len(ok),
    "SSIM_bg":  round(ok["SSIM_bg"].mean(),  4),
    "DINO_ref": round(ok["DINO_ref"].mean(), 4),
    "CLIP_I":   round(ok["CLIP_I"].mean(),   4),
})
df_summary = pd.DataFrame(summary_rows)

excel_path = os.path.join(SAVE_DIR, f"metrics_cross_{_run_name}.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="per_pair", index=False)
    df_summary.to_excel(writer, sheet_name="summary", index=False)

print(f"\n[Excel] Saved → {excel_path}")
