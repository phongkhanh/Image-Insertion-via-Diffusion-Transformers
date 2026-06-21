"""
Read metrics_cross_baseline/V2/V3.xlsx, find cases where:
  - V2 better than baseline
  - V3 better than both baseline and V2
Save filtered results to result/cross_comparison.xlsx
and result/cross_comparison_images/ (one stacked image per case).
"""

import os
import cv2
import numpy as np
import pandas as pd

EXCEL = {
    "baseline": "result/cross_person_baseline/metrics_cross_baseline.xlsx",
    "V2":       "result/cross_person_V2/metrics_cross_V2.xlsx",
    "V3":       "result/cross_person_V3/metrics_cross_V3.xlsx",
}
TEST_ROOT  = "/data1/stage/navsim_workspace/AnyInsertion/data_training_mask_prompt/test"
OUT_EXCEL  = "result/cross_comparison.xlsx"
OUT_IMAGES = "result/cross_comparison_images"
OUT_LIGHT  = "result/cross_lighting_images"   # cases with large lighting gap

METRICS_UP = ["SSIM_bg", "DINO_ref", "CLIP_I"]   # higher is better

LIGHTING_GAP_THRESHOLD = 20.0   # L channel diff (0-100); tune if needed

# ── Load ──────────────────────────────────────────────────────────────────────
dfs = {}
for name, path in EXCEL.items():
    df = pd.read_excel(path, sheet_name="per_pair")
    df = df.dropna(subset=METRICS_UP)
    dfs[name] = df[["class", "ref_file", "tar_file", "out_file",
                     "SSIM_bg", "DINO_ref", "CLIP_I", "composite_path"]]

# ── Merge on (class, ref_file, tar_file) ─────────────────────────────────────
merged = dfs["baseline"].merge(
    dfs["V2"],       on=["class", "ref_file", "tar_file"], suffixes=("_base", "_V2")
).merge(
    dfs["V3"][["class", "ref_file", "tar_file"] + METRICS_UP + ["composite_path"]],
    on=["class", "ref_file", "tar_file"]
)
merged = merged.rename(columns={
    "out_file_base":      "out_file",
    "composite_path_base":"composite_base",
    "composite_path_V2":  "composite_V2",
    "composite_path":     "composite_V3",
    **{m: f"{m}_V3" for m in METRICS_UP},
})

# ── Lighting gap ─────────────────────────────────────────────────────────────
def compute_lighting_gap(cls, ref_fname, tar_fname):
    """
    Returns (brightness_gap, color_gap, total_gap) in LAB space.
    brightness_gap = |mean L of ref_fg  - mean L of tar_bg|   (0-100)
    color_gap      = sqrt((dA)^2 + (dB)^2) of mean A,B       (approx 0-180)
    """
    ref_img  = cv2.imread(os.path.join(TEST_ROOT, cls, "ref_image", ref_fname))
    ref_mask = cv2.imread(os.path.join(TEST_ROOT, cls, "ref_mask",  ref_fname), cv2.IMREAD_GRAYSCALE)
    tar_img  = cv2.imread(os.path.join(TEST_ROOT, cls, "tar_image", tar_fname))
    tar_mask = cv2.imread(os.path.join(TEST_ROOT, cls, "tar_mask",  tar_fname), cv2.IMREAD_GRAYSCALE)

    if any(x is None for x in [ref_img, ref_mask, tar_img, tar_mask]):
        return None, None, None

    tar_mask = cv2.resize(tar_mask, (tar_img.shape[1], tar_img.shape[0]))
    ref_mask_bin = (ref_mask > 128)
    tar_bg_bin   = (cv2.resize(tar_mask, (tar_img.shape[1], tar_img.shape[0])) < 128)

    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(float)
    tar_lab = cv2.cvtColor(tar_img, cv2.COLOR_BGR2LAB).astype(float)

    if ref_mask_bin.sum() < 50 or tar_bg_bin.sum() < 50:
        return None, None, None

    ref_L, ref_A, ref_B = [ref_lab[:, :, c][ref_mask_bin].mean() for c in range(3)]
    tar_L, tar_A, tar_B = [tar_lab[:, :, c][tar_bg_bin].mean()   for c in range(3)]

    brightness_gap = abs(ref_L - tar_L)
    color_gap      = float(np.sqrt((ref_A - tar_A) ** 2 + (ref_B - tar_B) ** 2))
    total_gap      = brightness_gap + color_gap
    return round(brightness_gap, 2), round(color_gap, 2), round(total_gap, 2)

print("Computing lighting gaps...")
gaps = [compute_lighting_gap(r["class"], r["ref_file"], r["tar_file"])
        for _, r in merged.iterrows()]
merged["brightness_gap"], merged["color_gap"], merged["lighting_gap"] = zip(*gaps)
merged["brightness_gap"] = merged["brightness_gap"].astype(float)
merged["color_gap"]      = merged["color_gap"].astype(float)
merged["lighting_gap"]   = merged["lighting_gap"].astype(float)

# ── Delta columns ─────────────────────────────────────────────────────────────
for m in METRICS_UP:
    merged[f"delta_V2_base_{m}"] = (merged[f"{m}_V2"]   - merged[f"{m}_base"]).round(4)
    merged[f"delta_V3_V2_{m}"]   = (merged[f"{m}_V3"]   - merged[f"{m}_V2"]).round(4)
    merged[f"delta_V3_base_{m}"] = (merged[f"{m}_V3"]   - merged[f"{m}_base"]).round(4)

# ── Filter cases ──────────────────────────────────────────────────────────────
# V2 > baseline on at least 2 of 3 metrics
v2_better = (
    (merged["delta_V2_base_DINO_ref"] > 0).astype(int) +
    (merged["delta_V2_base_CLIP_I"]   > 0).astype(int) +
    (merged["delta_V2_base_SSIM_bg"]  > 0).astype(int)
) >= 2

# V3 > baseline AND V3 > V2 on at least 2 of 3 metrics
v3_better = (
    ((merged["delta_V3_base_DINO_ref"] > 0) & (merged["delta_V3_V2_DINO_ref"] > 0)).astype(int) +
    ((merged["delta_V3_base_CLIP_I"]   > 0) & (merged["delta_V3_V2_CLIP_I"]   > 0)).astype(int) +
    ((merged["delta_V3_base_SSIM_bg"]  > 0) & (merged["delta_V3_V2_SSIM_bg"]  > 0)).astype(int)
) >= 2

df_selected = merged[v2_better & v3_better].copy()

# rank by total improvement (V3 vs baseline, sum of all 3 metrics)
df_selected["score"] = (
    df_selected["delta_V3_base_DINO_ref"] +
    df_selected["delta_V3_base_CLIP_I"]   +
    df_selected["delta_V3_base_SSIM_bg"]
)
df_selected = df_selected.sort_values("score", ascending=False)

# lighting subset: cases with large lighting gap (harmonization most needed)
df_lighting = merged[
    merged["lighting_gap"].notna() &
    (merged["lighting_gap"] >= LIGHTING_GAP_THRESHOLD)
].copy()
df_lighting["score"] = (
    df_lighting["delta_V3_base_DINO_ref"] +
    df_lighting["delta_V3_base_CLIP_I"]   +
    df_lighting["delta_V3_base_SSIM_bg"]
)
df_lighting = df_lighting.sort_values("lighting_gap", ascending=False)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_EXCEL), exist_ok=True)

# column order for readability
cols_front = ["class", "ref_file", "tar_file", "score", "lighting_gap", "brightness_gap", "color_gap"]
cols_metrics = [
    "SSIM_bg_base", "SSIM_bg_V2", "SSIM_bg_V3",
    "DINO_ref_base","DINO_ref_V2","DINO_ref_V3",
    "CLIP_I_base",  "CLIP_I_V2",  "CLIP_I_V3",
]
cols_delta = [c for c in df_selected.columns if c.startswith("delta_")]
cols_paths = ["composite_base", "composite_V2", "composite_V3"]
df_out = df_selected[cols_front + cols_metrics + cols_delta + cols_paths]

cols_light = ["class", "ref_file", "tar_file", "lighting_gap", "brightness_gap", "color_gap", "score"] + \
             cols_metrics + cols_delta + cols_paths

with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    df_out.to_excel(writer, sheet_name="selected", index=False)
    df_lighting[cols_light].to_excel(writer, sheet_name="lighting_cases", index=False)
    merged.sort_values("lighting_gap", ascending=False) \
          .to_excel(writer, sheet_name="all_pairs", index=False)

# ── Save comparison images ────────────────────────────────────────────────────
os.makedirs(OUT_IMAGES, exist_ok=True)

LABEL_H    = 36                          # height of label bar above each row
LABEL_COLOR = (255, 255, 255)
BG_COLORS  = {"baseline": (60,  60,  60),
               "V2":       (30,  80,  30),
               "V3":       (30,  30, 100)}

def add_label_bar(img, text, bg):
    """Prepend a colored label bar on top of img."""
    bar = np.full((LABEL_H, img.shape[1], 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (10, LABEL_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL_COLOR, 2, cv2.LINE_AA)
    return np.vstack([bar, img])

def load_resize_w(path, width):
    img = cv2.imread(path)
    if img is None:
        return np.full((256, width, 3), 128, dtype=np.uint8)
    h, w = img.shape[:2]
    new_h = max(1, int(h * width / w))
    return cv2.resize(img, (width, new_h))

print(f"\nSaving comparison images → {OUT_IMAGES}/")
for rank, (_, row) in enumerate(df_selected.iterrows(), start=1):
    ref_stem = os.path.splitext(row["ref_file"])[0]
    tar_stem = os.path.splitext(row["tar_file"])[0]

    # target width = width of the baseline composite (all 3 should be same)
    base_img = cv2.imread(row["composite_base"])
    W = base_img.shape[1] if base_img is not None else 1536

    panels = []
    for run_name, path_col in [("baseline", "composite_base"),
                                ("V2",       "composite_V2"),
                                ("V3",       "composite_V3")]:
        img = load_resize_w(row[path_col], W)
        dino_val  = row[f"DINO_ref_{run_name if run_name != 'baseline' else 'base'}"]
        clip_val  = row[f"CLIP_I_{run_name if run_name != 'baseline' else 'base'}"]
        label = f"{run_name}   DINO={dino_val:.4f}  CLIP-I={clip_val:.4f}"
        panels.append(add_label_bar(img, label, BG_COLORS[run_name]))

    # pad panels to same width before stacking
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 40, dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    sep  = np.full((6, max_w, 3), 80, dtype=np.uint8)
    stack = np.vstack([padded[0], sep, padded[1], sep, padded[2]])

    fname = f"{rank:03d}_{row['class']}__{ref_stem}__into__{tar_stem}.jpg"
    cv2.imwrite(os.path.join(OUT_IMAGES, fname), stack, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  [{rank:03d}] {fname}")

print(f"\nSaving lighting-gap images → {OUT_LIGHT}/")
os.makedirs(OUT_LIGHT, exist_ok=True)
for rank, (_, row) in enumerate(df_lighting.iterrows(), start=1):
    ref_stem = os.path.splitext(row["ref_file"])[0]
    tar_stem = os.path.splitext(row["tar_file"])[0]
    base_img = cv2.imread(row["composite_base"])
    W = base_img.shape[1] if base_img is not None else 1536

    panels = []
    for run_name, path_col in [("baseline", "composite_base"),
                                ("V2",       "composite_V2"),
                                ("V3",       "composite_V3")]:
        img      = load_resize_w(row[path_col], W)
        dino_val = row[f"DINO_ref_{run_name if run_name != 'baseline' else 'base'}"]
        clip_val = row[f"CLIP_I_{run_name if run_name != 'baseline' else 'base'}"]
        label    = f"{run_name}   DINO={dino_val:.4f}  CLIP-I={clip_val:.4f}"
        panels.append(add_label_bar(img, label, BG_COLORS[run_name]))

    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 40, dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    sep   = np.full((6, max_w, 3), 80, dtype=np.uint8)
    stack = np.vstack([padded[0], sep, padded[1], sep, padded[2]])

    fname = f"{rank:03d}_light{row['lighting_gap']:.0f}_{row['class']}__{ref_stem}__into__{tar_stem}.jpg"
    cv2.imwrite(os.path.join(OUT_LIGHT, fname), stack, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  [{rank:03d}] gap={row['lighting_gap']:.1f}  {fname}")

print(f"\nTotal pairs      : {len(merged)}")
print(f"Selected cases   : {len(df_selected)}  (V2>base, V3>both)")
print(f"Lighting cases   : {len(df_lighting)}  (lighting_gap >= {LIGHTING_GAP_THRESHOLD})")
print(f"Excel   → {OUT_EXCEL}")
print(f"Images  → {OUT_IMAGES}/")
print(f"Lighting→ {OUT_LIGHT}/")
