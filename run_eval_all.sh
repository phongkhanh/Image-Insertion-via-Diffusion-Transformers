#!/bin/bash
set -e

STEPS=50
TEST_ROOT="/data1/stage/navsim_workspace/AnyInsertion/data_training_mask_prompt/test"

echo "============================================"
echo "  Running evaluation for all checkpoints"
echo "============================================"

# ── baseline ─────────────────────────────────────────────────────────────────
echo ""
echo "[1/3] baseline"
python evaluate.py \
    --lora_path  "runs/20260603-125049/ckpt/final" \
    --run_name   "baseline" \
    --save_dir   "result/baseline" \
    --test_root  "$TEST_ROOT" \
    --steps      $STEPS

# ── V2 ───────────────────────────────────────────────────────────────────────
echo ""
echo "[2/3] V2"
python evaluate.py \
    --lora_path  "runs/20260604-030052/ckpt/4000" \
    --run_name   "V2" \
    --save_dir   "result/V2" \
    --test_root  "$TEST_ROOT" \
    --steps      $STEPS

# ── V3 ───────────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] V3"
python evaluate.py \
    --lora_path  "runs/20260604-224425/ckpt/2000" \
    --run_name   "V3" \
    --save_dir   "result/V3" \
    --test_root  "$TEST_ROOT" \
    --steps      $STEPS

echo ""
echo "============================================"
echo "  All done!"
echo "  result/baseline/metrics_baseline.xlsx"
echo "  result/V2/metrics_V2.xlsx"
echo "  result/V3/metrics_V3.xlsx"
echo "============================================"
