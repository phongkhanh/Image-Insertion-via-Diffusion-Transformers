# Image Insertion via Diffusion Transformers

A modified training and inference pipeline for reference-based image insertion, built on top of [Insert Anything](https://github.com/song-wensong/insert-anything) (AAAI 2026 Oral).

This fork adds a **Scene-Aware Reference Harmonization** module (`MultiStageHarmonizer`) that injects cross-attention between reference and scene tokens at three stages of the FLUX dual-block stack (after blocks 6, 12, and 18), improving visual consistency between the inserted object and the target scene.

## Architecture

The pipeline is based on:
- **FLUX.1-Fill-dev** — inpainting backbone
- **FLUX.1-Redux-dev** — reference image encoder
- **LoRA** (rank 256) — fine-tuned on the AnyInsertion dataset
- **MultiStageHarmonizer (V3)** — cross-attention harmonization at dual-block stages 6, 12, 18 with shared Q/K/V projections and per-stage learnable gates (zero-initialized → identity at step 0)

## Installation

```bash
conda create -n insertanything python=3.10
conda activate insertanything
pip install -r requirements.txt
```

## Download Checkpoints

Download the following models to the `checkpoint/` directory:

| Model | Source |
|-------|--------|
| FLUX.1-Fill-dev | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) |
| FLUX.1-Redux-dev | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) |
| Insert Anything LoRA | [HuggingFace](https://huggingface.co/WensongSong/Insert-Anything) |

Expected structure:
```
checkpoint/
├── FLUX.1-Fill-dev/
├── FLUX.1-Redux-dev/
└── <lora_weights>.safetensors
```

## Inference

```bash
python inference.py
```

If `harmonizer.pt` is present alongside the LoRA weights, the harmonizer patch is applied automatically. Otherwise the pipeline runs in baseline mode.

## Gradio Demo

```bash
python app.py
```

## Training

Edit `experiments/config/insertanything.yaml` to set checkpoint paths and training config, then:

```bash
bash scripts/train.sh
```

Key training options in the config:
- `use_scene_harmonizer: true` — enables the MultiStageHarmonizer
- `train.max_steps` — number of training steps
- `model.lora_config.r` — LoRA rank (default 256)
- Optimizer: Prodigy with bias correction

## Contributors

- [phongkhanh](https://github.com/phongkhanh)

## Credits

This project builds on:
- [Insert Anything](https://github.com/song-wensong/insert-anything) (AAAI 2026)
- [FLUX.1](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [AnyInsertion dataset](https://huggingface.co/datasets/WensongSong/AnyInsertion_V1)
