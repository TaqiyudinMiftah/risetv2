# CAER-S Pipeline (CAER-Net Focus)

This repository contains a focused pipeline for the CAER-S part of:
"Context-Aware Emotion Recognition Networks" (ICCV 2019).

Current scope:
- Phase 1: project structure and reproducible config.
- Phase 2: CAER-S data pipeline (manifest builder, stratified val split, diagnostics, PyTorch dataset, and smoke check).
- Phase 3: minimal CAER-Net model (face stream, context stream, adaptive fusion), training, evaluation, and W&B monitoring.

## Expected CAER-S Layout

Point `dataset_root` to the folder that contains `train` and `test`.

Example:

```text
/path/to/CAER-S/
  train/
    Angry/
    Disgust/
    Fear/
    Happy/
    Neutral/
    Sad/
    Surprise/
  test/
    Angry/
    Disgust/
    Fear/
    Happy/
    Neutral/
    Sad/
    Surprise/
```

## Setup with UV (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment

```bash
./bin/setup_uv.sh
```

Or manually:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Generate Lockfile (Optional)

```bash
uv lock
```

### Alternative: pip

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline Steps

### 1. Build Manifest + Diagnostics

```bash
./bin/build_manifest.sh
```

Or manually:

```bash
python scripts/build_caers_manifest.py --config configs/caers_data.yaml
```

Outputs:
- `artifacts/caers/manifest_caers.jsonl` - JSONL manifest with per-sample metadata.
- `artifacts/caers/diagnostics_caers.json` - Diagnostics with split/class counts.

### 2. Smoke Test Data Pipeline

```bash
./bin/smoke_test.sh
```

Or manually:

```bash
python scripts/smoke_data_pipeline.py --config configs/caers_data.yaml
```

This checks:
- Manifest can be loaded.
- Label mapping is stable.
- DataLoader returns tensors for both face and context streams.

### 3. Train CAER-Net with W&B

```bash
./bin/train.sh
```

This will automatically log to Weights & Biases. The API key is already configured in the script.

#### Training Options

```bash
# Custom run name
./bin/train.sh --run-name "caers_resnet18_baseline"

# Resume from checkpoint
./bin/train.sh --resume checkpoints/caers/best_model.pt

# Offline mode (no W&B sync)
./bin/train.sh --offline

# Custom config
./bin/train.sh --config configs/my_config.yaml
```

Or manually with Python:

```bash
python scripts/train_caers.py \
  --config configs/caers_data.yaml \
  --wandb-project caers-emotion-recognition \
  --wandb-run-name "my_experiment"
```

#### W&B Configuration

The default W&B configuration is in `bin/train.sh`:
- **Project**: `caers-emotion-recognition`
- **API Key**: Configured in script

To use your own W&B account, edit `bin/train.sh` or set environment variables:

```bash
export WANDB_API_KEY="your_key_here"
export WANDB_PROJECT="your_project"
```

### 4. Evaluate

```bash
./bin/evaluate.sh
```

Options:

```bash
# Evaluate on validation split
./bin/evaluate.sh --split val

# Use custom checkpoint
./bin/evaluate.sh --checkpoint checkpoints/caers/best_model.pt
```

Or manually:

```bash
python scripts/evaluate_caers.py \
  --config configs/caers_data.yaml \
  --checkpoint checkpoints/caers/best_model.pt \
  --split test
```

This reports:
- Top-1 and Top-5 accuracy
- Overall accuracy
- Per-class accuracy

## Ablation Studies

Change `train.stream_mode` in `configs/caers_data.yaml`:

```yaml
train:
  stream_mode: face     # Face-only baseline
  # stream_mode: context  # Context-only baseline
  # stream_mode: multimodal  # Full two-stream (default)
```

Then run `./bin/train.sh` again.

## Project Structure

```text
.
├── bin/                          # Helper bash scripts
│   ├── setup_uv.sh              # UV environment setup
│   ├── build_manifest.sh        # Build data manifest
│   ├── smoke_test.sh            # Data pipeline smoke test
│   ├── train.sh                 # Training with W&B
│   └── evaluate.sh              # Evaluation
├── configs/
│   └── caers_data.yaml          # Main configuration
├── scripts/
│   ├── build_caers_manifest.py  # Manifest builder CLI
│   ├── smoke_data_pipeline.py   # Smoke test CLI
│   ├── train_caers.py           # Training CLI
│   └── evaluate_caers.py        # Evaluation CLI
├── src/caers_pipeline/
│   ├── __init__.py
│   ├── config.py                # Config loader
│   ├── data_manifest.py         # Manifest generation
│   ├── dataset.py               # PyTorch dataset
│   ├── engine.py                # Training/eval loops
│   ├── io_utils.py              # JSON/JSONL utilities
│   └── model.py                 # CAER-Net model
├── pyproject.toml               # UV/Python project config
├── .python-version              # Python version pin
├── requirements.txt             # Pip requirements fallback
└── README.md                    # This file
```

## Monitoring with W&B

During training, the following are logged automatically:
- **Metrics**: train/val loss, top-1/top-5 accuracy per epoch
- **Model**: Gradient histograms, parameter distributions
- **Artifacts**: Best model checkpoint uploaded to W&B
- **Config**: All hyperparameters from YAML config

View results at: https://wandb.ai

## Notes

- Context stream masking supports optional face bounding boxes if `face_bbox` is present in manifest rows.
- If no face bbox exists, context image equals original image.
- The adaptive fusion module learns soft weights between face and context features.
- Pretrained backbones are recommended due to the small size of CAER-S.
- For reproducibility, seeds are set for Python, NumPy, and PyTorch.
