# Multi-Method Emotion Recognition Pipeline (CAER-S)

Repository ini berisi pipeline modular untuk emosi recognition pada dataset **CAER-S**, dengan dukungan untuk **banyak metode** dari berbagai paper.

## Metode yang Tersedia

| Method | Paper | Status |
|--------|-------|--------|
| `caernet` | Lee et al., "Context-Aware Emotion Recognition Networks", ICCV 2019 | Ready |
| `zhou_cross_attention` | Zhou et al., "Emotion Recognition from Large-Scale Video Clips with Cross-Attention and Hybrid Feature Weighting Neural Networks", IJERPH 2023 | Ready |
| `yang_ccim` | Yang et al., "Context De-Confounded Emotion Recognition", CVPR 2023 | Ready |
| `glamor_net` | Le et al., "Global-Local Attention for Emotion Recognition", Neural Computing and Applications, 2022 | Ready |

## Struktur Repository (Multi-Method)

```text
.
├── bin/                          # Helper bash scripts
│   ├── setup_uv.sh              # UV environment setup
│   ├── build_manifest.sh        # Build data manifest
│   ├── smoke_test.sh            # Data pipeline smoke test
│   ├── train.sh                 # Unified training script
│   └── evaluate.sh              # Unified evaluation script
├── configs/
│   ├── caernet.yaml             # CAER-Net config
│   ├── zhou_cross_attention.yaml # Zhou et al. config
│   ├── yang_ccim.yaml           # Yang et al. (CCIM) config
│   └── glamor_net.yaml          # Le et al. (GLAMOR-Net) config
├── scripts/
│   ├── build_caers_manifest.py  # Manifest builder CLI
│   ├── smoke_data_pipeline.py   # Smoke test CLI
│   ├── train.py                 # Unified training CLI (multi-method)
│   └── evaluate.py              # Unified evaluation CLI (multi-method)
├── src/
│   ├── config/                  # Configuration loader
│   │   └── config.py
│   ├── datasets/                # Shared dataset utilities
│   │   ├── caers_dataset.py
│   │   └── transforms.py
│   ├── engine/                  # Shared training/eval engine
│   │   ├── metrics.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── models/                  # All methods/models
│   │   ├── common.py            # Shared encoder builder
│   │   ├── caernet/             # CAER-Net method
│   │   │   └── model.py
│   │   ├── zhou_cross_attention/ # Zhou et al. method
│   │   │   └── model.py
│   │   ├── yang_ccim/           # Yang et al. (CCIM) method
│   │   │   ├── model.py
│   │   │   └── confounder_builder.py
│   │   └── glamor_net/          # Le et al. (GLAMOR-Net) method
│   │       └── model.py
│   └── utils/
│       ├── io_utils.py
│       └── data_manifest.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Menambahkan Metode Baru

Untuk menambahkan metode dari paper baru:

1. **Buat direktori model**: `src/models/<method_name>/`
2. **Implementasikan model**: Buat `model.py` dengan class yang mengimplementasikan `forward(face_image, context_image) -> dict`
3. **Update config loader**: Tambahkan method-specific config di `src/config/config.py`
4. **Buat config file**: `configs/<method_name>.yaml`
5. **Register di scripts**: Update `build_model()` di `scripts/train.py` dan `scripts/evaluate.py`

Contoh minimal model:
```python
class MyNewModel(nn.Module):
    def forward(self, face_image, context_image):
        # ... your architecture ...
        return {"logits": logits}
```

## Setup

```bash
./bin/setup_uv.sh
# atau manual: uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"
```

## Pipeline Steps

### 1. Build Manifest + Diagnostics

```bash
./bin/build_manifest.sh
# atau dengan config tertentu:
./bin/build_manifest.sh --config configs/zhou_cross_attention.yaml
```

### 2. Smoke Test Data Pipeline

```bash
./bin/smoke_test.sh
```

### 3. Train

```bash
# CAER-Net
./bin/train.sh --config configs/caernet.yaml

# Zhou et al. Cross-Attention
./bin/train.sh --config configs/zhou_cross_attention.yaml

# Yang et al. CCIM (causal intervention)
./bin/train.sh --config configs/yang_ccim.yaml

# Le et al. GLAMOR-Net (global-local attention)
./bin/train.sh --config configs/glamor_net.yaml

# Dengan augmentasi
./bin/train.sh --config configs/zhou_cross_attention.yaml --augment

# Custom run name
./bin/train.sh --config configs/caernet.yaml --run-name "caernet_baseline"

# Resume dari checkpoint
./bin/train.sh --config configs/caernet.yaml --resume checkpoints/caernet/best_model.pt
```

### 4. Evaluate

```bash
# Auto-detect checkpoint berdasarkan method di config
./bin/evaluate.sh --config configs/caernet.yaml

# Evaluasi split val
./bin/evaluate.sh --config configs/zhou_cross_attention.yaml --split val

# Yang et al. CCIM
./bin/evaluate.sh --config configs/yang_ccim.yaml

# Le et al. GLAMOR-Net
./bin/evaluate.sh --config configs/glamor_net.yaml

# Custom checkpoint
./bin/evaluate.sh --config configs/caernet.yaml --checkpoint checkpoints/caernet/best_model.pt
```

## Ablation Studies (CAER-Net)

Ubah `train.stream_mode` di `configs/caernet.yaml`:

```yaml
train:
  stream_mode: face      # Face-only baseline
  # stream_mode: context # Context-only baseline
  # stream_mode: multimodal # Full two-stream (default)
```

## Monitoring dengan W&B

Training & evaluation otomatis log ke Weights & Biases:
- Metrics per epoch
- Model checkpoints sebagai artifact
- Per-class accuracy tables & charts

Set environment variables untuk konfigurasi W&B Anda sendiri:
```bash
export WANDB_API_KEY="your_key"
export WANDB_PROJECT="your_project"
```

## Notes

- Dataset CAER-S harus memiliki struktur `train/` dan `test/` dengan subfolder per kelas.
- Semua metode menggunakan **two-stream input** (face + context) kecuali ablasi single-stream.
- Pretrained backbones sangat direkomendasikan karena ukuran CAER-S yang relatif kecil.
- **Yang CCIM**: Confounder dictionary dibangun otomatis dari training data saat pertama kali training. File akan disimpan di `checkpoints/yang_ccim/confounder_dict.pt` dan bisa digunakan kembali untuk resume/evaluasi.
