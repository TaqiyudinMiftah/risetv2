# CAER-Net Reproduction on CAER-S

Repository ini difokuskan untuk reproduksi ulang paper:

**Context-Aware Emotion Recognition Networks**  
Lee et al., ICCV 2019

Pipeline utama ada di:

```text
CAER_S_CAERNet_Reproduction_ipynb.ipynb
```

Notebook tersebut berisi alur end-to-end:

1. setup environment dan seed,
2. pencarian root dataset CAER-S,
3. download/cache detector files ke `detectors/`,
4. build manifest `caers_manifest.jsonl`,
5. dataset two-stream face/context,
6. implementasi CAER-Net-S,
7. training,
8. evaluasi test set dan confusion matrix.

## Struktur Saat Ini

```text
.
├── CAER_S_CAERNet_Reproduction_ipynb.ipynb  # pipeline utama
├── detectors/                               # detector txt/pth cache, bisa di-download ulang notebook
├── paper/                                   # referensi paper
├── CAER-S/                                  # dataset lokal, di-ignore git
├── requirements.txt
├── pyproject.toml
└── README.md
```

Folder lama seperti `src/`, `scripts/`, `configs/`, `bin/`, `artifacts/`, `wandb/`, dan checkpoint lama sudah tidak menjadi bagian pipeline utama.

## Setup

Rekomendasi dengan `uv`:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m ipykernel install --user --name caer-net-reproduction --display-name "CAER-Net Reproduction"
```

Atau dengan `pip` jika environment Anda menyediakannya:

```bash
python -m pip install -r requirements.txt
```

## Menjalankan Reproduction

1. Pastikan dataset CAER-S tersedia secara lokal.
2. Buka `CAER_S_CAERNet_Reproduction_ipynb.ipynb`.
3. Pilih kernel `CAER-Net Reproduction`.
4. Jalankan cell dari atas ke bawah.
5. Notebook akan membuat ulang file runtime seperti:

```text
caers_manifest.jsonl
checkpoints/<run_name>/config.json
checkpoints/<run_name>/best.pt
checkpoints/<run_name>/last.pt
checkpoints/<run_name>/history.json
checkpoints/<run_name>/history.csv
checkpoints/<run_name>/metrics.json
checkpoints/<run_name>/test_predictions.csv
checkpoints/<run_name>/confusion_matrix.png
```

File runtime tersebut di-ignore agar repo tetap bersih.

## Evaluasi Test Set

Untuk mengevaluasi checkpoint tanpa membuka notebook:

```bash
python evaluate_test.py --checkpoint checkpoints/<run_name>/best.pt
```

Jika memakai checkpoint lama di root `checkpoints/best_caernet_s.pt`, cukup jalankan:

```bash
python evaluate_test.py
```

Output evaluasi disimpan ke `eval_test/` di folder checkpoint, termasuk `metrics.json`, `test_predictions.csv`, dan `confusion_matrix.png`. Untuk smoke test cepat:

```bash
python evaluate_test.py --max-samples 64
```

## Experiment Tracking dan Resume

Notebook memiliki satu `CFG` dictionary untuk mengatur hyperparameter, W&B, AMP, early stopping, gradient clipping, dan resume. Default W&B adalah `offline` jika `WANDB_API_KEY` tidak tersedia, sehingga training tetap bisa berjalan tanpa login.

Untuk W&B online:

```bash
export WANDB_API_KEY=...
export WANDB_MODE=online
```

Untuk resume training, isi `CFG["resume_from"]` dengan checkpoint `last.pt`, misalnya:

```python
CFG["resume_from"] = "checkpoints/<run_name>/last.pt"
```

## Dual GPU

Notebook otomatis memakai semua GPU CUDA yang terdeteksi melalui `torch.nn.DataParallel`.
Pada mesin dengan 2x RTX 3060, konfigurasi default menjadi:

```text
GPU_IDS = [0, 1]
PER_GPU_BATCH_SIZE = 32
BATCH_SIZE = 64
```

Jika memori GPU tidak cukup, turunkan `PER_GPU_BATCH_SIZE` di cell dataloader.
