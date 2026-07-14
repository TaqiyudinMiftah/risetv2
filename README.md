# CAER-S Research: CAER-Net to CD-ICA-Net

Repository ini memakai reproduksi CAER-Net sebagai baseline untuk meneliti iterative bidirectional face-context interaction dan post-interaction debiasing pada CD-ICA-Net. Baseline fondasinya adalah:

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
├── configs/caer_official.json               # config untuk pipeline official ndkhanh360/CAER
├── run_caer_official.py                     # wrapper train/test pipeline official
├── third_party/CAER/                        # upstream CAER repo sebagai submodule
├── agents.d/                                # scope dan protokol eksperimen
├── paper/                                   # PDF, matrix, notes, dan source snapshots
├── protocols/                               # definisi split penelitian
├── reports/                                 # audit dan laporan reproducibility
├── tests/                                   # unit test audit/protokol
├── CAER-S/                                  # dataset lokal, di-ignore git
├── requirements.txt
├── pyproject.toml
└── README.md
```

Output lokal seperti `artifacts/`, `wandb/`, dataset, dan checkpoint di-ignore. Jangan menghapus source protocol, laporan, atau literature package yang dilacak Git.

## Arah Penelitian

Baca `agents.d/LITERATURE_AND_EXPERIMENT_PLAN.md` sebelum menambah eksperimen. Matrix paper berada di `paper/LITERATURE_MATRIX.md`, inventaris sumber di `paper/SOURCE_INVENTORY.md`, dan hasil audit data di `reports/data_protocol_audit.md`.

Aturan utama: jangan memakai test set untuk tuning, pisahkan hasil paper dari hasil reproduksi, dan gunakan `caer_s_content_disjoint_v1` untuk controlled comparison. GLAMOR-Net, CAHFW-Net, CLEF, DSCT, dan AGCD-Net bersifat opsional sampai baseline interaksi dan debiasing wajib selesai.

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
checkpoints/<run_name>/train.log
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

Output evaluasi disimpan ke `eval_test/` di folder checkpoint, termasuk `eval.log`, `metrics.json`, `test_predictions.csv`, dan `confusion_matrix.png`. Untuk smoke test cepat:

```bash
python evaluate_test.py --max-samples 64
```

Untuk menentukan lokasi log sendiri:

```bash
python evaluate_test.py --log-file outputs/eval.log
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

## Pipeline Official ndkhanh360/CAER

Pipeline alternatif dari `https://github.com/ndkhanh360/CAER` tersedia sebagai submodule di `third_party/CAER/`. Setelah clone repo ini di mesin baru, ambil submodule dengan:

```bash
git submodule update --init --recursive
```

Untuk reproduksi historis upstream, siapkan symlink data default:

```bash
python run_caer_official.py prepare
```

Untuk seluruh eksperimen riset baru, gunakan split content-disjoint yang berversi. Generator tidak mengubah dataset mentah dan mencatat semua sampel duplikat yang dikeluarkan:

```bash
python prepare_content_disjoint_split.py
python run_caer_official.py prepare \
  --detector-dir artifacts/protocols/caer_s_content_disjoint_v1 \
  --force
```

Training dual-GPU dengan 2x RTX 3060 pada split bersih:

```bash
python run_caer_official.py train \
  --detector-dir artifacts/protocols/caer_s_content_disjoint_v1 \
  --device 0,1 --n-gpu 2
```

Config default `configs/caer_official.json` mengikuti checkpoint pretrained upstream:

```text
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
batch_size = 128
lr_scheduler = StepLR(step_size=15, gamma=0.5)
epochs = 150
```

`detectors/` asli hanya digunakan untuk reproduksi historis. Jangan resume checkpoint lama atau memakai split historis untuk controlled comparison. Reproduksi upstream pada protokol bersih tetap memakai `val_accuracy` agar sesuai kode official; implementasi in-repo pada Phase 1 akan memakai validation macro F1 sebagai aturan seleksi penelitian.

Untuk smoke test cepat:

```bash
python run_caer_official.py train \
  --detector-dir artifacts/protocols/caer_s_content_disjoint_v1 \
  --device 0 --n-gpu 1 --epochs 1 --batch-size 8 --no-tensorboard
```

Checkpoint dan TensorBoard log disimpan oleh kode upstream di:

```text
third_party/CAER/CAER/official_runs/models/CAER_S_Official_CAERNet/<run_id>/
third_party/CAER/CAER/official_runs/log/CAER_S_Official_CAERNet/<run_id>/
```

Evaluasi checkpoint official:

```bash
python run_caer_official.py test \
  --detector-dir artifacts/protocols/caer_s_content_disjoint_v1 \
  --device 0 \
  --n-gpu 1 \
  --resume third_party/CAER/CAER/official_runs/models/CAER_S_Official_CAERNet/<run_id>/model_best.pth
```
