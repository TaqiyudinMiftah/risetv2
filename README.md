# CAER-S Research: CAER-Net to CD-ICA-Net

Repository ini memakai reproduksi CAER-Net sebagai baseline untuk meneliti iterative bidirectional face-context interaction dan post-interaction debiasing pada CD-ICA-Net. Baseline fondasinya adalah:

**Context-Aware Emotion Recognition Networks**  
Lee et al., ICCV 2019

Source of truth untuk eksperimen baru berada di:

```text
caer_research/
```

`CAER_S_CAERNet_Reproduction_ipynb.ipynb` tetap dipertahankan sebagai demo dan
referensi checkpoint lama. Business logic reusable tidak lagi ditambahkan ke
notebook. Package memisahkan dua track:

1. `CAERNet`: port bersih arsitektur upstream-community/paper;
2. `NotebookCAERNet`: kompatibilitas arsitektur dan checkpoint notebook lama.

Jangan mencampur metrik kedua track.

## Struktur Saat Ini

```text
.
├── caer_research/                            # package data/model/trainer/metrics/checkpoint
├── CAER_S_CAERNet_Reproduction_ipynb.ipynb  # demo dan legacy reference
├── detectors/                               # detector txt/pth cache, bisa di-download ulang notebook
├── configs/caer_official.json               # config untuk pipeline official ndkhanh360/CAER
├── run_caer_official.py                     # wrapper train/test pipeline official
├── third_party/CAER/                        # upstream CAER repo sebagai submodule
├── agents.d/                                # scope dan protokol eksperimen
├── paper/                                   # PDF, matrix, notes, dan source snapshots
├── protocols/                               # definisi split penelitian
├── reports/                                 # audit dan laporan reproducibility
├── validate_clean_caer.py                    # parity check upstream dan notebook
├── tests/                                    # unit test audit/protokol/refactor
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
uv pip install -e .
uv run python -m ipykernel install --user --name caer-net-reproduction --display-name "CAER-Net Reproduction"
```

Atau dengan `pip` jika environment Anda menyediakannya:

```bash
python -m pip install -r requirements.txt
```

## Clean In-Repo Modules

Validasi refactor pada validation samples nyata tanpa training atau test access:

```bash
python validate_clean_caer.py --split val --samples 2 --device cpu
```

Command membandingkan logits port clean dengan source upstream serta logits
`NotebookCAERNet` dengan definisi kelas di notebook. Kedua perbandingan harus
memiliki `max_abs_logit_difference = 0.0`. Hasil lokal disimpan di
`artifacts/refactor/clean_caer_parity.json`.

Contoh penggunaan package:

```python
from caer_research import CAERNet, CAERSTwoStreamDataset, Trainer, build_transforms
```

`Trainer` menyimpan `best.pt`, `last.pt`, `history.json`, dan `history.csv` serta
menyertakan RNG state dan early-stopping counter agar resume dapat direproduksi.

## Menjalankan Notebook Demo

1. Pastikan dataset CAER-S tersedia secara lokal.
2. Buka `CAER_S_CAERNet_Reproduction_ipynb.ipynb`.
3. Pilih kernel `CAER-Net Reproduction`.
4. Jalankan cell sampai training/evaluasi validation. Jangan menjalankan bagian
   `# 9. Evaluasi test` selama tuning atau ablation.
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

## Legacy Test Evaluator

`evaluate_test.py` dipertahankan hanya untuk audit checkpoint notebook lama dan
bukan evaluator controlled research. Jangan menjalankannya ketika memilih model,
hyperparameter, atau ablation. Evaluasi test final harus menunggu seluruh
kandidat dibekukan berdasarkan validation dan memerlukan unlock eksplisit pada
evaluator yang terjaga.

Contoh berikut hanya untuk reproduksi audit historis:

```bash
python evaluate_test.py --checkpoint checkpoints/<run_name>/best.pt
```

Jika memakai checkpoint lama di root `checkpoints/best_caernet_s.pt`, cukup jalankan:

```bash
python evaluate_test.py
```

Output evaluasi historis disimpan ke `eval_test/` di folder checkpoint. Jangan
memakai `--max-samples` dari test sebagai smoke test model baru; gunakan
`validate_clean_caer.py` pada validation.

Command historis:

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

## Pipeline Upstream-Community ndkhanh360/CAER

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

Run eksploratori seed 42 pada 2x RTX 3060:

```bash
python run_caer_official.py train \
  --config configs/experiments/caernet_upstream_content_disjoint_exploratory_seed42.json \
  --seed 42 --device 0,1 --n-gpu 2 \
  --wandb-mode offline
```

Hasil eksplorasi dan keputusan scope dirangkum di
[`reports/research_execution_roadmap.md`](reports/research_execution_roadmap.md).
Reproduksi penuh difokuskan pada CAER-Net dan CAER-Net + CCIM bila protokolnya
dapat dicocokkan. Paper lain tetap masuk literature comparison; full reproduction
GLAMOR-Net, CAHFW-Net, CLEF, DSCT, dan AGCD-Net dikerjakan setelah controlled
interaction dan debiasing ablation selesai.

Validasi seluruh config final tanpa mengalokasikan training GPU:

```bash
python run_caer_final_multiseed.py --dry-run
```

Jalankan final upstream-community baseline untuk seed 42, 43, dan 44 secara
berurutan:

```bash
python run_caer_final_multiseed.py \
  --seeds 42 43 44 \
  --device 0,1 --n-gpu 2 \
  --wandb-mode offline
```

Run final memakai budget yang dibekukan dari eksperimen seed 42: 45 epoch,
early stopping 12, dan checkpoint selection berdasarkan validation accuracy.
Seed 42 dijalankan ulang karena run sebelumnya berstatus eksplorasi. Jangan
menambahkan override hyperparameter pada final multi-seed run.

Jika proses berhenti sebelum mencapai epoch 45 atau early stopping, jangan
masukkan checkpoint parsial ke hasil final. Jalankan ulang seed tersebut dari
awal; checkpoint upstream tidak menyimpan seluruh state RNG dan penghitung early
stopping yang diperlukan untuk resume identik. Audit progres terbaru tersedia di
[`reports/experiment1_caernet_final_results.md`](reports/experiment1_caernet_final_results.md).

Setelah validation setiap checkpoint selesai, validasi hash dan hitung agregat
tiga seed dengan:

```bash
python summarize_caer_multiseed.py \
  --run-id caernet__upstream_community__seed42__20260714_224828 \
  --run-id caernet__upstream_community__seed43__20260715_110722 \
  --run-id caernet__upstream_community__seed44__20260715_130832
```

Ringkasan JSON disimpan di `artifacts/experiments/` dan laporan Markdown di
`reports/experiment1_caernet_final_results.md`.

`--detector-dir` sekarang default ke protokol content-disjoint. Launcher menolak training jika salah satu GPU terpilih memiliki memori bebas kurang dari 6000 MiB, menyimpan hash input dan config di `artifacts/experiments/<run_id>/run_metadata.json`, lalu memperbarui `experiments/registry.csv`. Gunakan `--wandb-mode online` setelah `wandb login`; API key tidak diperlukan untuk mode offline.

Config dasar `configs/caer_official.json` mengikuti checkpoint pretrained upstream:

```text
optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
batch_size = 128
lr_scheduler = StepLR(step_size=15, gamma=0.5)
epochs = 150
```

`detectors/` asli hanya digunakan untuk reproduksi historis. Jangan resume checkpoint lama atau memakai split historis untuk controlled comparison. Reproduksi upstream-community pada protokol bersih tetap memakai `val_accuracy` agar sesuai kode sumber tersebut; implementasi in-repo akan memakai validation macro F1 sebagai aturan seleksi penelitian. Seed sekarang diterapkan oleh `run_caer_upstream_train.py`, karena source upstream mengunci seed `123` saat import.

Validasi command tanpa memulai training:

```bash
python run_caer_official.py train \
  --config configs/experiments/caernet_upstream_content_disjoint_exploratory_seed42.json \
  --seed 42 --device 0,1 --n-gpu 2 --dry-run
```

Evaluasi checkpoint pada validation saja, termasuk macro F1, per-class F1, NLL, ECE, predictions, dan confusion matrix:

```bash
python evaluate_caer_official.py \
  --checkpoint third_party/CAER/CAER/official_runs/models/<experiment_name>/<run_id>/model_best.pth \
  --split val --device cuda:0
```

Jika training selesai tetapi status registry belum terbarui, rekonsiliasi tanpa mengulang training:

```bash
python run_caer_official.py reconcile \
  --run-id <run_id> \
  --validation-metrics artifacts/experiments/<run_id>/val_evaluation/metrics.json
```

Evaluator mengunci test secara default. Flag `--allow-test` hanya digunakan setelah model dan protokol final dibekukan.

Checkpoint dan TensorBoard log disimpan oleh kode upstream di:

```text
third_party/CAER/CAER/official_runs/models/<experiment_name>/<run_id>/
third_party/CAER/CAER/official_runs/log/<experiment_name>/<run_id>/
```

Evaluasi checkpoint hanya setelah model dipilih dari validation dan kandidat dinyatakan final:

```bash
python run_caer_official.py test \
  --config configs/experiments/caernet_upstream_content_disjoint_exploratory_seed42.json \
  --seed 42 \
  --device 0 \
  --n-gpu 1 \
  --resume third_party/CAER/CAER/official_runs/models/<experiment_name>/<run_id>/model_best.pth
```
