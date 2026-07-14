# Phase 0 Baseline Audit: CAER-Net

## Decision

The selected checkpoint is an **upstream CAER-Net local reproduction**, not the clean in-repo notebook model. It is:

`third_party/CAER/CAER/official_runs/models/CAER_S_Official_CAERNet/0701_023049/model_best.pth`

It was selected at epoch 32 by validation accuracy `0.7674668`. The upstream `train.py` fixes the training seed to `123`; this report keeps the requested `seed42` filename because evaluation itself was run deterministically with seed `42`. It must not be described as a model trained with seed 42.

## Fresh Evaluation

The checkpoint was loaded in a new Python process with the checkpoint's exact saved config. The historical upstream protocol produced:

| Metric | Result |
| --- | ---: |
| Test accuracy | 77.59% |
| Macro F1 | 77.37% |
| Weighted F1 | 77.29% |
| Neutral F1 | 59.22% |
| NLL | 1.0107 |
| ECE (15 bins) | 0.1356 |
| Parameters | 2,390,028 |
| End-to-end latency | 10.98 ms/sample |

Per-class F1: Anger 76.99%, Disgust 86.30%, Fear 94.25%, Happy 70.95%, Neutral 59.22%, Sad 79.07%, Surprise 74.83%.

Artifacts are stored locally in `artifacts/phase0/caernet__upstream_official__seed123__historical_protocol/`: `metrics.json`, `predictions.csv`, `classification_report.json`, `confusion_matrix.csv`, `confusion_matrix.png`, `manifest_audit.json`, and input hashes. They are intentionally Git-ignored.

## Provenance

- Training and audit Git SHA: `aaf26453feaa46c18713c13716b9bafae83589aa` (the training SHA is inferred from the generated-config timestamp).
- Manifest SHA-256: `0821b1cecfcc5783e1270fe6cfeee1c0edff85226c8058aae5806e9bae5ac289`.
- Run config SHA-256: `2f94ed50fbaca4bfcc26a64ebb9b7b6855c7fd4b76b055cf4d1157c0b3b70f59`.
- Checkpoint SHA-256: `aaa815ec94d10d37c9e133eabcb6af9ccc3c285bc336ba6d3062fa3d4e9a04b1`.
- Detector SHA-256 values are saved in the artifact `input_hashes.json`.

The manifest has 48,850 train, 6,971 validation, and 13,942 test entries. It matches `train.txt`, `val.txt`, and `test.txt` exactly in order, label index, and raw bbox. Canonical class order is Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## Blocking Finding

This checkpoint is **not frozen as a clean research baseline**. Exact image-content hashes reveal 6 train-validation overlaps, 13 train-test overlaps, and 2 validation-test overlaps. There are also 34 duplicate train images and 3 duplicate test images within their respective splits. These are not filename aliases: the files are byte-identical. The historical metric above is retained only for upstream-protocol reproducibility and must not be used for model selection or controlled claims.

There are 333 detector boxes outside their source image bounds. They remain valid for the upstream baseline because `PIL.Image.crop` pads out-of-bounds regions; changing them would invalidate comparison with the official pipeline.

## Clean Protocol Follow-up

The data gate is now implemented as `caer_s_content_disjoint_v1`. It retains 48,816 train, 6,965 validation, and 13,925 test samples after removing 57 duplicates with fixed `train -> val -> test` priority. An independent audit reports zero train-validation, train-test, and validation-test content-hash overlap. Its generated manifest SHA-256 is `aa5d592c4c9a9cad556efa470c8fafc87c3fb6c076608b1dfb31509c20e90d32`.

This resolves the dataset-protocol gate, not the baseline-freeze acceptance. The next model run must train CAER-Net from scratch on `caer_s_content_disjoint_v1` and evaluate test only after validation-based selection. The upstream reproduction keeps its official `val_accuracy` selector; the clean in-repo implementation will adopt validation macro F1 during Phase 1. No Phase 1 refactor, multi-seed run, interaction baseline, or proposed CD-ICA-Net component has been started.
