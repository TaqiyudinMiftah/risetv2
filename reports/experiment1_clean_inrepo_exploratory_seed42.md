# Experiment 1: Clean In-Repository CAER-Net Exploratory Seed 42

## Decision

The clean in-repository CAER-Net exploratory run for seed 42 passes its
pre-registered validation gate. Its selected checkpoint reaches validation
macro F1 `0.756515`, above the required `0.70` threshold. This is a
single-seed exploratory result, not a final three-seed baseline and not a
claim about CD-ICA-Net.

No logical test samples, test images, test batches, test metrics, or test
predictions were loaded or evaluated. Under the CAER-S storage convention,
logical validation records are physically stored below `CAER-S/test/`; all
results in this report are nevertheless from the manifest's logical `val`
split only.

## Frozen Run

| Field | Value |
| --- | --- |
| Run ID | `caernet__clean_inrepo__seed42__20260722_043253` |
| Track / stage | clean in-repo / exploratory |
| Protocol | `caer_s_content_disjoint_v1` |
| Seed | 42 |
| Model | exact in-repository CAER-Net port, face + context with adaptive fusion |
| Parameters | 2,390,028 |
| Selection rule | maximum logical-validation macro F1 |
| Training budget | 45 epochs; patience 12; FP32 |
| Completed epochs | 45 contiguous epochs |
| Best epoch | 42 |
| Test use for selection | false |

The run began at Git SHA
`203ae652c34625ae67a1f0243e0cef9ac78144e9`. It was intentionally
interrupted after epoch 13 and resumed from its end-of-epoch `last.pt`; the
resume event records Git SHA `4b37fe859b48c4c2a980d4b8b0ed1eb77e998cf8`.
The original run SHA remains the training provenance and is not relabelled by
the later resume-support commit.

Training used one AMD Radeon RX 6600 LE (logical device 0) under ROCm with
`HSA_OVERRIDE_GFX_VERSION=10.3.0` and `ROCR_VISIBLE_DEVICES=0`. The frozen
source config specifies two GPUs, while the saved effective runtime config
records the required one-GPU override (`n_gpu: 1`).

## Validation Result

The saved best-checkpoint validation result contains 6,965 logical validation
samples.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.755779 |
| Macro F1 (primary) | 0.756515 |
| Weighted F1 | 0.758022 |
| Neutral F1 | 0.564175 |
| NLL / loss | 1.340341 |
| ECE (15 bins) | 0.174815 |

| Class | F1 |
| --- | ---: |
| Anger | 0.741750 |
| Disgust | 0.867683 |
| Fear | 0.928291 |
| Happy | 0.693790 |
| Neutral | 0.564175 |
| Sad | 0.772384 |
| Surprise | 0.727532 |

These values are stored in the ignored local artifact
`checkpoints/caernet__clean_inrepo__seed42__20260722_043253/val_metrics.json`.
The registry row has empty test-metric columns.

## Provenance And Reproduction Check

| Item | SHA-256 / value |
| --- | --- |
| Source config | `366cc043467bf2d3588edd15138b2f0907385ca917a106382fa0248c7d69d833` |
| Effective runtime config | `1f68ae1b23bab0f3235c7eed6c6875ca3fc5a63aa352661a80439d794ace2e16` |
| Content-disjoint manifest | `f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad` |
| Train detector annotation | `fe89efc8546f4febbaf9bf71566b3b37da84e0ab34314effd2be3e176eacea82` |
| Validation detector annotation | `85372913838eef0b8123ad86a8b10388175c4952835ea6f44e28f7c3fcadf2f1` |
| Best checkpoint (`best.pt`) | `b89d7df50b4a4a22b79eaf4e02753ac3696e50accb547d12bc004cf02b43f6ab` |
| Final resume checkpoint (`last.pt`) | `4fb228b35ec4c7b1bc5e8ae9a1ace5c3991d3b02692aa3069a438ce1c1ae2453` |

A fresh process loaded only `best.pt` and constructed only the logical `val`
dataloader. It reproduced the saved validation result exactly:

| Check | Result |
| --- | ---: |
| Maximum saved-versus-recomputed metric delta | 0.0 |
| Prediction-label mismatches | 0 |
| Maximum confidence delta | 0.0 |
| Logical test accessed | false |
| Logical test split loaded | false |
| Logical test images loaded | false |

The machine-readable validation-only record is
`artifacts/experiments/caernet__clean_inrepo__seed42__20260722_043253/validation_reproduction.json`;
it is deliberately ignored with the other runtime artifacts.

## Acceptance And Next Action

The gate is passed because the validation macro F1 is at least `0.70`, the
best checkpoint and saved predictions reproduce exactly in a fresh
validation-only process, and no test evaluation occurred. The result remains
separate from the frozen upstream-community three-seed baseline; it must not
be averaged with or substituted for that track.

The clean CAER-Net final configurations for seeds 42, 43, and 44 are frozen and
verified. Do not begin their three-run training allocation until its compute
launch is explicitly authorized. Only after the clean final baseline is frozen
may Experiment 2 input ablations begin; test evaluation remains locked until a
final candidate and evaluation protocol are frozen.
