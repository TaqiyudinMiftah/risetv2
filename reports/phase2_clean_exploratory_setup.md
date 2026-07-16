# Phase 2: Clean In-Repo Exploratory Setup

## Objective

Run one clean in-repository CAER-Net seed before allocating final three-seed
compute. This run validates the reusable trainer under the frozen
`caer_s_content_disjoint_v1` protocol; it does not replace or merge with the
upstream-community results.

## Frozen Setup

- Model: clean `CAERNet`, face and context enabled, adaptive fusion.
- Seed: 42.
- Hardware: two RTX 3060 GPUs through `torch.nn.DataParallel`.
- Batch size: 128 total.
- Optimizer: SGD, learning rate 0.01, momentum 0.9, Nesterov, no weight decay.
- Scheduler: StepLR, step size 15, gamma 0.5.
- Budget: at most 45 epochs, early-stopping patience 12.
- Checkpoint selection: maximum validation macro F1.
- Precision: FP32.
- W&B: offline.
- Test access: forbidden and not constructed by the launcher.

## Pre-Registered Gate

The clean pipeline passes the exploratory performance gate when validation macro
F1 is at least `0.70`, all hashes match, and checkpoint reload reproduces the
saved validation metrics. A lower result triggers a train/validation pipeline
audit; it does not permit test evaluation or ad hoc test-guided tuning.

The upstream-community seed 42 reference is validation accuracy `0.727064` and
macro F1 `0.729878`. Exact equality is not expected because the clean trainer
selects macro F1 and preserves a different deterministic training state.

## Structural Verification

- Config dry-run: passed.
- Dual-GPU validation forward: passed.
- Face batch: `[128, 3, 96, 96]`.
- Context batch: `[128, 3, 112, 112]`.
- Logits: `[128, 7]`.
- Test accessed: no.
