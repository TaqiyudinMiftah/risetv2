# Current Codex Handoff

Last updated: 2026-07-22 07:23 UTC. Live process state can change after this
timestamp; verify it before acting.

## Mission and Current Phase

The research goal is CD-ICA-Net for CAER-S: iterative bidirectional
face-context interaction followed by post-interaction debiasing and adaptive or
gated fusion. The clean in-repository CAER-Net exploratory seed-42 run has
completed and passed its pre-registered validation gate. The clean CAER-Net
final seed-42/43/44 configurations are frozen and verified; their three-run
compute allocation awaits explicit launch authorization. Do not implement
cross-attention, CCIM, or CD-ICA-Net before the clean final baseline is frozen.

Read these files before changing experiments:

1. `AGENTS.md`
2. `agents.d/CURRENT_HANDOFF.md`
3. `agents.d/LITERATURE_AND_EXPERIMENT_PLAN.md`
4. `agents.d/EXPERIMENT_PROTOCOL.md`
5. `reports/amd_rocm_migration_20260722.md`

## Server and Repository

- Active checkout: `/home/taqiyudinmiftah/riset/risetv2`
- Do not modify the older dirty checkout at `/home/taqiyudinmiftah/risetv2`.
- Branch: `main`; GitHub: `TaqiyudinMiftah/risetv2`.
- The active run was launched from Git SHA
  `203ae652c34625ae67a1f0243e0cef9ac78144e9`.
- Later commits only add migration reporting/handoff material. Do not relabel
  the run with a newer SHA.
- Runtime directories (`CAER-S`, `artifacts`, `checkpoints`, `wandb`) are local
  and ignored. Never commit datasets, checkpoints, W&B media, or credentials.

Before pulling, inspect `git status`. The trainer updates
`experiments/registry.csv`; preserve that runtime update and never reset other
user changes. The Acceptance Gate implementation is committed and pushed as
`25d05f5fc7339332c84e857849909e56b064f201`. All later code/document changes
must also be committed and pushed to GitHub.

## AMD ROCm Environment

- OS: Ubuntu 22.04.5; Python 3.12.13.
- Training device: RX 6600 LE, `gfx1032`, 8176 MiB VRAM, device 0.
- Never include device 1: it is an integrated `gfx1103` GPU with only 2 GiB.
- System ROCm: 7.2.1. PyTorch: `2.5.1+rocm6.2`; torchvision:
  `0.20.1+rocm6.2`.
- Required compatibility override: `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- The RX 6600 setup is an unsupported compatibility workaround, not an
  officially supported AMD configuration. Record the override in every run.
- PyTorch ROCm intentionally uses `torch.cuda` and `cuda:0`; do not replace
  these strings with `rocm:0` or `hip:0`.

Environment verification:

```bash
cd /home/taqiyudinmiftah/riset/risetv2
ROCR_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  .venv/bin/python scripts/check_accelerator.py \
  --require-backend rocm --min-devices 1
```

## Data and Protocol

- Protocol: `caer_s_content_disjoint_v1`.
- Counts: train 48,816; validation 6,965; test 13,925.
- Class order: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- Manifest:
  `artifacts/protocols/caer_s_content_disjoint_v1/manifest.jsonl`.
- Manifest SHA-256:
  `f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad`.
- Dataset filename/size inventory SHA-256:
  `40437886f4927e0584ebc23a64784204e32c4a46a10f60a9a74c54cb228874cb`.
- `CAER-S` is a symlink layout over the existing server dataset. It maps the
  physical `test/Anger` directory to canonical `test/Angry` without changing
  source data.
- The clean protocol has zero cross-split content-hash overlap. Historical
  detector splits contain leakage and are reproduction-only.
- Train optimizes parameters, validation selects checkpoints, and test remains
  locked until a final candidate and protocol are frozen.

## Frozen Upstream Baseline

The upstream-community three-seed validation baseline is complete and must not
be retrained unless a documented protocol defect is found. Results:

| Seed | Accuracy | Macro F1 | Neutral F1 | Best epoch |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.727064 | 0.729878 | 0.540762 | 16 |
| 43 | 0.753912 | 0.751444 | 0.572277 | 37 |
| 44 | 0.760804 | 0.756444 | 0.569024 | 39 |

Aggregate validation accuracy is `0.747260 +/- 0.017827`; macro F1 is
`0.745922 +/- 0.014118`. Test was not accessed. See
`reports/experiment1_caernet_final_results.md`.

The clean `caer_research.models.CAERNet` is an exact upstream architecture port
with 2,390,028 parameters. Checkpoint parity previously produced maximum logit
difference `0.0`; see `reports/phase1_clean_inrepo_refactor.md`.

## Completed Clean Exploratory Run

- Run ID: `caernet__clean_inrepo__seed42__20260722_043253`.
- Config:
  `configs/experiments/caernet_clean_content_disjoint_exploratory_seed42.json`.
- Output: `checkpoints/caernet__clean_inrepo__seed42__20260722_043253/`.
- Metadata:
  `artifacts/experiments/caernet__clean_inrepo__seed42__20260722_043253/run_metadata.json`.
- W&B: offline; one RX 6600; global batch 128; FP32.
- Optimizer: SGD, LR 0.01, momentum 0.9, Nesterov, no weight decay.
- Scheduler: StepLR, step 15, gamma 0.5.
- Budget: 45 epochs; early-stopping patience 12.
- Selection: maximum validation macro F1.
- Completed: `2026-07-22T06:32:46+00:00`; 45 contiguous epochs; best epoch 42.
- Metadata status: `completed`; `test_used_for_selection: false`.
- Live check at handoff update: no tmux server; RX 6600 device 0 idle.

The run was intentionally interrupted after epoch 13. One resume initialization
attempt failed before epoch 14 because its CPU RNG payload had been mapped to
CUDA; the corrected launcher restores it from CPU first. The successful resume
event is retained as recovery provenance. The resumed training reached epoch
45; it was not restarted under a duplicate run ID.

The original training Git SHA remains
`203ae652c34625ae67a1f0243e0cef9ac78144e9`. The successful resume event
records later Git SHA `4b37fe859b48c4c2a980d4b8b0ed1eb77e998cf8`; do not
relabel the original run with it.

### Completed-Artifacts Provenance

- Source config SHA-256:
  `366cc043467bf2d3588edd15138b2f0907385ca917a106382fa0248c7d69d833`.
- Effective saved runtime config (`n_gpu: 1`) SHA-256:
  `1f68ae1b23bab0f3235c7eed6c6875ca3fc5a63aa352661a80439d794ace2e16`.
- Manifest SHA-256:
  `f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad`.
- Train detector SHA-256:
  `fe89efc8546f4febbaf9bf71566b3b37da84e0ab34314effd2be3e176eacea82`.
- Validation detector SHA-256:
  `85372913838eef0b8123ad86a8b10388175c4952835ea6f44e28f7c3fcadf2f1`.
- Best checkpoint SHA-256:
  `b89d7df50b4a4a22b79eaf4e02753ac3696e50accb547d12bc004cf02b43f6ab`.
- Final `last.pt` SHA-256:
  `4fb228b35ec4c7b1bc5e8ae9a1ace5c3991d3b02692aa3069a438ce1c1ae2453`.

### Saved Validation Result

| Metric | Result |
| --- | ---: |
| Samples | 6,965 |
| Accuracy | 0.755779 |
| Macro F1 | 0.756515 |
| Weighted F1 | 0.758022 |
| Neutral F1 | 0.564175 |
| NLL / loss | 1.340341 |
| ECE (15 bins) | 0.174815 |

Per-class F1 is Anger `0.741750`, Disgust `0.867683`, Fear `0.928291`, Happy
`0.693790`, Neutral `0.564175`, Sad `0.772384`, and Surprise `0.727532`.

The saved `val_predictions.csv` has 6,965 validation rows (plus header). The
word `test` may appear in an image path because logical validation is stored
under physical `CAER-S/test/`; it does not make this logical-test evaluation.

### Fresh Validation-Only Reproduction

A fresh Python process loaded `best.pt` and constructed only the logical `val`
dataloader. It exactly reproduced the saved validation metrics and predictions:
maximum metric delta `0.0`, prediction mismatches `0`, and maximum confidence
delta `0.0`. It did not load the logical test split, logical-test images, or
batches and did not produce test metrics or predictions (`test_accessed: false`,
`test_split_loaded: false`, `test_images_loaded: false`). The ignored record is:

`artifacts/experiments/caernet__clean_inrepo__seed42__20260722_043253/validation_reproduction.json`

See `reports/experiment1_clean_inrepo_exploratory_seed42.md` for the full,
clean-track-only result report.

## Acceptance Gate and Next Work

The exploratory acceptance gate **passed**: validation macro F1 is `0.756515`
(`>= 0.70`), artifact hashes were checked, the validation-only fresh-process
reproduction is exact, and no logical-test evaluation occurred. The registry
test columns remain empty.

Next work:

1. The clean CAER-Net final configs for seeds 42, 43, and 44 are frozen and
   verified. Retain this track separately from upstream-community results.
2. Obtain or confirm authorization before allocating the final three-seed
   training compute. Do not duplicate the completed exploratory run.
3. After the clean final baseline is frozen, start Experiment 2 input ablations
   (face-only, context-only, face+context) under the same protocol.
4. Keep the logical test split locked until a final candidate and evaluation
   protocol are frozen.

## Verified Commands

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
  .venv/bin/python -m unittest discover -s tests -v

HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  .venv/bin/python run_caer_clean.py train \
  --config configs/experiments/caernet_clean_content_disjoint_exploratory_seed42.json \
  --seed 42 --device 0 --n-gpu 1 --wandb-mode disabled --smoke-only
```

The target server passed the resume-regression suite and the smoke forward with
face shape `[128, 3, 96, 96]`, context shape `[128, 3, 112, 112]`, logits shape
`[128, 7]`, and `test_accessed: false`. Re-run the suite after any code change
before launching subsequent training.
