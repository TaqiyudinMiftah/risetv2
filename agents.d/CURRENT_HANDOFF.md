# Current Codex Handoff

Last updated: 2026-07-22 05:20 UTC. Live process state can change after this
timestamp; verify it before acting.

## Mission and Current Phase

The research goal is CD-ICA-Net for CAER-S: iterative bidirectional
face-context interaction followed by post-interaction debiasing and adaptive or
gated fusion. The immediate work is still **Experiment 1 / clean in-repository
CAER-Net exploratory seed 42**. Do not implement cross-attention, CCIM, or
CD-ICA-Net until this clean baseline passes its pre-registered validation gate.

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
user changes. All code/document changes must be committed and pushed to GitHub.

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

## Active Clean Run

- tmux session: `caer-clean-s42-rocm`
- run ID: `caernet__clean_inrepo__seed42__20260722_043253`
- config:
  `configs/experiments/caernet_clean_content_disjoint_exploratory_seed42.json`
- config SHA-256:
  `366cc043467bf2d3588edd15138b2f0907385ca917a106382fa0248c7d69d833`
- output: `checkpoints/caernet__clean_inrepo__seed42__20260722_043253/`
- metadata:
  `artifacts/experiments/caernet__clean_inrepo__seed42__20260722_043253/run_metadata.json`
- launcher log:
  `artifacts/launch_logs/caernet__clean_inrepo__seed42__20260722_043253.log`
- W&B: offline.
- Hardware: one RX 6600; global batch 128; FP32.
- Optimizer: SGD, LR 0.01, momentum 0.9, Nesterov, no weight decay.
- Scheduler: StepLR, step 15, gamma 0.5.
- Budget: 45 epochs; early-stopping patience 12.
- Selection: maximum validation macro F1.

The run was intentionally interrupted after epoch 13 and is not completed.
The saved `last.pt` is an end-of-epoch checkpoint with history for epochs 1--13,
best validation macro F1 `0.577346` at epoch 8, and early-stopping count 5.
Its SHA-256 is
`56848a54ed329fe7468aeceac8c12ed9713c60ec731afc94e107a4a55997b832`.
The `best.pt` SHA-256 is
`7b69aa5535934443afe1b82ece01602bd9b42df6d3f67991130681722eaf2df8`.
This is historical observation, not proof of a current process state.

Verify before launching anything:

```bash
cd /home/taqiyudinmiftah/riset/risetv2
tmux list-sessions
tail -n 30 checkpoints/caernet__clean_inrepo__seed42__20260722_043253/train.log
tail -n 30 artifacts/launch_logs/caernet__clean_inrepo__seed42__20260722_043253.log
sed -n '1,240p' artifacts/experiments/caernet__clean_inrepo__seed42__20260722_043253/run_metadata.json
rocm-smi --showuse --showmeminfo vram
```

Do not launch a duplicate if the process is active. If tmux is absent, inspect
`history.csv`, `best.pt`, `last.pt`, metadata status, and launcher log before
deciding whether the run completed or was interrupted. Do not fabricate a
completion status.

The clean launcher now has a guarded state-resume path. It accepts only the
same run's `checkpoints/<run_id>/last.pt`, verifies the source config hash,
effective runtime config (including the one-GPU override), manifest hash,
protocol, seed, contiguous history, and model/optimizer/scheduler/RNG/DataLoader
generator state. It preserves the original run provenance and records every
resume event. A CPU split-vs-uninterrupted trajectory test covers the checkpoint
contract; ROCm execution remains deterministic-state resume rather than a claim
of bitwise ROCm determinism.

For the intentionally interrupted run, first verify no tmux trainer or GPU
process is active, then correct the stale runtime status:

```bash
.venv/bin/python run_caer_clean.py mark-interrupted \
  --run-id caernet__clean_inrepo__seed42__20260722_043253 \
  --reason "operator-requested interruption after epoch 13"
```

Then resume exactly that run, never `best.pt` and never a new run ID:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
  .venv/bin/python run_caer_clean.py train \
  --config configs/experiments/caernet_clean_content_disjoint_exploratory_seed42.json \
  --seed 42 --device 0 --n-gpu 1 --wandb-mode offline \
  --resume checkpoints/caernet__clean_inrepo__seed42__20260722_043253/last.pt
```

`--run-id`, if supplied, must match the run ID encoded in `last.pt`'s parent
directory. `KeyboardInterrupt` is recorded as `interrupted` with the latest
valid checkpoint instead of leaving metadata falsely `running`.

## Acceptance Gate and Next Work

After the clean run ends:

1. Confirm metadata status is `completed` and inspect `val_metrics.json`,
   `val_predictions.csv`, `history.csv`, and `best.pt`.
2. Reload `best.pt` in a fresh Python process and reproduce the saved full
   validation metrics and predictions.
3. Verify config, manifest, checkpoint, and detector hashes.
4. Pass the exploratory gate only if validation macro F1 is at least `0.70`.
5. Update `experiments/registry.csv` and a clean-baseline result report, then
   commit and push.
6. If the gate passes, freeze clean CAER-Net final configs for seeds 42, 43,
   and 44. Keep this track separate from upstream-community results.
7. Only after the clean baseline is frozen, implement Experiment 2 input
   ablations: face-only, context-only, and face+context under the same protocol.

If macro F1 is below `0.70`, stop and audit preprocessing, labels, bbox/crop,
transforms, optimizer behavior, and validation evaluation. Do not tune against
test and do not proceed directly to interaction/debiasing models.

## Verified Commands

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
  .venv/bin/python -m unittest discover -s tests -v

HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  .venv/bin/python run_caer_clean.py train \
  --config configs/experiments/caernet_clean_content_disjoint_exploratory_seed42.json \
  --seed 42 --device 0 --n-gpu 1 --wandb-mode disabled --smoke-only
```

The target server passed all 34 unit tests and the smoke forward with face
shape `[128, 3, 96, 96]`, context shape `[128, 3, 112, 112]`, logits shape
`[128, 7]`, and `test_accessed: false`.
