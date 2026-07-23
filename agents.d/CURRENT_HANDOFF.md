# Current Codex Handoff

Last updated: 2026-07-23 03:09 UTC. Live process state can change after this
timestamp; verify it before acting.

## Mission and Current Phase

The research goal is CD-ICA-Net for CAER-S: iterative bidirectional
face-context interaction followed by post-interaction debiasing and adaptive or
gated fusion. Experiment 1 has a completed, clean in-repository CAER-Net
three-seed final validation baseline. Experiment 2 input ablations are now
scoped and code/config acceptance is complete; exploratory compute has not yet
been launched. Do not implement cross-attention, CCIM, or CD-ICA-Net yet.

Read these files before changing experiments:

1. `AGENTS.md`
2. `agents.d/CURRENT_HANDOFF.md`
3. `agents.d/LITERATURE_AND_EXPERIMENT_PLAN.md`
4. `agents.d/EXPERIMENT_PROTOCOL.md`
5. `reports/amd_rocm_migration_20260722.md`
6. `reports/experiment1_clean_inrepo_final_results.md`
7. `reports/experiment2_input_ablation_plan.md`

## Live State and Repository

- Active checkout: `/home/taqiyudinmiftah/riset/risetv2`; branch `main`.
- Do not modify the older dirty checkout at `/home/taqiyudinmiftah/risetv2`.
- All three final runs below have `status: completed`; each has 45 contiguous
  epochs and a completed metadata record.
- At the 2026-07-23 03:09 UTC check, the old final-baseline tmux pane was
  `dead` with status 0 and GPU 0 had no KFD PID. Verify `tmux list-sessions`,
  `rocm-smi --showpids`, metadata, and `history.json` before taking any action;
  do not start a duplicate run.
- Runtime directories (`CAER-S`, `artifacts`, `checkpoints`, `wandb`) are local
  and ignored. Never commit datasets, checkpoints, W&B media, or credentials.
- `experiments/registry.csv` is a legitimate runtime update from the three
  final completions; preserve it and commit it with the result package.

## AMD ROCm Environment

- Training device: RX 6600 LE (`gfx1032`, 8176 MiB), device 0 only.
- Never include device 1 (integrated `gfx1103`, 2 GiB).
- Required environment: `HSA_OVERRIDE_GFX_VERSION=10.3.0` and
  `ROCR_VISIBLE_DEVICES=0`; PyTorch uses `cuda:0` on ROCm.
- PyTorch: `2.5.1+rocm6.2`; system ROCm: 7.2.1. The RX 6600 override is a
  compatibility workaround, not official hardware support.

## Data and Protocol

- Protocol: `caer_s_content_disjoint_v1`.
- Counts: train 48,816; validation 6,965; test 13,925.
- Logical validation is physically stored under `CAER-S/test/`; that does not
  make it logical-test evaluation.
- Manifest SHA-256:
  `f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad`.
- Generated train/validation detector hashes:
  `fe89efc8546f4febbaf9bf71566b3b37da84e0ab34314effd2be3e176eacea82` and
  `85372913838eef0b8123ad86a8b10388175c4952835ea6f44e28f7c3fcadf2f1`.
- Train optimizes, logical validation selects checkpoints, and logical test
  remains locked until a final candidate and one-time evaluation protocol are
  explicitly declared.

## Completed Clean Final Baseline

Final runs use code SHA `943e9592630444fed9e4b26f5260f22e447e934a`, one RX
6600, FP32, `n_gpu: 1`, validation macro-F1 checkpoint selection, and the
frozen clean configs. Every run records `test_used_for_selection: false`.

| Seed | Run ID | Best epoch | Val. accuracy | Val. macro F1 | Neutral F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 42 | `caernet__clean_inrepo_final__seed42__20260722_073316` | 42 | 0.755779 | 0.756515 | 0.564175 |
| 43 | `caernet__clean_inrepo_final__seed43__20260722_073316` | 38 | 0.781622 | 0.777142 | 0.602107 |
| 44 | `caernet__clean_inrepo_final__seed44__20260722_073316` | 33 | 0.772721 | 0.767930 | 0.572821 |

Aggregate logical-validation result, mean +/- sample SD:

- Accuracy: `0.770041 +/- 0.013129`.
- Macro F1: `0.767196 +/- 0.010333`.
- Weighted F1: `0.768715 +/- 0.010356`.
- Neutral F1: `0.579701 +/- 0.019880`.
- NLL / loss: `1.165652 +/- 0.151409`; ECE-15: `0.154298 +/- 0.017813`.

The full report is `reports/experiment1_clean_inrepo_final_results.md`.
Keep this track separate from the upstream-community baseline; do not pool the
two tracks or present them as a controlled comparison.

## Final Acceptance Evidence

- Metadata, histories, source/effective configs, manifest, train/validation
  detector hashes, and best/last checkpoints were audited for all three runs.
- Every `val_predictions.csv` has 6,965 rows. Registry rows are `completed`
  with both test-metric columns empty.
- A fresh process loaded each `best.pt` and constructed only logical `val`.
  For all seeds: metric maximum absolute delta `0.0`, prediction mismatches
  `0`, and confidence maximum absolute delta `0.0`.
- The ignored records are under
  `artifacts/experiments/<run_id>/validation_reproduction.json`; each says
  `test_accessed: false`, `test_split_loaded: false`, and
  `test_images_loaded: false`.
- `summarize_clean_final_multiseed.py` reads only completed metadata and
  `val_metrics.json`; it outputs the ignored aggregate JSON at
  `artifacts/experiments/caer_clean_final_multiseed_validation_summary.json`.

## Frozen Upstream Baseline

The upstream-community three-seed validation baseline remains separate and is
not to be retrained: accuracy `0.747260 +/- 0.017827`, macro F1
`0.745922 +/- 0.014118`, Neutral F1 `0.560688 +/- 0.017333`. Test was not
accessed. See `reports/experiment1_caernet_final_results.md`.

## Experiment 2: Strict Input Ablation Ready

- The old `CAERNet` `use_face` / `use_context` flags are invalid as strict
  ablations: fusion weights see both streams before an inactive feature is
  zeroed. Do not flip those booleans for Experiment 2.
- `CAERNetSingleStream` is now the strict component baseline. Face-only
  constructs only the face crop/tensor; context-only constructs only the
  face-masked context tensor and retains CAER-Net's context self-attention.
  The inactive tensor is absent from the batch and has no model path.
- Frozen exploratory configs, both seed 42 / one RX 6600 / FP32 / 45 epochs /
  validation macro-F1 selection:
  - `configs/experiments/caernet_clean_input_ablation_face_only_content_disjoint_exploratory_seed42.json`
    (`32d320cf3e8a0c41cbd0f0d3458598ff2c0a1086916faa74e0d9e113c60e8792`)
  - `configs/experiments/caernet_clean_input_ablation_context_only_content_disjoint_exploratory_seed42.json`
    (`04173be7b22f69eb50d431dc24ab54bd91dc6c6993210b1fe920c34bcbe9cb53`)
- Expected parameter counts: face-only 1,014,279; context-only 1,310,730;
  completed face+context CAER-Net control 2,390,028. Treat this as an
  input/component ablation, not capacity-matched causal evidence.
- Reuse `caernet__clean_inrepo__seed42__20260722_043253` for the exploratory
  face+context control, and reuse the three completed final clean controls for
  later accepted final comparisons. Do not retrain face+context.
- Isolation tests establish exact output invariance to a supplied inactive
  tensor, dataset tests show it is not constructed, launcher/registry tests
  keep test columns blank, and the validation-only verifier supports the new
  model type. The full suite passed 66 tests. Both configs passed `--dry-run`
  without runtime output and a GPU-0 one-batch logical-validation smoke with
  logits `[128, 7]` and `test_accessed: false`.
- See `reports/experiment2_input_ablation_plan.md` for the frozen contract and
  interpretation boundary.

## Next Work

1. Commit/push the Experiment 2 implementation and acceptance package, then
   launch face-only and context-only seed-42 exploratory runs serially on GPU
   0 with distinct run IDs. Do not launch a face+context duplicate.
2. On completion, audit metadata/histories and run
   `verify_clean_validation.py` for each selected checkpoint over logical
   validation only. Promote to final seeds only after accepting those results.
3. Keep logical test locked. There is no test metric for the clean baseline;
   final test evaluation requires a declared final candidate and explicit
   one-time unlock protocol.

## Verified Commands

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
  .venv/bin/python -m unittest discover -s tests -v

.venv/bin/python summarize_clean_final_multiseed.py \
  --output-json artifacts/experiments/caer_clean_final_multiseed_validation_summary.json
```
