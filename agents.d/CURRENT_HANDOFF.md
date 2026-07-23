# Current Codex Handoff

Last updated: 2026-07-23 02:44 UTC. Live process state can change after this
timestamp; verify it before acting.

## Mission and Current Phase

The research goal is CD-ICA-Net for CAER-S: iterative bidirectional
face-context interaction followed by post-interaction debiasing and adaptive or
gated fusion. Experiment 1 now has a completed, clean in-repository CAER-Net
three-seed final validation baseline. Do not implement cross-attention, CCIM,
or CD-ICA-Net until the next experiment is explicitly scoped.

Read these files before changing experiments:

1. `AGENTS.md`
2. `agents.d/CURRENT_HANDOFF.md`
3. `agents.d/LITERATURE_AND_EXPERIMENT_PLAN.md`
4. `agents.d/EXPERIMENT_PROTOCOL.md`
5. `reports/amd_rocm_migration_20260722.md`
6. `reports/experiment1_clean_inrepo_final_results.md`

## Live State and Repository

- Active checkout: `/home/taqiyudinmiftah/riset/risetv2`; branch `main`.
- Do not modify the older dirty checkout at `/home/taqiyudinmiftah/risetv2`.
- All three final runs below have `status: completed`; each has 45 contiguous
  epochs and a completed metadata record.
- GPU 0 had no KFD process at the final check. Verify `tmux list-sessions`,
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

## Next Work

1. Preserve the completed final result package and verify a clean worktree
   before changing experiments; do not stage ignored runtime artifacts.
2. Scope and freeze Experiment 2 input ablations (face-only, context-only, and
   face+context) under the same protocol/budget before allocating compute.
3. Keep logical test locked. Do not obtain a test score merely because this
   baseline is complete; final test evaluation belongs only to a declared final
   candidate and protocol.

## Verified Commands

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
  .venv/bin/python -m unittest discover -s tests -v

.venv/bin/python summarize_clean_final_multiseed.py \
  --output-json artifacts/experiments/caer_clean_final_multiseed_validation_summary.json
```
