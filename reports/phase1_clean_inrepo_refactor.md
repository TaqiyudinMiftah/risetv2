# Phase 1: Clean In-Repo Refactor

## Scope

Notebook business logic is now available as reusable, tested Python modules.
No cross-attention, debiasing, component ablation, long training, or test
evaluation was performed in this phase.

The repository keeps two explicitly separated model tracks:

- `CAERNet` is a dependency-free port of the upstream-community architecture and
  preserves its checkpoint key layout.
- `NotebookCAERNet` preserves the legacy notebook architecture only for old
  checkpoint compatibility and notebook demonstrations.

## Reusable Components

- `caer_research/data.py`: frozen-manifest loading, face crop, context masking,
  label order, and official-style transforms.
- `caer_research/models/`: upstream-compatible and notebook-compatible models.
- `caer_research/engine.py`: sample-weighted train/evaluation steps.
- `caer_research/metrics.py`: accuracy, macro/weighted/per-class F1, and ECE.
- `caer_research/checkpointing.py`: DataParallel compatibility, RNG state, and
  atomic checkpoint saves.
- `caer_research/trainer.py`: validation selection, early stopping, history,
  best/last checkpoints, and deterministic resume state.

## Parity Result

Parity used two real samples from the frozen validation manifest. Test was not
accessed.

| Track | Checkpoint SHA-256 | Parameters | Logit difference | Predictions equal |
| --- | --- | ---: | ---: | --- |
| Upstream-community port | `84892ec179848ac7e9fb6f423d3a2a20ea609a8eb75108a3f5ceceef461d1d00` | 2,390,028 | 0.0 | yes |
| Notebook compatibility | `e7d72d788e00856b4c74a2033e02fa078387b11d5e1d3db8fafbbf12d15a6ccf` | 2,387,401 | 0.0 | yes |

Input shapes were face `[2, 3, 96, 96]`, context `[2, 3, 112, 112]`, and logits
`[2, 7]`. The manifest SHA-256 was
`f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad`.

## Gate

The structural refactor and checkpoint parity gate passed. The next phase may
freeze the clean in-repo training config and run an exploratory seed before
allocating three-seed compute. Upstream and clean in-repo results must remain in
separate tables.
