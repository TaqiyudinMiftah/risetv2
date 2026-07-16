# Experiment 1: Final CAER-Net Training Audit

## Protocol

- Track: `upstream_community`
- Protocol: `caer_s_content_disjoint_v1`
- Seeds: 42, 43, 44
- Training budget: 45 epochs with early stopping 12
- Selection: maximum validation accuracy
- Test accessed: no

## Run Audit

| Seed | Run ID | Status | Latest epoch | Best epoch | Validation accuracy | Macro F1 | Neutral F1 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | `caernet__upstream_community__seed42__20260714_224828` | completed | 29 | 16 | 0.727064 | 0.729878 | 0.540762 |
| 43 | `caernet__upstream_community__seed43__20260715_110722` | completed | 45 | 37 | 0.753912 | 0.751444 | 0.572277 |
| 44 | `caernet__upstream_community__seed44__20260715_130832` | completed | 45 | 39 | 0.760804 | 0.756444 | 0.569024 |

Seed 42 was evaluated in a fresh process over all 6,965 validation samples. Its
checkpoint and detector hashes matched the frozen run config. The test split
remains locked.

An earlier seed 43 attempt, `caernet__upstream_community__seed43__20260715_030113`,
stopped at epoch 8 and remains marked `interrupted`. It is retained only as an
audit record and is excluded from aggregate statistics. The replacement run was
trained from scratch.

## Completion

All three final checkpoints were evaluated on validation using the same detector
hash and sample-weighted evaluator. Aggregate statistics and per-class results
are frozen in `reports/experiment1_caernet_final_results.md`. Do not evaluate
test until the controlled candidate set is frozen.
