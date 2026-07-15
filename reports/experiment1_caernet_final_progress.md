# Experiment 1: Final CAER-Net Progress

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
| 43 | `caernet__upstream_community__seed43__20260715_030113` | interrupted | 8 | 8 | not final | not evaluated | not evaluated |
| 44 | not created | pending | - | - | - | - | - |

Seed 42 was evaluated in a fresh process over all 6,965 validation samples. Its
checkpoint and detector hashes matched the frozen run config. The test split
remains locked.

Seed 43 stopped after epoch 8 without an active training process or a normal
early-stopping message. Its best validation accuracy at interruption was
`0.556692`, but this value is not a final result and must not enter aggregate
statistics. The upstream checkpoint does not preserve all RNG and early-stopping
state, so the final seed must be rerun from scratch rather than resumed.

## Required Next Run

```bash
python run_caer_final_multiseed.py \
  --seeds 43 44 \
  --device 0,1 --n-gpu 2 \
  --wandb-mode offline
```

After both runs complete, evaluate each `model_best.pth` on validation and
report mean and sample standard deviation across seeds 42, 43, and 44. Do not
evaluate test until the controlled candidate set is frozen.
