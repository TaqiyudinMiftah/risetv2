# Experiment 1: Final CAER-Net Results

## Protocol

- Track: `upstream_community`
- Protocol: `caer_s_content_disjoint_v1`
- Split: validation only (6,965 samples per seed)
- Seeds: 42, 43, 44
- Test accessed: no

## Per-Seed Results

| Seed | Best epoch | Accuracy | Macro F1 | Weighted F1 | Neutral F1 | NLL | ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 16 | 0.727064 | 0.729878 | 0.731325 | 0.540762 | 0.830470 | 0.081090 |
| 43 | 37 | 0.753912 | 0.751444 | 0.753001 | 0.572277 | 1.282266 | 0.168080 |
| 44 | 39 | 0.760804 | 0.756444 | 0.758040 | 0.569024 | 1.173349 | 0.160967 |

## Aggregate

Values are mean +/- sample standard deviation across three seeds.

| Metric | Mean +/- SD |
| --- | ---: |
| `accuracy` | 0.747260 +/- 0.017827 |
| `macro_f1` | 0.745922 +/- 0.014118 |
| `weighted_f1` | 0.747455 +/- 0.014195 |
| `nll` | 1.095362 +/- 0.235778 |
| `ece_15` | 0.136712 +/- 0.048302 |
| `neutral_f1` | 0.560688 +/- 0.017333 |

## Per-Class F1

| Class | Mean +/- SD |
| --- | ---: |
| Anger | 0.732663 +/- 0.003469 |
| Disgust | 0.840843 +/- 0.015797 |
| Fear | 0.931750 +/- 0.012069 |
| Happy | 0.678376 +/- 0.018294 |
| Neutral | 0.560688 +/- 0.017333 |
| Sad | 0.760045 +/- 0.009337 |
| Surprise | 0.717089 +/- 0.031139 |

Neutral remains the weakest class and the primary target for context-bias diagnostics. These validation results freeze the upstream-community baseline; they are not test results.
