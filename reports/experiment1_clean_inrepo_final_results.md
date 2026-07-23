# Experiment 1: Clean In-Repository CAER-Net Final Results

## Decision

The frozen clean in-repository CAER-Net baseline completed for final seeds 42,
43, and 44. All reported numbers below are from the content-disjoint logical
validation split only. This completes the three-seed clean baseline; it is not
a test result and it is not a claim about CD-ICA-Net or causal mechanisms.

No logical test samples, images, batches, metrics, or predictions were loaded.
Logical validation samples are physically stored beneath `CAER-S/test/` by the
upstream storage convention, but every evaluation used manifest `split="val"`
only.

## Protocol And Provenance

- Track / stage: clean in-repository / final.
- Protocol: `caer_s_content_disjoint_v1`.
- Model: exact CAER-Net port, face + context adaptive fusion; 2,390,028
  trainable parameters.
- Selection: maximum validation macro F1; 45 epochs; FP32; SGD with the frozen
  StepLR budget.
- Hardware: one RX 6600 LE (`cuda:0`) under ROCm with
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` and `ROCR_VISIBLE_DEVICES=0`.
- Run code SHA: `943e9592630444fed9e4b26f5260f22e447e934a`.
- Manifest SHA-256:
  `f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad`.
- Generated train/validation detector SHA-256 values:
  `fe89efc8546f4febbaf9bf71566b3b37da84e0ab34314effd2be3e176eacea82` and
  `85372913838eef0b8123ad86a8b10388175c4952835ea6f44e28f7c3fcadf2f1`.

Each run completed 45 contiguous epochs, records `test_used_for_selection:
false`, and has a completed registry row with empty test-metric columns.

| Seed | Run ID | Best epoch | Source config SHA-256 | Effective config SHA-256 | Best checkpoint SHA-256 |
| ---: | --- | ---: | --- | --- | --- |
| 42 | `caernet__clean_inrepo_final__seed42__20260722_073316` | 42 | `dfa484067059e8729e0308fd6337e1c530bfc73acc3453a3725831a328176f92` | `a73e11e91f4caa8a9ec6a194677af7c11b1b1da1b71d1ad27ab2aa70a796fe78` | `fdeddd41319de3ae7cb4c6ed31305da895223ea935edec6d0f7748f3e5a9cafe` |
| 43 | `caernet__clean_inrepo_final__seed43__20260722_073316` | 38 | `fb4caebedf7012e0cafe0665b6316f2313dded0ca0668eecb7e7eada0ed27a77` | `20cb600e47ed59547e6f8cd5ef37cf1539971bb9dcd613235ad61d85365c4a1f` | `fb0dfd8b960532d38f2e3a4066254b6e60fb19e61b08ebd5aa00f99d14b48267` |
| 44 | `caernet__clean_inrepo_final__seed44__20260722_073316` | 33 | `ea21146201355dde5c6d837b0abb1beac1b2649ff5ef77bdaf9ab91f2cf9c79b` | `ddad57040ba0934f0b83130beba8b9b31b158531c04a86296b0acae10cc4821a` | `67f12d50bc7010b033399364dbd297d5ab6a258d3129e2f2654cf3dfd60098da` |

## Per-Seed Validation Results

Each selected checkpoint was evaluated over 6,965 logical validation samples.

| Seed | Accuracy | Macro F1 | Weighted F1 | Neutral F1 | NLL / loss | ECE-15 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.755779 | 0.756515 | 0.758022 | 0.564175 | 1.340341 | 0.174815 |
| 43 | 0.781622 | 0.777142 | 0.778697 | 0.602107 | 1.072186 | 0.142776 |
| 44 | 0.772721 | 0.767930 | 0.769427 | 0.572821 | 1.084429 | 0.145302 |

## Aggregate Validation Results

Values are the mean plus/minus the sample standard deviation over the three
frozen seeds.

| Metric | Mean +/- SD |
| --- | ---: |
| Accuracy | 0.770041 +/- 0.013129 |
| Macro F1 | 0.767196 +/- 0.010333 |
| Weighted F1 | 0.768715 +/- 0.010356 |
| Neutral F1 | 0.579701 +/- 0.019880 |
| NLL / loss | 1.165652 +/- 0.151409 |
| ECE-15 | 0.154298 +/- 0.017813 |

| Class | F1 mean +/- SD |
| --- | ---: |
| Anger | 0.755128 +/- 0.016025 |
| Disgust | 0.867440 +/- 0.007802 |
| Fear | 0.938741 +/- 0.010838 |
| Happy | 0.695759 +/- 0.020876 |
| Neutral | 0.579701 +/- 0.019880 |
| Sad | 0.783698 +/- 0.014166 |
| Surprise | 0.749903 +/- 0.019660 |

## Fresh Validation-Only Reproduction

For every seed, a fresh process loaded its `best.pt`, instantiated only the
logical `val` dataset/dataloader, and reproduced the saved validation artifact
exactly: 6,965 samples, maximum metric delta `0.0`, prediction mismatches `0`,
and maximum confidence delta `0.0`. Every machine-readable record declares
`test_accessed: false`, `test_split_loaded: false`, and
`test_images_loaded: false`.

The ignored aggregate record is
`artifacts/experiments/caer_clean_final_multiseed_validation_summary.json`.
The reusable `summarize_clean_final_multiseed.py` utility reads only completed
metadata plus `val_metrics.json`; it rejects non-final/non-clean runs and any
run without an explicit test-selection lock.

## Interpretation And Next Work

The clean final baseline must remain separate from the upstream-community
baseline. Their results should not be pooled or presented as a controlled
implementation comparison because their checkpoint-selection and execution
paths differ. No causal conclusion follows from this baseline.

The next research step is Experiment 2 input ablations (face-only,
context-only, and face+context) under the same frozen protocol and comparable
training budget. The logical test split remains locked until a final candidate
and a one-time evaluation protocol are explicitly declared.
