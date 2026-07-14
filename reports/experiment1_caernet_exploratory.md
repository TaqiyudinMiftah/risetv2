# Experiment 1: CAER-Net Exploratory Run

## Scope

This run prepares the upstream-community `ndkhanh360/CAER` implementation for an exploratory CAER-Net baseline. It uses `caer_s_content_disjoint_v1`, seed 42, validation accuracy for checkpoint selection, and does not evaluate the test split during training.

Frozen config:
`configs/experiments/caernet_upstream_content_disjoint_exploratory_seed42.json`

Planned budget:

- 2x RTX 3060 through `torch.nn.DataParallel`;
- global batch size 128;
- SGD, learning rate 0.01, momentum 0.9;
- StepLR at epoch 15 with gamma 0.5;
- maximum 45 epochs and early stopping patience 12;
- TensorBoard plus W&B offline logging.

## Provenance

The launcher records the seed, git SHA, frozen/generated config hashes, and these detector hashes:

| Input | SHA-256 |
| --- | --- |
| Manifest | `f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad` |
| Train detector | `fe89efc8546f4febbaf9bf71566b3b37da84e0ab34314effd2be3e176eacea82` |
| Validation detector | `85372913838eef0b8123ad86a8b10388175c4952835ea6f44e28f7c3fcadf2f1` |
| Test detector | `e1941ea5000c0300092855a59b9c3567f0592b362264e4664cdc1bd617de96e2` |

The source submodule remains unmodified. `run_caer_upstream_train.py` resets the upstream hard-coded seed after import and before model/dataloader construction.

## Verification

- 16 unit tests passed after adding checkpoint/reconciliation and evaluation coverage.
- Dry-run generated a complete command and metadata record.
- A real CPU batch produced face `[2, 3, 96, 96]`, context `[2, 3, 112, 112]`, and logits `[2, 7]`.
- Fresh-process validation evaluated all 6,965 samples and matched the frozen detector hash.
- No test images or test metrics were accessed.

## Completed Run

Run ID: `caernet__upstream_community__seed42__20260714_142807`

Training selected epoch 16 and stopped at epoch 29 after 12 epochs without validation improvement. The checkpoint was evaluated independently using sample-weighted metrics:

| Metric | Validation result |
| --- | ---: |
| Accuracy | 0.727064 |
| Macro F1 | 0.729878 |
| Weighted F1 | 0.731325 |
| Neutral F1 | 0.540762 |
| NLL | 0.830470 |
| ECE, 15 bins | 0.081090 |
| Parameters | 2,390,028 |

The community trainer reported `0.726956` because it averages per-batch accuracy without weighting the smaller final batch. Controlled results use the independent sample-weighted value above.

## Per-Class Result

| Class | F1 |
| --- | ---: |
| Anger | 0.736668 |
| Disgust | 0.823262 |
| Fear | 0.918656 |
| Happy | 0.659277 |
| Neutral | 0.540762 |
| Sad | 0.749280 |
| Surprise | 0.681240 |

Neutral is the weakest class and therefore remains a primary diagnostic target. The final training epoch reached 0.9519 train accuracy but only 0.6103 validation accuracy; this is post-selection overfitting and does not replace the epoch-16 checkpoint.

## Artifact Status

The validation metrics, predictions, classification report, and confusion matrix are under `artifacts/experiments/<run_id>/val_evaluation/` and remain Git-ignored. Registry status is `completed`; run `...142718`, which produced no checkpoint, is marked `failed_incomplete`. The earlier `...042734` attempt remains `blocked_compute` as an audit trail.

This exploratory result is sufficient to freeze the upstream-community protocol for final seeds 42, 43, and 44. Test evaluation remains locked until the final baseline protocol and candidates are declared.
