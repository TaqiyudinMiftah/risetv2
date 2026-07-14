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

- 10 unit tests passed.
- Dry-run generated a complete command and metadata record.
- A real CPU batch produced face `[2, 3, 96, 96]`, context `[2, 3, 112, 112]`, and logits `[2, 7]`.
- No test images or test metrics were accessed by the training launcher.

## Compute Status

Run `caernet__upstream_community__seed42__20260714_042734` stopped at GPU preflight with status `blocked_compute`. GPU 0 had 1929 MiB free and GPU 1 had 2020 MiB free because VLLM occupied both cards; the required minimum is 6000 MiB per GPU. No training step ran.

Retry after the GPUs are released:

```bash
python run_caer_official.py train \
  --config configs/experiments/caernet_upstream_content_disjoint_exploratory_seed42.json \
  --seed 42 --device 0,1 --n-gpu 2 --wandb-mode offline
```
