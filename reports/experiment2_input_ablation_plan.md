# Experiment 2: Strict Input-Ablation Plan

## Decision

Experiment 2 will compare strict face-only and context-only component baselines
against the completed clean CAER-Net face+context adaptive-fusion control. The
face+context control is reused; it is not retrained. This avoids duplicate
compute and preserves the completed three-seed baseline as the reference.

The former `CAERNet` `use_face` and `use_context` flags are not used for this
experiment. They zero a feature only after both features have contributed to
the adaptive fusion weights, so flipping either flag would not be an isolated
input ablation.

## Frozen Exploratory Variants

| Variant | Model / data contract | Trainable parameters | Frozen config SHA-256 |
| --- | --- | ---: | --- |
| face-only | `CAERNetSingleStream(modality="face")`; dataset materializes only `face` | 1,014,279 | `32d320cf3e8a0c41cbd0f0d3458598ff2c0a1086916faa74e0d9e113c60e8792` |
| context-only | `CAERNetSingleStream(modality="context")`; dataset materializes only face-masked `context` | 1,310,730 | `04173be7b22f69eb50d431dc24ab54bd91dc6c6993210b1fe920c34bcbe9cb53` |
| face+context control | completed clean CAER-Net adaptive fusion | 2,390,028 | existing frozen run artifacts |

Frozen exploratory files:

- `configs/experiments/caernet_clean_input_ablation_face_only_content_disjoint_exploratory_seed42.json`
- `configs/experiments/caernet_clean_input_ablation_context_only_content_disjoint_exploratory_seed42.json`

Both new variants use the same content-disjoint manifest, transformations,
global batch size 128, SGD/StepLR schedule, 45-epoch budget, FP32, patience 12,
and validation macro-F1 selection rule as the completed clean baseline. They
run on one RX 6600 only (`ROCR_VISIBLE_DEVICES=0`, device 0) and explicitly set
`test_during_training: false`.

## Control Provenance

The exploratory seed-42 control is the completed run
`caernet__clean_inrepo__seed42__20260722_043253`, with source-config SHA-256
`366cc043467bf2d3588edd15138b2f0907385ca917a106382fa0248c7d69d833`.
For a final three-seed comparison after the exploratory gate, reuse:

- `caernet__clean_inrepo_final__seed42__20260722_073316`
- `caernet__clean_inrepo_final__seed43__20260722_073316`
- `caernet__clean_inrepo_final__seed44__20260722_073316`

The default face+context `CAERNet` path and state-dict layout remain unchanged;
the added single-stream implementation is a separate model type.

## Isolation and Protocol Gates

- A face-only model is exactly invariant to changing or omitting context;
  context-only is symmetric for face.
- The single-stream dataset does not crop or mask the inactive modality, so an
  inactive tensor is absent from the training and validation batch.
- A fresh validation-only verifier can instantiate either model type and is
  still hard-coded to logical `split="val"`.
- The launcher rejects legacy boolean modality flags as an ablation, records
  model type/variant/modalities in metadata, and leaves registry test columns
  blank.
- Dry runs create no metadata or checkpoint artifacts. Both frozen configs
  completed a no-training dry run and a ROCm GPU-0 one-batch logical-validation
  smoke successfully: face-only emitted `[128, 7]` from `[128, 3, 96, 96]`,
  while context-only emitted `[128, 7]` from `[128, 3, 112, 112]`. Both smoke
  records declared `test_accessed: false`.

Protocol remains `caer_s_content_disjoint_v1` (manifest SHA-256
`f18178e2dc374a7153cf08642bcd0408264186475326bcd350a83dd4569c29ad`), with
train/validation detector hashes
`fe89efc8546f4febbaf9bf71566b3b37da84e0ab34314effd2be3e176eacea82` and
`85372913838eef0b8123ad86a8b10388175c4952835ea6f44e28f7c3fcadf2f1`.
Logical test remains locked. Logical validation is the frozen `split="val"`
subset even though its images are physically stored beneath the upstream
`CAER-S/test/` directory.

## Interpretation and Next Compute Gate

These are input/component ablations, not capacity-matched causal evidence:
the strict single-stream models have fewer parameters than the full control and
context-only retains CAER-Net's context self-attention. Any observed difference
therefore quantifies performance under available input components within this
controlled budget; it must not be described as a causal effect.

After code acceptance, run the two seed-42 variants serially, audit their
metadata and histories, and reproduce each selected checkpoint on logical
validation only. Promote a variant to final seeds 42/43/44 only after that
exploratory result is accepted. Do not open or report a logical-test metric
during this sequence.
