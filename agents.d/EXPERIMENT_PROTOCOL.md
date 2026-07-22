# EXPERIMENT_PROTOCOL.md

## Dataset Rules

- Train split only for optimization.
- Validation split only for model selection.
- Test split only for final evaluation.
- Validation path must follow the upstream detector protocol: logical `val` samples are stored under physical `CAER-S/test/`, not `CAER-S/train/`.
- New experiments must use `protocols/caer_s_content_disjoint_v1.md`; the overlapping upstream split is historical reproduction only.
- Canonical class order:
  1. Anger
  2. Disgust
  3. Fear
  4. Happy
  5. Neutral
  6. Sad
  7. Surprise

Before training:
- verify all files exist;
- verify bbox is valid;
- verify no duplicate sample ID;
- verify no train/test overlap;
- save manifest hash.
- verify physical split provenance (`train -> train`, `val -> test`, `test -> test`).

## Model Selection

Default:
- maximize validation macro F1;
- validation accuracy as tie-breaker.

Do not use test accuracy for checkpoint selection.

## Metrics

Primary:
- accuracy;
- macro F1;
- per-class F1.

Secondary:
- weighted F1;
- precision;
- recall;
- NLL;
- ECE;
- parameter count;
- latency;
- peak GPU memory.

Robustness:
- performance drop under perturbation;
- prediction consistency;
- confidence change.

## Run Metadata

Each run saves:
- run_id;
- model;
- variant;
- seed;
- git SHA;
- config path and hash;
- manifest hash;
- detector hashes;
- checkpoint path;
- selection metric;
- best epoch;
- hardware;
- software versions.

## Run Naming

```text
<model>__<variant>__seed<seed>__<timestamp>
```

## Experiment Registry

Minimum columns:

```text
run_id,status,model,variant,seed,git_sha,config,config_sha256,
effective_config_sha256,manifest_sha256,detector_hashes,checkpoint,checkpoint_sha256,
val_accuracy,val_macro_f1,test_accuracy,test_macro_f1,
neutral_f1,params,latency_ms,notes
```

Test columns remain empty until final evaluation. For clean in-repository runs,
`detector_hashes` records only generated artifacts used for optimization or
validation while test is locked.

## Statistical Protocol

Exploratory:
- one seed;
- must be marked exploratory.

Final:
- at least three seeds;
- report mean ± std;
- preserve every individual run.

## Fair Comparison Checklist

Before comparing two models:
- same manifest;
- same split;
- same transforms;
- same input resolutions;
- same optimizer budget;
- same checkpoint rule;
- same seed count;
- same evaluator.

If not matched, label as non-controlled comparison.

## Research PR Requirements

Each PR must include:
- hypothesis;
- changed files;
- config;
- test command;
- smoke-test output;
- expected compute;
- acceptance criteria;
- no final test result unless experiment is finalized.
