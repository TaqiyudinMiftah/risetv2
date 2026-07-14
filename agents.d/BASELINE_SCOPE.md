# BASELINE_SCOPE.md

## Required Reproduction

The research must reproduce only the baselines needed to test the hypothesis.

Required:
- CAER-Net;
- face-only;
- context-only;
- unidirectional cross-attention;
- bidirectional cross-attention;
- iterative bidirectional cross-attention;
- CAER-Net + CCIM;
- raw-feature debiasing;
- post-interaction debiasing;
- CD-ICA-Net.

## Optional Reproduction

Optional:
- GLAMOR-Net;
- full CAHFW-Net;
- CLEF;
- DSCT;
- AGCD-Net.

Optional methods are implemented only after:
- baseline freeze;
- multi-seed CAER-Net;
- context diagnostics;
- core interaction ablations;
- core debiasing ablations.

## Literature Result Policy

Results copied from papers must be placed in a separate table.

Required columns:
- model;
- reported metric;
- source;
- reproduced?;
- protocol matched?;
- notes.

Do not present reported paper results as controlled comparison.

## Naming Rule

If only a component is reproduced, use:
- `CAHFW-inspired cross-attention`;
- `CCIM-inspired debiasing`.

Do not claim full reproduction unless architecture, preprocessing, training protocol, and evaluation have been matched.
