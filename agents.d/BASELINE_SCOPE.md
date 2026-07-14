# BASELINE_SCOPE.md

## Mandatory Full Reproduction

Run upstream-community CAER-Net and the clean in-repository CAER-Net as separate
tracks. Reproduce CAER-Net + CCIM as a full method only when its preprocessing,
training protocol, and intervention can be matched. Otherwise, label it
`CCIM-inspired` and report the mismatch.

## Mandatory Controlled Baselines

These models test the research hypothesis under one split, backbone, optimizer,
training budget, checkpoint rule, and evaluator:

- face-only;
- context-only;
- simple concatenation and CAER-Net adaptive fusion;
- unidirectional cross-attention;
- bidirectional cross-attention;
- iterative bidirectional cross-attention;
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

An optional paper becomes a full reproduction only when official code and a
matched CAER-S protocol are available. Otherwise, retain its published number
in the literature table or implement only a clearly named component baseline.

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
