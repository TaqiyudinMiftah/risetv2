# Literature And Paper Experiment Plan

This is the authoritative execution order for CD-ICA-Net research. The central question is whether iterative bidirectional face-context interaction followed by post-interaction debiasing improves robustness to context bias on CAER-S.

## Scientific Claims

Experiments must independently test whether:

1. bidirectional interaction improves over late fusion and one-way attention;
2. a limited number of interaction iterations improves refinement;
3. debiasing position changes the result;
4. post-interaction debiasing improves over raw-context debiasing;
5. context dependence decreases under controlled perturbations;
6. at least one of accuracy, macro F1, Neutral F1, calibration, robustness, or efficiency improves.

Masking and shuffling are diagnostics, not proof of causality. Use `causal-inspired` unless the graph, intervention, and identification assumptions are explicit and validated.

## Protocol Gate

Before training:

- use `caer_s_content_disjoint_v1` for clean experiments;
- verify split counts, class order, bbox policy, duplicate content, and overlap;
- keep logical validation samples in physical `CAER-S/test/`;
- optimize on train, select checkpoints on validation, and reserve test for finalized evaluation;
- record config, seed, git SHA, manifest/detector hashes, checkpoint, metrics, predictions, and confusion matrix;
- separate upstream historical reproduction, clean reproduction, and reported paper results.

Stop if validation provenance is ambiguous, preprocessing cannot be matched, a checkpoint cannot be reproduced, protocols differ in a claimed controlled comparison, or significant compute is required before an exploratory result is approved.

## Experiment Order

### Experiment 0: Data And Protocol Audit

Audit split sizes, distributions, labels, bbox source and validity, duplicate content, overlap, validation provenance, and hashes. Deliver `reports/data_protocol_audit.md`.

### Experiment 1: CAER-Net Reproduction

Keep upstream and clean tracks separate. Final baselines use seeds 42, 43, and 44. Report accuracy, macro/weighted/per-class F1, parameters, latency, memory, and epoch time.

### Experiment 2: Input Ablation

Compare face-only, context-only, face+context, and optionally a full-image classifier.

### Experiment 3: Fusion And Interaction

Compare concatenation, CAER adaptive fusion, face-guides-context attention, unidirectional cross-attention, bidirectional cross-attention, and iterative bidirectional cross-attention. Test `N=1,2,3`; `N=5` is optional.

### Experiment 4: Debiasing Position

Compare no debiasing, CAER-Net+CCIM, raw-context debiasing, and debiasing after uni-, bi-, and iterative interaction. Keep encoder, optimizer, split, seed, budget, and evaluator fixed.

### Experiment 5: Context Diagnostics

Evaluate original, face-only, context-only, face masked, context neutralized, shuffled context features, and swapped context features. Optional natural shift uses NCAER-S or another justified dataset. Measure accuracy/F1 drop, confidence shift, consistency, Neutral F1, NLL, and ECE.

### Experiment 6: CD-ICA-Net Ablation

Ablate iteration, bidirectionality, debiasing, pre/post position, gated fusion, auxiliary loss, and tied versus untied iteration weights.

### Experiments 7-11: Final Evidence

Run three-seed controlled comparison, calibration, bootstrap confidence intervals and paired prediction tests, efficiency profiling, and qualitative/error analysis. Final minimum models are CAER-Net, face-only, context-only, uni-/bidirectional attention, iterative attention, CAER-Net+CCIM, post-interaction debiasing, and CD-ICA-Net.

## Reproduction Scope

Mandatory full/component baselines are CAER-Net, face-only, context-only, uni-/bidirectional and iterative attention, and CAER-Net+CCIM when sufficiently matched. GLAMOR-Net, full CAHFW-Net, CLEF, DSCT, and AGCD-Net are optional until core ablations finish. Use names such as `CCIM-inspired` when only a component or unmatched approximation is implemented.

## Required Paper Package

Tables: dataset/protocol, literature results, controlled main results, interaction, debiasing position, robustness, and efficiency.

Figures: architecture, training curves, confusion matrices, per-class F1, reliability diagram, robustness degradation, face Grad-CAM, context attention, cross-attention across iterations, and Neutral error cases.

The final package must include frozen configs, a run registry, three-seed results, statistical analysis, qualitative cases where context helps or misleads, and a reproducibility appendix. A negative top-1 result remains publishable only when supported by a meaningful robustness, calibration, class-level, efficiency, or well-scoped negative finding.
