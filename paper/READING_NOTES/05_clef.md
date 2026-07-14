# Robust Emotion Recognition in Context Debiasing (CLEF)

**Source:** Yang et al., CVPR 2024, [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Robust_Emotion_Recognition_in_Context_Debiasing_CVPR_2024_paper.html), arXiv:2403.05963.

1. **Problem.** Remove harmful direct context effects without discarding useful context paths in a base CAER model.
2. **Architecture.** A non-invasive context-only branch is trained beside the base model; inference combines factual and no-treatment outputs.
3. **Representation.** The base model keeps its normal subject/context representation. A Places365 ResNet-152 receives images with the target subject masked.
4. **Interaction.** CLEF does not add face-context feature interaction; it operates in parallel with whatever interaction the base model has.
5. **Directionality.** Not applicable at feature level.
6. **Iteration.** None.
7. **Bias measurement.** Gains, per-class results, and qualitative corrections are reported. There is no dedicated shuffled/swapped-context or natural-shift evaluation.
8. **Debiasing position.** At prediction time, after the base model's interaction and fusion, by subtracting an estimated direct context effect.
9. **Causal evidence.** CLEF defines direct and indirect paths and counterfactual no-treatment inference. Ablations test the context branch, KL term, masking, and no-treatment embedding; they do not independently validate all identification assumptions.
10. **Gap for CD-ICA-Net.** CLEF motivates post-interaction debiasing but does not debias an enriched bidirectional representation inside the feature pipeline.

**Reported results:** CAER-Net improves from 73.47% to 75.86% on CAER-S. The paper's separate CCIM comparison reports 74.81% for CAER-Net+CCIM.

**Assets:** no official implementation, config, or checkpoint was found on the paper page or author repositories during this audit.
