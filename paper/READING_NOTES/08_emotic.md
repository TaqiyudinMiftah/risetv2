# EMOTIC Baseline

**Source:** Kosti et al., IEEE TPAMI 2020, [arXiv](https://arxiv.org/abs/2003.13401), DOI:10.1109/TPAMI.2019.2916866.

1. **Problem.** Establish a natural-image emotion dataset and quantify the value of scene context beyond a person's visible body.
2. **Architecture.** Independent body and whole-image CNNs followed by a fusion network for discrete and continuous emotion prediction.
3. **Representation.** The person bounding box encodes body cues; a Places-pretrained scene stream encodes the full image.
4. **Interaction.** No cross-attention. Body and scene features are concatenated for joint prediction.
5. **Directionality.** None.
6. **Iteration.** None.
7. **Bias measurement.** Input ablation measures context utility, not spurious context dependence.
8. **Debiasing position.** None.
9. **Causal evidence.** No causal graph or intervention.
10. **Gap for CD-ICA-Net.** The paper motivates face/context separation and input ablations, but its body-centric multi-label protocol is not a controlled CAER-S baseline.

**Reported result:** the best body+image variant reports 27.38 mean AP over 26 EMOTIC categories, versus 23.86 for the compared body-only variant.

**Assets:** official repository `paper/code_sources/emotic`, commit `69c3a5106aed08121cd12f6a5b359c745136931e`. Code is MIT; the dataset is restricted to non-commercial research and education.
