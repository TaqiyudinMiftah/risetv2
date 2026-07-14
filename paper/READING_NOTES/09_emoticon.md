# EmotiCon

**Source:** Mittal et al., CVPR 2020, [paper and supplement](https://openaccess.thecvf.com/content_CVPR_2020/html/Mittal_EmotiCon_Context-Aware_Multimodal_Emotion_Recognition_Using_Freges_Principle_CVPR_2020_paper.html), arXiv:2003.06692.

1. **Problem.** Combine multimodal, situational, and socio-dynamic interpretations of context for perceived emotion.
2. **Architecture.** Modality encoders, masked-scene Attention Branch Network, depth CNN or agent GCN, then joint classification.
3. **Representation.** Face landmarks and gait/pose represent the target; masked RGB represents scene semantics; depth or agent graphs represent proximity.
4. **Interaction.** Streams are independently encoded. Face and gait use multiplicative fusion, while the three context interpretations are concatenated.
5. **Directionality.** No explicit face-context feature update.
6. **Iteration.** None.
7. **Bias measurement.** Context-component ablations show utility; no spurious-context diagnostic is defined.
8. **Debiasing position.** None.
9. **Causal evidence.** No causal graph or intervention.
10. **Gap for CD-ICA-Net.** EmotiCon demonstrates broad context value, but relies on costly external modalities and does not isolate bidirectional face-context interaction or debiasing.

**Reported results:** 35.48 mAP on EMOTIC and 65.83 mAP on GroupWalk. The paper does not report an original CAER-S experiment.

**Assets:** the official CVF supplement is stored locally. The paper's project page did not expose a downloadable official code repository during this audit.
