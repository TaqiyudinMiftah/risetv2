# Context De-Confounded Emotion Recognition (CCIM)

**Source:** Yang et al., CVPR 2023, [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Context_De-Confounded_Emotion_Recognition_CVPR_2023_paper.html), arXiv:2303.11921.

1. **Problem.** Reduce spurious associations caused by uneven emotion distributions across contexts.
2. **Architecture.** A model-agnostic Contextual Causal Intervention Module (CCIM) is inserted before a base model's final classifier.
3. **Representation.** Subject and context form a joint feature `h`. Masked-context features from a Places365 ResNet-152 are clustered into a confounder dictionary.
4. **Interaction.** CCIM does not introduce face-context cross-attention. The joint feature queries context prototypes through dot-product or additive attention.
5. **Directionality.** Base-model dependent; prototype weighting is joint-feature to dictionary.
6. **Iteration.** No iterative interaction.
7. **Bias measurement.** The paper visualizes context-label imbalance and evaluates downstream performance, but does not provide a natural context-shift benchmark.
8. **Debiasing position.** After base-model feature fusion and before classification.
9. **Causal evidence.** An explicit `X,S,C,Z,Y` graph and backdoor-adjustment approximation are defined. Dictionary, masking, and attention ablations support implementation choices, but causal identification still depends on the prototype and graph assumptions.
10. **Gap for CD-ICA-Net.** CCIM is the mandatory causal-inspired baseline. It does not compare raw versus post-interaction placement or use iteratively enriched representations.

**Reported result:** CAER-Net improves from 73.47% to 74.81% CAER-S accuracy in the authors' reproduced protocol.

**Assets:** author-owned MIT repository `paper/code_sources/ccim`, commit `d6a651f91d1c1c91faca862ddeea915df9314919`. It releases the core module, not a full CAER-S training pipeline or checkpoint.
