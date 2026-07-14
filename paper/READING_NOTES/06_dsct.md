# Decoupled Subject-Context Transformer (DSCT)

**Source:** Li et al., ACM Multimedia 2024, [DOI](https://doi.org/10.1145/3664647.3680623), arXiv:2404.17205.

1. **Problem.** Replace disconnected detection and emotion-classification stages and coarse late fusion with a single-stage model.
2. **Architecture.** Deformable DETR encoder-decoder with Decoupled Subject-Context Transformer layers and joint localization/classification supervision.
3. **Representation.** Learnable subject and context queries sample multiscale full-image features. Spatial distance and semantic relevance select context queries.
4. **Interaction.** Context queries are fused into each subject query throughout the decoder.
5. **Directionality.** Primarily context to subject; both query sets evolve across layers, but the aggregation target is the subject query.
6. **Iteration.** Fusion occurs across six decoder layers. This is depth-wise progressive interaction, not a controlled tied recurrent block.
7. **Bias measurement.** No context-bias benchmark. Query and multi-subject visualizations study context use rather than spurious dependence.
8. **Debiasing position.** None.
9. **Causal evidence.** No causal graph or intervention.
10. **Gap for CD-ICA-Net.** DSCT shows progressive fine-grained interaction can help, but changes the task, architecture, initialization, and compute budget, so it cannot isolate CD-ICA-Net's hypothesis.

**Reported result:** 91.81% CAER-S accuracy with ResNet-50 and 39M parameters; training uses eight A6000 GPUs.

**Assets:** peer-reviewed venue and DOI are verified. No official source repository, config, or checkpoint was found.
