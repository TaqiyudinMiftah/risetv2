# AGCD-Net

**Source:** Devi et al., arXiv:2507.09248, [preprint](https://arxiv.org/abs/2507.09248).

1. **Problem.** Suppress instance-specific context features claimed to encode spurious emotion correlations.
2. **Architecture.** Hybrid ConvNeXt face/context encoders, self-attention, Attention Guided Causal Intervention Module (AG-CIM), addition fusion, and classifier.
3. **Representation.** Hybrid ConvNeXt adds a Spatial Transformer Network and squeeze-excitation blocks; face and context are encoded separately.
4. **Interaction.** Face features gate the correction applied to the difference between original and learned-perturbed context features.
5. **Directionality.** Face to context.
6. **Iteration.** One correction pass.
7. **Bias measurement.** The paper reports classification gains and a confusion matrix, but no swapped/shuffled-context robustness or calibration benchmark.
8. **Debiasing position.** On attended context features before face-context fusion.
9. **Causal evidence.** A learned linear perturbation is described as counterfactual, but no fully specified identification graph or real intervention validates that the feature difference isolates spurious context. Treat it as causal-inspired debiasing.
10. **Gap for CD-ICA-Net.** AGCD-Net is the nearest pre-fusion contrast. It does not test whether useful interaction should occur before debiasing or whether reciprocal iteration helps.

**Reported result:** 90.65% CAER-S accuracy; the backbone without AG-CIM reports 88.84%, and CAER-Net-S+AG-CIM reports 76.03%.

**Assets:** peer-reviewed venue and official code are unverified. The paper cites `ndkhanh360/CAER` only as a CAER-S implementation source; that is not AGCD-Net code.
