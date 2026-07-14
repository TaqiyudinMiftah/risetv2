# CAHFW-Net

**Source:** Zhou et al., IJERPH 2023, [DOI](https://doi.org/10.3390/ijerph20021400).

1. **Problem.** Replace simple face-context concatenation with explicit complementary-feature extraction and hybrid weighting.
2. **Architecture.** Dual-branch encoders, two interaction-rectification pairs, adaptive attention, and deep fusion.
3. **Representation.** Five-layer CNNs produce face and masked-context feature maps; context also receives a spatial highlight module.
4. **Interaction.** Cross-channel attention first rectifies facial features using context. A second tier updates context using the rectified face representation.
5. **Directionality.** Sequential bidirectional interaction, rather than simultaneous symmetric cross-attention.
6. **Iteration.** Two fixed, untied tiers. It does not test recurrent `N=1,2,3` refinement.
7. **Bias measurement.** No explicit context-bias diagnostic, perturbation benchmark, or calibration analysis.
8. **Debiasing position.** None. Element recalibration is attention-based feature selection, not a confounding adjustment.
9. **Causal evidence.** No causal graph or intervention.
10. **Gap for CD-ICA-Net.** CAHFW supports the interaction hypothesis but does not isolate directionality, iteration count, or debiasing position under a controlled protocol.

**Reported results:** CAER-S accuracy increases from 73.51% for `SE+AF`, to 80.26% with cross-attention/adaptive attention, and 83.76% with element recalibration added.

**Assets:** article and license are available under CC BY 4.0. No official code, config, or checkpoint was found.
