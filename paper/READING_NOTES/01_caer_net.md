# CAER-Net

**Source:** Lee et al., ICCV 2019, [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_Context-Aware_Emotion_Recognition_Networks_ICCV_2019_paper.html), arXiv:1908.05913.

1. **Problem.** Recognize emotion when facial evidence alone is ambiguous by adding scene context; the work also introduces CAER and CAER-S.
2. **Architecture.** Separate face and face-masked context CNN streams, a spatial context-attention module, and adaptive fusion.
3. **Representation.** A detected face is cropped to `96x96`; the original image is face-masked, resized, and cropped to `112x112`. Static and temporal 2D/3D variants are defined.
4. **Interaction.** Face and context are encoded independently. They meet only through learned stream weights and concatenation.
5. **Directionality.** There is no feature-level face-to-context or context-to-face interaction; context attention is self-contained.
6. **Iteration.** No iterative interaction. The static CAER-Net-S has one encoding and fusion pass.
7. **Bias measurement.** No explicit context-bias or context-shift metric is reported. Masking is used to force context discovery, not to diagnose spurious context reliance.
8. **Debiasing position.** None.
9. **Causal evidence.** No causal graph or intervention. Input ablations show context utility but do not identify a causal effect.
10. **Gap for CD-ICA-Net.** CAER-Net is the late/adaptive-fusion baseline. It leaves open whether bidirectional iterative refinement and post-interaction debiasing improve robustness.

**Reported result:** CAER-Net-S reaches 73.51% CAER-S accuracy. This repository's historical upstream-style result is separate because the available detector protocol contains exact-content overlap.

**Assets:** the paper links the dataset only. `third_party/CAER` is a community reproduction and must not be described as official author code.
