# GLAMOR-Net

**Source:** Le et al., Neural Computing and Applications, [DOI](https://doi.org/10.1007/s00521-021-06778-x), arXiv:2111.04129.

1. **Problem.** Improve context localization by conditioning global scene attention on local facial evidence; address train-test dependence through NCAER-S.
2. **Architecture.** Face and masked-context encoders, Global-Local Attention (GLA), adaptive fusion, and classifier.
3. **Representation.** A pooled face vector is paired with every spatial context vector. A shared MLP produces normalized context attention weights.
4. **Interaction.** Yes. Facial features condition the context saliency map before pooling.
5. **Directionality.** Unidirectional, face to context. Context does not update the face representation.
6. **Iteration.** One attention stage; no recurrent refinement.
7. **Bias measurement.** The paper identifies frame similarity between original CAER-S train and test sets and creates NCAER-S. It does not define context-bias interventions or calibration metrics.
8. **Debiasing position.** No model debiasing. NCAER-S is a dataset-level robustness measure.
9. **Causal evidence.** No causal graph or intervention.
10. **Gap for CD-ICA-Net.** GLA is the most direct face-guides-context baseline, but cannot test reciprocal or iterative interaction and does not debias enriched features.

**Reported results:** 77.90% CAER-S accuracy with the custom encoder and 89.88% with ResNet-18; on NCAER-S, the custom model reports 46.91%. Results are literature values, not matched comparisons.

**Assets:** official MIT repository `paper/code_sources/glamor-net`, frozen at `c12e1b97aa7354df126795f19402303a00166ec3`; it includes config and trained weights.
