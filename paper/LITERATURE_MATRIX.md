# Literature Matrix

This matrix separates reported paper results from this repository's controlled experiments. `Protocol matched` is never inferred from a similar dataset name; it requires matching split, preprocessing, training budget, and evaluator.

## Publication And Inputs

| Paper | Year | Venue/status | Dataset | Input modalities | Backbone |
|---|---:|---|---|---|---|
| CAER-Net | 2019 | ICCV | CAER-S, CAER, AFEW | Face crop; face-masked context; optional video clips | Custom five-layer 2D/3D CNN streams |
| GLAMOR-Net | 2022 | Neural Computing and Applications | CAER-S, NCAER-S | Face crop and face-masked context | Custom CNN or ResNet-18 |
| CAHFW-Net | 2023 | IJERPH, peer reviewed, CC BY | CAER-S | Face crop and face-masked context | Dual custom five-layer CNN encoders |
| CCIM | 2023 | CVPR | EMOTIC, CAER-S, GroupWalk | Base-model subject/context plus masked-context prototypes | Base model; Places365 ResNet-152 for dictionary |
| CLEF | 2024 | CVPR | EMOTIC, CAER-S | Base-model input plus masked-subject context image | Base model; Places365 ResNet-152 context branch |
| DSCT | 2024 | ACM MM | CAER-S, EMOTIC | Full image, subject boxes, emotion labels | Deformable DETR with ResNet-18/50/101 |
| AGCD-Net | 2025 | arXiv preprint; venue unverified | CAER-S | Face crop and context | Hybrid ConvNeXt with STN and SE blocks |
| EMOTIC baseline | 2020 | IEEE TPAMI | EMOTIC | Person body box and full scene | ImageNet/Places AlexNet streams |
| EmotiCon | 2020 | CVPR | EMOTIC, GroupWalk, IEMOCAP | Face, gait/pose, masked scene, depth or agent graph | Modality networks, ABN, CNN/GCN |

## Interaction And Fusion

| Paper | Face-context interaction | Directionality | Number of interaction stages | Fusion method |
|---|---|---|---:|---|
| CAER-Net | Separate encoders; context self-attention; stream weighting only at fusion | None | 0 | Learned scalar adaptive fusion and concatenation |
| GLAMOR-Net | Pooled face vector conditions spatial context attention | Face to context | 1 | Learned face/context scalar weights and concatenation |
| CAHFW-Net | Cross-channel attention first rectifies face, then rectified face informs context | Sequential bidirectional | 2 | Shallow/deep hybrid weighting plus hierarchical fusion |
| CCIM | Does not define interaction; attaches to a fused base-model representation | Base dependent | Base dependent | Base fusion plus prototype expectation before classifier |
| CLEF | Additional context-only branch runs parallel to the base CAER model | No feature interaction added | 0 | Counterfactual subtraction at prediction level |
| DSCT | Subject and context queries decouple, select relations, and fuse across decoder layers | Context queries to subject query across layers | 6 decoder layers | Early query fusion with spatial/semantic weighting |
| AGCD-Net | Face attention controls correction strength of context bias | Face to context | 1 | Element-wise addition of face and corrected context |
| EMOTIC baseline | Independent body and scene encoders | None | 0 | Feature concatenation and joint classifier |
| EmotiCon | Separate context interpretations; no cross-attention between face and scene | None | 0 | Multiplicative modality fusion, then concatenation |

## Debiasing And Causal Evidence

| Paper | Debiasing method | Debiasing position | Causal graph | Intervention type |
|---|---|---|---|---|
| CAER-Net | None; masking encourages non-face context | N/A | No | None |
| GLAMOR-Net | New content-disjoint NCAER-S split is a robustness response, not model debiasing | Dataset construction | No | None |
| CAHFW-Net | None | N/A | No | None |
| CCIM | Context prototype dictionary with backdoor-adjustment approximation | After base fusion, before classifier | Explicit `X,S,C,Z,Y` graph | Feature-level `do(X)` approximation using weighted prototypes |
| CLEF | Estimates and subtracts adverse direct context effect | Prediction-time, after base interaction/fusion | Explicit direct/indirect-effect graph | Factual versus no-treatment counterfactual inference |
| DSCT | None | N/A | No | None |
| AGCD-Net | Learned context perturbation and face-guided correction | Raw/attended context, before fusion | Informal causal framing, no fully identified graph | Learned feature perturbation called counterfactual |
| EMOTIC baseline | None | N/A | No | None |
| EmotiCon | None | N/A | No | None |

Masking, learned perturbations, and feature shuffling are diagnostics or causal-inspired operations unless the intervention assumptions and identification strategy are validated.

## Evidence And Research Fit

| Paper | Training objective | Evaluation metrics | Reported result | Official code? | Reproduced? | Protocol matched? | Limitation | Relevance to CD-ICA-Net |
|---|---|---|---|---|---|---|---|---|
| CAER-Net | Cross-entropy | Accuracy | CAER-S 73.51% | No; community code only | Historical upstream track: 77.59% test | Partial; leaky upstream split, no 3-seed clean result | Late fusion; exact-content leakage exists in available detector split | Foundation and adaptive-fusion baseline |
| GLAMOR-Net | Cross-entropy; staged branch/fusion training | Accuracy; Stuart-Maxwell test | 77.90% custom, 89.88% ResNet-18 on CAER-S | Yes | No | No | One-way attention; original CAER-S dependency; unstable fusion reported | Required face-guides-context reference; optional full reproduction |
| CAHFW-Net | Cross-entropy with flooding | Accuracy | CAER-S 83.76% | No | No | No | Only CAER-S; no released implementation; sequential rather than iterative | Strong interaction baseline and architecture contrast |
| CCIM | Base objective plus CCIM classifier training | Accuracy/mAP | CAER-Net 73.47% to 74.81% on CAER-S | Core module only | No | No | Dictionary/pretraining sensitive; full pipeline absent | Mandatory debiasing baseline |
| CLEF | Base loss, context-branch loss, KL regularization | Accuracy/mAP | CAER-Net 73.47% to 75.86% on CAER-S | No | No | No | Extra branch; no natural context-shift test; identification assumptions not independently validated | Post-model debiasing comparator; optional full reproduction |
| DSCT | Focal classification plus L1/GIoU localization | Accuracy/mAP, parameters | CAER-S 91.81%, 39M parameters | No | No | No | Single-stage task and 8x A6000 budget are not controlled against this two-stream pipeline | Shows value of repeated fine-grained interaction; optional only |
| AGCD-Net | Label-smoothed cross-entropy plus attention loss | Accuracy | CAER-S 90.65% | No | No | No | Preprint, one dataset, learned perturbation is not an identified intervention | Nearest pre-fusion debiasing contrast; causal claim treated cautiously |
| EMOTIC baseline | Weighted discrete loss plus continuous regression | mAP, Jaccard, VAD error | EMOTIC mean AP 27.38 for body+image | Yes | No | Not applicable to CAER-S | Body rather than face; multi-label task | Supports input ablation and context motivation |
| EmotiCon | Multiplicative loss and multi-label classification | mAP | EMOTIC 35.48 mAP | No code found | No | Not applicable to CAER-S | Expensive external modalities; no direct face-context interaction | Supporting multimodal context baseline |

## Research Gap

No reviewed work above isolates all three variables under one matched CAER-S protocol: interaction direction, repeated bidirectional refinement, and debiasing position. CD-ICA-Net should therefore compare late fusion, one-way attention, bidirectional attention, iteration count, raw-feature debiasing, and post-interaction debiasing while holding the encoder, split, optimizer budget, and evaluator fixed. Robustness and calibration evidence are required; top-1 accuracy alone cannot establish reduced context dependence.
