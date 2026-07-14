# RESEARCH_PLAN.md — CAER-S to CD-ICA-Net

> The authoritative scope, experiment order, deliverables, and stop conditions are in
> `agents.d/LITERATURE_AND_EXPERIMENT_PLAN.md`. This file retains the phase-level
> architecture roadmap and must not override that protocol.

## 1. Research Question

Pertanyaan utama:

> Apakah iterative bidirectional face-context interaction, diikuti debiasing setelah interaction, dapat meningkatkan robustness terhadap context bias tanpa menghilangkan context information yang benar-benar berguna?

## 2. Hypotheses

H1. Bidirectional interaction lebih informatif daripada late fusion.

H2. Iterative interaction dapat memperbaiki representasi wajah dan konteks, tetapi terlalu banyak iterasi dapat menyebabkan overfitting.

H3. Debiasing setelah interaction lebih efektif daripada debiasing pada raw feature.

H4. Manfaat utama metode usulan dapat terlihat pada macro F1, Neutral F1, calibration, dan robustness, bukan hanya top-1 accuracy.

## 3. End Goal

Penelitian selesai jika tersedia:

1. CAER-Net baseline reproducible minimal 3 seed.
2. Controlled comparison dengan baseline interaction dan debiasing.
3. Context-bias diagnostic benchmark.
4. CD-ICA-Net final.
5. Ablation lengkap.
6. Robustness, calibration, error analysis, dan interpretability.
7. Paper-ready tables, figures, dan reproducible code release.
8. Kesimpulan ilmiah yang tetap valid meskipun accuracy bukan SOTA.

## 4. Roadmap

### Phase 0 — Freeze Baseline 75%

Goal:
membekukan hasil CAER-Net saat ini sebagai baseline resmi penelitian.

Tasks:
- identifikasi checkpoint;
- tentukan apakah berasal dari upstream official atau clean reimplementation;
- simpan exact config;
- simpan seed;
- simpan optimizer/scheduler;
- hitung hash manifest, detector files, config, dan checkpoint;
- evaluasi checkpoint dari fresh process;
- validasi sample count, class order, bbox, dan split;
- buat baseline report;
- buat experiment registry.

Acceptance:
- hasil evaluasi ulang berbeda maksimal 0.1 percentage point;
- checkpoint dapat dimuat tanpa notebook state;
- tidak ada train/test leakage;
- output metrics lengkap.

### Phase 1 — Refactor Pipeline

Goal:
memindahkan business logic dari notebook ke modul Python reusable.

Extract:
- manifest parser;
- dataset;
- transforms;
- CAER-Net model;
- trainer;
- evaluator;
- checkpoint manager;
- metrics.

Notebook tetap dipertahankan sebagai demo.

Acceptance:
- checkpoint lama menghasilkan prediksi identik;
- unit test lolos;
- smoke train dan smoke evaluation lolos.

### Phase 2 — Baseline Multi-seed

Run:
- seed 42;
- seed 43;
- seed 44.

Track:
- upstream official CAER-Net;
- clean in-repo CAER-Net.

Report:
- mean ± std accuracy;
- macro F1;
- per-class F1;
- latency;
- parameter count;
- training curve.

### Phase 3 — Required Baseline Reproduction

Tidak perlu mereproduksi seluruh literatur.

Required controlled baselines:
1. CAER-Net.
2. Face-only.
3. Context-only.
4. Unidirectional cross-attention.
5. Bidirectional cross-attention.
6. Iterative bidirectional cross-attention.
7. CAER-Net + CCIM atau explicit CCIM reproduction.
8. Raw-feature debiasing.
9. Post-interaction debiasing.
10. CD-ICA-Net full.

Optional:
- GLAMOR-Net;
- full CAHFW-Net;
- CLEF;
- DSCT;
- AGCD-Net.

Optional model hanya dikerjakan setelah required baselines selesai.

### Phase 4 — Context-bias Diagnostics

Conditions:
- original face + context;
- face-only;
- context-only;
- face masked;
- context neutralized;
- shuffled context features;
- optional natural shift evaluation.

Measure:
- accuracy;
- macro F1;
- per-class F1;
- confidence shift;
- calibration;
- prediction consistency;
- Neutral confusion.

Decision gate:
jika context bias tidak terbukti, jangan memaksakan causal claim.

### Phase 5 — Interaction Ablation

Implement:
- adaptive fusion;
- unidirectional cross-attention;
- bidirectional cross-attention;
- iterative cross-attention.

Iterations:
- N=1;
- N=2;
- N=3;
- optional N=5.

Each block:
- projection;
- attention;
- residual;
- normalization;
- feed-forward;
- dropout;
- optional attention maps.

### Phase 6 — Debiasing Ablation

Compare:
- no debiasing;
- debiasing on raw context feature;
- debiasing after unidirectional interaction;
- debiasing after bidirectional interaction;
- debiasing after iterative interaction.

Semua varian harus memakai protocol identik.

### Phase 7 — Final CD-ICA-Net

Architecture:

```text
Face image -> Face encoder -> Face tokens -----------┐
                                                     |
Context image -> Context encoder -> Context tokens --┤
                                                     v
              Iterative bidirectional cross-attention
                   face <- context; context <- face
                              repeat N
                                 |
                      enriched representations
                                 |
                     post-interaction debiasing
                                 |
                       adaptive/gated fusion
                                 |
                          emotion classifier
```

Config options:
- num_iterations;
- num_heads;
- embedding_dim;
- tied_weights;
- debiasing_enabled;
- debiasing_position;
- fusion_type;
- auxiliary_loss_weight.

Required outputs:
- logits;
- optional causal_logits;
- fusion_weights;
- optional attention maps;
- embeddings;
- debiasing diagnostics.

### Phase 8 — Final Experiments

Minimum main table:
- CAER-Net;
- face-only;
- context-only;
- unidirectional CA;
- bidirectional CA;
- iterative CA;
- CAER-Net + CCIM;
- raw-feature debiasing;
- post-interaction debiasing;
- CD-ICA-Net.

Final candidates:
- minimum 3 seeds;
- same split;
- same evaluator;
- matched training budget.

Metrics:
- accuracy;
- macro F1;
- weighted F1;
- per-class F1;
- Neutral F1;
- NLL;
- ECE;
- robustness drop;
- parameters;
- latency;
- peak memory.

### Phase 9 — Paper Package

Required:
- main results table;
- ablation table;
- efficiency table;
- training curves;
- confusion matrices;
- Grad-CAM or attention maps;
- Neutral error analysis;
- context-helping examples;
- context-misleading examples;
- architecture diagram;
- reproducibility appendix.

Paper must answer:
1. Does bidirectional interaction help?
2. Does iteration help?
3. Does post-interaction debiasing help?
4. Which classes benefit?
5. Under what shift does it help?
6. What trade-offs and failure cases occur?
