# FIRST_TASK_FOR_CODEX.md

Read:
- `AGENTS.md`
- `agents.d/RESEARCH_PLAN.md`
- `agents.d/EXPERIMENT_PROTOCOL.md`
- `agents.d/BASELINE_SCOPE.md`

Execute Phase 0 only.

Tasks:
1. Audit the existing CAER-Net checkpoint with approximately 75% accuracy.
2. Determine whether it belongs to:
   - upstream official pipeline; or
   - clean in-repo reimplementation.
3. Validate:
   - manifest split;
   - class order;
   - bounding boxes;
   - sample counts;
   - train/test overlap.
4. Compute SHA-256 for:
   - manifest;
   - train.txt;
   - val.txt;
   - test.txt;
   - config;
   - checkpoint.
5. Evaluate the checkpoint from a fresh Python process.
6. Save:
   - metrics JSON;
   - prediction CSV;
   - confusion matrix;
   - classification report.
7. Create:
   - `reports/baseline_caernet_seed42.md`;
   - `experiments/registry.csv`.
8. Add tests for:
   - detector parsing;
   - label canonicalization;
   - bbox validation;
   - checkpoint loading;
   - deterministic evaluation.
9. Report findings.
10. Stop.

Do not:
- implement cross-attention;
- implement CCIM;
- implement CD-ICA-Net;
- run full multi-seed experiments;
- tune on test data.
