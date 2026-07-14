# CAER-S Content-Disjoint v1

This protocol is the required data gate for new research experiments. It removes exact byte-identical images from the source detector splits without changing the raw `CAER-S/` dataset.

The generator retains the first occurrence of an image hash in fixed priority order: `train`, then `val`, then `test`. Later duplicates are recorded in `removed_samples.csv`. It preserves retained labels and raw detector bboxes exactly, including coordinates outside image bounds, to remain compatible with the upstream Pillow crop behavior.

Generate the local protocol artifacts once:

```bash
python prepare_content_disjoint_split.py
```

This writes ignored artifacts under `artifacts/protocols/caer_s_content_disjoint_v1/`: a manifest, detector files, removal audit, hashes, and protocol report. Do not edit these files manually.

Use the generated detector files for all new upstream training and evaluation runs:

```bash
python run_caer_official.py train \
  --detector-dir artifacts/protocols/caer_s_content_disjoint_v1 \
  --device 0,1 --n-gpu 2
```

The original detector files remain valid only for historical upstream-protocol reproduction. Every clean run must record the generated `protocol.json` hashes in `experiments/registry.csv`.
