# Repository Guidelines

## Project Structure & Module Organization

This repository develops CD-ICA-Net from a reproducible CAER-Net baseline on CAER-S. Reusable data, model, metric, checkpoint, and training logic belongs in `caer_research/`; `CAER_S_CAERNet_Reproduction_ipynb.ipynb` remains a legacy executable demo. Research rules live in `agents.d/`, literature and frozen source snapshots in `paper/`, detector annotations in `detectors/`, and protocol definitions in `protocols/`. Local datasets, `artifacts/`, `checkpoints/`, `wandb/`, and generated manifests must remain uncommitted.

## Build, Test, and Development Commands

Use `uv` for environment setup:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m ipykernel install --user --name caer-net-reproduction --display-name "CAER-Net Reproduction"
```

Open the notebook with the `CAER-Net Reproduction` kernel and run cells top to bottom. For a quick dependency check:

```bash
python -c "import torch, torchvision, pandas, wandb"
```

To refresh dependency locking after editing `pyproject.toml`, run `uv lock`.

Generate the mandatory content-disjoint protocol and run tests with:

```bash
python prepare_content_disjoint_split.py
.venv/bin/python -m unittest discover -s tests -v
```

Validate the frozen exploratory run without allocating GPUs:

```bash
python run_caer_official.py train --config configs/experiments/caernet_upstream_content_disjoint_exploratory_seed42.json --seed 42 --dry-run
```

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, four-space indentation, `snake_case` functions, and `PascalCase` classes. Keep model, data, and optimization choices in frozen config files. Notebook cells should each handle one responsibility. Preserve checkpoint outputs such as `best.pt`, `last.pt`, `history.csv`, `metrics.json`, and `test_predictions.csv` under `checkpoints/<run_name>/`.

## Testing Guidelines

Tests use `unittest` under `tests/`; name files `test_*.py` and methods `test_*`. Before committing pipeline changes, run the suite plus a smoke path covering manifest validation, dataloader creation, and one forward pass with logits shape `[batch_size, 7]`. Full training is not a structural test.

## Research Protocol

At the start of a new Codex session, read `agents.d/CURRENT_HANDOFF.md` and verify its live-run status before taking action. Then read `agents.d/LITERATURE_AND_EXPERIMENT_PLAN.md` before experiment work. Use train for optimization, validation for checkpoint selection, and test only for finalized evaluation. New runs must use `caer_s_content_disjoint_v1`, record config/manifest/detector hashes, and keep reported literature results separate from reproduced results. Do not claim causality from masking, shuffling, or feature perturbation alone.

## Commit & Pull Request Guidelines

Use descriptive imperative commits, for example `fix validation split provenance`. Pull requests should state the hypothesis, protocol/config, test command, compute used, and whether test-set metrics were accessed. Keep one research objective per pull request.

## Security & Configuration Tips

Do not commit `WANDB_API_KEY`, dataset files, checkpoints, or W&B run directories. W&B should default to offline mode when no API key is available. Keep large runtime outputs ignored and reproducible from the notebook.
