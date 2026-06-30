# Repository Guidelines

## Project Structure & Module Organization

This repository is centered on a CAER-Net reproduction workflow for CAER-S. The main pipeline is `CAER_S_CAERNet_Reproduction_ipynb.ipynb`; keep experiment logic there unless a reusable Python module is intentionally introduced. Reference material lives in `paper/`. Detector cache files live in `detectors/`. Runtime outputs such as `caers_manifest.jsonl`, `checkpoints/`, `wandb/`, and `*.egg-info/` are generated artifacts and should not be committed. The local dataset is expected at `CAER-S/` with `train/` and `test/` subfolders.

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

## Coding Style & Naming Conventions

Use Python 3.12-compatible code. Prefer clear notebook cells with one responsibility: setup, data validation, model definition, training, and evaluation. Use `snake_case` for variables/functions and `PascalCase` for model classes. Keep configuration in the `CFG` dictionary rather than scattering constants across cells. Preserve checkpoint naming conventions: `best.pt`, `last.pt`, `history.csv`, `metrics.json`, and `test_predictions.csv` under `checkpoints/<run_name>/`.

## Testing Guidelines

There is no formal test suite. Before committing notebook changes, run a smoke path: imports, dataset/manifest validation, dataloader creation, and one forward pass. Confirm logits shape is `[BATCH_SIZE, 7]`. Do not run full 60-epoch training just to validate structural changes unless required.

## Commit & Pull Request Guidelines

Current Git history uses short messages such as `update`; prefer more descriptive imperative messages, for example `add wandb logging to notebook`. Pull requests should summarize notebook changes, mention whether training was run, include key metrics if available, and note any generated artifacts intentionally left uncommitted.

## Security & Configuration Tips

Do not commit `WANDB_API_KEY`, dataset files, checkpoints, or W&B run directories. W&B should default to offline mode when no API key is available. Keep large runtime outputs ignored and reproducible from the notebook.
