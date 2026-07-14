#!/usr/bin/env python3
"""Launch the unmodified upstream-community CAER trainer with an explicit seed."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
CAER_CODE_DIR = REPO_ROOT / "third_party" / "CAER" / "CAER"


def configure_seed(seed: int) -> None:
    """Reset every RNG used by the community implementation."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path, resume: Path | None) -> dict[str, Any]:
    """Match ConfigParser's resume semantics while allowing an explicit run id."""
    source_path = resume.parent / "config.json" if resume else config_path
    with source_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    if resume:
        with config_path.open(encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="offline",
    )
    parser.add_argument("--wandb-project", default="caer-net-reproduction")
    parser.add_argument("--wandb-entity", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    resume = args.resume.expanduser().resolve() if args.resume else None

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if not CAER_CODE_DIR.is_dir():
        raise FileNotFoundError(f"Upstream CAER source is missing: {CAER_CODE_DIR}")

    sys.path.insert(0, str(CAER_CODE_DIR))
    from parse_config import ConfigParser  # type: ignore[import-not-found]
    import train as upstream_train  # type: ignore[import-not-found]

    config_data = load_config(config_path, resume)
    configured_seed = config_data.get("seed")
    if configured_seed is not None and int(configured_seed) != args.seed:
        raise ValueError(
            f"CLI seed {args.seed} does not match config seed {configured_seed}."
        )

    # train.py sets seed=123 at import time. Reset it before constructing data/model.
    configure_seed(args.seed)

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        (REPO_ROOT / "wandb").mkdir(exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_id,
            mode=args.wandb_mode,
            dir=str(REPO_ROOT / "wandb"),
            config=config_data,
            sync_tensorboard=True,
            tags=[
                "caer-net",
                "upstream-community",
                config_data.get("experiment", {}).get("variant", "unspecified-protocol"),
            ],
        )

    try:
        config = ConfigParser(config_data, resume=resume, run_id=args.run_id)
        upstream_train.main(config)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
