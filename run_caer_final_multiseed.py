#!/usr/bin/env python3
"""Run the frozen upstream-community CAER-Net baseline for final seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
LAUNCHER = REPO_ROOT / "run_caer_official.py"
FINAL_CONFIGS = {
    seed: REPO_ROOT
    / "configs"
    / "experiments"
    / f"caernet_upstream_content_disjoint_final_seed{seed}.json"
    for seed in (42, 43, 44)
}


def build_command(args: argparse.Namespace, seed: int) -> list[str]:
    if seed not in FINAL_CONFIGS:
        raise ValueError(f"Unsupported final seed {seed}; choose from {sorted(FINAL_CONFIGS)}")

    command = [
        sys.executable,
        str(LAUNCHER),
        "train",
        "--config",
        str(FINAL_CONFIGS[seed]),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--n-gpu",
        str(args.n_gpu),
        "--wandb-mode",
        args.wandb_mode,
    ]
    if args.wandb_project:
        command.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Frozen final seeds to run sequentially.",
    )
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--n-gpu", type=int, default=2)
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="offline",
    )
    parser.add_argument("--wandb-project", default="caer-net-reproduction")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    invalid_seeds = sorted(set(args.seeds) - FINAL_CONFIGS.keys())
    if invalid_seeds:
        raise SystemExit(
            f"Unsupported final seed(s): {invalid_seeds}; choose from {sorted(FINAL_CONFIGS)}"
        )

    for seed in args.seeds:
        command = build_command(args, seed)
        print(f"Final CAER-Net seed {seed}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
