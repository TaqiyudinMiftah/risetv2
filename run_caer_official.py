#!/usr/bin/env python3
"""Prepare and run the upstream ndkhanh360/CAER training pipeline locally."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CAER_CODE_DIR = REPO_ROOT / "third_party" / "CAER" / "CAER"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "caer_official.json"
DATASET_ROOT = REPO_ROOT / "CAER-S"
DETECTOR_ROOT = REPO_ROOT / "detectors"
OFFICIAL_DATA_DIR = CAER_CODE_DIR / "data"
GENERATED_CONFIG_DIR = CAER_CODE_DIR / "official_runs" / "generated_configs"

CLASS_DIRS = {
    "Anger": "Angry",
    "Angry": "Angry",
    "Disgust": "Disgust",
    "Fear": "Fear",
    "Happy": "Happy",
    "Neutral": "Neutral",
    "Sad": "Sad",
    "Surprise": "Surprise",
}


def _relative_symlink(source: Path, target: Path, force: bool = False) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source:
            return
        if not force:
            raise FileExistsError(f"{target} already exists and does not point to {source}")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    rel_source = os.path.relpath(source, target.parent)
    target.symlink_to(rel_source, target_is_directory=source.is_dir())


def prepare_data(force: bool = False, detector_dir: Path = DETECTOR_ROOT) -> None:
    if not CAER_CODE_DIR.exists():
        raise FileNotFoundError(
            "third_party/CAER/CAER tidak ditemukan. Jalankan: "
            "git submodule update --init --recursive"
        )

    for split in ("train", "test"):
        split_root = DATASET_ROOT / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split tidak ditemukan: {split_root}")

        for official_name, local_name in CLASS_DIRS.items():
            source = split_root / local_name
            if not source.is_dir():
                raise FileNotFoundError(f"Folder kelas tidak ditemukan: {source}")
            target = OFFICIAL_DATA_DIR / "CAER-S" / split / official_name
            _relative_symlink(source, target, force=force)

    for name in ("train.txt", "val.txt", "test.txt"):
        source = detector_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Detector file tidak ditemukan: {source}")
        _relative_symlink(source, OFFICIAL_DATA_DIR / name, force=force)

    print(f"Prepared official CAER data layout at {OFFICIAL_DATA_DIR}")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_run_config(args: argparse.Namespace) -> Path:
    config = _load_config(args.config)

    if args.n_gpu is not None:
        config["n_gpu"] = args.n_gpu
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    if args.batch_size is not None:
        for key in ("train_loader", "val_loader", "test_loader"):
            config[key]["args"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        for key in ("train_loader", "val_loader", "test_loader"):
            config[key]["args"]["num_workers"] = args.num_workers
    if args.learning_rate is not None:
        config["optimizer"]["args"]["lr"] = args.learning_rate
    if args.no_tensorboard:
        config["trainer"]["tensorboard"] = False

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = GENERATED_CONFIG_DIR / f"caer_official_{stamp}.json"
    with run_config.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
    return run_config


def _run_official(command: list[str], device: str | None) -> None:
    env = os.environ.copy()
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    print("Running:", " ".join(command))
    subprocess.run(command, cwd=CAER_CODE_DIR, env=env, check=True)


def _resolve_cli_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def train(args: argparse.Namespace) -> None:
    prepare_data(detector_dir=args.detector_dir, force=args.force_prepare)
    run_config = _write_run_config(args)
    command = [sys.executable, "train.py", "--config", str(run_config)]
    if args.resume:
        command.extend(["--resume", _resolve_cli_path(args.resume)])
    if args.device:
        command.extend(["--device", args.device])
    _run_official(command, args.device)


def test(args: argparse.Namespace) -> None:
    prepare_data(detector_dir=args.detector_dir, force=args.force_prepare)
    if not args.resume:
        raise SystemExit("--resume wajib diisi untuk evaluasi official pipeline.")
    run_config = _write_run_config(args)
    command = [
        sys.executable,
        "test.py",
        "--config",
        str(run_config),
        "--resume",
        _resolve_cli_path(args.resume),
    ]
    if args.device:
        command.extend(["--device", args.device])
    _run_official(command, args.device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create upstream-compatible data symlinks.")
    prepare.add_argument("--force", action="store_true", help="Replace stale generated symlinks.")
    prepare.add_argument("--detector-dir", type=Path, default=DETECTOR_ROOT)
    prepare.set_defaults(func=lambda args: prepare_data(detector_dir=args.detector_dir, force=args.force))

    def add_run_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        subparser.add_argument(
            "--detector-dir",
            type=Path,
            default=DETECTOR_ROOT,
            help="Directory containing train.txt, val.txt, and test.txt.",
        )
        subparser.add_argument("--device", default="0,1", help="CUDA device list for upstream script.")
        subparser.add_argument("--n-gpu", type=int, default=None, help="Override config n_gpu.")
        subparser.add_argument("--batch-size", type=int, default=None)
        subparser.add_argument("--num-workers", type=int, default=None)
        subparser.add_argument("--learning-rate", type=float, default=None)
        subparser.add_argument("--no-tensorboard", action="store_true")
        subparser.add_argument("--force-prepare", action="store_true")

    train_parser = subparsers.add_parser("train", help="Run upstream CAER-Net training.")
    add_run_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser("test", help="Run upstream CAER-Net test evaluation.")
    add_run_args(test_parser)
    test_parser.add_argument("--epochs", type=int, default=None)
    test_parser.add_argument("--resume", type=Path, required=True)
    test_parser.set_defaults(func=test)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
