#!/usr/bin/env python3
"""Prepare and run the upstream ndkhanh360/CAER training pipeline locally."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
CAER_CODE_DIR = REPO_ROOT / "third_party" / "CAER" / "CAER"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "caer_official.json"
DATASET_ROOT = REPO_ROOT / "CAER-S"
DETECTOR_ROOT = REPO_ROOT / "detectors"
RESEARCH_DETECTOR_ROOT = REPO_ROOT / "artifacts" / "protocols" / "caer_s_content_disjoint_v1"
OFFICIAL_DATA_DIR = CAER_CODE_DIR / "data"
GENERATED_CONFIG_DIR = CAER_CODE_DIR / "official_runs" / "generated_configs"
UPSTREAM_TRAIN_LAUNCHER = REPO_ROOT / "run_caer_upstream_train.py"
EXPERIMENT_REGISTRY = REPO_ROOT / "experiments" / "registry.csv"
RUN_METADATA_ROOT = REPO_ROOT / "artifacts" / "experiments"

REGISTRY_COLUMNS = [
    "run_id",
    "status",
    "model",
    "variant",
    "seed",
    "git_sha",
    "config",
    "checkpoint",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "neutral_f1",
    "params",
    "latency_ms",
    "notes",
]

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ignored_runtime_paths = {"experiments/registry.csv"}
    for line in result.stdout.splitlines():
        changed_path = line[3:].split(" -> ")[-1]
        if changed_path not in ignored_runtime_paths:
            return True
    return False


def _make_run_id(seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"caernet__upstream_community__seed{seed}__{stamp}"


def _write_run_config(args: argparse.Namespace, run_id: str) -> Path:
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
    if args.early_stop is not None:
        config["trainer"]["early_stop"] = args.early_stop
    if args.save_period is not None:
        config["trainer"]["save_period"] = args.save_period
    if args.no_tensorboard:
        config["trainer"]["tensorboard"] = False

    detector_hashes = {
        name: _sha256(args.detector_dir / name)
        for name in ("train.txt", "val.txt", "test.txt")
    }
    manifest_path = args.detector_dir / "manifest.jsonl"
    config["seed"] = args.seed
    config["experiment"] = {
        "run_id": run_id,
        "track": "upstream_community",
        "variant": args.detector_dir.resolve().name,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "detector_dir": str(args.detector_dir.resolve()),
        "detector_hashes": detector_hashes,
        "manifest_hash": _sha256(manifest_path) if manifest_path.is_file() else None,
        "test_used_for_selection": False,
    }

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    run_config = GENERATED_CONFIG_DIR / f"{run_id}.json"
    with run_config.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
    return run_config


def _gpu_free_memory() -> dict[int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("GPU preflight requires a working nvidia-smi command.") from error
    free_memory: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index, free_mib = (part.strip() for part in line.split(",", maxsplit=1))
        free_memory[int(index)] = int(free_mib)
    return free_memory


def _gpu_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    inventory = []
    for line in result.stdout.splitlines():
        index, name, total_mib = (part.strip() for part in line.split(",", maxsplit=2))
        inventory.append({"index": int(index), "name": name, "memory_total_mib": int(total_mib)})
    return inventory


def _software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "wandb"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _check_gpu_capacity(args: argparse.Namespace) -> dict[int, int]:
    if args.skip_gpu_check or not args.device or args.n_gpu == 0:
        return {}
    selected = [int(item.strip()) for item in args.device.split(",") if item.strip()]
    requested_count = args.n_gpu if args.n_gpu is not None else len(selected)
    if len(selected) < requested_count:
        raise RuntimeError(
            f"Requested n_gpu={requested_count}, but --device only selects {len(selected)} GPU(s)."
        )
    selected = selected[:requested_count]
    free_memory = _gpu_free_memory()
    insufficient = {
        index: free_memory.get(index, 0)
        for index in selected
        if free_memory.get(index, 0) < args.min_free_gpu_mib
    }
    if insufficient:
        details = ", ".join(f"GPU {index}: {free_mib} MiB free" for index, free_mib in insufficient.items())
        raise RuntimeError(
            f"GPU preflight failed ({details}); require at least "
            f"{args.min_free_gpu_mib} MiB per selected GPU."
        )
    return {index: free_memory[index] for index in selected}


def _write_metadata(metadata: dict[str, Any]) -> Path:
    output_dir = RUN_METADATA_ROOT / metadata["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "run_metadata.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return output_path


def _update_registry(metadata: dict[str, Any], **updates: Any) -> None:
    EXPERIMENT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if EXPERIMENT_REGISTRY.is_file():
        with EXPERIMENT_REGISTRY.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    row = {column: "" for column in REGISTRY_COLUMNS}
    row.update(
        {
            "run_id": metadata["run_id"],
            "status": metadata["status"],
            "model": "CAER-Net",
            "variant": f"upstream_community_{metadata['variant']}",
            "seed": str(metadata["seed"]),
            "git_sha": metadata["git_sha"],
            "config": metadata["config"],
            "notes": metadata.get("notes", ""),
        }
    )
    row.update({key: str(value) for key, value in updates.items() if value is not None})
    rows = [existing for existing in rows if existing.get("run_id") != metadata["run_id"]]
    rows.append(row)
    with EXPERIMENT_REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _initial_metadata(
    args: argparse.Namespace,
    run_id: str,
    run_config: Path,
    command: list[str],
) -> dict[str, Any]:
    config = _load_config(run_config)
    try:
        frozen_config = str(args.config.expanduser().resolve().relative_to(REPO_ROOT))
    except ValueError:
        frozen_config = str(args.config.expanduser().resolve())
    return {
        "run_id": run_id,
        "status": "prepared",
        "model": "CAER-Net",
        "track": "upstream_community",
        "variant": config["experiment"]["variant"],
        "exploratory": config.get("research", {}).get("stage", "exploratory") == "exploratory",
        "seed": args.seed,
        "git_sha": config["experiment"]["git_sha"],
        "git_dirty": config["experiment"]["git_dirty"],
        "config": frozen_config,
        "config_sha256": _sha256(args.config.expanduser().resolve()),
        "generated_config": str(run_config.relative_to(REPO_ROOT)),
        "generated_config_sha256": _sha256(run_config),
        "detector_dir": str(args.detector_dir.resolve()),
        "detector_hashes": config["experiment"]["detector_hashes"],
        "manifest_hash": config["experiment"]["manifest_hash"],
        "selection_metric": config["trainer"]["monitor"],
        "test_used_for_selection": False,
        "command": command,
        "hardware": {"platform": platform.platform(), "gpus": _gpu_inventory()},
        "software": _software_versions(),
        "started_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": "Exploratory upstream-community run; test split is not evaluated during training.",
    }


def _load_checkpoint_summary(checkpoint: Path) -> dict[str, Any]:
    """Load metadata from an upstream checkpoint without depending on caller cwd."""
    upstream_path = str(CAER_CODE_DIR)
    added_to_path = upstream_path not in sys.path
    if added_to_path:
        sys.path.insert(0, upstream_path)
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        if added_to_path:
            sys.path.remove(upstream_path)

    monitor_best = payload.get("monitor_best")
    return {
        "best_epoch": int(payload["epoch"]),
        "monitor_best": float(monitor_best) if monitor_best is not None else None,
    }


def _best_checkpoint_summary(config: dict[str, Any], run_id: str) -> tuple[Path, dict[str, Any] | None]:
    checkpoint = (
        CAER_CODE_DIR
        / config["trainer"]["save_dir"]
        / "models"
        / config["name"]
        / run_id
        / "model_best.pth"
    )
    if not checkpoint.is_file():
        return checkpoint, None
    return checkpoint, _load_checkpoint_summary(checkpoint)


def _run_finished_at(config: dict[str, Any], run_id: str) -> str:
    info_log = (
        CAER_CODE_DIR
        / config["trainer"]["save_dir"]
        / "log"
        / config["name"]
        / run_id
        / "info.log"
    )
    if info_log.is_file():
        return datetime.fromtimestamp(info_log.stat().st_mtime, UTC).isoformat(timespec="seconds")
    return datetime.now(UTC).isoformat(timespec="seconds")


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
    run_id = args.run_id or _make_run_id(args.seed)
    run_config = _write_run_config(args, run_id)
    if args.n_gpu is None:
        args.n_gpu = int(_load_config(run_config)["n_gpu"])
    command = [
        sys.executable,
        str(UPSTREAM_TRAIN_LAUNCHER),
        "--config",
        str(run_config),
        "--seed",
        str(args.seed),
        "--run-id",
        run_id,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.resume:
        command.extend(["--resume", _resolve_cli_path(args.resume)])
    if args.device:
        command.extend(["--device", args.device])

    metadata = _initial_metadata(args, run_id, run_config, command)
    metadata_path = _write_metadata(metadata)
    print(f"Run metadata: {metadata_path}")
    print("Running:", " ".join(command))
    if args.dry_run:
        print("Dry run complete; training was not started.")
        return

    try:
        metadata["gpu_free_mib_at_start"] = _check_gpu_capacity(args)
    except RuntimeError as error:
        metadata["status"] = "blocked_compute"
        metadata["error"] = str(error)
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        _write_metadata(metadata)
        _update_registry(metadata)
        raise

    metadata["status"] = "running"
    _write_metadata(metadata)
    _update_registry(metadata)
    try:
        _run_official(command, args.device)
    except subprocess.CalledProcessError as error:
        metadata["status"] = "failed"
        metadata["return_code"] = error.returncode
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        _write_metadata(metadata)
        _update_registry(metadata)
        raise

    config = _load_config(run_config)
    checkpoint, checkpoint_summary = _best_checkpoint_summary(config, run_id)
    if not checkpoint.is_file():
        metadata["status"] = "failed_artifact"
        metadata["error"] = f"Training finished without best checkpoint: {checkpoint}"
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        _write_metadata(metadata)
        _update_registry(metadata)
        raise RuntimeError(metadata["error"])
    metadata["status"] = "completed"
    metadata["checkpoint"] = str(checkpoint.relative_to(REPO_ROOT))
    metadata["checkpoint_sha256"] = _sha256(checkpoint)
    metadata["best_epoch"] = checkpoint_summary["best_epoch"]
    metadata["val_accuracy"] = checkpoint_summary["monitor_best"]
    metadata["finished_at_utc"] = _run_finished_at(config, run_id)
    _write_metadata(metadata)
    _update_registry(
        metadata,
        checkpoint=metadata["checkpoint"],
        val_accuracy=metadata["val_accuracy"],
    )


def reconcile(args: argparse.Namespace) -> None:
    metadata_path = RUN_METADATA_ROOT / args.run_id / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    metadata = _load_config(metadata_path)
    run_config = REPO_ROOT / metadata["generated_config"]
    if not run_config.is_file():
        raise FileNotFoundError(f"Generated run config not found: {run_config}")
    config = _load_config(run_config)
    checkpoint, checkpoint_summary = _best_checkpoint_summary(config, args.run_id)

    metadata["finished_at_utc"] = _run_finished_at(config, args.run_id)
    metadata["reconciled_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    if checkpoint_summary is None:
        metadata["status"] = "failed_incomplete"
        metadata["error"] = f"Run ended without a best checkpoint: {checkpoint}"
        metadata["notes"] = "Incomplete upstream-community run; no checkpoint was produced."
        _write_metadata(metadata)
        _update_registry(metadata)
        print(json.dumps({"run_id": args.run_id, "status": metadata["status"]}, indent=2))
        return

    metadata.pop("error", None)
    metadata["status"] = "completed"
    metadata["checkpoint"] = str(checkpoint.relative_to(REPO_ROOT))
    metadata["checkpoint_sha256"] = _sha256(checkpoint)
    metadata["best_epoch"] = checkpoint_summary["best_epoch"]
    metadata["val_accuracy"] = checkpoint_summary["monitor_best"]
    metadata["notes"] = (
        "Exploratory upstream-community run completed; test split was not evaluated. "
        "Status reconciled after the post-training summary failure."
    )

    registry_updates: dict[str, Any] = {
        "checkpoint": metadata["checkpoint"],
        "val_accuracy": metadata["val_accuracy"],
    }
    if args.validation_metrics is not None:
        metrics_path = args.validation_metrics.expanduser().resolve()
        metrics = _load_config(metrics_path)
        if metrics.get("split") != "val":
            raise ValueError(f"Expected validation metrics, got split={metrics.get('split')!r}")
        if metrics.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
            raise ValueError("Validation metrics checkpoint hash does not match run checkpoint.")
        metadata["validation_metrics"] = str(metrics_path.relative_to(REPO_ROOT))
        metadata["validation_metrics_sha256"] = _sha256(metrics_path)
        metadata["val_accuracy"] = float(metrics["accuracy"])
        metadata["val_macro_f1"] = float(metrics["macro_f1"])
        metadata["val_weighted_f1"] = float(metrics["weighted_f1"])
        metadata["neutral_f1"] = float(metrics["per_class"]["Neutral"]["f1"])
        metadata["params"] = int(metrics["params"])
        registry_updates.update(
            {
                "val_accuracy": metadata["val_accuracy"],
                "val_macro_f1": metadata["val_macro_f1"],
                "neutral_f1": metadata["neutral_f1"],
                "params": metadata["params"],
            }
        )

    _write_metadata(metadata)
    _update_registry(metadata, **registry_updates)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": metadata["status"],
                "best_epoch": metadata["best_epoch"],
                "val_accuracy": metadata["val_accuracy"],
                "val_macro_f1": metadata.get("val_macro_f1"),
            },
            indent=2,
        )
    )


def test(args: argparse.Namespace) -> None:
    prepare_data(detector_dir=args.detector_dir, force=args.force_prepare)
    if not args.resume:
        raise SystemExit("--resume wajib diisi untuk evaluasi official pipeline.")
    run_id = args.run_id or _make_run_id(args.seed)
    run_config = _write_run_config(args, run_id)
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
            default=RESEARCH_DETECTOR_ROOT,
            help="Directory containing train.txt, val.txt, and test.txt.",
        )
        subparser.add_argument("--device", default="0,1", help="CUDA device list for upstream script.")
        subparser.add_argument("--n-gpu", type=int, default=None, help="Override config n_gpu.")
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--run-id", default=None)
        subparser.add_argument("--batch-size", type=int, default=None)
        subparser.add_argument("--num-workers", type=int, default=None)
        subparser.add_argument("--learning-rate", type=float, default=None)
        subparser.add_argument("--early-stop", type=int, default=None)
        subparser.add_argument("--save-period", type=int, default=None)
        subparser.add_argument("--no-tensorboard", action="store_true")
        subparser.add_argument("--force-prepare", action="store_true")
        subparser.add_argument(
            "--wandb-mode",
            choices=("disabled", "offline", "online"),
            default=os.environ.get("WANDB_MODE", "offline"),
        )
        subparser.add_argument("--wandb-project", default="caer-net-reproduction")
        subparser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
        subparser.add_argument("--min-free-gpu-mib", type=int, default=6000)
        subparser.add_argument("--skip-gpu-check", action="store_true")
        subparser.add_argument("--dry-run", action="store_true")

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

    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help="Finalize registry metadata for a completed or interrupted run.",
    )
    reconcile_parser.add_argument("--run-id", required=True)
    reconcile_parser.add_argument("--validation-metrics", type=Path, default=None)
    reconcile_parser.set_defaults(func=reconcile)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
