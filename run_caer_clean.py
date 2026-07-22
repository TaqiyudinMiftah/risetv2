#!/usr/bin/env python3
"""Train the clean in-repository CAER-Net implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from caer_research.checkpointing import load_checkpoint_payload, load_model_checkpoint
from caer_research.data import CAERSTwoStreamDataset, build_transforms
from caer_research.devices import (
    accelerator_snapshot,
    configure_visible_devices,
    parse_device_ids,
)
from caer_research.engine import evaluate
from caer_research.models import CAERNet
from caer_research.trainer import Trainer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "caernet_clean_content_disjoint_exploratory_seed42.json"
REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.csv"
METADATA_ROOT = REPO_ROOT / "artifacts" / "experiments"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return any(line[3:].split(" -> ")[-1] != "experiments/registry.csv" for line in result.stdout.splitlines())


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_run_id(seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"caernet__clean_inrepo__seed{seed}__{timestamp}"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("caer_clean_training")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_config(config: dict[str, Any], expected_seed: int) -> None:
    if int(config["seed"]) != expected_seed:
        raise ValueError(f"CLI seed {expected_seed} does not match config seed {config['seed']}.")
    research = config["research"]
    if research["protocol"] != "caer_s_content_disjoint_v1":
        raise ValueError("Clean experiments must use caer_s_content_disjoint_v1.")
    if research.get("test_during_training"):
        raise ValueError("Test access during training is forbidden.")
    if config["trainer"]["monitor"] != "macro_f1":
        raise ValueError("Clean in-repo checkpoint selection must use validation macro F1.")
    if config["model"]["type"] != "CAERNet":
        raise ValueError(f"Unsupported model type: {config['model']['type']}")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"Path must be inside the repository: {path}") from error


def interruption_details(checkpoint_path: Path, reason: str) -> dict[str, Any]:
    details: dict[str, Any] = {
        "interruption_reason": reason,
        "interrupted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if not checkpoint_path.is_file():
        return details

    checkpoint = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    details.update(
        {
            "last_checkpoint": repository_relative(checkpoint_path),
            "last_checkpoint_sha256": sha256(checkpoint_path),
            "last_completed_epoch": int(checkpoint["epoch"]),
        }
    )
    return details


def validate_resume_request(
    resume_path: Path,
    requested_run_id: str | None,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
) -> tuple[str, Path, dict[str, Any], Path]:
    """Validate that a resume request can only continue its exact frozen run."""
    checkpoint_path = resume_path.expanduser().resolve()
    checkpoint_root = CHECKPOINT_ROOT.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    if checkpoint_path.name != "last.pt":
        raise ValueError("Clean training may resume only from an end-of-epoch last.pt checkpoint.")
    if checkpoint_path.parent.parent != checkpoint_root:
        raise ValueError("Resume checkpoint must be directly under checkpoints/<run_id>/last.pt.")

    run_id = checkpoint_path.parent.name
    if requested_run_id is not None and requested_run_id != run_id:
        raise ValueError(
            f"--run-id {requested_run_id!r} conflicts with resume checkpoint run ID {run_id!r}."
        )
    output_dir = checkpoint_path.parent
    metadata_path = METADATA_ROOT / run_id / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Resume metadata not found: {metadata_path}")
    metadata = load_json(metadata_path)

    if metadata.get("status") != "interrupted":
        raise ValueError(
            "Resume requires metadata status 'interrupted'; verify the prior process is stopped first."
        )
    expected_metadata = {
        "run_id": run_id,
        "seed": int(config["seed"]),
        "protocol": config["research"]["protocol"],
        "config": repository_relative(config_path),
        "config_sha256": sha256(config_path),
        "manifest": repository_relative(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "test_used_for_selection": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Resume metadata mismatch for {key}: expected {expected!r}, "
                f"got {metadata.get(key)!r}."
            )

    frozen_runtime_config = output_dir / "config.json"
    if not frozen_runtime_config.is_file():
        raise FileNotFoundError(f"Frozen runtime config not found: {frozen_runtime_config}")
    if load_json(frozen_runtime_config) != config:
        raise ValueError("Effective resume config does not match the frozen runtime config.")

    checkpoint = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_metric",
        "early_stopping_count",
        "history",
        "config",
        "rng_state",
        "train_generator_state",
    }
    missing = sorted(key for key in required_keys if checkpoint.get(key) is None)
    if missing:
        raise ValueError(f"Resume checkpoint is missing required state: {missing}")
    if checkpoint["config"] != config:
        raise ValueError("Resume checkpoint config does not match the effective frozen config.")

    rng_state = checkpoint["rng_state"]
    if not isinstance(rng_state, dict) or not {"python", "numpy", "torch_cpu"} <= rng_state.keys():
        raise ValueError("Resume checkpoint does not contain the required RNG state.")
    epoch = int(checkpoint["epoch"])
    history = checkpoint["history"]
    if epoch < 1 or not isinstance(history, list):
        raise ValueError("Resume checkpoint must represent at least one completed epoch.")
    try:
        history_epochs = [int(row["epoch"]) for row in history]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Resume checkpoint history is malformed.") from error
    if history_epochs != list(range(1, epoch + 1)):
        raise ValueError("Resume checkpoint history is not contiguous through its saved epoch.")
    return run_id, output_dir, metadata, checkpoint_path


def mark_interrupted(args: argparse.Namespace) -> None:
    run_id = args.run_id
    if Path(run_id).name != run_id:
        raise ValueError("Run ID must not contain a path separator.")
    output_dir = CHECKPOINT_ROOT / run_id
    metadata_path = METADATA_ROOT / run_id / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    metadata = load_json(metadata_path)
    if metadata.get("run_id") != run_id:
        raise ValueError("Run metadata does not match the requested run ID.")
    if metadata.get("status") != "running":
        raise ValueError("Only a verified stale 'running' run may be marked interrupted.")
    checkpoint_path = output_dir / "last.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Cannot mark a run resumable without checkpoints/<run_id>/last.pt.")

    metadata.update({"status": "interrupted", **interruption_details(checkpoint_path, args.reason)})
    write_metadata(metadata)
    update_registry(metadata)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": metadata["status"],
                "last_completed_epoch": metadata["last_completed_epoch"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


def update_registry(metadata: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
    with REGISTRY_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    row = {field: "" for field in fieldnames}
    row.update(
        {
            "run_id": metadata["run_id"],
            "status": metadata["status"],
            "model": "CAER-Net",
            "variant": "clean_inrepo_caer_s_content_disjoint_v1",
            "seed": str(metadata["seed"]),
            "git_sha": metadata["git_sha"],
            "config": metadata["config"],
            "checkpoint": metadata.get("checkpoint", ""),
            "notes": metadata["notes"],
        }
    )
    if metrics is not None:
        row.update(
            {
                "val_accuracy": str(metrics["accuracy"]),
                "val_macro_f1": str(metrics["macro_f1"]),
                "neutral_f1": str(metrics["per_class"]["Neutral"]["f1"]),
                "params": str(metadata["params"]),
            }
        )
    rows = [existing for existing in rows if existing.get("run_id") != metadata["run_id"]]
    rows.append(row)
    with REGISTRY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(metadata: dict[str, Any]) -> Path:
    output_dir = METADATA_ROOT / metadata["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "run_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def public_evaluation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"labels", "predictions", "confidences", "image_paths"}
    }


def write_predictions(result: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "prediction", "confidence", "correct"],
            lineterminator="\n",
        )
        writer.writeheader()
        for image_path, label, prediction, confidence in zip(
            result["image_paths"],
            result["labels"],
            result["predictions"],
            result["confidences"],
        ):
            writer.writerow(
                {
                    "image_path": image_path,
                    "label": label,
                    "prediction": prediction,
                    "confidence": confidence,
                    "correct": int(label == prediction),
                }
            )


def train(args: argparse.Namespace) -> None:
    configure_visible_devices(args.device)
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    validate_config(config, args.seed)
    if args.n_gpu is not None:
        config["n_gpu"] = args.n_gpu
    manifest_path = resolve_path(config["data"]["manifest"])
    dataset_root = resolve_path(config["data"]["dataset_root"])
    if not manifest_path.is_file() or not dataset_root.is_dir():
        raise FileNotFoundError("Frozen manifest and CAER-S dataset root are required.")
    if args.resume is not None and args.smoke_only:
        raise ValueError("--resume cannot be combined with --smoke-only.")

    resume_path: Path | None = None
    if args.resume is not None:
        run_id, output_dir, metadata, resume_path = validate_resume_request(
            args.resume,
            args.run_id,
            config_path,
            config,
            manifest_path,
        )
        metadata_path = METADATA_ROOT / run_id / "run_metadata.json"
    else:
        run_id = args.run_id or make_run_id(args.seed)
        output_dir = CHECKPOINT_ROOT / run_id
        metadata = {
            "run_id": run_id,
            "status": "prepared",
            "seed": args.seed,
            "stage": config["research"]["stage"],
            "track": "clean_inrepo",
            "protocol": config["research"]["protocol"],
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "config": repository_relative(config_path),
            "config_sha256": sha256(config_path),
            "manifest": repository_relative(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "test_used_for_selection": False,
            "notes": "Exploratory clean in-repo run; test split is not loaded or evaluated.",
            "started_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        metadata_path = write_metadata(metadata)
    print(f"Run ID: {run_id}")
    print(f"Metadata: {metadata_path}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        status = "Resume preflight complete" if resume_path is not None else "Dry run complete"
        print(f"{status}; training was not started.")
        return

    n_gpu = int(config["n_gpu"])
    selected_count = len(parse_device_ids(args.device))
    if selected_count < n_gpu:
        raise RuntimeError(
            f"n_gpu={n_gpu}, but --device selects only {selected_count} accelerator(s)."
        )
    accelerator = accelerator_snapshot(args.device, n_gpu)
    gpu_memory = {
        int(device["requested_index"]): int(device["memory_free_mib"])
        for device in accelerator["devices"]
    }
    low_memory = {index: free for index, free in gpu_memory.items() if free < args.min_free_gpu_mib}
    if low_memory:
        raise RuntimeError(f"Accelerator preflight failed: {low_memory}")

    configure_determinism(args.seed)
    train_face_transform, train_context_transform = build_transforms(train=True)
    val_face_transform, val_context_transform = build_transforms(train=False)
    train_dataset = CAERSTwoStreamDataset(
        manifest_path,
        dataset_root,
        split="train",
        face_transform=train_face_transform,
        context_transform=train_context_transform,
    )
    val_dataset = CAERSTwoStreamDataset(
        manifest_path,
        dataset_root,
        split="val",
        face_transform=val_face_transform,
        context_transform=val_context_transform,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = {
        "batch_size": int(config["data"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    device = torch.device("cuda:0")
    base_model = CAERNet(**config["model"]["args"]).to(device)
    params = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    model: nn.Module = (
        nn.DataParallel(base_model, device_ids=list(range(n_gpu))) if n_gpu > 1 else base_model
    )
    optimizer = torch.optim.SGD(model.parameters(), **config["optimizer"]["args"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config["scheduler"]["args"])
    criterion = nn.CrossEntropyLoss()

    if args.smoke_only:
        model.eval()
        batch = next(iter(val_loader))
        with torch.inference_mode():
            logits = model(batch["face"].to(device), batch["context"].to(device))
        print(
            json.dumps(
                {
                    "status": "smoke_passed",
                    "n_gpu": n_gpu,
                    "face_shape": list(batch["face"].shape),
                    "context_shape": list(batch["context"].shape),
                    "logits_shape": list(logits.shape),
                    "test_accessed": False,
                },
                indent=2,
            )
        )
        return

    if resume_path is None:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logger = setup_logger(output_dir / "train.log")
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_id,
            mode=args.wandb_mode,
            dir=str(REPO_ROOT / "wandb"),
            config=config,
            tags=["caer-net", "clean-inrepo", "exploratory", config["research"]["protocol"]],
        )

    def log_epoch(row: dict[str, Any]) -> None:
        if wandb_run is not None:
            wandb_run.log(row)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        config=config,
        epochs=int(config["trainer"]["epochs"]),
        monitor=config["trainer"]["monitor"],
        patience=int(config["trainer"]["early_stopping_patience"]),
        use_amp=bool(config["trainer"]["use_amp"]),
        grad_clip_max_norm=config["trainer"]["grad_clip_max_norm"],
        epoch_callback=log_epoch,
        logger=logger,
        train_generator=generator,
    )
    try:
        if resume_path is not None:
            trainer.resume(resume_path)
            resume_event = {
                "at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "checkpoint": repository_relative(resume_path),
                "checkpoint_sha256": sha256(resume_path),
                "checkpoint_epoch": trainer.start_epoch - 1,
                "git_sha": git_sha(),
                "git_dirty": git_dirty(),
            }
            resumes = list(metadata.get("resumes", []))
            resumes.append(resume_event)
            metadata["resumes"] = resumes
            metadata["resume_count"] = len(resumes)
        metadata.update(
            {
                "status": "running",
                "accelerator": accelerator,
                "gpu_free_mib_at_start": gpu_memory,
                "params": params,
            }
        )
        write_metadata(metadata)
        update_registry(metadata)
        if resume_path is not None:
            logger.info(
                "resumed_from=%s next_epoch=%s",
                repository_relative(resume_path),
                trainer.start_epoch,
            )
        history = trainer.fit()
        load_model_checkpoint(model, trainer.best_path, map_location=device)
        val_result = evaluate(model, val_loader, criterion, device, use_amp=False)
        metrics = public_evaluation_metrics(val_result)
        (output_dir / "val_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        write_predictions(val_result, output_dir / "val_predictions.csv")
        metadata.update(
            {
                "status": "completed",
                "checkpoint": str(trainer.best_path.relative_to(REPO_ROOT)),
                "checkpoint_sha256": sha256(trainer.best_path),
                "best_epoch": int(max(history, key=lambda row: row["val_macro_f1"])["epoch"]),
                "val_metrics": str((output_dir / "val_metrics.json").relative_to(REPO_ROOT)),
                "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        write_metadata(metadata)
        update_registry(metadata, metrics)
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "best_val_accuracy": metrics["accuracy"],
                    "best_val_macro_f1": metrics["macro_f1"],
                    "best_val_neutral_f1": metrics["per_class"]["Neutral"]["f1"],
                }
            )
    except KeyboardInterrupt:
        metadata.update(
            {
                "status": "interrupted",
                **interruption_details(trainer.last_path, "KeyboardInterrupt"),
            }
        )
        write_metadata(metadata)
        update_registry(metadata)
        logger.warning("training interrupted; resumable checkpoint state has been recorded")
        raise
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "error": repr(error),
                "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        write_metadata(metadata)
        update_registry(metadata)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--run-id", default=None)
    train_parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue only the matching interrupted run from checkpoints/<run_id>/last.pt.",
    )
    train_parser.add_argument("--device", default="0,1")
    train_parser.add_argument("--n-gpu", type=int, default=None)
    train_parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="offline")
    train_parser.add_argument("--wandb-project", default="caer-net-reproduction")
    train_parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    train_parser.add_argument("--min-free-gpu-mib", type=int, default=6000)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--smoke-only", action="store_true")
    train_parser.set_defaults(func=train)

    interrupted_parser = subparsers.add_parser(
        "mark-interrupted",
        help="Record a verified stopped run as interrupted before a guarded resume.",
    )
    interrupted_parser.add_argument("--run-id", required=True)
    interrupted_parser.add_argument("--reason", required=True)
    interrupted_parser.set_defaults(func=mark_interrupted)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
