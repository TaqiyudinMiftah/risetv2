from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from caers_pipeline.config import load_config
from caers_pipeline.dataset import CAERSTwoStreamDataset
from caers_pipeline.engine import evaluate, train_one_epoch
from caers_pipeline.io_utils import ensure_parent_dir, write_json
from caers_pipeline.model import CAERNet, SingleStreamNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, cfg: object) -> nn.Module:
    if cfg.train.stream_mode == "multimodal":
        return CAERNet(
            num_classes=num_classes,
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            dropout=cfg.model.dropout,
        )
    return SingleStreamNet(
        num_classes=num_classes,
        stream=cfg.train.stream_mode,
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAER-Net on CAER-S")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=os.environ.get("WANDB_API_KEY", ""),
        help="W&B API key (falls back to WANDB_API_KEY env var)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="caers-emotion-recognition",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="W&B entity/team name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
        help="W&B run name (auto-generated if empty)",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run W&B in offline mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # Login to W&B if API key provided
    if args.wandb_api_key and not args.wandb_offline:
        wandb.login(key=args.wandb_api_key)

    # Initialize W&B
    wandb_mode = "offline" if args.wandb_offline else "online"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        mode=wandb_mode,
        config={
            "seed": cfg.seed,
            "backbone": cfg.model.backbone,
            "pretrained": cfg.model.pretrained,
            "dropout": cfg.model.dropout,
            "batch_size": cfg.train.batch_size,
            "num_epochs": cfg.train.num_epochs,
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "stream_mode": cfg.train.stream_mode,
            "image_size": cfg.dataset.image_size,
            "val_ratio": cfg.dataset.val_ratio,
        },
    )

    ds_train = CAERSTwoStreamDataset(
        manifest_path=cfg.outputs.manifest_path,
        dataset_root=cfg.dataset.dataset_root,
        split="train",
        image_size=cfg.dataset.image_size,
    )
    ds_val = CAERSTwoStreamDataset(
        manifest_path=cfg.outputs.manifest_path,
        dataset_root=cfg.dataset.dataset_root,
        split="val",
        image_size=cfg.dataset.image_size,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    num_classes = len(ds_train.label_to_index)
    model = build_model(num_classes, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # Log model architecture
    wandb.watch(model, log="all", log_freq=100)

    start_epoch = 0
    best_val_acc = 0.0
    history: list[dict[str, object]] = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
        wandb.run.summary["resumed_from_epoch"] = start_epoch
        wandb.run.summary["resumed_best_val_acc"] = best_val_acc

    ensure_parent_dir(cfg.train.save_dir / "placeholder")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.train.num_epochs}")
        train_metrics = train_one_epoch(model, loader_train, optimizer, criterion, device)
        val_metrics = evaluate(model, loader_val, criterion, device)

        print(f"  train loss={train_metrics['loss']:.4f} acc1={train_metrics['acc1']:.2f}%")
        print(f"  val   loss={val_metrics['loss']:.4f} acc1={val_metrics['acc1']:.2f}%")

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics["loss"],
            "train/acc1": train_metrics["acc1"],
            "train/acc5": train_metrics["acc5"],
            "val/loss": val_metrics["loss"],
            "val/acc1": val_metrics["acc1"],
            "val/acc5": val_metrics["acc5"],
        })

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["acc1"] > best_val_acc:
            best_val_acc = val_metrics["acc1"]
            ckpt_path = cfg.train.save_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "label_to_index": ds_train.label_to_index,
                "index_to_label": ds_train.index_to_label,
            }, ckpt_path)
            print(f"  Saved best model -> {ckpt_path}")

            # Log best model as W&B artifact
            artifact = wandb.Artifact(
                name=f"caers-model-{wandb.run.id}",
                type="model",
                metadata={
                    "epoch": epoch + 1,
                    "val_acc1": best_val_acc,
                    "stream_mode": cfg.train.stream_mode,
                },
            )
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact, aliases=["best"])

    history_path = cfg.train.save_dir / "history.json"
    write_json(history_path, {"history": history, "best_val_acc": best_val_acc})
    print(f"Training complete. History saved to {history_path}")

    # Log final history and summary
    wandb.run.summary["best_val_acc"] = best_val_acc
    wandb.run.summary["total_epochs"] = cfg.train.num_epochs
    wandb.save(str(history_path))

    wandb.finish()


if __name__ == "__main__":
    main()
