from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from caers_pipeline.engine import evaluate, evaluate_per_class
from caers_pipeline.io_utils import write_json
from caers_pipeline.model import CAERNet, SingleStreamNet


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
    parser = argparse.ArgumentParser(description="Evaluate CAER-Net on CAER-S test set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: test or val")
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
        job_type="evaluation",
        config={
            "eval_split": args.split,
            "checkpoint": args.checkpoint,
            "backbone": cfg.model.backbone,
            "pretrained": cfg.model.pretrained,
            "dropout": cfg.model.dropout,
            "batch_size": cfg.train.batch_size,
            "stream_mode": cfg.train.stream_mode,
            "image_size": cfg.dataset.image_size,
        },
    )

    ds_test = CAERSTwoStreamDataset(
        manifest_path=cfg.outputs.manifest_path,
        dataset_root=cfg.dataset.dataset_root,
        split=args.split,
        image_size=cfg.dataset.image_size,
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    num_classes = len(ds_test.label_to_index)
    model = build_model(num_classes, cfg).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    label_to_index = checkpoint.get("label_to_index", ds_test.label_to_index)
    index_to_label = {int(k): v for k, v in checkpoint.get("index_to_label", ds_test.index_to_label).items()}

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader_test, criterion, device)
    per_class = evaluate_per_class(model, loader_test, device, index_to_label)

    print(f"Results on '{args.split}':")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {metrics['acc1']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['acc5']:.2f}%")
    print("  Per-class Accuracy:")
    for label_name, acc in per_class["per_class_acc"].items():
        print(f"    {label_name}: {acc:.2f}%")

    # Log metrics to W&B
    wandb.log({
        f"{args.split}/loss": metrics["loss"],
        f"{args.split}/acc1": metrics["acc1"],
        f"{args.split}/acc5": metrics["acc5"],
        f"{args.split}/overall_acc": per_class["overall_acc"],
    })

    # Log per-class accuracy as a table
    class_data = []
    for label_name, acc in per_class["per_class_acc"].items():
        class_data.append([label_name, acc])
    
    class_table = wandb.Table(columns=["class", "accuracy"], data=class_data)
    wandb.log({f"{args.split}/per_class_accuracy": class_table})

    # Log per-class accuracy as bar chart
    wandb.log({
        f"{args.split}/per_class_acc_bar": wandb.plot.bar(
            class_table, "class", "accuracy", title=f"Per-Class Accuracy ({args.split})"
        )
    })

    # Log summary stats
    wandb.run.summary[f"{args.split}_loss"] = metrics["loss"]
    wandb.run.summary[f"{args.split}_acc1"] = metrics["acc1"]
    wandb.run.summary[f"{args.split}_acc5"] = metrics["acc5"]
    wandb.run.summary[f"{args.split}_overall_acc"] = per_class["overall_acc"]

    out = {
        "split": args.split,
        "metrics": metrics,
        "per_class_acc": per_class["per_class_acc"],
        "overall_acc": per_class["overall_acc"],
    }
    out_path = cfg.train.save_dir / f"eval_{args.split}.json"
    write_json(out_path, out)
    
    # Log evaluation JSON as artifact
    artifact = wandb.Artifact(
        name=f"caers-eval-{wandb.run.id}",
        type="evaluation",
        metadata={
            "split": args.split,
            "acc1": metrics["acc1"],
            "overall_acc": per_class["overall_acc"],
        },
    )
    artifact.add_file(str(out_path))
    wandb.log_artifact(artifact)
    
    print(f"Evaluation saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
