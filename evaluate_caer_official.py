#!/usr/bin/env python3
"""Evaluate an upstream-community CAER-Net checkpoint on a guarded split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
CAER_CODE_DIR = REPO_ROOT / "third_party" / "CAER" / "CAER"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "experiments"
CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expected_calibration_error(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    bins: int = 15,
) -> float:
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    ece = torch.tensor(0.0)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = confidence.gt(lower) & confidence.le(upper)
        if mask.any():
            weight = mask.float().mean()
            ece += weight * (correct[mask].float().mean() - confidence[mask].mean()).abs()
    return float(ece)


def evaluation_collate(batch: list[Any]) -> Any:
    valid = [sample for sample in batch if sample is not None]
    if not valid:
        raise RuntimeError("Every sample in an evaluation batch failed preprocessing.")
    return default_collate(valid)


def load_upstream_components() -> tuple[Any, Any, Any]:
    upstream_path = str(CAER_CODE_DIR)
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
    from data_loader.data_loaders import MyDataset
    from model.model import CAERSNet
    from utils.util import get_transform

    return MyDataset, CAERSNet, get_transform


def build_dataset(root: Path, detector: Path) -> Any:
    MyDataset, _, get_transform = load_upstream_components()

    class EvaluationDataset(MyDataset):
        def __getitem__(self, index: int) -> Any:
            sample = super().__getitem__(index)
            if sample is None:
                return None
            inputs, label = sample
            image_path = self.data[index].split(",", maxsplit=1)[0]
            return inputs, label, image_path

    return EvaluationDataset(str(root), str(detector), get_transform(train=False))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def validate_detector(config: dict[str, Any], split: str, detector: Path) -> str:
    actual_hash = sha256(detector)
    expected_hash = config.get("experiment", {}).get("detector_hashes", {}).get(f"{split}.txt")
    if expected_hash is None:
        raise ValueError("Checkpoint config does not contain frozen detector hashes.")
    if actual_hash != expected_hash:
        raise ValueError(
            f"Detector hash mismatch for {split}: expected {expected_hash}, got {actual_hash}."
        )
    return actual_hash


def load_model(checkpoint: Path, config: dict[str, Any], device: torch.device) -> tuple[Any, dict[str, Any]]:
    _, CAERSNet, _ = load_upstream_components()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in payload["state_dict"].items()
    }
    model = CAERSNet(**config["arch"]["args"])
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, payload


@torch.inference_mode()
def evaluate(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    paths: list[str] = []
    confidences: list[float] = []
    total_loss = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    for inputs, target, image_path in tqdm(loader, desc="validation", unit="batch"):
        face = inputs["face"].to(device, non_blocking=device.type == "cuda")
        context = inputs["context"].to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        logits = model(face, context)
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)

        total_loss += criterion(logits, target).item()
        total += target.numel()
        labels.extend(target.cpu().tolist())
        predictions.extend(prediction.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
        paths.extend(image_path)

    return {
        "loss": total_loss / total,
        "labels": labels,
        "predictions": predictions,
        "confidences": confidences,
        "paths": paths,
    }


def write_outputs(
    raw: dict[str, Any],
    output_dir: Path,
    checkpoint: Path,
    checkpoint_payload: dict[str, Any],
    checkpoint_hash: str,
    detector_hash: str,
    split: str,
    params: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = raw["labels"]
    predictions = raw["predictions"]
    confidence_tensor = torch.tensor(raw["confidences"])
    correct_tensor = torch.tensor(labels).eq(torch.tensor(predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    per_class = {
        CLASS_NAMES[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(CLASS_NAMES))
    }
    metrics = {
        "split": split,
        "evaluated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "detector_sha256": detector_hash,
        "samples": len(labels),
        "loss": float(raw["loss"]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "nll": float(raw["loss"]),
        "ece_15": expected_calibration_error(confidence_tensor, correct_tensor, bins=15),
        "params": params,
        "class_order": CLASS_NAMES,
        "per_class": per_class,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "label",
                "label_name",
                "prediction",
                "prediction_name",
                "confidence",
                "correct",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for image_path, label, prediction, confidence in zip(
            raw["paths"],
            labels,
            predictions,
            raw["confidences"],
        ):
            writer.writerow(
                {
                    "image_path": image_path,
                    "label": label,
                    "label_name": CLASS_NAMES[label],
                    "prediction": prediction,
                    "prediction_name": CLASS_NAMES[prediction],
                    "confidence": confidence,
                    "correct": int(label == prediction),
                }
            )

    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    figure, axis = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=45,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    return metrics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock final test evaluation; never use this during tuning.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def validate_split_access(split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise PermissionError(
            "Test evaluation is locked. Pass --allow-test only for a finalized model."
        )


def main() -> None:
    args = build_parser().parse_args()
    try:
        validate_split_access(args.split, args.allow_test)
    except PermissionError as error:
        raise SystemExit(str(error)) from error

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = checkpoint.parent / "config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError("Checkpoint and its sibling config.json are required.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    configure_determinism(seed)

    from run_caer_official import prepare_data

    detector_dir = Path(config["experiment"]["detector_dir"])
    prepare_data(detector_dir=detector_dir)
    loader_key = "val_loader" if args.split == "val" else "test_loader"
    loader_args = config[loader_key]["args"]
    data_root = CAER_CODE_DIR / loader_args["root"]
    detector = CAER_CODE_DIR / loader_args["detect_file"]
    detector_hash = validate_detector(config, args.split, detector)
    dataset = build_dataset(data_root, detector)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=evaluation_collate,
    )
    device = resolve_device(args.device)
    model, checkpoint_payload = load_model(checkpoint, config, device)
    raw = evaluate(model, loader, device)
    output_dir = args.output_dir or ARTIFACT_ROOT / checkpoint.parent.name / f"{args.split}_evaluation"
    metrics_path = write_outputs(
        raw=raw,
        output_dir=output_dir.resolve(),
        checkpoint=checkpoint,
        checkpoint_payload=checkpoint_payload,
        checkpoint_hash=sha256(checkpoint),
        detector_hash=detector_hash,
        split=args.split,
        params=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "split": metrics["split"],
                "samples": metrics["samples"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "neutral_f1": metrics["per_class"]["Neutral"]["f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
