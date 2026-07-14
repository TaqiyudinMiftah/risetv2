#!/usr/bin/env python3
"""Freeze and evaluate the selected upstream CAER-Net baseline for Phase 0."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from research_audit import (
    CLASS_NAMES,
    LABEL_TO_INDEX,
    DetectorSample,
    canonicalize_detector_path,
    canonicalize_label,
    configure_determinism,
    extract_state_dict,
    load_checkpoint_payload,
    read_detector_file,
    sha256_file,
    validate_bbox,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent
UPSTREAM_CODE_DIR = REPO_ROOT / "third_party" / "CAER" / "CAER"
DEFAULT_CHECKPOINT = (
    UPSTREAM_CODE_DIR
    / "official_runs"
    / "models"
    / "CAER_S_Official_CAERNet"
    / "0701_023049"
    / "model_best.pth"
)
DEFAULT_RUN_CONFIG = DEFAULT_CHECKPOINT.parent / "config.json"


class OfficialCAERSTestDataset(Dataset[tuple[dict[str, torch.Tensor], int, str]]):
    """Evaluation dataset that matches the upstream PIL crop and masking behavior."""

    def __init__(self, root: Path, samples: list[DetectorSample], transform: Any) -> None:
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], int, str]:
        sample = self.samples[index]
        image_path = self.root / sample.image_path
        with Image.open(image_path) as source:
            image = source.copy()

        x1, y1, x2, y2 = sample.face_bbox
        face = image.crop((x1, y1, x2, y2))
        context = image.copy()
        ImageDraw.Draw(context).rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
        transformed = self.transform({"face": face, "context": context})
        return transformed, sample.label_index, sample.image_path


def _json_dump(content: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _manifest_entry(row: dict[str, Any]) -> tuple[str, int, tuple[int, int, int, int]]:
    image_path = PurePosixPath(str(row["image_path"]))
    if len(image_path.parts) < 3:
        raise ValueError(f"Manifest image path is incomplete: {row['image_path']!r}")
    relative_path = canonicalize_detector_path(PurePosixPath(*image_path.parts[1:]).as_posix())
    label = canonicalize_label(str(row["label"]))
    bbox = tuple(int(value) for value in row["face_bbox"])
    return relative_path, LABEL_TO_INDEX[label], bbox


def audit_manifest(manifest_path: Path, dataset_root: Path, detector_dir: Path) -> dict[str, Any]:
    """Validate split provenance, bbox diagnostics, and detector-manifest parity."""
    rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    expected_splits = ("train", "val", "test")
    required_fields = {"sample_id", "image_path", "label", "split", "face_bbox"}
    grouped_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in expected_splits}
    issues: Counter[str] = Counter()
    examples: defaultdict[str, list[Any]] = defaultdict(list)
    sample_ids: set[str] = set()
    canonical_paths: dict[str, set[str]] = {split: set() for split in expected_splits}
    physical_paths: dict[str, list[Path]] = {split: [] for split in expected_splits}
    class_counts: dict[str, Counter[str]] = {split: Counter() for split in expected_splits}
    bbox_total = 0
    bbox_inside_image = 0

    for row_number, row in enumerate(rows, 1):
        missing_fields = required_fields.difference(row)
        if missing_fields:
            issues["missing_required_fields"] += 1
            examples["missing_required_fields"].append([row_number, sorted(missing_fields)])
            continue
        split = row["split"]
        if split not in grouped_rows:
            issues["unknown_split"] += 1
            examples["unknown_split"].append([row_number, split])
            continue
        grouped_rows[split].append(row)
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            issues["duplicate_sample_id"] += 1
            examples["duplicate_sample_id"].append([row_number, sample_id])
        sample_ids.add(sample_id)

        try:
            relative_path, label_index, bbox = _manifest_entry(row)
            label = canonicalize_label(str(row["label"]))
        except (TypeError, ValueError) as error:
            issues["invalid_manifest_row"] += 1
            examples["invalid_manifest_row"].append([row_number, str(error)])
            continue
        class_counts[split][label] += 1
        if relative_path in canonical_paths[split]:
            issues["duplicate_image_within_split"] += 1
            examples["duplicate_image_within_split"].append([row_number, relative_path])
        canonical_paths[split].add(relative_path)

        expected_disk_split = "train" if split == "train" else "test"
        actual_disk_split = PurePosixPath(str(row["image_path"])).parts[0]
        if actual_disk_split != expected_disk_split:
            issues["invalid_split_provenance"] += 1
            if len(examples["invalid_split_provenance"]) < 10:
                examples["invalid_split_provenance"].append(
                    [row_number, split, str(row["image_path"]), expected_disk_split]
                )
        manifest_image_path = PurePosixPath(str(row["image_path"]))
        image_path = (
            dataset_root
            / expected_disk_split
            / manifest_image_path.parts[-2]
            / manifest_image_path.name
        )
        if not image_path.is_file():
            issues["missing_image"] += 1
            examples["missing_image"].append([row_number, str(image_path)])
            continue
        physical_paths[split].append(image_path)
        with Image.open(image_path) as image:
            image_size = image.size
        bbox_total += 1
        if validate_bbox(bbox, image_size, require_inside_image=True):
            bbox_inside_image += 1
        else:
            issues["bbox_outside_image"] += 1
            if len(examples["bbox_outside_image"]) < 10:
                examples["bbox_outside_image"].append([row_number, str(row["image_path"]), list(bbox), list(image_size)])

    detector_matches: dict[str, bool] = {}
    detector_counts: dict[str, int] = {}
    for split in expected_splits:
        detector_samples = read_detector_file(detector_dir / f"{split}.txt")
        detector_entries = [(sample.image_path, sample.label_index, sample.face_bbox) for sample in detector_samples]
        manifest_entries = [_manifest_entry(row) for row in grouped_rows[split]]
        detector_matches[split] = detector_entries == manifest_entries
        detector_counts[split] = len(detector_entries)
        if not detector_matches[split]:
            issues["detector_manifest_mismatch"] += 1
            examples["detector_manifest_mismatch"].append(split)

    filename_namespace_collisions = {
        "train_val": len(canonical_paths["train"] & canonical_paths["val"]),
        "train_test": len(canonical_paths["train"] & canonical_paths["test"]),
        "val_test": len(canonical_paths["val"] & canonical_paths["test"]),
    }
    # CAER-S restarts image filenames in each split. A matching class/filename
    # is therefore not evidence of leakage; exact file-content hashes are.
    image_hash_index: dict[str, defaultdict[str, list[str]]] = {}
    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split in expected_splits:
            index: defaultdict[str, list[str]] = defaultdict(list)
            for image_path, digest in zip(physical_paths[split], executor.map(sha256_file, physical_paths[split])):
                index[digest].append(str(image_path.relative_to(dataset_root)))
            image_hash_index[split] = index
    image_hashes = {split: set(index) for split, index in image_hash_index.items()}
    content_overlaps = {
        "train_val": len(image_hashes["train"] & image_hashes["val"]),
        "train_test": len(image_hashes["train"] & image_hashes["test"]),
        "val_test": len(image_hashes["val"] & image_hashes["test"]),
    }
    for name, count in content_overlaps.items():
        if count:
            issues[f"content_overlap_{name}"] += count
    content_overlap_examples: dict[str, list[dict[str, Any]]] = {}
    for name, left, right in (("train_val", "train", "val"), ("train_test", "train", "test"), ("val_test", "val", "test")):
        content_overlap_examples[name] = [
            {
                "sha256": digest,
                left: image_hash_index[left][digest],
                right: image_hash_index[right][digest],
            }
            for digest in sorted(image_hashes[left] & image_hashes[right])[:25]
        ]

    blocking_issue_keys = {
        "missing_required_fields",
        "unknown_split",
        "duplicate_sample_id",
        "invalid_manifest_row",
        "invalid_split_provenance",
        "duplicate_image_within_split",
        "missing_image",
        "detector_manifest_mismatch",
        "content_overlap_train_val",
        "content_overlap_train_test",
        "content_overlap_val_test",
    }
    return {
        "manifest": str(manifest_path),
        "total_rows": len(rows),
        "split_counts": {split: len(grouped_rows[split]) for split in expected_splits},
        "detector_counts": detector_counts,
        "class_counts": {split: dict(class_counts[split]) for split in expected_splits},
        "detector_manifest_order_match": detector_matches,
        "filename_namespace_collisions": filename_namespace_collisions,
        "unique_image_content_hashes": {split: len(image_hashes[split]) for split in expected_splits},
        "duplicate_image_content_within_split": {
            split: sum(len(paths) - 1 for paths in image_hash_index[split].values())
            for split in expected_splits
        },
        "content_hash_overlap": content_overlaps,
        "content_hash_overlap_examples": content_overlap_examples,
        "bbox": {
            "total": bbox_total,
            "inside_image": bbox_inside_image,
            "outside_image": issues["bbox_outside_image"],
            "upstream_compatible": bbox_total == bbox_inside_image + issues["bbox_outside_image"],
            "note": "Out-of-image detector boxes are retained because upstream CAER uses PIL.Image.crop, which pads those crops.",
        },
        "issues": dict(issues),
        "examples": {key: value[:10] for key, value in examples.items()},
        "upstream_protocol_valid": not any(issues[key] for key in blocking_issue_keys),
    }


def _expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = 15) -> float:
    confidences_array = np.asarray(confidences)
    correct_array = np.asarray(correct, dtype=float)
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins)):
        mask = (confidences_array > lower) & (confidences_array <= upper)
        if mask.any():
            ece += abs(correct_array[mask].mean() - confidences_array[mask].mean()) * mask.mean()
    return float(ece)


def _import_upstream_components() -> tuple[type[torch.nn.Module], Any]:
    code_dir = str(UPSTREAM_CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from model.model import CAERSNet
    from utils.util import get_transform

    return CAERSNet, get_transform


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    samples: list[DetectorSample],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Evaluate from a newly loaded upstream checkpoint and retain every prediction."""
    CAERSNet, get_transform = _import_upstream_components()
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    checkpoint_config = payload["config"].config
    architecture_args = dict(checkpoint_config["arch"]["args"])
    model = CAERSNet(**architecture_args)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    model.to(device).eval()

    dataset = OfficialCAERSTestDataset(data_root, samples, get_transform(train=False))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    rows: list[dict[str, Any]] = []
    total_nll = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets, paths in loader:
            face = inputs["face"].to(device, non_blocking=device.type == "cuda")
            context = inputs["context"].to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            logits = model(face, context)
            total_nll += functional.cross_entropy(logits, targets, reduction="sum").item()
            probabilities = torch.softmax(logits, dim=1)
            batch_confidence, batch_predictions = probabilities.max(dim=1)
            batch_labels = targets.cpu().tolist()
            batch_predictions_list = batch_predictions.cpu().tolist()
            batch_confidence_list = batch_confidence.cpu().tolist()
            labels.extend(batch_labels)
            predictions.extend(batch_predictions_list)
            confidences.extend(batch_confidence_list)
            for image_path, label, prediction, confidence in zip(paths, batch_labels, batch_predictions_list, batch_confidence_list):
                rows.append(
                    {
                        "image_path": image_path,
                        "label": label,
                        "label_name": CLASS_NAMES[label],
                        "prediction": prediction,
                        "prediction_name": CLASS_NAMES[prediction],
                        "confidence": float(confidence),
                        "correct": label == prediction,
                    }
                )
    elapsed_seconds = time.perf_counter() - started

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    metrics = {
        "test_samples": len(labels),
        "test_nll": total_nll / len(labels),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "test_weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "test_ece_15_bins": _expected_calibration_error(confidences, [row["correct"] for row in rows]),
        "latency_seconds_total": elapsed_seconds,
        "latency_milliseconds_per_sample": elapsed_seconds * 1000 / len(labels),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(CLASS_NAMES))
        },
    }
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return metrics, rows, confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES)))), report


def _write_evaluation_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    report: dict[str, Any],
) -> None:
    _json_dump(metrics, output_dir / "metrics.json")
    _json_dump(report, output_dir / "classification_report.json")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *CLASS_NAMES])
        for label, values in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([label, *values])

    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="CAER-Net official reproduction - test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2
    for row_index, column_index in np.ndindex(matrix.shape):
        axis.text(
            column_index,
            row_index,
            str(matrix[row_index, column_index]),
            ha="center",
            va="center",
            color="white" if matrix[row_index, column_index] > threshold else "black",
        )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "caers_manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "CAER-S")
    parser.add_argument("--detector-dir", type=Path, default=REPO_ROOT / "detectors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=123)
    parser.add_argument(
        "--allow-known-content-overlap",
        action="store_true",
        help="Evaluate the historical upstream protocol after explicitly recording detected split overlap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    configure_determinism(args.evaluation_seed)

    from run_caer_official import prepare_data

    prepare_data()
    manifest_audit = audit_manifest(args.manifest, args.dataset_root, args.detector_dir)
    _json_dump(manifest_audit, output_dir / "manifest_audit.json")
    if not manifest_audit["upstream_protocol_valid"]:
        allowed_issue_keys = {"bbox_outside_image"}
        if args.allow_known_content_overlap:
            allowed_issue_keys.update(key for key in manifest_audit["issues"] if key.startswith("content_overlap_"))
        unexpected_issues = set(manifest_audit["issues"]).difference(allowed_issue_keys)
        if unexpected_issues or not args.allow_known_content_overlap:
            raise RuntimeError("Manifest audit failed; review manifest_audit.json before evaluating test data.")

    hashes = {
        "manifest": sha256_file(args.manifest),
        "train_detector": sha256_file(args.detector_dir / "train.txt"),
        "val_detector": sha256_file(args.detector_dir / "val.txt"),
        "test_detector": sha256_file(args.detector_dir / "test.txt"),
        "run_config": sha256_file(args.run_config),
        "checkpoint": sha256_file(args.checkpoint),
    }
    _json_dump(hashes, output_dir / "input_hashes.json")
    shutil.copy2(args.run_config, output_dir / "checkpoint_config.json")

    test_samples = read_detector_file(args.detector_dir / "test.txt")
    device = torch.device(args.device)
    metrics, rows, matrix, report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=UPSTREAM_CODE_DIR / "data" / "CAER-S" / "test",
        samples=test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    metadata = {
        "run_id": output_dir.name,
        "model": "CAER-Net",
        "variant": "upstream_official_local_reproduction",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": 32,
        "validation_accuracy_at_selection": 0.7674667758089367,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "historical_protocol_content_overlap_override": args.allow_known_content_overlap,
        "git_sha_at_audit": _git_sha(),
        "git_sha_at_training": "aaf26453feaa46c18713c13716b9bafae83589aa (inferred from generated-config timestamp)",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    metrics["metadata"] = metadata
    metrics["hashes"] = hashes
    _write_evaluation_outputs(output_dir, metrics, rows, matrix, report)
    _json_dump(metadata, output_dir / "run_metadata.json")
    print(json.dumps({key: metrics[key] for key in ("test_accuracy", "test_macro_f1", "test_weighted_f1", "test_ece_15_bins")}, indent=2))


if __name__ == "__main__":
    main()
