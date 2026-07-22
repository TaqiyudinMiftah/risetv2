#!/usr/bin/env python3
"""Reproduce a completed clean CAER-Net run on the logical validation split only.

The content-disjoint protocol stores logical validation images beneath the
physical ``CAER-S/test/`` directory.  This verifier always filters the frozen
manifest with ``split="val"`` and has no option to select the logical test
split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from caer_research.checkpointing import load_model_checkpoint
from caer_research.data import CAERSTwoStreamDataset, build_transforms
from caer_research.engine import evaluate
from caer_research.models import CAERNet


REPO_ROOT = Path(__file__).resolve().parent
VALIDATION_SPLIT = "val"
PREDICTION_FIELDS = ("image_path", "label", "prediction", "confidence", "correct")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a frozen relative path against the repository root."""

    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def validate_run_id(run_id: str) -> str:
    """Reject path-like run IDs before resolving their artifact directories."""

    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be a single non-empty run directory name.")
    return run_id


def run_paths(run_id: str, repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    """Return the fixed completed-run artifacts used by this verifier."""

    validated_run_id = validate_run_id(run_id)
    run_dir = repo_root / "checkpoints" / validated_run_id
    metadata_dir = repo_root / "artifacts" / "experiments" / validated_run_id
    return {
        "run_dir": run_dir,
        "metadata": metadata_dir / "run_metadata.json",
        "config": run_dir / "config.json",
        "checkpoint": run_dir / "best.pt",
        "metrics": run_dir / "val_metrics.json",
        "predictions": run_dir / "val_predictions.csv",
        "output": metadata_dir / "validation_reproduction.json",
    }


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, float | int]:
    """Flatten a nested numeric metric object into dot-separated metric names."""

    if isinstance(value, Mapping):
        flattened: dict[str, float | int] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_metrics(child, child_prefix))
        return flattened
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"Unsupported metric value at {prefix!r}: {type(value).__name__}.")
    return {prefix: value}


def compare_metrics(
    saved_metrics: Mapping[str, Any], reproduced_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare complete public metric schemas and return absolute deltas."""

    saved_flat = flatten_metrics(saved_metrics)
    reproduced_flat = flatten_metrics(reproduced_metrics)
    saved_keys = set(saved_flat)
    reproduced_keys = set(reproduced_flat)
    if saved_keys != reproduced_keys:
        raise ValueError(
            "Saved and reproduced metric schemas differ: "
            f"missing={sorted(saved_keys - reproduced_keys)}, "
            f"unexpected={sorted(reproduced_keys - saved_keys)}."
        )
    deltas = {
        key: abs(float(reproduced_flat[key]) - float(saved_flat[key]))
        for key in sorted(saved_flat)
    }
    return {
        "metric_deltas": deltas,
        "metric_max_abs_delta": max(deltas.values(), default=0.0),
    }


def load_saved_predictions(path: Path) -> list[dict[str, str]]:
    """Load the saved validation predictions and enforce their frozen schema."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
            raise ValueError(
                f"Unexpected prediction columns in {path}: {reader.fieldnames!r}; "
                f"expected {PREDICTION_FIELDS!r}."
            )
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if any(row.get(field) is None for field in PREDICTION_FIELDS):
                raise ValueError(f"Missing prediction value in {path} row {row_number}.")
            rows.append({field: str(row[field]) for field in PREDICTION_FIELDS})
    return rows


def compare_predictions(
    saved_rows: Sequence[Mapping[str, str]],
    image_paths: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    confidences: Sequence[float],
) -> dict[str, Any]:
    """Compare validation prediction rows while preserving their evaluation order."""

    reproduced_lengths = {len(image_paths), len(labels), len(predictions), len(confidences)}
    if len(reproduced_lengths) != 1:
        raise ValueError("Reproduced image paths, labels, predictions, and confidences differ in length.")

    reproduced_count = len(labels)
    prediction_mismatches = abs(len(saved_rows) - reproduced_count)
    max_confidence_delta = 0.0
    for saved, image_path, label, prediction, confidence in zip(
        saved_rows,
        image_paths,
        labels,
        predictions,
        confidences,
        strict=False,
    ):
        expected_correct = int(int(label) == int(prediction))
        if (
            saved["image_path"] != str(image_path)
            or int(saved["label"]) != int(label)
            or int(saved["prediction"]) != int(prediction)
            or int(saved["correct"]) != expected_correct
        ):
            prediction_mismatches += 1
        max_confidence_delta = max(
            max_confidence_delta,
            abs(float(saved["confidence"]) - float(confidence)),
        )
    return {
        "saved_prediction_count": len(saved_rows),
        "reproduced_prediction_count": reproduced_count,
        "prediction_mismatches": prediction_mismatches,
        "max_confidence_abs_delta": max_confidence_delta,
    }


def assert_logical_validation_only(dataset: CAERSTwoStreamDataset) -> None:
    """Prove the instantiated dataset contains only logical validation samples."""

    if dataset.split != VALIDATION_SPLIT:
        raise ValueError(f"Verifier requires split={VALIDATION_SPLIT!r}, got {dataset.split!r}.")
    non_validation_samples = [sample.sample_id for sample in dataset.samples if sample.split != VALIDATION_SPLIT]
    if non_validation_samples:
        raise ValueError(
            "Validation verifier dataset includes non-validation samples: "
            f"{non_validation_samples[:3]!r}."
        )


def public_evaluation_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove per-sample payloads before comparing the saved metric artifact."""

    excluded = {"labels", "predictions", "confidences", "image_paths"}
    return {key: value for key, value in result.items() if key not in excluded}


def validate_completed_run(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Ensure the target is a completed clean run whose test split stayed locked."""

    if metadata.get("status") != "completed":
        raise ValueError(f"Validation reproduction requires a completed run, got {metadata.get('status')!r}.")
    if metadata.get("test_used_for_selection") is not False:
        raise ValueError("Run metadata does not prove that the test split stayed out of selection.")
    research = config.get("research")
    if not isinstance(research, Mapping):
        raise ValueError("Frozen runtime config is missing research metadata.")
    if research.get("track") != "clean_inrepo":
        raise ValueError(f"Expected a clean_inrepo run, got {research.get('track')!r}.")
    if research.get("test_during_training") is not False:
        raise ValueError("Frozen runtime config permits test access during training.")


def reproduce_validation(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    repo_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], CAERSTwoStreamDataset]:
    """Evaluate ``best.pt`` over logical validation samples only."""

    data_config = config.get("data")
    model_config = config.get("model")
    if not isinstance(data_config, Mapping) or not isinstance(model_config, Mapping):
        raise ValueError("Frozen runtime config is missing data or model settings.")
    manifest_path = resolve_repo_path(str(data_config["manifest"]), repo_root)
    dataset_root = resolve_repo_path(str(data_config["dataset_root"]), repo_root)
    if not manifest_path.is_file() or not dataset_root.is_dir():
        raise FileNotFoundError("Frozen manifest and CAER-S dataset root are required for validation.")
    model_args = model_config.get("args")
    if not isinstance(model_args, Mapping):
        raise ValueError("Frozen runtime config is missing model arguments.")

    face_transform, context_transform = build_transforms(train=False)
    validation_dataset = CAERSTwoStreamDataset(
        manifest_path,
        dataset_root,
        split=VALIDATION_SPLIT,
        face_transform=face_transform,
        context_transform=context_transform,
    )
    assert_logical_validation_only(validation_dataset)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = CAERNet(**dict(model_args))
    load_model_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to(device)
    result = evaluate(model, validation_loader, nn.CrossEntropyLoss(), device, use_amp=False)
    return result, validation_dataset


def verification_result(
    run_id: str,
    paths: Mapping[str, Path],
    saved_metrics: Mapping[str, Any],
    reproduced_result: Mapping[str, Any],
    prediction_comparison: Mapping[str, Any],
    metric_comparison: Mapping[str, Any],
    metric_atol: float,
    confidence_atol: float,
) -> dict[str, Any]:
    """Build a portable validation-only reproducibility record."""

    reproduced_metrics = public_evaluation_metrics(reproduced_result)
    verification_passed = (
        float(metric_comparison["metric_max_abs_delta"]) <= metric_atol
        and int(prediction_comparison["prediction_mismatches"]) == 0
        and float(prediction_comparison["max_confidence_abs_delta"]) <= confidence_atol
    )
    return {
        "run_id": run_id,
        "status": "passed" if verification_passed else "failed",
        "logical_split": VALIDATION_SPLIT,
        "test_accessed": False,
        "test_split_loaded": False,
        "test_images_loaded": False,
        "effective_config": str(paths["config"].relative_to(REPO_ROOT)),
        "effective_config_sha256": sha256(paths["config"]),
        "checkpoint": str(paths["checkpoint"].relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "saved_metrics": dict(saved_metrics),
        "reproduced_metrics": reproduced_metrics,
        **dict(metric_comparison),
        **dict(prediction_comparison),
        "metric_atol": metric_atol,
        "confidence_atol": confidence_atol,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Completed clean run ID to reproduce.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for validation (default: cuda:0; CPU is supported for diagnostics).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (default: artifacts/experiments/<run-id>/validation_reproduction.json).",
    )
    parser.add_argument("--metric-atol", type=float, default=0.0)
    parser.add_argument("--confidence-atol", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metric_atol < 0.0 or args.confidence_atol < 0.0:
        raise ValueError("Verification tolerances must be non-negative.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but torch.cuda.is_available() is false.")

    paths = run_paths(args.run_id)
    required_paths = ("metadata", "config", "checkpoint", "metrics", "predictions")
    missing_paths = [str(paths[key]) for key in required_paths if not paths[key].is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Required completed-run artifacts are missing: {missing_paths!r}.")
    metadata = load_json(paths["metadata"])
    config = load_json(paths["config"])
    validate_completed_run(metadata, config)

    saved_metrics = load_json(paths["metrics"])
    saved_predictions = load_saved_predictions(paths["predictions"])
    reproduced_result, validation_dataset = reproduce_validation(
        config,
        paths["checkpoint"],
        REPO_ROOT,
        device,
    )
    reproduced_metrics = public_evaluation_metrics(reproduced_result)
    metric_comparison = compare_metrics(saved_metrics, reproduced_metrics)
    prediction_comparison = compare_predictions(
        saved_predictions,
        reproduced_result["image_paths"],
        reproduced_result["labels"],
        reproduced_result["predictions"],
        reproduced_result["confidences"],
    )
    result = verification_result(
        args.run_id,
        paths,
        saved_metrics,
        reproduced_result,
        prediction_comparison,
        metric_comparison,
        args.metric_atol,
        args.confidence_atol,
    )
    result["samples"] = len(validation_dataset)
    result["device"] = str(device)
    result["num_workers"] = 0

    output_path = args.output.expanduser().resolve() if args.output is not None else paths["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **result}, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit("Validation reproduction differs from the saved artifact; see the JSON output.")


if __name__ == "__main__":
    main()
