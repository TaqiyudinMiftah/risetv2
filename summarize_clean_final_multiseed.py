#!/usr/bin/env python3
"""Summarize completed clean CAER-Net final runs from validation artifacts only.

This utility deliberately opens exactly two artifacts per run: its completed
metadata record and ``checkpoints/<run_id>/val_metrics.json``.  It never loads
checkpoints, predictions, manifests, datasets, or logical-test artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent
METADATA_ROOT = REPO_ROOT / "artifacts" / "experiments"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
ARTIFACT_ROOT = REPO_ROOT / "artifacts"
EXPECTED_SEEDS = (42, 43, 44)
DEFAULT_RUN_IDS = (
    "caernet__clean_inrepo_final__seed42__20260722_073316",
    "caernet__clean_inrepo_final__seed43__20260722_073316",
    "caernet__clean_inrepo_final__seed44__20260722_073316",
)
REQUIRED_SCALAR_METRICS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "loss",
    "nll",
    "ece_15",
    "samples",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_value(values: Mapping[str, Any], key: str, label: str) -> float:
    value = values.get(key)
    if not _is_number(value):
        raise ValueError(f"{label} must contain a finite numeric {key!r} value.")
    return float(value)


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return {
        "values": list(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def validate_run_id(run_id: str) -> str:
    """Reject path-like IDs before resolving fixed local artifact paths."""

    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run IDs must be single non-empty run directory names.")
    return run_id


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError as error:
        raise ValueError(f"Path must remain inside the repository: {path}") from error


def _metadata_path(run_id: str, repo_root: Path) -> Path:
    return repo_root / "artifacts" / "experiments" / run_id / "run_metadata.json"


def _validation_metrics_path(run_id: str, repo_root: Path) -> Path:
    return repo_root / "checkpoints" / run_id / "val_metrics.json"


def _validate_metadata(metadata: Mapping[str, Any], run_id: str, repo_root: Path) -> Path:
    if metadata.get("status") != "completed":
        raise ValueError(f"Run {run_id} is not completed: {metadata.get('status')!r}.")
    if metadata.get("track") != "clean_inrepo":
        raise ValueError(f"Run {run_id} is not from the clean in-repository track.")
    if metadata.get("stage") != "final":
        raise ValueError(f"Run {run_id} is not a final run.")
    if metadata.get("test_used_for_selection") is not False:
        raise ValueError(f"Run {run_id} does not prove test_used_for_selection is false.")

    metrics_path = _validation_metrics_path(run_id, repo_root)
    expected_metrics_reference = _relative_to_repo(metrics_path, repo_root)
    if metadata.get("val_metrics") != expected_metrics_reference:
        raise ValueError(
            f"Run {run_id} must reference only its fixed validation metrics artifact "
            f"{expected_metrics_reference!r}."
        )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Validation metrics not found: {metrics_path}")
    return metrics_path


def _metric_schema(metrics: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    for metric in REQUIRED_SCALAR_METRICS:
        _numeric_value(metrics, metric, f"Run {run_id} validation metrics")

    scalar_metrics = tuple(sorted(key for key, value in metrics.items() if _is_number(value)))
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping) or not per_class:
        raise ValueError(f"Run {run_id} validation metrics must contain non-empty per_class results.")

    per_class_schema: dict[str, tuple[str, ...]] = {}
    for class_name, class_metrics in per_class.items():
        if not isinstance(class_name, str) or not isinstance(class_metrics, Mapping):
            raise ValueError(f"Run {run_id} has malformed per-class validation metrics.")
        numeric_metrics = tuple(
            sorted(key for key, value in class_metrics.items() if _is_number(value))
        )
        if not numeric_metrics:
            raise ValueError(f"Run {run_id} class {class_name!r} has no numeric metrics.")
        per_class_schema[class_name] = numeric_metrics
    if "Neutral" not in per_class_schema or "f1" not in per_class_schema["Neutral"]:
        raise ValueError(f"Run {run_id} validation metrics must contain Neutral F1.")

    return {
        "scalar_metrics": scalar_metrics,
        "per_class": per_class_schema,
    }


def _load_run(run_id: str, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load one completed clean final run without opening any test artifact."""

    run_id = validate_run_id(run_id)
    metadata_path = _metadata_path(run_id, repo_root)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    metadata = _load_json_object(metadata_path)
    if metadata.get("run_id") != run_id:
        raise ValueError(f"Run metadata ID mismatch for {run_id}.")
    metrics_path = _validate_metadata(metadata, run_id, repo_root)
    metrics = _load_json_object(metrics_path)
    schema = _metric_schema(metrics, run_id)
    return {
        "run_id": run_id,
        "metadata": metadata,
        "metrics": metrics,
        "schema": schema,
        "metrics_path": _relative_to_repo(metrics_path, repo_root),
    }


def _validate_run_consistency(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    reference_metadata = reference["metadata"]
    candidate_metadata = candidate["metadata"]
    run_id = candidate["run_id"]
    for key in ("protocol", "manifest_sha256", "detector_hashes"):
        if candidate_metadata.get(key) != reference_metadata.get(key):
            raise ValueError(f"Run {run_id} differs from the final run set in metadata field {key!r}.")
    if candidate["schema"] != reference["schema"]:
        raise ValueError(f"Run {run_id} validation metric schema differs from the final run set.")

    reference_metrics = reference["metrics"]
    candidate_metrics = candidate["metrics"]
    if int(_numeric_value(candidate_metrics, "samples", f"Run {run_id} validation metrics")) != int(
        _numeric_value(reference_metrics, "samples", "Reference validation metrics")
    ):
        raise ValueError(f"Run {run_id} has a different validation sample count.")


def _run_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    per_class = metrics["per_class"]
    schema = run["schema"]
    return {
        "seed": int(run["metadata"]["seed"]),
        "run_id": run["run_id"],
        "best_epoch": int(run["metadata"]["best_epoch"]),
        "validation_metrics": run["metrics_path"],
        "metrics": {
            metric: _numeric_value(metrics, metric, f"Run {run['run_id']} validation metrics")
            for metric in schema["scalar_metrics"]
        },
        "per_class": {
            class_name: {
                metric: _numeric_value(
                    per_class[class_name],
                    metric,
                    f"Run {run['run_id']} class {class_name!r}",
                )
                for metric in class_metrics
            }
            for class_name, class_metrics in schema["per_class"].items()
        },
    }


def summarize(
    run_ids: Sequence[str],
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Compute validation-only mean and sample standard deviation across final seeds."""

    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Each run ID may appear only once.")
    runs = [_load_run(run_id, repo_root) for run_id in run_ids]
    if not runs:
        raise ValueError("At least one completed clean final run is required.")
    runs.sort(key=lambda run: int(run["metadata"]["seed"]))
    seeds = [int(run["metadata"]["seed"]) for run in runs]
    required_seeds = sorted(int(seed) for seed in expected_seeds)
    if seeds != required_seeds:
        raise ValueError(f"Expected final seeds {required_seeds}, got {seeds}.")

    reference = runs[0]
    for run in runs[1:]:
        _validate_run_consistency(reference, run)

    schema = reference["schema"]
    aggregate = {
        metric: _metric_summary(
            [_numeric_value(run["metrics"], metric, f"Run {run['run_id']} validation metrics") for run in runs]
        )
        for metric in schema["scalar_metrics"]
    }
    per_class = {
        class_name: {
            metric: _metric_summary(
                [
                    _numeric_value(
                        run["metrics"]["per_class"][class_name],
                        metric,
                        f"Run {run['run_id']} class {class_name!r}",
                    )
                    for run in runs
                ]
            )
            for metric in class_metrics
        }
        for class_name, class_metrics in schema["per_class"].items()
    }
    reference_metadata = reference["metadata"]
    return {
        "model": "CAER-Net",
        "track": "clean_inrepo",
        "stage": "final",
        "protocol": reference_metadata["protocol"],
        "logical_split": "val",
        "test_used_for_selection": False,
        "test_accessed": False,
        "test_artifacts_read": False,
        "seeds": seeds,
        "samples_per_seed": int(
            _numeric_value(reference["metrics"], "samples", "Reference validation metrics")
        ),
        "manifest_sha256": reference_metadata["manifest_sha256"],
        "detector_hashes": reference_metadata["detector_hashes"],
        "runs": [_run_summary(run) for run in runs],
        "aggregate": aggregate,
        "per_class": per_class,
    }


def write_summary(summary: Mapping[str, Any], output_path: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Write an optional JSON record only beneath the ignored ``artifacts/`` tree."""

    candidate = output_path.expanduser()
    resolved_output = candidate if candidate.is_absolute() else repo_root / candidate
    resolved_output = resolved_output.resolve()
    artifact_root = (repo_root / "artifacts").resolve()
    try:
        resolved_output.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("--output-json must be located under the ignored artifacts/ directory.") from error
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Completed clean final run ID. Repeat once for each seed; defaults to the frozen 42/43/44 run IDs.",
    )
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional ignored JSON output path beneath artifacts/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_ids = args.run_ids if args.run_ids is not None else list(DEFAULT_RUN_IDS)
    summary = summarize(run_ids, args.expected_seeds)
    output_path = write_summary(summary, args.output_json) if args.output_json is not None else None
    print(
        json.dumps(
            {
                "output_json": str(output_path) if output_path is not None else None,
                "seeds": summary["seeds"],
                "accuracy": summary["aggregate"]["accuracy"],
                "macro_f1": summary["aggregate"]["macro_f1"],
                "weighted_f1": summary["aggregate"]["weighted_f1"],
                "test_accessed": summary["test_accessed"],
                "test_artifacts_read": summary["test_artifacts_read"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
