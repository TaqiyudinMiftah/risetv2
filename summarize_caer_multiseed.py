#!/usr/bin/env python3
"""Validate and aggregate final CAER-Net validation metrics across seeds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "experiments"
DEFAULT_JSON = ARTIFACT_ROOT / "caernet_final_multiseed_validation_summary.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "experiment1_caernet_final_results.md"
AGGREGATE_METRICS = ("accuracy", "macro_f1", "weighted_f1", "nll", "ece_15")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("seed", None)
    return normalized


def _metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _load_run(run_id: str) -> dict[str, Any]:
    metadata_path = ARTIFACT_ROOT / run_id / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "completed":
        raise ValueError(f"Run {run_id} is not completed: {metadata.get('status')}")
    if metadata.get("stage") != "final" or metadata.get("exploratory"):
        raise ValueError(f"Run {run_id} is not a final run.")
    if metadata.get("test_used_for_selection"):
        raise ValueError(f"Run {run_id} used test data for selection.")

    metrics_path = REPO_ROOT / metadata["validation_metrics"]
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Validation metrics not found: {metrics_path}")
    if _sha256(metrics_path) != metadata["validation_metrics_sha256"]:
        raise ValueError(f"Validation metrics hash mismatch for {run_id}.")
    metrics = _load_json(metrics_path)
    if metrics.get("split") != "val":
        raise ValueError(f"Run {run_id} metrics are not from validation.")
    if metrics.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
        raise ValueError(f"Checkpoint hash mismatch for {run_id}.")

    config_path = REPO_ROOT / metadata["config"]
    config = _load_json(config_path)
    if int(config["seed"]) != int(metadata["seed"]):
        raise ValueError(f"Seed mismatch for {run_id}.")
    return {
        "run_id": run_id,
        "metadata": metadata,
        "metrics": metrics,
        "config": config,
    }


def summarize(run_ids: list[str], expected_seeds: list[int]) -> dict[str, Any]:
    runs = sorted((_load_run(run_id) for run_id in run_ids), key=lambda run: run["metadata"]["seed"])
    seeds = [int(run["metadata"]["seed"]) for run in runs]
    if seeds != sorted(expected_seeds):
        raise ValueError(f"Expected seeds {sorted(expected_seeds)}, got {seeds}.")

    reference = runs[0]
    reference_config = _normalized_config(reference["config"])
    reference_metrics = reference["metrics"]
    reference_metadata = reference["metadata"]
    for run in runs[1:]:
        if _normalized_config(run["config"]) != reference_config:
            raise ValueError("Final configs differ by more than seed.")
        if run["metadata"]["manifest_hash"] != reference_metadata["manifest_hash"]:
            raise ValueError("Manifest hashes differ across final runs.")
        if run["metadata"]["detector_hashes"] != reference_metadata["detector_hashes"]:
            raise ValueError("Detector hashes differ across final runs.")
        if run["metrics"]["detector_sha256"] != reference_metrics["detector_sha256"]:
            raise ValueError("Validation detector hashes differ across evaluations.")
        if run["metrics"]["class_order"] != reference_metrics["class_order"]:
            raise ValueError("Class order differs across evaluations.")
        if run["metrics"]["samples"] != reference_metrics["samples"]:
            raise ValueError("Validation sample counts differ across evaluations.")

    aggregate = {
        metric: _metric_summary([float(run["metrics"][metric]) for run in runs])
        for metric in AGGREGATE_METRICS
    }
    aggregate["neutral_f1"] = _metric_summary(
        [float(run["metrics"]["per_class"]["Neutral"]["f1"]) for run in runs]
    )
    per_class_f1 = {
        class_name: _metric_summary(
            [float(run["metrics"]["per_class"][class_name]["f1"]) for run in runs]
        )
        for class_name in reference_metrics["class_order"]
    }
    return {
        "model": "CAER-Net",
        "track": "upstream_community",
        "protocol": reference["config"]["research"]["protocol"],
        "split": "val",
        "test_accessed": False,
        "seeds": seeds,
        "samples_per_seed": int(reference_metrics["samples"]),
        "manifest_sha256": reference_metadata["manifest_hash"],
        "validation_detector_sha256": reference_metrics["detector_sha256"],
        "runs": [
            {
                "seed": int(run["metadata"]["seed"]),
                "run_id": run["run_id"],
                "git_sha": run["metadata"]["git_sha"],
                "checkpoint_sha256": run["metrics"]["checkpoint_sha256"],
                "best_epoch": int(run["metrics"]["checkpoint_epoch"]),
                **{
                    metric: float(run["metrics"][metric])
                    for metric in AGGREGATE_METRICS
                },
                "neutral_f1": float(run["metrics"]["per_class"]["Neutral"]["f1"]),
            }
            for run in runs
        ],
        "aggregate": aggregate,
        "per_class_f1": per_class_f1,
    }


def _format_metric(summary: dict[str, Any]) -> str:
    return f"{summary['mean']:.6f} +/- {summary['std']:.6f}"


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Experiment 1: Final CAER-Net Results",
        "",
        "## Protocol",
        "",
        f"- Track: `{summary['track']}`",
        f"- Protocol: `{summary['protocol']}`",
        f"- Split: validation only ({summary['samples_per_seed']:,} samples per seed)",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        "- Test accessed: no",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Best epoch | Accuracy | Macro F1 | Weighted F1 | Neutral F1 | NLL | ECE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | {run['accuracy']:.6f} | "
            f"{run['macro_f1']:.6f} | {run['weighted_f1']:.6f} | "
            f"{run['neutral_f1']:.6f} | {run['nll']:.6f} | {run['ece_15']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "Values are mean +/- sample standard deviation across three seeds.",
            "",
            "| Metric | Mean +/- SD |",
            "| --- | ---: |",
        ]
    )
    for metric in (*AGGREGATE_METRICS, "neutral_f1"):
        lines.append(f"| `{metric}` | {_format_metric(summary['aggregate'][metric])} |")

    lines.extend(
        [
            "",
            "## Per-Class F1",
            "",
            "| Class | Mean +/- SD |",
            "| --- | ---: |",
        ]
    )
    for class_name, class_summary in summary["per_class_f1"].items():
        lines.append(f"| {class_name} | {_format_metric(class_summary)} |")
    lines.extend(
        [
            "",
            "Neutral remains the weakest class and the primary target for context-bias diagnostics. "
            "These validation results freeze the upstream-community baseline; they are not test results.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize(args.run_id, args.expected_seeds)
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.output_markdown.expanduser().resolve())
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_markdown": str(args.output_markdown.expanduser().resolve()),
                "accuracy": summary["aggregate"]["accuracy"],
                "macro_f1": summary["aggregate"]["macro_f1"],
                "neutral_f1": summary["aggregate"]["neutral_f1"],
                "test_accessed": summary["test_accessed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
