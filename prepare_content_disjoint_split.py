#!/usr/bin/env python3
"""Build a versioned CAER-S split with byte-identical images kept in one split."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from research_audit import (
    LABEL_TO_INDEX,
    canonicalize_detector_path,
    canonicalize_label,
    read_detector_file,
    sha256_file,
    validate_bbox,
)


REPO_ROOT = Path(__file__).resolve().parent
SPLIT_PRIORITY = ("train", "val", "test")


@dataclass(frozen=True)
class SourceSample:
    split: str
    row: dict[str, Any]
    detector_line: str
    content_sha256: str

    @property
    def image_path(self) -> str:
        return str(self.row["image_path"])

    @property
    def label(self) -> str:
        return canonicalize_label(str(self.row["label"]))


def _manifest_entry(row: dict[str, Any]) -> tuple[str, int, tuple[int, int, int, int]]:
    image_path = PurePosixPath(str(row["image_path"]))
    if len(image_path.parts) < 3:
        raise ValueError(f"Manifest image path is incomplete: {row['image_path']!r}")
    relative_path = canonicalize_detector_path(PurePosixPath(*image_path.parts[1:]).as_posix())
    label = canonicalize_label(str(row["label"]))
    bbox = tuple(int(value) for value in row["face_bbox"])
    if not validate_bbox(bbox):
        raise ValueError(f"Manifest bbox has no positive area: {bbox}")
    return relative_path, LABEL_TO_INDEX[label], bbox


def _image_path(row: dict[str, Any], dataset_root: Path) -> Path:
    split = str(row["split"])
    disk_split = "train" if split == "train" else "test"
    image_path = PurePosixPath(str(row["image_path"]))
    return dataset_root / disk_split / image_path.parts[-2] / image_path.name


def _read_sources(manifest_path: Path, dataset_root: Path, detector_dir: Path) -> dict[str, list[SourceSample]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_PRIORITY}
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            split = row.get("split")
            if split not in rows_by_split:
                raise ValueError(f"Unknown split {split!r} at manifest line {line_number}")
            rows_by_split[split].append(row)

    records: dict[str, list[tuple[dict[str, Any], str, Path]]] = {split: [] for split in SPLIT_PRIORITY}
    for split in SPLIT_PRIORITY:
        detector_path = detector_dir / f"{split}.txt"
        detector_lines = detector_path.read_text(encoding="utf-8").splitlines()
        detector_samples = read_detector_file(detector_path)
        manifest_rows = rows_by_split[split]
        if len(detector_samples) != len(manifest_rows):
            raise ValueError(f"{split} detector and manifest sample counts differ")
        for index, (row, detector_line, detector_sample) in enumerate(
            zip(manifest_rows, detector_lines, detector_samples), 1
        ):
            if _manifest_entry(row) != (
                detector_sample.image_path,
                detector_sample.label_index,
                detector_sample.face_bbox,
            ):
                raise ValueError(f"{split} detector and manifest differ at sample {index}")
            image_path = _image_path(row, dataset_root)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing source image: {image_path}")
            records[split].append((row, detector_line, image_path))

    all_records = [record for split in SPLIT_PRIORITY for record in records[split]]
    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(sha256_file, [image_path for _, _, image_path in all_records]))

    sources: dict[str, list[SourceSample]] = {split: [] for split in SPLIT_PRIORITY}
    for (row, detector_line, _), digest in zip(all_records, hashes):
        sources[str(row["split"])].append(
            SourceSample(
                split=str(row["split"]),
                row=row,
                detector_line=detector_line,
                content_sha256=digest,
            )
        )
    return sources


def build_content_disjoint_protocol(
    manifest_path: Path,
    dataset_root: Path,
    detector_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Retain the first sample in train -> validation -> test priority order.

    Source detector files and the source manifest remain untouched. Bboxes and
    class labels are copied exactly for retained samples.
    """
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    sources = _read_sources(manifest_path, dataset_root, detector_dir)

    retained: dict[str, list[SourceSample]] = {split: [] for split in SPLIT_PRIORITY}
    removed: list[dict[str, str]] = []
    first_by_hash: dict[str, SourceSample] = {}
    for split in SPLIT_PRIORITY:
        for sample in sources[split]:
            original = first_by_hash.get(sample.content_sha256)
            if original is None:
                first_by_hash[sample.content_sha256] = sample
                retained[split].append(sample)
                continue
            if sample.label != original.label:
                raise ValueError(
                    "Identical image content has conflicting labels: "
                    f"{original.image_path} ({original.label}) and {sample.image_path} ({sample.label})"
                )
            reason = f"duplicate_within_{split}" if original.split == split else f"duplicate_with_{original.split}"
            removed.append(
                {
                    "source_split": split,
                    "sample_id": str(sample.row["sample_id"]),
                    "image_path": sample.image_path,
                    "label": sample.label,
                    "content_sha256": sample.content_sha256,
                    "retained_split": original.split,
                    "retained_sample_id": str(original.row["sample_id"]),
                    "retained_image_path": original.image_path,
                    "reason": reason,
                }
            )

    output_dir.mkdir(parents=True)
    for split in SPLIT_PRIORITY:
        (output_dir / f"{split}.txt").write_text(
            "\n".join(sample.detector_line for sample in retained[split]) + "\n",
            encoding="utf-8",
        )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for split in SPLIT_PRIORITY:
            for sample in retained[split]:
                handle.write(json.dumps(sample.row, separators=(",", ":")) + "\n")
    with (output_dir / "removed_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source_split",
            "sample_id",
            "image_path",
            "label",
            "content_sha256",
            "retained_split",
            "retained_sample_id",
            "retained_image_path",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(removed)

    input_hashes = {
        "manifest": sha256_file(manifest_path),
        **{split: sha256_file(detector_dir / f"{split}.txt") for split in SPLIT_PRIORITY},
    }
    output_hashes = {
        "manifest": sha256_file(output_dir / "manifest.jsonl"),
        **{split: sha256_file(output_dir / f"{split}.txt") for split in SPLIT_PRIORITY},
        "removed_samples": sha256_file(output_dir / "removed_samples.csv"),
    }
    report = {
        "protocol_name": "caer_s_content_disjoint_v1",
        "selection_policy": "retain first exact-content occurrence in train -> val -> test source order",
        "source_manifest": str(manifest_path.resolve()),
        "source_detector_dir": str(detector_dir.resolve()),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "source_counts": {split: len(sources[split]) for split in SPLIT_PRIORITY},
        "retained_counts": {split: len(retained[split]) for split in SPLIT_PRIORITY},
        "removed_counts": {
            split: sum(1 for sample in removed if sample["source_split"] == split)
            for split in SPLIT_PRIORITY
        },
        "unique_content_hashes": len(first_by_hash),
        "content_hashes_disjoint_across_splits": True,
        "bbox_policy": "Preserve raw detector boxes; upstream Pillow crop pads out-of-image coordinates.",
    }
    (output_dir / "protocol.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "caers_manifest.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "CAER-S")
    parser.add_argument("--detector-dir", type=Path, default=REPO_ROOT / "detectors")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "protocols" / "caer_s_content_disjoint_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_content_disjoint_protocol(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        detector_dir=args.detector_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
