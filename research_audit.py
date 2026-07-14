"""Small, dependency-light helpers for the Phase 0 research audit."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch


CLASS_NAMES = ("Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
LABEL_ALIASES = {"Angry": "Anger"}


class DetectorParseError(ValueError):
    """Raised when one detector annotation cannot be used by the CAER protocol."""


@dataclass(frozen=True)
class DetectorSample:
    image_path: str
    label: str
    label_index: int
    face_bbox: tuple[int, int, int, int]


def canonicalize_label(label: str) -> str:
    """Map dataset aliases to the canonical research class order."""
    canonical = LABEL_ALIASES.get(label.strip(), label.strip())
    if canonical not in LABEL_TO_INDEX:
        raise DetectorParseError(f"Unknown CAER-S label: {label!r}")
    return canonical


def canonicalize_detector_path(image_path: str) -> str:
    """Canonicalize only the class directory while preserving the sample filename."""
    path = PurePosixPath(image_path)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise DetectorParseError(f"Invalid detector image path: {image_path!r}")
    return PurePosixPath(canonicalize_label(path.parts[0]), *path.parts[1:]).as_posix()


def bbox_has_positive_area(bbox: tuple[int, int, int, int] | list[int]) -> bool:
    """Return whether a crop rectangle is structurally usable by ``PIL.Image.crop``."""
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def validate_bbox(
    bbox: tuple[int, int, int, int] | list[int],
    image_size: tuple[int, int] | None = None,
    *,
    require_inside_image: bool = False,
) -> bool:
    """Validate a crop rectangle, optionally requiring all coordinates be in bounds.

    The upstream CAER loader delegates cropping to Pillow and therefore accepts
    rectangles outside image bounds. The strict mode is used only for diagnostics.
    """
    if not bbox_has_positive_area(bbox):
        return False
    if not require_inside_image or image_size is None:
        return True
    x1, y1, x2, y2 = bbox
    width, height = image_size
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def parse_detector_line(line: str, line_number: int | None = None) -> DetectorSample:
    """Parse one upstream ``path,label,x1,y1,x2,y2`` detector annotation."""
    parts = line.rstrip("\n").split(",")
    location = f" at line {line_number}" if line_number is not None else ""
    if len(parts) != 6:
        raise DetectorParseError(f"Expected six comma-separated fields{location}")

    raw_path, raw_label, *raw_bbox = parts
    canonical_path = canonicalize_detector_path(raw_path)
    class_label = canonicalize_label(PurePosixPath(raw_path).parts[0])
    try:
        label_index = int(raw_label)
        bbox = tuple(int(value) for value in raw_bbox)
    except ValueError as error:
        raise DetectorParseError(f"Non-integer label or bbox{location}") from error

    if label_index != LABEL_TO_INDEX[class_label]:
        raise DetectorParseError(
            f"Label index {label_index} disagrees with class {class_label!r}{location}"
        )
    if not bbox_has_positive_area(bbox):
        raise DetectorParseError(f"BBox must have positive area{location}: {bbox}")

    return DetectorSample(
        image_path=canonical_path,
        label=class_label,
        label_index=label_index,
        face_bbox=bbox,
    )


def read_detector_file(path: Path) -> list[DetectorSample]:
    """Read and validate an upstream detector text file."""
    with path.open(encoding="utf-8") as handle:
        return [parse_detector_line(line, line_number) for line_number, line in enumerate(handle, 1)]


def sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest without loading large artifacts in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_determinism(seed: int) -> None:
    """Set deterministic state for an evaluation process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_checkpoint_payload(path: Path, map_location: str | torch.device = "cpu") -> Mapping[str, Any]:
    """Load a PyTorch checkpoint while retaining the legacy upstream metadata."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping, got {type(payload).__name__}")
    return payload


def extract_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract and normalize a checkpoint state dict, including DataParallel keys."""
    state_dict = payload.get("state_dict", payload.get("model_state_dict"))
    if not isinstance(state_dict, Mapping):
        raise KeyError("Checkpoint does not contain state_dict or model_state_dict")
    normalized = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    if not normalized or not all(isinstance(value, torch.Tensor) for value in normalized.values()):
        raise TypeError("Checkpoint state dict must contain tensors")
    return normalized
