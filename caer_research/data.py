"""Manifest-backed CAER-S two-stream dataset and transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, LABEL_TO_INDEX


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    image_path: str
    label: str
    split: str
    face_bbox: tuple[int, int, int, int]

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ManifestSample":
        required = {"sample_id", "image_path", "label", "split", "face_bbox"}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Manifest row is missing fields: {missing}")
        if row["label"] not in CLASS_NAMES:
            raise ValueError(f"Unknown CAER-S label: {row['label']!r}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {row['split']!r}")
        bbox = tuple(int(value) for value in row["face_bbox"])
        if len(bbox) != 4:
            raise ValueError(f"Expected four bbox coordinates, got {bbox}")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid face bbox: {bbox}")
        return cls(
            sample_id=str(row["sample_id"]),
            image_path=str(row["image_path"]),
            label=str(row["label"]),
            split=str(row["split"]),
            face_bbox=bbox,
        )


def load_manifest(manifest_path: Path | str, split: str | None = None) -> list[ManifestSample]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: list[ManifestSample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = ManifestSample.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid manifest row {line_number} in {path}: {error}") from error
            if split is None or sample.split == split:
                rows.append(sample)
    if split is not None and not rows:
        raise ValueError(f"Manifest {path} contains no samples for split={split!r}")
    return rows


def mask_face(image: Image.Image, bbox: Iterable[int], fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    masked = image.copy()
    ImageDraw.Draw(masked).rectangle(tuple(int(value) for value in bbox), fill=fill)
    return masked


def crop_face(image: Image.Image, bbox: Iterable[int]) -> Image.Image:
    return image.crop(tuple(int(value) for value in bbox))


def build_transforms(train: bool) -> tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    face_transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    context_crop: transforms.RandomCrop | transforms.CenterCrop
    context_crop = transforms.RandomCrop(112) if train else transforms.CenterCrop(112)
    context_transform = transforms.Compose(
        [
            transforms.Resize((128, 171)),
            context_crop,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return face_transform, context_transform


def normalize_modalities(modalities: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return a canonical non-empty subset of the supported input modalities."""

    supported = ("face", "context")
    if modalities is None:
        return supported
    requested = {str(modality) for modality in modalities}
    if not requested:
        raise ValueError("At least one input modality is required.")
    unknown = sorted(requested - set(supported))
    if unknown:
        raise ValueError(f"Unsupported input modalities: {unknown!r}.")
    return tuple(modality for modality in supported if modality in requested)


class CAERSTwoStreamDataset(Dataset[dict[str, Any]]):
    """Load only the requested face crop and/or face-masked context tensors."""

    def __init__(
        self,
        manifest_path: Path | str,
        dataset_root: Path | str,
        split: str,
        face_transform: Callable[[Image.Image], Any] | None = None,
        context_transform: Callable[[Image.Image], Any] | None = None,
        modalities: Iterable[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.face_transform = face_transform
        self.context_transform = context_transform
        self.modalities = normalize_modalities(modalities)
        self.samples = load_manifest(self.manifest_path, split=split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self.dataset_root / sample.image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        item: dict[str, Any] = {
            "label": torch.tensor(LABEL_TO_INDEX[sample.label], dtype=torch.long),
            "label_name": sample.label,
            "sample_id": sample.sample_id,
            "image_path": sample.image_path,
            "bbox": torch.tensor(sample.face_bbox, dtype=torch.float32),
        }
        if "face" in self.modalities:
            face = crop_face(image, sample.face_bbox)
            if self.face_transform is not None:
                face = self.face_transform(face)
            item["face"] = face
        if "context" in self.modalities:
            context = mask_face(image, sample.face_bbox)
            if self.context_transform is not None:
                context = self.context_transform(context)
            item["context"] = context
        return item
