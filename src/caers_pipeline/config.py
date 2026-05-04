from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    dataset_root: Path
    image_extensions: tuple[str, ...]
    create_val_split: bool
    val_ratio: float
    image_size: int


@dataclass
class OutputConfig:
    manifest_path: Path
    diagnostics_path: Path


@dataclass
class ModelConfig:
    backbone: str
    pretrained: bool
    dropout: float


@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    device: str
    seed: int
    save_dir: Path
    stream_mode: str  # "multimodal", "face", "context"


@dataclass
class AppConfig:
    seed: int
    dataset: DatasetConfig
    outputs: OutputConfig
    model: ModelConfig
    train: TrainConfig


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("image_extensions must be a list of strings")
    ext = tuple(str(x).lower() for x in value)
    if len(ext) == 0:
        raise ValueError("image_extensions cannot be empty")
    return ext


def load_config(config_path: str | Path) -> AppConfig:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    seed = int(raw.get("seed", 42))

    dataset_raw = raw.get("dataset", {})
    outputs_raw = raw.get("outputs", {})
    model_raw = raw.get("model", {})
    train_raw = raw.get("train", {})

    if not isinstance(dataset_raw, dict) or not isinstance(outputs_raw, dict):
        raise ValueError("dataset and outputs config sections must be mappings")
    if not isinstance(model_raw, dict) or not isinstance(train_raw, dict):
        raise ValueError("model and train config sections must be mappings")

    dataset_cfg = DatasetConfig(
        dataset_root=Path(str(dataset_raw.get("dataset_root", ""))).expanduser(),
        image_extensions=_as_tuple(dataset_raw.get("image_extensions", [])),
        create_val_split=bool(dataset_raw.get("create_val_split", True)),
        val_ratio=float(dataset_raw.get("val_ratio", 0.1)),
        image_size=int(dataset_raw.get("image_size", 224)),
    )

    output_cfg = OutputConfig(
        manifest_path=Path(str(outputs_raw.get("manifest_path", "artifacts/manifest.jsonl"))),
        diagnostics_path=Path(str(outputs_raw.get("diagnostics_path", "artifacts/diagnostics.json"))),
    )

    model_cfg = ModelConfig(
        backbone=str(model_raw.get("backbone", "resnet18")),
        pretrained=bool(model_raw.get("pretrained", True)),
        dropout=float(model_raw.get("dropout", 0.5)),
    )

    train_cfg = TrainConfig(
        batch_size=int(train_raw.get("batch_size", 32)),
        num_epochs=int(train_raw.get("num_epochs", 30)),
        lr=float(train_raw.get("lr", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1e-4)),
        num_workers=int(train_raw.get("num_workers", 4)),
        device=str(train_raw.get("device", "cuda")),
        seed=seed,
        save_dir=Path(str(train_raw.get("save_dir", "checkpoints/caers"))),
        stream_mode=str(train_raw.get("stream_mode", "multimodal")),
    )

    if not 0.0 < dataset_cfg.val_ratio < 0.5:
        raise ValueError("val_ratio must be in (0.0, 0.5)")
    if train_cfg.stream_mode not in ("multimodal", "face", "context"):
        raise ValueError("stream_mode must be 'multimodal', 'face', or 'context'")

    return AppConfig(seed=seed, dataset=dataset_cfg, outputs=output_cfg, model=model_cfg, train=train_cfg)
