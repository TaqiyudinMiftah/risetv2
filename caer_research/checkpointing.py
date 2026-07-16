"""Checkpoint compatibility and deterministic training-state helpers."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def strip_data_parallel_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            return strip_data_parallel_prefix(state_dict)
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return strip_data_parallel_prefix(checkpoint)  # type: ignore[arg-type]
    raise ValueError("Checkpoint does not contain a recognized model state dict.")


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    module_search_path: Path | str | None = None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint_payload(
        checkpoint_path,
        map_location=map_location,
        module_search_path=module_search_path,
    )
    unwrap_model(model).load_state_dict(extract_model_state(checkpoint), strict=strict)
    return checkpoint


def load_checkpoint_payload(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
    module_search_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    search_path = str(Path(module_search_path).resolve()) if module_search_path else None
    added_to_path = search_path is not None and search_path not in sys.path
    if added_to_path:
        sys.path.insert(0, search_path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    finally:
        if added_to_path:
            sys.path.remove(search_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(checkpoint).__name__}")
    return checkpoint


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
    scaler: Any = None,
    history: list[dict[str, Any]] | None = None,
    early_stopping_count: int = 0,
) -> dict[str, Any]:
    return {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "early_stopping_count": early_stopping_count,
        "history": history or [],
        "config": dict(config),
        "rng_state": capture_rng_state(),
    }


def save_checkpoint(payload: Mapping[str, Any], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary_path)
    temporary_path.replace(path)
