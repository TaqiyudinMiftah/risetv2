"""Model implementations kept distinct by reproduction track."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .caernet import CAERNet, CAERNetSingleStream
from .notebook_caernet import NotebookCAERNet


def model_args(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a mutable, validated model argument mapping from frozen config."""

    args = model_config.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("Model configuration must contain an object-valued 'args' field.")
    return dict(args)


def model_type(model_config: Mapping[str, Any]) -> str:
    """Return a validated in-repository clean-model type."""

    value = model_config.get("type")
    if value not in {"CAERNet", "CAERNetSingleStream"}:
        raise ValueError(f"Unsupported model type: {value!r}.")
    return str(value)


def required_modalities(model_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact data tensors that a configured model may consume."""

    configured_type = model_type(model_config)
    args = model_args(model_config)
    if configured_type == "CAERNet":
        return ("face", "context")
    modality = args.get("modality")
    if modality not in {"face", "context"}:
        raise ValueError(
            "CAERNetSingleStream requires model.args.modality to be 'face' or 'context'."
        )
    return (str(modality),)


def build_model(model_config: Mapping[str, Any]) -> nn.Module:
    """Instantiate a frozen clean in-repository model configuration."""

    configured_type = model_type(model_config)
    args = model_args(model_config)
    if configured_type == "CAERNet":
        return CAERNet(**args)
    required_modalities(model_config)
    return CAERNetSingleStream(**args)


__all__ = [
    "CAERNet",
    "CAERNetSingleStream",
    "NotebookCAERNet",
    "build_model",
    "model_args",
    "model_type",
    "required_modalities",
]
