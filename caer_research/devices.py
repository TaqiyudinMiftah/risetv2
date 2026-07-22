"""Accelerator discovery shared by CUDA and ROCm training paths."""

from __future__ import annotations

import os
from typing import Any

import torch


MIB = 1024 * 1024


def parse_device_ids(value: str) -> list[int]:
    device_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not device_ids:
        raise ValueError("At least one accelerator index is required.")
    if any(index < 0 for index in device_ids):
        raise ValueError("Accelerator indices must be non-negative.")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("Accelerator indices must be unique.")
    return device_ids


def accelerator_backend() -> str:
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


def configure_visible_devices(value: str) -> None:
    """Set one vendor-appropriate visibility variable before GPU initialization."""
    parse_device_ids(value)
    if accelerator_backend() == "rocm":
        os.environ["ROCR_VISIBLE_DEVICES"] = value
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)


def accelerator_snapshot(value: str, requested_count: int) -> dict[str, Any]:
    requested_ids = parse_device_ids(value)
    if requested_count < 1:
        raise ValueError("requested_count must be at least one.")
    if len(requested_ids) < requested_count:
        raise RuntimeError(
            f"Requested n_gpu={requested_count}, but --device only selects "
            f"{len(requested_ids)} accelerator(s)."
        )
    if not torch.cuda.is_available():
        backend = accelerator_backend()
        raise RuntimeError(
            f"PyTorch accelerator support is unavailable (detected backend: {backend}). "
            "Install a CUDA or ROCm-enabled PyTorch build."
        )
    visible_count = torch.cuda.device_count()
    if visible_count < requested_count:
        raise RuntimeError(
            f"PyTorch sees {visible_count} accelerator(s), expected {requested_count}."
        )

    devices = []
    for logical_index, requested_index in enumerate(requested_ids[:requested_count]):
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
        properties = torch.cuda.get_device_properties(logical_index)
        devices.append(
            {
                "requested_index": requested_index,
                "logical_index": logical_index,
                "name": properties.name,
                "memory_free_mib": int(free_bytes // MIB),
                "memory_total_mib": int(total_bytes // MIB),
            }
        )

    backend = accelerator_backend()
    runtime_version = torch.version.hip if backend == "rocm" else torch.version.cuda
    return {
        "backend": backend,
        "runtime_version": str(runtime_version),
        "torch_version": torch.__version__,
        "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "visible_device_count": visible_count,
        "devices": devices,
    }
