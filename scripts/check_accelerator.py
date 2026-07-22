#!/usr/bin/env python3
"""Validate the installed PyTorch accelerator backend with a small compute test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caer_research.devices import accelerator_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-backend", choices=("cuda", "rocm"))
    parser.add_argument("--min-devices", type=int, default=1)
    args = parser.parse_args()

    backend = accelerator_backend()
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.require_backend and backend != args.require_backend:
        raise RuntimeError(f"Expected {args.require_backend}, detected {backend}.")
    if device_count < args.min_devices:
        raise RuntimeError(f"Expected at least {args.min_devices} accelerator(s), found {device_count}.")

    devices = []
    for index in range(device_count):
        left = torch.ones((32, 32), device=f"cuda:{index}")
        right = torch.ones((32, 32), device=f"cuda:{index}")
        checksum = float((left @ right).sum().item())
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "memory_total_mib": int(properties.total_memory // (1024 * 1024)),
                "matmul_checksum": checksum,
            }
        )

    print(
        json.dumps(
            {
                "backend": backend,
                "torch_version": torch.__version__,
                "runtime_version": (
                    str(torch.version.hip) if backend == "rocm" else str(torch.version.cuda)
                ),
                "device_count": device_count,
                "devices": devices,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
