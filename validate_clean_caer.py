#!/usr/bin/env python3
"""Verify clean in-repo models against upstream and notebook implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from caer_research.checkpointing import (
    extract_model_state,
    load_checkpoint_payload,
    load_model_checkpoint,
)
from caer_research.data import CAERSTwoStreamDataset, build_transforms
from caer_research.models import CAERNet, NotebookCAERNet


REPO_ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "CAER" / "CAER"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "protocols" / "caer_s_content_disjoint_v1" / "manifest.jsonl"
DEFAULT_UPSTREAM_CHECKPOINT = (
    UPSTREAM_ROOT
    / "official_runs"
    / "models"
    / "CAERNet_UpstreamCommunity_ContentDisjoint_Final"
    / "caernet__upstream_community__seed42__20260714_224828"
    / "model_best.pth"
)
DEFAULT_NOTEBOOK_CHECKPOINT = REPO_ROOT / "checkpoints" / "best_caernet_s.pt"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "refactor" / "clean_caer_parity.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_model() -> nn.Module:
    upstream_path = str(UPSTREAM_ROOT)
    added = upstream_path not in sys.path
    if added:
        sys.path.insert(0, upstream_path)
    try:
        from model.model import CAERSNet

        return CAERSNet()
    finally:
        if added:
            sys.path.remove(upstream_path)


def load_notebook_model(notebook_path: Path) -> nn.Module:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "class CAERNetS" in "".join(cell.get("source", []))
    )
    namespace: dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
    exec(compile(source, str(notebook_path), "exec"), namespace)
    return namespace["CAERNetS"](num_classes=7, dropout=0.5)


def compare_outputs(reference: Any, clean: Any) -> dict[str, Any]:
    reference_logits = reference if isinstance(reference, torch.Tensor) else reference["logits"]
    clean_logits = clean if isinstance(clean, torch.Tensor) else clean["logits"]
    torch.testing.assert_close(clean_logits, reference_logits, rtol=0.0, atol=0.0)
    result = {
        "logits_shape": list(clean_logits.shape),
        "max_abs_logit_difference": float((clean_logits - reference_logits).abs().max().item()),
        "predictions_equal": bool(torch.equal(clean_logits.argmax(1), reference_logits.argmax(1))),
    }
    if isinstance(reference, dict) and isinstance(clean, dict):
        torch.testing.assert_close(
            clean["fusion_weights"], reference["fusion_weights"], rtol=0.0, atol=0.0
        )
        result["max_abs_weight_difference"] = float(
            (clean["fusion_weights"] - reference["fusion_weights"]).abs().max().item()
        )
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "CAER-S")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--upstream-checkpoint", type=Path, default=DEFAULT_UPSTREAM_CHECKPOINT)
    parser.add_argument("--notebook-checkpoint", type=Path, default=DEFAULT_NOTEBOOK_CHECKPOINT)
    parser.add_argument("--notebook", type=Path, default=REPO_ROOT / "CAER_S_CAERNet_Reproduction_ipynb.ipynb")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    face_transform, context_transform = build_transforms(train=False)
    dataset = CAERSTwoStreamDataset(
        args.manifest,
        args.dataset_root,
        split=args.split,
        face_transform=face_transform,
        context_transform=context_transform,
    )
    loader = DataLoader(dataset, batch_size=args.samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    face = batch["face"].to(device)
    context = batch["context"].to(device)

    upstream_payload = load_checkpoint_payload(
        args.upstream_checkpoint,
        map_location="cpu",
        module_search_path=UPSTREAM_ROOT,
    )
    upstream_state = extract_model_state(upstream_payload)
    upstream_model = load_upstream_model().to(device).eval()
    upstream_model.load_state_dict(upstream_state)
    clean_upstream_model = CAERNet().to(device).eval()
    load_model_checkpoint(
        clean_upstream_model,
        args.upstream_checkpoint,
        map_location=device,
        module_search_path=UPSTREAM_ROOT,
    )

    notebook_reference = load_notebook_model(args.notebook).to(device).eval()
    notebook_payload = torch.load(args.notebook_checkpoint, map_location="cpu", weights_only=False)
    notebook_reference.load_state_dict(extract_model_state(notebook_payload))
    clean_notebook_model = NotebookCAERNet().to(device).eval()
    load_model_checkpoint(clean_notebook_model, args.notebook_checkpoint, map_location=device)

    with torch.inference_mode():
        upstream_result = compare_outputs(
            upstream_model(face, context),
            clean_upstream_model(face, context),
        )
        notebook_result = compare_outputs(
            notebook_reference(face, context),
            clean_notebook_model(face, context),
        )

    result = {
        "status": "passed",
        "split": args.split,
        "test_accessed": False,
        "samples": args.samples,
        "sample_ids": list(batch["sample_id"]),
        "face_shape": list(face.shape),
        "context_shape": list(context.shape),
        "manifest_sha256": sha256(args.manifest),
        "upstream": {
            "checkpoint_sha256": sha256(args.upstream_checkpoint),
            "parameters": parameter_count(clean_upstream_model),
            **upstream_result,
        },
        "notebook": {
            "checkpoint_sha256": sha256(args.notebook_checkpoint),
            "parameters": parameter_count(clean_notebook_model),
            **notebook_result,
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **result}, indent=2))


if __name__ == "__main__":
    main()
