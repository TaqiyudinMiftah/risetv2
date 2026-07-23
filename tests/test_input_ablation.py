from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import caer_research.data as data_module
import run_caer_clean
from caer_research.data import CAERSTwoStreamDataset
from caer_research.models import CAERNetSingleStream, build_model, required_modalities
from run_caer_clean import make_run_id, update_registry, validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CONFIG = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "caernet_clean_input_ablation_face_only_content_disjoint_exploratory_seed42.json"
)
CONTROL_CONFIG = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "caernet_clean_content_disjoint_exploratory_seed42.json"
)
CONTEXT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "caernet_clean_input_ablation_context_only_content_disjoint_exploratory_seed42.json"
)


class InputAblationTests(unittest.TestCase):
    def test_face_only_is_strictly_invariant_to_context_and_has_expected_size(self) -> None:
        torch.manual_seed(7)
        model = CAERNetSingleStream(modality="face").eval()
        face = torch.randn(2, 3, 96, 96)
        context_a = torch.randn(2, 3, 112, 112)
        context_b = torch.randn(2, 3, 112, 112)

        with torch.inference_mode():
            logits_a = model(face, context_a)
            logits_b = model(face, context_b)
            logits_none = model(face, None)

        self.assertEqual(tuple(logits_a.shape), (2, 7))
        torch.testing.assert_close(logits_a, logits_b, rtol=0.0, atol=0.0)
        torch.testing.assert_close(logits_a, logits_none, rtol=0.0, atol=0.0)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1_014_279)
        self.assertNotIn("attention_inference_module", dict(model.named_modules()))
        with self.assertRaisesRegex(ValueError, "face tensor"):
            model(None, context_a)

    def test_context_only_is_strictly_invariant_to_face_and_has_expected_size(self) -> None:
        torch.manual_seed(8)
        model = CAERNetSingleStream(modality="context").eval()
        face_a = torch.randn(2, 3, 96, 96)
        face_b = torch.randn(2, 3, 96, 96)
        context = torch.randn(2, 3, 112, 112)

        with torch.inference_mode():
            logits_a = model(face_a, context)
            logits_b = model(face_b, context)
            logits_none = model(None, context)

        self.assertEqual(tuple(logits_a.shape), (2, 7))
        torch.testing.assert_close(logits_a, logits_b, rtol=0.0, atol=0.0)
        torch.testing.assert_close(logits_a, logits_none, rtol=0.0, atol=0.0)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1_310_730)
        self.assertIsNotNone(model.attention_inference_module)
        with self.assertRaisesRegex(ValueError, "context tensor"):
            model(face_a, None)

    def test_dataset_constructs_only_the_requested_modality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "train" / "Angry" / "sample.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(image_path)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "train__sample",
                        "image_path": "train/Angry/sample.png",
                        "label": "Anger",
                        "split": "train",
                        "face_bbox": [5, 5, 15, 15],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(data_module, "mask_face", wraps=data_module.mask_face) as mask_face:
                face_item = CAERSTwoStreamDataset(
                    manifest_path, root, split="train", modalities=("face",)
                )[0]
            with patch.object(data_module, "crop_face", wraps=data_module.crop_face) as crop_face:
                context_item = CAERSTwoStreamDataset(
                    manifest_path, root, split="train", modalities=("context",)
                )[0]

        self.assertIn("face", face_item)
        self.assertNotIn("context", face_item)
        mask_face.assert_not_called()
        self.assertIn("context", context_item)
        self.assertNotIn("face", context_item)
        crop_face.assert_not_called()

    def test_frozen_ablation_configs_use_matching_strict_model_and_data_modalities(self) -> None:
        for path, modality in ((FACE_CONFIG, "face"), (CONTEXT_CONFIG, "context")):
            config = json.loads(path.read_text(encoding="utf-8"))

            validate_config(config, expected_seed=42)

            self.assertEqual(config["research"]["experiment"], "input_ablation")
            self.assertEqual(config["model"]["type"], "CAERNetSingleStream")
            self.assertEqual(config["model"]["args"], {"modality": modality})
            self.assertEqual(config["data"]["modalities"], [modality])
            self.assertEqual(required_modalities(config["model"]), (modality,))
            self.assertIsInstance(build_model(config["model"]), CAERNetSingleStream)
            self.assertEqual(config["n_gpu"], 1)
            self.assertFalse(config["research"]["test_during_training"])

    def test_launcher_rejects_legacy_boolean_modality_flags_as_an_ablation(self) -> None:
        config = json.loads(FACE_CONFIG.read_text(encoding="utf-8"))
        config["research"] = {
            "stage": "exploratory",
            "track": "clean_inrepo",
            "protocol": "caer_s_content_disjoint_v1",
            "selection_metric": "val_macro_f1",
            "test_during_training": False,
        }
        config["model"] = {
            "type": "CAERNet",
            "args": {"use_face": True, "use_context": False, "concat": False},
        }
        config["data"]["modalities"] = ["face", "context"]

        with self.assertRaisesRegex(ValueError, "exact frozen face\\+context"):
            validate_config(config, expected_seed=42)

    def test_launcher_rejects_non_control_two_stream_fusion_arguments(self) -> None:
        config = json.loads(CONTROL_CONFIG.read_text(encoding="utf-8"))
        config["model"]["args"]["concat"] = True

        with self.assertRaisesRegex(ValueError, "exact frozen face\\+context"):
            validate_config(config, expected_seed=42)

    def test_variant_run_id_and_registry_preserve_ablation_identity_with_blank_test_columns(self) -> None:
        run_id = make_run_id(42, "input_ablation_face_only_exploratory")
        self.assertRegex(
            run_id,
            r"^caernet__input_ablation_face_only_exploratory__seed42__\d{8}_\d{6}$",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry_path = Path(temporary_directory) / "registry.csv"
            fields = (
                "run_id,status,model,variant,seed,git_sha,config,checkpoint,"
                "val_accuracy,val_macro_f1,test_accuracy,test_macro_f1,neutral_f1,"
                "params,latency_ms,notes\n"
            )
            registry_path.write_text(fields, encoding="utf-8")
            metadata = {
                "run_id": run_id,
                "status": "prepared",
                "model_type": "CAERNetSingleStream",
                "variant": "input_ablation_face_only",
                "seed": 42,
                "git_sha": "fixture-sha",
                "config": "configs/face.json",
                "notes": "test split is not loaded or evaluated.",
            }
            with patch.object(run_caer_clean, "REGISTRY_PATH", registry_path):
                update_registry(metadata)
            with registry_path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["model"], "CAER-Net single-stream")
        self.assertEqual(row["variant"], "input_ablation_face_only")
        self.assertEqual(row["test_accuracy"], "")
        self.assertEqual(row["test_macro_f1"], "")

    def test_dry_run_leaves_no_metadata_or_checkpoint_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = json.loads(FACE_CONFIG.read_text(encoding="utf-8"))
            dataset_root = root / "CAER-S"
            manifest_path = root / "artifacts" / "protocols" / "manifest.jsonl"
            config_path = root / "configs" / "face.json"
            dataset_root.mkdir()
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("", encoding="utf-8")
            config["data"]["dataset_root"] = "CAER-S"
            config["data"]["manifest"] = "artifacts/protocols/manifest.jsonl"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run_id = "caernet__input_ablation_face_only_exploratory__seed42__fixture"
            args = Namespace(
                config=config_path,
                seed=42,
                run_id=run_id,
                resume=None,
                device="0",
                n_gpu=None,
                wandb_mode="disabled",
                wandb_project="unused",
                wandb_entity=None,
                min_free_gpu_mib=0,
                dry_run=True,
                smoke_only=False,
            )
            with (
                patch.object(run_caer_clean, "REPO_ROOT", root),
                patch.object(run_caer_clean, "CHECKPOINT_ROOT", root / "checkpoints"),
                patch.object(run_caer_clean, "METADATA_ROOT", root / "artifacts" / "experiments"),
                patch.object(run_caer_clean, "git_sha", return_value="fixture-sha"),
                patch.object(run_caer_clean, "git_dirty", return_value=False),
            ):
                run_caer_clean.train(args)

            self.assertFalse((root / "checkpoints" / run_id).exists())
            self.assertFalse((root / "artifacts" / "experiments" / run_id).exists())


if __name__ == "__main__":
    unittest.main()
