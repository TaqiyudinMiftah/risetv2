from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

import verify_clean_validation as verifier
from caer_research.models import CAERNetSingleStream


class ValidationOnlyVerifierTests(unittest.TestCase):
    def test_compare_metrics_reports_exact_nested_reproduction(self) -> None:
        saved = {
            "accuracy": 0.75,
            "samples": 4,
            "per_class": {"Neutral": {"f1": 0.5, "support": 2}},
        }

        comparison = verifier.compare_metrics(saved, saved)

        self.assertEqual(comparison["metric_max_abs_delta"], 0.0)
        self.assertEqual(comparison["metric_deltas"]["per_class.Neutral.f1"], 0.0)

    def test_compare_metrics_rejects_schema_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "schemas differ"):
            verifier.compare_metrics({"accuracy": 0.75}, {"macro_f1": 0.75})

    def test_load_and_compare_predictions_detects_row_and_confidence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            prediction_path = Path(temporary_directory) / "val_predictions.csv"
            with prediction_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=verifier.PREDICTION_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": "validation/sample-a.png",
                        "label": 1,
                        "prediction": 1,
                        "confidence": 0.9,
                        "correct": 1,
                    }
                )
                writer.writerow(
                    {
                        "image_path": "validation/sample-b.png",
                        "label": 2,
                        "prediction": 2,
                        "confidence": 0.8,
                        "correct": 1,
                    }
                )

            saved_rows = verifier.load_saved_predictions(prediction_path)
            comparison = verifier.compare_predictions(
                saved_rows,
                ["validation/sample-a.png", "validation/sample-b.png"],
                [1, 2],
                [1, 0],
                [0.9, 0.7],
            )

        self.assertEqual(comparison["saved_prediction_count"], 2)
        self.assertEqual(comparison["reproduced_prediction_count"], 2)
        self.assertEqual(comparison["prediction_mismatches"], 1)
        self.assertAlmostEqual(comparison["max_confidence_abs_delta"], 0.1)

    def test_validation_only_guard_rejects_non_validation_samples(self) -> None:
        validation_dataset = SimpleNamespace(
            split="val",
            samples=[SimpleNamespace(sample_id="validation-a", split="val")],
        )
        verifier.assert_logical_validation_only(validation_dataset)

        non_validation_dataset = SimpleNamespace(
            split="val",
            samples=[SimpleNamespace(sample_id="unexpected", split="train")],
        )
        with self.assertRaisesRegex(ValueError, "non-validation"):
            verifier.assert_logical_validation_only(non_validation_dataset)

    def test_run_id_cannot_escape_fixed_artifact_directories(self) -> None:
        with self.assertRaisesRegex(ValueError, "single non-empty"):
            verifier.run_paths("../other")

    def test_completed_run_validation_requires_locked_test_selection(self) -> None:
        config = {
            "research": {
                "track": "clean_inrepo",
                "test_during_training": False,
            }
        }
        verifier.validate_completed_run(
            {"status": "completed", "test_used_for_selection": False},
            config,
        )

        with self.assertRaisesRegex(ValueError, "test split stayed out"):
            verifier.validate_completed_run(
                {"status": "completed", "test_used_for_selection": True},
                config,
            )

    def test_validation_reproduction_supports_a_strict_single_stream_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "CAER-S" / "val" / "Anger" / "sample.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (160, 128), color=(128, 128, 128)).save(image_path)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "logical-val-sample",
                        "image_path": "val/Anger/sample.png",
                        "label": "Anger",
                        "split": "val",
                        "face_bbox": [20, 20, 100, 100],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            checkpoint_path = root / "single-stream.pt"
            source_model = CAERNetSingleStream(modality="face")
            torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)
            config = {
                "data": {
                    "dataset_root": "CAER-S",
                    "manifest": "manifest.jsonl",
                    "modalities": ["face"],
                    "batch_size": 1,
                },
                "model": {"type": "CAERNetSingleStream", "args": {"modality": "face"}},
            }

            result, dataset = verifier.reproduce_validation(
                config,
                checkpoint_path,
                root,
                torch.device("cpu"),
            )

        self.assertEqual(dataset.split, "val")
        self.assertEqual(dataset.modalities, ("face",))
        self.assertEqual(len(dataset), 1)
        self.assertEqual(result["samples"], 1)

    def test_validation_reproduction_rejects_mismatched_frozen_data_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.jsonl"
            dataset_root = root / "CAER-S"
            checkpoint_path = root / "single-stream.pt"
            manifest_path.write_text("", encoding="utf-8")
            dataset_root.mkdir()
            torch.save(
                {"model_state_dict": CAERNetSingleStream(modality="face").state_dict()},
                checkpoint_path,
            )
            config = {
                "data": {
                    "dataset_root": "CAER-S",
                    "manifest": "manifest.jsonl",
                    "modalities": ["context"],
                    "batch_size": 1,
                },
                "model": {"type": "CAERNetSingleStream", "args": {"modality": "face"}},
            }

            with self.assertRaisesRegex(ValueError, "modalities do not match"):
                verifier.reproduce_validation(config, checkpoint_path, root, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
