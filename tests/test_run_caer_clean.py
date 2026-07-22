from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

import run_caer_clean
from caer_research.checkpointing import training_checkpoint
from run_caer_clean import (
    clean_run_notes,
    completed_run_provenance,
    make_run_id,
    mark_interrupted,
    public_evaluation_metrics,
    reconcile_completed,
    sha256,
    validate_config,
    validate_resume_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "caernet_clean_content_disjoint_exploratory_seed42.json"


class CleanTrainingLauncherTests(unittest.TestCase):
    def _create_resume_fixture(self, root: Path) -> tuple[Path, Path, dict[str, object], Path]:
        run_id = "caernet__clean_inrepo__seed42__fixture"
        config_path = root / "configs" / "experiment.json"
        manifest_path = root / "artifacts" / "protocols" / "manifest.jsonl"
        output_dir = root / "checkpoints" / run_id
        metadata_dir = root / "artifacts" / "experiments" / run_id
        config: dict[str, object] = {
            "seed": 42,
            "n_gpu": 1,
            "research": {"protocol": "caer_s_content_disjoint_v1"},
        }
        config_path.parent.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest_path.write_text('{"split": "train"}\n', encoding="utf-8")
        (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        checkpoint = training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
            epoch=1,
            best_metric=0.5,
            early_stopping_count=0,
            history=[{"epoch": 1}],
            config=config,
        )
        checkpoint["train_generator_state"] = torch.Generator().manual_seed(42).get_state()
        checkpoint_path = output_dir / "last.pt"
        torch.save(checkpoint, checkpoint_path)
        metadata = {
            "run_id": run_id,
            "status": "interrupted",
            "seed": 42,
            "protocol": "caer_s_content_disjoint_v1",
            "git_sha": "fixture-git-sha",
            "config": str(config_path.relative_to(root)),
            "config_sha256": sha256(config_path),
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256(manifest_path),
            "test_used_for_selection": False,
            "notes": "fixture run; test split is not loaded or evaluated.",
        }
        (metadata_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return checkpoint_path, config_path, config, manifest_path

    def _create_completed_fixture(
        self,
        root: Path,
    ) -> tuple[str, Path, Path, list[dict[str, float]]]:
        run_id = "caernet__clean_inrepo__seed42__completed_fixture"
        config_path = root / "configs" / "experiment.json"
        manifest_path = root / "artifacts" / "protocols" / "fixture" / "manifest.jsonl"
        output_dir = root / "checkpoints" / run_id
        metadata_dir = root / "artifacts" / "experiments" / run_id
        config = {"seed": 42, "n_gpu": 1, "research": {"protocol": "caer_s_content_disjoint_v1"}}
        history: list[dict[str, float]] = [
            {"epoch": 1.0, "val_macro_f1": 0.5},
            {"epoch": 2.0, "val_macro_f1": 0.75},
        ]
        config_path.parent.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest_path.write_text('{"split": "train"}\n', encoding="utf-8")
        (manifest_path.parent / "train.txt").write_text("train detector\n", encoding="utf-8")
        (manifest_path.parent / "val.txt").write_text("validation detector\n", encoding="utf-8")
        # No test.txt is created: completed-run provenance must remain test-locked.
        (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (output_dir / "best.pt").write_bytes(b"best checkpoint")
        (output_dir / "last.pt").write_bytes(b"last checkpoint")
        (output_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
        metrics = {
            "accuracy": 0.8,
            "macro_f1": 0.75,
            "per_class": {"Neutral": {"f1": 0.7}},
        }
        (output_dir / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        metadata = {
            "run_id": run_id,
            "status": "completed",
            "seed": 42,
            "track": "clean_inrepo",
            "protocol": "caer_s_content_disjoint_v1",
            "git_sha": "original-run-sha",
            "config": str(config_path.relative_to(root)),
            "config_sha256": sha256(config_path),
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256(manifest_path),
            "test_used_for_selection": False,
            "params": 123,
            "notes": "completed fixture; test split is not loaded or evaluated.",
        }
        (metadata_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return run_id, output_dir, manifest_path, history

    def test_exploratory_config_uses_guarded_clean_protocol(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        validate_config(config, expected_seed=42)

        self.assertEqual(config["research"]["track"], "clean_inrepo")
        self.assertEqual(config["trainer"]["monitor"], "macro_f1")
        self.assertFalse(config["trainer"]["use_amp"])
        self.assertFalse(config["research"]["test_during_training"])

    def test_clean_run_notes_distinguish_final_from_exploratory(self) -> None:
        self.assertTrue(clean_run_notes("exploratory").startswith("Exploratory"))
        self.assertTrue(clean_run_notes("final").startswith("Final"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            clean_run_notes("draft")

    def test_run_id_identifies_clean_track_and_seed(self) -> None:
        run_id = make_run_id(42)

        self.assertRegex(run_id, re.compile(r"^caernet__clean_inrepo__seed42__\d{8}_\d{6}$"))

    def test_public_metrics_remove_prediction_payload(self) -> None:
        metrics = public_evaluation_metrics(
            {
                "accuracy": 0.75,
                "labels": [0],
                "predictions": [0],
                "confidences": [0.9],
                "image_paths": ["sample.png"],
            }
        )

        self.assertEqual(metrics, {"accuracy": 0.75})

    def test_resume_request_requires_matching_interrupted_run_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path, config_path, config, manifest_path = self._create_resume_fixture(root)
            with (
                patch.object(run_caer_clean, "REPO_ROOT", root),
                patch.object(run_caer_clean, "CHECKPOINT_ROOT", root / "checkpoints"),
                patch.object(run_caer_clean, "METADATA_ROOT", root / "artifacts" / "experiments"),
            ):
                run_id, output_dir, metadata, resolved_checkpoint = validate_resume_request(
                    checkpoint_path,
                    None,
                    config_path,
                    config,
                    manifest_path,
                )
                self.assertEqual(run_id, "caernet__clean_inrepo__seed42__fixture")
                self.assertEqual(output_dir, checkpoint_path.parent)
                self.assertEqual(metadata["status"], "interrupted")
                self.assertEqual(resolved_checkpoint, checkpoint_path)

                with self.assertRaisesRegex(ValueError, "conflicts"):
                    validate_resume_request(
                        checkpoint_path,
                        "other-run",
                        config_path,
                        config,
                        manifest_path,
                    )

                mismatched_config = dict(config)
                mismatched_config["n_gpu"] = 2
                with self.assertRaisesRegex(ValueError, "frozen runtime config"):
                    validate_resume_request(
                        checkpoint_path,
                        None,
                        config_path,
                        mismatched_config,
                        manifest_path,
                    )

                metadata_path = root / "artifacts" / "experiments" / run_id / "run_metadata.json"
                completed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                completed_metadata["status"] = "completed"
                metadata_path.write_text(json.dumps(completed_metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "status 'interrupted'"):
                    validate_resume_request(
                        checkpoint_path,
                        None,
                        config_path,
                        config,
                        manifest_path,
                    )

    def test_mark_interrupted_records_the_last_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path, _, _, _ = self._create_resume_fixture(root)
            run_id = checkpoint_path.parent.name
            metadata_path = root / "artifacts" / "experiments" / run_id / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "running"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            registry_path = root / "experiments" / "registry.csv"
            registry_path.parent.mkdir()
            registry_path.write_text(
                "run_id,status,model,variant,seed,git_sha,config,checkpoint,"
                "val_accuracy,val_macro_f1,test_accuracy,test_macro_f1,neutral_f1,"
                "params,latency_ms,notes\n",
                encoding="utf-8",
            )

            with (
                patch.object(run_caer_clean, "REPO_ROOT", root),
                patch.object(run_caer_clean, "CHECKPOINT_ROOT", root / "checkpoints"),
                patch.object(run_caer_clean, "METADATA_ROOT", root / "artifacts" / "experiments"),
                patch.object(run_caer_clean, "REGISTRY_PATH", registry_path),
            ):
                mark_interrupted(Namespace(run_id=run_id, reason="operator requested"))

            updated = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "interrupted")
            self.assertEqual(updated["last_completed_epoch"], 1)
            self.assertEqual(updated["last_checkpoint_sha256"], sha256(checkpoint_path))

    def test_completed_provenance_hashes_only_train_and_validation_detector_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, output_dir, manifest_path, history = self._create_completed_fixture(root)
            with patch.object(run_caer_clean, "REPO_ROOT", root):
                provenance = completed_run_provenance(output_dir, manifest_path, history)

            self.assertEqual(provenance["best_epoch"], 2)
            self.assertEqual(provenance["last_completed_epoch"], 2)
            self.assertEqual(
                provenance["effective_config_sha256"],
                sha256(output_dir / "config.json"),
            )
            self.assertEqual(set(provenance["detector_hashes"]), {"train.txt", "val.txt"})
            self.assertEqual(
                provenance["detector_hashes"]["val.txt"],
                sha256(manifest_path.parent / "val.txt"),
            )

    def test_reconcile_completed_repairs_stale_completion_provenance_without_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_id, output_dir, _, _ = self._create_completed_fixture(root)
            registry_path = root / "experiments" / "registry.csv"
            registry_path.parent.mkdir()
            registry_path.write_text(
                "run_id,status,model,variant,seed,git_sha,config,config_sha256,"
                "effective_config_sha256,manifest_sha256,detector_hashes,checkpoint,"
                "checkpoint_sha256,"
                "val_accuracy,val_macro_f1,test_accuracy,test_macro_f1,neutral_f1,"
                "params,latency_ms,notes\n",
                encoding="utf-8",
            )
            with (
                patch.object(run_caer_clean, "REPO_ROOT", root),
                patch.object(run_caer_clean, "CHECKPOINT_ROOT", root / "checkpoints"),
                patch.object(run_caer_clean, "METADATA_ROOT", root / "artifacts" / "experiments"),
                patch.object(run_caer_clean, "REGISTRY_PATH", registry_path),
                patch.object(run_caer_clean, "git_sha", return_value="reconcile-sha"),
            ):
                reconcile_completed(Namespace(run_id=run_id))

            metadata_path = root / "artifacts" / "experiments" / run_id / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["git_sha"], "original-run-sha")
            self.assertEqual(metadata["provenance_reconciled_by_git_sha"], "reconcile-sha")
            self.assertEqual(metadata["last_completed_epoch"], 2)
            self.assertEqual(metadata["last_checkpoint_sha256"], sha256(output_dir / "last.pt"))
            self.assertEqual(set(metadata["detector_hashes"]), {"train.txt", "val.txt"})

            with registry_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["val_macro_f1"], "0.75")
            self.assertEqual(rows[0]["effective_config_sha256"], sha256(output_dir / "config.json"))
            self.assertEqual(rows[0]["manifest_sha256"], metadata["manifest_sha256"])
            self.assertEqual(rows[0]["checkpoint_sha256"], sha256(output_dir / "best.pt"))
            self.assertEqual(
                json.loads(rows[0]["detector_hashes"]),
                metadata["detector_hashes"],
            )
            self.assertEqual(rows[0]["test_accuracy"], "")
            self.assertEqual(rows[0]["test_macro_f1"], "")


if __name__ == "__main__":
    unittest.main()
