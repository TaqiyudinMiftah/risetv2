from __future__ import annotations

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
    make_run_id,
    mark_interrupted,
    public_evaluation_metrics,
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

    def test_exploratory_config_uses_guarded_clean_protocol(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        validate_config(config, expected_seed=42)

        self.assertEqual(config["research"]["track"], "clean_inrepo")
        self.assertEqual(config["trainer"]["monitor"], "macro_f1")
        self.assertFalse(config["trainer"]["use_amp"])
        self.assertFalse(config["research"]["test_during_training"])

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


if __name__ == "__main__":
    unittest.main()
