from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_caer_official
from run_caer_upstream_train import load_config


class OfficialExperimentLauncherTests(unittest.TestCase):
    def test_git_dirty_ignores_runtime_registry_only(self) -> None:
        clean_result = Mock(stdout=" M experiments/registry.csv\n")
        dirty_result = Mock(
            stdout=" M experiments/registry.csv\n M run_caer_official.py\n"
        )
        with patch.object(run_caer_official.subprocess, "run", return_value=clean_result):
            self.assertFalse(run_caer_official._git_dirty())
        with patch.object(run_caer_official.subprocess, "run", return_value=dirty_result):
            self.assertTrue(run_caer_official._git_dirty())

    def test_generated_config_records_seed_and_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            detector_dir = root / "protocol"
            detector_dir.mkdir()
            for name in ("train.txt", "val.txt", "test.txt"):
                (detector_dir / name).write_text(f"{name}\n", encoding="utf-8")
            (detector_dir / "manifest.jsonl").write_text("{}\n", encoding="utf-8")

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "n_gpu": 1,
                        "train_loader": {"args": {"batch_size": 8, "num_workers": 0}},
                        "val_loader": {"args": {"batch_size": 8, "num_workers": 0}},
                        "test_loader": {"args": {"batch_size": 8, "num_workers": 0}},
                        "optimizer": {"args": {"lr": 0.01}},
                        "trainer": {
                            "epochs": 2,
                            "early_stop": 1,
                            "save_period": 1,
                            "tensorboard": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                config=config_path,
                detector_dir=detector_dir,
                n_gpu=2,
                epochs=3,
                batch_size=16,
                num_workers=2,
                learning_rate=0.02,
                early_stop=4,
                save_period=2,
                no_tensorboard=False,
                seed=42,
            )

            with (
                patch.object(run_caer_official, "GENERATED_CONFIG_DIR", root / "generated"),
                patch.object(run_caer_official, "_git_sha", return_value="abc123"),
                patch.object(run_caer_official, "_git_dirty", return_value=False),
            ):
                output_path = run_caer_official._write_run_config(args, "run-seed42")

            generated = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(generated["seed"], 42)
            self.assertEqual(generated["n_gpu"], 2)
            self.assertEqual(generated["trainer"]["epochs"], 3)
            self.assertEqual(generated["experiment"]["git_sha"], "abc123")
            self.assertFalse(generated["experiment"]["git_dirty"])
            self.assertEqual(generated["experiment"]["variant"], "protocol")
            self.assertEqual(
                set(generated["experiment"]["detector_hashes"]),
                {"train.txt", "val.txt", "test.txt"},
            )
            self.assertIsNotNone(generated["experiment"]["manifest_hash"])

    def test_gpu_preflight_rejects_busy_or_missing_selected_gpus(self) -> None:
        args = argparse.Namespace(
            skip_gpu_check=False,
            device="0,1",
            n_gpu=2,
            min_free_gpu_mib=6000,
        )
        with (
            patch.object(run_caer_official, "_gpu_free_memory", return_value={0: 5000, 1: 7000}),
            self.assertRaisesRegex(RuntimeError, "GPU 0: 5000 MiB free"),
        ):
            run_caer_official._check_gpu_capacity(args)

        args.device = "0"
        with self.assertRaisesRegex(RuntimeError, "only selects 1 GPU"):
            run_caer_official._check_gpu_capacity(args)

    def test_registry_upserts_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = Path(temporary_directory) / "registry.csv"
            metadata = {
                "run_id": "run-seed42",
                "status": "running",
                "seed": 42,
                "git_sha": "abc123",
                "variant": "caer_s_content_disjoint_v1",
                "config": "configs/seed42.json",
                "notes": "exploratory",
            }
            with patch.object(run_caer_official, "EXPERIMENT_REGISTRY", registry):
                run_caer_official._update_registry(metadata)
                metadata["status"] = "completed"
                run_caer_official._update_registry(
                    metadata,
                    checkpoint="checkpoints/best.pth",
                    val_accuracy=0.75,
                )

            with registry.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["val_accuracy"], "0.75")

    def test_training_progress_distinguishes_interruption_from_early_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = {
                "name": "CAERNet_Final",
                "trainer": {"save_dir": "official_runs", "epochs": 45},
            }
            info_log = (
                root
                / "official_runs"
                / "log"
                / "CAERNet_Final"
                / "seed43"
                / "info.log"
            )
            info_log.parent.mkdir(parents=True)
            info_log.write_text(
                "trainer - INFO -     epoch          : 8\n",
                encoding="utf-8",
            )

            with patch.object(run_caer_official, "CAER_CODE_DIR", root):
                interrupted = run_caer_official._training_progress(config, "seed43")
                info_log.write_text(
                    "trainer - INFO -     epoch          : 29\n"
                    "trainer - INFO - Validation performance didn't improve. Training stops.\n",
                    encoding="utf-8",
                )
                early_stopped = run_caer_official._training_progress(config, "seed43")

            self.assertEqual(interrupted["latest_epoch"], 8)
            self.assertFalse(interrupted["completed"])
            self.assertEqual(early_stopped["latest_epoch"], 29)
            self.assertTrue(early_stopped["completed"])

    def test_resume_config_is_overridden_by_frozen_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resume_dir = root / "previous"
            resume_dir.mkdir()
            resume = resume_dir / "model_best.pth"
            resume.touch()
            (resume_dir / "config.json").write_text(
                json.dumps({"name": "old", "optimizer": {"lr": 0.1}}),
                encoding="utf-8",
            )
            frozen = root / "frozen.json"
            frozen.write_text(
                json.dumps({"name": "new", "seed": 43}),
                encoding="utf-8",
            )

            config = load_config(frozen, resume)

            self.assertEqual(config["name"], "new")
            self.assertEqual(config["seed"], 43)
            self.assertEqual(config["optimizer"], {"lr": 0.1})


if __name__ == "__main__":
    unittest.main()
