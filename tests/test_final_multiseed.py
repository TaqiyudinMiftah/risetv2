from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_caer_final_multiseed
import run_caer_official


class FinalMultiseedTests(unittest.TestCase):
    def test_final_configs_differ_only_by_seed(self) -> None:
        configs = []
        for seed, path in run_caer_final_multiseed.FINAL_CONFIGS.items():
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["seed"], seed)
            self.assertEqual(config["research"]["stage"], "final")
            self.assertFalse(config["research"]["test_during_training"])
            config.pop("seed")
            configs.append(config)

        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[1], configs[2])

    def test_multiseed_command_uses_matching_frozen_config(self) -> None:
        args = argparse.Namespace(
            device="0,1",
            n_gpu=2,
            wandb_mode="offline",
            wandb_project="caer-net-reproduction",
            wandb_entity=None,
            dry_run=True,
        )

        command = run_caer_final_multiseed.build_command(args, 43)

        self.assertIn(str(run_caer_final_multiseed.FINAL_CONFIGS[43]), command)
        self.assertEqual(command[command.index("--seed") + 1], "43")
        self.assertEqual(command[command.index("--device") + 1], "0,1")
        self.assertIn("--dry-run", command)

    def test_final_metadata_is_not_labeled_exploratory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "final.json"
            config_path.write_text(
                json.dumps(
                    {
                        "research": {"stage": "final"},
                        "experiment": {
                            "variant": "caer_s_content_disjoint_v1",
                            "git_sha": "abc123",
                            "git_dirty": False,
                            "detector_hashes": {},
                            "manifest_hash": "manifest-hash",
                        },
                        "trainer": {"monitor": "max val_accuracy"},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                config=config_path,
                seed=42,
                detector_dir=root / "detectors",
            )
            with (
                patch.object(run_caer_official, "REPO_ROOT", root),
                patch.object(run_caer_official, "_gpu_inventory", return_value=[]),
                patch.object(run_caer_official, "_software_versions", return_value={}),
            ):
                metadata = run_caer_official._initial_metadata(
                    args,
                    "final-seed42",
                    config_path,
                    ["python", "train.py"],
                )

        self.assertEqual(metadata["stage"], "final")
        self.assertFalse(metadata["exploratory"])
        self.assertTrue(metadata["notes"].startswith("Final upstream-community"))


if __name__ == "__main__":
    unittest.main()
