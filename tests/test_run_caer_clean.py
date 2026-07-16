from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from run_caer_clean import make_run_id, public_evaluation_metrics, validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "caernet_clean_content_disjoint_exploratory_seed42.json"


class CleanTrainingLauncherTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
