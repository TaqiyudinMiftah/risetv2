from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from run_caer_clean import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPLORATORY_CONFIG = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "caernet_clean_content_disjoint_exploratory_seed42.json"
)
FINAL_CONFIGS = {
    seed: REPO_ROOT
    / "configs"
    / "experiments"
    / f"caernet_clean_content_disjoint_final_seed{seed}.json"
    for seed in (42, 43, 44)
}


class CleanFinalConfigTests(unittest.TestCase):
    def test_final_configs_match_the_clean_exploratory_control_except_frozen_fields(self) -> None:
        exploratory = json.loads(EXPLORATORY_CONFIG.read_text(encoding="utf-8"))
        expected = copy.deepcopy(exploratory)
        expected["name"] = "CAERNet_CleanInRepo_ContentDisjoint_Final"
        expected["research"]["stage"] = "final"

        for seed, path in FINAL_CONFIGS.items():
            config = json.loads(path.read_text(encoding="utf-8"))
            expected["seed"] = seed
            self.assertEqual(config, expected)

    def test_final_configs_differ_only_by_seed_and_pass_clean_validation(self) -> None:
        frozen_without_seed = []
        for seed, path in FINAL_CONFIGS.items():
            config = json.loads(path.read_text(encoding="utf-8"))

            validate_config(config, expected_seed=seed)

            self.assertEqual(config["seed"], seed)
            self.assertEqual(config["research"]["stage"], "final")
            self.assertEqual(config["research"]["track"], "clean_inrepo")
            self.assertEqual(config["research"]["selection_metric"], "val_macro_f1")
            self.assertFalse(config["research"]["test_during_training"])
            config.pop("seed")
            frozen_without_seed.append(config)

        self.assertEqual(frozen_without_seed[0], frozen_without_seed[1])
        self.assertEqual(frozen_without_seed[1], frozen_without_seed[2])


if __name__ == "__main__":
    unittest.main()
