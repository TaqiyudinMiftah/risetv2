from __future__ import annotations

import unittest

from summarize_caer_multiseed import _metric_summary, _normalized_config


class MultiseedSummaryTests(unittest.TestCase):
    def test_metric_summary_uses_sample_standard_deviation(self) -> None:
        summary = _metric_summary([1.0, 2.0, 3.0])

        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["std"], 1.0)

    def test_normalized_config_removes_only_seed(self) -> None:
        config = {"seed": 42, "optimizer": {"lr": 0.01}}

        normalized = _normalized_config(config)

        self.assertEqual(normalized, {"optimizer": {"lr": 0.01}})
        self.assertEqual(config["seed"], 42)


if __name__ == "__main__":
    unittest.main()
