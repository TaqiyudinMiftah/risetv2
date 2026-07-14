from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from evaluate_caer_official import (
    expected_calibration_error,
    validate_detector,
    validate_split_access,
)
from run_caer_official import _load_checkpoint_summary


class OfficialEvaluationTests(unittest.TestCase):
    def test_expected_calibration_error(self) -> None:
        confidence = torch.tensor([0.9, 0.6])
        correct = torch.tensor([True, False])

        value = expected_calibration_error(confidence, correct, bins=10)

        self.assertAlmostEqual(value, 0.35, places=6)

    def test_test_split_requires_explicit_unlock(self) -> None:
        validate_split_access("val", allow_test=False)
        with self.assertRaisesRegex(PermissionError, "Test evaluation is locked"):
            validate_split_access("test", allow_test=False)
        validate_split_access("test", allow_test=True)

    def test_detector_hash_must_match_checkpoint_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            detector = Path(temporary_directory) / "val.txt"
            detector.write_text("sample\n", encoding="utf-8")
            from evaluate_caer_official import sha256

            config = {"experiment": {"detector_hashes": {"val.txt": sha256(detector)}}}
            self.assertEqual(validate_detector(config, "val", detector), sha256(detector))

            detector.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Detector hash mismatch"):
                validate_detector(config, "val", detector)

    def test_checkpoint_summary_loads_epoch_and_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "model_best.pth"
            torch.save({"epoch": 16, "monitor_best": 0.72}, checkpoint)

            summary = _load_checkpoint_summary(checkpoint)

            self.assertEqual(summary["best_epoch"], 16)
            self.assertAlmostEqual(summary["monitor_best"], 0.72)


if __name__ == "__main__":
    unittest.main()
