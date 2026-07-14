from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from research_audit import (
    DetectorParseError,
    configure_determinism,
    extract_state_dict,
    load_checkpoint_payload,
    parse_detector_line,
    validate_bbox,
)


class ResearchAuditTests(unittest.TestCase):
    def test_detector_parsing_canonicalizes_angry(self) -> None:
        sample = parse_detector_line("Angry/0001.png,0,-2,3,20,30\n", 1)

        self.assertEqual(sample.image_path, "Anger/0001.png")
        self.assertEqual(sample.label, "Anger")
        self.assertEqual(sample.label_index, 0)
        self.assertEqual(sample.face_bbox, (-2, 3, 20, 30))

    def test_detector_parsing_rejects_mismatched_label_index(self) -> None:
        with self.assertRaises(DetectorParseError):
            parse_detector_line("Happy/0001.png,2,1,2,20,30\n", 3)

    def test_bbox_validation_supports_upstream_and_strict_modes(self) -> None:
        self.assertTrue(validate_bbox((-3, 1, 20, 30)))
        self.assertFalse(validate_bbox((-3, 1, 20, 30), (64, 64), require_inside_image=True))
        self.assertTrue(validate_bbox((0, 1, 20, 30), (64, 64), require_inside_image=True))
        self.assertFalse(validate_bbox((10, 1, 10, 30)))

    def test_checkpoint_loading_extracts_dataparallel_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
            torch.save({"state_dict": {"module.weight": torch.ones(2)}}, checkpoint_path)

            payload = load_checkpoint_payload(checkpoint_path)
            state_dict = extract_state_dict(payload)

        self.assertEqual(list(state_dict), ["weight"])
        self.assertTrue(torch.equal(state_dict["weight"], torch.ones(2)))

    def test_deterministic_evaluation_state_repeats_predictions(self) -> None:
        configure_determinism(42)
        model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Dropout(p=0.9))
        inputs = torch.rand(3, 4)
        model.eval()
        first_prediction = model(inputs).argmax(dim=1)

        configure_determinism(42)
        second_prediction = model(inputs).argmax(dim=1)

        self.assertTrue(torch.equal(first_prediction, second_prediction))


if __name__ == "__main__":
    unittest.main()
