from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_content_disjoint_split import build_content_disjoint_protocol


class ContentDisjointSplitTests(unittest.TestCase):
    def test_prioritizes_train_and_preserves_detector_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "CAER-S"
            detector_dir = root / "detectors"
            output_dir = root / "protocol"
            for directory in (
                dataset_root / "train" / "Angry",
                dataset_root / "test" / "Angry",
                detector_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            duplicate = Image.new("RGB", (20, 20), color=(255, 0, 0))
            duplicate.save(dataset_root / "train" / "Angry" / "train_duplicate.png")
            duplicate.save(dataset_root / "test" / "Angry" / "val_duplicate.png")
            Image.new("RGB", (20, 20), color=(0, 0, 255)).save(dataset_root / "test" / "Angry" / "test_unique.png")

            rows = [
                {
                    "sample_id": "train__duplicate",
                    "image_path": "train/Angry/train_duplicate.png",
                    "label": "Anger",
                    "split": "train",
                    "face_bbox": [1, 1, 10, 10],
                },
                {
                    "sample_id": "val__duplicate",
                    "image_path": "test/Angry/val_duplicate.png",
                    "label": "Anger",
                    "split": "val",
                    "face_bbox": [1, 1, 10, 10],
                },
                {
                    "sample_id": "test__unique",
                    "image_path": "test/Angry/test_unique.png",
                    "label": "Anger",
                    "split": "test",
                    "face_bbox": [1, 1, 10, 10],
                },
            ]
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            (detector_dir / "train.txt").write_text("Angry/train_duplicate.png,0,1,1,10,10\n", encoding="utf-8")
            (detector_dir / "val.txt").write_text("Anger/val_duplicate.png,0,1,1,10,10\n", encoding="utf-8")
            (detector_dir / "test.txt").write_text("Anger/test_unique.png,0,1,1,10,10\n", encoding="utf-8")

            report = build_content_disjoint_protocol(manifest_path, dataset_root, detector_dir, output_dir)

            self.assertEqual(report["retained_counts"], {"train": 1, "val": 0, "test": 1})
            self.assertEqual(report["removed_counts"], {"train": 0, "val": 1, "test": 0})
            self.assertEqual((output_dir / "val.txt").read_text(encoding="utf-8"), "\n")
            with (output_dir / "manifest.jsonl").open(encoding="utf-8") as handle:
                retained_rows = [json.loads(line) for line in handle]
            self.assertEqual(
                [row["sample_id"] for row in retained_rows],
                [
                    "train__train__Angry__train_duplicate_png",
                    "test__test__Angry__test_unique_png",
                ],
            )

    def test_normalizes_validation_manifest_to_test_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "CAER-S"
            detector_dir = root / "detectors"
            output_dir = root / "protocol"
            (dataset_root / "test" / "Angry").mkdir(parents=True)
            detector_dir.mkdir()

            Image.new("RGB", (20, 20), color=(0, 255, 0)).save(
                dataset_root / "test" / "Angry" / "val_unique.png"
            )
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "val__train__Angry__val_unique_png",
                        "image_path": "train/Angry/val_unique.png",
                        "label": "Anger",
                        "split": "val",
                        "face_bbox": [1, 1, 10, 10],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (detector_dir / "train.txt").write_text("", encoding="utf-8")
            (detector_dir / "val.txt").write_text(
                "Anger/val_unique.png,0,1,1,10,10\n", encoding="utf-8"
            )
            (detector_dir / "test.txt").write_text("", encoding="utf-8")

            report = build_content_disjoint_protocol(
                manifest_path, dataset_root, detector_dir, output_dir
            )

            output_row = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(output_row["image_path"], "test/Angry/val_unique.png")
            self.assertEqual(output_row["sample_id"], "val__test__Angry__val_unique_png")
            self.assertEqual(report["split_storage_roots"]["val"], "test")


if __name__ == "__main__":
    unittest.main()
