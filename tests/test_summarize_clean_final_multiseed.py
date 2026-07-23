from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import summarize_clean_final_multiseed as summarizer


class CleanFinalMultiseedSummaryTests(unittest.TestCase):
    def _write_run(self, root: Path, seed: int, accuracy: float) -> str:
        run_id = f"caernet__clean_inrepo_final__seed{seed}__fixture"
        metadata_path = root / "artifacts" / "experiments" / run_id / "run_metadata.json"
        metrics_path = root / "checkpoints" / run_id / "val_metrics.json"
        metadata_path.parent.mkdir(parents=True)
        metrics_path.parent.mkdir(parents=True)
        metadata = {
            "run_id": run_id,
            "status": "completed",
            "track": "clean_inrepo",
            "stage": "final",
            "seed": seed,
            "best_epoch": 45 - (seed - 42),
            "protocol": "caer_s_content_disjoint_v1",
            "manifest_sha256": "manifest-fixture-sha",
            "detector_hashes": {"train.txt": "train-fixture", "val.txt": "val-fixture"},
            "test_used_for_selection": False,
            "val_metrics": str(metrics_path.relative_to(root)),
        }
        metrics = {
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.01,
            "weighted_f1": accuracy - 0.02,
            "loss": 1.5 - accuracy,
            "nll": 1.5 - accuracy,
            "ece_15": 0.2 - (accuracy / 10),
            "samples": 6965,
            "per_class": {
                "Anger": {
                    "precision": accuracy - 0.03,
                    "recall": accuracy - 0.04,
                    "f1": accuracy - 0.035,
                    "support": 994,
                },
                "Neutral": {
                    "precision": accuracy - 0.08,
                    "recall": accuracy - 0.09,
                    "f1": accuracy - 0.085,
                    "support": 984,
                },
            },
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return run_id

    def _write_final_runs(self, root: Path) -> list[str]:
        return [
            self._write_run(root, 44, 0.90),
            self._write_run(root, 42, 0.70),
            self._write_run(root, 43, 0.80),
        ]

    def test_metric_summary_uses_sample_standard_deviation(self) -> None:
        summary = summarizer._metric_summary([1.0, 2.0, 3.0])

        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["sample_std"], 1.0)

    def test_summarize_uses_only_completed_clean_final_validation_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_ids = self._write_final_runs(root)

            summary = summarizer.summarize(run_ids, repo_root=root)

        self.assertEqual(summary["seeds"], [42, 43, 44])
        self.assertAlmostEqual(summary["aggregate"]["accuracy"]["mean"], 0.8)
        self.assertAlmostEqual(summary["aggregate"]["accuracy"]["sample_std"], 0.1)
        self.assertAlmostEqual(summary["per_class"]["Neutral"]["f1"]["mean"], 0.715)
        self.assertEqual(summary["samples_per_seed"], 6965)
        self.assertFalse(summary["test_used_for_selection"])
        self.assertFalse(summary["test_accessed"])
        self.assertFalse(summary["test_artifacts_read"])
        self.assertEqual(summary["runs"][0]["validation_metrics"], "checkpoints/caernet__clean_inrepo_final__seed42__fixture/val_metrics.json")

    def test_summarize_rejects_run_without_explicit_test_selection_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_ids = self._write_final_runs(root)
            metadata_path = (
                root / "artifacts" / "experiments" / run_ids[0] / "run_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["test_used_for_selection"] = True
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "test_used_for_selection is false"):
                summarizer.summarize(run_ids, repo_root=root)

    def test_summarize_rejects_nonstandard_metrics_reference_before_opening_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_ids = self._write_final_runs(root)
            metadata_path = (
                root / "artifacts" / "experiments" / run_ids[0] / "run_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["val_metrics"] = "unexpected_metrics.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "fixed validation metrics artifact"):
                summarizer.summarize(run_ids, repo_root=root)

    def test_write_summary_allows_only_ignored_artifacts_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_ids = self._write_final_runs(root)
            summary = summarizer.summarize(run_ids, repo_root=root)
            output_path = root / "artifacts" / "experiments" / "summary.json"

            written_path = summarizer.write_summary(summary, output_path, repo_root=root)

            self.assertEqual(json.loads(written_path.read_text(encoding="utf-8")), summary)
            with self.assertRaisesRegex(ValueError, "ignored artifacts"):
                summarizer.write_summary(summary, root / "summary.json", repo_root=root)


if __name__ == "__main__":
    unittest.main()
