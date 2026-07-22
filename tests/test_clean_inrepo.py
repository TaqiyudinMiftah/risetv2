from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from caer_research.checkpointing import load_model_checkpoint
from caer_research.data import CAERSTwoStreamDataset, load_manifest
from caer_research.engine import extract_logits
from caer_research.metrics import classification_metrics
from caer_research.models import CAERNet, NotebookCAERNet
from caer_research.trainer import Trainer


class TinyTwoStreamDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "face": torch.full((3, 2, 2), float(index)),
            "context": torch.full((3, 2, 2), float(index + 1)),
            "label": torch.tensor(index % 2),
            "image_path": f"sample-{index}.png",
        }


class TinyTwoStreamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 7)

    def forward(self, face: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = torch.stack([face.mean((1, 2, 3)), context.mean((1, 2, 3))], dim=1)
        return self.classifier(features)


class CleanInRepoTests(unittest.TestCase):
    def test_caernet_forward_matches_expected_contract(self) -> None:
        model = CAERNet().eval()
        face = torch.randn(2, 3, 96, 96)
        context = torch.randn(2, 3, 112, 112)

        with torch.inference_mode():
            logits = model(face, context)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 2_390_028)

    def test_notebook_model_returns_logits_and_normalized_fusion_weights(self) -> None:
        model = NotebookCAERNet().eval()
        face = torch.randn(2, 3, 96, 96)
        context = torch.randn(2, 3, 112, 112)

        with torch.inference_mode():
            output = model(face, context)

        self.assertEqual(tuple(extract_logits(output).shape), (2, 7))
        torch.testing.assert_close(output["fusion_weights"].sum(dim=1), torch.ones(2))

    def test_manifest_dataset_preserves_crop_mask_and_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "train" / "Angry" / "sample.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(image_path)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "train__sample",
                        "image_path": "train/Angry/sample.png",
                        "label": "Anger",
                        "split": "train",
                        "face_bbox": [5, 5, 15, 15],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = CAERSTwoStreamDataset(manifest_path, root, split="train")
            sample = dataset[0]

            self.assertEqual(sample["face"].size, (10, 10))
            self.assertEqual(sample["context"].getpixel((10, 10)), (0, 0, 0))
            self.assertEqual(sample["label"].item(), 0)
            self.assertEqual(load_manifest(manifest_path)[0].label, "Anger")

    def test_checkpoint_loader_accepts_data_parallel_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
            source = torch.nn.Linear(3, 2)
            state = {f"module.{key}": value for key, value in source.state_dict().items()}
            torch.save({"state_dict": state}, checkpoint_path)
            target = torch.nn.Linear(3, 2)

            load_model_checkpoint(target, checkpoint_path)

            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                torch.testing.assert_close(source_parameter, target_parameter)

    def test_classification_metrics_are_sample_weighted(self) -> None:
        metrics = classification_metrics(
            labels=[0, 0, 1, 1],
            predictions=[0, 1, 1, 1],
            confidences=[0.9, 0.6, 0.8, 0.7],
            class_names=("A", "B"),
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("ece_15", metrics)
        self.assertEqual(metrics["per_class"]["A"]["support"], 2)

    def test_trainer_writes_reproducible_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = TinyTwoStreamModel()
            loader = DataLoader(TinyTwoStreamDataset(), batch_size=2, shuffle=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            generator = torch.Generator().manual_seed(42)
            trainer = Trainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                optimizer=optimizer,
                criterion=torch.nn.CrossEntropyLoss(),
                device=torch.device("cpu"),
                output_dir=temporary_directory,
                config={"seed": 42},
                epochs=1,
                patience=2,
                train_generator=generator,
            )

            history = trainer.fit()
            checkpoint = torch.load(trainer.last_path, map_location="cpu", weights_only=False)

            self.assertEqual(len(history), 1)
            self.assertTrue(trainer.best_path.is_file())
            self.assertTrue(trainer.history_csv_path.is_file())
            self.assertIn("rng_state", checkpoint)
            self.assertIn("early_stopping_count", checkpoint)
            self.assertIn("train_generator_state", checkpoint)

    def test_trainer_resume_matches_uninterrupted_training(self) -> None:
        def build_trainer(output_dir: Path, epochs: int) -> Trainer:
            model = TinyTwoStreamModel()
            train_generator = torch.Generator().manual_seed(314159)
            train_loader = DataLoader(
                TinyTwoStreamDataset(),
                batch_size=2,
                shuffle=True,
                generator=train_generator,
            )
            val_loader = DataLoader(TinyTwoStreamDataset(), batch_size=2, shuffle=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            return Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=torch.nn.CrossEntropyLoss(),
                device=torch.device("cpu"),
                output_dir=output_dir,
                config={"seed": 42, "n_gpu": 1},
                epochs=epochs,
                patience=10,
                train_generator=train_generator,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            torch.manual_seed(2026)
            uninterrupted = build_trainer(root / "uninterrupted", epochs=3)
            uninterrupted_history = uninterrupted.fit()

            torch.manual_seed(2026)
            interrupted = build_trainer(root / "interrupted", epochs=1)
            interrupted.fit()
            resumed = build_trainer(root / "interrupted", epochs=3)
            resumed.resume(interrupted.last_path)
            resumed_history = resumed.fit()

            self.assertEqual(resumed.start_epoch, 2)
            self.assertEqual(resumed_history, uninterrupted_history)
            self.assertEqual(resumed.best_metric, uninterrupted.best_metric)
            self.assertEqual(resumed.early_stopping_count, uninterrupted.early_stopping_count)
            self.assertEqual(resumed.scheduler.state_dict(), uninterrupted.scheduler.state_dict())
            resumed_optimizer_state = resumed.optimizer.state_dict()["state"]
            uninterrupted_optimizer_state = uninterrupted.optimizer.state_dict()["state"]
            self.assertEqual(resumed_optimizer_state.keys(), uninterrupted_optimizer_state.keys())
            for parameter_id in resumed_optimizer_state:
                for key, value in resumed_optimizer_state[parameter_id].items():
                    expected = uninterrupted_optimizer_state[parameter_id][key]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    else:
                        self.assertEqual(value, expected)
            for resumed_parameter, uninterrupted_parameter in zip(
                resumed.model.parameters(), uninterrupted.model.parameters()
            ):
                torch.testing.assert_close(resumed_parameter, uninterrupted_parameter, rtol=0, atol=0)

    def test_trainer_resume_requires_loader_generator_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_model = TinyTwoStreamModel()
            trainer = Trainer(
                model=source_model,
                train_loader=DataLoader(TinyTwoStreamDataset(), batch_size=2, shuffle=False),
                val_loader=DataLoader(TinyTwoStreamDataset(), batch_size=2, shuffle=False),
                optimizer=torch.optim.SGD(source_model.parameters(), lr=0.01),
                criterion=torch.nn.CrossEntropyLoss(),
                device=torch.device("cpu"),
                output_dir=root / "source",
                config={"seed": 42},
                epochs=1,
                train_generator=torch.Generator().manual_seed(42),
            )
            trainer.fit()
            checkpoint = torch.load(trainer.last_path, map_location="cpu", weights_only=False)
            checkpoint.pop("train_generator_state")
            invalid_checkpoint = root / "missing-generator.pt"
            torch.save(checkpoint, invalid_checkpoint)

            target_model = TinyTwoStreamModel()
            target = Trainer(
                model=target_model,
                train_loader=DataLoader(TinyTwoStreamDataset(), batch_size=2, shuffle=False),
                val_loader=DataLoader(TinyTwoStreamDataset(), batch_size=2, shuffle=False),
                optimizer=torch.optim.SGD(target_model.parameters(), lr=0.01),
                criterion=torch.nn.CrossEntropyLoss(),
                device=torch.device("cpu"),
                output_dir=root / "target",
                config={"seed": 42},
                epochs=2,
                train_generator=torch.Generator().manual_seed(42),
            )
            with self.assertRaisesRegex(ValueError, "DataLoader generator state"):
                target.resume(invalid_checkpoint)


if __name__ == "__main__":
    unittest.main()
