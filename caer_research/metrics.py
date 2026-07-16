"""Sample-weighted classification and calibration metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .constants import CLASS_NAMES


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = 15,
) -> float:
    confidence_array = np.asarray(confidences, dtype=np.float64)
    correct_array = np.asarray(correct, dtype=np.float64)
    if confidence_array.shape != correct_array.shape:
        raise ValueError("Confidence and correctness arrays must have the same shape.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence_array > lower) & (confidence_array <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct_array[mask].mean()) - float(confidence_array[mask].mean())
            )
    return value


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    confidences: Sequence[float] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    class_indices = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_indices,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
    }
    if confidences is not None:
        correct = [label == prediction for label, prediction in zip(labels, predictions)]
        result["ece_15"] = expected_calibration_error(confidences, correct, bins=15)
    return result
