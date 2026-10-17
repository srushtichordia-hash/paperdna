"""Evaluation metrics for claim-relationship classification.

Pure-Python precision/recall/F1 per class and macro-F1 over the three
relationships, computed from labeled versus predicted labels. Intended
for measuring a classifier against a hand-labeled set — not for
claiming scientific truth.
"""

from __future__ import annotations

from collections.abc import Sequence

from paperdna.nli.schema import RELATIONSHIPS


def classification_report(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, object]:
    """Per-class precision/recall/F1 and macro-F1.

    ``y_true`` and ``y_pred`` are sequences of canonical relationship
    labels of equal length. Returns a dict with ``per_class`` (precision,
    recall, f1, support per label), ``macro_f1``, ``accuracy``, and
    ``total``. Classes with no predictions or no gold labels score 0.0
    rather than dividing by zero.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    total = len(y_true)
    per_class: dict[str, dict[str, float]] = {}
    f1_sum = 0.0
    correct = 0

    for label in RELATIONSHIPS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(tp + fn),
        }
        f1_sum += f1

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return {
        "per_class": per_class,
        "macro_f1": f1_sum / len(RELATIONSHIPS),
        "accuracy": correct / total if total else 0.0,
        "total": total,
    }
