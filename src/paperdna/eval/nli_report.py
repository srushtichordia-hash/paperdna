"""NLI evaluation: macro-F1 and per-class metrics from actual predictions.

Runs each labeled pair through ``classify_pair`` with a chosen
classifier, then computes ``classification_report`` on the ACTUAL
predicted labels versus the hand-labeled gold labels, per split.

System separation (mirroring the honesty rules elsewhere in PaperDNA):

- ``overlap``: the deterministic, non-circular floor baseline — its
  predictions do not derive from the gold labels, so the reported
  metrics are genuine measurements.
- ``scripted``: replays fixture scores authored FROM the gold labels.
  Agreement is circular by construction; the report flags this
  explicitly as ``circular`` and it is pipeline validation only.
- ``model``: the optional real cross-encoder (explicitly installed
  ``nli`` extra); genuine predictions, reported with the same caveats
  as any general-domain NLI model on synthetic scientific text.
"""

from __future__ import annotations

from paperdna.nli.classify import classify_pair
from paperdna.nli.eval import classification_report
from paperdna.nli.schema import RELATIONSHIPS


def evaluate_nli(
    pairs,  # Sequence[LabeledPair]
    classifier,
    *,
    threshold: float = 0.6,
    circular: bool = False,
) -> dict:
    """Classify every labeled pair and score predictions vs gold.

    Returns per-split and overall classification reports plus one row
    per pair (gold, prediction, confidence, uncertain). ``circular=True
    `` marks the whole report as plumbing validation, never a metric.
    """
    rows: list[dict] = []
    for item in pairs:
        classification = classify_pair(
            item.premise, item.hypothesis, classifier, threshold=threshold
        )
        rows.append(
            {
                "name": item.name,
                "split": item.split,
                "gold": item.expected_label,
                "predicted": classification.label,
                "match": classification.label == item.expected_label,
                "confidence": round(classification.confidence, 6),
                "uncertain": classification.uncertain,
                "uncertain_match": classification.uncertain == item.expected_uncertain,
            }
        )

    def _report(subset: list[dict]) -> dict:
        if not subset:
            return {"total": 0, "macro_f1": 0.0, "accuracy": 0.0, "per_class": {}}
        result = classification_report(
            [row["gold"] for row in subset], [row["predicted"] for row in subset]
        )
        result["macro_f1"] = round(result["macro_f1"], 6)
        result["accuracy"] = round(result["accuracy"], 6)
        result["uncertain_rate"] = round(
            sum(1 for row in subset if row["uncertain"]) / len(subset), 6
        )
        return result

    splits = sorted({row["split"] for row in rows})
    return {
        "classifier": classifier.name,
        "threshold": threshold,
        "circular": circular,
        "circular_note": (
            "Scripted scores were authored from the gold labels; agreement here "
            "validates pipeline plumbing only and is NOT a model-quality metric."
            if circular
            else None
        ),
        "labels": list(RELATIONSHIPS),
        "overall": _report(rows),
        "per_split": {split: _report([r for r in rows if r["split"] == split]) for split in splits},
        "per_pair": rows,
    }
