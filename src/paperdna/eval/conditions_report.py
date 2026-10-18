"""Condition-analysis correctness check against hand-labeled pairs.

Runs the deterministic rule baseline (``extract_conditions`` +
``compare_conditions``) over every labeled pair and counts per-category
agreement with the expected statuses. This is a CORRECTNESS check of
the transparent rule system, explicitly not a scientific-quality
score: the rules are deterministic, so expected == actual is the
regression signal.
"""

from __future__ import annotations

from paperdna.conditions.compare import compare_conditions
from paperdna.conditions.extract import extract_conditions
from paperdna.conditions.schema import CONDITION_CATEGORIES


def evaluate_conditions(pairs) -> dict:
    """Compare rule outputs to expected statuses for every pair."""
    rows: list[dict] = []
    for item in pairs:
        comparison = compare_conditions(
            extract_conditions(item.premise), extract_conditions(item.hypothesis)
        )
        actual = {field.category: field.status for field in comparison.fields}
        per_category = {
            category: {
                "expected": item.expected_statuses.get(category),
                "actual": actual.get(category),
                "match": actual.get(category) == item.expected_statuses.get(category),
            }
            for category in CONDITION_CATEGORIES
        }
        rows.append(
            {
                "name": item.name,
                "split": item.split,
                "match": all(entry["match"] for entry in per_category.values()),
                "per_category": per_category,
            }
        )

    matched = sum(1 for row in rows if row["match"])
    return {
        "system": "rule-baseline",
        "kind": "correctness_check_not_quality_score",
        "n_pairs": len(rows),
        "n_pairs_matching": matched,
        "agreement": round(matched / len(rows), 6) if rows else 0.0,
        "per_pair": rows,
    }
