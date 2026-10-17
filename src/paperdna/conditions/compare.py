"""Comparison of experimental conditions across two claims."""

from __future__ import annotations

import logging

from paperdna.conditions.patterns import normalize_value
from paperdna.conditions.schema import (
    CONDITION_CATEGORIES,
    FIELD_STATUSES,
    ClaimConditions,
    ConditionComparison,
    FieldComparison,
)

logger = logging.getLogger(__name__)


def _unique_normalized(values: tuple[str, ...]) -> frozenset[str]:
    return frozenset(normalize_value(v) for v in values)


def compare_conditions(
    premise: ClaimConditions, hypothesis: ClaimConditions
) -> ConditionComparison:
    """Compare conditions field-by-field across two claims.

    Status rules, in precedence order:

    1. ``missing`` — either side has no mention for the category. A
       missing condition is NOT evidence that the studies differed; it
       only means the text does not say.
    2. ``uncertain`` — both sides have mentions but any is ambiguous
       (generic wording), so equality/difference is unreliable.
    3. ``matching`` / ``differing`` — normalized value sets are equal or
       not.

    Non-``ClaimConditions`` input raises ``TypeError``. Field order is
    always ``CONDITION_CATEGORIES``; stats always sum to the category
    count (both covered by tests).
    """
    if not isinstance(premise, ClaimConditions):
        raise TypeError(f"expected ClaimConditions, got {type(premise).__name__}")
    if not isinstance(hypothesis, ClaimConditions):
        raise TypeError(f"expected ClaimConditions, got {type(hypothesis).__name__}")

    fields: list[FieldComparison] = []
    stats = {status: 0 for status in FIELD_STATUSES}

    for category in CONDITION_CATEGORIES:
        p_mentions = premise.get(category)
        h_mentions = hypothesis.get(category)
        p_values = tuple(dict.fromkeys(m.value for m in p_mentions))
        h_values = tuple(dict.fromkeys(m.value for m in h_mentions))

        if not p_values or not h_values:
            status = "missing"
        elif any(m.ambiguous for m in p_mentions) or any(
            m.ambiguous for m in h_mentions
        ):
            status = "uncertain"
        elif _unique_normalized(p_values) == _unique_normalized(h_values):
            status = "matching"
        else:
            status = "differing"

        stats[status] += 1
        fields.append(
            FieldComparison(
                category=category,
                status=status,
                premise_values=p_values,
                hypothesis_values=h_values,
            )
        )

    comparison = ConditionComparison(
        premise_source=premise.source,
        premise_source_id=premise.source_id,
        hypothesis_source=hypothesis.source,
        hypothesis_source_id=hypothesis.source_id,
        fields=tuple(fields),
        stats=stats,
    )
    logger.debug(
        "Compared %s:%s vs %s:%s -> %s",
        premise.source, premise.source_id, hypothesis.source, hypothesis.source_id,
        comparison.summary,
    )
    return comparison
