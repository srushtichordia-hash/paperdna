"""Experimental-condition extraction and comparison across claims.

A deterministic, offline, stdlib-only baseline: transparent regex rules
extract condition mentions (verbatim spans with exact offsets), claims
are compared field-by-field (matching / differing / missing / uncertain),
and NLI classifications can be annotated with condition context —
without ever overriding the NLI label.

Limitations: extraction is incomplete by construction; a condition
difference does not prove it explains divergent findings; missing
conditions are not evidence of difference. See the README "Experimental
conditions" section.
"""

from paperdna.conditions.compare import compare_conditions
from paperdna.conditions.context import contextualize_nli
from paperdna.conditions.extract import extract_conditions
from paperdna.conditions.patterns import normalize_value
from paperdna.conditions.schema import (
    CONDITION_CATEGORIES,
    FIELD_STATUSES,
    ClaimConditions,
    ConditionComparison,
    ConditionMention,
    FieldComparison,
    NliWithContext,
)

__all__ = [
    "CONDITION_CATEGORIES",
    "FIELD_STATUSES",
    "ClaimConditions",
    "ConditionComparison",
    "ConditionMention",
    "FieldComparison",
    "NliWithContext",
    "compare_conditions",
    "contextualize_nli",
    "extract_conditions",
    "normalize_value",
]
