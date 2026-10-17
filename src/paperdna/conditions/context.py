"""Expose condition differences as context for NLI classifications.

Goal: contextualize — never override. The NLI label is embedded
unchanged; a contradiction is never downgraded to neutral because a
condition differs. Differences are reported as contextual evidence whose
scientific impact requires interpretation.
"""

from __future__ import annotations

import logging

from paperdna.conditions.schema import ConditionComparison, NliWithContext
from paperdna.nli.schema import PairClassification

logger = logging.getLogger(__name__)


def contextualize_nli(
    classification: PairClassification, comparison: ConditionComparison
) -> NliWithContext:
    """Attach condition context to an NLI classification.

    The classification object is embedded by reference, unmodified: no
    label rewriting, no score changes, no automatic downgrades. Notes
    describe what differs (or is unknown) and explicitly state that a
    condition difference alone neither explains nor refutes the modeled
    relationship. Missing conditions are flagged as absence of
    information, not as evidence of difference.
    """
    if not isinstance(classification, PairClassification):
        raise TypeError(
            f"expected PairClassification, got {type(classification).__name__}"
        )
    if not isinstance(comparison, ConditionComparison):
        raise TypeError(f"expected ConditionComparison, got {type(comparison).__name__}")

    differing = [f.category for f in comparison.fields if f.status == "differing"]
    missing = [f.category for f in comparison.fields if f.status == "missing"]
    uncertain = [f.category for f in comparison.fields if f.status == "uncertain"]
    matching = [f.category for f in comparison.fields if f.status == "matching"]

    notes: list[str] = []
    if differing:
        notes.append(
            "Conditions differ (" + ", ".join(differing) + "): contextual "
            "evidence only. A difference does not by itself explain or refute "
            "the modeled relationship; scientific impact requires "
            "interpretation."
        )
    if missing:
        notes.append(
            "Missing condition mentions for (" + ", ".join(missing) + "): no "
            "mention found in at least one claim. Absence of information is "
            "not evidence that the studies used different conditions."
        )
    if uncertain:
        notes.append(
            "Ambiguously worded conditions (" + ", ".join(uncertain) + ") were "
            "extracted; their status is unreliable and should not be treated "
            "as verified."
        )
    if matching and not differing and not uncertain:
        notes.append(
            "All extracted conditions match (" + ", ".join(matching) + "). "
            "Because extraction is incomplete, this does not prove the "
            "studies used identical conditions."
        )
    elif matching:
        notes.append(
            "Matching conditions (" + ", ".join(matching) + ") provide no "
            "explanation for divergent findings on their own."
        )

    logger.debug(
        "Contextualized %s vs %s -> %d note(s)",
        comparison.premise_source_id, comparison.hypothesis_source_id, len(notes),
    )
    return NliWithContext(
        classification=classification, conditions=comparison, notes=tuple(notes)
    )
