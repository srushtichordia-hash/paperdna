"""Evidence-sentence correctness audit.

Programmatically verifies, over ACTUAL generated attributions, that
every evidence side keeps PaperDNA's traceability invariants:

1. verbatim text: the evidence sentence is non-empty whenever the
   side claims any verification state;
2. exact offsets: for ``verified`` / ``verified_by_search``,
   ``abstract[start:end] == sentence`` character-for-character;
3. honest gaps: ``unverified`` / ``unavailable`` always carry a stated
   reason; offsets on such sides are the claim's own (unverified)
   offsets, never presented as checked, and the classification-only
   path reports ``None`` offsets with the literal ``unknown`` claim-id
   component rather than guessing;
4. stable identity: ``claim_id == claim_id_for(source, source_id,
   start, end)`` — including the ``unknown`` offset component.

Zero violations is the pass condition. This audits textual
traceability, NOT scientific grounding.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from paperdna.corpus.schema import PaperRecord
from paperdna.evidence.schema import (
    RelationAttribution,
    SentenceEvidence,
    claim_id_for,
)

# Verification states that promise exact, checkable offsets.
_OFFSET_STATES = ("verified", "verified_by_search")


def audit_side(
    side: SentenceEvidence, abstracts: dict[tuple[str, str], str]
) -> list[str]:
    """Audit one evidence side; returns human-readable violations."""
    problems: list[str] = []
    where = f"{side.claim.source}:{side.claim.source_id} ({side.verification})"

    # Honest gaps: unverified/unavailable evidence must state why.
    if side.verification in ("unverified", "unavailable"):
        if not side.unavailable_reason:
            problems.append(f"{where}: missing unavailable_reason")
    elif not side.sentence.strip():
        problems.append(f"{where}: verified evidence sentence is empty")

    expected_id = claim_id_for(
        side.claim.source, side.claim.source_id, side.claim.start, side.claim.end
    )
    if side.claim.claim_id != expected_id:
        problems.append(
            f"{where}: claim_id {side.claim.claim_id!r} != expected {expected_id!r}"
        )

    if side.verification in _OFFSET_STATES:
        if side.claim.start is None or side.claim.end is None:
            problems.append(f"{where}: offsets promised but missing")
        else:
            abstract = abstracts.get((side.claim.source, side.claim.source_id))
            if abstract is not None and abstract[side.claim.start:side.claim.end] != side.sentence:
                problems.append(
                    f"{where}: abstract[{side.claim.start}:{side.claim.end}] "
                    "does not equal the evidence sentence"
                )

    return problems


def audit_attribution(
    attribution: RelationAttribution, abstracts: dict[tuple[str, str], str]
) -> list[str]:
    """Audit both evidence sides of one attribution."""
    return audit_side(attribution.premise, abstracts) + audit_side(
        attribution.hypothesis, abstracts
    )


def audit_attributions(
    attributions: Iterable[RelationAttribution], records: Sequence[PaperRecord]
) -> dict:
    """Audit many attributions; returns a report with zero-violation gate."""
    abstracts = {(r.source, r.source_id): r.abstract for r in records}
    violations: list[dict] = []
    status_counts: dict[str, int] = {}
    n_sides = 0
    n_attributions = 0

    for attribution in attributions:
        n_attributions += 1
        for side in (attribution.premise, attribution.hypothesis):
            n_sides += 1
            status_counts[side.verification] = status_counts.get(side.verification, 0) + 1
        for problem in audit_attribution(attribution, abstracts):
            violations.append({"detail": problem})

    return {
        "kind": "textual_traceability_audit_not_scientific_grounding",
        "n_attributions": n_attributions,
        "n_evidence_sides": n_sides,
        "verification_counts": dict(sorted(status_counts.items())),
        "n_violations": len(violations),
        "violations": violations,
    }
