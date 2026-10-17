"""Rule-based extraction of conditions from candidate claims."""

from __future__ import annotations

import logging

from paperdna.claims.schema import CandidateClaim
from paperdna.conditions.patterns import AMBIGUOUS_RULES, SPECIFIC_RULES
from paperdna.conditions.schema import ClaimConditions, ConditionMention

logger = logging.getLogger(__name__)

Candidate = tuple[int, int, int, str, str, bool]  # start, end, order, category, value, ambiguous


def _gather(sentence: str, rules, first_order: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for order, (category, pattern, ambiguous) in enumerate(rules, start=first_order):
        for match in pattern.finditer(sentence):
            candidates.append(
                (match.start(), match.end(), order, category, match.group(0), ambiguous)
            )
    return candidates


def _resolve(candidates: list[Candidate]) -> list[Candidate]:
    """Deterministic overlap resolution across ALL rules at once:
    earliest start wins; at the same start the longest match wins; then
    rule order breaks remaining ties. Overlapping shorter matches are
    dropped, so ambiguous rules never shadow specific ones."""
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))
    accepted: list[Candidate] = []
    last_end = -1
    for candidate in candidates:
        if candidate[0] >= last_end:
            accepted.append(candidate)
            last_end = candidate[1]
    return accepted


def extract_conditions(claim: CandidateClaim) -> ClaimConditions:
    """Extract experimental-condition mentions from one candidate claim.

    Every mention is a verbatim span of the claim's sentence with exact
    offsets; categories with no textual support are simply absent (never
    invented). Blank sentences produce empty conditions. Non-``CandidateClaim``
    input raises ``TypeError``.
    """
    if not isinstance(claim, CandidateClaim):
        raise TypeError(f"expected CandidateClaim, got {type(claim).__name__}")

    sentence = claim.sentence
    if not sentence.strip():
        return ClaimConditions(source=claim.source, source_id=claim.source_id, mentions=())

    candidates = _gather(sentence, SPECIFIC_RULES, first_order=0)
    candidates += _gather(sentence, AMBIGUOUS_RULES, first_order=len(SPECIFIC_RULES))
    resolved = _resolve(candidates)

    mentions = tuple(
        ConditionMention(
            category=category,
            value=value,
            start=start,
            end=end,
            ambiguous=ambiguous,
            source=claim.source,
            source_id=claim.source_id,
        )
        for start, end, _order, category, value, ambiguous in sorted(
            resolved, key=lambda c: (c[0], c[3])
        )
    )
    logger.debug(
        "Extracted %d condition mention(s) from %s:%s",
        len(mentions),
        claim.source,
        claim.source_id,
    )
    return ClaimConditions(
        source=claim.source, source_id=claim.source_id, mentions=mentions
    )
