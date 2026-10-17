"""Building validated evidence attributions from claims, classifications,
and paper records.

Validation strategy:

- **Reject** (raise): structurally invalid offsets (negative, or
  ``end < start``); spans that fail ``abstract[start:end] == sentence``
  when the abstract IS available (wrong pairing or fabricated claim);
  source-identity or text disagreement between claims and the
  classification/conditions passed alongside them; wrong types.
- **Mark unavailable, never fabricate**: no matching paper record →
  evidence is ``unverified`` with a stated reason; empty evidence text →
  ``unavailable``; missing URL/title stay ``None``.

The classification and condition comparison are embedded UNCHANGED —
attributions never rewrite model outputs.
"""

from __future__ import annotations

import logging
from typing import Iterable

from paperdna.claims.schema import CandidateClaim
from paperdna.conditions.schema import ConditionComparison, NliWithContext
from paperdna.evidence.errors import (
    AttributionMismatchError,
    InvalidSpanError,
)
from paperdna.evidence.resolver import PaperResolver, resolve_papers
from paperdna.evidence.schema import (
    ClaimRef,
    PaperRef,
    RelationAttribution,
    RelationInterpretation,
    SentenceEvidence,
    claim_id_for,
)
from paperdna.nli.schema import PairClassification

logger = logging.getLogger(__name__)

Records = Iterable["PaperRecord"] | PaperResolver  # noqa: F821 (string fwd ref)

_EMPTY_SENTENCE_REASON = "empty evidence sentence"
_RECORD_NOT_FOUND_REASON = "paper record not found"


def verify_span(abstract: str, sentence: str, start: int, end: int) -> bool:
    """True when ``abstract[start:end] == sentence`` with in-range offsets."""
    if start < 0 or end < start or end > len(abstract):
        return False
    return abstract[start:end] == sentence


def _occurrences(haystack: str, needle: str) -> list[int]:
    """All (possibly overlapping) start offsets of ``needle`` in ``haystack``."""
    positions: list[int] = []
    idx = haystack.find(needle)
    while idx != -1:
        positions.append(idx)
        idx = haystack.find(needle, idx + 1)
    return positions


def _as_resolver(records: Records | None) -> PaperResolver:
    if records is None:
        return PaperResolver(())
    if isinstance(records, PaperResolver):
        return records
    return resolve_papers(records)


def _paper_ref(source: str, source_id: str, resolver: PaperResolver) -> tuple[PaperRef, object]:
    record = resolver.get(source, source_id)
    if record is None:
        return PaperRef(source=source, source_id=source_id), None
    return (
        PaperRef(
            source=source,
            source_id=source_id,
            title=record.title,
            url=record.url,
            doi=record.doi,
            published=record.published,
        ),
        record,
    )


def _evidence_for_claim(claim: CandidateClaim, resolver: PaperResolver) -> SentenceEvidence:
    if claim.start < 0 or claim.end < claim.start:
        raise InvalidSpanError(
            f"invalid claim offsets: start={claim.start}, end={claim.end}"
        )
    paper, record = _paper_ref(claim.source, claim.source_id, resolver)
    ref = ClaimRef(
        claim_id=claim_id_for(claim.source, claim.source_id, claim.start, claim.end),
        source=claim.source,
        source_id=claim.source_id,
        start=claim.start,
        end=claim.end,
    )
    if not claim.sentence:
        return SentenceEvidence(
            paper=paper,
            claim=ref,
            sentence=claim.sentence,
            verification="unavailable",
            unavailable_reason=_EMPTY_SENTENCE_REASON,
        )
    if record is None:
        return SentenceEvidence(
            paper=paper,
            claim=ref,
            sentence=claim.sentence,
            verification="unverified",
            unavailable_reason=_RECORD_NOT_FOUND_REASON,
        )
    if not verify_span(record.abstract, claim.sentence, claim.start, claim.end):
        raise InvalidSpanError(
            f"claim span does not match abstract for {claim.source}:{claim.source_id}: "
            f"abstract[{claim.start}:{claim.end}] != sentence"
        )
    return SentenceEvidence(
        paper=paper,
        claim=ref,
        sentence=claim.sentence,
        verification="verified",
    )


def _evidence_for_text(
    source: str, source_id: str, text: str, resolver: PaperResolver
) -> SentenceEvidence:
    """Evidence reconstructed from a ``PairClassification`` (no offsets)."""
    paper, record = _paper_ref(source, source_id, resolver)

    def make(verification: str, reason: str | None, start: int | None, end: int | None):
        return SentenceEvidence(
            paper=paper,
            claim=ClaimRef(
                claim_id=claim_id_for(source, source_id, start, end),
                source=source,
                source_id=source_id,
                start=start,
                end=end,
            ),
            sentence=text,
            verification=verification,
            unavailable_reason=reason,
        )

    if not text:
        return make("unavailable", _EMPTY_SENTENCE_REASON, None, None)
    if record is None:
        return make("unverified", _RECORD_NOT_FOUND_REASON, None, None)
    positions = _occurrences(record.abstract, text)
    if len(positions) == 1:
        start = positions[0]
        return make("verified_by_search", None, start, start + len(text))
    if not positions:
        return make(
            "unavailable", "evidence sentence not found in abstract", None, None
        )
    return make(
        "unverified",
        f"sentence occurs {len(positions)} times in abstract; offsets unavailable",
        None,
        None,
    )


def _check_identity(kind: str, got_source: str, got_id: str, got_text: str | None,
                    want_source: str, want_id: str, want_text: str | None) -> None:
    if got_source != want_source or got_id != want_id:
        raise AttributionMismatchError(
            f"{kind} identity mismatch: {got_source}:{got_id} vs claim "
            f"{want_source}:{want_id}"
        )
    if want_text is not None and got_text != want_text:
        raise AttributionMismatchError(
            f"{kind} text does not match the claim's verbatim sentence"
        )


def _interpretation(
    classification: PairClassification,
    conditions: ConditionComparison | None,
    notes: tuple[str, ...],
) -> RelationInterpretation:
    return RelationInterpretation(
        label=classification.label,
        scores=dict(classification.scores),
        confidence=classification.confidence,
        uncertain=classification.uncertain,
        model=classification.model,
        condition_summary=conditions.summary if conditions is not None else None,
        condition_stats=dict(conditions.stats) if conditions is not None else None,
        notes=notes,
    )


def build_attribution(
    premise: CandidateClaim,
    hypothesis: CandidateClaim,
    *,
    records: Records | None = None,
    classification: PairClassification | None = None,
    conditions: ConditionComparison | None = None,
    notes: tuple[str, ...] = (),
) -> RelationAttribution:
    """Attribute one predicted relationship to its exact source sentences.

    ``records`` (any iterable of ``PaperRecord``, or a ``PaperResolver``)
    enriches each side with title/URL/DOI and enables abstract
    verification. When ``classification`` is given, its identity and
    evidence text are cross-checked against the claims
    (``AttributionMismatchError`` on disagreement) and it is embedded
    unchanged. ``conditions`` and ``notes`` are optional context; their
    identity is cross-checked the same way.
    """
    if not isinstance(premise, CandidateClaim):
        raise TypeError(f"expected CandidateClaim, got {type(premise).__name__}")
    if not isinstance(hypothesis, CandidateClaim):
        raise TypeError(f"expected CandidateClaim, got {type(hypothesis).__name__}")
    if classification is not None and not isinstance(classification, PairClassification):
        raise TypeError(f"expected PairClassification, got {type(classification).__name__}")
    if conditions is not None and not isinstance(conditions, ConditionComparison):
        raise TypeError(f"expected ConditionComparison, got {type(conditions).__name__}")

    if classification is not None:
        _check_identity(
            "classification premise", classification.premise_source,
            classification.premise_source_id, classification.premise_text,
            premise.source, premise.source_id, premise.sentence,
        )
        _check_identity(
            "classification hypothesis", classification.hypothesis_source,
            classification.hypothesis_source_id, classification.hypothesis_text,
            hypothesis.source, hypothesis.source_id, hypothesis.sentence,
        )
    if conditions is not None:
        _check_identity(
            "conditions premise", conditions.premise_source,
            conditions.premise_source_id, None,
            premise.source, premise.source_id, None,
        )
        _check_identity(
            "conditions hypothesis", conditions.hypothesis_source,
            conditions.hypothesis_source_id, None,
            hypothesis.source, hypothesis.source_id, None,
        )

    resolver = _as_resolver(records)
    premise_evidence = _evidence_for_claim(premise, resolver)
    hypothesis_evidence = _evidence_for_claim(hypothesis, resolver)

    interpretation = (
        _interpretation(classification, conditions, tuple(notes))
        if classification is not None
        else None
    )
    logger.debug(
        "Attributed %s:%s vs %s:%s (%s / %s)",
        premise.source, premise.source_id, hypothesis.source, hypothesis.source_id,
        premise_evidence.verification, hypothesis_evidence.verification,
    )
    return RelationAttribution(
        premise=premise_evidence,
        hypothesis=hypothesis_evidence,
        classification=classification,
        interpretation=interpretation,
        conditions=conditions,
    )


def attribute_from_classification(
    classification: PairClassification,
    *,
    records: Records | None = None,
    conditions: ConditionComparison | None = None,
) -> RelationAttribution:
    """Attribute a classification when the original claims are unavailable.

    Evidence sentences are taken verbatim from the classification.
    Offsets are INFERRED only when the sentence occurs exactly once in the
    resolvable abstract (``verified_by_search``); repeated occurrences or
    a missing record leave offsets ``None`` and mark the evidence
    ``unverified`` with a stated reason — offsets are never guessed.
    """
    if not isinstance(classification, PairClassification):
        raise TypeError(f"expected PairClassification, got {type(classification).__name__}")
    if conditions is not None and not isinstance(conditions, ConditionComparison):
        raise TypeError(f"expected ConditionComparison, got {type(conditions).__name__}")
    if conditions is not None:
        _check_identity(
            "conditions premise", conditions.premise_source,
            conditions.premise_source_id, None,
            classification.premise_source, classification.premise_source_id, None,
        )
        _check_identity(
            "conditions hypothesis", conditions.hypothesis_source,
            conditions.hypothesis_source_id, None,
            classification.hypothesis_source, classification.hypothesis_source_id, None,
        )

    resolver = _as_resolver(records)
    premise_evidence = _evidence_for_text(
        classification.premise_source, classification.premise_source_id,
        classification.premise_text, resolver,
    )
    hypothesis_evidence = _evidence_for_text(
        classification.hypothesis_source, classification.hypothesis_source_id,
        classification.hypothesis_text, resolver,
    )
    interpretation = (
        _interpretation(classification, conditions, ()) if classification else None
    )
    return RelationAttribution(
        premise=premise_evidence,
        hypothesis=hypothesis_evidence,
        classification=classification,
        interpretation=interpretation,
        conditions=conditions,
    )


def attribute_nli_with_context(
    premise: CandidateClaim,
    hypothesis: CandidateClaim,
    result: NliWithContext,
    *,
    records: Records | None = None,
) -> RelationAttribution:
    """Convenience integration with ``paperdna.conditions``.

    Unpacks an ``NliWithContext`` (from ``contextualize_nli``) into a full
    attribution: the classification and comparison are cross-checked
    against the claims and embedded unchanged; the condition-context
    notes are carried into the interpretation.
    """
    if not isinstance(result, NliWithContext):
        raise TypeError(f"expected NliWithContext, got {type(result).__name__}")
    return build_attribution(
        premise,
        hypothesis,
        records=records,
        classification=result.classification,
        conditions=result.conditions,
        notes=result.notes,
    )
