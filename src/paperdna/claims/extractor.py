"""Baseline candidate-claim extraction from paper abstracts.

Pipeline: segment the abstract into sentences, then classify each
sentence with lexical cues — finding cues select candidate claims,
methods and background cues reject other sentences. Deterministic and
offline: no models, no network, no downloads.

Limitations: this is a lexical baseline. It does not understand
scientific content or truth, paraphrased findings without the listed
cues are missed, and unusual phrasing can be misclassified. It exists to
produce auditable *candidates* for later stages, not conclusions.
"""

from __future__ import annotations

import logging

from paperdna.claims.cues import (
    BACKGROUND_CUES,
    FINDING_CUES,
    METHODS_CUES,
    first_match,
    find_hedges,
)
from paperdna.claims.schema import CandidateClaim, ExtractionResult
from paperdna.claims.segment import segment_sentences
from paperdna.corpus.schema import PaperRecord

logger = logging.getLogger(__name__)

_LABEL_BACKGROUND = "background"
_LABEL_METHODS = "methods"


def extract_claims(record: PaperRecord) -> ExtractionResult:
    """Extract candidate claims from one validated paper record.

    Every claim carries the paper's identity and the exact sentence text
    with offsets into the abstract; the record is never mutated. Empty
    abstracts produce empty results. Non-``PaperRecord`` input raises
    ``TypeError``.
    """
    if not isinstance(record, PaperRecord):
        raise TypeError(f"expected PaperRecord, got {type(record).__name__}")

    claims: list[CandidateClaim] = []
    stats = {
        "sentences": 0,
        "claims": 0,
        "background": 0,
        "methods": 0,
        "other": 0,
        "hedged": 0,
    }

    for span in segment_sentences(record.abstract):
        stats["sentences"] += 1
        cue = first_match(FINDING_CUES, span.sentence)
        if cue is not None:
            hedges = find_hedges(span.sentence)
            claims.append(
                CandidateClaim(
                    source=record.source,
                    source_id=record.source_id,
                    sentence=span.sentence,
                    start=span.start,
                    end=span.end,
                    cue=cue,
                    hedges=hedges,
                )
            )
            stats["claims"] += 1
            if hedges:
                stats["hedged"] += 1
            continue

        if first_match(METHODS_CUES, span.sentence) is not None:
            stats["methods"] += 1
        elif first_match(BACKGROUND_CUES, span.sentence) is not None:
            stats["background"] += 1
        else:
            stats["other"] += 1

    logger.debug(
        "Extracted %d claim(s) from %s:%s (%d sentences)",
        stats["claims"],
        record.source,
        record.source_id,
        stats["sentences"],
    )
    return ExtractionResult(claims=tuple(claims), stats=stats)


def extract_claims_batch(
    records: "list[PaperRecord] | tuple[PaperRecord, ...]",
) -> tuple[ExtractionResult, ...]:
    """Extract claims from multiple records, preserving input order."""
    return tuple(extract_claims(record) for record in records)
