"""Pairwise claim-relationship classification with uncertainty handling."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from paperdna.claims.schema import CandidateClaim
from paperdna.nli.classifier import NliClassifier
from paperdna.nli.errors import InvalidScoresError
from paperdna.nli.schema import RELATIONSHIPS, ClaimPair, PairClassification

logger = logging.getLogger(__name__)


def _validate_scores(scores: object) -> dict[str, float]:
    """Validate one RawPairScores mapping; returns a float-coerced copy.

    Raises InvalidScoresError for wrong keys, non-numeric or non-finite
    values, probabilities outside [0, 1], or a total far from 1.0.
    """
    if not isinstance(scores, dict):
        raise InvalidScoresError(
            f"expected a dict of scores, got {type(scores).__name__}"
        )
    if set(scores) != set(RELATIONSHIPS):
        raise InvalidScoresError(
            f"expected keys {sorted(RELATIONSHIPS)}, got {sorted(scores)}"
        )
    coerced: dict[str, float] = {}
    for label, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidScoresError(
                f"non-numeric score for {label!r}: {value!r}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise InvalidScoresError(f"non-finite score for {label!r}: {value!r}")
        if not 0.0 <= number <= 1.0:
            raise InvalidScoresError(
                f"score for {label!r} outside [0, 1]: {value!r}"
            )
        coerced[label] = number
    total = sum(coerced.values())
    if abs(total - 1.0) > 1e-3:
        raise InvalidScoresError(f"scores sum to {total}, expected ~1.0")
    return coerced


def _claim_text(claim: object, name: str) -> str:
    if not isinstance(claim, CandidateClaim):
        raise TypeError(f"{name} must be a CandidateClaim, got {type(claim).__name__}")
    text = claim.sentence.strip()
    if not text:
        raise ValueError(f"{name} has empty sentence text")
    return text


def classify_pair(
    premise: CandidateClaim,
    hypothesis: CandidateClaim,
    classifier: NliClassifier,
    *,
    threshold: float = 0.6,
) -> PairClassification:
    """Classify the relationship of two candidate claims.

    Premise is scored against hypothesis in that direction (NLI is not
    symmetric). The label is the highest-scoring relationship; scores for
    all three are preserved. ``uncertain`` is True when the label's score
    is below ``threshold`` or when either claim is hedged — tentative
    claims stay tentative, and low-confidence labels are never presented
    as established facts. Blank claims raise ``ValueError``; non-claims
    raise ``TypeError``; malformed classifier scores raise
    ``InvalidScoresError``.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    premise_text = _claim_text(premise, "premise")
    hypothesis_text = _claim_text(hypothesis, "hypothesis")

    raw = classifier.score_pairs([(premise_text, hypothesis_text)])[0]
    scores = _validate_scores(raw.scores)
    label = max(RELATIONSHIPS, key=lambda key: scores[key])  # ties: RELATIONSHIPS order
    confidence = scores[label]
    hedged = bool(premise.hedges) or bool(hypothesis.hedges)
    uncertain = bool(confidence < threshold or hedged)

    logger.debug(
        "Classified %s:%s vs %s:%s -> %s (confidence=%.3f, uncertain=%s)",
        premise.source, premise.source_id, hypothesis.source, hypothesis.source_id,
        label, confidence, uncertain,
    )
    return PairClassification(
        premise_source=premise.source,
        premise_source_id=premise.source_id,
        premise_text=premise.sentence,
        hypothesis_source=hypothesis.source,
        hypothesis_source_id=hypothesis.source_id,
        hypothesis_text=hypothesis.sentence,
        label=label,
        scores=scores,
        confidence=confidence,
        uncertain=uncertain,
        hedged=hedged,
        truncated=raw.truncated,
        model=classifier.name,
    )


def classify_pairs(
    pairs: Sequence[ClaimPair],
    classifier: NliClassifier,
    *,
    threshold: float = 0.6,
) -> list[PairClassification]:
    """Classify an ordered sequence of claim pairs, preserving order."""
    return [
        classify_pair(pair.premise, pair.hypothesis, classifier, threshold=threshold)
        for pair in pairs
    ]
