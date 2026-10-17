"""Data model for pairwise claim-relationship classification."""

from __future__ import annotations

from dataclasses import dataclass

from paperdna.claims.schema import CandidateClaim

# Canonical relationship labels. Order matters only for deterministic
# tie-breaking (first wins), never for interpreting model output indices.
RELATIONSHIPS: tuple[str, str, str] = ("entailment", "contradiction", "neutral")

DEFAULT_THRESHOLD = 0.6


@dataclass(frozen=True)
class ClaimPair:
    """Two candidate claims to compare: premise first, then hypothesis.

    NLI direction is not symmetric: "premise entails hypothesis" is not
    the same claim as the reverse. Both claims keep their source identity
    and exact evidence sentence.
    """

    premise: CandidateClaim
    hypothesis: CandidateClaim


@dataclass(frozen=True)
class RawPairScores:
    """One raw result from an NLI backend: probabilities per relationship.

    ``truncated`` records whether the backend had to truncate the pair to
    fit its maximum input length (informational; long inputs lose tail
    context).
    """

    scores: dict[str, float]
    truncated: bool = False


@dataclass(frozen=True)
class PairClassification:
    """The classified relationship between two candidate claims.

    Both claims' source identities and exact evidence sentences are copied
    out (serializable, unmutated sources). ``label`` is the argmax of the
    model's scores; ``scores`` preserves all three probabilities;
    ``confidence`` is the label's score.

    Uncertainty is preserved, never forced away: ``uncertain`` is True
    when ``confidence < threshold`` or when either claim is hedged. The
    label is still reported so callers can route pairs to review without
    losing the model's opinion. NLI softmax scores are *not* calibrated
    probabilities; the threshold is a heuristic, not a statistical
    guarantee.
    """

    premise_source: str
    premise_source_id: str
    premise_text: str
    hypothesis_source: str
    hypothesis_source_id: str
    hypothesis_text: str
    label: str
    scores: dict[str, float]
    confidence: float
    uncertain: bool
    hedged: bool
    truncated: bool
    model: str
