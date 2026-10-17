"""Data model for evidence attribution and sentence traceability.

Two strictly distinct kinds of information:

- **Observed evidence** (``PaperRef``, ``ClaimRef``, ``SentenceEvidence``):
  source text and metadata copied verbatim from source records, with
  offsets that can be verified against the original abstract.
- **Model interpretation** (``RelationInterpretation``): predictions and
  derived summaries from the NLI and condition stages. These are NEVER
  observed scientific facts and are typed as a separate structure so they
  can never be mistaken for quotations.
"""

from __future__ import annotations

from dataclasses import dataclass

from paperdna.conditions.schema import ConditionComparison
from paperdna.nli.schema import PairClassification

# How an evidence span's traceability was established:
# - "verified": an abstract was available and abstract[start:end] == sentence.
# - "verified_by_search": offsets were INFERRED from a unique occurrence of
#   the sentence in the abstract (only in the classification-only path).
# - "unverified": the sentence is verbatim but no abstract was available to
#   check (or the sentence occurs more than once, so offsets are unknown).
# - "unavailable": there is no usable evidence text; the reason is stated.
EVIDENCE_VERIFICATION: tuple[str, ...] = (
    "verified",
    "verified_by_search",
    "unverified",
    "unavailable",
)

# stats keys: "evidence_" + verification status. Invariant (tested):
# the four counts always sum to the number of evidence sides (always 2).
EVIDENCE_STATUSES: tuple[str, ...] = tuple(f"evidence_{v}" for v in EVIDENCE_VERIFICATION)

# claim_id component used when abstract offsets are unknown.
UNKNOWN_OFFSETS = "unknown"


def claim_id_for(
    source: str, source_id: str, start: int | None, end: int | None
) -> str:
    """Stable claim reference: ``"{source}:{source_id}:{start}-{end}"``.

    Deterministic given the same abstract and extractor. When offsets are
    unknown (classification-only path without a resolvable record), the
    offset part is the literal ``"unknown"`` — never invented.
    """
    if start is None or end is None:
        offsets = UNKNOWN_OFFSETS
    else:
        offsets = f"{start}-{end}"
    return f"{source}:{source_id}:{offsets}"


@dataclass(frozen=True)
class PaperRef:
    """Source-paper identity and whatever metadata was available.

    ``title``/``url``/``doi``/``published`` are copied from the paper
    record when one was found; they stay ``None`` when not — missing
    metadata is never fabricated.
    """

    source: str
    source_id: str
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    published: str | None = None

    @property
    def identity(self) -> tuple[str, str]:
        return (self.source.casefold(), self.source_id)


@dataclass(frozen=True)
class ClaimRef:
    """Stable reference to one claim: identity plus abstract offsets.

    ``start``/``end`` are ``None`` when abstract offsets are genuinely
    unavailable (no record to locate the sentence in); both-or-none is
    enforced. Structurally invalid offsets raise ``InvalidSpanError``.
    """

    claim_id: str
    source: str
    source_id: str
    start: int | None
    end: int | None

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError(
                "claim offsets must both be present or both absent "
                f"(got start={self.start!r}, end={self.end!r})"
            )
        if self.start is not None and (self.start < 0 or self.end < self.start):  # type: ignore[operator]
            from paperdna.evidence.errors import InvalidSpanError

            raise InvalidSpanError(
                f"invalid claim offsets: start={self.start}, end={self.end}"
            )


@dataclass(frozen=True)
class SentenceEvidence:
    """One observed evidence side: a verbatim sentence linked to its paper.

    ``sentence`` is the exact source text; ``claim`` carries the stable
    reference and (when known) the abstract offsets. ``verification``
    states how traceability was established (see ``EVIDENCE_VERIFICATION``);
    ``unavailable_reason`` is required for ``"unverified"`` and
    ``"unavailable"`` so missing evidence is always clearly indicated.
    """

    paper: PaperRef
    claim: ClaimRef
    sentence: str
    verification: str
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.verification not in EVIDENCE_VERIFICATION:
            raise ValueError(f"unknown verification status: {self.verification!r}")
        if self.verification in ("unverified", "unavailable") and not self.unavailable_reason:
            raise ValueError(
                f"verification {self.verification!r} requires an unavailable_reason"
            )

    @property
    def available(self) -> bool:
        """False when there is no usable evidence text."""
        return self.verification != "unavailable"

    @property
    def verified(self) -> bool:
        """True when offsets are confirmed (or uniquely inferred) against
        the original abstract."""
        return self.verification in ("verified", "verified_by_search")


@dataclass(frozen=True)
class RelationInterpretation:
    """Model-generated interpretation of a relationship — NOT observed facts.

    Every field here is a model output or derived summary: the NLI label
    and scores, the confidence flag, and condition-comparison summaries.
    Nothing here is source text; source text lives only in
    ``SentenceEvidence``. Condition differences do not rewrite the label.
    """

    label: str
    scores: dict[str, float]
    confidence: float
    uncertain: bool
    model: str
    condition_summary: str | None = None
    condition_stats: dict[str, int] | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationAttribution:
    """Evidence attribution for one predicted relationship.

    - ``premise`` / ``hypothesis``: the two observed evidence sides.
    - ``classification``: the embedded ``PairClassification``, unchanged
      (by reference; never rewritten).
    - ``interpretation``: the same prediction exposed in typed,
      clearly-labeled model-generated form (``None`` for evidence-only
      attribution).
    - ``conditions``: the raw ``ConditionComparison`` when available.
    - ``stats``: counts per verification status; the four counts always
      sum to 2 (both sides accounted for — covered by tests).
    """

    premise: SentenceEvidence
    hypothesis: SentenceEvidence
    classification: PairClassification | None = None
    interpretation: RelationInterpretation | None = None
    conditions: ConditionComparison | None = None
    stats: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.stats is None:
            counts = {key: 0 for key in EVIDENCE_STATUSES}
            for evidence in (self.premise, self.hypothesis):
                counts[f"evidence_{evidence.verification}"] += 1
            object.__setattr__(self, "stats", counts)
