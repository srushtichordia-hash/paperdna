"""Data model for experimental conditions and their comparison."""

from __future__ import annotations

from dataclasses import dataclass

from paperdna.nli.schema import PairClassification

# Condition categories identified in the extraction baseline. A category
# with no mention in a claim's text is simply absent — never invented.
CONDITION_CATEGORIES: tuple[str, ...] = (
    "dataset",
    "population",
    "model",
    "metric",
    "setting",
    "design",
)

# Comparison statuses. "missing" means one or both sides have no mention —
# it is NOT evidence that the studies used different conditions.
FIELD_STATUSES: tuple[str, ...] = ("matching", "differing", "missing", "uncertain")


@dataclass(frozen=True)
class ConditionMention:
    """One condition found in a claim's sentence.

    ``value`` is the VERBATIM span text (never rewritten or completed);
    ``start``/``end`` are offsets into the claim's sentence, so
    ``sentence[start:end] == value`` (global abstract offset =
    ``claim.start + start``). ``ambiguous`` marks generic wording ("our
    dataset", "various benchmarks") that cannot be compared reliably.
    Source identity is copied from the claim.
    """

    category: str
    value: str
    start: int
    end: int
    ambiguous: bool
    source: str
    source_id: str


@dataclass(frozen=True)
class ClaimConditions:
    """All conditions extracted from one candidate claim."""

    source: str
    source_id: str
    mentions: tuple[ConditionMention, ...]

    def get(self, category: str) -> tuple[ConditionMention, ...]:
        """Mentions for one category (empty tuple when not found)."""
        if category not in CONDITION_CATEGORIES:
            raise ValueError(f"unknown condition category: {category!r}")
        return tuple(m for m in self.mentions if m.category == category)


@dataclass(frozen=True)
class FieldComparison:
    """How one condition category compares across two claims."""

    category: str
    status: str  # one of FIELD_STATUSES
    premise_values: tuple[str, ...]  # verbatim, unique, in mention order
    hypothesis_values: tuple[str, ...]


@dataclass(frozen=True)
class ConditionComparison:
    """Field-by-field comparison of two claims' conditions.

    ``stats`` counts the four statuses; the counts always sum to
    ``len(CONDITION_CATEGORIES)`` (covered by tests).
    """

    premise_source: str
    premise_source_id: str
    hypothesis_source: str
    hypothesis_source_id: str
    fields: tuple[FieldComparison, ...]
    stats: dict[str, int]

    def field(self, category: str) -> FieldComparison:
        if category not in CONDITION_CATEGORIES:
            raise ValueError(f"unknown condition category: {category!r}")
        for item in self.fields:
            if item.category == category:
                return item
        raise KeyError(category)

    @property
    def summary(self) -> str:
        """Human-readable grouping of categories by status, in status order."""
        parts = []
        for status in FIELD_STATUSES:
            categories = [f.category for f in self.fields if f.status == status]
            if categories:
                parts.append(f"{status}: {', '.join(categories)}")
        return "; ".join(parts)


@dataclass(frozen=True)
class NliWithContext:
    """An NLI classification plus condition context.

    ``classification`` is embedded UNCHANGED — condition differences are
    exposed as contextual evidence in ``notes`` and never rewrite the NLI
    label (a contradiction stays a contradiction). The notes explain that
    scientific impact requires interpretation.
    """

    classification: PairClassification
    conditions: ConditionComparison
    notes: tuple[str, ...]
