"""Data model for candidate research claims."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentenceSpan:
    """One sentence with character offsets into the source text.

    Invariant (covered by tests): ``text[start:end] == sentence``.
    """

    sentence: str
    start: int
    end: int


@dataclass(frozen=True)
class CandidateClaim:
    """A candidate finding sentence extracted from a paper abstract.

    ``sentence`` is verbatim source text — never rewritten, summarized, or
    fabricated. ``start``/``end`` are character offsets into the paper's
    abstract, so ``abstract[start:end] == sentence``. ``source`` and
    ``source_id`` are copied from the originating ``PaperRecord`` to keep
    every claim traceable and serializable.

    ``cue`` is the verbatim matched finding cue (e.g. ``"we find"``).
    ``hedges`` lists detected uncertainty markers (e.g. ``"may"``,
    ``"suggests"``) in order of appearance, deduplicated. Hedges are
    recorded, never removed: tentative claims stay tentative.
    """

    source: str
    source_id: str
    sentence: str
    start: int
    end: int
    cue: str
    hedges: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionResult:
    """Claims extracted from one paper, plus statistics.

    ``stats`` keys and exact meanings:

    - ``sentences``: sentences considered in the abstract.
    - ``claims``: sentences selected as candidate claims.
    - ``background``: sentences classified as background.
    - ``methods``: sentences classified as methods/procedural.
    - ``other``: sentences with no classification.
    - ``hedged``: claims containing at least one uncertainty marker.

    Invariants (guaranteed by construction, covered by tests):

    - ``sentences == claims + background + methods + other``
    - ``hedged <= claims``
    """

    claims: tuple[CandidateClaim, ...]
    stats: dict[str, int]
