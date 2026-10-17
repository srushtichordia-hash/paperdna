"""Data model for PaperDNA research paper records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperRecord:
    """A validated, normalized research paper record.

    ``source`` + ``source_id`` form the stable identity of the paper within
    its origin system (e.g. ``"arxiv"`` + ``"2101.00001"``). ``doi`` is
    normalized (lowercase, without URL/prefix forms) and used as a
    cross-source deduplication key when present.
    """

    source: str
    source_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: str | None = None  # "YYYY" or "YYYY-MM-DD"
    url: str | None = None
    doi: str | None = None


@dataclass(frozen=True)
class RejectedRecord:
    """A record that failed validation, with its original payload preserved."""

    raw: object  # the original input, whatever it was (dict, str, ...)
    reason: str  # human-readable validation failure reason


@dataclass(frozen=True)
class DuplicateRecord:
    """A duplicate record paired with the retained record it duplicates.

    ``matched_on`` names the key that established the match: ``"doi"``,
    ``"source_id"``, or ``"title_year_author"``.
    """

    duplicate: PaperRecord
    retained_source: str
    retained_source_id: str
    matched_on: str


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of an ingestion batch.

    ``stats`` keys and their exact meanings:

    - ``loaded``: input records read from the batch.
    - ``validated``: records that passed validation (before deduplication).
    - ``rejected``: records that failed validation.
    - ``unique``: distinct papers retained after deduplication.
    - ``duplicates``: validated records dropped as duplicates.

    Invariants (guaranteed by construction and covered by tests):

    - ``loaded == validated + rejected``
    - ``validated == unique + duplicates``
    """

    papers: tuple[PaperRecord, ...]  # unique papers, input order preserved
    rejected: tuple[RejectedRecord, ...]
    duplicate_records: tuple[DuplicateRecord, ...]
    stats: dict[str, int]

    @property
    def duplicates(self) -> int:
        """Number of duplicates detected."""
        return self.stats["duplicates"]
