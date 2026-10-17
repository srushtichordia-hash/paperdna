"""Deduplication of validated paper records.

Deduplication uses a hierarchy of keys, checked in order for each record:

1. ``doi`` — the normalized DOI, which identifies a paper across sources.
2. ``(source, source_id)`` — the stable source-scoped identifier.
3. ``(title, year, first author)`` — a conservative fallback for records
   lacking both a DOI and a matching source ID (e.g. the same paper seen
   from two sources with independent IDs).

The fallback is intentionally conservative — it requires the case-folded
title, publication year, and first-author name all to match — because a
looser rule (e.g. title only) would merge distinct papers with generic
titles. It can still produce false matches for genuinely different papers
that share all three; this is an accepted, documented trade-off, and the
``matched_on`` field of every ``DuplicateRecord`` makes fallback matches
auditable.
"""

from __future__ import annotations

import logging

from paperdna.corpus.normalize import normalize_doi
from paperdna.corpus.schema import DuplicateRecord, PaperRecord

logger = logging.getLogger(__name__)

KEY_DOI = "doi"
KEY_SOURCE_ID = "source_id"
KEY_FALLBACK = "title_year_author"


def candidate_keys(record: PaperRecord) -> list[tuple]:
    """Return the deduplication keys for a record, in priority order."""
    keys: list[tuple] = []
    if record.doi:
        # Re-normalize defensively so dedup is correct even for records
        # constructed outside the validation pipeline.
        doi = normalize_doi(record.doi)
        if doi:
            keys.append((KEY_DOI, doi))
    keys.append((KEY_SOURCE_ID, record.source.casefold(), record.source_id))
    year = record.published[:4] if record.published else ""
    keys.append(
        (
            KEY_FALLBACK,
            record.title.casefold(),
            year,
            record.authors[0].casefold(),
        )
    )
    return keys


def deduplicate(
    records: list[PaperRecord],
) -> tuple[list[PaperRecord], list[DuplicateRecord]]:
    """Remove duplicate records, keeping the first occurrence of each paper.

    Returns ``(unique, duplicates)`` in input order. Every duplicate is
    reported with the retained record it matches and the key that matched,
    so provenance is never silently discarded.
    """
    unique: list[PaperRecord] = []
    duplicates: list[DuplicateRecord] = []
    seen: dict[tuple, PaperRecord] = {}

    for record in records:
        matched_key: tuple | None = None
        retained: PaperRecord | None = None
        for key in candidate_keys(record):
            if key in seen:
                matched_key = key
                retained = seen[key]
                break

        if retained is not None:
            duplicates.append(
                DuplicateRecord(
                    duplicate=record,
                    retained_source=retained.source,
                    retained_source_id=retained.source_id,
                    matched_on=matched_key[0],
                )
            )
            logger.debug(
                "Duplicate %s:%s matched %s:%s on %s",
                record.source,
                record.source_id,
                retained.source,
                retained.source_id,
                matched_key[0],
            )
            continue

        unique.append(record)
        for key in candidate_keys(record):
            seen.setdefault(key, record)

    return unique, duplicates
