"""Batch ingestion: load -> validate -> deduplicate -> report statistics."""

from __future__ import annotations

import json
import logging

from paperdna.corpus.dedup import deduplicate
from paperdna.corpus.schema import IngestionResult, PaperRecord, RejectedRecord
from paperdna.corpus.validation import validate_record

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[object]:
    """Read a JSONL file into a list of raw records.

    Blank lines are skipped. A line that is not valid JSON is kept as its
    raw text so that downstream validation rejects it with provenance rather
    than aborting the batch.
    """
    records: list[object] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Unparseable JSONL line kept for rejection: %r", line)
                records.append(line)
    return records


def ingest_records(raw_records: object) -> IngestionResult:
    """Ingest an iterable of raw records.

    Malformed inputs (non-dict records, wrongly typed fields, anything at
    all) are rejected individually and never abort the batch.
    """
    papers: list[PaperRecord] = []
    rejected: list[RejectedRecord] = []
    loaded = 0

    for item in raw_records:
        loaded += 1
        outcome = validate_record(item)
        if isinstance(outcome, RejectedRecord):
            rejected.append(outcome)
            logger.debug("Rejected record: %s", outcome.reason)
        else:
            papers.append(outcome)

    unique, duplicate_records = deduplicate(papers)

    # By construction: every loaded record is either validated or rejected,
    # and every validated record is either unique or a duplicate.
    stats = {
        "loaded": loaded,
        "validated": len(papers),
        "rejected": len(rejected),
        "unique": len(unique),
        "duplicates": len(duplicate_records),
    }
    logger.info(
        "Ingested %d record(s): %d unique, %d duplicate, %d rejected",
        stats["loaded"],
        stats["unique"],
        stats["duplicates"],
        stats["rejected"],
    )
    return IngestionResult(
        papers=tuple(unique),
        rejected=tuple(rejected),
        duplicate_records=tuple(duplicate_records),
        stats=stats,
    )
