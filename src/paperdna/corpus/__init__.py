"""Corpus ingestion, validation, normalization, and deduplication."""

from paperdna.corpus.dedup import deduplicate
from paperdna.corpus.ingest import ingest_records, load_jsonl
from paperdna.corpus.normalize import normalize_authors, normalize_doi, normalize_text
from paperdna.corpus.schema import (
    DuplicateRecord,
    IngestionResult,
    PaperRecord,
    RejectedRecord,
)
from paperdna.corpus.validation import validate_record

__all__ = [
    "DuplicateRecord",
    "IngestionResult",
    "PaperRecord",
    "RejectedRecord",
    "deduplicate",
    "ingest_records",
    "load_jsonl",
    "normalize_authors",
    "normalize_doi",
    "normalize_text",
    "validate_record",
]
