"""Tests for the corpus data model."""

import dataclasses

import pytest

from paperdna.corpus.schema import (
    DuplicateRecord,
    IngestionResult,
    PaperRecord,
    RejectedRecord,
)


def make_record(**overrides) -> PaperRecord:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "title": "A Title",
        "abstract": "An abstract.",
        "authors": ("Ada Lovelace",),
    }
    fields.update(overrides)
    return PaperRecord(**fields)


def test_paper_record_is_immutable():
    record = make_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.title = "Changed"  # type: ignore[misc]


def test_paper_record_optional_fields_default_to_none():
    record = make_record()
    assert record.published is None
    assert record.url is None
    assert record.doi is None


def test_rejected_record_preserves_raw_and_reason():
    raw = {"bad": True}
    rejected = RejectedRecord(raw=raw, reason="title: missing")
    assert rejected.raw is raw
    assert rejected.reason == "title: missing"


def test_duplicate_record_provenance_fields():
    dup = DuplicateRecord(
        duplicate=make_record(),
        retained_source="fixture",
        retained_source_id="p001",
        matched_on="doi",
    )
    assert dup.retained_source == "fixture"
    assert dup.retained_source_id == "p001"
    assert dup.matched_on == "doi"


def test_ingestion_result_duplicates_property_matches_stats():
    result = IngestionResult(
        papers=(),
        rejected=(),
        duplicate_records=(),
        stats={
            "loaded": 2,
            "validated": 2,
            "rejected": 0,
            "unique": 1,
            "duplicates": 1,
        },
    )
    assert result.duplicates == 1
