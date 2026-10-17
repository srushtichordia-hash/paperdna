"""Tests for deterministic text preprocessing."""

import pytest

from paperdna.retrieval.preprocess import record_to_text
from tests.retrieval.conftest import make_record


def test_format_is_title_two_newlines_abstract():
    record = make_record(title="My Title", abstract="My abstract.")
    assert record_to_text(record) == "My Title\n\nMy abstract."


def test_deterministic_across_calls():
    record = make_record()
    assert record_to_text(record) == record_to_text(record)


def test_does_not_mutate_record():
    record = make_record()
    before = (record.title, record.abstract)
    record_to_text(record)
    assert (record.title, record.abstract) == before


def test_rejects_non_record():
    with pytest.raises(TypeError):
        record_to_text({"title": "x", "abstract": "y"})  # type: ignore[arg-type]
