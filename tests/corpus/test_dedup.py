"""Tests for deduplication."""

from paperdna.corpus.dedup import (
    KEY_DOI,
    KEY_FALLBACK,
    KEY_SOURCE_ID,
    deduplicate,
)
from paperdna.corpus.schema import PaperRecord


def make_record(**overrides) -> PaperRecord:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "title": "Sparse Attention Mechanisms",
        "abstract": "An abstract.",
        "authors": ("Ada Lovelace", "Grace Hopper"),
        "published": "2021-01-05",
        "doi": "10.1234/ml.001",
    }
    fields.update(overrides)
    return PaperRecord(**fields)


def run(records):
    unique, duplicates = deduplicate(records)
    return unique, duplicates


class TestDoiMatching:
    def test_same_doi_across_sources_is_duplicate(self):
        a = make_record()
        b = make_record(
            source="openalex",
            source_id="W999",
            title="Sparse Attention for Long Sequences",
            authors=("Someone Else",),
            doi="10.1234/ML.001",  # same DOI, different case
        )
        unique, duplicates = run([a, b])
        assert len(unique) == 1
        assert len(duplicates) == 1
        assert duplicates[0].matched_on == KEY_DOI
        assert duplicates[0].retained_source == "arxiv"
        assert duplicates[0].retained_source_id == "2101.00001"

    def test_different_dois_are_not_duplicates(self):
        other = make_record(
            source_id="2101.00002",
            title="An Entirely Different Paper",
            authors=("Alan Turing",),
            doi="10.1234/ml.002",
        )
        unique, duplicates = run([make_record(), other])
        assert len(unique) == 2
        assert not duplicates


class TestSourceScopedMatching:
    def test_same_source_and_id_is_duplicate_even_without_doi(self):
        a = make_record(doi=None)
        b = make_record(doi=None, title="A Different Title")
        unique, duplicates = run([a, b])
        assert len(unique) == 1
        assert duplicates[0].matched_on == KEY_SOURCE_ID

    def test_same_id_in_different_sources_is_not_a_duplicate(self):
        a = make_record(doi=None)
        b = make_record(
            doi=None,
            source="openalex",
            title="A Different Paper",
            authors=("Alan Turing",),
        )
        unique, duplicates = run([a, b])
        assert len(unique) == 2
        assert not duplicates


class TestFallbackMatching:
    def test_same_title_year_first_author_is_duplicate(self):
        a = make_record(doi=None)
        b = make_record(
            doi=None,
            source="openalex",
            source_id="W999",
            title="sparse attention mechanisms",  # case variation
            published="2021-06-01",  # same year
        )
        unique, duplicates = run([a, b])
        assert len(unique) == 1
        assert duplicates[0].matched_on == KEY_FALLBACK

    def test_different_year_is_not_a_duplicate(self):
        a = make_record(doi=None)
        b = make_record(
            doi=None,
            source="openalex",
            source_id="W999",
            published="2022-01-01",
        )
        unique, _ = run([a, b])
        assert len(unique) == 2

    def test_different_first_author_is_not_a_duplicate(self):
        a = make_record(doi=None)
        b = make_record(
            doi=None, source="openalex", source_id="W999", authors=("Alan Turing",)
        )
        unique, _ = run([a, b])
        assert len(unique) == 2

    def test_different_title_is_not_a_duplicate(self):
        a = make_record(doi=None)
        b = make_record(
            doi=None,
            source="openalex",
            source_id="W999",
            title="An Entirely Different Paper",
        )
        unique, _ = run([a, b])
        assert len(unique) == 2


class TestGeneral:
    def test_first_occurrence_is_retained(self):
        first = make_record(title="First Version")
        # Different source_id but shared DOI forces a duplicate; the first
        # record must be the one retained.
        second = make_record(source_id="2101.00002", title="Second Version")
        unique, duplicates = run([first, second])
        assert len(unique) == 1
        assert unique[0].title == "First Version"
        assert duplicates[0].duplicate.title == "Second Version"

    def test_input_order_is_preserved(self):
        records = [
            make_record(source_id="a", title="Paper A", doi="10.1234/a"),
            make_record(
                source_id="b", title="Paper B", authors=("Alan Turing",), doi="10.1234/b"
            ),
            make_record(
                source_id="c", title="Paper C", authors=("Katherine Johnson",),
                doi="10.1234/c",
            ),
        ]
        unique, _ = run(records)
        assert [r.source_id for r in unique] == ["a", "b", "c"]

    def test_empty_input(self):
        unique, duplicates = run([])
        assert unique == []
        assert duplicates == []

    def test_three_way_duplicate_keeps_one(self):
        a = make_record()
        b = make_record(source_id="x", source="other")
        c = make_record(source_id="y", source="third")
        unique, duplicates = run([a, b, c])
        assert len(unique) == 1
        assert len(duplicates) == 2
        for dup in duplicates:
            assert dup.retained_source == "arxiv"
            assert dup.retained_source_id == "2101.00001"
