"""Tests for PaperIndex building, dedup, and searching."""

import math

import pytest

from paperdna.retrieval import PaperIndex
from paperdna.retrieval.errors import InvalidVectorError
from paperdna.retrieval.index import SearchResult
from tests.retrieval.conftest import FakeEmbedder, StaticEmbedder, make_record


def test_build_indexes_all_unique_records(fake_embedder):
    records = [make_record(source_id=str(i)) for i in range(3)]
    index = PaperIndex.build(records, fake_embedder)
    assert index.stats == {"indexed": 3, "duplicate_ids_dropped": 0, "dimension": 8}
    assert index.records == tuple(records)


def test_build_embeds_corpus_exactly_once(fake_embedder):
    records = [make_record(source_id=str(i)) for i in range(4)]
    PaperIndex.build(records, fake_embedder)
    assert len(fake_embedder.embed_calls) == 1
    assert len(fake_embedder.embed_calls[0]) == 4


def test_duplicate_ids_dropped_and_counted():
    embedder = FakeEmbedder()
    kept = make_record()
    duplicate = make_record()  # same (source, source_id)
    index = PaperIndex.build([kept, duplicate], embedder)
    assert index.stats["indexed"] == 1
    assert index.stats["duplicate_ids_dropped"] == 1
    assert index.records[0] is kept  # original object, unmutated


def test_source_identity_is_case_insensitive():
    embedder = FakeEmbedder()
    upper = make_record(source="ARXIV")
    lower = make_record(source="arxiv")
    index = PaperIndex.build([upper, lower], embedder)
    assert index.stats["indexed"] == 1
    assert index.stats["duplicate_ids_dropped"] == 1
    assert index.records[0] is upper


def test_same_source_id_in_different_sources_kept():
    embedder = FakeEmbedder()
    a = make_record(source="arxiv")
    b = make_record(source="openalex")
    index = PaperIndex.build([a, b], embedder)
    assert index.stats["indexed"] == 2
    assert index.stats["duplicate_ids_dropped"] == 0


def test_empty_corpus_makes_no_embed_calls():
    embedder = FakeEmbedder()
    index = PaperIndex.build([], embedder)
    assert index.stats == {"indexed": 0, "duplicate_ids_dropped": 0, "dimension": 0}
    assert embedder.embed_calls == []
    assert index.search("anything") == []


def test_non_record_input_rejected(fake_embedder):
    with pytest.raises(TypeError):
        PaperIndex.build([{"title": "not a record"}], fake_embedder)  # type: ignore[list-item]


class TestSearch:
    RECORDS = [
        make_record(
            source_id="1",
            title="Attention Mechanisms",
            abstract="We study attention mechanisms for documents.",
        ),
        make_record(
            source_id="2",
            title="Gradient Stabilization",
            abstract="A study of gradient stabilization in networks.",
        ),
        make_record(
            source_id="3",
            title="Retrieval Benchmarks",
            abstract="We benchmark retrieval pipelines with benchmarks.",
        ),
    ]

    def build_index(self) -> tuple[PaperIndex, FakeEmbedder]:
        embedder = FakeEmbedder()
        return PaperIndex.build(self.RECORDS, embedder), embedder

    def test_returns_search_results_with_records_and_scores(self):
        index, _ = self.build_index()
        results = index.search("attention mechanisms for documents", top_k=3)
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].record is self.RECORDS[0]
        assert results[0].score > results[1].score

    def test_ranking_is_descending(self):
        index, _ = self.build_index()
        results = index.search("attention mechanisms for documents", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates(self):
        index, _ = self.build_index()
        assert len(index.search("attention", top_k=2)) == 2
        assert len(index.search("attention", top_k=1)) == 1

    def test_top_k_larger_than_corpus_is_fine(self):
        index, _ = self.build_index()
        assert len(index.search("attention", top_k=99)) == 3

    def test_repeated_searches_embed_query_only(self):
        index, embedder = self.build_index()
        total_calls_before = len(embedder.embed_calls)
        index.search("attention mechanisms", top_k=3)
        index.search("gradient stabilization", top_k=3)
        new_calls = embedder.embed_calls[total_calls_before:]
        assert new_calls == [["attention mechanisms"], ["gradient stabilization"]]

    def test_empty_query_raises(self):
        index, _ = self.build_index()
        for bad in ("", "   ", "\n\t"):
            with pytest.raises(ValueError):
                index.search(bad)

    @pytest.mark.parametrize("bad_k", [0, -1, 2.5, True, "3"])
    def test_invalid_top_k_raises(self, bad_k):
        index, _ = self.build_index()
        with pytest.raises(ValueError):
            index.search("attention", top_k=bad_k)  # type: ignore[arg-type]

    def test_query_dimension_mismatch_raises(self):
        embedder = StaticEmbedder({"q": [1.0, 0.0, 0.0], "doc\n\ndoc": [1.0, 0.0, 0.0]})
        index = PaperIndex.build([make_record(title="doc", abstract="doc")], embedder)
        embedder.vectors["q"] = [1.0, 0.0]  # shrink after build
        with pytest.raises(InvalidVectorError):
            index.search("q")

    def test_non_finite_query_vector_raises(self):
        embedder = StaticEmbedder(
            {"T\n\nA": [1.0, 0.0], "q": [math.nan, 1.0]}, dimension=2
        )
        index = PaperIndex.build([make_record(title="T", abstract="A")], embedder)
        with pytest.raises(InvalidVectorError):
            index.search("q")

    def test_zero_norm_query_vector_raises(self):
        embedder = StaticEmbedder({"T\n\nA": [1.0, 0.0], "q": [0.0, 0.0]}, dimension=2)
        index = PaperIndex.build([make_record(title="T", abstract="A")], embedder)
        with pytest.raises(InvalidVectorError):
            index.search("q")


class TestBuildValidation:
    def test_wrong_vector_count_raises(self):
        class ShortEmbedder(FakeEmbedder):
            def embed(self, texts):
                self.embed_calls.append(list(texts))
                return [[0.1] * 8]  # too few

        with pytest.raises(InvalidVectorError):
            PaperIndex.build(
                [make_record(source_id="1"), make_record(source_id="2")], ShortEmbedder()
            )

    def test_mismatched_corpus_dimensions_raise(self):
        class RaggedEmbedder(FakeEmbedder):
            def embed(self, texts):
                self.embed_calls.append(list(texts))
                vectors = [[0.1] * 8 for _ in texts]
                vectors[-1] = [0.1] * 4
                return vectors

        with pytest.raises(InvalidVectorError):
            PaperIndex.build(
                [make_record(source_id="1"), make_record(source_id="2")], RaggedEmbedder()
            )

    def test_non_finite_corpus_vector_raises(self):
        class NanEmbedder(FakeEmbedder):
            def embed(self, texts):
                self.embed_calls.append(list(texts))
                vectors = [[0.1] * 8 for _ in texts]
                vectors[0] = [math.nan] * 8
                return vectors

        with pytest.raises(InvalidVectorError):
            PaperIndex.build([make_record()], NanEmbedder())

    def test_zero_norm_corpus_vector_raises(self):
        class ZeroEmbedder(FakeEmbedder):
            def embed(self, texts):
                self.embed_calls.append(list(texts))
                return [[0.0] * 8 for _ in texts]

        with pytest.raises(InvalidVectorError):
            PaperIndex.build([make_record()], ZeroEmbedder())

    def test_declared_dimension_is_enforced(self):
        class LyingEmbedder(FakeEmbedder):
            dimension = 99

        with pytest.raises(InvalidVectorError):
            PaperIndex.build([make_record()], LyingEmbedder())
