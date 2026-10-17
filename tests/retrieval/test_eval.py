"""Tests for Precision@k and Recall@k, including documented edge cases."""

import json
from pathlib import Path

import pytest

from paperdna.corpus import ingest_records, load_jsonl
from paperdna.retrieval import PaperIndex, precision_at_k, recall_at_k, record_to_text
from tests.retrieval.conftest import FakeEmbedder

EVAL_PATH = Path(__file__).parent.parent / "fixtures" / "retrieval_eval.json"
CORPUS_PATH = Path(__file__).parent.parent / "fixtures" / "sample_corpus.jsonl"


class TestPrecisionAtK:
    def test_basic_case(self):
        # 2 of top-3 relevant -> 2/3
        assert precision_at_k(["a", "b", "c"], ["b", "c", "d"], 3) == pytest.approx(
            2 / 3
        )

    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], 2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_partial_k_slicing(self):
        # Only the first k retrieved count.
        assert precision_at_k(["x", "a"], ["a"], 1) == 0.0

    def test_fewer_than_k_retrieved_divides_by_k(self):
        # Documented convention: missing results count against the metric.
        assert precision_at_k(["a"], ["a", "b"], 3) == pytest.approx(1 / 3)

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], 3) == 0.0

    def test_empty_relevance_set(self):
        assert precision_at_k(["a", "b"], [], 2) == 0.0

    def test_both_empty(self):
        assert precision_at_k([], [], 2) == 0.0

    @pytest.mark.parametrize("bad_k", [0, -1, 2.5, True, "2"])
    def test_invalid_k_raises(self, bad_k):
        with pytest.raises(ValueError):
            precision_at_k(["a"], ["a"], bad_k)  # type: ignore[arg-type]


class TestRecallAtK:
    def test_basic_case(self):
        # 2 of 3 relevant found in top-3 -> 2/3
        assert recall_at_k(["a", "b", "x"], ["a", "b", "c"], 3) == pytest.approx(
            2 / 3
        )

    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 5) == 1.0

    def test_fewer_than_k_retrieved(self):
        assert recall_at_k(["a"], ["a", "b"], 3) == pytest.approx(0.5)

    def test_cutoff_respected(self):
        # "b" is relevant but ranked below k.
        assert recall_at_k(["x", "y", "b"], ["b"], 2) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["a"], 3) == 0.0

    def test_empty_relevance_set(self):
        assert recall_at_k(["a", "b"], [], 2) == 0.0

    @pytest.mark.parametrize("bad_k", [0, -3, 1.5, False, None])
    def test_invalid_k_raises(self, bad_k):
        with pytest.raises(ValueError):
            recall_at_k(["a"], ["a"], bad_k)  # type: ignore[arg-type]


class TestEvalFixture:
    """Run the hand-labeled sanity-check set through a deterministic fake.

    The fake maps each query to the exact embedding of its top paper when
    the query shares its distinctive tokens, making the expected winner
    deterministic without any model.
    """

    def load_eval(self) -> dict:
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    def test_fixture_shape(self):
        data = self.load_eval()
        assert len(data["queries"]) >= 3
        paper_ids = {paper["id"] for paper in data["papers"]}
        for item in data["queries"]:
            assert item["relevant"], "every sanity query needs a label"
            assert set(item["relevant"]) <= paper_ids

    def test_fixture_describes_itself_as_sanity_check(self):
        assert "sanity" in self.load_eval()["_comment"]

    def test_perfect_retrieval_scores_one(self):
        data = self.load_eval()
        embedder = FakeEmbedder()
        index = PaperIndex.build(
            ingest_records(load_jsonl(str(CORPUS_PATH))).papers, embedder
        )
        # Sanity: fixture paper ids correspond to fixture corpus records.
        corpus_ids = {f"{p.source}:{p.source_id}" for p in index.records}
        assert {p["id"] for p in data["papers"]} <= corpus_ids

        precisions = []
        for item in data["queries"]:
            results = index.search(item["query"], top_k=len(index.records))
            retrieved = [f"{r.record.source}:{r.record.source_id}" for r in results]
            precisions.append(precision_at_k(retrieved, item["relevant"], 1))
            assert recall_at_k(retrieved, item["relevant"], len(retrieved)) > 0
        # FakeEmbedder is token-overlap based; every labeled query shares
        # its distinctive tokens with its labeled paper, so P@1 = 1.0 here.
        assert precisions == [1.0] * len(precisions)

    def test_preprocessing_is_deterministic_on_fixture(self):
        data = self.load_eval()
        records = ingest_records(load_jsonl(str(CORPUS_PATH))).papers
        texts = [record_to_text(p) for p in records]
        assert texts == [record_to_text(p) for p in records]
        assert all(isinstance(t, str) and t for t in texts)
        assert len(data["papers"]) == 3
