"""Tests for Precision@k and Recall@k, including documented edge cases."""

import json
from pathlib import Path

import pytest

from paperdna.corpus import ingest_records, load_jsonl
from paperdna.eval import evaluate_retrieval, lexical_rank, load_retrieval_benchmark
from paperdna.retrieval import precision_at_k, recall_at_k, record_to_text

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
    """Structural and end-to-end checks on the labeled benchmark.

    The benchmark is self-contained (papers + queries in one file) and
    measured with the app's real lexical fallback; anchor queries share
    distinctive tokens with their labeled paper, so token-overlap
    ranking must put that paper first.
    """

    def load_eval(self) -> dict:
        return json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    def test_fixture_shape(self):
        data = self.load_eval()
        assert len(data["papers"]) >= 10
        assert len(data["queries"]) >= 12
        paper_ids = {paper["id"] for paper in data["papers"]}
        for item in data["queries"]:
            assert item["relevant"], "every sanity query needs a label"
            assert set(item["relevant"]) <= paper_ids
            assert item["split"] in ("dev", "test")

    def test_fixture_describes_itself_as_sanity_check(self):
        assert "sanity" in self.load_eval()["_comment"]

    def test_splits_are_disjoint_and_nonempty(self):
        data = self.load_eval()
        by_split: dict[str, list[str]] = {}
        for item in data["queries"]:
            by_split.setdefault(item["split"], []).append(item["query"])
        assert by_split.get("dev") and by_split.get("test")
        assert not set(by_split["dev"]) & set(by_split["test"])

    def test_benchmark_is_self_contained(self):
        records, queries = load_retrieval_benchmark(EVAL_PATH)
        data = self.load_eval()
        loaded_ids = {f"{r.source}:{r.source_id}" for r in records}
        assert loaded_ids == {p["id"] for p in data["papers"]}
        assert len(queries) == len(data["queries"])

    def test_anchor_queries_rank_their_paper_first(self):
        records, queries = load_retrieval_benchmark(EVAL_PATH)
        anchors = [q for q in queries if q.anchor]
        assert anchors, "the benchmark must keep token-anchored sanity queries"
        for item in anchors:
            retrieved = lexical_rank(item.query, records)
            assert precision_at_k(retrieved, item.relevant, 1) == 1.0, item.query

    def test_every_query_retrieves_a_relevant_paper(self):
        records, queries = load_retrieval_benchmark(EVAL_PATH)
        for item in queries:
            retrieved = lexical_rank(item.query, records)
            assert retrieved, f"lexical baseline returned nothing for {item.query!r}"
            assert (
                recall_at_k(retrieved, item.relevant, len(retrieved)) > 0
            ), item.query

    def test_lexical_report_is_deterministic_and_bounded(self):
        records, queries = load_retrieval_benchmark(EVAL_PATH)
        run = evaluate_retrieval(
            records, queries, lexical_rank, system="lexical", ranking_kind="lexical"
        )
        again = evaluate_retrieval(
            records, queries, lexical_rank, system="lexical", ranking_kind="lexical"
        )
        assert run == again
        assert run["ranking_kind"] == "lexical"
        for row in run["per_query"]:
            for value in row["metrics"].values():
                assert 0.0 <= value <= 1.0
        n = len(run["per_query"])
        for key, value in run["aggregate_mean"].items():
            mean = sum(r["metrics"][key] for r in run["per_query"]) / n
            assert value == pytest.approx(mean, abs=1e-6)

    def test_preprocessing_is_deterministic_on_fixture(self):
        records = ingest_records(load_jsonl(str(CORPUS_PATH))).papers
        texts = [record_to_text(p) for p in records]
        assert texts == [record_to_text(p) for p in records]
        assert all(isinstance(t, str) and t for t in texts)
