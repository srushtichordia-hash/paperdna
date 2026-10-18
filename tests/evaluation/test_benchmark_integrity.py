"""Structural integrity of the hand-labeled evaluation benchmarks.

These tests guard the DATA itself: loader validation, split hygiene,
label coverage, and the fixture's internal consistency (e.g. that
``expected_uncertain`` matches the documented uncertainty rule).
"""

from __future__ import annotations

import json

import pytest

from paperdna.eval.benchmark import (
    load_conditions_pairs,
    load_nli_pairs,
    load_retrieval_benchmark,
)
from paperdna.nli.schema import RELATIONSHIPS


class TestRetrievalBenchmark:
    def test_loads_and_is_self_contained(self):
        records, queries = load_retrieval_benchmark()
        assert len(records) >= 10
        assert len(queries) >= 12
        ids = {f"{r.source}:{r.source_id}" for r in records}
        for item in queries:
            assert set(item.relevant) <= ids

    def test_every_paper_is_labeled_relevant_at_least_once(self):
        records, queries = load_retrieval_benchmark()
        ids = {f"{r.source}:{r.source_id}" for r in records}
        covered = {pid for q in queries for pid in q.relevant}
        assert covered == ids

    def test_splits_are_disjoint_nonempty_and_sized(self):
        _, queries = load_retrieval_benchmark()
        dev = [q for q in queries if q.split == "dev"]
        test = [q for q in queries if q.split == "test"]
        assert dev and test
        assert not {q.query for q in dev} & {q.query for q in test}
        # Test is the reporting split: it should be the larger one.
        assert len(test) > len(dev)

    def test_anchor_queries_exist_in_dev(self):
        _, queries = load_retrieval_benchmark()
        anchors = [q for q in queries if q.anchor]
        assert anchors
        assert all(q.split == "dev" for q in anchors)


class TestNliBenchmark:
    def test_loads_with_expected_coverage(self):
        pairs = load_nli_pairs()
        assert len(pairs) >= 24
        names = [p.name for p in pairs]
        assert len(names) == len(set(names))

    def test_each_split_contains_every_relationship(self):
        pairs = load_nli_pairs()
        for split in ("dev", "test"):
            labels = {p.expected_label for p in pairs if p.split == split}
            assert labels == set(RELATIONSHIPS), split

    def test_scripted_scores_are_circular_but_valid(self):
        for pair in load_nli_pairs():
            assert set(pair.scripted_scores) == set(RELATIONSHIPS)
            total = sum(pair.scripted_scores.values())
            assert total == pytest.approx(1.0, abs=1e-9), pair.name

    def test_expected_uncertain_follows_the_documented_rule(self):
        # uncertain = max score < 0.6 OR hedges present (classify.py rule).
        for pair in load_nli_pairs():
            expected = (
                max(pair.scripted_scores.values()) < 0.6
                or bool(pair.premise.hedges)
                or bool(pair.hypothesis.hedges)
            )
            assert pair.expected_uncertain == expected, pair.name

    def test_pairs_keep_verbatim_identity(self):
        for pair in load_nli_pairs():
            assert pair.premise.source and pair.premise.source_id
            assert pair.hypothesis.source and pair.hypothesis.source_id
            assert pair.premise.sentence.strip() and pair.hypothesis.sentence.strip()


class TestConditionsBenchmark:
    def test_loads_all_pairs(self):
        pairs = load_conditions_pairs()
        assert len(pairs) >= 6
        assert len({p.name for p in pairs}) == len(pairs)

    def test_empty_side_is_preserved(self):
        # "incomplete_claim_side" exercises the empty-premise path.
        pairs = load_conditions_pairs()
        assert any(not p.premise.sentence for p in pairs)


class TestLoaderValidation:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_nli_pairs(tmp_path / "nope.json")

    def test_unknown_split_raises(self, tmp_path):
        path = tmp_path / "nli.json"
        path.write_text(json.dumps({
            "pairs": [{
                "name": "x",
                "premise": {"source": "s", "source_id": "1", "sentence": "A claim.", "hedges": []},
                "hypothesis": {"source": "s", "source_id": "2", "sentence": "Another claim.", "hedges": []},
                "expected_label": "neutral",
                "scripted_scores": {"entailment": 0.2, "contradiction": 0.2, "neutral": 0.6},
                "expected_uncertain": False,
                "split": "holdout",
            }]
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown split"):
            load_nli_pairs(path)

    def test_unknown_relevant_paper_raises(self, tmp_path):
        path = tmp_path / "retrieval.json"
        path.write_text(json.dumps({
            "papers": [{"id": "s:1", "title": "T", "abstract": "A."}],
            "queries": [{"query": "q", "relevant": ["s:999"], "split": "dev"}]
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown papers"):
            load_retrieval_benchmark(path)

    def test_uncovered_paper_raises(self, tmp_path):
        path = tmp_path / "retrieval.json"
        path.write_text(json.dumps({
            "papers": [
                {"id": "s:1", "title": "T", "abstract": "A."},
                {"id": "s:2", "title": "U", "abstract": "B."},
            ],
            "queries": [{"query": "q", "relevant": ["s:1"], "split": "dev"}]
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="never labeled relevant"):
            load_retrieval_benchmark(path)

    def test_duplicate_pair_name_raises(self, tmp_path):
        pair = {
            "name": "dup",
            "premise": {"source": "s", "source_id": "1", "sentence": "A claim.", "hedges": []},
            "hypothesis": {"source": "s", "source_id": "2", "sentence": "Another claim.", "hedges": []},
            "expected_label": "neutral",
            "scripted_scores": {"entailment": 0.2, "contradiction": 0.2, "neutral": 0.6},
            "expected_uncertain": False,
            "split": "dev",
        }
        path = tmp_path / "nli.json"
        path.write_text(json.dumps({"pairs": [pair, pair]}), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate pair name"):
            load_nli_pairs(path)
