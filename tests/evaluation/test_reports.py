"""Correctness of the evaluation reports' metric computation.

Uses tiny synthetic inputs with known expected values so the metric
math is verified independently of the real benchmarks.
"""

from __future__ import annotations

import dataclasses

import pytest

from paperdna.corpus.schema import PaperRecord
from paperdna.eval import evaluate_conditions, evaluate_nli, evaluate_retrieval
from paperdna.eval.baselines import OverlapNliClassifier, ScriptedNliClassifier
from paperdna.eval.benchmark import RetrievalQuery, load_conditions_pairs


def _record(source_id: str, title: str, abstract: str) -> PaperRecord:
    return PaperRecord(
        source="toy", source_id=source_id, title=title, abstract=abstract,
        authors=("Toy Author",),
    )


TOY_RECORDS = [
    _record("p1", "Attention sparsity", "Sparse attention improves long documents."),
    _record("p2", "Gradient clipping", "Clipping stabilizes deep network training."),
    _record("p3", "Retrieval benchmarks", "Retrieval augmentation for question answering."),
]

TOY_QUERIES = [
    RetrievalQuery("sparse attention", ("toy:p1",), "dev", False),
    RetrievalQuery("stabilizing training", ("toy:p2", "toy:p3"), "test", False),
]


def _fixed_rank(order_map):
    def rank_fn(query: str, records) -> list[str]:
        return order_map[query]
    return rank_fn


class TestRetrievalReport:
    def test_metric_math_on_fixed_rankings(self):
        order = {
            "sparse attention": ["toy:p1", "toy:p2", "toy:p3"],
            "stabilizing training": ["toy:p3", "toy:p1", "toy:p2"],
        }
        report = evaluate_retrieval(
            TOY_RECORDS, TOY_QUERIES, _fixed_rank(order),
            system="fixed", ranking_kind="lexical", ks=(1, 2, 3),
        )
        rows = {r["query"]: r for r in report["per_query"]}
        # q1: relevant={p1}; top-1 hit -> P@1=1, R@1=1
        assert rows["sparse attention"]["metrics"]["precision@1"] == 1.0
        assert rows["sparse attention"]["metrics"]["recall@1"] == 1.0
        # q2: relevant={p2,p3}; top-1 is p3 -> P@1=1/1, R@1=1/2
        q2 = rows["stabilizing training"]["metrics"]
        assert q2["precision@1"] == 1.0
        assert q2["recall@1"] == pytest.approx(0.5)
        # q2 @2: retrieved [p3,p1]; 1 relevant of 2 -> P@2=0.5, R@2=0.5
        assert q2["precision@2"] == pytest.approx(0.5)
        assert q2["recall@2"] == pytest.approx(0.5)

    def test_short_results_divide_by_k(self):
        order = {"sparse attention": ["toy:p1"], "stabilizing training": ["toy:p1"]}
        report = evaluate_retrieval(
            TOY_RECORDS, TOY_QUERIES, _fixed_rank(order),
            system="fixed", ranking_kind="lexical", ks=(3,),
        )
        rows = {r["query"]: r for r in report["per_query"]}
        # Documented convention: 1 retrieved of k=3 -> 1/3.
        assert rows["sparse attention"]["metrics"]["precision@3"] == pytest.approx(1 / 3)

    def test_aggregate_is_mean_of_per_query(self):
        order = {"sparse attention": ["toy:p1"], "stabilizing training": ["toy:p2"]}
        report = evaluate_retrieval(
            TOY_RECORDS, TOY_QUERIES, _fixed_rank(order),
            system="fixed", ranking_kind="lexical", ks=(1,),
        )
        per = [r["metrics"]["precision@1"] for r in report["per_query"]]
        assert report["aggregate_mean"]["precision@1"] == pytest.approx(sum(per) / len(per))

    def test_ranking_kind_is_enforced(self):
        with pytest.raises(ValueError, match="ranking_kind"):
            evaluate_retrieval(
                TOY_RECORDS, TOY_QUERIES, _fixed_rank({}), system="x", ranking_kind="fancy"
            )

    def test_invalid_ks_rejected(self):
        with pytest.raises(ValueError, match="ks"):
            evaluate_retrieval(
                TOY_RECORDS, TOY_QUERIES, _fixed_rank({}),
                system="x", ranking_kind="lexical", ks=(0,),
            )


class TestNliReport:
    def test_overlap_baseline_predictions_are_text_only(self):
        # Identical sentences must score as pure entailment regardless of
        # any fixture: the classifier sees ONLY the two strings.
        clf = OverlapNliClassifier()
        scores = clf.score_pairs([("exact same words here", "exact same words here")])[0]
        assert scores.scores["entailment"] == pytest.approx(1.0)

    def test_overlap_baseline_negation_rule(self):
        clf = OverlapNliClassifier()
        scores = clf.score_pairs([
            ("the model did not improve accuracy", "the model improves accuracy")
        ])[0]
        assert scores.scores["contradiction"] >= 0.5

    def test_report_structure_from_actual_predictions(self):
        from paperdna.eval.benchmark import load_nli_pairs

        pairs = load_nli_pairs()[:6]
        report = evaluate_nli(pairs, OverlapNliClassifier())
        assert report["circular"] is False
        assert report["overall"]["total"] == 6
        assert 0.0 <= report["overall"]["macro_f1"] <= 1.0
        assert len(report["per_pair"]) == 6
        for row in report["per_pair"]:
            assert row["gold"] != row["predicted"] or row["match"]
            assert 0.0 <= row["confidence"] <= 1.0

    def test_scripted_system_is_flagged_circular(self):
        from paperdna.eval.benchmark import load_nli_pairs

        pairs = load_nli_pairs()
        script = {(p.premise.sentence, p.hypothesis.sentence): p.scripted_scores for p in pairs}
        report = evaluate_nli(pairs, ScriptedNliClassifier(script), circular=True)
        assert report["circular"] is True
        assert report["circular_note"]
        # Circular agreement is 1.0 by construction — plumbing only.
        assert report["overall"]["accuracy"] == 1.0

    def test_per_split_reports_cover_both_splits(self):
        from paperdna.eval.benchmark import load_nli_pairs

        report = evaluate_nli(load_nli_pairs(), OverlapNliClassifier())
        assert set(report["per_split"]) == {"dev", "test"}


class TestConditionsReport:
    def test_rule_baseline_matches_all_labeled_pairs(self):
        report = evaluate_conditions(load_conditions_pairs())
        assert report["kind"] == "correctness_check_not_quality_score"
        assert report["agreement"] == 1.0
        assert report["n_pairs_matching"] == report["n_pairs"]

    def test_wrong_expectation_is_detected(self):
        pairs = load_conditions_pairs()
        corrupted = [
            dataclasses.replace(
                p, expected_statuses={**p.expected_statuses, "dataset": "differing"}
            )
            if p is pairs[0] else p
            for p in pairs
        ]
        report = evaluate_conditions(corrupted)
        assert report["n_pairs_matching"] == report["n_pairs"] - 1
        first = report["per_pair"][0]
        assert not first["match"]
