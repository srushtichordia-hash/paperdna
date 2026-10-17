"""Tests for the labeled fixture pipeline, metrics, and lazy loading."""

import importlib.util
import sys

import pytest

import paperdna.nli
from paperdna.nli import (
    ClaimPair,
    NliUnavailableError,
    CrossEncoderNliClassifier,
    DEFAULT_NLI_MODEL,
    classification_report,
    classify_pairs,
)
from tests.nli.conftest import FakeNliClassifier, claim_from


def scripted_classifier(fixture_pairs) -> FakeNliClassifier:
    return FakeNliClassifier(
        script={
            (pair["premise"]["sentence"], pair["hypothesis"]["sentence"]): pair[
                "scripted_scores"
            ]
            for pair in fixture_pairs
        }
    )


class TestFixturePipeline:
    def test_fixture_shape(self, fixture_pairs):
        assert len(fixture_pairs) >= 8
        names = [pair["name"] for pair in fixture_pairs]
        assert len(names) == len(set(names))
        expected = {"entailment", "contradiction", "neutral"}
        assert {pair["expected_label"] for pair in fixture_pairs} == expected

    def test_fixture_describes_itself_as_sanity_check(self):
        from tests.nli.conftest import NLI_FIXTURE

        assert "sanity-check" in NLI_FIXTURE.read_text(encoding="utf-8")

    def test_pipeline_matches_fixture_expectations(self, fixture_pairs):
        pairs = [
            ClaimPair(
                premise=claim_from(item["premise"]),
                hypothesis=claim_from(item["hypothesis"]),
            )
            for item in fixture_pairs
        ]
        results = classify_pairs(pairs, scripted_classifier(fixture_pairs))

        for item, result in zip(fixture_pairs, results):
            assert result.label == item["expected_label"], item["name"]
            assert result.uncertain == item["expected_uncertain"], item["name"]
            assert result.premise_source == item["premise"]["source"]
            assert result.premise_source_id == item["premise"]["source_id"]
            assert result.premise_text == item["premise"]["sentence"]
            assert result.hypothesis_source_id == item["hypothesis"]["source_id"]
            assert result.hypothesis_text == item["hypothesis"]["sentence"]
            assert sum(result.scores.values()) == pytest.approx(1.0)

    def test_metrics_over_fixture(self, fixture_pairs):
        pairs = [
            ClaimPair(
                premise=claim_from(item["premise"]),
                hypothesis=claim_from(item["hypothesis"]),
            )
            for item in fixture_pairs
        ]
        results = classify_pairs(pairs, scripted_classifier(fixture_pairs))
        report = classification_report(
            [item["expected_label"] for item in fixture_pairs],
            [result.label for result in results],
        )
        assert report["macro_f1"] == pytest.approx(1.0)
        assert report["accuracy"] == pytest.approx(1.0)


class TestClassificationReport:
    def test_perfect_predictions(self):
        labels = ["entailment", "contradiction", "neutral"]
        report = classification_report(labels, labels)
        assert report["macro_f1"] == pytest.approx(1.0)
        assert report["accuracy"] == pytest.approx(1.0)
        for stats in report["per_class"].values():
            assert stats["precision"] == pytest.approx(1.0)
            assert stats["recall"] == pytest.approx(1.0)

    def test_hand_computed_values(self):
        # y_true: e c n e | y_pred: e c c e
        report = classification_report(
            ["entailment", "contradiction", "neutral", "entailment"],
            ["entailment", "contradiction", "contradiction", "entailment"],
        )
        per = report["per_class"]
        assert per["entailment"]["f1"] == pytest.approx(1.0)
        assert per["contradiction"]["precision"] == pytest.approx(0.5)
        assert per["contradiction"]["recall"] == pytest.approx(1.0)
        assert per["contradiction"]["f1"] == pytest.approx(2 / 3)
        assert per["neutral"]["f1"] == pytest.approx(0.0)
        # (1 + 2/3 + 0) / 3 = 5/9
        assert report["macro_f1"] == pytest.approx(5 / 9)
        assert report["accuracy"] == pytest.approx(0.75)
        assert report["total"] == 4

    def test_empty_input(self):
        report = classification_report([], [])
        assert report["macro_f1"] == 0.0
        assert report["accuracy"] == 0.0
        assert report["total"] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            classification_report(["entailment"], ["entailment", "neutral"])


class TestLazyLoading:
    def test_importing_nli_does_not_import_heavy_deps(self):
        assert "sentence_transformers" not in sys.modules
        assert "torch" not in sys.modules

    def test_default_model_is_the_researched_choice(self):
        assert DEFAULT_NLI_MODEL == "cross-encoder/nli-deberta-v3-base"

    @pytest.mark.skipif(
        importlib.util.find_spec("sentence_transformers") is not None,
        reason="sentence-transformers installed; missing-dep test not applicable",
    )
    def test_missing_dependency_raises_helpful_error(self):
        with pytest.raises(NliUnavailableError) as excinfo:
            CrossEncoderNliClassifier()
        assert 'pip install -e ".[nli]"' in str(excinfo.value)

    @pytest.mark.slow
    def test_real_model_smoke(self):
        """Requires the nli extra and a model download; skipped by default."""
        pytest.importorskip("sentence_transformers")
        classifier = CrossEncoderNliClassifier()
        results = classifier.score_pairs(
            [("A man is eating pizza", "A man eats something")]
        )
        assert len(results) == 1
        assert set(results[0].scores) == {"entailment", "contradiction", "neutral"}
        assert sum(results[0].scores.values()) == pytest.approx(1.0)
