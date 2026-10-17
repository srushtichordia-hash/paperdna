"""Tests for pair classification: labels, uncertainty, validation."""

import pytest

from paperdna.nli import (
    ClaimPair,
    InvalidScoresError,
    PairClassification,
    RawPairScores,
    classify_pair,
    classify_pairs,
)
from tests.nli.conftest import FakeNliClassifier, claim_from, make_claim


class StubClassifier:
    """Classifier with a fixed return value, for validation tests."""

    name = "stub"
    max_length = 512

    def __init__(self, result):
        self.result = result

    def score_pairs(self, pairs):
        return [self.result for _ in pairs]


GOOD_SCORES = {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}


class TestLabels:
    @pytest.mark.parametrize(
        "scores,label",
        [
            ({"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}, "entailment"),
            ({"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05}, "contradiction"),
            ({"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9}, "neutral"),
        ],
    )
    def test_all_three_labels(self, scores, label):
        classifier = FakeNliClassifier(default_scores=scores)
        result = classify_pair(make_claim(), make_claim(), classifier)
        assert result.label == label
        assert result.confidence == pytest.approx(max(scores.values()))
        assert result.scores == scores

    def test_ties_break_deterministically_by_relationship_order(self):
        tied = {"entailment": 1 / 3, "contradiction": 1 / 3, "neutral": 1 / 3}
        classifier = FakeNliClassifier(default_scores=tied)
        result = classify_pair(make_claim(), make_claim(), classifier)
        assert result.label == "entailment"  # first in RELATIONSHIPS order

    def test_classification_is_a_pair_relationship_not_a_verdict(self):
        classifier = FakeNliClassifier()
        result = classify_pair(make_claim(), make_claim(), classifier)
        assert isinstance(result, PairClassification)


class TestAttribution:
    def test_both_sources_and_texts_preserved(self):
        premise = make_claim(source="arxiv", source_id="a1", sentence="Premise text.")
        hypothesis = make_claim(
            source="openalex", source_id="b9", sentence="Hypothesis text."
        )
        result = classify_pair(premise, hypothesis, FakeNliClassifier())
        assert result.premise_source == "arxiv"
        assert result.premise_source_id == "a1"
        assert result.premise_text == "Premise text."
        assert result.hypothesis_source == "openalex"
        assert result.hypothesis_source_id == "b9"
        assert result.hypothesis_text == "Hypothesis text."

    def test_model_name_recorded(self):
        result = classify_pair(make_claim(), make_claim(), FakeNliClassifier())
        assert result.model == "fake-nli"

    def test_premise_and_hypothesis_sent_to_classifier_in_order(self):
        classifier = FakeNliClassifier()
        premise = make_claim(sentence="Premise one.")
        hypothesis = make_claim(sentence="Hypothesis two.")
        classify_pair(premise, hypothesis, classifier)
        assert classifier.calls == [[("Premise one.", "Hypothesis two.")]]


class TestUncertainty:
    def test_high_confidence_not_uncertain(self):
        classifier = FakeNliClassifier(default_scores=GOOD_SCORES)
        result = classify_pair(make_claim(), make_claim(), classifier)
        assert result.confidence >= 0.6
        assert not result.uncertain

    def test_low_confidence_marks_uncertain_but_keeps_label(self):
        scores = {"entailment": 0.42, "contradiction": 0.31, "neutral": 0.27}
        classifier = FakeNliClassifier(default_scores=scores)
        result = classify_pair(make_claim(), make_claim(), classifier)
        assert result.label == "entailment"  # opinion preserved
        assert result.uncertain  # but flagged for review, not stated as fact

    def test_hedged_premise_marks_uncertain_despite_high_confidence(self):
        classifier = FakeNliClassifier(default_scores=GOOD_SCORES)
        hedged = make_claim(hedges=("may",))
        result = classify_pair(hedged, make_claim(), classifier)
        assert result.hedged
        assert result.uncertain

    def test_hedged_hypothesis_marks_uncertain(self):
        classifier = FakeNliClassifier(default_scores=GOOD_SCORES)
        result = classify_pair(make_claim(), make_claim(hedges=("suggests",)), classifier)
        assert result.hedged and result.uncertain

    def test_threshold_boundary_is_inclusive(self):
        scores = {"entailment": 0.6, "contradiction": 0.2, "neutral": 0.2}
        classifier = FakeNliClassifier(default_scores=scores)
        result = classify_pair(make_claim(), make_claim(), classifier, threshold=0.6)
        assert not result.uncertain  # exactly at threshold -> confident

    def test_custom_threshold_respected(self):
        classifier = FakeNliClassifier(default_scores=GOOD_SCORES)
        result = classify_pair(make_claim(), make_claim(), classifier, threshold=0.95)
        assert result.uncertain

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_threshold_raises(self, bad):
        with pytest.raises(ValueError):
            classify_pair(make_claim(), make_claim(), FakeNliClassifier(), threshold=bad)


class TestTruncation:
    def test_truncated_flag_propagated(self):
        class Truncating(FakeNliClassifier):
            def score_pairs(self, pairs):
                self.calls.append(list(pairs))
                return [RawPairScores(scores=self.default_scores, truncated=True)]

        result = classify_pair(make_claim(), make_claim(), Truncating())
        assert result.truncated


class TestMalformedInput:
    def test_non_claim_raises_type_error(self):
        with pytest.raises(TypeError):
            classify_pair({"sentence": "x"}, make_claim(), FakeNliClassifier())  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            classify_pair(make_claim(), None, FakeNliClassifier())  # type: ignore[arg-type]

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_blank_claim_text_raises_value_error(self, blank):
        with pytest.raises(ValueError):
            classify_pair(make_claim(sentence=blank), make_claim(), FakeNliClassifier())

    def test_classifier_returning_wrong_count_raises(self):
        class Short(StubClassifier):
            def score_pairs(self, pairs):
                return []

        with pytest.raises(IndexError):
            classify_pair(make_claim(), make_claim(), Short(GOOD_SCORES))

    @pytest.mark.parametrize(
        "bad_scores",
        [
            {"entailment": 0.5, "neutral": 0.5},  # missing key
            {"entailment": 0.5, "contradiction": 0.3, "neutral": 0.1, "extra": 0.1},
            {"entailment": float("nan"), "contradiction": 0.5, "neutral": 0.5},
            {"entailment": float("inf"), "contradiction": 0.0, "neutral": 0.0},
            {"entailment": 1.5, "contradiction": 0.0, "neutral": 0.0},
            {"entailment": 0.9, "contradiction": 0.9, "neutral": 0.9},  # sums to 2.7
        ],
    )
    def test_invalid_scores_raise(self, bad_scores):
        classifier = StubClassifier(RawPairScores(scores=bad_scores))
        with pytest.raises(InvalidScoresError):
            classify_pair(make_claim(), make_claim(), classifier)

    def test_non_dict_scores_raise(self):
        classifier = StubClassifier(RawPairScores(scores=[0.1, 0.2, 0.7]))  # type: ignore[arg-type]
        with pytest.raises(InvalidScoresError):
            classify_pair(make_claim(), make_claim(), classifier)


class TestClassifyPairs:
    def test_preserves_order_and_pairs_with_fixture(self):
        pairs = [
            ClaimPair(premise=make_claim(sentence="P1."), hypothesis=make_claim(sentence="H1.")),
            ClaimPair(premise=make_claim(sentence="P2."), hypothesis=make_claim(sentence="H2.")),
        ]
        classifier = FakeNliClassifier(
            script={
                ("P1.", "H1."): GOOD_SCORES,
                ("P2.", "H2."): {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1},
            }
        )
        results = classify_pairs(pairs, classifier)
        assert [r.label for r in results] == ["entailment", "contradiction"]
        # One call per pair, preserving order within the pipeline.
        assert classifier.calls == [[("P1.", "H1.")], [("P2.", "H2.")]]
