"""Tests for condition extraction (verbatim spans, offsets, categories)."""

from __future__ import annotations

import pytest

from paperdna.conditions import extract_conditions
from paperdna.conditions.schema import CONDITION_CATEGORIES
from tests.conditions.conftest import make_claim


def values(claim, category: str) -> tuple[str, ...]:
    return tuple(m.value for m in claim.get(category))


class TestExtraction:
    def test_extracts_each_category(self):
        cases = [
            ("We find that accuracy improves on the GLUE benchmark.", "dataset", "GLUE benchmark"),
            ("We find that treatment reduces mortality in 200 patients.", "population", "200 patients"),
            ("We show that the BERT classifier improves F1 score.", "model", "the BERT classifier"),
            ("We find that accuracy improves in the low-resource setting.", "setting", "low-resource setting"),
            ("In a randomized controlled trial, we find a clear effect.", "design", "randomized controlled trial"),
        ]
        for sentence, category, expected in cases:
            conditions = extract_conditions(make_claim(sentence=sentence))
            assert expected in values(conditions, category), (
                f"{category}: expected {expected!r} in {values(conditions, category)}"
            )

    def test_metric_extraction(self):
        conditions = extract_conditions(
            make_claim(sentence="We find that F1 score improves markedly.")
        )
        assert values(conditions, "metric") == ("F1 score",)

    def test_offsets_are_verbatim(self):
        sentence = "We show that the BERT classifier improves F1 score on the SQuAD corpus."
        conditions = extract_conditions(make_claim(sentence=sentence))
        assert conditions.mentions, "expected at least one mention"
        for mention in conditions.mentions:
            assert sentence[mention.start : mention.end] == mention.value

    def test_values_preserve_verbatim_text(self):
        conditions = extract_conditions(
            make_claim(sentence="We find that Accuracy improves on the GLUE benchmark.")
        )
        # Casing/wording is never rewritten.
        assert "Accuracy" in values(conditions, "metric")
        assert "GLUE benchmark" in values(conditions, "dataset")

    def test_source_identity_is_copied(self):
        conditions = extract_conditions(
            make_claim(source="arxiv", source_id="2101.00001")
        )
        assert conditions.source == "arxiv"
        assert conditions.source_id == "2101.00001"
        for mention in conditions.mentions:
            assert mention.source == "arxiv"
            assert mention.source_id == "2101.00001"

    def test_blank_sentence_yields_no_conditions(self):
        conditions = extract_conditions(make_claim(sentence=""))
        assert conditions.mentions == ()
        for category in CONDITION_CATEGORIES:
            assert conditions.get(category) == ()

    def test_no_conditions_found_is_empty_not_fabricated(self):
        conditions = extract_conditions(
            make_claim(sentence="We find a clear and interesting effect here.")
        )
        assert conditions.mentions == ()

    def test_non_claim_input_raises(self):
        with pytest.raises(TypeError):
            extract_conditions("not a claim")  # type: ignore[arg-type]

    def test_unknown_category_raises(self):
        conditions = extract_conditions(make_claim())
        with pytest.raises(ValueError):
            conditions.get("bogus")


class TestAmbiguity:
    def test_generic_dataset_wording_is_flagged_ambiguous(self):
        conditions = extract_conditions(
            make_claim(sentence="We find that our dataset improves performance.")
        )
        mentions = conditions.get("dataset")
        assert mentions and all(m.ambiguous for m in mentions)

    def test_specific_dataset_wording_is_not_ambiguous(self):
        conditions = extract_conditions(
            make_claim(sentence="We find that accuracy improves on the GLUE benchmark.")
        )
        mentions = conditions.get("dataset")
        assert mentions and not any(m.ambiguous for m in mentions)

    def test_various_benchmarks_is_ambiguous(self):
        conditions = extract_conditions(
            make_claim(sentence="We find that our approach improves on various benchmarks.")
        )
        mentions = conditions.get("dataset")
        assert mentions and all(m.ambiguous for m in mentions)

    def test_generic_model_wording_is_flagged(self):
        conditions = extract_conditions(
            make_claim(sentence="We find that our model improves performance.")
        )
        mentions = conditions.get("model")
        assert mentions and all(m.ambiguous for m in mentions)


class TestOverlapResolution:
    def test_specific_rule_wins_over_shorter_known_name(self):
        conditions = extract_conditions(
            make_claim(sentence="We show that the BERT classifier improves results.")
        )
        # "BERT" (shorter) must not be emitted alongside "the BERT classifier".
        assert values(conditions, "model") == ("the BERT classifier",)

    def test_specific_rule_wins_over_shorter_dataset_name(self):
        conditions = extract_conditions(
            make_claim(sentence="We find that accuracy improves on the SQuAD corpus.")
        )
        assert values(conditions, "dataset") == ("SQuAD corpus",)

    def test_deterministic_on_repeated_runs(self):
        sentence = (
            "In a randomized controlled trial, our model improved F1 "
            "in a low-resource setting on the GLUE benchmark."
        )
        first = extract_conditions(make_claim(sentence=sentence))
        second = extract_conditions(make_claim(sentence=sentence))
        assert first == second

    def test_mentions_sorted_by_position(self):
        sentence = "We show that the BERT classifier improves F1 score on the SQuAD corpus."
        conditions = extract_conditions(make_claim(sentence=sentence))
        starts = [m.start for m in conditions.mentions]
        assert starts == sorted(starts)
