"""Tests for condition comparison across claims."""

from __future__ import annotations

import pytest

from paperdna.conditions import compare_conditions, extract_conditions
from paperdna.conditions.schema import CONDITION_CATEGORIES, FIELD_STATUSES
from tests.conditions.conftest import make_claim


def extract(sentence: str, **overrides):
    return extract_conditions(make_claim(sentence=sentence, **overrides))


class TestComparisonStatuses:
    def test_matching_dataset_and_metric(self):
        premise = extract("We find that accuracy improves on the GLUE benchmark.")
        hypothesis = extract("We find that accuracy improves on the GLUE benchmark.")
        comparison = compare_conditions(premise, hypothesis)
        assert comparison.field("dataset").status == "matching"
        assert comparison.field("metric").status == "matching"

    def test_differing_dataset_and_metric(self):
        premise = extract("We show that accuracy improves on the SQuAD corpus.")
        hypothesis = extract("We show that F1 score improves on the ImageNet dataset.")
        comparison = compare_conditions(premise, hypothesis)
        assert comparison.field("dataset").status == "differing"
        assert comparison.field("metric").status == "differing"
        assert comparison.field("dataset").premise_values == ("SQuAD corpus",)
        assert comparison.field("dataset").hypothesis_values == ("ImageNet dataset",)

    def test_missing_is_not_differing(self):
        # Premise states a population; hypothesis says nothing about one.
        premise = extract("We find that treatment reduces mortality in 200 patients.")
        hypothesis = extract("We find that treatment reduces mortality.")
        comparison = compare_conditions(premise, hypothesis)
        field = comparison.field("population")
        assert field.status == "missing"
        assert field.status != "differing"
        assert field.premise_values == ("200 patients",)
        assert field.hypothesis_values == ()

    def test_missing_on_both_sides(self):
        premise = extract("We find a clear effect.")
        hypothesis = extract("We find a clear effect.")
        comparison = compare_conditions(premise, hypothesis)
        for category in CONDITION_CATEGORIES:
            assert comparison.field(category).status == "missing"

    def test_ambiguous_on_either_side_is_uncertain(self):
        premise = extract("We find that our dataset improves performance.")
        hypothesis = extract("We find that accuracy improves on the GLUE benchmark.")
        comparison = compare_conditions(premise, hypothesis)
        assert comparison.field("dataset").status == "uncertain"

    def test_ambiguous_on_both_sides_is_uncertain(self):
        premise = extract("We find that our approach improves on various benchmarks.")
        hypothesis = extract("We find that the method improves on the standard benchmark.")
        comparison = compare_conditions(premise, hypothesis)
        assert comparison.field("dataset").status == "uncertain"

    def test_values_are_deduplicated_within_field(self):
        premise = extract("We find that accuracy and accuracy improve results.")
        hypothesis = extract("We find that accuracy improves results.")
        comparison = compare_conditions(premise, hypothesis)
        field = comparison.field("metric")
        assert field.premise_values == ("accuracy",)
        assert field.status == "matching"

    def test_normalized_equality_ignores_case(self):
        premise = extract("We find that Accuracy improves results.")
        hypothesis = extract("We find that accuracy improves results.")
        comparison = compare_conditions(premise, hypothesis)
        assert comparison.field("metric").status == "matching"


class TestStatsAndShape:
    def test_stats_cover_all_categories(self):
        premise = extract(
            "In a randomized controlled trial, our model improved F1 "
            "in a low-resource setting on the GLUE benchmark."
        )
        hypothesis = extract("We find that accuracy improves.")
        comparison = compare_conditions(premise, hypothesis)
        assert sum(comparison.stats.values()) == len(CONDITION_CATEGORIES)
        assert set(comparison.stats) == set(FIELD_STATUSES)

    def test_fields_are_in_category_order(self):
        premise = extract("We find that accuracy improves.")
        hypothesis = extract("We find that accuracy improves.")
        comparison = compare_conditions(premise, hypothesis)
        assert tuple(f.category for f in comparison.fields) == CONDITION_CATEGORIES

    def test_summary_lists_each_status_group(self):
        premise = extract("We find that treatment reduces mortality in 200 patients.")
        hypothesis = extract("We find that accuracy improves.")
        comparison = compare_conditions(premise, hypothesis)
        assert "missing: " in comparison.summary
        assert "population" in comparison.summary

    def test_source_ids_are_propagated(self):
        premise = extract("We find that accuracy improves.", source_id="p1")
        hypothesis = extract("We find that accuracy improves.", source_id="h1")
        comparison = compare_conditions(premise, hypothesis)
        assert comparison.premise_source_id == "p1"
        assert comparison.hypothesis_source_id == "h1"

    def test_non_conditions_input_raises(self):
        with pytest.raises(TypeError):
            compare_conditions("a", "b")  # type: ignore[arg-type]


class TestHandLabeledFixture:
    """Sanity check against synthetic hand-labeled pairs (not a quality claim)."""

    def test_expected_statuses(self, fixture_pairs):
        for item in fixture_pairs:
            premise = extract_conditions(
                make_claim(
                    source=item["premise"]["source"],
                    source_id=item["premise"]["source_id"],
                    sentence=item["premise"]["sentence"],
                )
            )
            hypothesis = extract_conditions(
                make_claim(
                    source=item["hypothesis"]["source"],
                    source_id=item["hypothesis"]["source_id"],
                    sentence=item["hypothesis"]["sentence"],
                )
            )
            comparison = compare_conditions(premise, hypothesis)
            actual = {f.category: f.status for f in comparison.fields}
            assert actual == item["expected_statuses"], item["name"]
            assert sum(comparison.stats.values()) == len(CONDITION_CATEGORIES)
