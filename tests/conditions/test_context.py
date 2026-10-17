"""Tests for exposing condition differences as NLI context (never override)."""

from __future__ import annotations

import pytest

from paperdna.conditions import compare_conditions, contextualize_nli, extract_conditions
from tests.conditions.conftest import make_claim, make_classification


def compare(premise_sentence: str, hypothesis_sentence: str):
    premise = extract_conditions(make_claim(sentence=premise_sentence))
    hypothesis = extract_conditions(
        make_claim(sentence=hypothesis_sentence, source="fixture")
    )
    return compare_conditions(premise, hypothesis)


class TestContextualize:
    def test_contradiction_label_is_preserved(self):
        classification = make_classification(label="contradiction", confidence=0.91)
        comparison = compare(
            "We show that accuracy improves on the SQuAD corpus.",
            "We show that accuracy improves on the ImageNet dataset.",
        )
        result = contextualize_nli(classification, comparison)
        assert result.classification is classification
        assert result.classification.label == "contradiction"
        assert result.classification.confidence == 0.91
        assert result.classification.scores == classification.scores

    def test_condition_difference_is_context_not_downgrade(self):
        classification = make_classification(label="contradiction")
        comparison = compare(
            "We show that accuracy improves on the SQuAD corpus.",
            "We show that accuracy improves on the ImageNet dataset.",
        )
        result = contextualize_nli(classification, comparison)
        # The label must remain a contradiction regardless of conditions.
        assert result.classification.label == "contradiction"
        differing_notes = [n for n in result.notes if "differ" in n]
        assert differing_notes
        # The note must frame the difference as interpretation-required
        # context, not as an automatic explanation.
        assert "interpretation" in differing_notes[0]

    def test_missing_conditions_are_absence_of_information(self):
        classification = make_classification(label="entailment")
        comparison = compare(
            "We find that treatment reduces mortality in 200 patients.",
            "We find that treatment reduces mortality.",
        )
        result = contextualize_nli(classification, comparison)
        missing_notes = [n for n in result.notes if "missing" in n.lower()]
        assert missing_notes
        assert "not evidence" in missing_notes[0]
        assert result.classification.label == "entailment"

    def test_all_matching_conditions_note(self):
        classification = make_classification(label="neutral")
        comparison = compare(
            "We find that accuracy improves on the GLUE benchmark.",
            "We find that accuracy improves on the GLUE benchmark.",
        )
        result = contextualize_nli(classification, comparison)
        matching_notes = [n for n in result.notes if "match" in n.lower()]
        assert matching_notes
        assert "does not prove" in matching_notes[0]

    def test_partial_matching_conditions_note(self):
        classification = make_classification(label="contradiction")
        comparison = compare(
            "In a randomized controlled trial, our model improved F1 "
            "in a low-resource setting.",
            "In a randomized controlled trial, the transformer model "
            "gains F1 in a low-resource setting.",
        )
        result = contextualize_nli(classification, comparison)
        assert any("Matching conditions" in n for n in result.notes)
        # model is uncertain because of "our model" on the premise side
        assert any("Ambiguously worded" in n for n in result.notes)

    def test_uncertain_conditions_note(self):
        classification = make_classification(label="contradiction")
        comparison = compare(
            "We find that our dataset improves performance.",
            "We find that accuracy improves on the GLUE benchmark.",
        )
        result = contextualize_nli(classification, comparison)
        uncertain_notes = [n for n in result.notes if "Ambiguously worded" in n]
        assert uncertain_notes
        assert result.classification.label == "contradiction"

    def test_comparison_is_embedded(self):
        classification = make_classification()
        comparison = compare(
            "We find that accuracy improves on the GLUE benchmark.",
            "We find that accuracy improves on the GLUE benchmark.",
        )
        result = contextualize_nli(classification, comparison)
        assert result.conditions is comparison

    def test_bad_types_raise(self):
        comparison = compare("We find a clear effect.", "We find a clear effect.")
        with pytest.raises(TypeError):
            contextualize_nli("not a classification", comparison)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            contextualize_nli(make_classification(), "not a comparison")  # type: ignore[arg-type]
