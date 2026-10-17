"""Integration tests: conditions API -> evidence attribution (read-only)."""

from __future__ import annotations

import pytest

from paperdna.conditions import (
    compare_conditions,
    contextualize_nli,
    extract_conditions,
)
from paperdna.evidence import (
    attribute_nli_with_context,
    build_attribution,
)
from tests.evidence.conftest import (
    make_classification,
    make_claim,
    make_hypothesis_claim,
    make_hypothesis_record,
    make_record,
)


def _pipeline():
    premise = make_claim()
    hypothesis = make_hypothesis_claim()
    classification = make_classification()
    comparison = compare_conditions(
        extract_conditions(premise), extract_conditions(hypothesis)
    )
    context = contextualize_nli(classification, comparison)
    return premise, hypothesis, classification, comparison, context


class TestNliWithContextPath:
    def test_full_pipeline_attribution(self):
        premise, hypothesis, classification, comparison, context = _pipeline()
        attribution = attribute_nli_with_context(
            premise, hypothesis, context,
            records=[make_record(), make_hypothesis_record()],
        )
        # Observed side: both sentences verified against their abstracts.
        assert attribution.premise.verification == "verified"
        assert attribution.hypothesis.verification == "verified"
        # Model side: classification embedded unchanged, plus condition context.
        assert attribution.classification is classification
        assert attribution.conditions is comparison
        interpretation = attribution.interpretation
        assert interpretation is not None
        assert interpretation.label == "entailment"  # unchanged from classification
        assert interpretation.condition_summary == comparison.summary
        assert interpretation.condition_stats == comparison.stats
        assert interpretation.notes == context.notes
        assert sum(attribution.stats.values()) == 2

    def test_condition_difference_never_rewrites_label(self):
        premise, hypothesis, classification, comparison, context = _pipeline()
        attribution = attribute_nli_with_context(premise, hypothesis, context)
        # The dataset differs (GLUE vs ImageNet) yet the label is untouched.
        assert comparison.field("dataset").status == "differing"
        assert attribution.interpretation.label == classification.label

    def test_bad_type_raises(self):
        premise, hypothesis, _, _, _ = _pipeline()
        with pytest.raises(TypeError):
            attribute_nli_with_context(premise, hypothesis, "not a context")


class TestConditionsOnlyPath:
    def test_conditions_without_classification(self):
        premise = make_claim()
        hypothesis = make_hypothesis_claim()
        comparison = compare_conditions(
            extract_conditions(premise), extract_conditions(hypothesis)
        )
        attribution = build_attribution(
            premise, hypothesis, conditions=comparison
        )
        assert attribution.conditions is comparison
        assert attribution.classification is None
        assert attribution.interpretation is None  # interpretation requires a model output
