"""Tests for serialization and exact round-trip behavior."""

from __future__ import annotations

import json

import pytest

from paperdna.conditions import compare_conditions, contextualize_nli, extract_conditions
from paperdna.evidence import (
    attribution_from_dict,
    attribution_to_dict,
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


def _full_attribution():
    premise = make_claim()
    hypothesis = make_hypothesis_claim()
    classification = make_classification()
    comparison = compare_conditions(
        extract_conditions(premise), extract_conditions(hypothesis)
    )
    context = contextualize_nli(classification, comparison)
    return attribute_nli_with_context(
        premise, hypothesis, context,
        records=[make_record(), make_hypothesis_record()],
    )


class TestRoundTrip:
    def test_full_round_trip_is_exact(self):
        attribution = _full_attribution()
        data = attribution_to_dict(attribution)
        assert attribution_from_dict(data) == attribution

    def test_json_safe_and_round_trips(self):
        attribution = _full_attribution()
        text = json.dumps(attribution_to_dict(attribution))
        assert attribution_from_dict(json.loads(text)) == attribution

    def test_evidence_only_round_trip(self):
        attribution = build_attribution(make_claim(), make_hypothesis_claim())
        data = attribution_to_dict(attribution)
        assert data["classification"] is None
        assert data["interpretation"] is None
        assert data["conditions"] is None
        assert attribution_from_dict(data) == attribution

    def test_unavailable_evidence_round_trips(self):
        hypothesis = make_hypothesis_claim(sentence="", start=0, end=0)
        attribution = build_attribution(make_claim(), hypothesis)
        data = attribution_to_dict(attribution)
        rebuilt = attribution_from_dict(data)
        assert rebuilt == attribution
        assert rebuilt.hypothesis.unavailable_reason == "empty evidence sentence"

    def test_model_interpretation_stays_labeled_as_model_output(self):
        data = attribution_to_dict(_full_attribution())
        # The interpretation is a separate key from the observed evidence.
        assert "interpretation" in data and "premise" in data and "hypothesis" in data
        assert data["interpretation"]["label"] == "entailment"
        assert data["interpretation"]["scores"]["entailment"] == 0.8


class TestStrictness:
    def test_to_dict_requires_attribution(self):
        with pytest.raises(TypeError):
            attribution_to_dict("not an attribution")

    def test_from_dict_requires_dict(self):
        with pytest.raises(ValueError):
            attribution_from_dict([1, 2, 3])

    def test_from_dict_missing_key_raises(self):
        data = attribution_to_dict(_full_attribution())
        del data["premise"]
        with pytest.raises(ValueError):
            attribution_from_dict(data)

    def test_from_dict_missing_nested_key_raises(self):
        data = attribution_to_dict(_full_attribution())
        del data["premise"]["paper"]["source"]
        with pytest.raises(ValueError):
            attribution_from_dict(data)

    def test_from_dict_bad_verification_raises(self):
        data = attribution_to_dict(_full_attribution())
        data["premise"]["verification"] = "definitely"
        with pytest.raises(ValueError):
            attribution_from_dict(data)
