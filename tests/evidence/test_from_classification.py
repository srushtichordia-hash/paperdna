"""Tests for the classification-only attribution path (no original claims)."""

from __future__ import annotations

import pytest

from paperdna.evidence import (
    AttributionMismatchError,
    attribute_from_classification,
)
from tests.evidence.conftest import (
    DEFAULT_SENTENCE,
    HYPOTHESIS_SENTENCE,
    make_classification,
    make_hypothesis_record,
    make_record,
)


class TestUniqueOccurrence:
    def test_offsets_inferred_from_unique_match(self):
        classification = make_classification()
        attribution = attribute_from_classification(
            classification, records=[make_record(), make_hypothesis_record()]
        )
        assert attribution.premise.verification == "verified_by_search"
        assert attribution.premise.claim.start is not None
        record = make_record()
        assert record.abstract[attribution.premise.claim.start:attribution.premise.claim.end] == DEFAULT_SENTENCE
        assert attribution.premise.claim.claim_id.endswith(
            f":{attribution.premise.claim.start}-{attribution.premise.claim.end}"
        )

    def test_interpretation_carried_over(self):
        attribution = attribute_from_classification(make_classification())
        assert attribution.interpretation is not None
        assert attribution.interpretation.label == "entailment"


class TestRepeatedSentences:
    def test_repeated_sentence_leaves_offsets_none(self):
        abstract = f"{HYPOTHESIS_SENTENCE} Middle text. {HYPOTHESIS_SENTENCE}"
        record = make_hypothesis_record(abstract=abstract)
        attribution = attribute_from_classification(
            make_classification(), records=[make_record(), record]
        )
        hypothesis = attribution.hypothesis
        assert hypothesis.verification == "unverified"
        assert hypothesis.claim.start is None and hypothesis.claim.end is None
        assert "occurs 2 times" in hypothesis.unavailable_reason
        assert hypothesis.claim.claim_id.endswith(":unknown")
        # Verbatim text is still preserved for the reader.
        assert hypothesis.sentence == HYPOTHESIS_SENTENCE

    def test_sentence_absent_from_abstract_is_unavailable(self):
        record = make_record(abstract="An entirely different abstract body.")
        attribution = attribute_from_classification(
            make_classification(), records=[record, make_hypothesis_record()]
        )
        assert attribution.premise.verification == "unavailable"
        assert "not found in abstract" in attribution.premise.unavailable_reason
        assert not attribution.premise.available


class TestMissingProvenance:
    def test_missing_record_leaves_offsets_none(self):
        attribution = attribute_from_classification(make_classification())
        assert attribution.premise.verification == "unverified"
        assert attribution.premise.unavailable_reason == "paper record not found"
        assert attribution.premise.claim.start is None
        assert attribution.premise.paper.title is None

    def test_empty_text_is_unavailable(self):
        attribution = attribute_from_classification(
            make_classification(hypothesis_text="")
        )
        assert attribution.hypothesis.verification == "unavailable"
        assert attribution.hypothesis.unavailable_reason == "empty evidence sentence"


class TestCrossChecksAndTypes:
    def test_conditions_identity_mismatch_raises(self):
        from paperdna.conditions import compare_conditions, extract_conditions
        from paperdna.evidence import build_attribution
        from tests.evidence.conftest import make_claim, make_hypothesis_claim

        conditions = compare_conditions(
            extract_conditions(make_claim()),
            extract_conditions(make_hypothesis_claim(source="other")),
        )
        with pytest.raises(AttributionMismatchError):
            attribute_from_classification(
                make_classification(), conditions=conditions
            )

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            attribute_from_classification("not a classification")
