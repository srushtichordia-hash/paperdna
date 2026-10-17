"""Tests for building attributions from claims and paper records."""

from __future__ import annotations

import pytest

from paperdna.evidence import (
    AttributionMismatchError,
    InvalidSpanError,
    PaperResolver,
    build_attribution,
    resolve_papers,
    verify_span,
)
from tests.evidence.conftest import (
    DEFAULT_ABSTRACT,
    DEFAULT_SENTENCE,
    make_classification,
    make_claim,
    make_hypothesis_claim,
    make_hypothesis_record,
    make_record,
)


class TestVerifySpan:
    def test_matching_span(self):
        start = DEFAULT_ABSTRACT.index(DEFAULT_SENTENCE)
        assert verify_span(DEFAULT_ABSTRACT, DEFAULT_SENTENCE, start, start + len(DEFAULT_SENTENCE))

    def test_mismatched_span(self):
        assert not verify_span(DEFAULT_ABSTRACT, DEFAULT_SENTENCE, 0, 10)

    def test_out_of_range_offsets(self):
        assert not verify_span(DEFAULT_ABSTRACT, DEFAULT_SENTENCE, 0, 10_000)
        assert not verify_span(DEFAULT_ABSTRACT, DEFAULT_SENTENCE, -1, 5)
        assert not verify_span(DEFAULT_ABSTRACT, DEFAULT_SENTENCE, 10, 5)


class TestValidAttribution:
    def test_verified_with_full_metadata(self):
        attribution = build_attribution(
            make_claim(),
            make_hypothesis_claim(),
            records=[make_record(), make_hypothesis_record()],
        )
        assert attribution.premise.verification == "verified"
        assert attribution.hypothesis.verification == "verified"
        assert attribution.premise.paper.title == "Sparse Attention Studies"
        assert attribution.premise.paper.url == "https://arxiv.org/abs/2101.00001"
        assert attribution.hypothesis.paper.url is None  # genuinely missing, preserved
        assert attribution.premise.sentence == DEFAULT_SENTENCE

    def test_claim_ids_and_offsets(self):
        claim = make_claim()
        attribution = build_attribution(
            claim, make_hypothesis_claim(), records=[make_record(), make_hypothesis_record()]
        )
        assert attribution.premise.claim.claim_id == (
            f"arxiv:2101.00001:{claim.start}-{claim.end}"
        )
        assert attribution.premise.claim.start == claim.start

    def test_classification_embedded_unchanged(self):
        classification = make_classification()
        attribution = build_attribution(
            make_claim(),
            make_hypothesis_claim(),
            records=[make_record(), make_hypothesis_record()],
            classification=classification,
        )
        assert attribution.classification is classification
        interpretation = attribution.interpretation
        assert interpretation is not None
        assert interpretation.label == classification.label
        assert interpretation.scores == classification.scores
        assert interpretation.confidence == classification.confidence
        assert interpretation.uncertain == classification.uncertain
        assert interpretation.model == classification.model

    def test_evidence_only_attribution(self):
        attribution = build_attribution(make_claim(), make_hypothesis_claim())
        assert attribution.classification is None
        assert attribution.interpretation is None
        assert attribution.conditions is None


class TestMissingProvenance:
    def test_missing_record_is_unverified_not_fabricated(self):
        attribution = build_attribution(make_claim(), make_hypothesis_claim())
        for side in (attribution.premise, attribution.hypothesis):
            assert side.verification == "unverified"
            assert side.unavailable_reason == "paper record not found"
            assert side.paper.title is None
            assert side.paper.url is None
            # The verbatim sentence is still preserved.
            assert side.sentence

    def test_missing_url_stays_none(self):
        record = make_record(url=None)
        attribution = build_attribution(make_claim(), make_hypothesis_claim(), records=[record])
        assert attribution.premise.verification == "verified"
        assert attribution.premise.paper.url is None

    def test_empty_sentence_is_unavailable(self):
        hypothesis = make_hypothesis_claim(sentence="", start=0, end=0)
        attribution = build_attribution(
            make_claim(), hypothesis, records=[make_record()]
        )
        assert attribution.hypothesis.verification == "unavailable"
        assert not attribution.hypothesis.available
        assert attribution.hypothesis.unavailable_reason == "empty evidence sentence"
        assert attribution.premise.verification == "verified"


class TestRejectedSpans:
    def test_span_mismatching_abstract_raises(self):
        bad = make_claim(start=0, end=len(DEFAULT_SENTENCE))
        with pytest.raises(InvalidSpanError):
            build_attribution(bad, make_hypothesis_claim(), records=[make_record()])

    def test_negative_offsets_raise(self):
        bad = make_claim(start=-5, end=10)
        with pytest.raises(InvalidSpanError):
            build_attribution(bad, make_hypothesis_claim())

    def test_end_before_start_raises(self):
        bad = make_claim(start=50, end=10)
        with pytest.raises(InvalidSpanError):
            build_attribution(bad, make_hypothesis_claim())


class TestCrossChecks:
    def test_classification_text_mismatch_raises(self):
        wrong = make_classification(premise_text="A different sentence entirely.")
        with pytest.raises(AttributionMismatchError):
            build_attribution(
                make_claim(), make_hypothesis_claim(), classification=wrong
            )

    def test_classification_identity_mismatch_raises(self):
        wrong = make_classification(hypothesis_source="elsewhere")
        with pytest.raises(AttributionMismatchError):
            build_attribution(
                make_claim(), make_hypothesis_claim(), classification=wrong
            )

    def test_conditions_identity_mismatch_raises(self):
        from paperdna.conditions import compare_conditions, extract_conditions

        conditions = compare_conditions(
            extract_conditions(make_claim(source="other")),
            extract_conditions(make_hypothesis_claim()),
        )
        with pytest.raises(AttributionMismatchError):
            build_attribution(
                make_claim(), make_hypothesis_claim(), conditions=conditions
            )

    def test_same_claim_on_both_sides_is_deterministic(self):
        claim = make_claim()
        first = build_attribution(claim, claim, records=[make_record()])
        second = build_attribution(claim, claim, records=[make_record()])
        assert first == second
        assert first.premise.claim.claim_id == first.hypothesis.claim.claim_id


class TestResolver:
    def test_duplicate_records_first_wins(self):
        resolver = resolve_papers([make_record(title="First"), make_record(title="Second")])
        assert resolver.duplicate_ids_dropped == 1
        assert resolver.get("arxiv", "2101.00001").title == "First"

    def test_lookup_casefolds_source(self):
        resolver = resolve_papers([make_record(source="ArXiv")])
        assert resolver.get("arxiv", "2101.00001") is not None

    def test_missing_record_returns_none(self):
        assert resolve_papers([make_record()]).get("fixture", "nope") is None

    def test_resolver_instance_accepted_directly(self):
        resolver = PaperResolver([make_record()])
        attribution = build_attribution(
            make_claim(), make_hypothesis_claim(), records=resolver
        )
        assert attribution.premise.verification == "verified"

    def test_non_record_entries_rejected(self):
        with pytest.raises(TypeError):
            resolve_papers(["not a record"])


class TestTypes:
    def test_bad_claim_types_raise(self):
        with pytest.raises(TypeError):
            build_attribution("a", make_hypothesis_claim())
        with pytest.raises(TypeError):
            build_attribution(make_claim(), 42)

    def test_bad_classification_type_raises(self):
        with pytest.raises(TypeError):
            build_attribution(
                make_claim(), make_hypothesis_claim(), classification="entailment"
            )

    def test_bad_conditions_type_raises(self):
        with pytest.raises(TypeError):
            build_attribution(
                make_claim(), make_hypothesis_claim(), conditions={}
            )
