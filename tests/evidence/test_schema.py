"""Tests for the evidence data model."""

from __future__ import annotations

import pytest

from paperdna.evidence import (
    EVIDENCE_STATUSES,
    EVIDENCE_VERIFICATION,
    ClaimRef,
    InvalidSpanError,
    PaperRef,
    SentenceEvidence,
    build_attribution,
    claim_id_for,
)
from tests.evidence.conftest import make_claim, make_hypothesis_claim, make_record


class TestClaimId:
    def test_format_with_offsets(self):
        assert claim_id_for("arxiv", "2101.00001", 10, 40) == "arxiv:2101.00001:10-40"

    def test_unknown_offsets_are_not_invented(self):
        assert claim_id_for("arxiv", "2101.00001", None, None) == "arxiv:2101.00001:unknown"

    def test_stable_across_calls(self):
        a = claim_id_for("arxiv", "x", 0, 5)
        b = claim_id_for("arxiv", "x", 0, 5)
        assert a == b


class TestClaimRef:
    def test_offsets_both_or_none(self):
        with pytest.raises(ValueError):
            ClaimRef(claim_id="c", source="s", source_id="i", start=0, end=None)

    def test_negative_offsets_rejected(self):
        with pytest.raises(InvalidSpanError):
            ClaimRef(claim_id="c", source="s", source_id="i", start=-1, end=5)

    def test_end_before_start_rejected(self):
        with pytest.raises(InvalidSpanError):
            ClaimRef(claim_id="c", source="s", source_id="i", start=9, end=5)

    def test_invalid_span_is_also_value_error(self):
        # Subclasses ValueError so malformed input can be caught uniformly.
        assert issubclass(InvalidSpanError, ValueError)


class TestPaperRef:
    def test_identity_casefolds_source(self):
        ref = PaperRef(source="ArXiv", source_id="2101.00001")
        assert ref.identity == ("arxiv", "2101.00001")

    def test_missing_metadata_stays_none(self):
        ref = PaperRef(source="arxiv", source_id="x")
        assert ref.title is None and ref.url is None and ref.doi is None


class TestSentenceEvidence:
    def _evidence(self, **overrides):
        fields = {
            "paper": PaperRef(source="arxiv", source_id="x"),
            "claim": ClaimRef(claim_id="arxiv:x:0-1", source="arxiv", source_id="x", start=0, end=1),
            "sentence": "ab",
            "verification": "verified",
        }
        fields.update(overrides)
        return SentenceEvidence(**fields)

    def test_unknown_verification_rejected(self):
        with pytest.raises(ValueError):
            self._evidence(verification="probably")

    def test_unverified_requires_reason(self):
        with pytest.raises(ValueError):
            self._evidence(verification="unverified", unavailable_reason=None)

    def test_unavailable_requires_reason(self):
        with pytest.raises(ValueError):
            self._evidence(verification="unavailable", unavailable_reason=None)

    def test_availability_and_verified_flags(self):
        assert self._evidence().available and self._evidence().verified
        unverified = self._evidence(verification="unverified", unavailable_reason="r")
        assert unverified.available and not unverified.verified
        unavailable = self._evidence(verification="unavailable", unavailable_reason="r")
        assert not unavailable.available and not unavailable.verified

    def test_verification_statuses_are_documented(self):
        assert set(EVIDENCE_VERIFICATION) == {
            "verified", "verified_by_search", "unverified", "unavailable",
        }
        assert all(f"evidence_{v}" in EVIDENCE_STATUSES for v in EVIDENCE_VERIFICATION)


class TestStatsInvariant:
    def test_counts_sum_to_two(self):
        attribution = build_attribution(
            make_claim(), make_hypothesis_claim(), records=[make_record()]
        )
        assert sum(attribution.stats.values()) == 2
        assert set(attribution.stats) == set(EVIDENCE_STATUSES)

    def test_counts_reflect_verifications(self):
        attribution = build_attribution(make_claim(), make_hypothesis_claim())
        assert attribution.stats["evidence_unverified"] == 2
        assert attribution.stats["evidence_verified"] == 0
