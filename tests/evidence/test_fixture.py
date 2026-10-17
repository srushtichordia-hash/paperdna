"""Sanity check against synthetic hand-labeled attribution cases
(not a claim of scientific quality)."""

from __future__ import annotations

import pytest

from paperdna.claims.schema import CandidateClaim
from paperdna.corpus.schema import PaperRecord
from paperdna.evidence import (
    EVIDENCE_STATUSES,
    attribute_from_classification,
    build_attribution,
)
from tests.evidence.conftest import make_classification


def _records(case: dict) -> list[PaperRecord]:
    return [
        PaperRecord(
            source=r["source"],
            source_id=r["source_id"],
            title=r["title"],
            abstract=r["abstract"],
            authors=tuple(r["authors"]),
            published=r.get("published"),
            url=r.get("url"),
            doi=r.get("doi"),
        )
        for r in case["records"]
    ]


def _claim(data: dict, records: list[PaperRecord]) -> CandidateClaim:
    by_id = {(r.source.casefold(), r.source_id): r for r in records}
    sentence = data["sentence"]
    record = by_id.get((data["source"].casefold(), data["source_id"]))
    if sentence and record is not None:
        start = record.abstract.index(sentence)
    else:
        # No record to locate the sentence in (missing-record or empty
        # cases); the builder marks such evidence unverified/unavailable.
        start = 0
    return CandidateClaim(
        source=data["source"],
        source_id=data["source_id"],
        sentence=sentence,
        start=start,
        end=start + len(sentence),
        cue="",
        hedges=(),
    )


class TestHandLabeledFixture:
    def test_expected_statuses(self, fixture_cases):
        for case in fixture_cases:
            records = _records(case)
            classification = (
                make_classification(**case["classification"])
                if "classification" in case
                else None
            )
            if case["mode"] == "build":
                attribution = build_attribution(
                    _claim(case["premise"], records),
                    _claim(case["hypothesis"], records),
                    records=records,
                    classification=classification,
                )
            else:
                attribution = attribute_from_classification(
                    classification, records=records
                )
            expected = case["expected"]
            assert attribution.premise.verification == expected["premise_verification"], case["name"]
            assert attribution.hypothesis.verification == expected["hypothesis_verification"], case["name"]
            assert sum(attribution.stats.values()) == 2, case["name"]
            assert set(attribution.stats) == set(EVIDENCE_STATUSES)
            if "label" in expected:
                assert attribution.interpretation is not None
                assert attribution.interpretation.label == expected["label"]
            if expected.get("url_is_none"):
                assert attribution.premise.paper.url is None
                assert attribution.hypothesis.paper.url is None
