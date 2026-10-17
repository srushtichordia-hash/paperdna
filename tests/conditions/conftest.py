"""Shared helpers for condition-analysis tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperdna.claims.schema import CandidateClaim
from paperdna.nli.schema import PairClassification

CONDITIONS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "conditions_eval.json"


def make_claim(**overrides) -> CandidateClaim:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "sentence": "We find that accuracy improves on the GLUE benchmark.",
        "start": 0,
        "cue": "we find",
        "hedges": (),
    }
    fields.update(overrides)
    fields["end"] = fields["start"] + len(fields["sentence"])
    return CandidateClaim(**fields)


def make_classification(**overrides) -> PairClassification:
    fields = {
        "premise_source": "arxiv",
        "premise_source_id": "c1",
        "premise_text": "Premise.",
        "hypothesis_source": "fixture",
        "hypothesis_source_id": "c2",
        "hypothesis_text": "Hypothesis.",
        "label": "contradiction",
        "scores": {"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05},
        "confidence": 0.9,
        "uncertain": False,
        "hedged": False,
        "truncated": False,
        "model": "fake-nli",
    }
    fields.update(overrides)
    return PairClassification(**fields)


@pytest.fixture
def fixture_pairs() -> list[dict]:
    return json.loads(CONDITIONS_FIXTURE.read_text(encoding="utf-8"))["pairs"]
