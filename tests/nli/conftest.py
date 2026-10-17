"""Shared helpers for NLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperdna.claims.schema import CandidateClaim
from paperdna.nli.schema import RawPairScores

NLI_FIXTURE = Path(__file__).parent.parent / "fixtures" / "nli_pairs.json"


class FakeNliClassifier:
    """Deterministic offline classifier: scripted scores per pair.

    Scores are keyed by the exact (premise, hypothesis) text pair. Pairs
    not in the script fall back to ``default_scores``. Records every call
    so tests can assert batching behavior. No model, no downloads.
    """

    name = "fake-nli"
    max_length = 512

    def __init__(
        self,
        script: dict[tuple[str, str], dict[str, float]] | None = None,
        default_scores: dict[str, float] | None = None,
    ):
        self.script = script or {}
        self.default_scores = default_scores or {
            "entailment": 0.8,
            "contradiction": 0.1,
            "neutral": 0.1,
        }
        self.calls: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[RawPairScores]:
        self.calls.append(list(pairs))
        return [
            RawPairScores(scores=self.script.get(pair, self.default_scores))
            for pair in pairs
        ]


def make_claim(**overrides) -> CandidateClaim:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "sentence": "We find that the method works.",
        "start": 0,
        "end": 35,
        "cue": "we find",
        "hedges": (),
    }
    fields.update(overrides)
    return CandidateClaim(**fields)


@pytest.fixture
def fixture_pairs() -> list[dict]:
    return json.loads(NLI_FIXTURE.read_text(encoding="utf-8"))["pairs"]


def claim_from(spec: dict) -> CandidateClaim:
    sentence = spec["sentence"]
    return CandidateClaim(
        source=spec["source"],
        source_id=spec["source_id"],
        sentence=sentence,
        start=0,
        end=len(sentence),
        cue="fixture",
        hedges=tuple(spec.get("hedges", ())),
    )
