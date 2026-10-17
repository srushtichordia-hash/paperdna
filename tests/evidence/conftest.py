"""Shared helpers for evidence-attribution tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperdna.claims.schema import CandidateClaim
from paperdna.corpus.schema import PaperRecord
from paperdna.nli.schema import PairClassification

EVIDENCE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "evidence_attributions.json"

DEFAULT_ABSTRACT = (
    "We study sparse attention for long documents. "
    "We find that accuracy improves on the GLUE benchmark. "
    "Prior work is extensive."
)
DEFAULT_SENTENCE = "We find that accuracy improves on the GLUE benchmark."
HYPOTHESIS_ABSTRACT = (
    "Background sentences here. "
    "We find that accuracy improves on the ImageNet dataset. "
    "Closing remarks."
)
HYPOTHESIS_SENTENCE = "We find that accuracy improves on the ImageNet dataset."


def make_record(**overrides) -> PaperRecord:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "title": "Sparse Attention Studies",
        "abstract": DEFAULT_ABSTRACT,
        "authors": ("A. Author",),
        "published": "2021-05",
        "url": "https://arxiv.org/abs/2101.00001",
        "doi": None,
    }
    fields.update(overrides)
    return PaperRecord(**fields)


def make_hypothesis_record(**overrides) -> PaperRecord:
    fields = {
        "source": "fixture",
        "source_id": "f2",
        "title": "Vision Findings",
        "abstract": HYPOTHESIS_ABSTRACT,
        "authors": ("B. Researcher",),
        "published": None,
        "url": None,
        "doi": None,
    }
    fields.update(overrides)
    return PaperRecord(**fields)


def make_claim(**overrides) -> CandidateClaim:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "sentence": DEFAULT_SENTENCE,
        "start": DEFAULT_ABSTRACT.index(DEFAULT_SENTENCE),
        "cue": "we find",
        "hedges": (),
    }
    fields["end"] = fields["start"] + len(fields["sentence"])
    fields.update(overrides)
    return CandidateClaim(**fields)


def make_hypothesis_claim(**overrides) -> CandidateClaim:
    fields = {
        "source": "fixture",
        "source_id": "f2",
        "sentence": HYPOTHESIS_SENTENCE,
        "start": HYPOTHESIS_ABSTRACT.index(HYPOTHESIS_SENTENCE),
        "cue": "we find",
        "hedges": (),
    }
    fields["end"] = fields["start"] + len(fields["sentence"])
    fields.update(overrides)
    return CandidateClaim(**fields)


def make_classification(**overrides) -> PairClassification:
    fields = {
        "premise_source": "arxiv",
        "premise_source_id": "2101.00001",
        "premise_text": DEFAULT_SENTENCE,
        "hypothesis_source": "fixture",
        "hypothesis_source_id": "f2",
        "hypothesis_text": HYPOTHESIS_SENTENCE,
        "label": "entailment",
        "scores": {"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1},
        "confidence": 0.8,
        "uncertain": False,
        "hedged": False,
        "truncated": False,
        "model": "fake-nli",
    }
    fields.update(overrides)
    return PairClassification(**fields)


@pytest.fixture
def fixture_cases() -> list[dict]:
    return json.loads(EVIDENCE_FIXTURE.read_text(encoding="utf-8"))["cases"]
