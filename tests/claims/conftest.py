"""Shared helpers for claim-extraction tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperdna.corpus.schema import PaperRecord

CLAIMS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "claims_eval.json"


def make_record(**overrides) -> PaperRecord:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "title": "A Title",
        "abstract": "We find that the method works.",
        "authors": ("Ada Lovelace",),
    }
    fields.update(overrides)
    return PaperRecord(**fields)


@pytest.fixture
def labeled_abstracts() -> list[dict]:
    data = json.loads(CLAIMS_FIXTURE.read_text(encoding="utf-8"))
    return data["abstracts"]
