"""Shared test helpers for retrieval tests."""

from __future__ import annotations

import pytest

from paperdna.corpus.schema import PaperRecord


class FakeEmbedder:
    """Deterministic offline embedder: token-overlap hashing.

    Maps each distinct input text to a fixed vector derived from its
    tokens, so identical texts always get identical vectors and similar
    texts (sharing tokens) get higher cosine similarity. No model, no
    downloads, no randomness.
    """

    name = "fake"
    dimension = 8

    _TOKENS = (
        "attention",
        "gradient",
        "retrieval",
        "networks",
        "documents",
        "sparse",
        "stabilization",
        "benchmarks",
    )

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            lowered.count(token) + 0.1  # nonzero baseline keeps vectors valid
            for token in self._TOKENS
        ]


class StaticEmbedder:
    """Embedder returning pre-assigned vectors keyed by exact text."""

    name = "static"

    def __init__(self, vectors: dict[str, list[float]], dimension: int | None = None):
        self.vectors = vectors
        self.dimension = dimension
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self.vectors[text] for text in texts]


def make_record(**overrides) -> PaperRecord:
    fields = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "title": "Attention Mechanisms",
        "abstract": "We study attention mechanisms for documents.",
        "authors": ("Ada Lovelace",),
    }
    fields.update(overrides)
    return PaperRecord(**fields)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
