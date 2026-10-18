"""Shared fakes for the API and pipeline test suite.

The fakes implement the same Protocols as the real backends
(``paperdna.retrieval.embedder.Embedder`` and
``paperdna.nli.classifier.NliClassifier``) but are deterministic,
offline, and import nothing from the optional ML stack. No test in this
package downloads a model or touches the network.
"""

from __future__ import annotations

import math

import pytest

from paperdna.corpus.ingest import ingest_records
from paperdna.nli.schema import RELATIONSHIPS, RawPairScores

# httpx + fastapi TestClient are only imported inside fixtures so that the
# pure-pipeline tests in this package never require the HTTP stack.


def _words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w.strip(".,;:()")}


class FakeEmbedder:
    """Deterministic embedding: a small hashed bag-of-words vector.

    Same text always maps to the same vector; overlapping vocabulary
    yields higher cosine similarity. No model, no download.
    """

    name = "fake-embedder"
    dimension = 24

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for word in _words(text):
                index = hash(word) % self.dimension
                vector[index] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


class FakeClassifier:
    """Deterministic NLI: overlap between premise and hypothesis.

    Entailment rises with shared vocabulary; contradiction is high when
    the premise contains a negation marker ("no", "not") absent from the
    hypothesis; remainder goes to neutral. Scores always sum to 1.
    """

    name = "fake-nli"
    max_length = 512

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[RawPairScores]:
        results = []
        for premise, hypothesis in pairs:
            p_words = _words(premise)
            h_words = _words(hypothesis)
            union = p_words | h_words
            overlap = len(p_words & h_words) / (len(union) or 1)
            negated = ("no" in p_words or "not" in p_words) and ("no" not in h_words)
            if negated:
                scores = {
                    "entailment": 0.1,
                    "contradiction": 0.8,
                    "neutral": 0.1,
                }
            else:
                scores = {
                    "entailment": overlap,
                    "contradiction": 0.05,
                    "neutral": max(0.0, 0.95 - overlap),
                }
            total = sum(scores.values()) or 1.0
            scores = {key: value / total for key, value in scores.items()}
            # Guard against float drift pushing a value outside [0, 1].
            scores = {key: min(1.0, max(0.0, value)) for key, value in scores.items()}
            results.append(RawPairScores(scores=scores))
        return results


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier()


def make_records(count: int = 4):
    """A small deterministic corpus of valid PaperRecords."""
    raw = [
        {
            "source": "test",
            "source_id": f"p{i:03d}",
            "title": f"Test paper {i} on retrieval",
            "abstract": (
                f"We find that retrieval augmentation improves question "
                f"answering accuracy on dataset d{i} by a clear margin. "
                f"Results hold across model sizes."
            ),
            "authors": [f"Author {i}"],
            "published": "2023",
            "url": f"https://example.org/p{i:03d}",
        }
        for i in range(count)
    ]
    return ingest_records(raw).papers


@pytest.fixture
def corpus():
    return make_records()


# ---------------------------------------------------------------------------
# FastAPI test-client fixtures (HTTP stack imported lazily)
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """A ModelRegistry with no models loaded (lexical baseline mode)."""
    from paperdna.api.models import ModelRegistry

    return ModelRegistry()


@pytest.fixture
def semantic_registry():
    """A ModelRegistry with the deterministic fakes injected, so the
    pipeline exercises the semantic + NLI path without any real model."""
    from paperdna.api.models import ModelRegistry

    reg = ModelRegistry()
    reg.embedder = FakeEmbedder()
    reg.classifier = FakeClassifier()
    return reg


@pytest.fixture
def client(registry):
    """TestClient over an app whose model registry is empty (no models)."""
    from fastapi.testclient import TestClient

    from paperdna.api.app import create_app

    return TestClient(create_app(registry=registry))


@pytest.fixture
def semantic_client(semantic_registry):
    """TestClient over an app with fakes injected as loaded models."""
    from fastapi.testclient import TestClient

    from paperdna.api.app import create_app

    return TestClient(create_app(registry=semantic_registry))
