"""Shared deterministic fakes for the evaluation/integration suite.

Self-contained copies implementing the same Protocols as the real
backends (``Embedder``, ``NliClassifier``) — no model, no network, no
imports from the optional ML stack.
"""

from __future__ import annotations

import math

import pytest

from paperdna.nli.schema import RawPairScores


def _words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w.strip(".,;:()")}


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (offline)."""

    name = "fake-embedder"
    dimension = 24

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for word in _words(text):
                vector[hash(word) % self.dimension] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


class FakeClassifier:
    """Deterministic overlap-based NLI fake (offline).

    Entailment rises with shared vocabulary; a premise negation absent
    from the hypothesis flips the mass to contradiction.
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
                scores = {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1}
            else:
                scores = {
                    "entailment": overlap,
                    "contradiction": 0.05,
                    "neutral": max(0.0, 0.95 - overlap),
                }
            total = sum(scores.values()) or 1.0
            results.append(
                RawPairScores(scores={k: v / total for k, v in scores.items()})
            )
        return results


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier()
