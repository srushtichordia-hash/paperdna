"""Tests for the model lifecycle endpoints: opt-in confirmation gating.

The core honesty rule: a model is NEVER constructed or downloaded without
an explicit ``confirmDownload: true``. These tests enforce that the
endpoint refuses to load otherwise, and that loading is idempotent/gated.
"""

from __future__ import annotations


def test_load_without_confirmation_is_refused(client):
    response = client.post("/api/models/load", json={"confirmDownload": False})
    assert response.status_code == 400
    # Nothing was loaded.
    status = client.get("/api/models").json()
    assert status["embedding"]["loaded"] is False
    assert status["nli"]["loaded"] is False


def test_load_with_default_body_is_refused(client):
    # An empty body defaults confirmDownload to False.
    response = client.post("/api/models/load", json={})
    assert response.status_code == 400


def test_load_with_confirmation_when_not_installed_degrades(client):
    """With confirmation but no optional deps, load degrades gracefully."""
    response = client.post("/api/models/load", json={"confirmDownload": True})
    assert response.status_code == 200
    body = response.json()
    # sentence-transformers is not installed in the test venv, so nothing
    # loads — but the call must not raise.
    if body["embedding"]["installed"] is False:
        assert body["embedding"]["loaded"] is False


def test_unload_resets_registry(semantic_client):
    # Fakes are loaded.
    assert semantic_client.get("/api/models").json()["embedding"]["loaded"] is True
    semantic_client.post("/api/models/unload")
    status = semantic_client.get("/api/models").json()
    assert status["embedding"]["loaded"] is False
    assert status["nli"]["loaded"] is False


def test_load_confirmation_actually_constructs(monkeypatch, client, registry):
    """With confirmation and a stubbed constructor, the model is built."""
    constructed = {"embedder": 0, "classifier": 0}

    class StubEmbedder:
        name = "stub-embedder"
        dimension = 8

        def __init__(self, *a, **k):
            constructed["embedder"] += 1

        def embed(self, texts):
            return [[0.0] * self.dimension for _ in texts]

    class StubClassifier:
        name = "stub-nli"
        max_length = 512

        def __init__(self, *a, **k):
            constructed["classifier"] += 1

        def score_pairs(self, pairs):
            from paperdna.nli.schema import RawPairScores

            return [
                RawPairScores(
                    scores={"entailment": 0.3, "contradiction": 0.3, "neutral": 0.4}
                )
                for _ in pairs
            ]

    # Pretend the optional dependency is present.
    monkeypatch.setattr(
        "paperdna.api.models._sentence_transformers_installed", lambda: True
    )
    monkeypatch.setattr(
        "paperdna.retrieval.embedder.SentenceTransformerEmbedder", StubEmbedder
    )
    monkeypatch.setattr(
        "paperdna.nli.classifier.CrossEncoderNliClassifier", StubClassifier
    )

    response = client.post("/api/models/load", json={"confirmDownload": True})
    assert response.status_code == 200
    assert constructed["embedder"] == 1
    assert constructed["classifier"] == 1
    assert client.get("/api/models").json()["embedding"]["loaded"] is True
