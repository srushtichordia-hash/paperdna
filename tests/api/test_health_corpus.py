"""Tests for the read-only endpoints: health, corpus, models status."""

from __future__ import annotations


def test_health_reports_ok_and_corpus_stats(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    stats = body["corpus"]
    # The bundled demo corpus has 10 valid papers.
    assert stats["loaded"] == 10
    assert stats["unique"] == 10
    assert stats["duplicates"] == 0
    assert stats["rejected"] == 0


def test_health_reports_model_status(client):
    body = client.get("/api/health").json()
    models = body["models"]
    assert set(models) == {"embedding", "nli"}
    for entry in models.values():
        assert isinstance(entry["installed"], bool)
        assert entry["loaded"] is False  # nothing loaded at startup
        assert entry["name"] is None


def test_corpus_lists_all_demo_papers(client):
    response = client.get("/api/corpus")
    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["unique"] == 10
    assert len(body["papers"]) == 10
    for paper in body["papers"]:
        assert paper["source"]
        assert paper["sourceId"]
        # Ranked fields are absent for a plain corpus listing.
        assert paper["score"] is None
        assert paper["rankingKind"] is None


def test_models_status_reflects_empty_registry(client):
    body = client.get("/api/models").json()
    assert body["embedding"]["loaded"] is False
    assert body["nli"]["loaded"] is False


def test_models_status_reflects_injected_fakes(semantic_client):
    body = semantic_client.get("/api/models").json()
    assert body["embedding"]["loaded"] is True
    assert body["embedding"]["name"] == "fake-embedder"
    assert body["nli"]["loaded"] is True
    assert body["nli"]["name"] == "fake-nli"
