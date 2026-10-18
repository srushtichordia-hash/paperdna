"""Tests for POST /api/analyze: degraded mode, semantic mode, honesty rules."""

from __future__ import annotations

CLAIM = (
    "We find that retrieval augmentation improves question answering "
    "accuracy on dataset d1 by a clear margin."
)


def test_analyze_degraded_uses_lexical_ranking(client):
    response = client.post("/api/analyze", json={"claim": CLAIM})
    assert response.status_code == 200
    body = response.json()
    assert body["rankingKind"] == "lexical"
    # A message must state that semantic retrieval is off.
    assert any("lexical" in m.lower() for m in body["messages"])


def test_analyze_degraded_has_no_relations(client):
    body = client.post("/api/analyze", json={"claim": CLAIM}).json()
    assert body["relations"] == []
    # A message must state that relationship classification is off.
    assert any("relationship" in m.lower() for m in body["messages"])


def test_analyze_semantic_uses_semantic_ranking(semantic_client):
    body = semantic_client.post("/api/analyze", json={"claim": CLAIM}).json()
    assert body["rankingKind"] == "semantic"
    assert body["capabilities"]["embedderName"] == "fake-embedder"
    assert body["capabilities"]["nliModel"] == "fake-nli"


def test_analyze_semantic_produces_relations(semantic_client):
    body = semantic_client.post(
        "/api/analyze", json={"claim": CLAIM, "comparePapers": True, "maxPairs": 50}
    ).json()
    assert len(body["relations"]) > 0
    for rel in body["relations"]:
        interp = rel["interpretation"]
        assert interp is not None
        assert interp["label"] in ("entailment", "contradiction", "neutral")
        # Predictions are never presented as calibrated facts.
        assert interp["scoresAreCalibrated"] is False
        assert isinstance(interp["uncertain"], bool)


def test_analyze_text_only_user_claim_has_null_offsets(semantic_client):
    """A claim not found verbatim in the abstract keeps null offsets."""
    # The claim is a paraphrase, so it will not match verbatim.
    body = semantic_client.post(
        "/api/analyze", json={"claim": "Retrieval helps QA.", "comparePapers": True}
    ).json()
    assert body["userClaimHasRealOffsets"] is False
    # Every user-side evidence claim id is "unknown" with null offsets.
    for rel in body["relations"]:
        if rel["kind"] == "user_vs_paper":
            hyp = rel["hypothesis"]
            assert hyp["claim"]["source"] == "user"
            assert hyp["claim"]["start"] is None
            assert hyp["claim"]["end"] is None
            assert hyp["claim"]["claimId"] == "user:query:unknown"


def test_analyze_paper_premise_keeps_verified_provenance(semantic_client):
    """The paper premise of a user_vs_paper relation stays verified."""
    body = semantic_client.post(
        "/api/analyze", json={"claim": CLAIM, "comparePapers": True, "maxPairs": 50}
    ).json()
    user_rels = [r for r in body["relations"] if r["kind"] == "user_vs_paper"]
    assert user_rels
    for rel in user_rels:
        # The premise is a real paper claim with real abstract offsets.
        assert rel["premise"]["verification"] == "verified"
        assert rel["premise"]["claim"]["start"] is not None
        assert rel["premise"]["claim"]["end"] is not None


def test_analyze_graph_is_deterministic(semantic_client):
    payload = {"claim": CLAIM, "comparePapers": True, "maxPairs": 50}
    first = semantic_client.post("/api/analyze", json=payload).json()["graph"]
    second = semantic_client.post("/api/analyze", json=payload).json()["graph"]
    assert first == second


def test_analyze_graph_node_positions_present(semantic_client):
    body = semantic_client.post(
        "/api/analyze", json={"claim": CLAIM, "comparePapers": True, "maxPairs": 50}
    ).json()
    for node in body["graph"]["nodes"]:
        assert set(node["position"]) == {"x", "y"}
        assert node["type"] in ("paper", "claim", "evidence", "condition")


def test_analyze_empty_claim_is_rejected(client):
    response = client.post("/api/analyze", json={"claim": ""})
    assert response.status_code == 422


def test_analyze_too_long_claim_is_rejected(client):
    response = client.post("/api/analyze", json={"claim": "x" * 2000})
    assert response.status_code == 422


def test_analyze_top_k_out_of_range_is_rejected(client):
    response = client.post("/api/analyze", json={"claim": CLAIM, "topK": 99})
    assert response.status_code == 422


def test_analyze_missing_claim_is_rejected(client):
    response = client.post("/api/analyze", json={})
    assert response.status_code == 422
