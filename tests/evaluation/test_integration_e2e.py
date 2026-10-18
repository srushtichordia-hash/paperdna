"""End-to-end integration tests across every pipeline module.

One claim flows through retrieval → claim extraction → NLI → condition
comparison → evidence attribution → knowledge graph → API response,
and cross-module invariants are asserted on the ACTUAL outputs
(verbatim evidence, honest user-claim provenance, graph/claim count
consistency, deterministic reruns).
"""

from __future__ import annotations

from paperdna.api.pipeline import AppInputs, run_discovery
from paperdna.nli.schema import RELATIONSHIPS

CLAIM = (
    "Retrieval augmentation improves factuality in closed-book "
    "question answering."
)


class TestPipelineEndToEnd:
    def test_full_mode_cross_module_invariants(self, fake_embedder, fake_classifier):
        inputs = AppInputs(claim=CLAIM, top_k=5, max_pairs=50)
        result = run_discovery(
            inputs, embedder=fake_embedder, classifier=fake_classifier
        )

        # Retrieval: ranked papers come from the ingested corpus, bounded.
        paper_ids = {f"{p.source}:{p.source_id}" for p in result.papers}
        ranked_ids = [f"{p.record.source}:{p.record.source_id}" for p in result.ranked]
        assert set(ranked_ids) <= paper_ids
        assert len(ranked_ids) <= inputs.top_k
        assert result.capabilities.ranking == "semantic"

        # Claim extraction: every claim is a verbatim abstract span.
        abstracts = {(p.source, p.source_id): p.abstract for p in result.papers}
        assert result.paper_claims, "demo corpus must produce claims"
        for claim in result.paper_claims:
            abstract = abstracts[(claim.source, claim.source_id)]
            assert abstract[claim.start:claim.end] == claim.sentence

        # Relations: labels are real relationships; evidence is honest.
        assert result.relations, "full mode must predict relationships"
        for view in result.relations:
            assert view.label in RELATIONSHIPS
            assert 0.0 <= view.confidence <= 1.0
            premise = view.attribution.premise
            hypothesis = view.attribution.hypothesis
            for side in (premise, hypothesis):
                if side.verification == "verified":
                    record_abstract = abstracts.get((side.claim.source, side.claim.source_id))
                    if record_abstract is not None:
                        assert record_abstract[side.claim.start:side.claim.end] == side.sentence
                else:
                    assert side.unavailable_reason
            if view.kind == "user_vs_paper":
                # Text-only user claim: paper side keeps real provenance,
                # the user side never invents offsets.
                assert premise.verification == "verified"
                assert hypothesis.claim.start is None
                assert hypothesis.claim.claim_id.endswith(":unknown")

        # Graph: built from the same outputs, counts are consistent.
        # Every ranked paper is a paper node; each user_vs_paper relation
        # contributes the user's own pseudo-paper node as well.
        from paperdna.graph.schema import PaperNode

        stats = result.graph.stats
        paper_node_ids = {n.id for n in result.graph.nodes if isinstance(n, PaperNode)}
        assert {f"paper:{pid}" for pid in ranked_ids} <= paper_node_ids
        has_user_relation = any(v.kind == "user_vs_paper" for v in result.relations)
        assert stats["paper"] == len(ranked_ids) + (1 if has_user_relation else 0)
        assert stats["paper"] == len(paper_node_ids), "no duplicate paper nodes"
        assert stats["claim"] >= len(result.paper_claims)
        node_ids = {node.id for node in result.graph.nodes}
        for edge in result.graph.edges:
            assert edge.source in node_ids and edge.target in node_ids

    def test_full_mode_is_deterministic(self, fake_embedder, fake_classifier):
        inputs = AppInputs(claim=CLAIM, top_k=5, max_pairs=50)
        first = run_discovery(
            inputs, embedder=fake_embedder, classifier=fake_classifier
        )
        second = run_discovery(
            inputs, embedder=fake_embedder, classifier=fake_classifier
        )
        assert first == second

    def test_degraded_mode_is_honest(self):
        inputs = AppInputs(claim=CLAIM, top_k=5)
        result = run_discovery(inputs)  # no embedder, no classifier

        assert result.capabilities.ranking == "lexical"
        assert result.relations == ()
        assert any("lexical" in m.lower() for m in result.messages)
        assert any("relationship" in m.lower() for m in result.messages)
        # The graph still reflects the claims that WERE produced.
        assert result.graph.stats["claim"] >= 1
        assert all(
            p.kind == "lexical" for p in result.ranked
        ), "fallback ranking must never be labeled semantic"

    def test_degraded_mode_is_deterministic(self):
        inputs = AppInputs(claim=CLAIM, top_k=5)
        assert run_discovery(inputs) == run_discovery(inputs)


class TestApiRoundTrip:
    def _client(self, registry):
        from fastapi.testclient import TestClient
        from paperdna.api.app import create_app

        return TestClient(create_app(registry=registry))

    def test_analyze_response_matches_pipeline_counts(self, fake_embedder, fake_classifier):
        from paperdna.api.models import ModelRegistry

        registry = ModelRegistry()
        registry.embedder = fake_embedder
        registry.classifier = fake_classifier
        client = self._client(registry)

        body = client.post("/api/analyze", json={"claim": CLAIM, "topK": 5}).json()
        direct = run_discovery(
            AppInputs(claim=CLAIM, top_k=5),
            embedder=fake_embedder,
            classifier=fake_classifier,
        )
        assert body["rankingKind"] == direct.capabilities.ranking
        assert len(body["papers"]) == len(direct.ranked)
        assert len(body["paperClaims"]) == len(direct.paper_claims)
        assert len(body["relations"]) == len(direct.relations)
        assert len(body["graph"]["nodes"]) == len(direct.graph.nodes)
        assert len(body["graph"]["edges"]) == len(direct.graph.edges)

    def test_serialized_relations_keep_honesty_flags(self, fake_embedder, fake_classifier):
        from paperdna.api.models import ModelRegistry

        registry = ModelRegistry()
        registry.embedder = fake_embedder
        registry.classifier = fake_classifier
        client = self._client(registry)

        body = client.post(
            "/api/analyze", json={"claim": CLAIM, "topK": 5, "maxPairs": 50}
        ).json()
        assert body["relations"], "semantic mode must serialize relations"
        for rel in body["relations"]:
            interp = rel["interpretation"]
            assert interp["label"] in RELATIONSHIPS
            # Uncertainty is surfaced, never suppressed.
            assert isinstance(interp["uncertain"], bool)
            # Every relation links verbatim evidence on both sides.
            for side in (rel["premise"], rel["hypothesis"]):
                assert side["sentence"].strip()
                assert side["verification"]

    def test_degraded_analyze_round_trip(self):
        from paperdna.api.models import ModelRegistry

        client = self._client(ModelRegistry())
        body = client.post("/api/analyze", json={"claim": CLAIM}).json()
        assert body["rankingKind"] == "lexical"
        assert body["relations"] == []
        assert body["graph"]["nodes"], "graph is served even in degraded mode"
