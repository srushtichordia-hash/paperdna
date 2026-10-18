"""Tests for graph node/edge schemas, ids, and invariants."""

from __future__ import annotations

import pytest

from paperdna.evidence import claim_id_for
from paperdna.graph import (
    NODE_TYPES,
    RELATION_ROLES,
    ClaimConditionEdge,
    ClaimEvidenceEdge,
    ClaimNode,
    ClaimPaperEdge,
    ClaimRelationEdge,
    ConditionNode,
    EvidenceNode,
    KnowledgeGraph,
    PaperNode,
    claim_node_id,
    condition_node_id,
    edge_id,
    evidence_node_id,
    paper_node_id,
)
from tests.graph.conftest import (
    DEFAULT_SENTENCE,
    HYPOTHESIS_SENTENCE,
    make_classification,
)


class TestIds:
    def test_paper_id_casefolds_source(self):
        assert paper_node_id("arXiv", "2101.00001") == "paper:arxiv:2101.00001"
        assert PaperNode(source="arXiv", source_id="2101.00001").id == (
            "paper:arxiv:2101.00001"
        )

    def test_claim_id_reuses_evidence_format(self):
        assert claim_node_id("arxiv", "2101.00001", 25, 80) == claim_id_for(
            "arxiv", "2101.00001", 25, 80
        )
        assert claim_node_id("arxiv", "2101.00001", None, None) == (
            "arxiv:2101.00001:unknown"
        )

    def test_evidence_and_condition_id_formats(self):
        assert evidence_node_id("a:b:1-2") == "evidence:a:b:1-2"
        assert condition_node_id("a:b:1-2", 3, 8) == "condition:a:b:1-2:3-8"

    def test_edge_id_is_uniform(self):
        assert (
            edge_id("claim_relation", "a:b:1-2", "c:d:3-4")
            == "edge:claim_relation:a:b:1-2->c:d:3-4"
        )
        assert (
            ClaimRelationEdge(source="a:b:1-2", target="c:d:3-4").id
            == "edge:claim_relation:a:b:1-2->c:d:3-4"
        )

    def test_ids_are_deterministic_and_order_independent(self):
        first = ClaimNode("arxiv", "1", "text", 0, 4)
        second = ClaimNode("arxiv", "1", "text", 0, 4)
        assert first.id == second.id
        assert first == second


class TestClaimNode:
    def test_offsets_must_both_be_present(self):
        with pytest.raises(ValueError):
            ClaimNode("arxiv", "1", "text", None, 5)
        with pytest.raises(ValueError):
            ClaimNode("arxiv", "1", "text", 0, None)

    def test_offsets_must_be_sane(self):
        with pytest.raises(ValueError):
            ClaimNode("arxiv", "1", "text", -1, 5)
        with pytest.raises(ValueError):
            ClaimNode("arxiv", "1", "text", 6, 5)

    def test_unknown_offsets_allowed(self):
        node = ClaimNode("arxiv", "1", "text", None, None)
        assert node.id.endswith(":unknown")


class TestEvidenceNode:
    def test_verification_must_be_known(self):
        with pytest.raises(ValueError):
            EvidenceNode("a:b:1-2", "text", "maybe")

    def test_unverified_and_unavailable_require_reason(self):
        with pytest.raises(ValueError):
            EvidenceNode("a:b:1-2", "text", "unverified")
        with pytest.raises(ValueError):
            EvidenceNode("a:b:1-2", "text", "unavailable")
        unverified = EvidenceNode(
            "a:b:1-2", "text", "unverified", "paper record not found"
        )
        # Verbatim text exists, but offsets were never checked.
        assert unverified.available and not unverified.verified
        unavailable = EvidenceNode(
            "a:b:1-2", "", "unavailable", "empty evidence sentence"
        )
        assert not unavailable.available and not unavailable.verified

    def test_available_and_verified_flags(self):
        verified = EvidenceNode("a:b:1-2", "text", "verified", start=1, end=2)
        assert verified.available and verified.verified
        by_search = EvidenceNode("a:b:1-2", "text", "verified_by_search")
        assert by_search.available and by_search.verified
        unavailable = EvidenceNode(
            "a:b:1-2", "", "unavailable", "empty evidence sentence"
        )
        assert not unavailable.available and not unavailable.verified

    def test_offsets_must_both_be_present(self):
        with pytest.raises(ValueError):
            EvidenceNode("a:b:1-2", "text", "verified", start=None, end=3)


class TestConditionNode:
    def test_category_must_be_known(self):
        with pytest.raises(ValueError):
            ConditionNode("a:b:1-2", "weather", "sunny", 0, 5, False)

    def test_offsets_sane_and_global_both_or_none(self):
        with pytest.raises(ValueError):
            ConditionNode("a:b:1-2", "dataset", "GLUE", -1, 4, False)
        with pytest.raises(ValueError):
            ConditionNode("a:b:1-2", "dataset", "GLUE", 0, 4, False, global_start=3)
        node = ConditionNode(
            "a:b:1-2", "dataset", "GLUE", 0, 4, False, global_start=3, global_end=7
        )
        assert node.global_start == 3 and node.global_end == 7


class TestClaimRelationEdge:
    def test_role_derived_from_label(self):
        assert (
            ClaimRelationEdge(source="a", target="b", label="entailment").role
            == "supports"
        )
        assert (
            ClaimRelationEdge(source="a", target="b", label="contradiction").role
            == "contradicts"
        )
        assert ClaimRelationEdge(source="a", target="b", label="neutral").role == "neutral"
        assert ClaimRelationEdge(source="a", target="b").role is None

    def test_relation_roles_are_the_three_labels(self):
        assert sorted(RELATION_ROLES) == ["contradiction", "entailment", "neutral"]

    def test_edge_is_directed_premise_to_hypothesis(self):
        edge = ClaimRelationEdge(source="premise-id", target="hypothesis-id")
        assert edge.source == "premise-id"
        assert edge.target == "hypothesis-id"
        assert edge.type == "claim_relation"


class TestKnowledgeGraphBasics:
    def test_empty_graph_is_valid(self):
        graph = KnowledgeGraph((), ())
        assert len(graph) == 0
        assert graph.nodes == () and graph.edges == ()
        assert graph.stats["nodes"] == 0 and graph.stats["edges"] == 0

    def test_node_lookup_and_contains(self):
        paper = PaperNode("arxiv", "1", title="T")
        graph = KnowledgeGraph([paper], [])
        assert paper.id in graph
        assert graph.node(paper.id) == paper
        assert graph.node("nope") is None
        assert "nope" not in graph

    def test_stats_fixed_keys(self):
        claim = ClaimNode("arxiv", "1", DEFAULT_SENTENCE, 0, len(DEFAULT_SENTENCE))
        graph = KnowledgeGraph([claim], [])
        stats = graph.stats
        for node_type in NODE_TYPES:
            assert node_type in stats
        assert stats["claim"] == 1
        assert stats["nodes"] == 1

    def test_repr_mentions_counts(self):
        graph = KnowledgeGraph([], [])
        assert "papers=0" in repr(graph) and "claims=0" in repr(graph)

    def test_equality_is_order_insensitive(self):
        a = ClaimNode("arxiv", "1", "one", 0, 3)
        b = ClaimNode("arxiv", "2", "two", 0, 3)
        assert KnowledgeGraph([a, b], []) == KnowledgeGraph([b, a], [])
        assert KnowledgeGraph([a], []) != KnowledgeGraph([b], [])

    def test_edge_types_have_expected_endpoints(self):
        # Smoke: schema edge classes expose typed endpoints via docs;
        # endpoint compatibility is enforced by KnowledgeGraph tests.
        claim = ClaimNode("arxiv", "1", "x", 0, 1)
        paper = PaperNode("arxiv", "1")
        edge = ClaimPaperEdge(source=claim.id, target=paper.id)
        graph = KnowledgeGraph([claim, paper], [edge])
        assert graph.edges == (edge,)


class TestFixtureShapes:
    def test_claim_matches_conftest_factory(self):
        from tests.graph.conftest import make_claim

        claim = make_claim()
        node = ClaimNode(
            claim.source,
            claim.source_id,
            claim.sentence,
            claim.start,
            claim.end,
            claim.cue,
            tuple(claim.hedges),
        )
        assert node.id == claim_node_id(
            claim.source, claim.source_id, claim.start, claim.end
        )
        assert claim.sentence == DEFAULT_SENTENCE
        assert HYPOTHESIS_SENTENCE != DEFAULT_SENTENCE
        classification = make_classification()
        assert classification.premise_text == DEFAULT_SENTENCE

    def test_claim_condition_edge_types(self):
        claim = ClaimNode("arxiv", "1", "x", 0, 1)
        condition = ConditionNode(claim.id, "dataset", "GLUE", 0, 4, False)
        edge = ClaimConditionEdge(source=claim.id, target=condition.id)
        assert edge.type == "claim_condition"
        evidence = EvidenceNode(claim.id, "x", "verified", start=0, end=1)
        assert (
            ClaimEvidenceEdge(source=claim.id, target=evidence.id).type
            == "claim_evidence"
        )
