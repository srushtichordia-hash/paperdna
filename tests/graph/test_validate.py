"""Tests for graph validation: dangling refs, types, labels, aggregates."""

from __future__ import annotations

import pytest

from paperdna.graph import (
    ClaimNode,
    ClaimPaperEdge,
    ClaimRelationEdge,
    GraphValidationError,
    KnowledgeGraph,
    PaperNode,
    claim_node_id,
    evidence_node_id,
    paper_node_id,
)
from paperdna.evidence.schema import claim_id_for


def _claim(source="arxiv", source_id="1", start=0, end=4) -> ClaimNode:
    return ClaimNode(source, source_id, "text", start, end)


def _paper(source="arxiv", source_id="1") -> PaperNode:
    return PaperNode(source, source_id)


class TestDanglingReferences:
    def test_unknown_target_rejected(self):
        claim = _claim()
        edge = ClaimPaperEdge(source=claim.id, target=paper_node_id("arxiv", "2"))
        with pytest.raises(GraphValidationError) as excinfo:
            KnowledgeGraph([claim], [edge])
        assert "references unknown target node" in str(excinfo.value)

    def test_unknown_source_rejected(self):
        paper = _paper()
        edge = ClaimPaperEdge(source=claim_node_id("arxiv", "9", 0, 4), target=paper.id)
        with pytest.raises(GraphValidationError) as excinfo:
            KnowledgeGraph([paper], [edge])
        assert "references unknown source node" in str(excinfo.value)

    def test_claim_relation_needs_claim_endpoints(self):
        claim = _claim()
        paper = _paper()
        edge = ClaimRelationEdge(source=claim.id, target=paper.id, label="neutral")
        with pytest.raises(GraphValidationError) as excinfo:
            KnowledgeGraph([claim, paper], [edge])
        message = str(excinfo.value)
        assert "expects a claim target node" in message


class TestWrongEndpointTypes:
    def test_reversed_claim_paper_edge(self):
        claim = _claim()
        paper = _paper()
        # source must be a claim, target a paper — reversed fails.
        edge = ClaimPaperEdge(source=paper.id, target=claim.id)
        with pytest.raises(GraphValidationError) as excinfo:
            KnowledgeGraph([claim, paper], [edge])
        message = str(excinfo.value)
        assert "expects a claim source node" in message
        assert "expects a paper target node" in message


class TestRelationPayload:
    def _edge(self, **overrides):
        fields = {
            "source": claim_node_id("arxiv", "1", 0, 4),
            "target": claim_node_id("arxiv", "2", 0, 4),
            "label": "neutral",
            "scores": {"entailment": 0.2, "contradiction": 0.2, "neutral": 0.6},
            "confidence": 0.6,
        }
        fields.update(overrides)
        return ClaimRelationEdge(**fields)

    def _graph_with(self, edge):
        return KnowledgeGraph(
            [
                _claim(source_id="1"),
                _claim(source="arxiv", source_id="2"),
            ],
            [edge],
        )

    def test_invalid_label_rejected(self):
        with pytest.raises(GraphValidationError) as excinfo:
            self._graph_with(self._edge(label="proves"))
        assert "invalid relation label" in str(excinfo.value)

    def test_none_label_allowed(self):
        graph = self._graph_with(self._edge(label=None, scores=None, confidence=None))
        assert graph.stats["claim_relation"] == 1

    def test_incomplete_scores_rejected(self):
        with pytest.raises(GraphValidationError) as excinfo:
            self._graph_with(self._edge(scores={"neutral": 1.0}))
        assert "scores must cover exactly" in str(excinfo.value)

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(GraphValidationError):
            self._graph_with(self._edge(confidence=1.5))
        with pytest.raises(GraphValidationError):
            self._graph_with(self._edge(confidence=-0.1))

    def test_conditions_must_be_condition_comparison(self):
        with pytest.raises(GraphValidationError) as excinfo:
            self._graph_with(self._edge(conditions="lots of them"))
        assert "ConditionComparison" in str(excinfo.value)


class TestDuplicates:
    def test_duplicate_node_ids_rejected(self):
        with pytest.raises(GraphValidationError) as excinfo:
            KnowledgeGraph([_claim(), _claim()], [])
        assert "duplicate node id" in str(excinfo.value)

    def test_duplicate_edge_ids_rejected(self):
        claim = _claim()
        paper = _paper()
        edge = ClaimPaperEdge(source=claim.id, target=paper.id)
        with pytest.raises(GraphValidationError) as excinfo:
            KnowledgeGraph([claim, paper], [edge, ClaimPaperEdge(source=claim.id, target=paper.id)])
        assert "duplicate edge id" in str(excinfo.value)


class TestAggregation:
    def test_all_problems_reported_together(self):
        orphan_edge = ClaimPaperEdge(
            source=claim_node_id("arxiv", "9", 0, 4),
            target=paper_node_id("arxiv", "1"),
        )
        bad_relation = ClaimRelationEdge(
            source=claim_node_id("arxiv", "1", 0, 4),
            target=paper_node_id("arxiv", "1"),  # wrong: must be a claim
            label="proves",  # wrong: not a RELATIONSHIPS label
        )
        with pytest.raises(GraphValidationError) as excinfo:
            KnowledgeGraph(
                [_claim(), _claim()],  # duplicate node id too
                [orphan_edge, bad_relation],
            )
        message = str(excinfo.value)
        assert "duplicate node id" in message
        assert "references unknown source node" in message
        assert "invalid relation label" in message

    def test_validate_rechecks_constructed_graph(self):
        graph = KnowledgeGraph([_claim()], [])
        graph.validate()  # no raise


class TestValidGraphPasses:
    def test_linked_claim_paper_pair(self):
        claim = _claim()
        paper = _paper()
        graph = KnowledgeGraph(
            [claim, paper], [ClaimPaperEdge(source=claim.id, target=paper.id)]
        )
        graph.validate()
        assert graph.stats["claim_paper"] == 1

    def test_evidence_id_derived_from_claim_id(self):
        # Evidence ids incorporate the claim id, including "unknown".
        unknown_claim = claim_id_for("arxiv", "1", None, None)
        assert evidence_node_id(unknown_claim) == "evidence:arxiv:1:unknown"
