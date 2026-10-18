"""Research evidence knowledge graph.

Papers, claims, evidence sentences, and experimental conditions as a
navigable, validated graph — built from the outputs of the upstream
pipeline stages (corpus, claims, NLI, conditions, evidence attribution)
through their public APIs, with no changes to their behavior.

What the graph guarantees:

- **Stable, typed identity**: every node and edge has a deterministic id
  derived from its content (claim/evidence ids reuse
  ``paperdna.evidence.claim_id_for``), independent of insertion order.
- **Verbatim evidence**: evidence nodes carry the exact sentence, the
  upstream verification status (``verified`` / ``verified_by_search`` /
  ``unverified`` / ``unavailable``), the stated unavailability reason,
  and abstract offsets when known. Missing provenance is preserved as
  ``None`` — never fabricated.
- **Predictions vs. facts**: ``CLAIM_RELATION`` edges hold NLI labels,
  softmax scores, confidence, and uncertainty flags as explicitly
  *model-generated* attributes; they never rewrite or assert scientific
  truth. Condition comparisons ride on the edge as context.
- **Referential integrity**: ``KnowledgeGraph`` construction rejects
  edges pointing at nonexistent nodes and invalid labels/scores.

Limitations: this is a *textual* evidence graph, not a scientific-truth
database — NLI labels are uncalibrated predictions, and verification
proves a sentence's provenance, not its correctness. Claim ids depend
on the current extractor's segmentation. Graphs are data structures
only; visualization is deferred (the deterministic JSON is its input).
See the README "Research evidence knowledge graph" section.
"""

from paperdna.graph.builder import GraphBuilder, build_graph
from paperdna.graph.errors import GraphConflictError, GraphError, GraphValidationError
from paperdna.graph.queries import (
    claims,
    condition_nodes,
    conditions_for_claim,
    conditions_for_relation,
    connected_components,
    contradicting_relations,
    evidence_for_claim,
    evidence_for_relation,
    evidence_nodes,
    neutral_relations,
    papers,
    paper_for_claim,
    relations,
    supporting_relations,
)
from paperdna.graph.schema import (
    EDGE_TYPES,
    GRAPH_FORMAT,
    GRAPH_VERSION,
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
from paperdna.graph.serialize import (
    graph_from_dict,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
)

__all__ = [
    "EDGE_TYPES",
    "GRAPH_FORMAT",
    "GRAPH_VERSION",
    "NODE_TYPES",
    "RELATION_ROLES",
    "ClaimConditionEdge",
    "ClaimEvidenceEdge",
    "ClaimNode",
    "ClaimPaperEdge",
    "ClaimRelationEdge",
    "ConditionNode",
    "EvidenceNode",
    "GraphBuilder",
    "GraphConflictError",
    "GraphError",
    "GraphValidationError",
    "KnowledgeGraph",
    "PaperNode",
    "build_graph",
    "claim_node_id",
    "claims",
    "condition_node_id",
    "condition_nodes",
    "conditions_for_claim",
    "conditions_for_relation",
    "connected_components",
    "contradicting_relations",
    "edge_id",
    "evidence_for_claim",
    "evidence_for_relation",
    "evidence_node_id",
    "evidence_nodes",
    "graph_from_dict",
    "graph_from_json",
    "graph_to_dict",
    "graph_to_json",
    "neutral_relations",
    "paper_for_claim",
    "paper_node_id",
    "papers",
    "relations",
    "supporting_relations",
]
