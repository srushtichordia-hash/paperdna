"""Project a ``KnowledgeGraph`` into a React Flow nodes/edges payload.

Deterministic: the same graph always produces the same positions. The
layout is a simple, dependency-free column layout — one column per node
type (papers, claims, evidence, conditions) — computed here in Python so
the frontend needs no layout library. Node ``data`` and edge ``data``
carry the full upstream fields (including model-generated relation
attributes on edges), so the frontend can render detail panels without
re-deriving anything.

Positions are pure presentation; every node/edge ``data`` payload is a
verbatim projection of the graph node/edge, never embellished.
"""

from __future__ import annotations

from paperdna.graph.schema import (
    ClaimConditionEdge,
    ClaimEvidenceEdge,
    ClaimNode,
    ClaimPaperEdge,
    ClaimRelationEdge,
    ConditionNode,
    Edge,
    EvidenceNode,
    KnowledgeGraph,
    Node,
    PaperNode,
)

# Column order (left -> right) and horizontal gap between columns.
_COLUMN_ORDER = ("paper", "claim", "evidence", "condition")
_COLUMN_X = {node_type: 260.0 * index for index, node_type in enumerate(_COLUMN_ORDER)}
_ROW_GAP = 150.0
_ROW_OFFSET = 80.0


def _layout_rows(graph: KnowledgeGraph) -> dict[str, tuple[float, float]]:
    """Assign each node a (x, y) position, deterministic by node order.

    Nodes are walked in graph order; within each type-column they stack
    top-to-bottom. This keeps identical graphs pixel-identical.
    """
    row_counters: dict[str, int] = {node_type: 0 for node_type in _COLUMN_ORDER}
    positions: dict[str, tuple[float, float]] = {}
    for node in graph.nodes:
        column = _COLUMN_X[node.type]
        row = row_counters[node.type]
        positions[node.id] = (column, _ROW_OFFSET + _ROW_GAP * row)
        row_counters[node.type] += 1
    return positions


def _node_data(node: Node) -> dict[str, object]:
    """Type-specific node payload, copied verbatim from the graph node."""
    if isinstance(node, PaperNode):
        return {
            "kind": "paper",
            "source": node.source,
            "sourceId": node.source_id,
            "title": node.title,
            "url": node.url,
            "doi": node.doi,
            "published": node.published,
            "authors": list(node.authors),
        }
    if isinstance(node, ClaimNode):
        return {
            "kind": "claim",
            "source": node.source,
            "sourceId": node.source_id,
            "sentence": node.sentence,
            "start": node.start,
            "end": node.end,
            "cue": node.cue,
            "hedges": list(node.hedges) if node.hedges is not None else None,
        }
    if isinstance(node, EvidenceNode):
        return {
            "kind": "evidence",
            "claimId": node.claim_id,
            "sentence": node.sentence,
            "verification": node.verification,
            "unavailableReason": node.unavailable_reason,
            "start": node.start,
            "end": node.end,
        }
    if isinstance(node, ConditionNode):
        return {
            "kind": "condition",
            "claimId": node.claim_id,
            "category": node.category,
            "value": node.value,
            "start": node.start,
            "end": node.end,
            "ambiguous": node.ambiguous,
            "globalStart": node.global_start,
            "globalEnd": node.global_end,
        }
    raise TypeError(f"unknown node type: {type(node).__name__}")


def _edge_payload(edge: Edge) -> tuple[str | None, dict[str, object]]:
    """Return (label, data) for one edge.

    ``label`` is a short display string (e.g. the relation role); ``data``
    carries the full typed payload. Provenance edges carry no model
    output; ``claim_relation`` carries the NLI prediction verbatim.
    """
    if isinstance(edge, ClaimPaperEdge):
        return "extracted from", {"kind": "claim_paper"}
    if isinstance(edge, ClaimEvidenceEdge):
        return "has evidence", {"kind": "claim_evidence"}
    if isinstance(edge, ClaimConditionEdge):
        return "has condition", {"kind": "claim_condition"}
    if isinstance(edge, ClaimRelationEdge):
        conditions = None
        if edge.conditions is not None:
            conditions = {
                "premiseSource": edge.conditions.premise_source,
                "premiseSourceId": edge.conditions.premise_source_id,
                "hypothesisSource": edge.conditions.hypothesis_source,
                "hypothesisSourceId": edge.conditions.hypothesis_source_id,
                "fields": [
                    {
                        "category": field.category,
                        "status": field.status,
                        "premiseValues": list(field.premise_values),
                        "hypothesisValues": list(field.hypothesis_values),
                    }
                    for field in edge.conditions.fields
                ],
                "stats": dict(edge.conditions.stats),
            }
        return edge.role, {
            "kind": "claim_relation",
            "label": edge.label,
            "role": edge.role,
            "scores": dict(edge.scores) if edge.scores is not None else None,
            "confidence": edge.confidence,
            "uncertain": edge.uncertain,
            "hedged": edge.hedged,
            "truncated": edge.truncated,
            "model": edge.model,
            "conditions": conditions,
            "notes": list(edge.notes),
            "scoresAreCalibrated": False,
        }
    raise TypeError(f"unknown edge type: {type(edge).__name__}")


def graph_to_flow(graph: KnowledgeGraph) -> dict[str, list[dict[str, object]]]:
    """Project one ``KnowledgeGraph`` into ``{"nodes": [...], "edges": [...]}``.

    Every node gets an ``id``, ``type``, deterministic ``position``, and
    typed ``data``. Every edge gets ``id``, ``source``, ``target``,
    ``type``, an optional display ``label``, and typed ``data``. The
    result is JSON-serializable and stable for equal graphs.
    """
    if not isinstance(graph, KnowledgeGraph):
        raise TypeError(f"expected KnowledgeGraph, got {type(graph).__name__}")

    positions = _layout_rows(graph)
    nodes: list[dict[str, object]] = [
        {
            "id": node.id,
            "type": node.type,
            "position": {"x": positions[node.id][0], "y": positions[node.id][1]},
            "data": _node_data(node),
        }
        for node in graph.nodes
    ]
    edges: list[dict[str, object]] = []
    for edge in graph.edges:
        label, data = _edge_payload(edge)
        edges.append(
            {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "label": label,
                "data": data,
            }
        )
    return {"nodes": nodes, "edges": edges}
