"""Deterministic JSON-safe serialization and exact round-trip for graphs.

The dict shape is a documented public surface (mirroring
:mod:`paperdna.evidence.serialize`): plain lists/dicts/str/float/bool/
None only, so ``json.dumps`` always succeeds. Nodes and edges are
emitted in a canonical order (by type rank, then id), so two graphs
with the same content serialize to byte-identical JSON regardless of
insertion order.

``graph_from_dict`` is strict — missing or malformed fields, unknown
types or versions, and declared ids that disagree with the payload all
raise ``ValueError``; dangling edge references raise
``GraphValidationError`` (from ``KnowledgeGraph`` construction). Nothing
is guessed.
"""

from __future__ import annotations

import json
from typing import Any

from paperdna.conditions.schema import ConditionComparison, FieldComparison
from paperdna.graph.schema import (
    EDGE_TYPES,
    GRAPH_FORMAT,
    GRAPH_VERSION,
    NODE_TYPES,
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

# Rank of each node/edge type for canonical ordering.
_NODE_TYPE_RANK = {node_type: rank for rank, node_type in enumerate(NODE_TYPES)}
_EDGE_TYPE_RANK = {edge_type: rank for rank, edge_type in enumerate(EDGE_TYPES)}


def _require(data: dict, key: str) -> object:
    if key not in data:
        raise ValueError(f"malformed graph dict: missing key {key!r}")
    return data[key]


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _node_to_dict(node: Node) -> dict:
    if isinstance(node, PaperNode):
        attrs: dict[str, Any] = {
            "source": node.source,
            "source_id": node.source_id,
            "title": node.title,
            "url": node.url,
            "doi": node.doi,
            "published": node.published,
            "authors": list(node.authors),
        }
    elif isinstance(node, ClaimNode):
        attrs = {
            "source": node.source,
            "source_id": node.source_id,
            "sentence": node.sentence,
            "start": node.start,
            "end": node.end,
            "cue": node.cue,
            "hedges": list(node.hedges) if node.hedges is not None else None,
        }
    elif isinstance(node, EvidenceNode):
        attrs = {
            "claim_id": node.claim_id,
            "sentence": node.sentence,
            "verification": node.verification,
            "unavailable_reason": node.unavailable_reason,
            "start": node.start,
            "end": node.end,
        }
    elif isinstance(node, ConditionNode):
        attrs = {
            "claim_id": node.claim_id,
            "category": node.category,
            "value": node.value,
            "start": node.start,
            "end": node.end,
            "ambiguous": node.ambiguous,
            "global_start": node.global_start,
            "global_end": node.global_end,
        }
    else:  # pragma: no cover - defensive; KnowledgeGraph is typed
        raise TypeError(f"unknown node type: {type(node).__name__}")
    return {"id": node.id, "type": node.type, "attrs": attrs}


def _node_from_dict(data: dict) -> Node:
    node_type = str(_require(data, "type"))
    attrs = _require(data, "attrs")
    if not isinstance(attrs, dict):
        raise ValueError("malformed graph dict: node 'attrs' must be a dict")
    if node_type == "paper":
        authors = _require(attrs, "authors")
        if not isinstance(authors, list):
            raise ValueError("malformed graph dict: 'authors' must be a list")
        node: Node = PaperNode(
            source=str(_require(attrs, "source")),
            source_id=str(_require(attrs, "source_id")),
            title=_optional_str(attrs.get("title")),
            url=_optional_str(attrs.get("url")),
            doi=_optional_str(attrs.get("doi")),
            published=_optional_str(attrs.get("published")),
            authors=tuple(str(a) for a in authors),
        )
    elif node_type == "claim":
        hedges = attrs.get("hedges")
        if hedges is not None and not isinstance(hedges, list):
            raise ValueError("malformed graph dict: 'hedges' must be a list or None")
        node = ClaimNode(
            source=str(_require(attrs, "source")),
            source_id=str(_require(attrs, "source_id")),
            sentence=str(_require(attrs, "sentence")),
            start=_optional_int(_require(attrs, "start")),
            end=_optional_int(_require(attrs, "end")),
            cue=_optional_str(attrs.get("cue")),
            hedges=tuple(str(h) for h in hedges) if hedges is not None else None,
        )
    elif node_type == "evidence":
        node = EvidenceNode(
            claim_id=str(_require(attrs, "claim_id")),
            sentence=str(_require(attrs, "sentence")),
            verification=str(_require(attrs, "verification")),
            unavailable_reason=_optional_str(attrs.get("unavailable_reason")),
            start=_optional_int(_require(attrs, "start")),
            end=_optional_int(_require(attrs, "end")),
        )
    elif node_type == "condition":
        node = ConditionNode(
            claim_id=str(_require(attrs, "claim_id")),
            category=str(_require(attrs, "category")),
            value=str(_require(attrs, "value")),
            start=int(_require(attrs, "start")),  # type: ignore[arg-type]
            end=int(_require(attrs, "end")),  # type: ignore[arg-type]
            ambiguous=bool(_require(attrs, "ambiguous")),  # type: ignore[arg-type]
            global_start=_optional_int(attrs.get("global_start")),
            global_end=_optional_int(attrs.get("global_end")),
        )
    else:
        raise ValueError(f"malformed graph dict: unknown node type {node_type!r}")

    declared_id = str(_require(data, "id"))
    if node.id != declared_id:
        raise ValueError(
            f"malformed graph dict: node id {declared_id!r} does not match "
            f"payload-derived id {node.id!r}"
        )
    return node


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def _conditions_to_dict(comparison: ConditionComparison) -> dict:
    """Same dict shape as ``paperdna.evidence.serialize`` conditions."""
    return {
        "premise_source": comparison.premise_source,
        "premise_source_id": comparison.premise_source_id,
        "hypothesis_source": comparison.hypothesis_source,
        "hypothesis_source_id": comparison.hypothesis_source_id,
        "fields": [
            {
                "category": field.category,
                "status": field.status,
                "premise_values": list(field.premise_values),
                "hypothesis_values": list(field.hypothesis_values),
            }
            for field in comparison.fields
        ],
        "stats": dict(comparison.stats),
    }


def _conditions_from_dict(data: dict) -> ConditionComparison:
    fields = _require(data, "fields")
    if not isinstance(fields, list):
        raise ValueError("malformed graph dict: conditions 'fields' must be a list")
    stats = _require(data, "stats")
    if not isinstance(stats, dict):
        raise ValueError("malformed graph dict: conditions 'stats' must be a dict")
    return ConditionComparison(
        premise_source=str(_require(data, "premise_source")),
        premise_source_id=str(_require(data, "premise_source_id")),
        hypothesis_source=str(_require(data, "hypothesis_source")),
        hypothesis_source_id=str(_require(data, "hypothesis_source_id")),
        fields=tuple(
            FieldComparison(
                category=str(_require(item, "category")),
                status=str(_require(item, "status")),
                premise_values=tuple(str(v) for v in _require(item, "premise_values")),  # type: ignore[union-attr]
                hypothesis_values=tuple(str(v) for v in _require(item, "hypothesis_values")),  # type: ignore[union-attr]
            )
            for item in fields  # type: ignore[union-attr]
        ),
        stats={str(k): int(v) for k, v in stats.items()},
    )


def _edge_to_dict(edge: Edge) -> dict:
    attrs: dict[str, Any] = {}
    if isinstance(edge, ClaimRelationEdge):
        scores = dict(edge.scores) if edge.scores is not None else None
        attrs = {
            "label": edge.label,
            "scores": scores,
            "confidence": edge.confidence,
            "uncertain": edge.uncertain,
            "hedged": edge.hedged,
            "truncated": edge.truncated,
            "model": edge.model,
            "conditions": (
                _conditions_to_dict(edge.conditions)
                if edge.conditions is not None
                else None
            ),
            "notes": list(edge.notes),
        }
    return {
        "id": edge.id,
        "type": edge.type,
        "source": edge.source,
        "target": edge.target,
        "attrs": attrs,
    }


def _edge_from_dict(data: dict) -> Edge:
    edge_type = str(_require(data, "type"))
    source = str(_require(data, "source"))
    target = str(_require(data, "target"))
    attrs = _require(data, "attrs")
    if not isinstance(attrs, dict):
        raise ValueError("malformed graph dict: edge 'attrs' must be a dict")

    if edge_type == "claim_paper":
        edge: Edge = ClaimPaperEdge(source=source, target=target)
    elif edge_type == "claim_evidence":
        edge = ClaimEvidenceEdge(source=source, target=target)
    elif edge_type == "claim_condition":
        edge = ClaimConditionEdge(source=source, target=target)
    elif edge_type == "claim_relation":
        scores = attrs.get("scores")
        if scores is not None and not isinstance(scores, dict):
            raise ValueError("malformed graph dict: 'scores' must be a dict or None")
        notes = attrs.get("notes", [])
        if not isinstance(notes, list):
            raise ValueError("malformed graph dict: 'notes' must be a list")
        conditions = attrs.get("conditions")
        edge = ClaimRelationEdge(
            source=source,
            target=target,
            label=_optional_str(attrs.get("label")),
            scores={str(k): float(v) for k, v in scores.items()}  # type: ignore[union-attr]
            if scores is not None
            else None,
            confidence=_optional_float(attrs.get("confidence")),
            uncertain=_optional_bool(attrs.get("uncertain")),
            hedged=_optional_bool(attrs.get("hedged")),
            truncated=_optional_bool(attrs.get("truncated")),
            model=_optional_str(attrs.get("model")),
            conditions=(
                _conditions_from_dict(conditions)
                if conditions is not None
                else None
            ),
            notes=tuple(str(n) for n in notes),
        )
    else:
        raise ValueError(f"malformed graph dict: unknown edge type {edge_type!r}")

    declared_id = str(_require(data, "id"))
    if edge.id != declared_id:
        raise ValueError(
            f"malformed graph dict: edge id {declared_id!r} does not match "
            f"payload-derived id {edge.id!r}"
        )
    return edge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def graph_to_dict(graph: KnowledgeGraph) -> dict:
    """JSON-safe dict for one graph (round-trips exactly).

    Nodes and edges are emitted in canonical order: nodes by (type
    rank, id), edges by (type rank, source, target, id).
    """
    if not isinstance(graph, KnowledgeGraph):
        raise TypeError(f"expected KnowledgeGraph, got {type(graph).__name__}")
    nodes = sorted(graph.nodes, key=lambda n: (_NODE_TYPE_RANK[n.type], n.id))
    edges = sorted(
        graph.edges,
        key=lambda e: (_EDGE_TYPE_RANK[e.type], e.source, e.target, e.id),
    )
    return {
        "format": GRAPH_FORMAT,
        "version": GRAPH_VERSION,
        "nodes": [_node_to_dict(node) for node in nodes],
        "edges": [_edge_to_dict(edge) for edge in edges],
    }


def graph_from_dict(data: dict) -> KnowledgeGraph:
    """Rebuild a graph from its dict form.

    Round-trips exactly (``graph_from_dict(graph_to_dict(g)) == g``).
    Missing or malformed fields raise ``ValueError``; edges referencing
    nonexistent nodes raise ``GraphValidationError``; nothing is
    guessed.
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected dict, got {type(data).__name__}")
    if _require(data, "format") != GRAPH_FORMAT:
        raise ValueError(
            f"unsupported graph format {_require(data, 'format')!r} "
            f"(expected {GRAPH_FORMAT!r})"
        )
    version = _require(data, "version")
    if version != GRAPH_VERSION:
        raise ValueError(
            f"unsupported graph version {version!r} (expected {GRAPH_VERSION})"
        )
    raw_nodes = _require(data, "nodes")
    raw_edges = _require(data, "edges")
    if not isinstance(raw_nodes, list):
        raise ValueError("malformed graph dict: 'nodes' must be a list")
    if not isinstance(raw_edges, list):
        raise ValueError("malformed graph dict: 'edges' must be a list")
    nodes = [_node_from_dict(item) for item in raw_nodes]  # type: ignore[arg-type]
    edges = [_edge_from_dict(item) for item in raw_edges]  # type: ignore[arg-type]
    return KnowledgeGraph(nodes, edges)


def graph_to_json(graph: KnowledgeGraph, *, indent: int = 2) -> str:
    """Deterministic JSON text: sorted keys, fixed indentation, trailing
    newline. Byte-identical for equal graphs."""
    return json.dumps(graph_to_dict(graph), sort_keys=True, indent=indent) + "\n"


def graph_from_json(text: str) -> KnowledgeGraph:
    """Parse JSON text produced by ``graph_to_json`` (or any conforming
    writer); use ``path.read_text()`` for files."""
    if not isinstance(text, str):
        raise ValueError(f"expected str, got {type(text).__name__}")
    return graph_from_dict(json.loads(text))
