"""Query functions over a :class:`~paperdna.graph.schema.KnowledgeGraph`.

All queries are deterministic (results sorted by node/edge id), return
empty results for unknown claims or empty graphs (never raise for a
missing claim), and validate only their own arguments. Relation queries
return ``(edge, other_claim)`` pairs: the edge is the directed
``CLAIM_RELATION`` (source = premise, target = hypothesis) and the other
claim is the node at the far end.
"""

from __future__ import annotations

from paperdna.conditions.schema import ConditionComparison
from paperdna.graph.schema import (
    ClaimNode,
    ClaimRelationEdge,
    ConditionNode,
    EvidenceNode,
    KnowledgeGraph,
    PaperNode,
)

_DIRECTIONS = ("out", "in", "both")


def _check_direction(direction: str) -> None:
    if direction not in _DIRECTIONS:
        raise ValueError(
            f"direction must be one of {_DIRECTIONS}, got {direction!r}"
        )


# ---------------------------------------------------------------------------
# Typed node listings
# ---------------------------------------------------------------------------


def papers(graph: KnowledgeGraph) -> tuple[PaperNode, ...]:
    """All paper nodes, sorted by id."""
    return tuple(
        sorted((n for n in graph.nodes if isinstance(n, PaperNode)), key=lambda n: n.id)
    )


def claims(graph: KnowledgeGraph) -> tuple[ClaimNode, ...]:
    """All claim nodes, sorted by id."""
    return tuple(
        sorted((n for n in graph.nodes if isinstance(n, ClaimNode)), key=lambda n: n.id)
    )


def evidence_nodes(graph: KnowledgeGraph) -> tuple[EvidenceNode, ...]:
    """All evidence nodes, sorted by id."""
    return tuple(
        sorted(
            (n for n in graph.nodes if isinstance(n, EvidenceNode)), key=lambda n: n.id
        )
    )


def condition_nodes(graph: KnowledgeGraph) -> tuple[ConditionNode, ...]:
    """All condition nodes, sorted by id."""
    return tuple(
        sorted(
            (n for n in graph.nodes if isinstance(n, ConditionNode)), key=lambda n: n.id
        )
    )


# ---------------------------------------------------------------------------
# Claim-centered lookups
# ---------------------------------------------------------------------------


def paper_for_claim(graph: KnowledgeGraph, claim_id: str) -> PaperNode | None:
    """The source paper of a claim via its ``EXTRACTED_FROM`` edge.

    ``None`` when the claim or its paper link is absent.
    """
    claim = graph.node(claim_id)
    if not isinstance(claim, ClaimNode):
        return None
    for edge in graph.edges:
        if edge.type == "claim_paper" and edge.source == claim_id:
            paper = graph.node(edge.target)
            if isinstance(paper, PaperNode):
                return paper
    return None


def evidence_for_claim(graph: KnowledgeGraph, claim_id: str) -> EvidenceNode | None:
    """The evidence side attached to a claim via ``HAS_EVIDENCE``.

    ``None`` when the claim or its evidence link is absent.
    """
    claim = graph.node(claim_id)
    if not isinstance(claim, ClaimNode):
        return None
    for edge in graph.edges:
        if edge.type == "claim_evidence" and edge.source == claim_id:
            evidence = graph.node(edge.target)
            if isinstance(evidence, EvidenceNode):
                return evidence
    return None


def conditions_for_claim(graph: KnowledgeGraph, claim_id: str) -> tuple[ConditionNode, ...]:
    """All condition mentions of a claim, sorted by id."""
    claim = graph.node(claim_id)
    if not isinstance(claim, ClaimNode):
        return ()
    targets = sorted(
        edge.target
        for edge in graph.edges
        if edge.type == "claim_condition" and edge.source == claim_id
    )
    nodes = []
    for target in targets:
        node = graph.node(target)
        if isinstance(node, ConditionNode):
            nodes.append(node)
    return tuple(nodes)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def relations(
    graph: KnowledgeGraph, claim_id: str, *, direction: str = "out"
) -> tuple[tuple[ClaimRelationEdge, ClaimNode], ...]:
    """All classified relations touching a claim, sorted by edge id.

    ``direction="out"`` (default) returns relations where the claim is
    the premise; ``"in"`` where it is the hypothesis; ``"both"`` union.
    Each item pairs the directed edge with the claim at the other end.
    Evidence-only (unlabeled) relations are included here; the
    label-specific helpers below exclude them. Unknown claims and empty
    graphs return ``()``.
    """
    _check_direction(direction)
    claim = graph.node(claim_id)
    if not isinstance(claim, ClaimNode):
        return ()
    pairs: dict[str, tuple[ClaimRelationEdge, ClaimNode]] = {}
    for edge in graph.edges:
        if not isinstance(edge, ClaimRelationEdge):
            continue
        other_id: str | None = None
        if direction in ("out", "both") and edge.source == claim_id:
            other_id = edge.target
        if direction in ("in", "both") and edge.target == claim_id:
            other_id = edge.source
        if other_id is None:
            continue
        other = graph.node(other_id)
        if isinstance(other, ClaimNode):
            pairs[edge.id] = (edge, other)
    return tuple(pairs[edge_id] for edge_id in sorted(pairs))


def _relations_with_label(
    graph: KnowledgeGraph, claim_id: str, label: str, direction: str
) -> tuple[tuple[ClaimRelationEdge, ClaimNode], ...]:
    return tuple(
        (edge, other)
        for edge, other in relations(graph, claim_id, direction=direction)
        if edge.label == label
    )


def supporting_relations(
    graph: KnowledgeGraph, claim_id: str, *, direction: str = "out"
) -> tuple[tuple[ClaimRelationEdge, ClaimNode], ...]:
    """Relations where the model predicted entailment (a *prediction*
    that the other claim supports this one — not a verified fact)."""
    return _relations_with_label(graph, claim_id, "entailment", direction)


def contradicting_relations(
    graph: KnowledgeGraph, claim_id: str, *, direction: str = "out"
) -> tuple[tuple[ClaimRelationEdge, ClaimNode], ...]:
    """Relations where the model predicted contradiction."""
    return _relations_with_label(graph, claim_id, "contradiction", direction)


def neutral_relations(
    graph: KnowledgeGraph, claim_id: str, *, direction: str = "out"
) -> tuple[tuple[ClaimRelationEdge, ClaimNode], ...]:
    """Relations where the model predicted neutrality."""
    return _relations_with_label(graph, claim_id, "neutral", direction)


def evidence_for_relation(
    graph: KnowledgeGraph, edge: ClaimRelationEdge
) -> tuple[EvidenceNode | None, EvidenceNode | None]:
    """The premise and hypothesis evidence nodes of a relation edge.

    ``None`` for either side when that claim has no evidence link (for
    example classification-only relations).
    """
    if not isinstance(edge, ClaimRelationEdge):
        raise TypeError(f"expected ClaimRelationEdge, got {type(edge).__name__}")
    return (
        evidence_for_claim(graph, edge.source),
        evidence_for_claim(graph, edge.target),
    )


def conditions_for_relation(
    graph: KnowledgeGraph, edge: ClaimRelationEdge
) -> ConditionComparison | None:
    """The embedded ``ConditionComparison`` of a relation edge, or
    ``None`` when no comparison was provided. Exposed unchanged — a
    condition difference is context, never a rewritten label."""
    if not isinstance(edge, ClaimRelationEdge):
        raise TypeError(f"expected ClaimRelationEdge, got {type(edge).__name__}")
    return edge.conditions


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def connected_components(graph: KnowledgeGraph) -> tuple[tuple[str, ...], ...]:
    """Node-id sets of the connected components, treating edges as
    undirected.

    Deterministic: each component is a sorted tuple of node ids, and the
    outer tuple is sorted by ``(size, members)``. An empty graph yields
    ``()``; isolated nodes are components of size one.
    """
    adjacency: dict[str, set[str]] = {node.id: set() for node in graph.nodes}
    for edge in graph.edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)

    components: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        members: list[str] = []
        seen.add(start)
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(members)))
    return tuple(sorted(components, key=lambda members: (len(members), members)))
