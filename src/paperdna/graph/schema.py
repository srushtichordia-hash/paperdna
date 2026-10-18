"""Data model for the research evidence knowledge graph.

The graph links papers, claims, evidence sentences, and experimental
conditions into one navigable structure, keeping two kinds of
information strictly separate (mirroring :mod:`paperdna.evidence`):

- **Observed** (paper/claim/evidence/condition node attributes): source
  text and metadata copied verbatim from upstream records, with the
  upstream verification status preserved.
- **Model-generated** (``ClaimRelationEdge`` attributes): the NLI label,
  scores, confidence, uncertainty flags, and condition-comparison
  summaries. These are predictions, never asserted scientific facts.

Node identity is deterministic: the same upstream records always produce
the same ids, regardless of insertion order. Claim and evidence ids reuse
the stable ``"{source}:{source_id}:{start}-{end}"`` format from
:func:`paperdna.evidence.claim_id_for`, so graph ids match
``RelationAttribution`` claim references exactly. Paper ids casefold the
source, matching ``PaperResolver``/``PaperIndex``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from paperdna.conditions.schema import CONDITION_CATEGORIES, ConditionComparison
from paperdna.evidence.schema import EVIDENCE_VERIFICATION, claim_id_for
from paperdna.graph.errors import GraphValidationError
from paperdna.nli.schema import RELATIONSHIPS

# Node and edge type names, as they appear in serialized form.
NODE_TYPES: tuple[str, ...] = ("paper", "claim", "evidence", "condition")
EDGE_TYPES: tuple[str, ...] = (
    "claim_paper",
    "claim_evidence",
    "claim_condition",
    "claim_relation",
)

# Serialization envelope markers.
GRAPH_FORMAT = "paperdna.graph"
GRAPH_VERSION = 1

# NLI label -> the query-facing role name. The raw label is always
# preserved on the edge; this mapping only names what a query helper
# returns ("supporting relations" are entailment *predictions*).
RELATION_ROLES: dict[str, str] = {
    "entailment": "supports",
    "contradiction": "contradicts",
    "neutral": "neutral",
}


def paper_node_id(source: str, source_id: str) -> str:
    """Stable paper id: ``"paper:{source.casefold()}:{source_id}"``.

    Casefolding matches ``PaperResolver`` and ``PaperIndex`` identity.
    """
    return f"paper:{source.casefold()}:{source_id}"


def claim_node_id(
    source: str, source_id: str, start: int | None, end: int | None
) -> str:
    """Stable claim id; delegates to ``paperdna.evidence.claim_id_for``.

    The id is the claim reference itself
    (``"{source}:{source_id}:{start}-{end}"``, offsets part ``"unknown"``
    when abstract offsets are unavailable), so graph claim nodes and
    evidence claim references share one identity format.
    """
    return claim_id_for(source, source_id, start, end)


def evidence_node_id(claim_id: str) -> str:
    """Stable evidence id: ``"evidence:{claim_id}"``."""
    return f"evidence:{claim_id}"


def condition_node_id(claim_id: str, start: int, end: int) -> str:
    """Stable condition id: ``"condition:{claim_id}:{start}-{end}"``.

    ``start``/``end`` are sentence-relative offsets of the mention (as
    extracted by ``paperdna.conditions``); they disambiguate multiple
    mentions within one claim.
    """
    return f"condition:{claim_id}:{start}-{end}"


def edge_id(kind: str, source: str, target: str) -> str:
    """Uniform stable edge id: ``"edge:{kind}:{source}->{target}"``."""
    return f"edge:{kind}:{source}->{target}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperNode:
    """One source paper.

    ``title``/``url``/``doi``/``published`` are copied from the upstream
    record or paper reference when available and stay ``None`` otherwise
    — missing metadata is never fabricated. ``authors`` is empty when the
    graph was built from a ``PaperRef`` (which carries no authors).
    """

    source: str
    source_id: str
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    published: str | None = None
    authors: tuple[str, ...] = ()

    @property
    def type(self) -> str:
        return "paper"

    @property
    def id(self) -> str:
        return paper_node_id(self.source, self.source_id)


@dataclass(frozen=True)
class ClaimNode:
    """One candidate claim (a finding sentence from a paper abstract).

    ``sentence`` is verbatim source text. ``start``/``end`` are character
    offsets into the paper's abstract when known (both present or both
    ``None`` — never invented). ``cue``/``hedges`` come from the claim
    extractor when the claim was built from a ``CandidateClaim``; they
    are ``None`` when the graph only saw evidence or a classification.
    """

    source: str
    source_id: str
    sentence: str
    start: int | None
    end: int | None
    cue: str | None = None
    hedges: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError(
                "claim offsets must both be present or both absent "
                f"(got start={self.start!r}, end={self.end!r})"
            )
        if self.start is not None and (self.start < 0 or self.end < self.start):  # type: ignore[operator]
            raise ValueError(
                f"invalid claim offsets: start={self.start}, end={self.end}"
            )

    @property
    def type(self) -> str:
        return "claim"

    @property
    def id(self) -> str:
        return claim_node_id(self.source, self.source_id, self.start, self.end)


@dataclass(frozen=True)
class EvidenceNode:
    """One observed evidence side: a verbatim sentence and its status.

    ``verification`` is copied from the upstream attribution and stays
    within ``paperdna.evidence.EVIDENCE_VERIFICATION``;
    ``unavailable_reason`` is carried verbatim (required upstream for
    ``unverified``/``unavailable``). ``start``/``end`` are the abstract
    offsets when known. The sentence is exact source text, never
    rewritten; this node holds no model output.
    """

    claim_id: str
    sentence: str
    verification: str
    unavailable_reason: str | None = None
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        if self.verification not in EVIDENCE_VERIFICATION:
            raise ValueError(f"unknown verification status: {self.verification!r}")
        if self.verification in ("unverified", "unavailable") and not self.unavailable_reason:
            raise ValueError(
                f"verification {self.verification!r} requires an unavailable_reason"
            )
        if (self.start is None) != (self.end is None):
            raise ValueError(
                "evidence offsets must both be present or both absent "
                f"(got start={self.start!r}, end={self.end!r})"
            )
        if self.start is not None and (self.start < 0 or self.end < self.start):  # type: ignore[operator]
            raise ValueError(
                f"invalid evidence offsets: start={self.start}, end={self.end}"
            )

    @property
    def type(self) -> str:
        return "evidence"

    @property
    def id(self) -> str:
        return evidence_node_id(self.claim_id)

    @property
    def available(self) -> bool:
        """False when there is no usable evidence text."""
        return self.verification != "unavailable"

    @property
    def verified(self) -> bool:
        """True when offsets are confirmed (or uniquely inferred)
        against the original abstract."""
        return self.verification in ("verified", "verified_by_search")


@dataclass(frozen=True)
class ConditionNode:
    """One experimental-condition mention extracted from a claim.

    ``value`` is the verbatim span text (never rewritten); ``start``/
    ``end`` are sentence-relative offsets (``sentence[start:end] ==
    value``), matching ``paperdna.conditions.ConditionMention``.
    ``global_start``/``global_end`` add the claim's abstract offset when
    it is known (both present or both ``None``).
    """

    claim_id: str
    category: str
    value: str
    start: int
    end: int
    ambiguous: bool
    global_start: int | None = None
    global_end: int | None = None

    def __post_init__(self) -> None:
        if self.category not in CONDITION_CATEGORIES:
            raise ValueError(f"unknown condition category: {self.category!r}")
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"invalid condition offsets: start={self.start}, end={self.end}"
            )
        if (self.global_start is None) != (self.global_end is None):
            raise ValueError(
                "global offsets must both be present or both absent "
                f"(got global_start={self.global_start!r}, "
                f"global_end={self.global_end!r})"
            )

    @property
    def type(self) -> str:
        return "condition"

    @property
    def id(self) -> str:
        return condition_node_id(self.claim_id, self.start, self.end)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimPaperEdge:
    """``claim --[EXTRACTED_FROM]--> paper``: provenance of a claim."""

    source: str  # claim node id
    target: str  # paper node id

    @property
    def type(self) -> str:
        return "claim_paper"

    @property
    def id(self) -> str:
        return edge_id(self.type, self.source, self.target)


@dataclass(frozen=True)
class ClaimEvidenceEdge:
    """``claim --[HAS_EVIDENCE]--> evidence``: one evidence side of a claim."""

    source: str  # claim node id
    target: str  # evidence node id

    @property
    def type(self) -> str:
        return "claim_evidence"

    @property
    def id(self) -> str:
        return edge_id(self.type, self.source, self.target)


@dataclass(frozen=True)
class ClaimConditionEdge:
    """``claim --[HAS_CONDITION]--> condition``: one condition mention."""

    source: str  # claim node id
    target: str  # condition node id

    @property
    def type(self) -> str:
        return "claim_condition"

    @property
    def id(self) -> str:
        return edge_id(self.type, self.source, self.target)


@dataclass(frozen=True)
class ClaimRelationEdge:
    """``premise claim --[CLAIM_RELATION]--> hypothesis claim``.

    The edge is directed: NLI is not symmetric, ``source`` is the
    premise and ``target`` the hypothesis. Everything here is
    **model-generated** — the NLI label (one of
    ``paperdna.nli.RELATIONSHIPS``), softmax scores, confidence, and the
    embedded condition comparison (``paperdna.conditions.
    ConditionComparison``, unchanged). ``label`` is ``None`` for an
    evidence-only attribution (no classification was provided). Scores
    and confidence are ``None`` in that case too; they are predictions,
    never verified scientific facts — see ``RELATION_ROLES`` for the
    query-facing role names.
    """

    source: str  # premise claim node id
    target: str  # hypothesis claim node id
    label: str | None = None
    scores: dict[str, float] | None = None
    confidence: float | None = None
    uncertain: bool | None = None
    hedged: bool | None = None
    truncated: bool | None = None
    model: str | None = None
    conditions: ConditionComparison | None = None
    notes: tuple[str, ...] = ()

    @property
    def type(self) -> str:
        return "claim_relation"

    @property
    def id(self) -> str:
        return edge_id(self.type, self.source, self.target)

    @property
    def role(self) -> str | None:
        """Query-facing role name (``"supports"``/``"contradicts"``/
        ``"neutral"``) derived from the label, or ``None`` when the
        edge carries no classification."""
        if self.label is None:
            return None
        return RELATION_ROLES.get(self.label)


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------

# Any node / any edge in the graph.
Node = PaperNode | ClaimNode | EvidenceNode | ConditionNode
Edge = ClaimPaperEdge | ClaimEvidenceEdge | ClaimConditionEdge | ClaimRelationEdge

# Expected (source type, target type) per edge type.
_EXPECTED_ENDPOINTS: dict[str, tuple[str, str]] = {
    "claim_paper": ("claim", "paper"),
    "claim_evidence": ("claim", "evidence"),
    "claim_condition": ("claim", "condition"),
    "claim_relation": ("claim", "claim"),
}


def _collect_problems(nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> list[str]:
    """All structural problems in one pass (empty list when valid)."""
    problems: list[str] = []
    node_by_id: dict[str, Node] = {}
    for node in nodes:
        if node.id in node_by_id:
            problems.append(f"duplicate node id: {node.id!r}")
        node_by_id[node.id] = node

    edge_ids: set[str] = set()
    for edge in edges:
        if edge.id in edge_ids:
            problems.append(f"duplicate edge id: {edge.id!r}")
        edge_ids.add(edge.id)

        source_node = node_by_id.get(edge.source)
        target_node = node_by_id.get(edge.target)
        if source_node is None:
            problems.append(
                f"edge {edge.id!r} references unknown source node {edge.source!r}"
            )
        if target_node is None:
            problems.append(
                f"edge {edge.id!r} references unknown target node {edge.target!r}"
            )

        want_source_type, want_target_type = _EXPECTED_ENDPOINTS[edge.type]
        if source_node is not None and source_node.type != want_source_type:
            problems.append(
                f"edge {edge.id!r} expects a {want_source_type} source node, "
                f"got {source_node.type}"
            )
        if target_node is not None and target_node.type != want_target_type:
            problems.append(
                f"edge {edge.id!r} expects a {want_target_type} target node, "
                f"got {target_node.type}"
            )

        if isinstance(edge, ClaimRelationEdge):
            if edge.label is not None and edge.label not in RELATIONSHIPS:
                problems.append(
                    f"edge {edge.id!r} has invalid relation label {edge.label!r}"
                )
            if edge.scores is not None and set(edge.scores) != set(RELATIONSHIPS):
                problems.append(
                    f"edge {edge.id!r} scores must cover exactly "
                    f"{sorted(RELATIONSHIPS)}, got {sorted(edge.scores)}"
                )
            if edge.confidence is not None and not 0.0 <= edge.confidence <= 1.0:
                problems.append(
                    f"edge {edge.id!r} confidence outside [0, 1]: {edge.confidence}"
                )
            if edge.conditions is not None and not isinstance(
                edge.conditions, ConditionComparison
            ):
                problems.append(
                    f"edge {edge.id!r} conditions must be a ConditionComparison "
                    f"or None, got {type(edge.conditions).__name__}"
                )
    return problems


class KnowledgeGraph:
    """An immutable, validated graph of papers, claims, evidence, and
    conditions.

    Construction validates structure and raises ``GraphValidationError``
    listing every problem: duplicate node/edge ids, edges referencing
    nonexistent nodes, wrong endpoint node types, and invalid relation
    labels/scores. Disconnected components are legal (and common); there
    is deliberately no connectivity requirement.
    """

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        self._nodes = tuple(nodes)
        self._edges = tuple(edges)
        problems = _collect_problems(self._nodes, self._edges)
        if problems:
            raise GraphValidationError(
                f"{len(problems)} graph validation problem(s): " + "; ".join(problems)
            )
        self._node_by_id: dict[str, Node] = {node.id: node for node in self._nodes}

    # -- basic accessors ---------------------------------------------------

    @property
    def nodes(self) -> tuple[Node, ...]:
        """All nodes, in insertion order."""
        return self._nodes

    @property
    def edges(self) -> tuple[Edge, ...]:
        """All edges, in insertion order."""
        return self._edges

    def node(self, node_id: str) -> Node | None:
        """The node with this id, or ``None``."""
        return self._node_by_id.get(node_id)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._node_by_id

    def __len__(self) -> int:
        return len(self._nodes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KnowledgeGraph):
            return NotImplemented
        return {n.id: n for n in self._nodes} == {
            n.id: n for n in other._nodes
        } and {e.id: e for e in self._edges} == {e.id: e for e in other._edges}

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"KnowledgeGraph(papers={stats['paper']}, claims={stats['claim']}, "
            f"evidence={stats['evidence']}, conditions={stats['condition']}, "
            f"edges={stats['edges']})"
        )

    @property
    def stats(self) -> dict[str, int]:
        """Fixed-key counts by node and edge type."""
        counts = {node_type: 0 for node_type in NODE_TYPES}
        for node in self._nodes:
            counts[node.type] += 1
        edge_counts = {edge_type: 0 for edge_type in EDGE_TYPES}
        for edge in self._edges:
            edge_counts[edge.type] += 1
        return {
            **counts,
            **edge_counts,
            "nodes": len(self._nodes),
            "edges": len(self._edges),
        }

    def validate(self) -> None:
        """Re-check structural validity; raise ``GraphValidationError``.

        Nodes and edges are immutable, so a graph that constructed
        successfully stays valid; this exists for loaded graphs and for
        callers who want an explicit check.
        """
        problems = _collect_problems(self._nodes, self._edges)
        if problems:
            raise GraphValidationError(
                f"{len(problems)} graph validation problem(s): " + "; ".join(problems)
            )
