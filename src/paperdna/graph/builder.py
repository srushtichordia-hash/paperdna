"""Incremental assembly of a knowledge graph from pipeline outputs.

``GraphBuilder`` consumes the public outputs of the upstream modules —
``PaperRecord``/``PaperRef``, ``CandidateClaim``, ``ClaimRef``,
``SentenceEvidence``, ``PairClassification``, ``ClaimConditions``, and
``RelationAttribution`` — and turns them into a validated
:class:`~paperdna.graph.schema.KnowledgeGraph`. No upstream module is
changed or re-behaved: model outputs and evidence text are embedded
unchanged.

Duplicate policy (matching the loud-rejection style of
:mod:`paperdna.evidence`):

- Re-adding an **identical** node or edge under the same id is an
  idempotent no-op, counted in ``duplicates_ignored``.
- Paper and claim nodes support **enrichment**: a second add that only
  fills in previously missing metadata (e.g. adding a ``PaperRecord``
  after a metadata-free stub) is merged, counted in ``claims_enriched``
  for claims (and in ``duplicates_ignored`` for papers). Enrichment is
  order-independent and never overwrites a conflicting non-empty value.
- Any other payload disagreement under an existing id raises
  ``GraphConflictError`` — the builder never silently keeps one of two
  different payloads.
"""

from __future__ import annotations

import logging
from typing import Iterable

from paperdna.claims.schema import CandidateClaim
from paperdna.conditions.schema import ClaimConditions, ConditionComparison
from paperdna.corpus.schema import PaperRecord
from paperdna.evidence.schema import (
    ClaimRef,
    PaperRef,
    RelationAttribution,
    SentenceEvidence,
)
from paperdna.graph.errors import GraphConflictError
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
    evidence_node_id,
    paper_node_id,
)
from paperdna.nli.schema import PairClassification

logger = logging.getLogger(__name__)

# Sentinel marking two non-empty values that disagree.
_CONFLICT = object()


def _merge_optional(a: object, b: object, empty: object = None) -> object:
    """Merge two optional field values; ``_CONFLICT`` when they disagree.

    ``None`` (or ``empty``, e.g. ``()`` for author tuples) means
    "unknown" and defers to the other side; two different known values
    conflict.
    """
    if a == b:
        return a
    if a is None or a == empty:
        return b
    if b is None or b == empty:
        return a
    return _CONFLICT


def _merged_paper(existing: PaperNode, new: PaperNode) -> PaperNode | None:
    """Enrichment merge for two paper nodes of the same identity."""
    title = _merge_optional(existing.title, new.title)
    url = _merge_optional(existing.url, new.url)
    doi = _merge_optional(existing.doi, new.doi)
    published = _merge_optional(existing.published, new.published)
    authors = _merge_optional(existing.authors, new.authors, empty=())
    if _CONFLICT in (title, url, doi, published, authors):
        return None
    return PaperNode(
        source=existing.source,
        source_id=existing.source_id,
        title=title,  # type: ignore[arg-type]
        url=url,  # type: ignore[arg-type]
        doi=doi,  # type: ignore[arg-type]
        published=published,  # type: ignore[arg-type]
        authors=authors,  # type: ignore[arg-type]
    )


def _merged_claim(existing: ClaimNode, new: ClaimNode) -> ClaimNode | None:
    """Enrichment merge for two claim nodes of the same identity.

    The id fixes source, source_id, and offsets; only the sentence must
    match exactly, while ``cue``/``hedges`` may be filled in from the
    side that had them.
    """
    if existing.sentence != new.sentence:
        return None
    cue = _merge_optional(existing.cue, new.cue, empty="")
    hedges = _merge_optional(existing.hedges, new.hedges, empty=())
    if _CONFLICT in (cue, hedges):
        return None
    return ClaimNode(
        source=existing.source,
        source_id=existing.source_id,
        sentence=existing.sentence,
        start=existing.start,
        end=existing.end,
        cue=cue,  # type: ignore[arg-type]
        hedges=hedges,  # type: ignore[arg-type]
    )


class GraphBuilder:
    """Incrementally assemble a validated knowledge graph.

    Every ``add_*`` method is idempotent for identical input and returns
    the id of the affected claim (or paper). ``build()`` validates the
    whole structure and returns an immutable ``KnowledgeGraph``.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._duplicates_ignored = 0
        self._claims_enriched = 0

    # -- counters ----------------------------------------------------------

    @property
    def duplicates_ignored(self) -> int:
        """Identical re-adds and metadata enrichments that were merged."""
        return self._duplicates_ignored

    @property
    def claims_enriched(self) -> int:
        """Claim nodes whose missing cue/hedges were filled in later."""
        return self._claims_enriched

    @property
    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "duplicates_ignored": self._duplicates_ignored,
            "claims_enriched": self._claims_enriched,
        }

    # -- low-level add with duplicate/conflict policy ----------------------

    def _add_node(self, node: Node) -> str:
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
            return node.id
        if existing == node:
            self._duplicates_ignored += 1
            return node.id
        if isinstance(existing, PaperNode) and isinstance(node, PaperNode):
            merged = _merged_paper(existing, node)
            if merged is not None:
                self._nodes[node.id] = merged
                self._duplicates_ignored += 1
                return node.id
        if isinstance(existing, ClaimNode) and isinstance(node, ClaimNode):
            merged = _merged_claim(existing, node)
            if merged is not None:
                self._nodes[node.id] = merged
                self._duplicates_ignored += 1
                self._claims_enriched += 1
                return node.id
        raise GraphConflictError(
            f"conflicting payload for existing node id {node.id!r}: "
            f"{existing!r} vs {node!r}"
        )

    def _add_edge(self, edge: Edge) -> str:
        existing = self._edges.get(edge.id)
        if existing is None:
            self._edges[edge.id] = edge
            return edge.id
        if existing == edge:
            self._duplicates_ignored += 1
            return edge.id
        raise GraphConflictError(
            f"conflicting payload for existing edge id {edge.id!r}: "
            f"{existing!r} vs {edge!r}"
        )

    # -- papers ------------------------------------------------------------

    def add_paper(self, paper: PaperRecord | PaperRef | PaperNode) -> str:
        """Add (or enrich) one paper node; returns the node id.

        ``PaperRecord`` contributes authors; ``PaperRef`` and missing
        metadata fields stay empty — never fabricated.
        """
        if isinstance(paper, PaperNode):
            node = paper
        elif isinstance(paper, PaperRecord):
            node = PaperNode(
                source=paper.source,
                source_id=paper.source_id,
                title=paper.title,
                url=paper.url,
                doi=paper.doi,
                published=paper.published,
                authors=tuple(paper.authors),
            )
        elif isinstance(paper, PaperRef):
            node = PaperNode(
                source=paper.source,
                source_id=paper.source_id,
                title=paper.title,
                url=paper.url,
                doi=paper.doi,
                published=paper.published,
            )
        else:
            raise TypeError(
                "expected PaperRecord, PaperRef, or PaperNode, "
                f"got {type(paper).__name__}"
            )
        return self._add_node(node)

    # -- claims ------------------------------------------------------------

    def _ensure_claim(
        self,
        source: str,
        source_id: str,
        sentence: str,
        start: int | None,
        end: int | None,
        cue: str | None,
        hedges: tuple[str, ...] | None,
    ) -> str:
        """Add the claim node, its paper node, and the EXTRACTED_FROM edge."""
        claim = ClaimNode(
            source=source,
            source_id=source_id,
            sentence=sentence,
            start=start,
            end=end,
            cue=cue,
            hedges=hedges,
        )
        self._add_node(claim)
        # Paper stub: identity is known from the claim even when no
        # metadata is; a later add_paper with the real record enriches it.
        self._add_node(PaperNode(source=source, source_id=source_id))
        self._add_edge(ClaimPaperEdge(source=claim.id, target=paper_node_id(source, source_id)))
        return claim.id

    def add_claim(self, claim: CandidateClaim) -> str:
        """Add a candidate claim with its paper and provenance edge.

        Also ensures a paper node exists for the claim's identity (as a
        metadata-free stub when the paper was not added explicitly) and
        links the two with an ``EXTRACTED_FROM`` edge. Returns the claim
        node id.
        """
        if not isinstance(claim, CandidateClaim):
            raise TypeError(f"expected CandidateClaim, got {type(claim).__name__}")
        return self._ensure_claim(
            claim.source,
            claim.source_id,
            claim.sentence,
            claim.start,
            claim.end,
            claim.cue,
            tuple(claim.hedges),
        )

    def add_claim_ref(self, ref: ClaimRef, sentence: str) -> str:
        """Add a claim from a stable claim reference plus its sentence.

        For the evidence path where only the reference and verbatim text
        are available: offsets come from the ref (``None`` on the
        classification-only path), and cue/hedges stay unknown (``None``).
        """
        if not isinstance(ref, ClaimRef):
            raise TypeError(f"expected ClaimRef, got {type(ref).__name__}")
        if not isinstance(sentence, str):
            raise TypeError(f"expected str sentence, got {type(sentence).__name__}")
        return self._ensure_claim(
            ref.source, ref.source_id, sentence, ref.start, ref.end, None, None
        )

    # -- evidence ----------------------------------------------------------

    def add_evidence(self, evidence: SentenceEvidence) -> str:
        """Add one observed evidence side and its links.

        Creates (idempotently) the evidence node, the claim node, the
        paper node, and the ``HAS_EVIDENCE`` and ``EXTRACTED_FROM``
        edges. Returns the claim node id (the hub other queries use).
        """
        if not isinstance(evidence, SentenceEvidence):
            raise TypeError(
                f"expected SentenceEvidence, got {type(evidence).__name__}"
            )
        claim = evidence.claim
        claim_id = self._ensure_claim(
            claim.source,
            claim.source_id,
            evidence.sentence,
            claim.start,
            claim.end,
            None,
            None,
        )
        # Re-assert the paper with whatever metadata the reference has
        # (merges with / enriches any stub already present).
        self.add_paper(evidence.paper)
        self._add_node(
            EvidenceNode(
                claim_id=claim_id,
                sentence=evidence.sentence,
                verification=evidence.verification,
                unavailable_reason=evidence.unavailable_reason,
                start=claim.start,
                end=claim.end,
            )
        )
        self._add_edge(ClaimEvidenceEdge(source=claim_id, target=evidence_node_id(claim_id)))
        return claim_id

    # -- conditions ---------------------------------------------------------

    def add_claim_conditions(
        self, claim: CandidateClaim, conditions: ClaimConditions
    ) -> str:
        """Add one claim's condition mentions as nodes with link edges.

        The claim is required (mention offsets are sentence-relative, so
        the claim's abstract offsets are needed to compute global ones
        and the exact claim id). ``conditions`` must belong to the same
        source identity as ``claim``; a mismatch raises ``ValueError``.
        Returns the claim node id.
        """
        if not isinstance(claim, CandidateClaim):
            raise TypeError(f"expected CandidateClaim, got {type(claim).__name__}")
        if not isinstance(conditions, ClaimConditions):
            raise TypeError(
                f"expected ClaimConditions, got {type(conditions).__name__}"
            )
        if conditions.source != claim.source or conditions.source_id != claim.source_id:
            raise ValueError(
                "conditions identity mismatch: "
                f"{conditions.source}:{conditions.source_id} vs claim "
                f"{claim.source}:{claim.source_id}"
            )
        claim_id = self.add_claim(claim)
        for mention in conditions.mentions:
            node = ConditionNode(
                claim_id=claim_id,
                category=mention.category,
                value=mention.value,
                start=mention.start,
                end=mention.end,
                ambiguous=mention.ambiguous,
                global_start=claim.start + mention.start,
                global_end=claim.start + mention.end,
            )
            self._add_node(node)
            self._add_edge(ClaimConditionEdge(source=claim_id, target=node.id))
        return claim_id

    # -- relations ----------------------------------------------------------

    def add_relation(
        self,
        classification: PairClassification,
        *,
        conditions: ConditionComparison | None = None,
        notes: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        """Add a classified claim pair (classification-only path).

        Both claims are created from the classification's verbatim
        texts; abstract offsets are unknown here, so the claim ids use
        the ``unknown`` suffix (never guessed). ``conditions`` is an
        optional ``ConditionComparison`` embedded unchanged. Returns the
        (premise, hypothesis) claim node ids.
        """
        if not isinstance(classification, PairClassification):
            raise TypeError(
                f"expected PairClassification, got {type(classification).__name__}"
            )
        if conditions is not None and not isinstance(conditions, ConditionComparison):
            raise TypeError(
                f"expected ConditionComparison, got {type(conditions).__name__}"
            )
        premise_id = self._ensure_claim(
            classification.premise_source,
            classification.premise_source_id,
            classification.premise_text,
            None,
            None,
            None,
            None,
        )
        hypothesis_id = self._ensure_claim(
            classification.hypothesis_source,
            classification.hypothesis_source_id,
            classification.hypothesis_text,
            None,
            None,
            None,
            None,
        )
        self._add_edge(
            ClaimRelationEdge(
                source=premise_id,
                target=hypothesis_id,
                label=classification.label,
                scores=dict(classification.scores),
                confidence=classification.confidence,
                uncertain=classification.uncertain,
                hedged=classification.hedged,
                truncated=classification.truncated,
                model=classification.model,
                conditions=conditions,
                notes=tuple(notes),
            )
        )
        return premise_id, hypothesis_id

    def add_attribution(self, attribution: RelationAttribution) -> tuple[str, str]:
        """Add a full evidence attribution from the evidence stage.

        Both evidence sides become claim + evidence + paper nodes with
        their link edges; the predicted relationship (when the
        attribution carries a classification or interpretation) becomes
        a directed ``CLAIM_RELATION`` edge from premise to hypothesis,
        embedding the classification and condition comparison unchanged.
        An evidence-only attribution (no classification) produces an
        unlabeled relation edge. Returns the (premise, hypothesis) claim
        node ids.
        """
        if not isinstance(attribution, RelationAttribution):
            raise TypeError(
                f"expected RelationAttribution, got {type(attribution).__name__}"
            )
        premise_id = self.add_evidence(attribution.premise)
        hypothesis_id = self.add_evidence(attribution.hypothesis)

        label = scores = confidence = uncertain = hedged = truncated = model = None
        notes: tuple[str, ...] = ()
        classification = attribution.classification
        interpretation = attribution.interpretation
        if classification is not None:
            label = classification.label
            scores = dict(classification.scores)
            confidence = classification.confidence
            uncertain = classification.uncertain
            hedged = classification.hedged
            truncated = classification.truncated
            model = classification.model
        elif interpretation is not None:
            label = interpretation.label
            scores = dict(interpretation.scores)
            confidence = interpretation.confidence
            uncertain = interpretation.uncertain
            model = interpretation.model
        if interpretation is not None:
            notes = tuple(interpretation.notes)

        self._add_edge(
            ClaimRelationEdge(
                source=premise_id,
                target=hypothesis_id,
                label=label,
                scores=scores,
                confidence=confidence,
                uncertain=uncertain,
                hedged=hedged,
                truncated=truncated,
                model=model,
                conditions=attribution.conditions,
                notes=notes,
            )
        )
        return premise_id, hypothesis_id

    # -- output -------------------------------------------------------------

    def build(self) -> KnowledgeGraph:
        """Validate and return the immutable graph."""
        graph = KnowledgeGraph(self._nodes.values(), self._edges.values())
        logger.debug("Built %r from %s", graph, self.stats)
        return graph


def build_graph(
    *,
    papers: Iterable[PaperRecord | PaperRef | PaperNode] = (),
    claims: Iterable[CandidateClaim] = (),
    claim_refs: Iterable[tuple[ClaimRef, str]] = (),
    evidences: Iterable[SentenceEvidence] = (),
    attributions: Iterable[RelationAttribution] = (),
    conditions: Iterable[tuple[CandidateClaim, ClaimConditions]] = (),
    relations: Iterable[PairClassification] = (),
) -> KnowledgeGraph:
    """Build a validated graph from upstream pipeline outputs.

    Each keyword accepts an iterable of the corresponding upstream
    objects (empty by default, so ``build_graph()`` yields a valid empty
    graph); ``claim_refs`` takes ``(ClaimRef, sentence)`` pairs and
    ``conditions`` takes ``(CandidateClaim, ClaimConditions)`` pairs.
    Order of inputs does not affect the resulting graph.
    """
    builder = GraphBuilder()
    for paper in papers:
        builder.add_paper(paper)
    for claim in claims:
        builder.add_claim(claim)
    for ref, sentence in claim_refs:
        builder.add_claim_ref(ref, sentence)
    for evidence in evidences:
        builder.add_evidence(evidence)
    for attribution in attributions:
        builder.add_attribution(attribution)
    for claim, claim_conditions in conditions:
        builder.add_claim_conditions(claim, claim_conditions)
    for classification in relations:
        builder.add_relation(classification)
    return builder.build()
