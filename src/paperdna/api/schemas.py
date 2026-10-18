"""Typed request/response models for the PaperDNA web API.

These Pydantic models are the JSON contract between the FastAPI backend
and the React frontend. They mirror the pure result schema in
:mod:`paperdna.api.pipeline` field-for-field but use camelCase keys (via
an alias generator) so the frontend sees idiomatic TypeScript names.

Honesty rules enforced here:

- ``rankingKind`` is always present and is either ``"semantic"`` or
  ``"lexical"``; the frontend must never present lexical overlap as
  semantic similarity.
- Relationship ``label``/``confidence``/``uncertain`` come from the NLI
  model and are labeled predictions; ``scoresAreCalibrated`` is always
  ``False`` for NLI softmax scores.
- Evidence ``verification`` and ``unavailableReason`` are copied verbatim
  from the evidence module; a missing-evidence side is represented
  explicitly, never dropped or invented.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared enums (plain string literals for easy TS mapping)
# ---------------------------------------------------------------------------

RankingKind = Literal["semantic", "lexical"]
RelationLabel = Literal["entailment", "contradiction", "neutral"]
RelationKind = Literal["user_vs_paper", "paper_vs_paper"]
# Copied verbatim from paperdna.evidence.EVIDENCE_VERIFICATION.
VerificationStatus = Literal[
    "verified", "verified_by_search", "unverified", "unavailable"
]
ConditionStatus = Literal["matching", "differing", "uncertain", "missing"]
GraphNodeType = Literal["paper", "claim", "evidence", "condition"]
GraphEdgeType = Literal[
    "claim_paper", "claim_evidence", "claim_condition", "claim_relation"
]


class _CamelModel(BaseModel):
    """Base model that serializes snake_case fields to camelCase keys.

    Field names in this module are already camelCase (matching the
    TypeScript contract), so no alias generator is needed; this base
    exists to keep ``populate_by_name`` behavior uniform and to leave a
    single place to add validation config later.
    """

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AnalyzeRequest(_CamelModel):
    """POST /api/analyze request body."""

    claim: str = Field(..., min_length=1, description="Scientific claim to analyze.")
    abstract: str | None = Field(
        default=None,
        description="Optional abstract for verifying the claim's provenance.",
    )
    topK: int = Field(default=5, ge=1, le=10)
    maxPairs: int = Field(default=20, ge=1, le=50)
    comparePapers: bool = Field(default=False)


class LoadModelsRequest(_CamelModel):
    """POST /api/models/load request body.

    ``confirmDownload`` must be ``True``; the endpoint refuses to
    construct a model without this explicit acknowledgment.
    """

    confirmDownload: bool = Field(
        default=False,
        description="Must be true to construct/download models.",
    )


# ---------------------------------------------------------------------------
# Corpus / paper models
# ---------------------------------------------------------------------------


class PaperOut(_CamelModel):
    """One ranked paper (or a corpus paper in ``/api/corpus``).

    ``score``/``rankingKind`` are present for ranked results and
    ``None``/``None`` for plain corpus listings.
    """

    source: str
    sourceId: str
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = []
    published: str | None = None
    url: str | None = None
    doi: str | None = None
    score: float | None = None
    rankingKind: RankingKind | None = None


class CorpusStatsOut(_CamelModel):
    loaded: int
    unique: int
    duplicates: int
    rejected: int


class CorpusResponse(_CamelModel):
    stats: CorpusStatsOut
    papers: list[PaperOut]


# ---------------------------------------------------------------------------
# Claim / evidence / condition models
# ---------------------------------------------------------------------------


class ClaimOut(_CamelModel):
    """One candidate claim. ``start``/``end`` are ``None`` when abstract
    offsets are genuinely unknown — never invented."""

    source: str
    sourceId: str
    sentence: str
    start: int | None = None
    end: int | None = None
    cue: str | None = None
    hedges: list[str] = []


class PaperRefOut(_CamelModel):
    source: str
    sourceId: str
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    published: str | None = None


class ClaimRefOut(_CamelModel):
    claimId: str
    source: str
    sourceId: str
    start: int | None = None
    end: int | None = None


class EvidenceOut(_CamelModel):
    """One observed evidence side, copied verbatim from the evidence API."""

    paper: PaperRefOut
    claim: ClaimRefOut
    sentence: str
    verification: VerificationStatus
    unavailableReason: str | None = None


class ConditionMentionOut(_CamelModel):
    category: str
    value: str
    start: int
    end: int
    ambiguous: bool
    source: str
    sourceId: str


class ClaimConditionsOut(_CamelModel):
    source: str
    sourceId: str
    mentions: list[ConditionMentionOut]


class FieldComparisonOut(_CamelModel):
    category: str
    status: ConditionStatus
    premiseValues: list[str]
    hypothesisValues: list[str]


class ConditionComparisonOut(_CamelModel):
    premiseSource: str
    premiseSourceId: str
    hypothesisSource: str
    hypothesisSourceId: str
    fields: list[FieldComparisonOut]
    stats: dict[str, int]


# ---------------------------------------------------------------------------
# Relation (predicted relationship) models
# ---------------------------------------------------------------------------


class InterpretationOut(_CamelModel):
    """Model-generated interpretation — NOT an observed scientific fact."""

    label: RelationLabel
    scores: dict[str, float]
    confidence: float
    uncertain: bool
    model: str
    conditionSummary: str | None = None
    conditionStats: dict[str, int] | None = None
    notes: list[str] = []
    scoresAreCalibrated: bool = False


class RelationOut(_CamelModel):
    kind: RelationKind
    premise: EvidenceOut
    hypothesis: EvidenceOut
    interpretation: InterpretationOut | None = None
    conditions: ConditionComparisonOut | None = None


# ---------------------------------------------------------------------------
# Graph (React Flow) models
# ---------------------------------------------------------------------------


class GraphNodeOut(_CamelModel):
    id: str
    type: GraphNodeType
    position: dict[str, float]  # {"x": float, "y": float}
    data: dict[str, object]  # type-specific payload (see graph_payload.py)


class GraphEdgeOut(_CamelModel):
    id: str
    source: str
    target: str
    type: GraphEdgeType
    label: str | None = None
    data: dict[str, object]


class GraphOut(_CamelModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


# ---------------------------------------------------------------------------
# Capabilities / discovery response
# ---------------------------------------------------------------------------


class CapabilitiesOut(_CamelModel):
    rankingKind: RankingKind
    embedderName: str | None = None
    nliModel: str | None = None
    messages: list[str] = []


class DiscoveryResponse(_CamelModel):
    """Full POST /api/analyze response."""

    claim: str
    abstractProvided: bool
    rankingKind: RankingKind
    papers: list[PaperOut]
    userClaim: ClaimOut | None = None
    userClaimHasRealOffsets: bool = False
    paperClaims: list[ClaimOut] = []
    claimConditions: list[dict[str, object]] = []
    relations: list[RelationOut] = []
    graph: GraphOut
    capabilities: CapabilitiesOut
    messages: list[str] = []
    stats: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Health / model-status models
# ---------------------------------------------------------------------------


class ModelStatusOut(_CamelModel):
    installed: bool
    loaded: bool
    name: str | None = None
    downloadMB: int | None = None


class ModelsStatusOut(_CamelModel):
    embedding: ModelStatusOut
    nli: ModelStatusOut


class HealthResponse(_CamelModel):
    status: str
    corpus: CorpusStatsOut
    models: ModelsStatusOut
