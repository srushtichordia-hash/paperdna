"""Convert a pure :class:`~paperdna.api.pipeline.DiscoveryResult` into
the typed API response (:class:`~paperdna.api.schemas.DiscoveryResponse`).

Every field is a verbatim projection of the pipeline result — evidence
text, offsets, verification status, ranking kind, and NLI prediction
flags are copied unchanged. Nothing is invented: the text-only user
claim keeps ``None`` offsets and its "unknown" provenance, and every
model prediction carries its uncertainty flag and the explicit
``scoresAreCalibrated=False`` marker.
"""

from __future__ import annotations

from paperdna.api.graph_payload import graph_to_flow
from paperdna.api.pipeline import (
    Capabilities,
    DiscoveryResult,
    RankedPaper,
    RelationView,
)
from paperdna.api.schemas import (
    CapabilitiesOut,
    ClaimConditionsOut,
    ClaimOut,
    ClaimRefOut,
    ConditionComparisonOut,
    ConditionMentionOut,
    CorpusStatsOut,
    DiscoveryResponse,
    EvidenceOut,
    FieldComparisonOut,
    GraphEdgeOut,
    GraphOut,
    GraphNodeOut,
    InterpretationOut,
    PaperOut,
    PaperRefOut,
    RelationOut,
)
from paperdna.claims.schema import CandidateClaim
from paperdna.conditions.schema import ClaimConditions, ConditionComparison
from paperdna.corpus.schema import IngestionResult, PaperRecord
from paperdna.evidence.schema import RelationAttribution, SentenceEvidence


def _paper_out(
    record: PaperRecord, *, score: float | None = None, ranking_kind: str | None = None
) -> PaperOut:
    return PaperOut(
        source=record.source,
        sourceId=record.source_id,
        title=record.title,
        abstract=record.abstract,
        authors=list(record.authors),
        published=record.published,
        url=record.url,
        doi=record.doi,
        score=score,
        rankingKind=ranking_kind,  # type: ignore[arg-type]
    )


def _ranked_paper_out(item: RankedPaper) -> PaperOut:
    return _paper_out(item.record, score=item.score, ranking_kind=item.kind)


def _claim_out(claim: CandidateClaim) -> ClaimOut:
    return ClaimOut(
        source=claim.source,
        sourceId=claim.source_id,
        sentence=claim.sentence,
        start=claim.start,
        end=claim.end,
        cue=claim.cue or None,
        hedges=list(claim.hedges),
    )


def _evidence_out(evidence: SentenceEvidence) -> EvidenceOut:
    paper = evidence.paper
    ref = evidence.claim
    return EvidenceOut(
        paper=PaperRefOut(
            source=paper.source,
            sourceId=paper.source_id,
            title=paper.title,
            url=paper.url,
            doi=paper.doi,
            published=paper.published,
        ),
        claim=ClaimRefOut(
            claimId=ref.claim_id,
            source=ref.source,
            sourceId=ref.source_id,
            start=ref.start,
            end=ref.end,
        ),
        sentence=evidence.sentence,
        verification=evidence.verification,  # type: ignore[arg-type]
        unavailableReason=evidence.unavailable_reason,
    )


def _interpretation_out(attribution: RelationAttribution) -> InterpretationOut | None:
    interp = attribution.interpretation
    if interp is None:
        return None
    return InterpretationOut(
        label=interp.label,  # type: ignore[arg-type]
        scores=dict(interp.scores),
        confidence=interp.confidence,
        uncertain=interp.uncertain,
        model=interp.model,
        conditionSummary=interp.condition_summary,
        conditionStats=dict(interp.condition_stats) if interp.condition_stats else None,
        notes=list(interp.notes),
        scoresAreCalibrated=False,
    )


def _conditions_out(comparison: ConditionComparison) -> ConditionComparisonOut:
    return ConditionComparisonOut(
        premiseSource=comparison.premise_source,
        premiseSourceId=comparison.premise_source_id,
        hypothesisSource=comparison.hypothesis_source,
        hypothesisSourceId=comparison.hypothesis_source_id,
        fields=[
            FieldComparisonOut(
                category=field.category,
                status=field.status,  # type: ignore[arg-type]
                premiseValues=list(field.premise_values),
                hypothesisValues=list(field.hypothesis_values),
            )
            for field in comparison.fields
        ],
        stats=dict(comparison.stats),
    )


def _relation_out(view: RelationView) -> RelationOut:
    attribution = view.attribution
    conditions = (
        _conditions_out(attribution.conditions)
        if attribution.conditions is not None
        else None
    )
    return RelationOut(
        kind=view.kind,  # type: ignore[arg-type]
        premise=_evidence_out(attribution.premise),
        hypothesis=_evidence_out(attribution.hypothesis),
        interpretation=_interpretation_out(attribution),
        conditions=conditions,
    )


def _claim_conditions_entries(
    result: DiscoveryResult,
) -> list[dict[str, object]]:
    """(claim, conditions) pairs as JSON-safe dicts."""
    entries: list[dict[str, object]] = []
    for claim, conditions in result.claim_conditions:
        entries.append(
            {
                "claim": _claim_out(claim).model_dump(),
                "conditions": _conditions_to_dict(conditions),
            }
        )
    return entries


def _conditions_to_dict(conditions: ClaimConditions) -> dict[str, object]:
    return {
        "source": conditions.source,
        "sourceId": conditions.source_id,
        "mentions": [
            ConditionMentionOut(
                category=mention.category,
                value=mention.value,
                start=mention.start,
                end=mention.end,
                ambiguous=mention.ambiguous,
                source=mention.source,
                sourceId=mention.source_id,
            ).model_dump()
            for mention in conditions.mentions
        ],
    }


def _capabilities_out(capabilities: Capabilities) -> CapabilitiesOut:
    return CapabilitiesOut(
        rankingKind=capabilities.ranking,  # type: ignore[arg-type]
        embedderName=capabilities.embedder_name,
        nliModel=capabilities.nli_model,
        messages=list(capabilities.messages),
    )


def _graph_out(result: DiscoveryResult) -> GraphOut:
    flow = graph_to_flow(result.graph)
    return GraphOut(
        nodes=[GraphNodeOut(**node) for node in flow["nodes"]],  # type: ignore[arg-type]
        edges=[GraphEdgeOut(**edge) for edge in flow["edges"]],  # type: ignore[arg-type]
    )


def _corpus_stats_out(ingestion: IngestionResult) -> CorpusStatsOut:
    return CorpusStatsOut(
        loaded=ingestion.stats.get("loaded", 0),
        unique=ingestion.stats.get("unique", 0),
        duplicates=ingestion.stats.get("duplicates", 0),
        rejected=ingestion.stats.get("rejected", 0),
    )


def discovery_to_response(result: DiscoveryResult) -> DiscoveryResponse:
    """Project a full pipeline result into the typed API response."""
    user_claim_out = (
        _claim_out(result.user_claim) if result.user_claim is not None else None
    )
    return DiscoveryResponse(
        claim=result.inputs.claim,
        abstractProvided=result.inputs.abstract is not None,
        rankingKind=result.capabilities.ranking,  # type: ignore[arg-type]
        papers=[_ranked_paper_out(item) for item in result.ranked],
        userClaim=user_claim_out,
        userClaimHasRealOffsets=result.user_claim_has_real_offsets,
        paperClaims=[_claim_out(claim) for claim in result.paper_claims],
        claimConditions=_claim_conditions_entries(result),
        relations=[_relation_out(view) for view in result.relations],
        graph=_graph_out(result),
        capabilities=_capabilities_out(result.capabilities),
        messages=list(result.messages),
        stats=dict(result.stats),
    )


def corpus_stats_dict(stats: dict[str, int]) -> CorpusStatsOut:
    """Project an ingestion-stats dict (``IngestionResult.stats``) into
    the response shape. Accepts the ``{loaded, unique, duplicates,
    rejected}`` dict produced by ``ingest_records``."""
    return CorpusStatsOut(
        loaded=stats.get("loaded", 0),
        unique=stats.get("unique", 0),
        duplicates=stats.get("duplicates", 0),
        rejected=stats.get("rejected", 0),
    )
