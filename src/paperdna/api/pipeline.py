"""Offline-first orchestration of the PaperDNA pipeline for the web API.

Pure, deterministic Python: imports only the ``paperdna`` core package —
never ``fastapi``, never the optional ML stack. Models are NEVER
constructed here; the caller (the API layer, or a test) injects an
``Embedder`` and/or ``NliClassifier``, or leaves them ``None`` to run in
degraded mode. This guarantees nothing is ever downloaded implicitly.

What each mode does:

- **No embedder** -> papers are ranked by a deterministic token-overlap
  score computed here (``kind="lexical"``). This is app-layer
  presentation ranking, clearly labeled; it does not touch or replace
  the retrieval module's semantic index.
- **No classifier** -> no relationships are predicted (a relationship
  needs an NLI label); the result explains why and still carries
  ranked papers, extracted claims, conditions, and a graph of
  papers/claims/conditions.
- **Both present** -> full pipeline via the real public APIs:
  ``PaperIndex``/``search`` -> ``extract_claims`` -> ``classify_pair``
  -> ``extract_conditions``/``compare_conditions`` ->
  ``build_attribution``/``attribute_from_classification`` ->
  ``build_graph``.

User-claim honesty: when the user's claim appears verbatim (unique
occurrence) in their pasted abstract, it gets real abstract offsets and
full evidence verification. Otherwise it is a *text-only* claim: it is
paired via a transient, pairing-only ``CandidateClaim`` (whose dummy
offsets are never surfaced, never serialized, and never enter the graph
— enforced by tests). Its provenance comes from the hybrid attribution
path, which keeps the paper premise's real verified evidence and
recomputes only the user hypothesis side from verbatim text, honestly
reporting ``verified_by_search`` or ``unverified``/``unavailable`` with a
stated reason.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from paperdna.api.validation import AppInputs
from paperdna.claims.extractor import extract_claims
from paperdna.claims.schema import CandidateClaim
from paperdna.conditions.compare import compare_conditions
from paperdna.conditions.extract import extract_conditions
from paperdna.conditions.schema import ClaimConditions, ConditionComparison
from paperdna.corpus.ingest import ingest_records, load_jsonl
from paperdna.corpus.schema import PaperRecord
from paperdna.evidence.build import attribute_from_classification, build_attribution
from paperdna.evidence.schema import RelationAttribution
from paperdna.graph.builder import build_graph
from paperdna.graph.schema import KnowledgeGraph
from paperdna.nli.classify import classify_pair
from paperdna.nli.schema import PairClassification

logger = logging.getLogger(__name__)

# Identity used for the user's own claim/abstract; distinct from any
# corpus source so user content can never collide with paper content.
USER_SOURCE = "user"
USER_SOURCE_ID = "query"

DEMO_CORPUS_PATH = Path(__file__).parent / "data" / "demo_corpus.jsonl"

# How ranking kind is labeled in the schema and the UI.
RANKING_SEMANTIC = "semantic"
RANKING_LEXICAL = "lexical"

_TOKENS = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the "
    "to was were we with which who".split()
)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):  # mirrors paperdna.retrieval.embedder.Embedder
    name: str
    dimension: int | None

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class NliClassifier(Protocol):  # mirrors paperdna.nli.classifier.NliClassifier
    name: str
    max_length: int | None

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[object]: ...


@dataclass(frozen=True)
class RankedPaper:
    """One retrieval result. ``kind`` is ``"semantic"`` (model embedding
    similarity) or ``"lexical"`` (offline token-overlap fallback) — the
    two are never conflated. ``score`` is in ``[0, 1]`` for lexical and
    is the raw cosine similarity for semantic (usually ``[-1, 1]``)."""

    record: PaperRecord
    score: float
    kind: str


@dataclass(frozen=True)
class Capabilities:
    """What the current run could and could not do, with user-facing
    messages explaining anything unavailable."""

    embedder_name: str | None
    nli_model: str | None
    ranking: str  # RANKING_SEMANTIC | RANKING_LEXICAL
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationView:
    """One predicted (or evidence-only) relationship for display."""

    premise: CandidateClaim
    hypothesis: CandidateClaim
    attribution: RelationAttribution
    kind: str  # "user_vs_paper" | "paper_vs_paper"

    @property
    def classification(self) -> PairClassification | None:
        return self.attribution.classification

    @property
    def label(self) -> str | None:
        if self.attribution.interpretation is not None:
            return self.attribution.interpretation.label
        return None

    @property
    def confidence(self) -> float | None:
        if self.attribution.interpretation is not None:
            return self.attribution.interpretation.confidence
        return None

    @property
    def uncertain(self) -> bool | None:
        if self.attribution.interpretation is not None:
            return self.attribution.interpretation.uncertain
        return None

    @property
    def conditions(self) -> ConditionComparison | None:
        return self.attribution.conditions


@dataclass(frozen=True)
class DiscoveryResult:
    """Everything the UI renders. Fully materialized and immutable."""

    inputs: AppInputs
    papers: tuple[PaperRecord, ...]  # unique ingested corpus
    ranked: tuple[RankedPaper, ...]
    user_claim: CandidateClaim | None  # None when no claim could be built
    user_claim_has_real_offsets: bool
    user_conditions: ClaimConditions | None
    paper_claims: tuple[CandidateClaim, ...]
    claim_conditions: tuple[tuple[CandidateClaim, ClaimConditions], ...]
    relations: tuple[RelationView, ...]
    graph: KnowledgeGraph
    capabilities: Capabilities
    messages: tuple[str, ...] = ()
    stats: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lexical fallback ranking (app-layer, clearly labeled)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKENS.findall(text.casefold())
        if token not in _STOPWORDS and len(token) > 1
    }


def lexical_overlap(claim: str, record: PaperRecord) -> float:
    """Deterministic token-overlap score in ``[0, 1]``.

    Fraction of the claim's distinct content tokens that appear in the
    paper's ``record_to_text``-style text (title + abstract). This is a
    transparent heuristic, NOT semantic similarity.
    """
    query = _tokens(claim)
    if not query:
        return 0.0
    document = _tokens(f"{record.title} {record.abstract}")
    return len(query & document) / len(query)


def rank_lexically(
    claim: str, records: Iterable[PaperRecord], top_k: int
) -> tuple[RankedPaper, ...]:
    """Rank papers by lexical overlap; corpus order breaks ties.

    Deterministic and offline. Scores of 0.0 are dropped only when they
    would pad the list past genuinely related papers — papers with no
    overlap at all are excluded, so an empty result is a valid,
    honest state.
    """
    scored = [
        RankedPaper(record=record, score=lexical_overlap(claim, record), kind=RANKING_LEXICAL)
        for record in records
    ]
    scored = [item for item in scored if item.score > 0.0]
    scored.sort(key=lambda item: -item.score)  # stable: corpus order for ties
    return tuple(scored[:top_k])


def rank_semantically(
    embedder: Embedder, claim: str, records: list[PaperRecord], top_k: int
) -> tuple[RankedPaper, ...]:
    """Rank via the retrieval module's real public API (index + search)."""
    from paperdna.retrieval.index import PaperIndex

    if not records:
        return ()
    index = PaperIndex.build(records, embedder)
    return tuple(
        RankedPaper(record=result.record, score=result.score, kind=RANKING_SEMANTIC)
        for result in index.search(claim, top_k)
    )


# ---------------------------------------------------------------------------
# User claim handling
# ---------------------------------------------------------------------------


def _user_record(abstract: str) -> PaperRecord:
    """A synthetic record for the user's pasted abstract.

    Direct construction is deliberate: identity ``user:query`` cannot
    collide with corpus sources, and the evidence module treats a
    matching record as authoritative for verification.
    """
    return PaperRecord(
        source=USER_SOURCE,
        source_id=USER_SOURCE_ID,
        title="Your pasted abstract",
        abstract=abstract,
        authors=(),
    )


def _verbatim_user_claim(claim: str, abstract: str) -> CandidateClaim | None:
    """Real-offset user claim when the claim occurs verbatim (unique) in
    the abstract; ``None`` otherwise — offsets are never guessed."""
    occurrences = abstract.count(claim)
    if occurrences != 1:
        return None
    start = abstract.index(claim)
    return CandidateClaim(
        source=USER_SOURCE,
        source_id=USER_SOURCE_ID,
        sentence=claim,
        start=start,
        end=start + len(claim),
        cue="",  # no extractor cue: this is user input, not extracted text
        hedges=(),
    )


def _pairing_only_user_claim(claim: str) -> CandidateClaim:
    """A transient claim used ONLY to pair with ``classify_pair``.

    The dummy offsets (``0``/``len``) are never surfaced to the user,
    never serialized, and never enter the graph: user-side provenance
    comes from :func:`_hybrid_attribution`, which keeps only the paper
    side's real offsets and recomputes the user side from verbatim text,
    reporting honestly (``None`` offsets / ``unknown`` claim ids). A test
    enforces this non-leakage.
    """
    return CandidateClaim(
        source=USER_SOURCE,
        source_id=USER_SOURCE_ID,
        sentence=claim,
        start=0,
        end=len(claim),
        cue="",
        hedges=(),
    )


def _hybrid_attribution(
    premise: CandidateClaim,
    classification: PairClassification,
    *,
    records: Iterable[PaperRecord],
    conditions: ConditionComparison | None,
) -> RelationAttribution:
    """Attribute a ``user_vs_paper`` relation where the user claim is
    text-only (no verifiable abstract offsets).

    The premise is always a real paper claim: its evidence must keep the
    ``verified`` provenance that ``build_attribution`` establishes against
    the source abstract, so the same paper claim yields one consistent
    evidence node regardless of which relation it appears in. The
    hypothesis is the transient user claim, whose dummy offsets must never
    be trusted; its evidence is recomputed from verbatim text via the
    classification-only path (honest ``unverified``/``verified_by_search``/
    ``unavailable`` statuses). We therefore build the premise side with
    ``build_attribution`` and replace only the hypothesis side with the
    text-derived evidence — never re-deriving the paper side by search.
    """
    full = build_attribution(
        premise,
        premise,  # self-pair: we only need the premise's verified evidence
        records=records,
        classification=None,
        conditions=None,
    )
    text_only = attribute_from_classification(
        classification, records=records, conditions=conditions
    )
    return RelationAttribution(
        premise=full.premise,
        hypothesis=text_only.hypothesis,
        classification=classification,
        interpretation=text_only.interpretation,
        conditions=conditions,
    )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus(path: str | Path | None = None) -> tuple[PaperRecord, ...]:
    """Load and ingest a JSONL corpus through the real public API."""
    raw = load_jsonl(str(path or DEMO_CORPUS_PATH))
    result = ingest_records(raw)
    return result.papers


def _dedupe_records(records: Iterable[PaperRecord]) -> tuple[tuple[PaperRecord, ...], dict[str, int]]:
    """De-duplicate already-validated records by ``(source, source_id)``.

    Tests and callers that pass ``records=`` provide ``PaperRecord``
    objects (already validated), so we must NOT re-run them through
    ``ingest_records`` (which expects raw dicts and would reject them all).
    Duplicate identities are dropped, first occurrence wins, matching the
    upstream ``PaperIndex``/``PaperResolver`` identity rule.
    """
    unique: list[PaperRecord] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for record in records:
        if not isinstance(record, PaperRecord):
            raise TypeError(
                f"records must contain PaperRecord objects, got "
                f"{type(record).__name__}; pass raw dicts via corpus_path instead"
            )
        identity = (record.source.casefold(), record.source_id)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        unique.append(record)
    stats = {
        "loaded": len(unique) + duplicates,
        "validated": len(unique) + duplicates,
        "rejected": 0,
        "unique": len(unique),
        "duplicates": duplicates,
    }
    return tuple(unique), stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_discovery(
    inputs: AppInputs,
    *,
    records: Iterable[PaperRecord] | None = None,
    corpus_path: str | Path | None = None,
    embedder: Embedder | None = None,
    classifier: NliClassifier | None = None,
) -> DiscoveryResult:
    """Run the discovery pipeline over one set of validated inputs.

    ``records`` overrides the corpus (tests use this); otherwise the
    bundled demo corpus (or ``corpus_path``) is ingested via
    ``load_jsonl`` + ``ingest_records``. ``embedder``/``classifier`` are
    pre-constructed by the caller — this function never imports or
    loads a model, so nothing is ever downloaded here.
    """
    if not isinstance(inputs, AppInputs):
        raise TypeError(f"expected AppInputs, got {type(inputs).__name__}")

    messages: list[str] = []

    # 1. Corpus ------------------------------------------------------------
    if records is not None:
        # Already-validated records: de-duplicate, never re-ingest.
        papers, stats = _dedupe_records(records)
    else:
        raw = load_jsonl(str(corpus_path or DEMO_CORPUS_PATH))
        ingestion = ingest_records(raw)
        papers = ingestion.papers
        stats = dict(ingestion.stats)

    if not papers:
        messages.append(
            "No papers were ingested"
            + (
                f" ({stats.get('rejected', 0)} rejected, "
                f"{stats.get('duplicates', 0)} duplicates)."
                if stats.get("loaded", 0)
                else " — the corpus is empty."
            )
        )
        empty_graph = build_graph()
        return DiscoveryResult(
            inputs=inputs,
            papers=(),
            ranked=(),
            user_claim=None,
            user_claim_has_real_offsets=False,
            user_conditions=None,
            paper_claims=(),
            claim_conditions=(),
            relations=(),
            graph=empty_graph,
            capabilities=_capabilities(embedder, classifier, RANKING_LEXICAL, messages),
            messages=tuple(messages),
            stats=stats,
        )

    # 2. Ranking -----------------------------------------------------------
    if embedder is not None:
        ranked = rank_semantically(embedder, inputs.claim, list(papers), inputs.top_k)
        ranking_kind = RANKING_SEMANTIC
    else:
        ranked = rank_lexically(inputs.claim, papers, inputs.top_k)
        ranking_kind = RANKING_LEXICAL
        messages.append(
            "Semantic retrieval is off, so papers are ranked by offline "
            "token overlap — a rough lexical heuristic, NOT meaning-aware "
            "similarity. Enable the embedding model for semantic ranking."
        )
    if not ranked:
        messages.append(
            "No paper in the corpus is lexically related to the claim. "
            "Try rephrasing with the keywords used by the papers."
        )

    # 3. User claim and abstract -------------------------------------------
    user_record: PaperRecord | None = None
    user_claim: CandidateClaim | None = None
    if inputs.abstract is not None:
        user_record = _user_record(inputs.abstract)
    real_user_claim = (
        _verbatim_user_claim(inputs.claim, inputs.abstract)
        if inputs.abstract is not None
        else None
    )
    user_claim_has_real_offsets = real_user_claim is not None
    if user_claim_has_real_offsets:
        user_claim = real_user_claim
    else:
        # Pairing-only claim: provenance comes from the classification-only
        # attribution path, never from these placeholder offsets.
        user_claim = _pairing_only_user_claim(inputs.claim)
        if inputs.abstract is not None:
            messages.append(
                "Your claim does not appear verbatim (exactly once) in the "
                "pasted abstract, so its offsets cannot be verified against "
                "it; evidence for your claim is marked accordingly."
            )

    user_conditions = extract_conditions(user_claim)

    # 4. Claims from retrieved papers --------------------------------------
    paper_claims: list[CandidateClaim] = []
    claim_conditions: list[tuple[CandidateClaim, ClaimConditions]] = []
    for item in ranked:
        for claim in extract_claims(item.record).claims:
            paper_claims.append(claim)
            claim_conditions.append((claim, extract_conditions(claim)))
    if ranked and not paper_claims:
        messages.append(
            "The retrieved papers yielded no candidate claims: the rule-based "
            "extractor found no finding-cue sentences. This is a baseline "
            "limitation, not evidence of absence."
        )

    # Deterministic pair order: corpus ranking order, then claim offsets.
    relations: list[RelationView] = []
    all_records = list(papers) + ([user_record] if user_record is not None else [])

    # 5. Relations ---------------------------------------------------------
    if classifier is None:
        messages.append(
            "Relationship classification is off, so no support/"
            "contradiction/neutral labels are shown. Enable the NLI model "
            "to compare claims pairwise."
        )
    else:
        pairs: list[tuple[CandidateClaim, CandidateClaim, str]] = []
        for claim in paper_claims:
            pairs.append((claim, user_claim, "user_vs_paper"))
        if inputs.compare_papers:
            for i, premise in enumerate(paper_claims):
                for hypothesis in paper_claims[i + 1 :]:
                    pairs.append((premise, hypothesis, "paper_vs_paper"))
        if len(pairs) > inputs.max_pairs:
            messages.append(
                f"Comparison capped at {inputs.max_pairs} of {len(pairs)} "
                "possible claim pairs (limit setting)."
            )
            pairs = pairs[: inputs.max_pairs]

        conditions_cache: dict[
            tuple[tuple[str, str, int, int], tuple[str, str, int, int]],
            ConditionComparison,
        ] = {}
        for premise, hypothesis, kind in pairs:
            classification = classify_pair(premise, hypothesis, classifier)
            cache_key = (
                (premise.source, premise.source_id, premise.start, premise.end),
                (hypothesis.source, hypothesis.source_id, hypothesis.start, hypothesis.end),
            )
            comparison = conditions_cache.get(cache_key)
            if comparison is None:
                comparison = compare_conditions(
                    extract_conditions(premise), extract_conditions(hypothesis)
                )
                conditions_cache[cache_key] = comparison

            involves_pairing_only_user = (
                kind == "user_vs_paper" and not user_claim_has_real_offsets
            )
            if involves_pairing_only_user:
                # User claim is text-only: keep the paper premise's real
                # verified provenance; recompute only the user hypothesis
                # side from verbatim text (honest None offsets).
                attribution = _hybrid_attribution(
                    premise,
                    classification,
                    records=all_records,
                    conditions=comparison,
                )
            else:
                attribution = build_attribution(
                    premise,
                    hypothesis,
                    records=all_records,
                    classification=classification,
                    conditions=comparison,
                )
            relations.append(
                RelationView(
                    premise=premise,
                    hypothesis=hypothesis,
                    attribution=attribution,
                    kind=kind,
                )
            )

    # 6. Graph -------------------------------------------------------------
    graph_records: list[PaperRecord] = [item.record for item in ranked]
    if user_record is not None:
        graph_records.append(user_record)
    graph_claims: list[CandidateClaim] = list(paper_claims)
    graph_conditions = list(claim_conditions)
    if user_claim_has_real_offsets and user_claim is not None:
        graph_claims.append(user_claim)
        graph_conditions.append((user_claim, user_conditions))
    graph = build_graph(
        papers=graph_records,
        claims=graph_claims,
        conditions=graph_conditions,
        attributions=[view.attribution for view in relations],
    )

    return DiscoveryResult(
        inputs=inputs,
        papers=papers,
        ranked=ranked,
        user_claim=user_claim,
        user_claim_has_real_offsets=user_claim_has_real_offsets,
        user_conditions=user_conditions,
        paper_claims=tuple(paper_claims),
        claim_conditions=tuple(claim_conditions),
        relations=tuple(relations),
        graph=graph,
        capabilities=_capabilities(embedder, classifier, ranking_kind, messages),
        messages=tuple(messages),
        stats=stats,
    )


def _capabilities(
    embedder: Embedder | None,
    classifier: NliClassifier | None,
    ranking: str,
    messages: list[str],
) -> Capabilities:
    return Capabilities(
        embedder_name=getattr(embedder, "name", None),
        nli_model=getattr(classifier, "name", None),
        ranking=ranking,
        messages=tuple(messages),
    )
