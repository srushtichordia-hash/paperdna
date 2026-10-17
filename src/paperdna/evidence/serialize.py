"""JSON-safe serialization and exact round-trip for attributions.

The dict shape is a small, documented public surface intended for storage
and the future UI: plain lists/dicts/str/float/bool/None only, so
``json.dumps`` always succeeds. ``attribution_from_dict`` is strict —
missing or malformed fields raise ``ValueError``, never guesses.
"""

from __future__ import annotations

from paperdna.conditions.schema import ConditionComparison, FieldComparison
from paperdna.evidence.schema import (
    ClaimRef,
    PaperRef,
    RelationAttribution,
    RelationInterpretation,
    SentenceEvidence,
)
from paperdna.nli.schema import PairClassification


def _require(data: dict, key: str) -> object:
    if key not in data:
        raise ValueError(f"malformed attribution dict: missing key {key!r}")
    return data[key]


def _paper_to_dict(paper: PaperRef) -> dict:
    return {
        "source": paper.source,
        "source_id": paper.source_id,
        "title": paper.title,
        "url": paper.url,
        "doi": paper.doi,
        "published": paper.published,
    }


def _claim_to_dict(ref: ClaimRef) -> dict:
    return {
        "claim_id": ref.claim_id,
        "source": ref.source,
        "source_id": ref.source_id,
        "start": ref.start,
        "end": ref.end,
    }


def _evidence_to_dict(evidence: SentenceEvidence) -> dict:
    return {
        "paper": _paper_to_dict(evidence.paper),
        "claim": _claim_to_dict(evidence.claim),
        "sentence": evidence.sentence,
        "verification": evidence.verification,
        "unavailable_reason": evidence.unavailable_reason,
    }


def _classification_to_dict(c: PairClassification) -> dict:
    return {
        "premise_source": c.premise_source,
        "premise_source_id": c.premise_source_id,
        "premise_text": c.premise_text,
        "hypothesis_source": c.hypothesis_source,
        "hypothesis_source_id": c.hypothesis_source_id,
        "hypothesis_text": c.hypothesis_text,
        "label": c.label,
        "scores": dict(c.scores),
        "confidence": c.confidence,
        "uncertain": c.uncertain,
        "hedged": c.hedged,
        "truncated": c.truncated,
        "model": c.model,
    }


def _interpretation_to_dict(i: RelationInterpretation) -> dict:
    return {
        "label": i.label,
        "scores": dict(i.scores),
        "confidence": i.confidence,
        "uncertain": i.uncertain,
        "model": i.model,
        "condition_summary": i.condition_summary,
        "condition_stats": (
            dict(i.condition_stats) if i.condition_stats is not None else None
        ),
        "notes": list(i.notes),
    }


def _conditions_to_dict(c: ConditionComparison) -> dict:
    return {
        "premise_source": c.premise_source,
        "premise_source_id": c.premise_source_id,
        "hypothesis_source": c.hypothesis_source,
        "hypothesis_source_id": c.hypothesis_source_id,
        "fields": [
            {
                "category": f.category,
                "status": f.status,
                "premise_values": list(f.premise_values),
                "hypothesis_values": list(f.hypothesis_values),
            }
            for f in c.fields
        ],
        "stats": dict(c.stats),
    }


def attribution_to_dict(attribution: RelationAttribution) -> dict:
    """JSON-safe dict for one attribution (round-trips exactly)."""
    if not isinstance(attribution, RelationAttribution):
        raise TypeError(f"expected RelationAttribution, got {type(attribution).__name__}")
    return {
        "premise": _evidence_to_dict(attribution.premise),
        "hypothesis": _evidence_to_dict(attribution.hypothesis),
        "classification": (
            _classification_to_dict(attribution.classification)
            if attribution.classification is not None
            else None
        ),
        "interpretation": (
            _interpretation_to_dict(attribution.interpretation)
            if attribution.interpretation is not None
            else None
        ),
        "conditions": (
            _conditions_to_dict(attribution.conditions)
            if attribution.conditions is not None
            else None
        ),
        "stats": dict(attribution.stats),
    }


def _paper_from_dict(data: dict) -> PaperRef:
    return PaperRef(
        source=str(_require(data, "source")),
        source_id=str(_require(data, "source_id")),
        title=data.get("title"),
        url=data.get("url"),
        doi=data.get("doi"),
        published=data.get("published"),
    )


def _claim_from_dict(data: dict) -> ClaimRef:
    return ClaimRef(
        claim_id=str(_require(data, "claim_id")),
        source=str(_require(data, "source")),
        source_id=str(_require(data, "source_id")),
        start=data.get("start"),
        end=data.get("end"),
    )


def _evidence_from_dict(data: dict) -> SentenceEvidence:
    return SentenceEvidence(
        paper=_paper_from_dict(_require(data, "paper")),  # type: ignore[arg-type]
        claim=_claim_from_dict(_require(data, "claim")),  # type: ignore[arg-type]
        sentence=str(_require(data, "sentence")),
        verification=str(_require(data, "verification")),
        unavailable_reason=data.get("unavailable_reason"),
    )


def _classification_from_dict(data: dict) -> PairClassification:
    scores = _require(data, "scores")
    if not isinstance(scores, dict):
        raise ValueError("malformed attribution dict: 'scores' must be a dict")
    return PairClassification(
        premise_source=str(_require(data, "premise_source")),
        premise_source_id=str(_require(data, "premise_source_id")),
        premise_text=str(_require(data, "premise_text")),
        hypothesis_source=str(_require(data, "hypothesis_source")),
        hypothesis_source_id=str(_require(data, "hypothesis_source_id")),
        hypothesis_text=str(_require(data, "hypothesis_text")),
        label=str(_require(data, "label")),
        scores={str(k): float(v) for k, v in scores.items()},
        confidence=float(_require(data, "confidence")),
        uncertain=bool(_require(data, "uncertain")),
        hedged=bool(_require(data, "hedged")),
        truncated=bool(_require(data, "truncated")),
        model=str(_require(data, "model")),
    )


def _interpretation_from_dict(data: dict) -> RelationInterpretation:
    stats = data.get("condition_stats")
    if stats is not None and not isinstance(stats, dict):
        raise ValueError("malformed attribution dict: 'condition_stats' must be a dict or None")
    scores = _require(data, "scores")
    if not isinstance(scores, dict):
        raise ValueError("malformed attribution dict: 'scores' must be a dict")
    return RelationInterpretation(
        label=str(_require(data, "label")),
        scores={str(k): float(v) for k, v in scores.items()},
        confidence=float(_require(data, "confidence")),
        uncertain=bool(_require(data, "uncertain")),
        model=str(_require(data, "model")),
        condition_summary=data.get("condition_summary"),
        condition_stats=(
            {str(k): int(v) for k, v in stats.items()} if stats is not None else None
        ),
        notes=tuple(str(n) for n in data.get("notes", ())),
    )


def _conditions_from_dict(data: dict) -> ConditionComparison:
    fields = _require(data, "fields")
    if not isinstance(fields, list):
        raise ValueError("malformed attribution dict: 'fields' must be a list")
    stats = _require(data, "stats")
    if not isinstance(stats, dict):
        raise ValueError("malformed attribution dict: 'stats' must be a dict")
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


def attribution_from_dict(data: dict) -> RelationAttribution:
    """Rebuild an attribution from its dict form.

    Round-trips exactly (``from_dict(to_dict(x)) == x``). Missing or
    malformed fields raise ``ValueError``; nothing is guessed.
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected dict, got {type(data).__name__}")
    classification = data.get("classification")
    interpretation = data.get("interpretation")
    conditions = data.get("conditions")
    stats = _require(data, "stats")
    if not isinstance(stats, dict):
        raise ValueError("malformed attribution dict: 'stats' must be a dict")
    return RelationAttribution(
        premise=_evidence_from_dict(_require(data, "premise")),  # type: ignore[arg-type]
        hypothesis=_evidence_from_dict(_require(data, "hypothesis")),  # type: ignore[arg-type]
        classification=(
            _classification_from_dict(classification)
            if classification is not None
            else None
        ),
        interpretation=(
            _interpretation_from_dict(interpretation)
            if interpretation is not None
            else None
        ),
        conditions=(_conditions_from_dict(conditions) if conditions is not None else None),
        stats={str(k): int(v) for k, v in stats.items()},
    )
