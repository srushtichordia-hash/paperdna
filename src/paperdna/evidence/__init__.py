"""Evidence attribution and sentence traceability.

Links every predicted relationship to the exact source sentences behind
it: verbatim evidence text with verified abstract offsets, source-paper
identity (with title/URL/DOI when available), stable claim references,
and the model's interpretation kept strictly separate from observed
text. Missing provenance is marked, never fabricated; invalid spans are
rejected loudly.

Limitations: verification is textual identity against the abstract that
was available at attribution time, not scientific grounding; a
``PairClassification`` on its own carries no offsets, so the
classification-only path can only infer them from a unique sentence
occurrence (labeled ``verified_by_search``); claim IDs are deterministic
but tied to the extractor's current segmentation. See the README
"Evidence attribution & traceability" section.
"""

from paperdna.evidence.build import (
    attribute_from_classification,
    attribute_nli_with_context,
    build_attribution,
    verify_span,
)
from paperdna.evidence.errors import (
    AttributionMismatchError,
    EvidenceError,
    InvalidSpanError,
)
from paperdna.evidence.resolver import PaperResolver, resolve_papers
from paperdna.evidence.schema import (
    EVIDENCE_STATUSES,
    EVIDENCE_VERIFICATION,
    ClaimRef,
    PaperRef,
    RelationAttribution,
    RelationInterpretation,
    SentenceEvidence,
    claim_id_for,
)
from paperdna.evidence.serialize import attribution_from_dict, attribution_to_dict

__all__ = [
    "EVIDENCE_STATUSES",
    "EVIDENCE_VERIFICATION",
    "AttributionMismatchError",
    "ClaimRef",
    "EvidenceError",
    "InvalidSpanError",
    "PaperRef",
    "PaperResolver",
    "RelationAttribution",
    "RelationInterpretation",
    "SentenceEvidence",
    "attribute_from_classification",
    "attribute_nli_with_context",
    "attribution_from_dict",
    "attribution_to_dict",
    "build_attribution",
    "claim_id_for",
    "resolve_papers",
    "verify_span",
]
