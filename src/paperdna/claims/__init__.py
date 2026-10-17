"""Candidate claim extraction from research abstracts.

A lightweight, deterministic, offline baseline: rule-based sentence
segmentation plus lexical cue classification. It preserves exact source
sentences and cautious language; it does not understand scientific
truth. See the README "Claim extraction" section for limitations.
"""

from paperdna.claims.extractor import extract_claims, extract_claims_batch
from paperdna.claims.schema import CandidateClaim, ExtractionResult, SentenceSpan
from paperdna.claims.segment import segment_sentences

__all__ = [
    "CandidateClaim",
    "ExtractionResult",
    "SentenceSpan",
    "extract_claims",
    "extract_claims_batch",
    "segment_sentences",
]
