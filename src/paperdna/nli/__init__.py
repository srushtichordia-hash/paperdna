"""Natural Language Inference for comparing candidate research claims.

The heavy model stack (PyTorch, sentence-transformers) is optional:
install it with ``pip install -e ".[nli]"``. Nothing here imports it
until a ``CrossEncoderNliClassifier`` is explicitly instantiated.

This module estimates textual relationships between claim pairs; it does
not establish scientific truth.
"""

from paperdna.nli.classifier import (
    DEFAULT_NLI_MODEL,
    CrossEncoderNliClassifier,
    NliClassifier,
)
from paperdna.nli.classify import classify_pair, classify_pairs
from paperdna.nli.errors import InvalidScoresError, NliError, NliUnavailableError
from paperdna.nli.eval import classification_report
from paperdna.nli.labels import DEFAULT_LABEL_ORDER, canonical_label, resolve_label_map
from paperdna.nli.schema import (
    RELATIONSHIPS,
    ClaimPair,
    PairClassification,
    RawPairScores,
)

__all__ = [
    "DEFAULT_LABEL_ORDER",
    "DEFAULT_NLI_MODEL",
    "ClaimPair",
    "CrossEncoderNliClassifier",
    "InvalidScoresError",
    "NliClassifier",
    "NliError",
    "NliUnavailableError",
    "PairClassification",
    "RawPairScores",
    "RELATIONSHIPS",
    "canonical_label",
    "classify_pair",
    "classify_pairs",
    "classification_report",
    "resolve_label_map",
]
