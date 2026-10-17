"""Error types for evidence attribution and traceability.

``InvalidSpanError`` also subclasses ``ValueError`` so callers can treat
structurally bad offsets as malformed input uniformly.
"""

from __future__ import annotations


class EvidenceError(Exception):
    """Base class for evidence-attribution failures."""


class InvalidSpanError(EvidenceError, ValueError):
    """Offsets are structurally invalid or do not match the known abstract.

    Raised rather than silently repaired: a span that fails
    ``abstract[start:end] == sentence`` means a wrong record pairing or a
    fabricated claim, never a condition to paper over.
    """


class AttributionMismatchError(EvidenceError):
    """An input (classification or condition comparison) does not belong to
    the claims being attributed — source identity or evidence text disagrees.
    """
