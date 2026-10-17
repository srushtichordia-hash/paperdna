"""Cosine similarity and deterministic ranking.

Vectors are plain sequences of floats; the core stays dependency-free.
Invalid vectors (wrong dimension, non-finite values, zero norm) raise
``InvalidVectorError`` and are never silently coerced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from paperdna.retrieval.errors import InvalidVectorError


def validate_vector(
    vector: Sequence[float], *, dimension: int | None = None
) -> tuple[float, ...]:
    """Validate one vector and return it as a float tuple.

    Raises ``InvalidVectorError`` if the vector is empty, contains
    non-numeric or non-finite values, has zero norm, or does not match
    ``dimension`` when one is given.
    """
    try:
        values = tuple(float(x) for x in vector)
    except (TypeError, ValueError) as exc:
        raise InvalidVectorError(f"vector contains non-numeric values: {exc}") from exc

    if not values:
        raise InvalidVectorError("vector is empty")
    if dimension is not None and len(values) != dimension:
        raise InvalidVectorError(
            f"vector has dimension {len(values)}, expected {dimension}"
        )
    if not all(math.isfinite(x) for x in values):
        raise InvalidVectorError("vector contains non-finite values")
    if sum(x * x for x in values) == 0.0:
        raise InvalidVectorError("vector has zero norm")
    return values


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors, in [-1, 1].

    Both vectors are validated (including matching each other's length);
    zero-norm vectors raise rather than producing NaN.
    """
    va = validate_vector(a)
    vb = validate_vector(b, dimension=len(va))
    dot = sum(x * y for x, y in zip(va, vb))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(x * x for x in vb))
    return dot / (norm_a * norm_b)


def rank(
    query_vector: Sequence[float], vectors: Sequence[Sequence[float]]
) -> list[tuple[int, float]]:
    """Score every corpus vector against the query and rank descending.

    Returns ``(input_index, score)`` pairs sorted by score. Ties keep input
    order (stable sort), so ranking is deterministic. Corpus vectors are
    validated against the query vector's dimension.
    """
    query = validate_vector(query_vector)
    scored = [
        (index, cosine(query, validate_vector(vector, dimension=len(query))))
        for index, vector in enumerate(vectors)
    ]
    scored.sort(key=lambda pair: -pair[1])  # stable: ties keep input order
    return scored
