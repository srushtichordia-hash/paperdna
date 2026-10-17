"""Retrieval quality metrics: Precision@k and Recall@k.

Conventions (documented and covered by tests):

- ``k`` must be a positive integer (``k >= 1``); anything else raises
  ``ValueError``.
- Both metrics are computed over the first ``k`` retrieved items. When
  fewer than ``k`` items were retrieved, the denominator stays ``k``
  (a missing result counts against the metric).
- An empty relevance set yields ``0.0`` for both metrics: nothing can be
  retrieved correctly, and nothing can be missed.
- Relevance and retrieval membership are exact-match on paper identity
  strings (e.g. ``"arxiv:2101.00001"``).
"""

from __future__ import annotations

from collections.abc import Iterable


def _normalize_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    return k


def precision_at_k(
    retrieved: Iterable[str], relevant: Iterable[str], k: int
) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    k = _normalize_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(retrieved)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top-k retrieved."""
    k = _normalize_k(k)
    relevant_list = list(relevant)
    if not relevant_list:
        return 0.0
    top_k_set = set(list(retrieved)[:k])
    hits = sum(1 for item in relevant_list if item in top_k_set)
    return hits / len(relevant_list)
