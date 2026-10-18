"""Retrieval evaluation: Precision@k and Recall@k from actual rankings.

Runs labeled benchmark queries through a ranking system, collects the
ACTUAL retrieved order, and computes ``paperdna.retrieval``'s
documented P@k / R@k metrics against the hand-labeled relevance sets.

Systems measured offline by default:

- ``lexical``: the app's real offline fallback, ``rank_lexically``
  (token overlap; NOT semantic similarity — labeled as such in output).

Opt-in (requires the ``embed`` extra, explicitly installed):

- ``semantic``: ``PaperIndex`` + ``SentenceTransformerEmbedder``.

Conventions (from ``paperdna.retrieval.eval``): fewer than ``k``
retrieved divides by ``k``; an empty relevance set scores 0.0 — and the
benchmark loader rejects empty relevance sets, so that case cannot
occur here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from paperdna.api.pipeline import rank_lexically
from paperdna.corpus.schema import PaperRecord
from paperdna.retrieval.eval import precision_at_k, recall_at_k

DEFAULT_KS = (1, 3, 5)

# Ordered paper ids for a query over a corpus.
RankFn = Callable[[str, Sequence[PaperRecord]], list[str]]


def lexical_rank(query: str, records: Sequence[PaperRecord]) -> list[str]:
    """The app's offline fallback ranking, as ordered paper ids."""
    ranked = rank_lexically(query, records, top_k=len(records))
    return [f"{item.record.source}:{item.record.source_id}" for item in ranked]


def semantic_rank(embedder, query: str, records: Sequence[PaperRecord]) -> list[str]:
    """Optional real-model ranking via the retrieval module's public API."""
    from paperdna.retrieval.index import PaperIndex

    index = PaperIndex.build(list(records), embedder)
    return [
        f"{result.record.source}:{result.record.source_id}"
        for result in index.search(query, top_k=len(records))
    ]


def evaluate_retrieval(
    records: Sequence[PaperRecord],
    queries: Sequence,  # Sequence[RetrievalQuery]
    rank_fn: RankFn,
    *,
    system: str,
    ranking_kind: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Compute per-query and aggregate P@k / R@k for one system.

    ``ranking_kind`` must be ``"semantic"`` or ``"lexical"`` and is
    copied into the report so downstream readers can never conflate a
    token-overlap score with model similarity.
    """
    if ranking_kind not in ("semantic", "lexical"):
        raise ValueError(f"ranking_kind must be 'semantic' or 'lexical', got {ranking_kind!r}")
    ks = tuple(int(k) for k in ks)
    if not ks or any(k < 1 for k in ks):
        raise ValueError(f"ks must be positive integers, got {ks!r}")

    rows: list[dict] = []
    for item in queries:
        retrieved = rank_fn(item.query, records)
        row: dict = {
            "query": item.query,
            "split": item.split,
            "relevant": list(item.relevant),
            "retrieved": retrieved,
            "metrics": {},
        }
        for k in ks:
            row["metrics"][f"precision@{k}"] = precision_at_k(retrieved, item.relevant, k)
            row["metrics"][f"recall@{k}"] = recall_at_k(retrieved, item.relevant, k)
        rows.append(row)

    aggregate: dict[str, float] = {}
    n = len(rows)
    for k in ks:
        p_key, r_key = f"precision@{k}", f"recall@{k}"
        aggregate[p_key] = round(sum(r["metrics"][p_key] for r in rows) / n, 6) if n else 0.0
        aggregate[r_key] = round(sum(r["metrics"][r_key] for r in rows) / n, 6) if n else 0.0

    return {
        "system": system,
        "ranking_kind": ranking_kind,
        "scores_are_calibrated": False,
        "n_papers": len(records),
        "n_queries": n,
        "ks": list(ks),
        "aggregate_mean": aggregate,
        "per_query": rows,
    }
