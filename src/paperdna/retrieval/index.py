"""Paper index: embed a corpus once, then answer similarity queries.

Model loading lives in :mod:`paperdna.retrieval.embedder`; this module only
consumes an :class:`Embedder`, so retrieval logic never touches a specific
model or its dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from paperdna.corpus.schema import PaperRecord
from paperdna.retrieval.embedder import Embedder
from paperdna.retrieval.errors import InvalidVectorError
from paperdna.retrieval.preprocess import record_to_text
from paperdna.retrieval.similarity import rank, validate_vector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """One ranked paper: the original (unmutated) record plus its score."""

    record: PaperRecord
    score: float


class PaperIndex:
    """A corpus embedded once and searched many times.

    Paper identity is ``(source.casefold(), source_id)``. Records repeating
    an already-seen identity are dropped at build time: the first
    occurrence is kept, later ones are counted in ``duplicate_ids_dropped``
    and logged at DEBUG. Original ``PaperRecord`` objects are stored and
    returned unmutated, preserving full source traceability.
    """

    def __init__(
        self,
        records: Sequence[PaperRecord],
        vectors: Sequence[Sequence[float]],
        embedder: Embedder,
    ) -> None:
        self._records = tuple(records)
        self._vectors = tuple(tuple(float(x) for x in v) for v in vectors)
        self._embedder = embedder
        self._duplicate_ids_dropped = 0

    @property
    def dimension(self) -> int:
        return len(self._vectors[0]) if self._vectors else 0

    @property
    def records(self) -> tuple[PaperRecord, ...]:
        """The indexed (unique) records, in input order."""
        return self._records

    @property
    def duplicate_ids_dropped(self) -> int:
        """Number of records dropped for repeating an existing identity."""
        return self._duplicate_ids_dropped

    @property
    def stats(self) -> dict[str, int]:
        return {
            "indexed": len(self._records),
            "duplicate_ids_dropped": self._duplicate_ids_dropped,
            "dimension": self.dimension,
        }

    @classmethod
    def build(cls, records: Iterable[PaperRecord], embedder: Embedder) -> "PaperIndex":
        """Validate, deduplicate by identity, and embed the corpus.

        Every record must be a ``PaperRecord`` (validated corpus output).
        Embeddings are computed exactly once per index; invalid vectors
        (wrong shape, non-finite values, zero norm) raise
        ``InvalidVectorError`` instead of being coerced. An empty corpus
        produces an empty index with no embedding calls.
        """
        unique: list[PaperRecord] = []
        seen: set[tuple[str, str]] = set()
        dropped = 0
        for record in records:
            if not isinstance(record, PaperRecord):
                raise TypeError(f"expected PaperRecord, got {type(record).__name__}")
            identity = (record.source.casefold(), record.source_id)
            if identity in seen:
                dropped += 1
                logger.debug(
                    "Dropping duplicate paper %s:%s", record.source, record.source_id
                )
                continue
            seen.add(identity)
            unique.append(record)

        index = cls(records=unique, vectors=(), embedder=embedder)
        index._duplicate_ids_dropped = dropped
        if not unique:
            return index

        texts = [record_to_text(record) for record in unique]
        raw_vectors = embedder.embed(texts)
        if len(raw_vectors) != len(texts):
            raise InvalidVectorError(
                f"embedder returned {len(raw_vectors)} vectors for {len(texts)} texts"
            )
        # First vector realizes the dimension; all others (and future query
        # vectors) must match it exactly. If the embedder declares a
        # dimension, it must match reality.
        first = validate_vector(raw_vectors[0], dimension=embedder.dimension)
        dimension = len(first)
        index._vectors = (first,) + tuple(
            validate_vector(vector, dimension=dimension) for vector in raw_vectors[1:]
        )
        return index

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the ``top_k`` most similar papers to the query text.

        Ranking is by descending cosine similarity; ties keep corpus order
        (stable sort), so results are deterministic. An empty corpus
        returns ``[]``. A blank query or ``top_k < 1`` raises
        ``ValueError``. The query is embedded at call time; corpus vectors
        are reused from build time, so repeated searches never re-embed the
        corpus.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not self._records:
            return []

        raw_query_vectors = self._embedder.embed([query])
        if len(raw_query_vectors) != 1:
            raise InvalidVectorError(
                f"embedder returned {len(raw_query_vectors)} vectors for 1 query"
            )
        query_vector = validate_vector(raw_query_vectors[0], dimension=self.dimension)

        scored = rank(query_vector, self._vectors)[:top_k]
        return [
            SearchResult(record=self._records[index], score=score)
            for index, score in scored
        ]
