"""Resolving (source, source_id) to paper records for attribution."""

from __future__ import annotations

import logging
from typing import Iterable

from paperdna.corpus.schema import PaperRecord

logger = logging.getLogger(__name__)


class PaperResolver:
    """Lookup of papers by ``(source.casefold(), source_id)``.

    Same identity convention as ``PaperIndex``. Duplicate identities are
    dropped deterministically: the first occurrence wins; later ones are
    counted in ``duplicate_ids_dropped`` and never overwrite the first.
    """

    def __init__(self, records: Iterable[PaperRecord]) -> None:
        self._by_identity: dict[tuple[str, str], PaperRecord] = {}
        self._duplicate_ids_dropped = 0
        for record in records:
            if not isinstance(record, PaperRecord):
                raise TypeError(f"expected PaperRecord, got {type(record).__name__}")
            key = (record.source.casefold(), record.source_id)
            if key in self._by_identity:
                self._duplicate_ids_dropped += 1
                logger.debug(
                    "Duplicate paper identity %s:%s dropped during resolution",
                    record.source,
                    record.source_id,
                )
                continue
            self._by_identity[key] = record

    def get(self, source: str, source_id: str) -> PaperRecord | None:
        """The paper for this identity, or ``None`` when not in the corpus."""
        return self._by_identity.get((source.casefold(), source_id))

    @property
    def duplicate_ids_dropped(self) -> int:
        return self._duplicate_ids_dropped

    def __len__(self) -> int:
        return len(self._by_identity)


def resolve_papers(records: Iterable[PaperRecord]) -> PaperResolver:
    """Build a :class:`PaperResolver` from any iterable of paper records."""
    return PaperResolver(records)
