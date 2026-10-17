"""Deterministic mapping of paper records to embedding input text.

The format is stable and documented: title, two newlines, abstract. This
keeps preprocessing reproducible and makes it easy to re-derive the exact
text a stored vector was computed from.
"""

from __future__ import annotations

from paperdna.corpus.schema import PaperRecord


def record_to_text(record: PaperRecord) -> str:
    """Render a record as embedding input: ``"{title}\\n\\n{abstract}"``.

    Fields are used as validated (already whitespace-normalized). The record
    is never mutated.
    """
    if not isinstance(record, PaperRecord):
        raise TypeError(
            f"expected PaperRecord, got {type(record).__name__}"
        )
    return f"{record.title}\n\n{record.abstract}"
