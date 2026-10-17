"""Text, author-list, and DOI normalization helpers."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")

# Common DOI presentation forms seen in metadata feeds and citations.
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)

# Shape of a DOI: "10." registrar prefix, 4-9 digit suffix, "/" separator.
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def normalize_text(value: str) -> str:
    """Normalize a text field.

    Applies Unicode NFC normalization, collapses all internal whitespace
    runs (spaces, tabs, newlines) to single spaces, and strips the ends.
    """
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def normalize_authors(value: object) -> tuple[str, ...] | None:
    """Normalize an author field into a tuple of non-empty names.

    Accepts a comma-separated string or a list/tuple of strings, preserving
    order. Returns ``None`` if the value is of an unusable type or contains
    non-string members; an empty tuple means all entries were blank.
    """
    if isinstance(value, str):
        items: list[object] = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return None

    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            return None
        name = normalize_text(item)
        if name:
            result.append(name)
    return tuple(result)


def normalize_doi(value: str) -> str | None:
    """Normalize a DOI into its canonical lowercase form.

    Strips ``doi:`` and ``https://doi.org/``-style URL prefixes, lowercases,
    and removes a trailing period (common when text was scraped from
    citations). Returns ``None`` if the remainder does not look like a DOI.
    """
    doi = normalize_text(value)
    lowered = doi.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    doi = doi.strip().lower().rstrip(".")
    if _DOI_RE.match(doi):
        return doi
    return None
