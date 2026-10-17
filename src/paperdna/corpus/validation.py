"""Validation of raw payload dictionaries into ``PaperRecord`` objects.

``validate_record`` is a pure function that never raises on bad input: any
malformed or incomplete payload produces a ``RejectedRecord`` describing the
problem, with the original payload preserved.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re

from paperdna.corpus.normalize import normalize_authors, normalize_doi, normalize_text
from paperdna.corpus.schema import PaperRecord, RejectedRecord

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"^\d{4}$")
_FULL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _reject(raw: object, reason: str) -> RejectedRecord:
    return RejectedRecord(raw=raw, reason=reason)


def _valid_date(value: str) -> bool:
    """Check ``YYYY`` or a real ``YYYY-MM-DD`` calendar date."""
    if _FULL_DATE_RE.match(value):
        try:
            _dt.date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if _YEAR_RE.match(value):
        return 1 <= int(value) <= 9999
    return False


def _require_str(raw: dict, field_name: str) -> tuple[str | None, str | None]:
    """Read a required string field; returns (normalized_value, error_reason)."""
    value = raw.get(field_name)
    if value is None:
        return None, f"{field_name}: missing"
    if not isinstance(value, str):
        return None, (
            f"{field_name}: expected a string, got {type(value).__name__}"
        )
    normalized = normalize_text(value)
    if not normalized:
        return None, f"{field_name}: empty after normalization"
    return normalized, None


def validate_record(raw: object) -> PaperRecord | RejectedRecord:
    """Validate one raw record into a ``PaperRecord`` or ``RejectedRecord``.

    Required fields: ``source``, ``source_id``, ``title``, ``abstract``,
    ``authors``. Optional: ``published`` ("YYYY" or "YYYY-MM-DD"), ``url``
    (http/https), ``doi``. Unknown extra keys are ignored. A malformed
    optional ``doi`` is dropped (with a warning) rather than failing the
    record, since DOIs are optional; a malformed ``published`` or ``url``
    fails the record so bad data is never silently rewritten.
    """
    if not isinstance(raw, dict):
        return _reject(raw, "record is not a dictionary")

    for field_name in ("source", "source_id", "title", "abstract"):
        _, error = _require_str(raw, field_name)
        if error:
            return _reject(raw, error)
    source = normalize_text(raw["source"])
    source_id = normalize_text(raw["source_id"])
    title = normalize_text(raw["title"])
    abstract = normalize_text(raw["abstract"])

    authors_raw = raw.get("authors")
    if authors_raw is None:
        return _reject(raw, "authors: missing")
    if not isinstance(authors_raw, (str, list, tuple)):
        return _reject(
            raw,
            "authors: expected a string or list of strings, "
            f"got {type(authors_raw).__name__}",
        )
    authors = normalize_authors(authors_raw)
    if authors is None:
        return _reject(raw, "authors: expected a list of strings")
    if not authors:
        return _reject(raw, "authors: empty after normalization")

    published = raw.get("published")
    if published is not None:
        if not isinstance(published, str):
            return _reject(
                raw,
                f"published: expected a string, got {type(published).__name__}",
            )
        published = published.strip()
        if not _valid_date(published):
            return _reject(
                raw,
                "published: not a valid date (expected YYYY or YYYY-MM-DD)",
            )

    url = raw.get("url")
    if url is not None:
        if not isinstance(url, str):
            return _reject(raw, f"url: expected a string, got {type(url).__name__}")
        url = url.strip()
        if url and not url.startswith(("http://", "https://")):
            return _reject(raw, "url: must start with http:// or https://")
        url = url or None

    doi = raw.get("doi")
    if doi is not None:
        if not isinstance(doi, str):
            return _reject(raw, f"doi: expected a string, got {type(doi).__name__}")
        normalized_doi = normalize_doi(doi)
        if normalized_doi is None:
            logger.warning(
                "Dropping unparseable DOI %r for %s:%s", doi, source, source_id
            )
        doi = normalized_doi

    return PaperRecord(
        source=source,
        source_id=source_id,
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        url=url,
        doi=doi,
    )
