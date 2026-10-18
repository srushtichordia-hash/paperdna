"""Input validation for the PaperDNA web API.

Pure and deterministic: no FastAPI, no models. Every limit exists to
turn a hostile or sloppy input into an actionable message before it
reaches the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Length caps. The claim is the query; the abstract is optional context.
MAX_CLAIM_CHARS = 1_000
MAX_ABSTRACT_CHARS = 12_000

TOP_K_MIN, TOP_K_MAX = 1, 10
MAX_PAIRS_MIN, MAX_PAIRS_MAX = 1, 50

_WS = re.compile(r"\s+")


class AppValidationError(ValueError):
    """An input problem the user can fix; message is user-facing."""


@dataclass(frozen=True)
class AppInputs:
    """Validated user inputs, ready for the pipeline.

    ``claim`` is whitespace-normalized and non-empty. ``abstract`` is
    ``None`` when the user left it blank (the pipeline works without it).
    """

    claim: str
    abstract: str | None = None
    top_k: int = 5
    max_pairs: int = 20
    compare_papers: bool = False


def normalize_text(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return _WS.sub(" ", text).strip()


def validate_claim(text: object) -> str:
    """Return the normalized claim, or raise ``AppValidationError``.

    A claim must be present and within the length cap. Normalization
    collapses newlines/tabs so a pasted multi-line sentence becomes one
    line.
    """
    if not isinstance(text, str):
        raise AppValidationError("Claim must be text.")
    claim = normalize_text(text)
    if not claim:
        raise AppValidationError("Enter a scientific claim to analyze.")
    if len(claim) > MAX_CLAIM_CHARS:
        raise AppValidationError(
            f"Claim is too long ({len(claim)} characters; the limit is "
            f"{MAX_CLAIM_CHARS}). Paste the finding sentence only."
        )
    return claim


def validate_abstract(text: object) -> str | None:
    """Return the normalized abstract, or ``None`` when blank.

    The abstract is optional; leaving it blank is valid. When present it
    must be within the length cap.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        raise AppValidationError("Abstract must be text.")
    abstract = normalize_text(text)
    if not abstract:
        return None
    if len(abstract) > MAX_ABSTRACT_CHARS:
        raise AppValidationError(
            f"Abstract is too long ({len(abstract)} characters; the limit "
            f"is {MAX_ABSTRACT_CHARS}). Trim it to the abstract only."
        )
    return abstract


def _bounded_int(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AppValidationError(f"{name} must be a whole number.")
    if not low <= value <= high:
        raise AppValidationError(f"{name} must be between {low} and {high}.")
    return value


def validate_top_k(value: object) -> int:
    """Return ``top_k`` bounded to ``[TOP_K_MIN, TOP_K_MAX]``."""
    return _bounded_int(value, "Number of papers", TOP_K_MIN, TOP_K_MAX)


def validate_max_pairs(value: object) -> int:
    """Return ``max_pairs`` bounded to ``[MAX_PAIRS_MIN, MAX_PAIRS_MAX]``."""
    return _bounded_int(value, "Maximum comparisons", MAX_PAIRS_MIN, MAX_PAIRS_MAX)


def make_inputs(
    claim: object,
    abstract: object = None,
    top_k: object = 5,
    max_pairs: object = 20,
    compare_papers: object = False,
) -> AppInputs:
    """Validate every field at once, raising the first problem found.

    This is the single entry point the UI calls on submit; catching
    ``AppValidationError`` around it yields the user-facing message.
    """
    return AppInputs(
        claim=validate_claim(claim),
        abstract=validate_abstract(abstract),
        top_k=validate_top_k(top_k),
        max_pairs=validate_max_pairs(max_pairs),
        compare_papers=bool(compare_papers),
    )
