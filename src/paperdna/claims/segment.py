"""Lightweight, deterministic sentence segmentation with char offsets.

Pure stdlib, offline, and non-raising on any string input. Splits on
terminal punctuation (``.``, ``!``, ``?``) and blank-line paragraph
breaks, with guards for common scientific abbreviations, decimal
numbers, single-letter initials, and closing quotes/parentheses.

Known limitations (documented in the README): unusual abbreviations,
non-standard capitalization after a boundary, and unbalanced punctuation
can cause over- or under-splitting. This is a baseline, not a
linguistic parser.
"""

from __future__ import annotations

import re

from paperdna.claims.schema import SentenceSpan

# Common scientific abbreviations that end with a period but do not end
# a sentence. Multi-word forms are handled by their final token ("al.").
_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "al",
        "fig",
        "eq",
        "eqs",
        "vs",
        "cf",
        "ca",
        "approx",
        "no",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "etc",
        "inc",
        "ltd",
        "ref",
        "refs",
        "sec",
        "ch",
        "vol",
    }
)

_CLOSERS = "\"'”’)]}"
_OPENERS = "([{\"'“‘"

# A punctuation run followed (optionally by closing chars) by whitespace
# or end of text.
_BOUNDARY_RE = re.compile(r"[.!?]+(?=[\"'”’)\]}]*(?:\s|$))")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z]+)$")
_PARAGRAPH_RE = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")


def _is_abbreviation_period(text: str, match_end: int) -> bool:
    """True if the period run ending at ``match_end`` should not split.

    Single-letter initials ("A. Newton") never split. Multi-word
    abbreviations ("et al.") split when the next sentence clearly starts
    with an uppercase letter, and merge otherwise ("Fig. 2", "e.g., ...").
    """
    before = text[:match_end].rstrip(".")
    token_match = _WORD_BEFORE_RE.search(before)
    if not token_match:
        return False
    token = token_match.group(1)
    if len(token) == 1:
        # Single-letter initial ("A.") — treat as abbreviation.
        return True
    if token.lower() not in _ABBREVIATIONS:
        return False
    # Abbreviation followed by a sentence-like start ("et al. Prior ...")
    # is a boundary; followed by lowercase/digits/punctuation it is not.
    following = _next_non_space(text, match_end)
    return not (following.isupper() or following in _OPENERS)


def _next_non_space(text: str, position: int) -> str:
    for char in text[position:]:
        if not char.isspace():
            return char
    return ""


def _split_paragraph(paragraph: str, base: int) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(paragraph):
        # Abbreviation check applies only to period runs.
        if "." in match.group() and _is_abbreviation_period(paragraph, match.end()):
            continue

        # Extend the boundary over closing quotes/parentheses.
        end = match.end()
        while end < len(paragraph) and paragraph[end] in _CLOSERS:
            end += 1

        # The next sentence must start with an uppercase letter, digit, or
        # an opener followed by one; lowercase means a mid-sentence period.
        following = _next_non_space(paragraph, end)
        if following and not (
            following.isupper() or following.isdigit() or following in _OPENERS
        ):
            continue

        spans.append((start, end))
        start = end

    spans.append((start, len(paragraph)))

    result: list[SentenceSpan] = []
    for raw_start, raw_end in spans:
        chunk = paragraph[raw_start:raw_end]
        stripped = chunk.strip()
        if not stripped:
            continue
        offset = raw_start + (len(chunk) - len(chunk.lstrip()))
        result.append(
            SentenceSpan(
                sentence=stripped,
                start=base + offset,
                end=base + offset + len(stripped),
            )
        )
    return result


def segment_sentences(text: str) -> tuple[SentenceSpan, ...]:
    """Split ``text`` into sentences with global character offsets.

    Empty or whitespace-only input yields an empty tuple. Offsets refer to
    the original ``text``, so ``text[span.start:span.end] == span.sentence``
    for every returned span.
    """
    if not isinstance(text, str) or not text.strip():
        return ()

    spans: list[SentenceSpan] = []
    for paragraph in _PARAGRAPH_RE.finditer(text):
        spans.extend(_split_paragraph(paragraph.group(), paragraph.start()))
    return tuple(spans)
