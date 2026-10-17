"""Rule dictionaries for baseline claim classification.

Each rule is a (name, compiled-regex) pair matched case-insensitively
against a sentence. These are lexical heuristics for a reproducible
baseline — they do not understand scientific content, and paraphrases
without these cues will be missed.
"""

from __future__ import annotations

import re

Finding = tuple[str, re.Pattern[str]]


def _rules(*patterns: str) -> tuple[Finding, ...]:
    return tuple(
        (pattern, re.compile(pattern, re.IGNORECASE)) for pattern in patterns
    )


# Sentences matching any finding cue are candidate claims (highest priority).
FINDING_CUES: tuple[Finding, ...] = _rules(
    r"\bwe (find|found|show|shows|demonstrate|demonstrates|observe|observes|"
    r"confirm|confirms|report|reports)\b",
    r"\bour (results?|analysis|findings?|experiments?|study) "
    r"(show|shows|indicate|indicates|suggest|suggests|reveal|reveals|"
    r"demonstrate|demonstrates|confirm|confirms)\b",
    r"\bresults? (show|shows|indicate|indicates|suggest|suggests|reveal|reveals)\b",
    r"\b(?:is|are) ((?:\w+ ){0,2}associated with)\b",
    r"\bwe (conclude|present|introduce)\b",
)

# Sentences matching these (and no finding cue) are background.
BACKGROUND_CUES: tuple[Finding, ...] = _rules(
    r"\bprevious(ly)? (studies|study|work|research|approaches|methods)\b",
    r"\bit is (?:well )?known\b",
    r"\bhas been proposed\b",
    r"\bprior (work|studies|research)\b",
    r"\brecent(ly)?\b",
)

# Sentences matching these (and no finding cue) are methods/procedural.
METHODS_CUES: tuple[Finding, ...] = _rules(
    r"\bwe (trained|used|conducted|collected|evaluated|implemented)\b",
    r"\bdatasets?\b",
    r"\bwe then\b",
    r"\bfirst,",
    r"\bmethods?\b",
    r"\bimplementation\b",
)

# Uncertainty markers, recorded on claims without altering the text.
HEDGE_CUES: tuple[Finding, ...] = _rules(
    r"\b(may|might|could|possibly|potentially)\b",
    r"\bsuggests?\b",
    r"\b(?:appears?|seems?)\b",
    r"\b(?:is|are) ((?:\w+ ){0,2}associated with)\b",
    r"\bconsistent with\b",
    r"\bwe hypothesize\b",
)


def first_match(rules: tuple[Finding, ...], sentence: str) -> str | None:
    """Return the verbatim text matched by the first matching rule, or None."""
    for _name, pattern in rules:
        match = pattern.search(sentence)
        if match:
            return match.group(0)
    return None


def find_hedges(sentence: str) -> tuple[str, ...]:
    """Return hedge markers in order of appearance, deduplicated."""
    found: list[str] = []
    positions: list[tuple[int, str]] = []
    for _name, pattern in HEDGE_CUES:
        for match in pattern.finditer(sentence):
            positions.append((match.start(), match.group(0).lower()))
    positions.sort()
    for _pos, text in positions:
        if text not in found:
            found.append(text)
    return tuple(found)
