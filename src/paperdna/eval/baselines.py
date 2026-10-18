"""Deterministic evaluation baselines for relation classification.

Two clearly separated systems:

- ``OverlapNliClassifier``: a transparent token-overlap heuristic whose
  predictions are INDEPENDENT of the gold labels. It exists so the
  reported macro-F1 / per-class metrics are measured from genuine
  predictions, even fully offline. It is a floor baseline, not a
  scientific-quality model.

- ``ScriptedNliClassifier``: replays the fixture's ``scripted_scores``.
  Those scores were authored from the gold labels, so agreement is
  circular BY CONSTRUCTION. It validates pipeline plumbing only and
  must never appear in a model-performance table.

The real-model tier (``CrossEncoderNliClassifier``) is optional,
explicitly installed, and never loaded from here implicitly.
"""

from __future__ import annotations

import re

from paperdna.nli.schema import RawPairScores

_TOKENS = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the "
    "to was were we with which who".split()
)
_NEGATIONS = frozenset("not no never cannot without fails failed".split())


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKENS.findall(text.casefold())
        if token not in _STOPWORDS and len(token) > 1
    }


def _has_negation(tokens: set[str]) -> bool:
    return bool(tokens & _NEGATIONS)


class OverlapNliClassifier:
    """Jaccard-overlap baseline with a negation-parity rule.

    Scoring is fully deterministic and text-only:

    - ``entailment`` = Jaccard overlap of the two sentences' content
      tokens (|A ∩ B| / |A ∪ B|), unless negation parity differs;
    - when exactly one side carries a negation marker, ``contradiction``
      is raised to ``max(0.5, overlap)`` and entailment shrinks to
      ``max(0.0, 0.4 - overlap)``;
    - ``neutral`` takes the remaining mass.

    This encodes almost nothing about language; its measured metrics
    are a LOWER BOUND that a real NLI model should beat.
    """

    name = "lexical-overlap-baseline"
    max_length = None

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[RawPairScores]:
        return [RawPairScores(scores=self._score(a, b)) for a, b in pairs]

    @staticmethod
    def _score(premise: str, hypothesis: str) -> dict[str, float]:
        a = _content_tokens(premise)
        b = _content_tokens(hypothesis)
        union = a | b
        overlap = (len(a & b) / len(union)) if union else 0.0
        if _has_negation(a) != _has_negation(b):
            contradiction = max(0.5, overlap)
            entailment = max(0.0, 0.4 - overlap)
        else:
            entailment = overlap
            contradiction = 0.0
        neutral = max(0.0, 1.0 - entailment - contradiction)
        total = entailment + contradiction + neutral
        return {
            "entailment": entailment / total,
            "contradiction": contradiction / total,
            "neutral": neutral / total,
        }


class ScriptedNliClassifier:
    """Replays fixture-authored scores (circular; plumbing validation).

    Pairs missing from the script get uniform scores, which classify as
    ``neutral`` with ``uncertain=True`` — an obvious, deterministic
    fallback that would stand out in a conformance report.
    """

    name = "scripted-fixture"
    max_length = None

    def __init__(self, script: dict[tuple[str, str], dict[str, float]]) -> None:
        self.script = script

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[RawPairScores]:
        uniform = {"entailment": 1 / 3, "contradiction": 1 / 3, "neutral": 1 / 3}
        return [
            RawPairScores(scores=self.script.get(pair, dict(uniform)))
            for pair in pairs
        ]
