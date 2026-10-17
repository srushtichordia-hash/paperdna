"""Label-order resolution for NLI model outputs.

Different NLI models emit their three logits in different orders (e.g.
the recommended model uses ``["contradiction", "entailment", "neutral"]``
while many SciFact-tuned models use ``["entailment", "neutral",
"contradiction"]``). Numeric label IDs are therefore never trusted on
their own: this module resolves the model's declared ``id2label`` config
into canonical relationship names, with documented synonym handling
(SciFact-style ``SUPPORTS``/``REFUTES``/``NOT_ENOUGH_INFO``). If a config
declares labels that cannot be mapped, resolution *raises* rather than
guessing. Only when a config declares no labels at all do we fall back to
the recommended model's documented default order.
"""

from __future__ import annotations

from collections.abc import Mapping

from paperdna.nli.errors import NliError

# Documented output order of the default model
# (cross-encoder/nli-deberta-v3-base, per its model card). Used ONLY when
# the model config declares no labels.
DEFAULT_LABEL_ORDER: tuple[str, str, str] = ("contradiction", "entailment", "neutral")

_CANONICAL_BY_NAME = {
    "entailment": "entailment",
    "entailed": "entailment",
    "support": "entailment",
    "supports": "entailment",
    "supported": "entailment",
    "contradiction": "contradiction",
    "contradictory": "contradiction",
    "refute": "contradiction",
    "refutes": "contradiction",
    "refuted": "contradiction",
    "neutral": "neutral",
    "not enough info": "neutral",
    "not_enough_info": "neutral",
    "nei": "neutral",
    "unknown": "neutral",
}


def canonical_label(raw: object) -> str | None:
    """Map a raw model label to a canonical relationship name, or None."""
    if not isinstance(raw, str):
        return None
    return _CANONICAL_BY_NAME.get(raw.strip().lower())


def resolve_label_map(
    id2label: Mapping[object, object] | None,
) -> tuple[str, str, str]:
    """Resolve a model's ``id2label`` config into canonical label order.

    Returns a 3-tuple of canonical labels indexed by the model's numeric
    output index. Raises ``NliError`` if the config declares labels but
    they cannot be mapped one-to-one onto the three relationships. An
    empty/missing ``id2label`` falls back to ``DEFAULT_LABEL_ORDER``.
    """
    if not id2label:
        return DEFAULT_LABEL_ORDER

    resolved: dict[int, str] = {}
    for key, raw_label in dict(id2label).items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise NliError(f"label index {key!r} is not an integer") from exc
        canonical = canonical_label(raw_label)
        if canonical is None:
            raise NliError(
                f"cannot map model label {raw_label!r} to a relationship "
                "(expected entailment/contradiction/neutral or a documented "
                "synonym such as SUPPORTS/REFUTES/NOT_ENOUGH_INFO)"
            )
        if canonical in resolved.values():
            raise NliError(f"duplicate canonical label {canonical!r} in model config")
        resolved[index] = canonical

    expected = set(range(3))
    if set(resolved) != expected:
        raise NliError(
            f"expected labels for indices {sorted(expected)}, "
            f"got {sorted(resolved)}"
        )
    return (resolved[0], resolved[1], resolved[2])
