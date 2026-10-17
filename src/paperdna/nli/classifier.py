"""NLI classifier interface and the optional real-model backend.

The core package never imports torch or sentence-transformers: the real
backend loads the model only when explicitly instantiated, with the
import inside ``__init__`` (same pattern as the retrieval embedder).

Default model: ``cross-encoder/nli-deberta-v3-base`` (Apache-2.0;
184M parameters; 512-token inputs; trained on SNLI + MultiNLI, i.e.
general-domain text — scientific claim pairs are out-of-distribution).
"""

from __future__ import annotations

import logging
import math
from typing import Protocol, runtime_checkable

from paperdna.nli.errors import InvalidScoresError, NliUnavailableError
from paperdna.nli.labels import resolve_label_map
from paperdna.nli.schema import RawPairScores

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


@runtime_checkable
class NliClassifier(Protocol):
    """Anything that scores (premise, hypothesis) text pairs."""

    name: str
    max_length: int | None

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[RawPairScores]:
        """Return one RawPairScores per input pair, in order."""
        ...


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


class CrossEncoderNliClassifier:
    """NLI backend using a sentence-transformers CrossEncoder.

    Installing the ``nli`` (or ``embed``) extra is required:
    ``pip install -e ".[nli]"``. The model is downloaded on first use
    (several hundred MB, cached by Hugging Face) — never at module import
    time.

    Label order is resolved from the model's own ``id2label`` config via
    :mod:`paperdna.nli.labels`; the model-card default order is used only
    when the config declares no labels. Logits are softmaxed locally so
    scores are always probabilities summing to ~1.
    """

    def __init__(
        self, model_name: str | None = None, max_length: int | None = None
    ) -> None:
        self.name = model_name or DEFAULT_NLI_MODEL
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise NliUnavailableError(
                "sentence-transformers is not installed. Install the optional "
                'extra with: pip install -e ".[nli]"'
            ) from exc
        logger.info("Loading NLI model %r (downloads on first use)", self.name)
        self._model = CrossEncoder(self.name)
        self.max_length: int | None = int(
            max_length or getattr(self._model, "max_seq_length", None) or 512
        )
        config = getattr(self._model, "config", None)
        self._label_order = resolve_label_map(getattr(config, "id2label", None))

    def _pair_truncated(self, pair: tuple[str, str]) -> bool:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None or self.max_length is None:
            return False
        untruncated = tokenizer(
            pair[0], pair[1], truncation=False, add_special_tokens=True
        )["input_ids"]
        return len(untruncated) > self.max_length

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[RawPairScores]:
        results: list[RawPairScores] = []
        logits_rows = self._model.predict(
            pairs, max_length=self.max_length, show_progress_bar=False
        )
        for pair, logits in zip(pairs, logits_rows):
            values = [float(value) for value in logits]
            if len(values) != 3:
                raise InvalidScoresError(
                    f"model returned {len(values)} logits, expected 3"
                )
            probabilities = _softmax(values)
            scores = {
                label: probabilities[index]
                for index, label in enumerate(self._label_order)
            }
            results.append(
                RawPairScores(scores=scores, truncated=self._pair_truncated(pair))
            )
        return results
