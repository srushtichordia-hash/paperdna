"""Optional ML model lifecycle: availability, opt-in load, unload.

This module is the ONLY place a model may be constructed, and it does so
only when the user has explicitly confirmed the download via
``load_models(confirm_download=True)``. Nothing is loaded at import time
or app start. When models are unavailable (not installed, or the user
declines), the pipeline runs its lexical/label-free baseline — the
frontend is told via capabilities, never misled.

Model download sizes are exposed so the UI can show an informed opt-in
notice. Load failures degrade to "unavailable" with a captured message
rather than raising, so the app stays usable.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Rough Hugging Face download sizes (MB), for the opt-in notice.
EMBEDDING_DOWNLOAD_MB = 90
NLI_DOWNLOAD_MB = 700


def _sentence_transformers_installed() -> bool:
    """True when the optional ``embed``/``nli`` extra is present."""
    return importlib.util.find_spec("sentence_transformers") is not None


@dataclass
class ModelRegistry:
    """Holds the (possibly absent) embedder and NLI classifier.

    ``embedder``/``classifier`` are ``None`` until :meth:`load_models`
    succeeds. Tests inject fakes by setting these attributes directly.
    """

    embedder: object | None = None
    classifier: object | None = None
    embedder_error: str | None = None
    classifier_error: str | None = None

    def status(self) -> dict[str, dict[str, object]]:
        """Availability + load state for each model (JSON-safe)."""
        installed = _sentence_transformers_installed()
        return {
            "embedding": {
                "installed": installed,
                "loaded": self.embedder is not None,
                "name": getattr(self.embedder, "name", None),
                "downloadMB": EMBEDDING_DOWNLOAD_MB if installed else None,
            },
            "nli": {
                "installed": installed,
                "loaded": self.classifier is not None,
                "name": getattr(self.classifier, "name", None),
                "downloadMB": NLI_DOWNLOAD_MB if installed else None,
            },
        }

    def load_models(self, confirm_download: bool) -> dict[str, dict[str, object]]:
        """Construct the real models after explicit user confirmation.

        Returns the post-load status. Refuses (leaving models unloaded)
        when ``confirm_download`` is False or the optional dependency is
        missing. A construction failure is captured, logged, and reported
        via the returned status rather than raised.
        """
        if not confirm_download:
            logger.info("Model load skipped: user has not confirmed download.")
            return self.status()

        if not _sentence_transformers_installed():
            self.embedder_error = (
                "sentence-transformers is not installed; run "
                '`pip install -e ".[embed,nli]"` to enable semantic models.'
            )
            self.classifier_error = self.embedder_error
            return self.status()

        if self.embedder is None and self.embedder_error is None:
            try:
                from paperdna.retrieval.embedder import SentenceTransformerEmbedder

                self.embedder = SentenceTransformerEmbedder()
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the app
                self.embedder_error = str(exc)
                logger.warning("Embedding model load failed: %s", exc)

        if self.classifier is None and self.classifier_error is None:
            try:
                from paperdna.nli.classifier import CrossEncoderNliClassifier

                self.classifier = CrossEncoderNliClassifier()
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the app
                self.classifier_error = str(exc)
                logger.warning("NLI model load failed: %s", exc)

        return self.status()

    def unload_models(self) -> dict[str, dict[str, object]]:
        """Drop loaded models and reset error state."""
        self.embedder = None
        self.classifier = None
        self.embedder_error = None
        self.classifier_error = None
        return self.status()
