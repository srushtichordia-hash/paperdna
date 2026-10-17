"""Basic logging setup for PaperDNA."""

from __future__ import annotations

import logging

from paperdna.config import load_settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure and return the root PaperDNA logger.

    Args:
        level: A logging level name (e.g. ``"DEBUG"``). If omitted, the level
            is read from the ``PAPERDNA_LOG_LEVEL`` environment variable, and
            defaults to ``INFO``.

    Returns:
        The configured ``"paperdna"`` logger.
    """
    if level is None:
        level = load_settings().log_level

    logger = logging.getLogger("paperdna")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
