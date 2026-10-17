"""Environment-based configuration with safe defaults.

All configuration is read from ``PAPERDNA_*`` environment variables. No
secrets are stored in code, and no values are logged. Defaults are chosen so
the package works out of the box without any environment setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_PREFIX = "PAPERDNA_"

_DEFAULT_LOG_LEVEL = "INFO"
_VALID_LOG_LEVELS = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    log_level: str = _DEFAULT_LOG_LEVEL


def _read_log_level() -> str:
    """Read the log level from the environment, falling back to a safe default."""
    raw = os.environ.get(f"{ENV_PREFIX}LOG_LEVEL", _DEFAULT_LOG_LEVEL)
    level = raw.strip().upper()
    if level not in _VALID_LOG_LEVELS:
        # Never raise on misconfiguration; degrade to the default instead.
        return _DEFAULT_LOG_LEVEL
    return level


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(log_level=_read_log_level())
