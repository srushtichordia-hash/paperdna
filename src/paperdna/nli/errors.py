"""Exceptions for the NLI subsystem."""

from __future__ import annotations


class NliError(Exception):
    """Base class for NLI errors."""


class NliUnavailableError(NliError):
    """The NLI backend is not installed or cannot be loaded."""


class InvalidScoresError(NliError):
    """A classifier returned malformed scores (wrong labels, non-finite
    values, or probabilities that do not sum to ~1)."""
