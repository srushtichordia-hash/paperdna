"""Exceptions for the retrieval subsystem."""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for retrieval errors."""


class EmbedderUnavailableError(RetrievalError):
    """The embedding backend is not installed or cannot be loaded."""


class InvalidVectorError(RetrievalError):
    """An embedding vector is malformed: wrong dimension, non-finite
    values, or zero norm."""
