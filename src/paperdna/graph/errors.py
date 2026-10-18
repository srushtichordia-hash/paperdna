"""Error types for the research evidence knowledge graph."""

from __future__ import annotations


class GraphError(Exception):
    """Base class for knowledge-graph errors."""


class GraphValidationError(GraphError):
    """The graph is structurally invalid.

    The message lists every problem found (unknown edge endpoints, wrong
    endpoint node types, duplicate ids, invalid relation labels or
    scores), never just the first one.
    """


class GraphConflictError(GraphError):
    """A node or edge id was re-added with a conflicting payload.

    Re-adding an *identical* payload under the same id is an idempotent
    no-op (counted in ``GraphBuilder.duplicates_ignored``); re-adding a
    *different* payload under an existing id is rejected loudly rather
    than silently keeping one of the two.
    """
