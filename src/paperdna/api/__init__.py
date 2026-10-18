"""PaperDNA web API: a thin FastAPI layer over the research pipeline.

Two strictly separated layers:

- **Pure core** (:mod:`paperdna.api.pipeline`,
  :mod:`paperdna.api.validation`): deterministic Python that imports only
  the ``paperdna`` research package — never ``fastapi``, never the
  optional ML stack. Fully testable offline.
- **HTTP layer** (:mod:`paperdna.api.app`, :mod:`paperdna.api.schemas`,
  :mod:`paperdna.api.graph_payload`): FastAPI app factory, typed
  request/response models, and the ``KnowledgeGraph`` -> React Flow
  projection.

Honesty rules baked into this package: ranking kind (semantic vs
lexical fallback) is always explicit; NLI labels are model *predictions*
with their uncertainty flags and the calibration caveat, never asserted
facts; evidence text, offsets, verification status, and missing-evidence
reasons are copied verbatim from the evidence module; nothing is
fabricated. Models are constructed only behind an explicit user
confirmation, never at import or app start.
"""
