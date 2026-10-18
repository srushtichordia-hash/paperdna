"""FastAPI application factory for the PaperDNA web API.

Thin HTTP layer: validates requests with Pydantic, calls the pure
pipeline, and returns typed JSON. No research logic lives here — it all
comes from :mod:`paperdna.api.pipeline` and the upstream ``paperdna``
modules. Models are only constructed when a request explicitly confirms
the download (``POST /api/models/load``); app import and startup never
touch them.

Endpoints
---------
GET  /api/health        -> liveness + corpus stats + model status
GET  /api/corpus        -> corpus stats + paper summaries
POST /api/analyze       -> full discovery result for one claim
GET  /api/models        -> model availability/load status
POST /api/models/load   -> construct models (requires confirmDownload)
POST /api/models/unload -> drop loaded models
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from paperdna.api.models import ModelRegistry
from paperdna.api.pipeline import DEMO_CORPUS_PATH, load_corpus, run_discovery
from paperdna.api.schemas import (
    AnalyzeRequest,
    CorpusResponse,
    DiscoveryResponse,
    HealthResponse,
    LoadModelsRequest,
    ModelsStatusOut,
    PaperOut,
)
from paperdna.api.serializers import (
    _paper_out,
    corpus_stats_dict,
    discovery_to_response,
)
from paperdna.api.validation import AppValidationError, make_inputs

logger = logging.getLogger(__name__)

# Populated per-app so tests can inject fakes and the running app holds
# one shared registry.
_registry = ModelRegistry()


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a ``ModelRegistry`` to override models
    (used by tests to inject fakes)."""
    global _registry
    if registry is not None:
        _registry = registry

    app = FastAPI(title="PaperDNA API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        papers = load_corpus()
        stats = corpus_stats_dict(
            {"loaded": len(papers), "unique": len(papers), "duplicates": 0, "rejected": 0}
        )
        return HealthResponse(
            status="ok",
            corpus=stats,
            models=ModelsStatusOut(**_registry.status()),  # type: ignore[arg-type]
        )

    @app.get("/api/corpus", response_model=CorpusResponse)
    def corpus() -> CorpusResponse:
        papers = load_corpus()
        stats = corpus_stats_dict(
            {"loaded": len(papers), "unique": len(papers), "duplicates": 0, "rejected": 0}
        )
        return CorpusResponse(
            stats=stats,
            papers=[_paper_out(record) for record in papers],
        )

    @app.get("/api/models", response_model=ModelsStatusOut)
    def models_status() -> ModelsStatusOut:
        return ModelsStatusOut(**_registry.status())  # type: ignore[arg-type]

    @app.post("/api/models/load", response_model=ModelsStatusOut)
    def models_load(body: LoadModelsRequest) -> ModelsStatusOut:
        if not body.confirmDownload:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Model download must be confirmed explicitly. Set "
                    "confirmDownload to true after the user acknowledges the "
                    "download notice."
                ),
            )
        return ModelsStatusOut(**_registry.load_models(True))  # type: ignore[arg-type]

    @app.post("/api/models/unload", response_model=ModelsStatusOut)
    def models_unload() -> ModelsStatusOut:
        return ModelsStatusOut(**_registry.unload_models())  # type: ignore[arg-type]

    @app.post("/api/analyze", response_model=DiscoveryResponse)
    def analyze(body: AnalyzeRequest) -> DiscoveryResponse:
        try:
            inputs = make_inputs(
                claim=body.claim,
                abstract=body.abstract,
                top_k=body.topK,
                max_pairs=body.maxPairs,
                compare_papers=body.comparePapers,
            )
        except AppValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            result = run_discovery(
                inputs,
                corpus_path=DEMO_CORPUS_PATH,
                embedder=_registry.embedder,
                classifier=_registry.classifier,
            )
        except AppValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface as a 500 with context
            logger.exception("Discovery pipeline failed")
            raise HTTPException(
                status_code=500, detail=f"Discovery failed: {type(exc).__name__}: {exc}"
            ) from exc

        return discovery_to_response(result)

    return app


# Module-level app for `uvicorn paperdna.api.app:app`.
app = create_app()
