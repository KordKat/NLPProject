"""STEM OS FastAPI application entrypoint.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000   (from the backend/ directory)
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.model_engine import get_engine
from app.api import artifacts, assistant, auth, knowledge_graph, preferences, projects, skills
from app.config import get_settings
from app.database.session import init_db
from app.knowledge_graph.neo4j_client import get_graph
from app.retrieval.faiss_search import get_arxiv_index
from app.services.pdf_converter import fail_stale_conversions
from app.tools.registry import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stemos")
settings = get_settings()


def _warm_imports() -> None:
    import faiss  # noqa: F401
    import torch  # noqa: F401
    from transformers import (  # noqa: F401
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.secret_key == "change-me-in-production" and settings.environment == "production":
        raise RuntimeError("SECRET_KEY must be set in production")
    init_db()
    fail_stale_conversions()
    registry.load()
    # Heavy services load in the background so auth/projects are usable immediately; their status
    # ("loading" → "ready"/"error") is reported by /api/health and the UI. Failures are not fatal.
    # transformers uses lazy imports that are not thread-safe: resolve them once before loader threads start
    await asyncio.to_thread(_warm_imports)
    await asyncio.to_thread(get_graph().connect)
    get_engine().status = "loading" if settings.llm_load_on_startup else "not_loaded"
    loaders = [asyncio.to_thread(get_arxiv_index().load)]
    if settings.llm_load_on_startup:
        loaders.append(asyncio.to_thread(get_engine().load))

    async def _load_all() -> None:
        await asyncio.gather(*loaders, return_exceptions=True)
        log.info("STEM OS ready — model=%s faiss=%s neo4j=%s", get_engine().status, get_arxiv_index().status, get_graph().status)

    app.state.loader = asyncio.create_task(_load_all())
    yield
    get_graph().close()


app = FastAPI(title="STEM OS API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth.router, preferences.router, projects.router, assistant.router, assistant.tools_router, artifacts.router, knowledge_graph.router, skills.router):
    app.include_router(r)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/api/health")
def health():
    e = get_engine()
    g = get_graph()
    return {
        "status": "ok",
        "model": {"name": e.model_name, "status": e.status, "error": e.error, "device": e.device, "quantization": e.quantization},
        "faiss": get_arxiv_index().info(),
        "neo4j": {"status": "ready" if g._driver is not None else g.status, "error": g.error},
        "sandbox": settings.sandbox_mode,
        "tools": registry.names(),
    }
