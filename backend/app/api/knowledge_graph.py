from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.database.models import Project
from app.deps import get_owned_project
from app.knowledge_graph import graph_retriever
from app.knowledge_graph.neo4j_client import GraphUnavailableError, get_graph

router = APIRouter(prefix="/api/projects/{project_id}/graph", tags=["knowledge-graph"])


def _require_graph():
    g = get_graph()
    if not g.available:
        raise HTTPException(status_code=503, detail=f"Knowledge graph unavailable: {g.error}")


@router.get("")
async def project_graph(limit: int = Query(300, ge=1, le=1000), project: Project = Depends(get_owned_project)):
    _require_graph()
    try:
        return await asyncio.to_thread(graph_retriever.project_graph, project.id, limit)
    except GraphUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/search")
async def search_graph(q: str = Query(..., min_length=2, max_length=200), project: Project = Depends(get_owned_project)):
    _require_graph()
    try:
        return await asyncio.to_thread(graph_retriever.search, q, project.id, 12, 1)
    except GraphUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/node")
async def node(uid: str = Query(..., max_length=400), project: Project = Depends(get_owned_project)):
    _require_graph()
    try:
        d = await asyncio.to_thread(graph_retriever.node_detail, uid, project.id)
    except GraphUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if d is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return d
