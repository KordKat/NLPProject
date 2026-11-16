"""Helpers shared by tools: source bookkeeping and artifact registration."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.database.session import db_session
from app.services.artifact_storage import artifact_to_dict, register_artifact
from app.tools.base import ToolContext


def add_source(ctx: ToolContext, src: dict[str, Any]) -> dict[str, Any]:
    key = src["source_key"]
    if key in ctx.sources:
        prev = ctx.sources[key]
        prev["relevance"] = max(prev.get("relevance", 0) or 0, src.get("relevance", 0) or 0)
        return prev
    src.setdefault("ref", len(ctx.sources) + 1)
    ctx.sources[key] = src
    return src


def arxiv_source(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": f"arxiv:{paper['paper_id']}",
        "source_type": "arxiv",
        "paper_id": paper["paper_id"],
        "title": paper.get("title", ""),
        "url": paper.get("url", ""),
        "relevance": paper.get("score", 0.0),
        "meta": {
            "authors": paper.get("authors", []),
            "published": paper.get("published", ""),
            "abstract": (paper.get("abstract") or "")[:800],
            "byline": paper.get("byline", ""),
        },
    }


def web_source(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": f"web:{r['url']}",
        "source_type": "web",
        "title": r.get("title", ""),
        "url": r["url"],
        "relevance": r.get("relevance", 0.0),
        "meta": {"snippet": r.get("snippet", ""), "fetched": r.get("fetched", False), "site": r.get("site", "")},
    }


def _register_sync(project_id: str, rel_path: str, artifact_type: Optional[str], meta: Optional[dict], status: str, error: str) -> dict[str, Any]:
    with db_session() as db:
        a = register_artifact(db, project_id, rel_path, artifact_type=artifact_type, meta=meta, status=status, error=error)
        return artifact_to_dict(a)


async def register_file_artifact(
    ctx: ToolContext,
    rel_path: str,
    *,
    artifact_type: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
    status: str = "ready",
    error: str = "",
    announce: bool = True,
) -> dict[str, Any]:
    meta = {**(meta or {}), "run_id": ctx.run_id}
    art = await asyncio.to_thread(_register_sync, ctx.project_id, rel_path, artifact_type, meta, status, error)
    if not any(a["id"] == art["id"] for a in ctx.artifacts):
        ctx.artifacts.append(art)
    else:
        ctx.artifacts = [art if a["id"] == art["id"] else a for a in ctx.artifacts]
    if announce:
        await ctx.emit({"type": "artifact", "artifact": art})
        if art["type"] == "image":
            await ctx.emit({"type": "image", "url": art["url"], "artifact": art})
    return art
