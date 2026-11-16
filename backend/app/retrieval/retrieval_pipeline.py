"""Candidate assembly for one retrieval round: runs arXiv / graph / web queries in parallel through the
tool registry (so every call is visible as a tool event), then expands vector hits through the graph."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from app.knowledge_graph import graph_retriever
from app.knowledge_graph.neo4j_client import GraphUnavailableError, get_graph
from app.retrieval.faiss_search import get_arxiv_index
from app.tools.base import ToolContext, ToolResult
from app.tools.common import add_source, arxiv_source
from app.tools.registry import registry

TOOL_FOR = {"arxiv": "search_arxiv", "graph": "search_knowledge_graph", "web": "search_web"}

Executor = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


async def run_round(ctx: ToolContext, queries: list[dict[str, str]], question: str, execute: Executor, seen_keys: set[str]) -> dict[str, Any]:
    """Returns {"candidates": [...], "graph": {...}, "notes": [...]}."""
    calls = []
    for q in queries:
        tool = TOOL_FOR.get(q.get("tool", "arxiv"))
        if not tool:
            continue
        args: dict[str, Any] = {"query": q["query"]}
        if tool == "search_arxiv":
            args["top_k"] = 5
        if tool == "search_web":
            args["limit"] = 4
        calls.append((q, tool, args))
    results = await asyncio.gather(*[execute(tool, args) for _, tool, args in calls])

    candidates: list[dict[str, Any]] = []
    notes: list[str] = []
    graph_nodes: dict[str, dict] = {}
    graph_edges: list[dict] = []
    arxiv_ids: list[str] = []

    for (q, tool, _), res in zip(calls, results):
        if not res.ok:
            notes.append(f"{tool}('{q['query']}') failed: {res.error}")
            continue
        data = res.data or {}
        if tool == "search_arxiv":
            for p in data.get("papers", []):
                arxiv_ids.append(p["paper_id"])
                key = f"arxiv:{p['paper_id']}"
                if key in seen_keys:
                    continue
                candidates.append({**p, "kind": "arxiv", "source_key": key, "ref": p.get("ref")})
        elif tool == "search_web":
            for r in data.get("results", []):
                key = f"web:{r['url']}"
                if key in seen_keys:
                    continue
                candidates.append({
                    "kind": "web_fetched" if r.get("fetched") else "web_snippet",
                    "source_key": key, "ref": r.get("ref"), "title": r["title"], "url": r["url"],
                    "abstract": r.get("snippet", ""), "chunks": [{"text": r.get("content", "")}] if r.get("content") else [],
                    "score": r.get("relevance", 0.5) * 0.6, "published": "",
                })
        elif tool == "search_knowledge_graph":
            for n in data.get("nodes", []):
                graph_nodes[n["uid"]] = n
            graph_edges += data.get("edges", [])

    # graph-connected papers (from graph search + expansion of vector hits) → hybrid candidates
    index = get_arxiv_index()
    if arxiv_ids and get_graph().available:
        try:
            exp = await asyncio.to_thread(graph_retriever.expand_from_papers, list(dict.fromkeys(arxiv_ids)), ctx.project_id)
            for n in exp["nodes"]:
                graph_nodes.setdefault(n["uid"], n)
            graph_edges += exp["edges"]
            for rp in exp["related_papers"]:
                graph_nodes.setdefault(rp["uid"], rp)
                rp["_shared"] = rp.get("shared_count", 0)
        except GraphUnavailableError as exc:
            notes.append(f"graph expansion unavailable: {exc}")
            exp = {"related_papers": []}
    else:
        exp = {"related_papers": []}

    graph_papers = [n for n in graph_nodes.values() if n.get("type") == "Paper" and n.get("paper_id")]
    graph_papers += [p for p in exp.get("related_papers", []) if p.get("paper_id")]
    for gp in graph_papers[:6]:
        key = f"arxiv:{gp['paper_id']}"
        if key in seen_keys or any(c["source_key"] == key for c in candidates):
            continue
        if not index.available:
            continue
        paper = await asyncio.to_thread(index.get_paper, gp["paper_id"])
        if not paper:
            continue
        chunks = await asyncio.to_thread(index.chunks_for_paper, gp["paper_id"], question, 2)
        if chunks:
            paper["chunks"] = chunks
            paper["score"] = max(c["score"] for c in chunks)
        src = add_source(ctx, {**arxiv_source(paper), "meta": {**arxiv_source(paper)["meta"], "via": "knowledge_graph"}})
        candidates.append({**paper, "kind": "graph", "via_graph": True, "shared_count": gp.get("shared_count", 1), "source_key": key, "ref": src["ref"], "shared_concepts": gp.get("shared_concepts", [])})

    # dedupe edges
    uniq = {}
    for e in graph_edges:
        uniq[(e["source"], e["type"], e["target"])] = e
    return {"candidates": candidates, "graph": {"nodes": list(graph_nodes.values()), "edges": list(uniq.values())}, "notes": notes, "graph_paper_count": len(graph_papers)}
