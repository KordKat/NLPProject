"""search_knowledge_graph / upsert_knowledge_graph — Neo4j tools."""
from __future__ import annotations

import asyncio
from typing import Any

from app.knowledge_graph import graph_retriever, graph_writer
from app.knowledge_graph.neo4j_client import GraphUnavailableError, get_graph
from app.knowledge_graph.schema import RELATIONSHIPS, normalize_relationship
from app.retrieval.faiss_search import get_arxiv_index
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.common import add_source
from app.tools.registry import compact_json, registry


async def search_knowledge_graph(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not get_graph().available:
        return ToolResult.failure(f"Knowledge graph unavailable ({get_graph().error}). Graph retrieval was NOT performed.")
    query = str(args["query"])
    # project scope always comes from the authenticated context, never from model arguments
    try:
        res = await asyncio.to_thread(graph_retriever.search, query, ctx.project_id, int(args.get("limit", 10) or 10), 2)
    except GraphUnavailableError as exc:
        return ToolResult.failure(str(exc))
    for p in res.get("papers", []):
        if p.get("paper_id"):
            add_source(ctx, {"source_key": f"graph:{p['uid']}", "source_type": "graph", "paper_id": p["paper_id"], "title": p["name"], "url": p.get("url") or "", "relevance": 0.4, "meta": {"via": "knowledge_graph"}})
    if not res["nodes"]:
        return ToolResult(content=compact_json({"query": query, "concepts": [], "relationships": [], "note": "No matching knowledge in the graph yet."}), data=res)
    names = {n["uid"]: n["name"] for n in res["nodes"]}
    view = {
        "query": query,
        "concepts": [{"name": n["name"], "type": n["type"], "description": n["description"][:160]} for n in res["nodes"] if n["type"] != "Paper"][:20],
        "papers": [{"title": n["name"], "arxiv_id": n.get("paper_id")} for n in res["nodes"] if n["type"] == "Paper"][:10],
        "relationships": [
            {"from": names.get(e["source"], e["source"]), "relation": e["type"], "to": names.get(e["target"], e["target"]), "papers": e["source_papers"][:3], "evidence": (e["evidence"][0][:200] if e["evidence"] else "")}
            for e in res["edges"][:30]
        ],
    }
    return ToolResult(content=compact_json(view, 7000), data=res)


async def upsert_knowledge_graph(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not get_graph().available:
        return ToolResult.failure(f"Knowledge graph unavailable ({get_graph().error}). Nothing was written.")
    concepts = args.get("concepts") or []
    if isinstance(concepts, str):
        concepts = [c.strip() for c in concepts.split(",") if c.strip()]
    if len(concepts) < 1:
        return ToolResult.failure("Provide at least one concept.")
    relationship = str(args.get("relationship", "RELATED_TO"))
    if len(concepts) >= 2 and normalize_relationship(relationship) is None:
        return ToolResult.failure(f"Unknown relationship '{relationship}'. Allowed: {', '.join(RELATIONSHIPS)}")
    paper_id = str(args.get("paper_id", "") or "").strip()
    evidence = str(args.get("evidence", "") or "").strip()
    types = args.get("concept_types") or []
    if isinstance(types, str):
        types = [t.strip() for t in types.split(",")]
    paper = None
    if paper_id:
        idx = get_arxiv_index()
        paper = idx.get_paper(paper_id) if idx.available else None
        if paper is None:
            paper = {"paper_id": paper_id, "title": f"arXiv:{paper_id}", "url": f"https://arxiv.org/abs/{paper_id}"}
    private = not paper_id  # knowledge without a public paper source stays private to this project
    ex = {
        "concepts": [{"name": c, "type": (types[i] if i < len(types) and types[i] else "Concept")} for i, c in enumerate(concepts)],
        "relations": [],
    }
    if len(concepts) >= 2:
        ev = evidence or (f"Asserted in project with reference to arXiv:{paper_id}" if paper_id else "User/project note")
        for i in range(len(concepts) - 1):
            ex["relations"].append({"subject": concepts[i], "subject_type": ex["concepts"][i]["type"], "relation": relationship, "object": concepts[i + 1], "object_type": ex["concepts"][i + 1]["type"], "evidence": ev, "confidence": 0.6})
    try:
        out = await asyncio.to_thread(graph_writer.write_extraction, ex, project_id=ctx.project_id, owner_id=ctx.user_id, paper=paper, private=private)
    except GraphUnavailableError as exc:
        return ToolResult.failure(str(exc))
    await ctx.emit({"type": "graph_updated", "entities": out["entities"], "relationships": out["relationships"]})
    return ToolResult(content=compact_json({"entities_upserted": out["entities"], "relationships_upserted": out["relationships"], "scope": "project-private" if private else "shared (paper-derived)"}), data=out)


registry.register(ToolSpec(
    name="search_knowledge_graph",
    description="Search the Neo4j scientific knowledge graph for concepts, methods, datasets and papers related to a topic, and traverse their relationships (with provenance).",
    parameters={"properties": {
        "query": {"type": "string", "description": "Concepts to look up, e.g. 'self-attention long-range dependencies'."},
        "project_id": {"type": "string", "description": "Optional; the current project is always used."},
        "limit": {"type": "integer", "description": "Max seed concepts (default 10)."},
    }, "required": ["query"]},
    handler=search_knowledge_graph, category="graph",
    when_to_use="Every research question (alongside search_arxiv) to find connected concepts and papers that vector search may miss; when the user asks how ideas relate.",
    when_not_to_use="Arithmetic or file operations.",
    returns="JSON {concepts:[{name,type,description}], papers:[{title, arxiv_id}], relationships:[{from, relation, to, papers, evidence}]}.",
    example_args={"query": "self-attention and long-range dependencies"},
    example_result='{"concepts":[{"name":"Self-Attention","type":"Concept"},{"name":"Positional Encoding","type":"Method"}],"papers":[{"title":"Attention Is All You Need","arxiv_id":"1706.03762"}],"relationships":[{"from":"Attention Is All You Need","relation":"INTRODUCES","to":"Transformer","papers":["1706.03762"],"evidence":"We propose a new simple network architecture, the Transformer"}]}',
    interpretation="Relationships are backed by the listed papers/evidence. Use connected papers as leads (search_arxiv for details). An empty graph is normal for new topics.",
))

registry.register(ToolSpec(
    name="upsert_knowledge_graph",
    description="Create or update knowledge-graph entities and a typed relationship between consecutive concepts, with provenance. Idempotent.",
    parameters={"properties": {
        "concepts": {"type": "array", "description": "Concept names, e.g. [\"Transformer\", \"Self-Attention\"]. A relationship links each concept to the next."},
        "relationship": {"type": "string", "description": f"One of: {', '.join(RELATIONSHIPS)}"},
        "paper_id": {"type": "string", "description": "arXiv id supporting the relationship (makes it shared knowledge). Omit for private project notes."},
        "evidence": {"type": "string", "description": "Short supporting quote or statement."},
        "concept_types": {"type": "array", "description": "Optional types matching concepts: Concept, Method, Algorithm, Dataset, Model, ResearchProblem, Equation, ScientificField."},
    }, "required": ["concepts"]},
    handler=upsert_knowledge_graph, category="graph", read_only=False,
    when_to_use="The user asks to remember/record how concepts relate, or you established a well-supported relationship worth saving.",
    when_not_to_use="Speculative links or mere co-occurrence of terms.",
    returns='JSON {"entities_upserted", "relationships_upserted", "scope"}.',
    example_args={"concepts": ["Transformer", "Self-Attention"], "relationship": "USES_METHOD", "paper_id": "1706.03762", "evidence": "The Transformer relies entirely on self-attention."},
    example_result='{"entities_upserted": 3, "relationships_upserted": 2, "scope": "shared (paper-derived)"}',
    interpretation="Saved knowledge becomes visible in the project's Knowledge Graph panel.",
))
