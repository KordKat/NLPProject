"""search_arxiv — semantic search over the FAISS arXiv index."""
from __future__ import annotations

import asyncio
from typing import Any

from app.retrieval.faiss_search import get_arxiv_index
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.common import add_source, arxiv_source
from app.tools.registry import compact_json, registry


async def search_arxiv(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    index = get_arxiv_index()
    if not index.available:
        return ToolResult.failure(
            f"ArXiv index unavailable ({index.error or index.status}). Paper retrieval cannot be performed; say so to the user."
        )
    query = str(args["query"]).strip()
    top_k = int(args.get("top_k", 5) or 5)
    papers = await asyncio.to_thread(
        index.search,
        query,
        top_k,
        args.get("min_score"),
        str(args.get("date_from", "") or ""),
        str(args.get("date_to", "") or ""),
    )
    # results may come from a shared cache: copy before annotating with run-specific refs
    papers = [dict(p) for p in papers]
    if not papers:
        return ToolResult(
            content=compact_json({"query": query, "results": [], "note": "No sufficiently relevant papers in the indexed corpus (demo subset)."}),
            data={"query": query, "papers": []},
        )
    model_view = []
    for p in papers:
        src = add_source(ctx, arxiv_source(p))
        p["ref"] = src["ref"]
        model_view.append(
            {
                "ref": src["ref"],
                "arxiv_id": p["arxiv_id"],
                "title": p["title"],
                "authors": p["authors"][:6],
                "published": p["published"],
                "score": p["score"],
                "abstract": (p["abstract"] or "")[:500],
                "relevant_chunk": (p["chunks"][0]["text"][:700] if p["chunks"] else ""),
                "url": p["url"],
            }
        )
    return ToolResult(content=compact_json({"query": query, "results": model_view}, 7000), data={"query": query, "papers": papers}, sources=[arxiv_source(p) for p in papers])


registry.register(
    ToolSpec(
        name="search_arxiv",
        description="Semantic search over the local arXiv paper database (FAISS index). Returns papers with metadata, abstract, the most relevant text chunks and a similarity score.",
        parameters={
            "properties": {
                "query": {"type": "string", "description": "A focused scientific search query (key terms, not a full sentence)."},
                "top_k": {"type": "integer", "description": "Max number of papers (1-20, default 5)."},
                "date_from": {"type": "string", "description": "Optional earliest publication date YYYY-MM-DD."},
                "date_to": {"type": "string", "description": "Optional latest publication date YYYY-MM-DD."},
                "min_score": {"type": "number", "description": "Optional minimum cosine similarity (default 0.30)."},
            },
            "required": ["query"],
        },
        handler=search_arxiv,
        category="research",
        when_to_use="Scientific/technical questions that need evidence from research papers: methods, mechanisms, results, comparisons.",
        when_not_to_use="Pure arithmetic, coding tasks, file operations, or very recent news (use search_web).",
        returns="JSON {query, results:[{ref, arxiv_id, title, authors, published, score, abstract, relevant_chunk, url}]}. Empty results means nothing relevant exists in the corpus.",
        example_args={"query": "transformer self-attention long-range dependencies", "top_k": 5},
        example_result='{"query":"transformer self-attention long-range dependencies","results":[{"ref":1,"arxiv_id":"1706.03762","title":"Attention Is All You Need","published":"2017-06-12","score":0.71,"abstract":"The dominant sequence transduction models ...","relevant_chunk":"Self-attention ... path length between long-range dependencies ...","url":"https://arxiv.org/abs/1706.03762"}]}',
        interpretation="Higher score = more similar (0.5+ is strong, 0.3-0.5 partial). Cite papers by their [ref] number. Only state what the abstract/chunk actually says.",
    )
)
