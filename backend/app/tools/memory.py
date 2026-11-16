"""recall_project — retrieve earlier research from the current project (memory)."""
from __future__ import annotations

import asyncio
from typing import Any

from app.services.project_memory import recall
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.common import add_source
from app.tools.registry import compact_json, registry


async def recall_project(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query", "") or "")
    data = await asyncio.to_thread(recall, ctx.project_id, query)
    for s in data.get("sources", []):
        src = {k: v for k, v in s.items() if k not in ("ref", "order")}
        # renumber into this run's citation space so [n] stays unambiguous
        s["ref"] = add_source(ctx, src)["ref"]
    return ToolResult(content=compact_json(data, 7000), data=data)


registry.register(ToolSpec(
    name="recall_project",
    description="Retrieve earlier work from the current project: past questions, previously retrieved papers/sources, extracted evidence, decisions and saved concepts.",
    parameters={"properties": {"query": {"type": "string", "description": "What to look for, e.g. 'the second paper' or 'diffusion sampling method'."}}, "required": ["query"]},
    handler=recall_project, category="research",
    when_to_use="The user refers to earlier work ('the previous method', 'the second paper', 'continue', 'as before').",
    when_not_to_use="A brand new topic unrelated to the project.",
    returns="JSON {summary, past_questions:[...], sources:[{ref,title,paper_id,url}], evidence:[{ref,title,findings}], decisions:[...]} ",
    example_args={"query": "previously discussed papers on denoising diffusion"},
    example_result='{"summary":"Project investigates diffusion models ...","past_questions":["Explain how diffusion models work"],"sources":[{"ref":1,"title":"Denoising Diffusion Probabilistic Models","paper_id":"2006.11239"},{"ref":2,"title":"Score-Based Generative Modeling through SDEs","paper_id":"2011.13456"}],"evidence":[{"ref":2,"findings":"unifies score matching and diffusion via SDEs"}]}',
    interpretation="'The second paper' means the second source listed in earlier work. Use the paper ids with search_arxiv / search_knowledge_graph for more detail.",
))
