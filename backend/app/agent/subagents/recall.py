"""Recall agent: reads and updates what this project remembers."""
from __future__ import annotations

import re

from app.agent.runtime import Handoff, Runtime, call_prefill
from app.agent.tool_parser import parse_prefilled_call

SAVE = re.compile(r"\b(remember|record|save|note)\b.*\b(that|relationship|relation)\b|\badd\b.*\b(knowledge )?graph\b", re.I)
ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
ORDINAL = re.compile(r"\b(first|second|third|fourth|fifth|last) (?:paper|source|article|study|reference)\b", re.I)
_SAVE_SYSTEM ="Record what the user states in the knowledge graph. concepts is a JSON list of the concept names, relationship is one of RELATED_TO, USES_METHOD, EXTENDS, SOLVES, BELONGS_TO, DERIVED_FROM, CONTRADICTS."


class RecallAgent:
    name = "recall"
    label = "Recall agent"
    description = "Look up earlier work saved in this project."

    async def _save(self, rt: Runtime, task: str) -> Handoff:
        out = await rt.llm(
            [{"role": "system", "content": _SAVE_SYSTEM}, {"role": "user", "content": task[:600]}],
            max_new_tokens=140, tools=rt.schemas("upsert_knowledge_graph"), prefill=call_prefill("upsert_knowledge_graph", "concepts"),
        )
        args = parse_prefilled_call(out, "upsert_knowledge_graph")
        if not args.get("concepts"):
            rt.board.problems.append("The relationship could not be understood, so nothing was saved.")
            return Handoff(False, "Nothing saved.")
        args["evidence"] = task[:300]
        res = await rt.execute("upsert_knowledge_graph", args)
        if not res.ok:
            rt.board.problems.append(res.error)
            return Handoff(False, res.error[:120])
        rt.board.memory = f"Saved to this project's knowledge graph: {args.get('concepts')} ({args.get('relationship', 'RELATED_TO')})."
        return Handoff(True, "Saved to the knowledge graph.")

    async def run(self, rt: Runtime, task: str) -> Handoff:
        if SAVE.search(task) and rt.has_tool("upsert_knowledge_graph"):
            return await self._save(rt, task)
        res = await rt.execute("recall_project", {"query": task[:300]})
        data = res.data if isinstance(res.data, dict) else {}
        if not res.ok:
            rt.board.problems.append(res.error)
            return Handoff(False, "Project memory could not be read.")
        lines: list[str] = []
        if data.get("summary"):
            lines.append(f"Summary: {data['summary']}")
        sources = sorted(data.get("sources") or [], key=lambda s: s.get("order") or 999)
        if sources:
            lines.append("Saved sources, oldest first:")
            for s in sources[:10]:
                ident = f"arXiv:{s['paper_id']}" if s.get("paper_id") else s.get("url", "")
                lines.append(f"{s.get('order')}. {s.get('title', '')[:100]} ({ident})")
        for ev in (data.get("evidence") or [])[:4]:
            text = ev.get("findings") or ev.get("method") or ev.get("core_concept")
            if text:
                lines.append(f"Finding from \"{(ev.get('title') or '')[:60]}\": {str(text)[:260]}")
        if not lines:
            rt.board.memory = "This project has no saved research yet."
            return Handoff(True, "The project has no saved research.")
        ordinal = ORDINAL.search(task)
        if ordinal and sources:  # "the second paper" has one exact answer: no model needed
            word = ordinal.group(1).lower()
            index = -1 if word == "last" else ORDINALS[word]
            if -len(sources) <= index < len(sources):
                s = sources[index]
                ident = f"arXiv:{s['paper_id']}" if s.get("paper_id") else s.get("url", "")
                rt.board.exact = f"The {word} source saved in this project is **{s.get('title', '').strip()}** ({ident})."
            else:
                rt.board.exact = f"This project only has {len(sources)} saved source{'s' if len(sources) != 1 else ''}, so there is no {word} one."
            return Handoff(True, "Found the exact source.", rt.board.exact)
        rt.board.memory = "\n".join(lines)
        return Handoff(True, f"Found earlier work ({len(sources)} saved sources).", rt.board.memory)
