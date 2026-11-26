"""Research agent: queries → retrieval → relevant facts → ONE cited paragraph.

The only model calls are the query writer and a single summary at the end. Relevance filtering,
gap detection and citation are done with embeddings, so extra retrieval rounds cost almost nothing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.agent.prompts import EXPECTED_SHOTS, EXPECTED_SYSTEM, QUERY_SHOTS, QUERY_SYSTEM, SUMMARY_SHOT_IN, SUMMARY_SHOT_OUT, SUMMARY_SYSTEM
from app.agent.runtime import Handoff, Runtime
from app.agent.tool_parser import extract_json
from app.config import get_settings
from app.knowledge_graph.graph_retriever import format_for_context
from app.knowledge_graph.neo4j_client import get_graph
from app.retrieval import reranking
from app.retrieval.evidence import Fact, attribute_paragraph, extractive_paragraph, select_facts, states_no_answer, uncovered
from app.retrieval.retrieval_pipeline import run_round

log = logging.getLogger("stemos.research")
settings = get_settings()

RECENT = re.compile(r"\b(latest|newest|recent(ly)?|currently|today|this (year|month|week)|news|20[2-3]\d|approved|regulation|price|release[ds]?|won|winner)\b", re.I)
IMAGES = re.compile(r"\b(photos?|pictures?|images?)\s+of\b|\bshow me\b.*\b(photo|picture|image|diagram)", re.I)
_QSTOP = {"the", "and", "for", "with", "from", "into", "that", "this", "what", "how", "why", "are", "does", "about", "latest", "recent", "research", "overview", "using", "based"}


_EXAMPLE_QUERIES = {q for _, queries in QUERY_SHOTS for q in queries}


def _qtokens(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", q.lower())) - _QSTOP


def dedupe_queries(queries: list[str], previous: list[str] | tuple = ()) -> list[str]:
    """Drop queries that would retrieve the same material (token Jaccard ≥ 0.6)."""
    pool = [_qtokens(p) for p in previous]
    kept: list[str] = []
    for q in queries:
        tk = _qtokens(q)
        if not tk or any(len(tk & o) / max(1, len(tk | o)) >= 0.6 for o in pool):
            continue
        kept.append(q)
        pool.append(tk)
    return kept


def keywords(text: str, n: int = 8) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9][\w'\-]*", text) if w.lower() not in _QSTOP and len(w) > 2]
    return " ".join(list(dict.fromkeys(words))[:n]) or text[:100]


class ResearchAgent:
    name = "research"
    label = "Research agent"
    description = "Find facts in papers and on the web; returns one cited paragraph. For recent events, specific studies, statistics, or when sources are requested."

    async def _write_queries(self, rt: Runtime, task: str, used: list[str]) -> list[str]:
        messages: list[dict[str, str]] = [{"role": "system", "content": QUERY_SYSTEM}]
        for t, q in QUERY_SHOTS:
            messages += [{"role": "user", "content": t}, {"role": "assistant", "content": json.dumps(q)}]
        ask = task[:900] if not used else f"{task[:500]}\nAlready searched: {'; '.join(used[-6:])}\nWrite queries for what is still missing."
        messages.append({"role": "user", "content": ask})
        out = await rt.llm(messages, max_new_tokens=80, temperature=0.0, prefill='["', stop=("]",))
        parsed = extract_json(out if out.rstrip().endswith("]") else out.rstrip().rstrip(",") + '"]')
        found = [str(q).strip()[:140] for q in parsed if isinstance(q, str)] if isinstance(parsed, list) else []
        found = [q for q in found if q not in _EXAMPLE_QUERIES]  # copied from the in-context examples
        found = dedupe_queries([q for q in found if 2 <= len(q.split()) <= 16], used)[:4]
        return found or ([] if used else [keywords(task)])

    async def _expected_answer(self, rt: Runtime, task: str) -> str:
        """A guess at the answer's wording, used only to rank real sentences. Never shown or cited."""
        messages: list[dict[str, str]] = [{"role": "system", "content": EXPECTED_SYSTEM}]
        for q, a in EXPECTED_SHOTS:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": task[:400]})
        out = await rt.llm(messages, max_new_tokens=45, temperature=0.0)
        return (out.strip().split("\n") or [""])[0][:300]

    def _route(self, rt: Runtime, queries: list[str], task: str, first: bool) -> list[dict[str, str]]:
        """Send each query where it can actually be answered."""
        plan: list[dict[str, str]] = []
        use_arxiv = settings.arxiv_search_enabled and rt.has_tool("search_arxiv")
        use_web = rt.has_tool("search_web") and (not use_arxiv or bool(RECENT.search(task)))
        if use_arxiv:
            plan += [{"tool": "arxiv", "query": q} for q in queries]
        if use_web:
            plan += [{"tool": "web", "query": q} for q in queries[:3]]
        if first and rt.has_tool("search_knowledge_graph") and get_graph().available:
            plan.append({"tool": "graph", "query": keywords(queries[0], 4)})
        return plan

    async def _summarize(self, rt: Runtime, task: str, facts: list[Fact]) -> str:
        bullet = "\n".join(f"- {f.text}" for f in facts)
        out = await rt.llm(
            [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": SUMMARY_SHOT_IN},
                {"role": "assistant", "content": SUMMARY_SHOT_OUT},
                {"role": "user", "content": f"QUESTION: {task[:400]}\nFACTS:\n{bullet}"},
            ],
            max_new_tokens=280,
            temperature=0.2,
        )
        draft = out.split("\n\n")[0]
        # judged on what survives verification, before any source is marked as cited
        probe = attribute_paragraph(draft, facts, lambda _: 0) or extractive_paragraph(facts, lambda _: 0)
        if states_no_answer(draft) or states_no_answer(probe, task):
            return ""
        return attribute_paragraph(draft, facts, rt.cite) or extractive_paragraph(facts, rt.cite)

    def _store_evidence(self, rt: Runtime, facts: list[Fact]) -> None:
        """Per-source records used by the report writer, project memory and the graph curator."""
        by_key: dict[str, list[Fact]] = {}
        for f in facts:
            by_key.setdefault(f.source_key, []).append(f)
        known = {e["source_key"] for e in rt.ctx.evidence}
        for key in rt.board.cited:
            src = rt.ctx.sources.get(key)
            if src is None or key in known or key not in by_key:
                continue
            meta = src.get("meta") or {}
            rt.ctx.evidence.append({
                "ref": src["ref"], "source_key": key, "source_type": src.get("source_type"), "paper_id": src.get("paper_id", ""),
                "url": src.get("url", ""), "title": src.get("title", ""), "published": meta.get("published", ""), "authors": meta.get("authors", []),
                "relevant": True, "relevance": max(f.score for f in by_key[key]), "findings": " ".join(f.text for f in by_key[key]),
            })

    async def run(self, rt: Runtime, task: str) -> Handoff:
        board = rt.board
        failed: list[str] = []
        if IMAGES.search(task) and rt.has_tool("search_images"):
            res = await rt.execute("search_images", {"query": keywords(task, 6), "limit": 3})
            if res.ok:
                board.built += [f"{i['title']} ({i['attribution']})" for i in (res.data or {}).get("images", [])]
        await rt.step("Writing search queries", stage="retrieval")
        queries = await self._write_queries(rt, task, [q["query"] for q in board.queries])
        expected = await self._expected_answer(rt, task)
        pool: list[Fact] = list(board.facts)
        seen: set[str] = {f.source_key for f in pool}
        nodes: dict[str, dict] = {}
        edges: dict[tuple, dict] = {}
        quiet = lambda name, args: rt.execute(name, args, announce_sources=False)  # noqa: E731

        for rnd in range(1, rt.research_rounds + 1):
            if rt.ctx.cancel_event.is_set() or not queries:
                break
            plan = self._route(rt, queries, task, first=rnd == 1)
            if not plan:
                board.problems.append("No search tool is enabled for this project.")
                break
            board.rounds += 1
            board.queries += [{**q, "round": rnd} for q in plan]
            await rt.step(f"Searching (round {rnd}): " + "; ".join(queries), stage="retrieval", round=rnd)
            batch = await run_round(rt.ctx, plan, task, quiet, seen)
            failed += [n for n in batch["notes"] if "failed" in n or "unavailable" in n]
            for n in batch["graph"]["nodes"]:
                nodes.setdefault(n["uid"], n)
            for e in batch["graph"]["edges"]:
                edges.setdefault((e["source"], e["type"], e["target"]), e)
            ranked = reranking.rank(batch["candidates"], task, bool(RECENT.search(task)), limit=8)
            seen.update(c["source_key"] for c in ranked)
            fresh = await asyncio.to_thread(select_facts, task, ranked, expected=expected)
            pool = sorted(pool + fresh, key=lambda f: f.score, reverse=True)[:14]
            await rt.step(f"Kept {len(fresh)} relevant facts from {len(ranked)} sources", stage="filter")
            if rnd >= rt.research_rounds or not fresh:
                break
            gaps = await asyncio.to_thread(uncovered, queries, pool)
            if not gaps:
                break
            await rt.step("Missing evidence: " + "; ".join(gaps[:2]), stage="gap")
            queries = await self._write_queries(rt, f"{task}\nMissing: {'; '.join(gaps[:2])}", [q["query"] for q in board.queries])

        if nodes and edges:
            board.graph_notes = format_for_context({"nodes": list(nodes.values()), "edges": list(edges.values())}, max_edges=6)
        if failed:
            log.info("retrieval problems: %s", failed)
        if not pool:
            msg = "No relevant evidence was found" + (" (a search service failed)" if failed else "")
            # research runs when the answer depends on sources: the model would invent the specifics
            board.exact = f"{msg}, so I can't give a reliable answer to this."
            return Handoff(False, msg)

        await rt.step(f"Writing one cited paragraph from {len(pool)} facts", stage="filter")
        paragraph = await self._summarize(rt, task, pool)
        if not paragraph:
            board.exact = "I searched, but the sources I found do not state this, so I can't give a reliable answer."
            return Handoff(False, "The sources do not answer this.")
        board.facts = pool
        board.brief = f"{board.brief} {paragraph}".strip()
        self._store_evidence(rt, pool)
        await rt.emit({"type": "retrieval_sources", "sources": [_source_view(s) for s in rt.cited_sources()]})
        return Handoff(True, f"Cited paragraph stored ({len(board.cited)} sources).", paragraph)


def _source_view(s: dict[str, Any]) -> dict[str, Any]:
    meta = s.get("meta") or {}
    return {"ref": s.get("ref"), "title": s.get("title"), "url": s.get("url"), "kind": s.get("source_type"), "paper_id": s.get("paper_id"), "published": meta.get("published", "")}
