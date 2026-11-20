"""Recursive research workflow (Stages B–F):

query generation → parallel retrieval (FAISS + Neo4j + web) → hybrid ranking → KeawGPT relevance
filtering/extraction → knowledge augmentation (+ graph construction with provenance) → gap analysis →
additional retrieval … until evidence coverage is sufficient, retrieval value diminishes, or the
round limit is reached.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agent.model_engine import GenerationParams, KeawEngine
from app.agent.prompts import FILTER_EXAMPLE_IN, FILTER_EXAMPLE_OUT, FILTER_SYSTEM, GAP_EXAMPLE_IN, GAP_EXAMPLE_OUT, GAP_SYSTEM
from app.agent.tool_parser import extract_json
from app.config import get_settings
from app.knowledge_graph import graph_writer
from app.knowledge_graph.graph_retriever import format_for_context
from app.knowledge_graph.neo4j_client import GraphUnavailableError, get_graph
from app.retrieval import reranking
from app.retrieval.retrieval_pipeline import run_round
from app.tools.base import ToolContext, ToolResult

log = logging.getLogger("stemos.research")
settings = get_settings()

Emit = Callable[[dict[str, Any]], Awaitable[None]]
Executor = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]

TEXT_FIELDS = ("title", "published", "problem", "core_concept", "method", "results", "findings", "limitations")
LIST_FIELDS = ("authors", "mechanisms", "equations", "connections")


@dataclass
class ResearchOutcome:
    evidence: list[dict[str, Any]] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=lambda: {"nodes": [], "edges": []})
    graph_context: str = ""
    rounds: int = 0
    queries: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stop_reason: str = ""


def _norm_query(q: dict[str, str]) -> str:
    return f"{q.get('tool', 'arxiv')}::{' '.join(sorted(re.findall(r'[a-z0-9]+', q.get('query', '').lower())))}"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", s.lower()))


_QSTOP = {"the", "and", "for", "with", "from", "into", "that", "this", "what", "how", "why", "are", "does", "about", "latest", "recent", "research", "overview"}


def _qtokens(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", q.lower())) - _QSTOP


def dedupe_queries(queries: list[dict[str, Any]], previous: list[dict[str, Any]] | tuple = ()) -> list[dict[str, Any]]:
    """Drop queries that would retrieve essentially the same material as an earlier query for the
    same tool (token Jaccard ≥ 0.6), e.g. paraphrases a small model tends to produce."""
    pool = [(p.get("tool"), _qtokens(p.get("query", ""))) for p in previous]
    kept: list[dict[str, Any]] = []
    for q in queries:
        tk = _qtokens(q.get("query", ""))
        if not tk:
            continue
        if any(tool == q.get("tool") and len(tk & other) / max(1, len(tk | other)) >= 0.6 for tool, other in pool):
            continue
        kept.append(q)
        pool.append((q.get("tool"), tk))
    return kept


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "relevant")
    return bool(v)


def _num(v: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _grounded(claim: str, source_text: str, threshold: float) -> bool:
    ct = _tokens(claim)
    if not ct:
        return False
    return len(ct & _tokens(source_text)) / len(ct) >= threshold


class ResearchLoop:
    def __init__(self, engine: KeawEngine, ctx: ToolContext, emit: Emit, execute: Executor, *, question: str, plan: dict[str, Any], max_rounds: int, owner_id: str):
        self.engine = engine
        self.ctx = ctx
        self.emit = emit
        self.execute = execute
        self.question = question
        self.plan = plan
        self.max_rounds = max(1, min(max_rounds, 8))
        self.owner_id = owner_id
        self.out = ResearchOutcome()
        self.used_queries: set[str] = set()
        self.seen_keys: set[str] = set()

    async def step(self, stage: str, text: str, **extra: Any) -> None:
        await self.emit({"type": "research_step", "stage": stage, "text": text, **extra})

    # ------------------------------------------------------------------ main loop
    async def run(self) -> ResearchOutcome:
        queries = self._initial_queries()
        for rnd in range(1, self.max_rounds + 1):
            if self.ctx.cancel_event.is_set():
                self.out.stop_reason = "cancelled"
                break
            queries = dedupe_queries([q for q in queries if _norm_query(q) not in self.used_queries], self.out.queries)[:6]
            if not queries:
                self.out.stop_reason = "no new queries"
                break
            for q in queries:
                self.used_queries.add(_norm_query(q))
                self.out.queries.append({**q, "round": rnd})
            self.out.rounds = rnd
            sources_label = "scientific papers and knowledge graph" if settings.arxiv_search_enabled else "the web and knowledge graph"
            label = f"Searching {sources_label}" if rnd == 1 else "Performing additional retrieval"
            await self.step("retrieval", f"{label} (round {rnd}: {len(queries)} queries)", round=rnd, queries=queries)

            batch = await run_round(self.ctx, queries, self.question, self.execute, self.seen_keys)
            self.out.notes += batch["notes"]
            self._merge_graph(batch["graph"])
            if batch["graph"]["edges"] or batch.get("graph_paper_count"):
                await self.step("graph", f"Exploring related knowledge: {len(batch['graph']['nodes'])} concepts, {len(batch['graph']['edges'])} relationships, {batch.get('graph_paper_count', 0)} graph-linked papers")

            candidates = reranking.rank(batch["candidates"], self.question, bool(self.plan.get("needs_current_info")), limit=8)
            for c in candidates:
                self.seen_keys.add(c["source_key"])
            await self.emit({"type": "retrieval_sources", "round": rnd, "sources": [self._source_view(c) for c in candidates]})

            new_relevant = 0
            if candidates:
                await self.step("filter", f"Filtering {len(candidates)} retrieved sources for relevance")
                items = await self._filter(candidates)
                new_relevant = sum(1 for it in items if it.get("relevant"))
                for it in items:
                    if not it.get("relevant"):
                        it.pop("_kg", None)
                kept = [it for it in items if it.get("relevant")]
                self.out.evidence += kept
                self.ctx.evidence += kept
                await self.step("augment", f"Kept {new_relevant} of {len(candidates)} sources; augmenting the evidence base ({len(self.out.evidence)} items)", kept=[{"ref": k["ref"], "title": k.get("title", "")} for k in kept])
                await self._write_graph(kept, candidates)
            else:
                await self.step("filter", "No new relevant sources retrieved this round")

            if rnd >= self.max_rounds:
                self.out.stop_reason = f"round limit ({self.max_rounds}) reached"
                break
            if rnd > 1 and new_relevant < settings.research_min_new_evidence:
                self.out.stop_reason = "diminishing returns (no new relevant evidence)"
                break
            await self.step("gap", "Identifying missing evidence")
            gap = await self._gap_analysis()
            if gap.get("sufficient") or not gap.get("new_queries"):
                self.out.stop_reason = "evidence sufficient" if gap.get("sufficient") else "no further gaps identified"
                if gap.get("missing"):
                    self.out.notes.append("Remaining gaps: " + "; ".join(map(str, gap["missing"][:4])))
                break
            await self.step("gap", "Missing: " + "; ".join(map(str, gap.get("missing", [])[:3])) if gap.get("missing") else "Refining queries")
            queries = gap["new_queries"]

        self.out.graph_context = format_for_context(self.out.graph)
        await self.step("done", f"Research complete after {self.out.rounds} round(s): {len(self.out.evidence)} evidence items — {self.out.stop_reason}")
        return self.out

    # ------------------------------------------------------------------ helpers
    def _initial_queries(self) -> list[dict[str, str]]:
        qs = [q for q in (self.plan.get("queries") or []) if isinstance(q, dict) and q.get("query")]
        if not settings.arxiv_search_enabled:
            qs = [q for q in qs if q.get("tool") != "arxiv"]
        tools = {q.get("tool") for q in qs}
        from app.agent.orchestrator import keyword_query  # lazy: avoids a circular import

        main = self.plan.get("main_question") or self.question
        if "arxiv" not in tools and settings.arxiv_search_enabled:
            qs.append({"tool": "arxiv", "query": keyword_query(main)})
        if "graph" not in tools:
            qs.append({"tool": "graph", "query": keyword_query(main, 4)})
        if (not settings.arxiv_search_enabled or self.plan.get("needs_current_info")) and "web" not in tools:
            qs.append({"tool": "web", "query": keyword_query(main)})
        return qs

    def _merge_graph(self, g: dict[str, Any]) -> None:
        nodes = {n["uid"]: n for n in self.out.graph["nodes"]}
        for n in g.get("nodes", []):
            nodes.setdefault(n["uid"], n)
        edges = {(e["source"], e["type"], e["target"]): e for e in self.out.graph["edges"]}
        for e in g.get("edges", []):
            edges.setdefault((e["source"], e["type"], e["target"]), e)
        self.out.graph = {"nodes": list(nodes.values()), "edges": list(edges.values())}

    @staticmethod
    def _source_view(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "ref": c.get("ref"), "title": c.get("title", ""), "url": c.get("url", ""), "kind": c.get("kind"),
            "paper_id": c.get("paper_id"), "published": c.get("published", ""), "score": c.get("hybrid_score"),
            "authors": (c.get("authors") or [])[:4], "via_graph": bool(c.get("via_graph")),
        }

    def _doc_text(self, c: dict[str, Any]) -> str:
        head = f"[{c['ref']}] {c.get('title', '')}"
        if c.get("paper_id"):
            head += f" (arXiv:{c['paper_id']}, {c.get('published', '')})"
        elif c.get("url"):
            head += f" ({c['url']})"
        parts = [head]
        if c.get("authors") or c.get("byline"):
            parts.append(f"Authors/byline: {', '.join(c.get('authors') or []) or c.get('byline', '')}")
        if c.get("shared_concepts"):
            parts.append(f"Found via knowledge graph; shares concepts: {', '.join(c['shared_concepts'][:5])}")
        if c.get("abstract"):
            parts.append(f"Abstract: {c['abstract'][:900]}")
        for ch in (c.get("chunks") or [])[:2]:
            if ch.get("text"):
                parts.append(f"Excerpt: {ch['text'][:1100]}")
        if c.get("kind") == "web_snippet":
            parts.append("(only a search snippet is available for this web result)")
        return "\n".join(parts)

    async def _filter(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        size = max(1, settings.research_sources_per_filter_call)
        batches = [candidates[i: i + size] for i in range(0, len(candidates), size)]
        items: list[dict[str, Any]] = []
        subq = "; ".join(self.plan.get("subquestions") or [])
        for b in batches:
            if self.ctx.cancel_event.is_set():
                break
            docs = "\n\n".join(self._doc_text(c) for c in b)
            user = f"QUESTION: {self.question}\nSUBQUESTIONS: {subq or '-'}\nDOCUMENTS:\n{docs}"
            msgs = [
                {"role": "system", "content": FILTER_SYSTEM},
                {"role": "user", "content": FILTER_EXAMPLE_IN},
                {"role": "assistant", "content": FILTER_EXAMPLE_OUT},
                {"role": "user", "content": user},
            ]
            try:
                out = await self.engine.complete(msgs, GenerationParams(max_new_tokens=1800, temperature=0.1), cancel_event=self.ctx.cancel_event)
                parsed = extract_json(out)
            except Exception as exc:
                log.warning("filter call failed: %s", exc)
                parsed = None
            items += self._validate_items(parsed, b)
        return items

    def _validate_items(self, parsed: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_ref = {c["ref"]: c for c in batch}
        raw_items = parsed.get("items") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else None
        results: dict[Any, dict[str, Any]] = {}
        for it in raw_items or []:
            if not isinstance(it, dict):
                continue
            try:
                ref = int(it.get("ref"))
            except (TypeError, ValueError):
                continue
            c = by_ref.get(ref)
            # small models sometimes attach an extraction to the wrong document number:
            # trust the title the model wrote and re-attach the item to the document it matches
            mtitle = _tokens(str(it.get("title", "")))
            if mtitle:
                def overlap(cand: dict[str, Any]) -> float:
                    ct = _tokens(cand.get("title", ""))
                    return len(mtitle & ct) / max(1, len(mtitle))
                best = max(batch, key=overlap)
                if c is None or (overlap(c) < 0.4 and overlap(best) >= 0.6):
                    c, ref = best, best["ref"]
            if c is None or ref in results:
                continue
            source_text = self._doc_text(c)
            ev: dict[str, Any] = {
                "ref": ref, "source_key": c["source_key"], "source_type": "web" if c.get("kind", "").startswith("web") else "arxiv",
                "paper_id": c.get("paper_id", ""), "url": c.get("url", ""),
                "relevant": _truthy(it.get("relevant")), "relevance": _num(it.get("relevance"), 0.5) if _truthy(it.get("relevant")) else 0.0,
                "via_graph": bool(c.get("via_graph")),
            }
            for k in TEXT_FIELDS:
                v = it.get(k)
                ev[k] = str(v).strip()[:900] if isinstance(v, (str, int, float)) and str(v).strip() else ""
            for k in LIST_FIELDS:
                v = it.get(k)
                ev[k] = [str(x)[:400] for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:8] if isinstance(v, list) else []
            # anti-fabrication: keep metadata from the source and only grounded equations
            ev["title"] = c.get("title") or ev["title"]
            ev["published"] = c.get("published") or ev["published"]
            if c.get("authors"):
                ev["authors"] = c["authors"]
            ev["equations"] = [e for e in ev["equations"] if _grounded(e, source_text, 0.5)]
            concepts = [x for x in (it.get("concepts") or []) if isinstance(x, dict) and x.get("name")][:12]
            relations = [r for r in (it.get("relations") or []) if isinstance(r, dict) and _grounded(str(r.get("evidence", "")), source_text, 0.6)][:12]
            ev["_kg"] = {"concepts": concepts, "relations": relations, "authors": ev["authors"]}
            results[ref] = ev
        # sources the model skipped are treated as low-confidence relevant when strongly similar
        for ref, c in by_ref.items():
            if ref not in results and (c.get("hybrid_score") or 0) >= 0.55:
                results[ref] = {
                    "ref": ref, "source_key": c["source_key"], "source_type": "arxiv" if c.get("paper_id") else "web", "paper_id": c.get("paper_id", ""),
                    "url": c.get("url", ""), "relevant": True, "relevance": 0.4, "title": c.get("title", ""), "published": c.get("published", ""),
                    "authors": c.get("authors", []), "core_concept": "", "method": "", "mechanisms": [], "equations": [], "results": "",
                    "findings": (c.get("abstract") or "")[:500], "limitations": "", "connections": [], "problem": "", "_kg": {"concepts": [], "relations": [], "authors": []},
                    "unfiltered": True,
                }
        return list(results.values())

    async def _write_graph(self, kept: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
        if not get_graph().available:
            self.out.notes.append("Knowledge graph unavailable — extracted concepts were not saved.")
            return
        by_key = {c["source_key"]: c for c in candidates}
        total_e = total_r = 0
        for ev in kept:
            kg = ev.pop("_kg", None)
            c = by_key.get(ev["source_key"])
            if not kg or c is None:
                continue
            paper = c if c.get("paper_id") else None
            if paper is None and not kg["relations"]:
                continue
            try:
                res = await asyncio.to_thread(graph_writer.write_extraction, kg, project_id=self.ctx.project_id, owner_id=self.owner_id, paper=paper, private=paper is None)
                total_e += res["entities"]
                total_r += res["relationships"]
            except GraphUnavailableError as exc:
                self.out.notes.append(f"Graph write failed: {exc}")
                break
            except Exception as exc:
                log.warning("graph write failed: %s", exc)
        for ev in kept:
            ev.pop("_kg", None)
        if total_e:
            await self.emit({"type": "graph_updated", "entities": total_e, "relationships": total_r})

    async def _gap_analysis(self) -> dict[str, Any]:
        ev_lines = []
        for ev in self.out.evidence[-14:]:
            ev_lines.append(f"[{ev['ref']}] {ev.get('title', '')[:90]} — {(ev.get('core_concept') or ev.get('findings') or '')[:220]}")
        used = " | ".join(f"{q['tool']}: {q['query']}" for q in self.out.queries)
        user = (
            f"QUESTION: {self.question}\nSUBQUESTIONS: {'; '.join(self.plan.get('subquestions') or []) or '-'}\n"
            f"EVIDENCE:\n{chr(10).join(ev_lines) or '(none yet)'}\nQUERIES USED: {used}"
        )
        if self.out.graph["edges"]:
            user += f"\nGRAPH RELATIONS FOUND: {len(self.out.graph['edges'])} (unexplored concepts: {', '.join(n['name'] for n in self.out.graph['nodes'][:8] if n.get('type') != 'Paper')})"
        msgs = [
            {"role": "system", "content": GAP_SYSTEM},
            {"role": "user", "content": GAP_EXAMPLE_IN},
            {"role": "assistant", "content": GAP_EXAMPLE_OUT},
            {"role": "user", "content": user},
        ]
        try:
            out = await self.engine.complete(msgs, GenerationParams(max_new_tokens=500, temperature=0.2), cancel_event=self.ctx.cancel_event)
            data = extract_json(out)
        except Exception as exc:
            log.warning("gap analysis failed: %s", exc)
            data = None
        if not isinstance(data, dict):
            return {"sufficient": len(self.out.evidence) >= 3, "new_queries": []}
        nq = []
        for q in data.get("new_queries") or []:
            allowed = ("arxiv", "graph", "web") if settings.arxiv_search_enabled else ("graph", "web")
            if isinstance(q, dict) and q.get("query") and q.get("tool") in allowed:
                if _norm_query(q) not in self.used_queries:
                    nq.append({"tool": q["tool"], "query": str(q["query"])[:200]})
        data["new_queries"] = nq[:3]
        return data


def evidence_for_storage(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps([{k: v for k, v in e.items() if not k.startswith("_")} for e in evidence], default=str))
