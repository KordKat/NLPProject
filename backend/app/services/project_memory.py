"""Project memory: persistent summaries, recall of earlier research, and summary maintenance."""
from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.model_engine import GenerationParams, get_engine
from app.agent.prompts import MEMORY_SYSTEM
from app.database.models import Project
from app.database.repositories import projects as repo
from app.database.session import db_session

log = logging.getLogger("stemos.memory")

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_STOP = {"the", "and", "for", "with", "that", "this", "what", "how", "why", "are", "was", "from", "previous", "paper", "papers", "method", "continue", "compare", "second", "first", "third"}


def _terms(q: str) -> list[str]:
    return [w for w in _WORD.findall(q.lower()) if w not in _STOP][:8]


def recall(project_id: str, query: str) -> dict[str, Any]:
    with db_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            return {"error": "project not found"}
        runs = repo.list_runs(db, project_id, 8)
        sources = repo.list_sources(db, project_id)
        terms = _terms(query)
        matched = repo.search_sources(db, project_id, terms, 10) if terms else []
        # chronological order so "the second paper" is meaningful
        ordered = sorted(sources, key=lambda s: s.created_at)
        listing = []
        seen = set()
        for s in matched + ordered:
            if s.id in seen:
                continue
            seen.add(s.id)
            listing.append({
                "order": ordered.index(s) + 1 if s in ordered else None,
                "source_key": s.source_key,
                "source_type": s.source_type,
                "title": s.title,
                "paper_id": s.paper_id,
                "url": s.source_url,
                "relevance": s.relevance,
                "meta": {k: v for k, v in (s.meta or {}).items() if k in ("authors", "published")},
            })
            if len(listing) >= 15:
                break
        evidence = []
        for r in runs:
            for ev in r.evidence or []:
                text = " ".join(str(ev.get(k, "")) for k in ("title", "core_concept", "method", "findings"))
                if not terms or any(t in text.lower() for t in terms):
                    evidence.append({k: ev.get(k) for k in ("title", "paper_id", "source_key", "core_concept", "method", "findings", "limitations") if ev.get(k)})
        return {
            "summary": project.summary or "",
            "past_questions": [r.user_question[:160] for r in reversed(runs)][-8:],
            "sources": [{**s, "ref": i + 1} for i, s in enumerate(listing)],
            "evidence": evidence[:10],
            "decisions": (project.context or {}).get("decisions", [])[-10:],
            "saved_concepts": (project.context or {}).get("concepts", [])[-20:],
        }


def project_line(project: Project) -> str:
    """The little the Answer agent must know about the project."""
    parts = []
    if project.description:
        parts.append(f"Project focus: {project.description[:240]}")
    if project.summary:
        parts.append(f"Project memory: {project.summary[:600]}")
    return " ".join(parts)


async def update_summary(project_id: str, question: str, answer: str) -> None:
    """Rolling project memory: one rewritten paragraph."""
    engine = get_engine()
    if not engine.available:
        return
    with db_session() as db:
        project = db.get(Project, project_id)
        if project is None:
            return
        prev = project.summary or "(empty)"
    try:
        out = await engine.complete(
            [
                {"role": "system", "content": MEMORY_SYSTEM},
                {"role": "user", "content": f"MEMORY:\n{prev}\n\nNEW REQUEST:\n{question[:500]}\n\nNEW RESULT:\n{answer[:1500]}"},
            ],
            GenerationParams(max_new_tokens=150, temperature=0.1),
        )
    except Exception as exc:
        log.warning("summary update failed: %s", exc)
        return
    paragraph = re.sub(r"\s*\[\d+\]", "", out.split("\n\n")[0]).strip()
    if len(paragraph) < 20:
        return
    with db_session() as db:
        project = db.get(Project, project_id)
        if project is not None:
            project.summary = paragraph[:1200]
