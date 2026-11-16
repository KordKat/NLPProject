"""Hybrid candidate ranking: vector similarity + graph connectivity + lexical relevance + source quality
(+ recency when the question needs current information)."""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = {"the", "and", "for", "with", "that", "this", "what", "how", "why", "are", "does", "between", "explain", "compare", "about", "from", "into", "their", "using"}


def terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def lexical_overlap(question_terms: set[str], text: str) -> float:
    if not question_terms:
        return 0.0
    t = terms(text)
    return len(question_terms & t) / math.sqrt(len(question_terms) * max(len(t), 1)) if t else 0.0


def recency(date: str) -> float:
    try:
        year = int(date[:4])
    except (ValueError, TypeError):
        return 0.0
    now = datetime.utcnow().year
    return max(0.0, 1.0 - (now - year) / 10.0)


QUALITY = {"arxiv": 1.0, "graph": 0.85, "web_fetched": 0.7, "web_snippet": 0.4}


def score_candidate(c: dict[str, Any], question: str, needs_current: bool) -> float:
    qt = terms(question)
    text = " ".join([c.get("title", ""), c.get("abstract", ""), " ".join(ch.get("text", "")[:600] for ch in c.get("chunks", []))])
    vec = float(c.get("score") or 0.0)
    graph = min(1.0, (c.get("shared_count") or 0) / 4.0) if c.get("via_graph") else 0.0
    lex = lexical_overlap(qt, text)
    kind = c.get("kind", "arxiv")
    quality = QUALITY.get(kind, 0.5)
    s = 0.50 * vec + 0.15 * graph + 0.20 * lex + 0.15 * quality
    if needs_current:
        s += 0.15 * recency(c.get("published", ""))
    return round(s, 4)


def rank(candidates: list[dict[str, Any]], question: str, needs_current: bool, limit: int) -> list[dict[str, Any]]:
    for c in candidates:
        c["hybrid_score"] = score_candidate(c, question, needs_current)
    return sorted(candidates, key=lambda c: c["hybrid_score"], reverse=True)[:limit]
