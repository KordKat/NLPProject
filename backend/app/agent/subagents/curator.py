"""Background agent: grows the Neo4j graph from a finished run's verified paragraph.

Runs after the answer is delivered. Each relation must be supported by one retrieved fact (both of
its concepts appear in that sentence), which also provides its source and evidence quote.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.agent.model_engine import GenerationParams, get_engine
from app.agent.prompts import GRAPH_SHOT_IN, GRAPH_SHOT_OUT, GRAPH_SYSTEM
from app.agent.tool_parser import extract_json
from app.knowledge_graph import graph_writer
from app.knowledge_graph.neo4j_client import GraphUnavailableError, get_graph
from app.knowledge_graph.schema import normalize_relationship

log = logging.getLogger("stemos.curator")


# Two concepts in one sentence prove they are related, not HOW. A specific relation needs its verb in
# that sentence ("RNNs suffer from vanishing gradients" must never become RNN SOLVES vanishing gradients).
_RELATION_CUES = {
    "SOLVES": r"\b(solv|address|overcom|mitigat|resolv|prevent|avoid|fix|counter)",
    "INTRODUCES": r"\b(introduc|propos|present|develop|invent)",
    "EXTENDS": r"\b(extend|generali[sz]|build(s)? (up)?on|improv)",
    "USES_METHOD": r"\b(use|uses|using|employ|rel(y|ies) on|based on|appl(y|ies))\b",
    "USES_DATASET": r"\b(data ?set|corpus|benchmark|imagenet|mnist|cifar)",
    "CONTRADICTS": r"\b(contradict|refut|disagree|inconsistent with)",
    "DERIVED_FROM": r"\b(deriv)",
    "BELONGS_TO": r"\b(is an?|are|type of|kind of|class of|family of|member of)\b",
}


def _checked_relation(relation: str, evidence: str) -> str:
    cue = _RELATION_CUES.get(relation.upper())
    return relation if cue is None or re.search(cue, evidence, re.I) else "RELATED_TO"


def _same_concept(a: str, b: str) -> bool:
    x, y = (set(re.findall(r"[a-z0-9]{3,}", t.lower())) for t in (a, b))
    return x <= y or y <= x


def _appears(term: str, text: str) -> bool:
    """All words of a short term, or two thirds of a longer one ("vanishing gradient problem")."""
    words = re.findall(r"[a-z0-9]{3,}", term.lower())
    low = text.lower()
    return bool(words) and sum(w in low for w in words) >= (len(words) if len(words) < 3 else (2 * len(words) + 2) // 3)


async def curate(*, brief: str, facts: list[dict[str, str]], evidence: list[dict[str, Any]], project_id: str, owner_id: str) -> None:
    """`facts` = [{"text", "source_key"}]; `evidence` = per-source records of the run."""
    graph = get_graph()
    engine = get_engine()
    if not brief or not evidence or not graph.available or not engine.available:
        return
    clean = re.sub(r"\s*\[\d+\]", "", brief)
    out = await engine.complete(
        [
            {"role": "system", "content": GRAPH_SYSTEM},
            {"role": "user", "content": GRAPH_SHOT_IN},
            {"role": "assistant", "content": GRAPH_SHOT_OUT},
            {"role": "user", "content": clean[:1500]},
        ],
        GenerationParams(max_new_tokens=160, temperature=0.0, stop=("]]",)),
        prefill='[["',
    )
    parsed = extract_json(out if out.rstrip().endswith("]]") else out.rstrip().rstrip(",") + "]")
    while isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list) and parsed[0] and isinstance(parsed[0][0], list):
        parsed = parsed[0]  # the model sometimes wraps the list once more
    triples = [t for t in parsed if isinstance(t, list) and len(t) == 3] if isinstance(parsed, list) else []
    by_key = {e["source_key"]: e for e in evidence}
    per_source: dict[str, list[dict[str, Any]]] = {key: [] for key in by_key}
    for subject, relation, obj in triples[:8]:
        subject, obj = str(subject).strip()[:80], str(obj).strip()[:80]
        if not subject or not obj or _same_concept(subject, obj) or normalize_relationship(str(relation)) is None:
            continue
        support = next((f for f in facts if f["source_key"] in by_key and _appears(subject, f["text"]) and _appears(obj, f["text"])), None)
        if support:
            relation = _checked_relation(str(relation), support["text"])
            per_source[support["source_key"]].append({"subject": subject, "relation": relation, "object": obj, "evidence": support["text"][:400], "confidence": 0.7})
    written = 0
    for key, relations in per_source.items():
        ev = by_key[key]
        is_paper = bool(ev.get("paper_id"))
        if not relations and not is_paper:
            continue
        paper = {"paper_id": ev["paper_id"], "title": ev.get("title"), "published": ev.get("published", ""), "url": ev.get("url", ""), "abstract": ev.get("findings", "")} if is_paper else None
        try:
            result = await asyncio.to_thread(
                graph_writer.write_extraction,
                {"concepts": [], "relations": relations, "authors": ev.get("authors", [])},
                project_id=project_id, owner_id=owner_id, paper=paper, private=not is_paper,
            )
            written += result["relationships"]
        except GraphUnavailableError as exc:
            log.warning("graph write skipped: %s", exc)
            return
    log.info("graph curator wrote %d relationships", written)
