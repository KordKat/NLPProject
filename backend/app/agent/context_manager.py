"""Conversation context for the Answer agent: short, labelled history. Everything else is routed
through the blackboard (agent/runtime.py)."""
from __future__ import annotations

from typing import Any

from app.agent.model_engine import get_engine
from app.database.repositories import projects as repo
from app.database.session import db_session

EARLIER_ANSWER_TAG = "[Earlier answer, already delivered. Do not repeat it.]"


def _assistant_digest(m: Any, max_tokens: int) -> str:
    """A small model copies full earlier answers verbatim, so only a short labelled digest is kept."""
    eng = get_engine()
    meta = m.meta or {}
    status = meta.get("status") or "completed"
    if status in ("failed", "cancelled") or not (m.content or "").strip():
        return f"[Earlier turn: no answer was delivered, the run was {status if status != 'completed' else 'empty'}.]"
    files = [a.get("path") for a in meta.get("artifacts", []) if a.get("path")]
    tail = f"\n[Files produced: {', '.join(files[:8])}]" if files else ""
    return f"{EARLIER_ANSWER_TAG}\n{eng.truncate_to_tokens(m.content, max_tokens)}{tail}"


def history_messages(project_id: str, n: int = 4) -> list[dict[str, str]]:
    """The last `n` messages before the current request; the latest exchange is kept longer."""
    eng = get_engine()
    with db_session() as db:
        msgs = repo.recent_messages(db, project_id, n + 1)
    if msgs and msgs[-1].role == "user":
        msgs = msgs[:-1]
    msgs = msgs[-n:]
    out: list[dict[str, str]] = []
    last = len(msgs) - 1
    for i, m in enumerate(msgs):
        recent = i >= last - 1
        if m.role == "assistant":
            out.append({"role": "assistant", "content": _assistant_digest(m, 220 if recent else 80)})
        elif m.role == "user":
            out.append({"role": "user", "content": eng.truncate_to_tokens(m.content or "", 220 if recent else 80)})
    return out


def previous_exchanges(project_id: str, n: int = 6) -> list[tuple[str, str]]:
    """(user request, assistant answer) pairs of earlier turns, used to detect repeated answers."""
    with db_session() as db:
        msgs = repo.recent_messages(db, project_id, n * 2 + 1)
    return [(p.content or "", c.content) for p, c in zip(msgs, msgs[1:]) if p.role == "user" and c.role == "assistant" and (c.content or "").strip()]


def previous_turn(project_id: str) -> tuple[str, str, bool]:
    """What "continue" refers to: (previous request, its answer, whether that turn finished its work)."""
    with db_session() as db:
        msgs = repo.recent_messages(db, project_id, 12)
    if msgs and msgs[-1].role == "user":
        msgs = msgs[:-1]
    answer = next((m for m in reversed(msgs) if m.role == "assistant"), None)
    request = next((m.content or "" for m in reversed(msgs) if m.role == "user"), "")
    if answer is None:
        return request, "", False
    finished = (answer.meta or {}).get("status") == "completed" and bool((answer.content or "").strip()) and "> **Note:**" not in answer.content
    return request, answer.content or "", finished
