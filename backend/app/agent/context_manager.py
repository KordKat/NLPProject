"""Conversation context for the Answer agent: short, labelled history. Everything else is routed
through the blackboard (agent/runtime.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class PreviousTurn:
    request: str = ""
    answer: str = ""
    status: str = "failed"
    attachments: list[str] = field(default_factory=list)


def previous_turn(project_id: str) -> PreviousTurn:
    """Find the latest prior request and the reply that belongs to it."""
    with db_session() as db:
        msgs = repo.recent_messages(db, project_id, 12)
    if msgs and msgs[-1].role == "user":
        msgs = msgs[:-1]
    for i in range(len(msgs) - 1, -1, -1):
        user = msgs[i]
        if user.role != "user":
            continue
        reply = msgs[i + 1] if i + 1 < len(msgs) and msgs[i + 1].role == "assistant" else None
        meta = (reply.meta or {}) if reply is not None else {}
        answer = (reply.content or "") if reply is not None else ""
        status = meta.get("status", "completed" if answer.strip() else "failed")
        if status == "completed" and (not answer.strip() or ("outcomes" not in meta and "> **Note:**" in answer)):
            status = "failed"
        return PreviousTurn(meta.get("effective_request") or user.content or "", answer, status,
                            list(meta.get("effective_attachments") or (user.meta or {}).get("attachments") or []))
    return PreviousTurn()
