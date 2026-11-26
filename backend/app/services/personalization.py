"""Personalization: preference rendering for the system prompt and preference learning.

Learning is conservative: only messages that explicitly express a lasting preference ("from now on",
"always", "I prefer", "I don't like" …) or explicit dislike feedback with a comment are analysed, so
temporary instructions are not turned into permanent preferences.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.agent.model_engine import GenerationParams, get_engine
from app.agent.tool_parser import extract_json
from app.database.models import Project, UserPreferences
from app.database.repositories import users as users_repo
from app.database.session import db_session

log = logging.getLogger("stemos.personalization")

PERSISTENT_MARKERS = re.compile(
    r"\b(from now on|always|never|in (the )?future|i (generally |usually )?prefer|i (really )?(like|love|hate|dislike)|"
    r"i don'?t like|please stop|stop using|i want you to always|remember that i|my preference)\b",
    re.I,
)
DEPTH_HINTS = {
    "detailed": re.compile(r"\b(more detail|go deeper|more rigorous|full derivation|derive it|step[- ]by[- ]step)\b", re.I),
    "concise": re.compile(r"\b(shorter|too long|be concise|tl;?dr|briefly)\b", re.I),
}


def preferences_text(user_id: str, project: Optional[Project] = None) -> str:
    with db_session() as db:
        p = users_repo.get_or_create_preferences(db, user_id)
        # only what differs from the defaults: unchanged settings tell the model nothing
        lines = [
            text
            for value, default, text in (
                (p.preferred_language, "English", f"- Respond in {p.preferred_language}."),
                (p.response_style, "academic", f"- Writing style: {p.response_style}."),
                (p.explanation_depth, "balanced", f"- Explanation depth: {p.explanation_depth}."),
                (p.theory_vs_practice, "balanced", f"- Orientation: {p.theory_vs_practice}."),
                (p.math_notation, "LaTeX", f"- Math notation: {p.math_notation}."),
                (p.units, "SI", f"- Units: {p.units}."),
                (p.citation_style, "numeric", f"- Citation style: {p.citation_style}."),
            )
            if value != default
        ]
        if p.likes:
            lines.append(f"- Likes: {'; '.join(p.likes[:10])}.")
        if p.dislikes:
            lines.append(f"- Dislikes (avoid): {'; '.join(p.dislikes[:10])}.")
        if p.formatting:
            lines.append(f"- Formatting: {'; '.join(p.formatting[:10])}.")
        for c in (p.custom or [])[:15]:
            lines.append(f"- {c.get('text')}")
    if project is not None:
        proj_prefs = (project.settings or {}).get("preferences") or {}
        if proj_prefs:
            lines.append("Project-specific preferences (take precedence over the general ones above):")
            for k, v in proj_prefs.items():
                if v not in (None, "", []):
                    lines.append(f"- {k.replace('_', ' ')}: {v if not isinstance(v, list) else ', '.join(map(str, v))}")
    return "\n".join(lines)


EXTRACT_PROMPT = """Decide whether the user's message states a LASTING personal preference about how the assistant should respond
(language, depth, style, notation, units, citation style, topics they like/dislike, formatting).
Temporary, one-off requests ("explain this one briefly") are NOT lasting preferences.
Return ONLY JSON: {"lasting": true|false, "preference": "<short imperative statement, e.g. 'Avoid analogies; prefer formal derivations.'>",
"field": "explanation_depth|response_style|preferred_language|math_notation|units|citation_style|likes|dislikes|formatting|custom",
"value": "<value for the field>"}"""


async def maybe_learn(user_id: str, text: str, origin: str = "message") -> Optional[dict[str, Any]]:
    with db_session() as db:
        prefs = users_repo.get_or_create_preferences(db, user_id)
        if not prefs.learning_enabled:
            return None
    if origin == "message" and not PERSISTENT_MARKERS.search(text):
        return None
    engine = get_engine()
    if not engine.available:
        return None
    try:
        out = await engine.complete(
            [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": text[:1500]}],
            GenerationParams(max_new_tokens=200, temperature=0.1),
        )
    except Exception as exc:
        log.debug("preference extraction failed: %s", exc)
        return None
    data = extract_json(out)
    if not isinstance(data, dict) or not data.get("lasting") or not data.get("preference"):
        return None
    return apply_learned(user_id, data, evidence=text[:300], origin=origin)


def apply_learned(user_id: str, data: dict[str, Any], evidence: str, origin: str) -> dict[str, Any]:
    field = str(data.get("field", "custom"))
    value = str(data.get("value", "") or "").strip()
    statement = str(data["preference"]).strip()[:300]
    with db_session() as db:
        p: UserPreferences = users_repo.get_or_create_preferences(db, user_id)
        if field == "explanation_depth" and value in ("concise", "balanced", "detailed"):
            p.explanation_depth = value
        elif field in ("likes", "dislikes", "formatting") and value:
            cur = list(getattr(p, field) or [])
            if value not in cur:
                setattr(p, field, (cur + [value])[-20:])
        elif field in ("preferred_language", "response_style", "math_notation", "units", "citation_style") and value and len(value) < 40:
            setattr(p, field, value)
        else:
            custom = list(p.custom or [])
            if not any(c.get("text", "").lower() == statement.lower() for c in custom):
                custom.append({"id": uuid.uuid4().hex[:10], "text": statement, "source": "learned", "origin": origin, "evidence": evidence, "created_at": datetime.now(timezone.utc).isoformat()})
                p.custom = custom[-30:]
    return {"field": field, "value": value, "statement": statement}


def nudge_depth(user_id: str, text: str) -> None:
    """Lightweight signal: repeated explicit requests for more/less depth shift the default."""
    for depth, rx in DEPTH_HINTS.items():
        if rx.search(text):
            with db_session() as db:
                p = users_repo.get_or_create_preferences(db, user_id)
                user = users_repo.get(db, user_id)
                if not p.learning_enabled or user is None:
                    return
                acct = dict(user.account_settings or {})
                counts = dict(acct.get("depth_signals", {}))
                counts[depth] = counts.get(depth, 0) + 1
                if counts[depth] >= 3:
                    if p.explanation_depth != depth:
                        p.explanation_depth = depth
                    counts[depth] = 0
                acct["depth_signals"] = counts
                user.account_settings = acct
            return
