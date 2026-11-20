"""Skill library + retrieval (inspired by Claude Skills).

A skill is a Markdown file with a small front-matter header:

    ---
    name: pptx
    title: PowerPoint presentations with python-pptx
    description: One or two sentences saying what the skill teaches and when it applies.
    triggers: powerpoint, pptx, slides, presentation
    tools: run_python, list_files
    ---
    # Instructions …

* Built-in skills live in the repo-level `skills/` folder (shared, read-only from the UI).
* Each user's own skills live in `storage/skills/<user_id>/` (private to that user).
* A user can hide a built-in skill ("remove") and restore it later; hidden skills are never retrieved.

Retrieval never loads the whole library into the prompt: only the catalog line (name + description) is
compared with the request (MiniLM embedding similarity + trigger keywords + the planner's own picks),
and the full text of the top few matching skills is injected.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import get_settings

log = logging.getLogger("stemos.skills")
settings = get_settings()

MAX_SKILL_BYTES = 40_000
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{1,48}[a-z0-9]$")
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


class SkillError(ValueError):
    pass


@dataclass
class Skill:
    id: str  # "builtin:<name>" | "user:<name>"
    name: str
    title: str
    description: str
    triggers: list[str]
    tools: list[str]
    source: str  # builtin | user
    path: Path
    body: str = ""
    hidden: bool = False
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "id": self.id, "name": self.name, "title": self.title, "description": self.description,
            "triggers": self.triggers, "tools": self.tools, "source": self.source, "hidden": self.hidden,
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:50]


def parse_skill(text: str, fallback_name: str) -> dict:
    meta: dict[str, str] = {}
    body = text
    m = _FM.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
        body = text[m.end():]
    name = slugify(meta.get("name") or fallback_name)
    title = meta.get("title") or ""
    if not title:
        h = re.search(r"^#\s+(.+)$", body, re.M)
        title = h.group(1).strip() if h else name.replace("-", " ").title()
    desc = meta.get("description") or ""
    if not desc:
        para = next((p.strip() for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("#")), "")
        desc = re.sub(r"\s+", " ", para)[:240]
    split = lambda s: [x.strip() for x in re.split(r"[,;]", s or "") if x.strip()]  # noqa: E731
    return {"name": name, "title": title[:120], "description": desc[:400], "triggers": split(meta.get("triggers", ""))[:20], "tools": split(meta.get("tools", ""))[:10], "body": body.strip()}


def render_skill_file(name: str, title: str, description: str, triggers: list[str], tools: list[str], body: str) -> str:
    def clean(s: str) -> str:
        return re.sub(r"[\r\n]+", " ", s).strip()

    return (
        "---\n"
        f"name: {name}\n"
        f"title: {clean(title)}\n"
        f"description: {clean(description)}\n"
        f"triggers: {', '.join(clean(t) for t in triggers)}\n"
        f"tools: {', '.join(clean(t) for t in tools)}\n"
        "---\n\n" + body.strip() + "\n"
    )


class SkillStore:
    def __init__(self) -> None:
        self._emb_cache: dict[tuple[str, float], np.ndarray] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ discovery
    def _user_dir(self, user_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", user_id):
            raise SkillError("invalid user")
        d = settings.user_skills_path / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load(self, path: Path, source: str) -> Optional[Skill]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("cannot read skill %s: %s", path, exc)
            return None
        d = parse_skill(text, path.stem)
        return Skill(id=f"{source}:{d['name']}", name=d["name"], title=d["title"], description=d["description"], triggers=d["triggers"], tools=d["tools"], source=source, path=path, body=d["body"])

    def list_for_user(self, user_id: str, hidden: Optional[list[str]] = None, include_hidden: bool = True) -> list[Skill]:
        hidden_set = set(hidden or [])
        out: dict[str, Skill] = {}
        bdir = settings.skills_path
        if bdir.exists():
            for p in sorted(bdir.glob("*.md")):
                if p.name.lower() == "readme.md":
                    continue
                s = self._load(p, "builtin")
                if s:
                    s.hidden = s.id in hidden_set
                    out[s.name] = s
        for p in sorted(self._user_dir(user_id).glob("*.md")):
            s = self._load(p, "user")
            if s:
                out[s.name] = s  # a user skill with the same name overrides the built-in one
        skills = list(out.values())
        return skills if include_hidden else [s for s in skills if not s.hidden]

    def get(self, user_id: str, skill_id: str, hidden: Optional[list[str]] = None) -> Optional[Skill]:
        return next((s for s in self.list_for_user(user_id, hidden) if s.id == skill_id or s.name == skill_id), None)

    # ------------------------------------------------------------------ management
    def add(self, user_id: str, *, name: str, content: str, title: str = "", description: str = "", triggers: Optional[list[str]] = None, tools: Optional[list[str]] = None, overwrite: bool = False) -> Skill:
        if len(content.encode("utf-8")) > MAX_SKILL_BYTES:
            raise SkillError(f"Skill is too large (max {MAX_SKILL_BYTES // 1000} KB)")
        d = parse_skill(content, name or "skill")
        slug = slugify(name) if name else d["name"]
        if not NAME_RE.match(slug):
            raise SkillError("Name must be 3-50 characters: lowercase letters, digits and hyphens")
        body = d["body"]
        if len(body) < 40:
            raise SkillError("Skill instructions are too short — write real guidance for the agent")
        final = render_skill_file(
            slug,
            title or d["title"],
            description or d["description"],
            triggers if triggers is not None else d["triggers"],
            tools if tools is not None else d["tools"],
            body,
        )
        path = self._user_dir(user_id) / f"{slug}.md"
        if path.exists() and not overwrite:
            raise SkillError(f"You already have a skill named '{slug}'")
        path.write_text(final, encoding="utf-8")
        s = self._load(path, "user")
        assert s is not None
        return s

    def delete_user_skill(self, user_id: str, name: str) -> bool:
        path = self._user_dir(user_id) / f"{slugify(name)}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------ retrieval
    def _embed(self, skill: Skill) -> Optional[np.ndarray]:
        try:
            from app.retrieval.embedding import get_encoder

            key = (str(skill.path), skill.path.stat().st_mtime)
            with self._lock:
                if key in self._emb_cache:
                    return self._emb_cache[key]
            text = f"{skill.title}. {skill.description} Keywords: {', '.join(skill.triggers)}"
            vec = get_encoder().encode([text])[0]
            with self._lock:
                self._emb_cache[key] = vec
            return vec
        except Exception as exc:  # encoder unavailable → keyword-only retrieval
            log.debug("skill embedding unavailable: %s", exc)
            return None

    def retrieve(self, user_id: str, request: str, hidden: Optional[list[str]] = None, planner_picks: Optional[list[str]] = None, k: Optional[int] = None) -> list[Skill]:
        k = k or settings.skills_max_per_request
        skills = self.list_for_user(user_id, hidden, include_hidden=False)
        if not skills:
            return []
        low = request.lower()
        picks = {slugify(p) for p in (planner_picks or []) if isinstance(p, str)}
        qvec = None
        try:
            from app.retrieval.embedding import get_encoder

            qvec = get_encoder().encode([request[:1000]])[0]
        except Exception:
            pass
        for s in skills:
            s.score, s.reasons = 0.0, []
            if qvec is not None:
                v = self._embed(s)
                if v is not None:
                    sim = float(np.dot(qvec, v))
                    s.score += sim
                    s.reasons.append(f"similarity {sim:.2f}")
            hits = [t for t in s.triggers if re.search(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", low)]
            if hits:
                s.score += 0.25 + 0.05 * min(len(hits), 3)
                s.reasons.append("keywords: " + ", ".join(hits[:3]))
            if s.name in picks:
                # a planner choice counts strongly, but only with some support in the request itself
                # (keyword hit or non-trivial similarity) — protects against a confused plan
                supported = bool(hits) or (qvec is not None and s.score >= 0.12)
                s.score += 0.40 if supported else 0.10
                s.reasons.append("chosen by planner")
        ranked = sorted((s for s in skills if s.score >= settings.skills_min_score), key=lambda s: s.score, reverse=True)
        return ranked[:k]

    def catalog(self, user_id: str, hidden: Optional[list[str]] = None) -> list[tuple[str, str]]:
        return [(s.name, s.description) for s in self.list_for_user(user_id, hidden, include_hidden=False)]


skill_store = SkillStore()


def hidden_skills_for(user_id: str) -> list[str]:
    from app.database.repositories import users as users_repo
    from app.database.session import db_session

    with db_session() as db:
        u = users_repo.get(db, user_id)
        return list(((u.account_settings or {}) if u else {}).get("hidden_skills", []))
