"""Skill library management (per user): list, view, add, remove/hide, restore, test retrieval."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.models import User
from app.database.session import get_db
from app.deps import get_current_user
from app.services.skills import SkillError, skill_store
from app.tools.registry import registry

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillIn(BaseModel):
    name: str = Field(min_length=3, max_length=50)
    content: str = Field(min_length=40, max_length=40_000, description="Markdown instructions (front matter optional)")
    title: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=400)
    triggers: Optional[list[str]] = None
    tools: Optional[list[str]] = None
    overwrite: bool = False


def _hidden(user: User) -> list[str]:
    return list((user.account_settings or {}).get("hidden_skills", []))


def _set_hidden(db: Session, user: User, hidden: list[str]) -> None:
    s = dict(user.account_settings or {})
    s["hidden_skills"] = sorted(set(hidden))
    user.account_settings = s
    db.commit()


@router.get("")
def list_skills(user: User = Depends(get_current_user)):
    return [s.summary() for s in skill_store.list_for_user(user.id, _hidden(user))]


@router.get("/match")
async def match_skills(q: str = Query(..., min_length=2, max_length=2000), user: User = Depends(get_current_user)):
    """Which skills the agent would retrieve for this request (same retrieval the agent uses)."""
    skills = await asyncio.to_thread(skill_store.retrieve, user.id, q, _hidden(user), None)
    return [{**s.summary(), "score": round(s.score, 3), "reasons": s.reasons} for s in skills]


@router.get("/{skill_id}")
def get_skill(skill_id: str, user: User = Depends(get_current_user)):
    s = skill_store.get(user.id, skill_id, _hidden(user))
    if s is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {**s.summary(), "content": s.path.read_text(encoding="utf-8")}


@router.post("", status_code=201)
def add_skill(body: SkillIn, user: User = Depends(get_current_user)):
    tools = None
    if body.tools is not None:
        known = set(registry.names())
        unknown = [t for t in body.tools if t not in known]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown tools: {', '.join(unknown)}. Available: {', '.join(sorted(known))}")
        tools = body.tools
    try:
        s = skill_store.add(
            user.id, name=body.name, content=body.content, title=body.title, description=body.description,
            triggers=[t.strip() for t in body.triggers if t.strip()] if body.triggers is not None else None,
            tools=tools, overwrite=body.overwrite,
        )
    except SkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return s.summary()


@router.delete("/{skill_id}")
def remove_skill(skill_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    u = db.merge(user)
    source, _, name = skill_id.partition(":")
    if source == "user":
        if not skill_store.delete_user_skill(u.id, name):
            raise HTTPException(status_code=404, detail="Skill not found")
        return {"removed": skill_id, "mode": "deleted"}
    if source == "builtin":
        if skill_store.get(u.id, skill_id, []) is None:
            raise HTTPException(status_code=404, detail="Skill not found")
        _set_hidden(db, u, _hidden(u) + [skill_id])
        return {"removed": skill_id, "mode": "hidden"}
    raise HTTPException(status_code=400, detail="Invalid skill id")


@router.post("/{skill_id}/restore")
def restore_skill(skill_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    u = db.merge(user)
    _set_hidden(db, u, [h for h in _hidden(u) if h != skill_id])
    return {"restored": skill_id}
