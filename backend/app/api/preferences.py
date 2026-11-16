from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.models import User
from app.database.repositories import projects as projects_repo
from app.database.repositories import users as users_repo
from app.database.session import get_db
from app.deps import get_current_user
from app.schemas import CustomPreferenceIn, FeedbackIn, PreferencesOut, PreferencesUpdate
from app.services import personalization

router = APIRouter(prefix="/api", tags=["preferences"])


@router.get("/preferences", response_model=PreferencesOut)
def get_preferences(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return users_repo.get_or_create_preferences(db, user.id)


@router.put("/preferences", response_model=PreferencesOut)
def update_preferences(body: PreferencesUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = users_repo.get_or_create_preferences(db, user.id)
    for k, v in body.model_dump(exclude_unset=True).items():
        if isinstance(v, list):
            v = [str(x).strip()[:120] for x in v if str(x).strip()][:30]
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p


@router.post("/preferences/custom", response_model=PreferencesOut)
def add_custom(body: CustomPreferenceIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = users_repo.get_or_create_preferences(db, user.id)
    custom = list(p.custom or [])
    custom.append({"id": uuid.uuid4().hex[:10], "text": body.text.strip(), "source": "explicit", "created_at": datetime.now(timezone.utc).isoformat()})
    p.custom = custom[-30:]
    db.commit()
    db.refresh(p)
    return p


@router.delete("/preferences/custom/{item_id}", response_model=PreferencesOut)
def delete_custom(item_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = users_repo.get_or_create_preferences(db, user.id)
    p.custom = [c for c in (p.custom or []) if c.get("id") != item_id]
    db.commit()
    db.refresh(p)
    return p


@router.post("/feedback")
async def feedback(body: FeedbackIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from app.database.models import Message

    msg = db.get(Message, body.message_id)
    if msg is None or projects_repo.get_owned(db, msg.project_id, user.id) is None:
        raise HTTPException(status_code=404, detail="Message not found")
    fb = projects_repo.upsert_feedback(db, user.id, msg.project_id, msg.id, body.rating, body.content)
    meta = dict(msg.meta or {})
    meta["feedback"] = {"rating": body.rating, "content": body.content}
    msg.meta = meta
    db.commit()
    learned = None
    if body.rating < 0 and body.content.strip():
        learned = await personalization.maybe_learn(user.id, f"Feedback on an answer I disliked: {body.content}", origin="feedback")
    elif body.rating > 0 and body.content.strip():
        learned = await personalization.maybe_learn(user.id, f"Feedback on an answer I liked: {body.content}", origin="feedback")
    return {"id": fb.id, "rating": fb.rating, "learned": learned}
