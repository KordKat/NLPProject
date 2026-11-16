from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.database.models import AuthSession, User, UserPreferences


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_by_email(db: Session, email: str) -> Optional[User]:
    return db.scalar(select(User).where(User.email == email.lower().strip()))


def get(db: Session, user_id: str) -> Optional[User]:
    return db.get(User, user_id)


def create(db: Session, email: str, password_hash: str, display_name: str) -> User:
    user = User(email=email.lower().strip(), password_hash=password_hash, display_name=display_name.strip())
    db.add(user)
    db.flush()
    db.add(UserPreferences(user_id=user.id))
    db.commit()
    db.refresh(user)
    return user


def get_or_create_preferences(db: Session, user_id: str) -> UserPreferences:
    prefs = db.get(UserPreferences, user_id)
    if prefs is None:
        prefs = UserPreferences(user_id=user_id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return prefs


def create_session(db: Session, user_id: str, ttl_days: int, user_agent: str = "") -> tuple[str, AuthSession]:
    token = secrets.token_urlsafe(48)
    sess = AuthSession(
        user_id=user_id,
        token_hash=_hash_token(token),
        expires_at=datetime.now(timezone.utc) + timedelta(days=ttl_days),
        user_agent=user_agent[:300],
    )
    db.add(sess)
    db.commit()
    return token, sess


def user_for_token(db: Session, token: str) -> Optional[User]:
    sess = db.scalar(select(AuthSession).where(AuthSession.token_hash == _hash_token(token)))
    if sess is None:
        return None
    expires = sess.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        db.delete(sess)
        db.commit()
        return None
    user = db.get(User, sess.user_id)
    if user is None or not user.is_active:
        return None
    return user


def revoke_token(db: Session, token: str) -> None:
    db.execute(delete(AuthSession).where(AuthSession.token_hash == _hash_token(token)))
    db.commit()
