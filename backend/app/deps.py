"""FastAPI dependencies: DB session, current user, owned project."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database.models import Project, User
from app.database.repositories import projects as projects_repo
from app.database.repositories import users as users_repo
from app.database.session import get_db

SESSION_COOKIE = "stemos_session"


def extract_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return request.cookies.get(SESSION_COOKIE)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = extract_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = users_repo.user_for_token(db, token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid")
    return user


def get_owned_project(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Project:
    project = projects_repo.get_owned(db, project_id, user.id)
    if project is None:
        # 404 rather than 403 so project ids of other users are not disclosed
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project
