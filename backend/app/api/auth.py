from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import User
from app.database.repositories import users as users_repo
from app.database.session import get_db
from app.deps import SESSION_COOKIE, extract_token, get_current_user
from app.schemas import AuthOut, LoginIn, ProfileUpdate, RegisterIn, UserOut
from app.security import hash_password, rate_limiter, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
settings = get_settings()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, secure=settings.cookie_secure, samesite="lax",
        max_age=settings.session_ttl_days * 86400, path="/",
    )


@router.post("/register", response_model=AuthOut, status_code=201)
def register(body: RegisterIn, request: Request, response: Response, db: Session = Depends(get_db)):
    rate_limiter.check(f"auth:{_client_ip(request)}", settings.rate_limit_auth_per_minute)
    if users_repo.get_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = users_repo.create(db, body.email, hash_password(body.password), body.display_name)
    token, _ = users_repo.create_session(db, user.id, settings.session_ttl_days, request.headers.get("user-agent", ""))
    _set_cookie(response, token)
    return {"user": UserOut.model_validate(user), "token": token}


@router.post("/login", response_model=AuthOut)
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    rate_limiter.check(f"auth:{_client_ip(request)}", settings.rate_limit_auth_per_minute)
    user = users_repo.get_by_email(db, body.email)
    if user is None or not verify_password(body.password, user.password_hash) or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token, _ = users_repo.create_session(db, user.id, settings.session_ttl_days, request.headers.get("user-agent", ""))
    _set_cookie(response, token)
    return {"user": UserOut.model_validate(user), "token": token}


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = extract_token(request)
    if token:
        users_repo.revoke_token(db, token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.status_code = 204
    return response


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/me", response_model=UserOut)
def update_me(body: ProfileUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    u = db.merge(user)
    if body.display_name is not None:
        u.display_name = body.display_name.strip()
    if body.account_settings is not None:
        merged = dict(u.account_settings or {})
        merged.update(body.account_settings)
        u.account_settings = merged
    if body.new_password:
        if not body.current_password or not verify_password(body.current_password, u.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        u.password_hash = hash_password(body.new_password)
    db.commit()
    db.refresh(u)
    return u
