"""Pydantic request/response schemas (API contract)."""
from __future__ import annotations

from datetime import datetime as _dt
from datetime import timezone
from typing import Annotated, Any, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field


def _as_utc(d: _dt) -> _dt:
    # SQLite drops tzinfo; all stored timestamps are UTC
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d


datetime = Annotated[_dt, AfterValidator(_as_utc)]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ auth / users

class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=120)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(ORM):
    id: str
    email: str
    display_name: str
    created_at: datetime
    account_settings: dict[str, Any] = {}


class AuthOut(BaseModel):
    user: UserOut
    token: str


class ProfileUpdate(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    account_settings: Optional[dict[str, Any]] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(default=None, min_length=8, max_length=128)


# ------------------------------------------------------------------ preferences

class PreferencesOut(ORM):
    preferred_language: str
    response_style: str
    explanation_depth: str
    theory_vs_practice: str
    math_notation: str
    units: str
    citation_style: str
    preferred_topics: list[str]
    likes: list[str]
    dislikes: list[str]
    formatting: list[str]
    custom: list[dict[str, Any]]
    learning_enabled: bool
    updated_at: datetime


class PreferencesUpdate(BaseModel):
    preferred_language: Optional[str] = Field(default=None, max_length=40)
    response_style: Optional[str] = Field(default=None, max_length=60)
    explanation_depth: Optional[Literal["concise", "balanced", "detailed"]] = None
    theory_vs_practice: Optional[Literal["theoretical", "balanced", "practical"]] = None
    math_notation: Optional[str] = Field(default=None, max_length=60)
    units: Optional[str] = Field(default=None, max_length=30)
    citation_style: Optional[str] = Field(default=None, max_length=30)
    preferred_topics: Optional[list[str]] = None
    likes: Optional[list[str]] = None
    dislikes: Optional[list[str]] = None
    formatting: Optional[list[str]] = None
    learning_enabled: Optional[bool] = None


class CustomPreferenceIn(BaseModel):
    text: str = Field(min_length=2, max_length=400)


# ------------------------------------------------------------------ projects

class ProjectCreate(BaseModel):
    title: str = Field(default="Untitled Project", max_length=200)
    description: str = Field(default="", max_length=4000)


class ProjectUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=4000)
    settings: Optional[dict[str, Any]] = None


class ProjectOut(ORM):
    id: str
    title: str
    description: str
    status: str
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    summary: str = ""
    settings: dict[str, Any] = {}
    context: dict[str, Any] = {}


class ProjectDetail(ProjectOut):
    stats: dict[str, int] = {}


class MessageOut(ORM):
    id: str
    role: str
    content: str
    meta: dict[str, Any] = {}
    created_at: datetime


class SourceOut(ORM):
    id: str
    source_key: str
    source_type: str
    source_url: str
    paper_id: str
    title: str
    meta: dict[str, Any]
    relevance: float
    created_at: datetime


class RunOut(ORM):
    id: str
    user_question: str
    status: str
    task_type: str
    retrieval_rounds: int
    queries: list[dict[str, Any]]
    source_refs: list[str]
    activity: list[dict[str, Any]]
    error: str
    created_at: datetime
    finished_at: Optional[datetime]


class ArtifactOut(BaseModel):
    id: str
    project_id: str
    filename: str
    path: str
    type: str
    mime_type: str
    size_bytes: int
    status: str
    error: str
    meta: dict[str, Any]
    url: str
    download_url: str
    created_at: datetime


# ------------------------------------------------------------------ assistant

class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    research_mode: Literal["auto", "deep", "quick", "off"] = "auto"
    attachments: list[str] = Field(default_factory=list, description="workspace-relative file paths")
    enabled_tools: Optional[list[str]] = None


class ApprovalDecision(BaseModel):
    approve: bool


class FeedbackIn(BaseModel):
    message_id: str
    rating: Literal[1, -1]
    content: str = Field(default="", max_length=2000)
