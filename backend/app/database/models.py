"""SQLAlchemy ORM models for STEM OS application data."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    account_settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    projects: Mapped[list["Project"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    preferences: Mapped[Optional["UserPreferences"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )


class AuthSession(Base):
    """Opaque session tokens (stored hashed) — supports real logout/revocation."""

    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str] = mapped_column(String(300), default="")


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    preferred_language: Mapped[str] = mapped_column(String(40), default="English")
    response_style: Mapped[str] = mapped_column(String(60), default="academic")
    explanation_depth: Mapped[str] = mapped_column(String(30), default="balanced")  # concise|balanced|detailed
    theory_vs_practice: Mapped[str] = mapped_column(String(30), default="balanced")
    math_notation: Mapped[str] = mapped_column(String(60), default="LaTeX")
    units: Mapped[str] = mapped_column(String(30), default="SI")
    citation_style: Mapped[str] = mapped_column(String(30), default="numeric")
    preferred_topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    likes: Mapped[list[str]] = mapped_column(JSON, default=list)
    dislikes: Mapped[list[str]] = mapped_column(JSON, default=list)
    formatting: Mapped[list[str]] = mapped_column(JSON, default=list)
    # free-form preference statements: [{id, text, source: explicit|learned, created_at, evidence}]
    custom: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    learning_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="preferences")


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_user_status_updated", "user_id", "status", "updated_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | archived
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # persistent compressed memory of the project
    summary: Mapped[str] = mapped_column(Text, default="")
    # key decisions / saved concepts: {"decisions": [...], "concepts": [...]}
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # project-level preferences + agent settings
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # neo4j entity uids explored in this project (cache; source of truth is the graph)
    graph_refs: Mapped[list[str]] = mapped_column(JSON, default=list)

    owner: Mapped[User] = relationship(back_populates="projects")
    messages: Mapped[list["Message"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_project_created", "project_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text, default="")
    # activity, tool calls, sources, artifacts, attachments, run_id ...
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="messages")


class ResearchRun(Base):
    __tablename__ = "research_runs"
    __table_args__ = (Index("ix_runs_project_created", "project_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    user_question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running|completed|partial|failed|cancelled
    task_type: Mapped[str] = mapped_column(String(30), default="chat")
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    retrieval_rounds: Mapped[int] = mapped_column(Integer, default=0)
    queries: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    source_refs: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    activity: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (
        UniqueConstraint("project_id", "source_key", name="uq_source_project_key"),
        Index("ix_sources_project_type", "project_id", "source_type"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    source_key: Mapped[str] = mapped_column(String(600))  # arxiv:<id> | web:<url> | graph:<uid>
    source_type: Mapped[str] = mapped_column(String(20))  # arxiv | web | graph | image
    source_url: Mapped[str] = mapped_column(Text, default="")
    paper_id: Mapped[str] = mapped_column(String(64), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    relevance: Mapped[float] = mapped_column(Float, default=0.0)
    first_run_id: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("project_id", "rel_path", name="uq_artifact_project_path"),
        Index("ix_artifacts_project_created", "project_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(300))
    rel_path: Mapped[str] = mapped_column(String(600))  # path inside the project workspace
    artifact_type: Mapped[str] = mapped_column(String(30))  # image|report|pdf|code|data|document|file
    mime_type: Mapped[str] = mapped_column(String(120), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="ready")  # ready|pending|processing|failed
    error: Mapped[str] = mapped_column(Text, default="")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # attribution, run_id, ...
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("user_id", "message_id", name="uq_feedback_user_message"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(Integer)  # +1 / -1
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
