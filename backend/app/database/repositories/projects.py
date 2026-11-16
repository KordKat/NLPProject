from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.database.models import (
    Artifact,
    Feedback,
    Message,
    Project,
    ResearchRun,
    Source,
    utcnow,
)


def get_owned(db: Session, project_id: str, user_id: str) -> Optional[Project]:
    """Return project only if it belongs to user — the core ownership check."""
    return db.scalar(select(Project).where(Project.id == project_id, Project.user_id == user_id))


def list_for_user(db: Session, user_id: str, status: Optional[str] = None, q: str = "", limit: int = 200) -> list[Project]:
    stmt = select(Project).where(Project.user_id == user_id)
    if status:
        stmt = stmt.where(Project.status == status)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(or_(func.lower(Project.title).like(like), func.lower(Project.description).like(like)))
    stmt = stmt.order_by(Project.last_activity_at.desc()).limit(limit)
    return list(db.scalars(stmt))


def create(db: Session, user_id: str, title: str, description: str = "", settings: Optional[dict[str, Any]] = None) -> Project:
    p = Project(user_id=user_id, title=title.strip() or "Untitled Project", description=description.strip(), settings=settings or {})
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def touch(db: Session, project: Project) -> None:
    project.last_activity_at = utcnow()
    db.commit()


def stats(db: Session, project_id: str) -> dict[str, int]:
    def count(model) -> int:
        return int(db.scalar(select(func.count()).select_from(model).where(model.project_id == project_id)) or 0)

    return {
        "messages": count(Message),
        "artifacts": count(Artifact),
        "sources": count(Source),
        "runs": count(ResearchRun),
    }


# ----------------------------------------------------------------- messages

def list_messages(db: Session, project_id: str, limit: int = 500) -> list[Message]:
    stmt = select(Message).where(Message.project_id == project_id).order_by(Message.created_at.asc()).limit(limit)
    return list(db.scalars(stmt))


def recent_messages(db: Session, project_id: str, n: int) -> list[Message]:
    stmt = select(Message).where(Message.project_id == project_id).order_by(Message.created_at.desc()).limit(n)
    return list(reversed(list(db.scalars(stmt))))


def add_message(db: Session, project_id: str, role: str, content: str, meta: Optional[dict] = None) -> Message:
    m = Message(project_id=project_id, role=role, content=content, meta=meta or {})
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


# ----------------------------------------------------------------- sources

def upsert_source(db: Session, project_id: str, run_id: str, src: dict[str, Any]) -> Source:
    key = src["source_key"][:600]
    row = db.scalar(select(Source).where(Source.project_id == project_id, Source.source_key == key))
    if row is None:
        row = Source(
            project_id=project_id,
            source_key=key,
            source_type=src.get("source_type", "web"),
            source_url=src.get("url", "") or "",
            paper_id=src.get("paper_id", "") or "",
            title=src.get("title", "") or "",
            meta=src.get("meta", {}),
            relevance=float(src.get("relevance", 0.0) or 0.0),
            first_run_id=run_id,
        )
        db.add(row)
    else:
        row.relevance = max(row.relevance, float(src.get("relevance", 0.0) or 0.0))
        merged = dict(row.meta or {})
        merged.update(src.get("meta", {}))
        row.meta = merged
    db.commit()
    return row


def list_sources(db: Session, project_id: str, source_type: Optional[str] = None) -> list[Source]:
    stmt = select(Source).where(Source.project_id == project_id)
    if source_type:
        stmt = stmt.where(Source.source_type == source_type)
    return list(db.scalars(stmt.order_by(Source.created_at.desc())))


def search_sources(db: Session, project_id: str, terms: list[str], limit: int = 8) -> list[Source]:
    if not terms:
        return list(db.scalars(select(Source).where(Source.project_id == project_id).order_by(Source.created_at.desc()).limit(limit)))
    conds = [func.lower(Source.title).like(f"%{t.lower()}%") for t in terms if len(t) > 2]
    if not conds:
        return []
    stmt = select(Source).where(Source.project_id == project_id, or_(*conds)).order_by(Source.relevance.desc()).limit(limit)
    return list(db.scalars(stmt))


# ----------------------------------------------------------------- runs

def create_run(db: Session, project_id: str, question: str) -> ResearchRun:
    r = ResearchRun(project_id=project_id, user_question=question)
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def list_runs(db: Session, project_id: str, limit: int = 50) -> list[ResearchRun]:
    stmt = select(ResearchRun).where(ResearchRun.project_id == project_id).order_by(ResearchRun.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))


def recent_evidence(db: Session, project_id: str, limit_runs: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in list_runs(db, project_id, limit_runs):
        for ev in run.evidence or []:
            out.append(ev)
    return out


# ----------------------------------------------------------------- artifacts

def list_artifacts(db: Session, project_id: str, artifact_type: Optional[str] = None) -> list[Artifact]:
    stmt = select(Artifact).where(Artifact.project_id == project_id)
    if artifact_type:
        stmt = stmt.where(Artifact.artifact_type == artifact_type)
    return list(db.scalars(stmt.order_by(Artifact.created_at.desc())))


def get_artifact_owned(db: Session, artifact_id: str, user_id: str) -> Optional[Artifact]:
    stmt = (
        select(Artifact)
        .join(Project, Project.id == Artifact.project_id)
        .where(Artifact.id == artifact_id, Project.user_id == user_id)
    )
    return db.scalar(stmt)


def artifact_by_path(db: Session, project_id: str, rel_path: str) -> Optional[Artifact]:
    return db.scalar(select(Artifact).where(Artifact.project_id == project_id, Artifact.rel_path == rel_path))


# ----------------------------------------------------------------- feedback

def upsert_feedback(db: Session, user_id: str, project_id: str, message_id: str, rating: int, content: str) -> Feedback:
    fb = db.scalar(select(Feedback).where(Feedback.user_id == user_id, Feedback.message_id == message_id))
    if fb is None:
        fb = Feedback(user_id=user_id, project_id=project_id, message_id=message_id, rating=rating, content=content)
        db.add(fb)
    else:
        fb.rating = rating
        fb.content = content
    db.commit()
    db.refresh(fb)
    return fb
