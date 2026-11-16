"""Per-project workspace storage + artifact registration.

Each project owns an isolated workspace directory: storage/projects/<project_id>/workspace.
All file paths coming from users or from the model are resolved through `resolve_in_workspace`
which rejects traversal, absolute paths and symlink escapes.
"""
from __future__ import annotations

import mimetypes
import shutil
from datetime import timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import Artifact, utcnow
from app.database.repositories import projects as repo

settings = get_settings()

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
CODE_EXT = {".py", ".sh", ".js", ".ts", ".ipynb", ".c", ".cpp", ".r", ".jl", ".m", ".tex"}
DATA_EXT = {".csv", ".json", ".tsv", ".parquet", ".npy", ".txt", ".xlsx"}

mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("image/webp", ".webp")


class WorkspaceError(ValueError):
    pass


def project_root(project_id: str) -> Path:
    return settings.projects_path / project_id


def workspace_dir(project_id: str) -> Path:
    ws = project_root(project_id) / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def delete_project_storage(project_id: str) -> None:
    root = project_root(project_id)
    if root.exists() and root.resolve().is_relative_to(settings.projects_path.resolve()):
        shutil.rmtree(root, ignore_errors=True)


def normalize_rel_path(path: str) -> str:
    raw = (path or "").replace("\\", "/").strip()
    if raw in ("", ".", "./", "/"):
        return ""
    if raw.startswith("workspace/"):
        raw = raw[len("workspace/"):]
    p = PurePosixPath(raw)
    if p.is_absolute() or (len(raw) > 1 and raw[1] == ":"):
        raise WorkspaceError("Absolute paths are not allowed; use a path relative to the project workspace.")
    parts = [x for x in p.parts if x not in ("", ".")]
    if any(x == ".." for x in parts):
        raise WorkspaceError("Path traversal ('..') is not allowed.")
    if parts and parts[0] == ".stemos":
        raise WorkspaceError("The .stemos directory is reserved.")
    return "/".join(parts)


def resolve_in_workspace(project_id: str, path: str) -> tuple[Path, str]:
    ws = workspace_dir(project_id).resolve()
    rel = normalize_rel_path(path)
    full = (ws / rel).resolve() if rel else ws
    if not (full == ws or full.is_relative_to(ws)):
        raise WorkspaceError("Path escapes the project workspace.")
    return full, rel


def classify(path: str) -> str:
    ext = Path(path).suffix.lower()
    name = Path(path).name.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext == ".md":
        return "report" if "report" in name else "document"
    if ext in CODE_EXT:
        return "code"
    if ext in DATA_EXT:
        return "data"
    return "file"


def guess_mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def artifact_to_dict(a: Artifact) -> dict[str, Any]:
    return {
        "id": a.id,
        "project_id": a.project_id,
        "filename": a.filename,
        "path": a.rel_path,
        "type": a.artifact_type,
        "name": a.filename,
        "mime_type": a.mime_type,
        "size_bytes": a.size_bytes,
        "status": a.status,
        "error": a.error or "",
        "meta": a.meta or {},
        "url": f"/api/artifacts/{a.id}",
        "download_url": f"/api/artifacts/{a.id}/download",
        "created_at": (a.created_at if a.created_at.tzinfo else a.created_at.replace(tzinfo=timezone.utc)).isoformat() if a.created_at else None,
    }


def register_artifact(
    db: Session,
    project_id: str,
    rel_path: str,
    *,
    artifact_type: Optional[str] = None,
    status: str = "ready",
    meta: Optional[dict[str, Any]] = None,
    error: str = "",
) -> Artifact:
    """Create or update the artifact row for a workspace file (idempotent by path)."""
    full, rel = resolve_in_workspace(project_id, rel_path)
    size = full.stat().st_size if full.exists() and full.is_file() else 0
    a = repo.artifact_by_path(db, project_id, rel)
    if a is None:
        a = Artifact(
            project_id=project_id,
            filename=Path(rel).name,
            rel_path=rel,
            artifact_type=artifact_type or classify(rel),
            mime_type=guess_mime(rel),
            size_bytes=size,
            status=status,
            error=error,
            meta=meta or {},
        )
        db.add(a)
    else:
        a.size_bytes = size
        a.status = status
        a.error = error
        if artifact_type:
            a.artifact_type = artifact_type
        if meta:
            merged = dict(a.meta or {})
            merged.update(meta)
            a.meta = merged
        a.updated_at = utcnow()
    db.commit()
    db.refresh(a)
    return a


def unregister_path(db: Session, project_id: str, rel_path: str) -> None:
    a = repo.artifact_by_path(db, project_id, rel_path)
    if a is not None:
        db.delete(a)
        db.commit()


def snapshot(project_id: str) -> dict[str, float]:
    """mtime snapshot of the workspace, used to detect files produced by code execution."""
    ws = workspace_dir(project_id)
    out: dict[str, float] = {}
    for p in ws.rglob("*"):
        if p.is_file() and ".stemos" not in p.parts:
            try:
                out[p.relative_to(ws).as_posix()] = p.stat().st_mtime
            except OSError:
                pass
    return out


def changed_files(before: dict[str, float], after: dict[str, float]) -> list[str]:
    return sorted(k for k, v in after.items() if k not in before or before[k] != v)
