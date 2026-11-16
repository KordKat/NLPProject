"""Artifacts + project workspace files (ownership-checked, safe delivery)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import Project, User
from app.database.repositories import projects as repo
from app.database.session import get_db
from app.deps import get_current_user, get_owned_project
from app.services import pdf_converter
from app.services.artifact_storage import (
    WorkspaceError,
    artifact_to_dict,
    classify,
    guess_mime,
    register_artifact,
    resolve_in_workspace,
    workspace_dir,
)

router = APIRouter(tags=["artifacts"])
settings = get_settings()

SAFE_INLINE = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf", "text/plain", "text/markdown", "text/csv", "application/json"}
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; sandbox", "Cache-Control": "private, max-age=300"}


def _serve(full: Path, filename: str, mime: str, download: bool) -> FileResponse:
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    headers = dict(SECURITY_HEADERS)
    # anything that could execute script in the browser (html, svg, js…) is only ever sent as a download
    if download or mime not in SAFE_INLINE:
        return FileResponse(full, media_type="application/octet-stream" if mime not in SAFE_INLINE else mime, filename=filename, headers=headers)
    if mime == "application/pdf":
        # browsers refuse to render PDFs under a sandbox CSP; PDFs here are produced by our converter
        headers["Content-Security-Policy"] = "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; object-src 'self'"
    headers["Content-Disposition"] = f'inline; filename="{filename}"'
    return FileResponse(full, media_type=mime, headers=headers)


@router.get("/api/projects/{project_id}/artifacts")
def list_artifacts(type: Optional[str] = None, project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    return [artifact_to_dict(a) for a in repo.list_artifacts(db, project.id, type)]


@router.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = repo.get_artifact_owned(db, artifact_id, user.id)
    if a is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    full, _ = resolve_in_workspace(a.project_id, a.rel_path)
    return _serve(full, a.filename, a.mime_type, download=False)


@router.get("/api/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = repo.get_artifact_owned(db, artifact_id, user.id)
    if a is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    full, _ = resolve_in_workspace(a.project_id, a.rel_path)
    return _serve(full, a.filename, a.mime_type, download=True)


@router.get("/api/artifacts/{artifact_id}/meta")
def artifact_meta(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = repo.get_artifact_owned(db, artifact_id, user.id)
    if a is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact_to_dict(a)


@router.get("/api/artifacts/{artifact_id}/content")
def artifact_content(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = repo.get_artifact_owned(db, artifact_id, user.id)
    if a is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    full, _ = resolve_in_workspace(a.project_id, a.rel_path)
    if not full.exists():
        raise HTTPException(status_code=404, detail="File missing")
    if a.artifact_type in ("image", "pdf") or full.stat().st_size > 2_000_000:
        raise HTTPException(status_code=415, detail="Not a text artifact")
    return {"artifact": artifact_to_dict(a), "content": full.read_text(encoding="utf-8", errors="replace")}


@router.post("/api/artifacts/{artifact_id}/convert-pdf")
async def reconvert(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = repo.get_artifact_owned(db, artifact_id, user.id)
    if a is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    src = a.rel_path if a.rel_path.endswith(".md") else (a.meta or {}).get("source_markdown")
    if not src:
        raise HTTPException(status_code=400, detail="No Markdown source for this artifact")
    out = src[:-3] + ".pdf"
    return pdf_converter.start_conversion(a.project_id, src, out)


# ------------------------------------------------------------------ workspace files

@router.get("/api/projects/{project_id}/files")
def list_files(project: Project = Depends(get_owned_project)):
    ws = workspace_dir(project.id).resolve()
    out = []
    for p in sorted(ws.rglob("*")):
        rel = p.relative_to(ws)
        if ".stemos" in rel.parts or not p.is_file():
            continue
        r = rel.as_posix()
        out.append({"path": r, "name": p.name, "type": classify(r), "mime_type": guess_mime(r), "size_bytes": p.stat().st_size, "modified": p.stat().st_mtime})
        if len(out) >= 1000:
            break
    return out


@router.get("/api/projects/{project_id}/files/raw")
def raw_file(path: str = Query(..., max_length=600), download: bool = False, project: Project = Depends(get_owned_project)):
    try:
        full, rel = resolve_in_workspace(project.id, path)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _serve(full, Path(rel).name, guess_mime(rel), download)


@router.post("/api/projects/{project_id}/files/upload", status_code=201)
async def upload_file(file: UploadFile = File(...), folder: str = "uploads", project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    name = Path(file.filename or "upload.bin").name.replace("..", "_")[:150] or "upload.bin"
    try:
        full, rel = resolve_in_workspace(project.id, f"{folder.strip('/')}/{name}" if folder else name)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    limit = settings.max_upload_mb * 1024 * 1024
    full.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with full.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                full.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")
            f.write(chunk)
    a = register_artifact(db, project.id, rel, meta={"origin": "upload"})
    return artifact_to_dict(a)
