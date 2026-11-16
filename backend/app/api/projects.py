from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.models import Project, User, utcnow
from app.database.repositories import projects as repo
from app.database.session import get_db
from app.deps import get_current_user, get_owned_project
from app.schemas import MessageOut, ProjectCreate, ProjectDetail, ProjectOut, ProjectUpdate, RunOut, SourceOut
from app.services.artifact_storage import delete_project_storage, workspace_dir

router = APIRouter(prefix="/api/projects", tags=["projects"])

ALLOWED_SETTINGS = {"preferences", "research_mode", "max_rounds", "enabled_tools", "pinned"}


@router.get("", response_model=list[ProjectOut])
def list_projects(status: Optional[str] = Query(default=None, pattern="^(active|archived)$"), q: str = "", user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return repo.list_for_user(db, user.id, status, q.strip()[:100])


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = repo.create(db, user.id, body.title, body.description)
    workspace_dir(p.id)
    return p


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    d = ProjectDetail.model_validate(project)
    d.stats = repo.stats(db, project.id)
    return d


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(body: ProjectUpdate, project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    if body.title is not None:
        project.title = body.title.strip()
    if body.description is not None:
        project.description = body.description.strip()
    if body.settings is not None:
        merged = dict(project.settings or {})
        for k, v in body.settings.items():
            if k in ALLOWED_SETTINGS:
                merged[k] = v
        project.settings = merged
    project.updated_at = utcnow()
    db.commit()
    db.refresh(project)
    return project


@router.post("/{project_id}/archive", response_model=ProjectOut)
def archive_project(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    project.status = "archived"
    project.updated_at = utcnow()
    db.commit()
    db.refresh(project)
    return project


@router.post("/{project_id}/unarchive", response_model=ProjectOut)
def unarchive_project(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    project.status = "active"
    project.updated_at = utcnow()
    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=204)
def delete_project(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    pid = project.id
    db.delete(project)
    db.commit()
    delete_project_storage(pid)
    try:
        from app.knowledge_graph.neo4j_client import get_graph

        g = get_graph()
        if g.available:
            # drop the project's links and its private knowledge; shared paper knowledge stays
            g.write("MATCH (p:Project {id: $pid}) DETACH DELETE p", pid=pid)
            g.write("MATCH ()-[r]->() WHERE r.project_id = $pid DELETE r", pid=pid)
            g.write("MATCH (n:Entity {private_project: $pid}) DETACH DELETE n", pid=pid)
    except Exception:
        pass
    return None


@router.get("/{project_id}/messages", response_model=list[MessageOut])
def list_messages(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    return repo.list_messages(db, project.id)


@router.get("/{project_id}/sources", response_model=list[SourceOut])
def list_sources(source_type: Optional[str] = None, project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    return repo.list_sources(db, project.id, source_type)


@router.get("/{project_id}/runs", response_model=list[RunOut])
def list_runs(project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    return repo.list_runs(db, project.id)


@router.get("/{project_id}/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, project: Project = Depends(get_owned_project), db: Session = Depends(get_db)):
    from app.database.models import ResearchRun

    r = db.get(ResearchRun, run_id)
    if r is None or r.project_id != project.id:
        raise HTTPException(status_code=404, detail="Run not found")
    return r
