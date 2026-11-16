"""Assistant execution: SSE streaming runs, cancellation, approvals, agent status."""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.model_engine import get_engine
from app.agent.orchestrator import ActiveRun, execute_run, run_manager
from app.config import get_settings
from app.database.models import Project, User, utcnow
from app.database.repositories import projects as repo
from app.database.session import get_db
from app.deps import get_current_user, get_owned_project
from app.schemas import ApprovalDecision, AssistantRequest
from app.security import rate_limiter
from app.services.approvals import approvals
from app.services.artifact_storage import WorkspaceError, resolve_in_workspace
from app.tools.registry import registry

router = APIRouter(prefix="/api/projects", tags=["assistant"])
settings = get_settings()


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/{project_id}/assistant")
async def run_assistant(body: AssistantRequest, request: Request, project: Project = Depends(get_owned_project), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rate_limiter.check(f"assistant:{user.id}", settings.rate_limit_assistant_per_minute)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")
    if run_manager.active_for_project(project.id):
        raise HTTPException(status_code=409, detail="A run is already in progress in this project. Cancel it or wait for it to finish.")
    attachments = []
    for a in body.attachments[:10]:
        try:
            full, rel = resolve_in_workspace(project.id, a)
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if full.exists():
            attachments.append(rel)

    psettings = project.settings or {}
    research_mode = body.research_mode if body.research_mode != "auto" else psettings.get("research_mode", "auto")
    enabled_tools = body.enabled_tools if body.enabled_tools is not None else psettings.get("enabled_tools")
    if enabled_tools is not None:
        enabled_tools = [t for t in enabled_tools if t in registry.names()]

    user_msg = repo.add_message(db, project.id, "user", message, {"attachments": attachments, "research_mode": research_mode})
    run = repo.create_run(db, project.id, message)
    project.last_activity_at = utcnow()
    db.commit()
    db.expunge(project)

    queue: asyncio.Queue = asyncio.Queue()
    active = ActiveRun(run_id=run.id, project_id=project.id, user_id=user.id)
    run_manager.runs[run.id] = active

    async def emit(event: dict[str, Any]) -> None:
        await queue.put(event)

    async def runner() -> None:
        lock = run_manager.lock_for(project.id)
        try:
            async with lock:
                await execute_run(project=project, user_id=user.id, message=message, research_mode=research_mode, attachments=attachments, enabled_tools=enabled_tools, emit=emit, active=active)
        finally:
            run_manager.runs.pop(run.id, None)
            await queue.put(None)

    active.task = asyncio.create_task(runner())

    async def stream() -> AsyncIterator[str]:
        yield _sse({"type": "accepted", "run_id": run.id, "user_message_id": user_msg.id})
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    break
                yield _sse(event)
        finally:
            # client went away → stop generation (the run persists what it has)
            if active.task and not active.task.done():
                active.cancel_event.set()
                approvals.cancel_project(project.id)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@router.post("/{project_id}/runs/{run_id}/cancel")
def cancel_run(run_id: str, project: Project = Depends(get_owned_project), user: User = Depends(get_current_user)):
    active = run_manager.runs.get(run_id)
    if active is None or active.project_id != project.id:
        raise HTTPException(status_code=404, detail="No active run with this id")
    run_manager.cancel(run_id, user.id)
    return {"cancelled": True}


@router.post("/{project_id}/approvals/{approval_id}")
def decide_approval(approval_id: str, body: ApprovalDecision, project: Project = Depends(get_owned_project), user: User = Depends(get_current_user)):
    if not approvals.decide(approval_id, project.id, user.id, body.approve):
        raise HTTPException(status_code=404, detail="Approval request not found or expired")
    return {"approved": body.approve}


@router.get("/{project_id}/active-run")
def active_run(project: Project = Depends(get_owned_project)):
    r = run_manager.active_for_project(project.id)
    return {"run_id": r.run_id if r else None}


tools_router = APIRouter(prefix="/api/agent", tags=["assistant"])


@tools_router.get("/tools")
def list_tools(user: User = Depends(get_current_user)):
    return [
        {"name": s.name, "description": s.description, "category": s.category, "read_only": s.read_only, "requires_approval": s.needs_approval is not None}
        for s in registry.specs()
    ]


@tools_router.get("/status")
def agent_status(user: User = Depends(get_current_user)):
    e = get_engine()
    return {"model": e.model_name, "status": e.status, "error": e.error, "device": e.device, "context_tokens": e.max_context_len}
