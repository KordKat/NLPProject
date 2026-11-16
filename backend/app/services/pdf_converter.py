"""Non-blocking PDF conversion: spawns app.services.pdf_worker in a separate OS process and tracks
its status on the PDF artifact row (pending → processing → ready | failed)."""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Optional

from sqlalchemy import update

from app.config import BACKEND_DIR, get_settings
from app.database.models import Artifact
from app.database.session import db_session
from app.services.artifact_storage import artifact_to_dict, register_artifact, resolve_in_workspace

log = logging.getLogger("stemos.pdf")
settings = get_settings()
_tasks: set[asyncio.Task] = set()


def _set_status(project_id: str, rel: str, status: str, error: str = "", meta: Optional[dict] = None) -> dict[str, Any]:
    with db_session() as db:
        a = register_artifact(db, project_id, rel, artifact_type="pdf", status=status, error=error, meta=meta)
        return artifact_to_dict(a)


async def _run(project_id: str, in_rel: str, out_rel: str) -> None:
    in_full, _ = resolve_in_workspace(project_id, in_rel)
    out_full, _ = resolve_in_workspace(project_id, out_rel)
    await asyncio.to_thread(_set_status, project_id, out_rel, "processing")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "app.services.pdf_worker", str(in_full), str(out_full), settings.pdf_engine,
            cwd=str(BACKEND_DIR), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("PDF conversion timed out")
        if proc.returncode != 0 or not out_full.exists():
            raise RuntimeError(err.decode("utf-8", "replace")[-1500:] or f"converter exited with {proc.returncode}")
        await asyncio.to_thread(_set_status, project_id, out_rel, "ready", "", {"source_markdown": in_rel})
        log.info("PDF ready: %s/%s", project_id, out_rel)
    except Exception as exc:
        log.warning("PDF conversion failed for %s: %s", in_rel, exc)
        await asyncio.to_thread(_set_status, project_id, out_rel, "failed", str(exc)[:2000], {"source_markdown": in_rel})


def start_conversion(project_id: str, in_rel: str, out_rel: str) -> dict[str, Any]:
    """Register the PDF artifact as pending and launch conversion in the background."""
    art = _set_status(project_id, out_rel, "pending", "", {"source_markdown": in_rel})
    task = asyncio.get_running_loop().create_task(_run(project_id, in_rel, out_rel))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return art


def fail_stale_conversions() -> None:
    """On startup, conversions interrupted by a restart are marked failed (they can be re-triggered)."""
    with db_session() as db:
        db.execute(
            update(Artifact)
            .where(Artifact.status.in_(["pending", "processing"]))
            .values(status="failed", error="Interrupted by server restart — re-run conversion.")
        )
