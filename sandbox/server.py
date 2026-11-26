"""STEM OS execution sandbox service.

Runs inside its own container (see docker-compose.yml): unprivileged user, no capabilities, read-only
root filesystem, no internet (internal network only), memory/pid/cpu limits. It can only see the
project workspaces volume mounted at /workspaces. The backend calls POST /run with a shared token.
"""
from __future__ import annotations

import os
import resource
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("WORKSPACES_ROOT", "/workspaces")).resolve()
TOKEN = os.environ.get("SANDBOX_TOKEN", "")
MEM_MB = int(os.environ.get("SANDBOX_MEMORY_MB", "1024"))
MAX_OUT = 20_000

app = FastAPI(title="STEM OS sandbox")


class RunRequest(BaseModel):
    kind: str = Field(pattern="^(bash|python)$")
    project_id: str = Field(pattern="^[a-f0-9]{32}$")
    cwd: str = ""
    payload: str = Field(max_length=200_000)
    timeout: int = Field(default=60, ge=1, le=600)


def _limits(timeout: int):
    def apply():
        mem = MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
        os.setsid()
    return apply


@app.post("/run")
def run(req: RunRequest, x_sandbox_token: str = Header(default="")):
    if not TOKEN or x_sandbox_token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")
    ws = (ROOT / req.project_id / "workspace").resolve()
    if not ws.is_relative_to(ROOT):
        raise HTTPException(status_code=400, detail="bad project")
    ws.mkdir(parents=True, exist_ok=True)
    cwd = (ws / req.cwd).resolve() if req.cwd else ws
    if not (cwd == ws or cwd.is_relative_to(ws)):
        raise HTTPException(status_code=400, detail="cwd escapes workspace")
    cwd.mkdir(parents=True, exist_ok=True)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(ws), "LANG": "C.UTF-8", "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/mpl", "PYTHONDONTWRITEBYTECODE": "1"}
    script = None
    if req.kind == "python":
        script = Path(tempfile.gettempdir()) / f"run_{uuid.uuid4().hex}.py"
        script.write_text(req.payload, encoding="utf-8")
        argv = ["python", "-X", "utf8", "-I", str(script)]
    else:
        argv = ["/bin/bash", "-c", req.payload]
    try:
        p = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, timeout=req.timeout, preexec_fn=_limits(req.timeout))
        out, err, code, to = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as exc:
        out, err, code, to = exc.stdout or b"", (exc.stderr or b"") + f"\n[timed out after {req.timeout}s]".encode(), -1, True
    finally:
        if script:
            script.unlink(missing_ok=True)
    return {"exit_code": code, "stdout": out.decode("utf-8", "replace")[:MAX_OUT], "stderr": err.decode("utf-8", "replace")[:MAX_OUT], "timed_out": to}


@app.get("/health")
def health():
    return {"ok": True}
