"""Isolated command execution for the bash / run_python tools.

Two modes:
* remote (recommended, used by docker-compose): commands are sent to the `sandbox` service — a
  separate container with no internet access, no capabilities, a read-only root FS, memory/pid/cpu
  limits, running as an unprivileged user, that can only see the project workspaces volume.
* local (development only): a subprocess confined to the project workspace directory with a scrubbed
  environment (no secrets), timeouts and, on POSIX, rlimits. This is NOT a security boundary against
  a hostile model; it is meant for a single trusted developer machine.
"""
from __future__ import annotations

import asyncio
import functools
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import get_settings
from app.services.artifact_storage import resolve_in_workspace, workspace_dir

settings = get_settings()

MAX_OUTPUT = 20_000


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    sandbox: str

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "sandbox": self.sandbox,
        }


def _trim(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + f"\n... [truncated {len(s) - MAX_OUTPUT} chars]"


def _scrubbed_env(cwd: Path) -> dict[str, str]:
    env = {
        "HOME": str(cwd),
        "PWD": str(cwd),
        "LANG": "C.UTF-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(cwd / ".stemos" / "mpl"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    for key in ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "PATHEXT"):
        if key in os.environ:
            env[key] = os.environ[key]
    # numeric libraries start one thread per core and reserve memory for each: on a 256-core server
    # that alone exceeds the memory limit ("OpenBLAS: Memory allocation still failed")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = _MATH_THREADS
    return env


_MATH_THREADS = str(min(4, os.cpu_count() or 1))
_CHILD_TASKS = 256  # processes and threads the code itself may start


def _user_tasks() -> int:
    """Threads already running as this user. RLIMIT_NPROC counts all of them, not only the child's,
    so a fixed limit fails when the server itself (e.g. as root in a container) runs hundreds."""
    uid, count = os.getuid(), 0
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return 0
    for entry in entries:
        if entry.name.isdigit():
            try:
                if entry.stat().st_uid == uid:
                    count += len(os.listdir(f"/proc/{entry.name}/task"))
            except OSError:
                pass
    return count


def _posix_limits(max_tasks: int):  # pragma: no cover - POSIX only
    import resource

    mem = settings.sandbox_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_CPU, (settings.sandbox_timeout, settings.sandbox_timeout + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 1024 * 1024, 200 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (max_tasks, max_tasks))
    os.setsid()


def _local_run(argv: list[str], cwd: Path, timeout: int) -> ExecResult:
    (cwd / ".stemos" / "mpl").mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(cwd=str(cwd), env=_scrubbed_env(cwd), capture_output=True, timeout=timeout)
    if os.name == "posix":
        kwargs["preexec_fn"] = functools.partial(_posix_limits, _user_tasks() + _CHILD_TASKS)
    try:
        p = subprocess.run(argv, **kwargs)
        return ExecResult(
            exit_code=p.returncode,
            stdout=_trim(p.stdout.decode("utf-8", "replace")),
            stderr=_trim(p.stderr.decode("utf-8", "replace")),
            timed_out=False,
            sandbox="local",
        )
    except subprocess.TimeoutExpired as exc:
        return ExecResult(
            exit_code=-1,
            stdout=_trim((exc.stdout or b"").decode("utf-8", "replace")),
            stderr=_trim((exc.stderr or b"").decode("utf-8", "replace")) + f"\n[timed out after {timeout}s]",
            timed_out=True,
            sandbox="local",
        )


def _shell_argv(command: str) -> list[str]:
    if os.name == "nt":
        bash = shutil.which("bash")
        if bash:
            return [bash, "-c", command]
        return ["cmd.exe", "/c", command]
    return ["/bin/bash", "-c", command]


async def run_command(project_id: str, command: str, cwd: str = "", timeout: int | None = None) -> ExecResult:
    timeout = min(int(timeout or settings.sandbox_timeout), 600)
    full_cwd, rel_cwd = resolve_in_workspace(project_id, cwd)
    full_cwd.mkdir(parents=True, exist_ok=True)
    if settings.sandbox_mode == "remote":
        return await _remote("bash", project_id, rel_cwd, command, timeout)
    return await asyncio.to_thread(_local_run, _shell_argv(command), full_cwd, timeout)


async def run_python(project_id: str, code: str, timeout: int | None = None, *, cwd: Path | None = None) -> ExecResult:
    timeout = min(int(timeout or settings.sandbox_timeout), 600)
    ws = workspace_dir(project_id)
    run_dir = cwd.resolve().relative_to(ws.resolve()) if cwd is not None else None
    if settings.sandbox_mode == "remote":
        return await _remote("python", project_id, run_dir.as_posix() if run_dir is not None else "", code, timeout)
    script_dir = ws / ".stemos" / "runs"
    script_dir.mkdir(parents=True, exist_ok=True)
    script = script_dir / f"run_{os.getpid()}_{abs(hash(code)) % 10**8}.py"
    script.write_text(code, encoding="utf-8")
    try:
        return await asyncio.to_thread(_local_run, [sys.executable, "-X", "utf8", "-I", str(script)], cwd or ws, timeout)
    finally:
        try:
            script.unlink()
        except OSError:
            pass


async def _remote(kind: str, project_id: str, rel_cwd: str, payload: str, timeout: int) -> ExecResult:
    async with httpx.AsyncClient(timeout=timeout + 15) as client:
        try:
            r = await client.post(
                f"{settings.sandbox_url.rstrip('/')}/run",
                json={"kind": kind, "project_id": project_id, "cwd": rel_cwd, "payload": payload, "timeout": timeout},
                headers={"X-Sandbox-Token": settings.sandbox_token},
            )
            r.raise_for_status()
            d = r.json()
            return ExecResult(
                exit_code=int(d.get("exit_code", -1)),
                stdout=_trim(d.get("stdout", "")),
                stderr=_trim(d.get("stderr", "")),
                timed_out=bool(d.get("timed_out", False)),
                sandbox="remote",
            )
        except httpx.HTTPError as exc:
            return ExecResult(exit_code=-1, stdout="", stderr=f"Sandbox service unavailable: {exc}", timed_out=False, sandbox="remote")
