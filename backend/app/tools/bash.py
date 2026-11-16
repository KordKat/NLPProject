"""bash — run shell commands inside the project's isolated sandbox workspace."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

from app.services import sandbox
from app.services.artifact_storage import WorkspaceError, changed_files, snapshot
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.common import register_file_artifact
from app.tools.registry import compact_json, registry

FORBIDDEN = re.compile(r"\b(sudo|su|doas|chroot|mount|umount|nsenter|setcap|passwd|useradd|usermod)\b|docker\s|/var/run/docker\.sock")
DESTRUCTIVE = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*\s+|.*\*)"), "recursive/forced or wildcard delete"),
    (re.compile(r"\brm\s"), "deletes files"),
    (re.compile(r"\b(rmdir|shred|truncate|dd|mkfs\S*)\b"), "destructive filesystem operation"),
    (re.compile(r"\bfind\b.*\s-delete\b"), "find -delete"),
    (re.compile(r"\b(mv)\s"), "moves/overwrites files"),
    (re.compile(r"\b(kill|pkill|killall|shutdown|reboot)\b"), "process/system control"),
    (re.compile(r"\b(curl|wget|ssh|scp|rsync|nc|ncat|telnet|ftp)\b"), "network access / external side effects"),
    (re.compile(r"\b(pip|pip3|npm|yarn|apt|apt-get|conda)\s+(install|uninstall|remove)"), "installs/removes software"),
    (re.compile(r"\bgit\s+(push|reset\s+--hard|clean)"), "git operation with external or destructive effect"),
    (re.compile(r"(^|[^>])>\s*[^&\s>]"), "redirect may overwrite a file"),
]


def bash_needs_approval(args: dict[str, Any]) -> Optional[str]:
    cmd = str(args.get("command", ""))
    reasons = [why for rx, why in DESTRUCTIVE if rx.search(cmd)]
    return "; ".join(dict.fromkeys(reasons)) or None


async def register_changes(ctx: ToolContext, before: dict[str, float]) -> list[dict[str, Any]]:
    after = await asyncio.to_thread(snapshot, ctx.project_id)
    arts = []
    for rel in changed_files(before, after)[:20]:
        arts.append(await register_file_artifact(ctx, rel, meta={"origin": "execution"}))
    return arts


async def bash_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    command = str(args["command"]).strip()
    if FORBIDDEN.search(command):
        return ToolResult.failure("Command refused: privilege escalation / host access is not permitted in the sandbox.")
    before = await asyncio.to_thread(snapshot, ctx.project_id)
    try:
        res = await sandbox.run_command(ctx.project_id, command, str(args.get("cwd", "") or ""), args.get("timeout"))
    except WorkspaceError as exc:
        return ToolResult.failure(str(exc))
    arts = await register_changes(ctx, before)
    payload = {**res.to_dict(), "command": command, "new_or_modified_files": [a["path"] for a in arts]}
    return ToolResult(content=compact_json(payload, 8000), data=payload, ok=res.exit_code == 0, error="" if res.exit_code == 0 else f"exit code {res.exit_code}", artifacts=arts)


registry.register(
    ToolSpec(
        name="bash",
        description="Execute a shell command in the project's isolated workspace (sandboxed: no host access, limited CPU/memory/time). Returns stdout, stderr and exit code.",
        parameters={
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "cwd": {"type": "string", "description": "Working directory relative to the project workspace (default: workspace root)."},
                "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 60, max 600)."},
            },
            "required": ["command"],
        },
        handler=bash_tool,
        category="compute",
        timeout=620,
        needs_approval=bash_needs_approval,
        read_only=False,
        when_to_use="Running scripts that already exist in the workspace, inspecting files with standard CLI tools (ls, head, wc, grep), compiling code.",
        when_not_to_use="Simple math (eval), Python computations/plots (run_python), reading/writing files (use the file tools).",
        returns='JSON {"exit_code", "stdout", "stderr", "timed_out", "new_or_modified_files"}.',
        example_args={"command": "python analysis.py", "cwd": "."},
        example_result='{"exit_code": 0, "stdout": "mean=4.2\\n", "stderr": "", "timed_out": false, "new_or_modified_files": ["results/summary.csv"]}',
        interpretation="exit_code 0 = success. Read stderr when it is non-zero and fix the problem. Destructive or network commands need user approval and may be rejected.",
    )
)
