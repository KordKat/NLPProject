"""Compute agent: math, data analysis, simulations and plots, always by running real code."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from app.agent.prompts import COMPUTE_SYSTEM
from app.agent.runtime import Handoff, Runtime, call_prefill, tool_turns
from app.agent.tool_parser import parse_tool_calls
from app.services.artifact_storage import WorkspaceError, resolve_in_workspace, workspace_dir

_ARITH = re.compile(r"(?<![\w.])(\(?-?\d[\d.,]*\)?(?:\s*(?:\*\*|[+\-*/^%×÷])\s*\(?-?\d[\d.,]*\)?)+)(?![\w])")
_COMMAND = re.compile(r"\b(?:run|execute)\b[^`\n]{0,30}`([^`\n]+)`", re.I)
_SCRIPT = re.compile(r"\b(?:run|execute)\s+(?:the\s+|my\s+)?(?:script\s+)?([\w\-./]+\.py)\b", re.I)
_DATA = re.compile(r"[\w\-./\\]+\.(?:csv|tsv|json|xlsx)\b", re.I)


def pure_arithmetic(task: str) -> str:
    """The expression when the task is nothing but plain arithmetic (needs no model call)."""
    m = _ARITH.search(task)
    if not m or len(task) > 80:
        return ""
    rest = _ARITH.sub(" ", task, count=1).lower()
    words = set(re.findall(r"[a-z]+", rest)) - {"what", "is", "calculate", "compute", "evaluate", "the", "value", "of", "please", "whats", "s"}
    return m.group(1).replace(",", "") if not words else ""


def data_preview(project_id: str, task: str) -> str:
    """Files in the workspace, plus the header of any data file the task names."""
    ws = workspace_dir(project_id)
    names = [p.relative_to(ws).as_posix() for p in ws.rglob("*") if p.is_file() and ".stemos" not in p.parts][:25]
    lines = []
    if names:
        lines.append("Workspace files: " + ", ".join(names))
    for name in _DATA.findall(task)[:2]:
        try:
            full, rel = resolve_in_workspace(project_id, name)
        except WorkspaceError:
            continue
        if full.suffix.lower() in (".csv", ".tsv") and full.exists():
            with full.open(encoding="utf-8", errors="replace") as f:
                head = [next(f, "").rstrip() for _ in range(3)]
            lines.append(f"First lines of {rel}:\n" + "\n".join(head))
    return "\n".join(lines)


class ComputeAgent:
    name = "compute"
    label = "Compute agent"
    description = "Run math or Python: calculations, simulations, data analysis, plots."

    async def run(self, rt: Runtime, task: str) -> Handoff:
        expr = pure_arithmetic(task)
        if expr and rt.has_tool("eval"):
            res = await rt.execute("eval", {"expression": expr})
            if res.ok:
                value = (res.data or {}).get("result")
                line = f"{expr} = {value}"
                rt.board.computed.append(line)
                return Handoff(True, f"Result: {value}", line)
        command = _COMMAND.search(task)
        script = _SCRIPT.search(task)
        if (command or script) and rt.has_tool("bash"):
            cmd = command.group(1) if command else f"python {script.group(1)}"
            res = await rt.execute("bash", {"command": cmd})
            data = res.data if isinstance(res.data, dict) else {}
            output = ((data.get("stdout") or "") + (data.get("stderr") or ""))[:1800].strip()
            rt.board.computed.append(f"$ {cmd}\n{output or res.error}")
            if not res.ok:
                rt.board.problems.append(f"The command failed: {(res.error or output)[:200]}")
            return Handoff(res.ok, "Command finished." if res.ok else "Command failed.", output)
        if not rt.has_tool("run_python"):
            rt.board.problems.append("Python execution is disabled for this project.")
            return Handoff(False, "Python execution is disabled.")

        skills = await rt.skill_text(task, limit=1)
        system = COMPUTE_SYSTEM + (f"\n\n{skills}" if skills else "")
        context = "\n".join(x for x in (rt.board.for_compute(), await asyncio.to_thread(data_preview, rt.project_id, task)) if x)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": f"{context}\n\nTASK: {task}".strip()}]
        schemas = rt.schemas("run_python")
        last_error = ""
        for attempt in range(3):
            out = await rt.llm(messages, max_new_tokens=1500, temperature=0.1, tools=schemas, prefill=call_prefill("run_python", "code"))
            calls = parse_tool_calls(out)
            code = (calls[0].arguments.get("code") or "").strip() if calls else ""
            if not code:
                last_error = "no code was produced"
                continue
            res = await rt.execute("run_python", {"code": code})
            data = res.data if isinstance(res.data, dict) else {}
            stdout = (data.get("stdout") or "").strip()
            images = [Path(i.get("path") or i.get("name", "")).as_posix() for i in data.get("images", [])]
            if res.ok and (stdout or images or data.get("files")):
                files = images + list(data.get("files", []))
                rt.board.built += [f for f in files if f not in rt.board.built]
                summary = stdout[:1800] + (f"\nFigures saved: {', '.join(images)}" if images else "")
                rt.board.computed.append(summary.strip())
                return Handoff(True, ("Output produced. " + (f"Saved {', '.join(files)}." if files else "")).strip(), summary)
            last_error = (data.get("stderr") or res.error or "the program printed nothing")[-700:]
            feedback = json.dumps({"exit_code": data.get("exit_code"), "error": last_error, "fix": "Rewrite the complete program so it runs and print()s the results."})
            messages += tool_turns("run_python", {"code": code}, feedback)
            if attempt < 2:
                await rt.step(f"Code failed, retrying (attempt {attempt + 2} of 3)", stage="verify")
        rt.board.problems.append(f"The computation failed: {last_error[:300]}")
        return Handoff(False, "Computation failed.")
