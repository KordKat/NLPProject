"""Compute agent: math, data analysis, simulations and plots, always by running real code."""
from __future__ import annotations

import asyncio
import ast
import json
import re
from pathlib import Path

from app.agent.prompts import COMPUTE_SYSTEM
from app.agent.runtime import Handoff, Runtime, call_prefill, tool_turns
from app.agent.tool_parser import parse_prefilled_call
from app.services.artifact_storage import WorkspaceError, resolve_in_workspace, workspace_dir

_ARITH = re.compile(r"(?<![\w.])(\(?-?\d[\d.,]*\)?(?:\s*(?:\*\*|[+\-*/^%×÷])\s*\(?-?\d[\d.,]*\)?)+)(?![\w])")
_COMMAND = re.compile(r"\b(?:run|execute)\b[^`\n]{0,30}`([^`\n]+)`", re.I)
_SCRIPT = re.compile(r"\b(?:run|execute)\s+(?:the\s+|my\s+)?(?:script\s+)?([\w\-./]+\.py)\b", re.I)
_DATA = re.compile(r"[\w\-./\\]+\.(?:csv|tsv|json|xlsx|parquet|npy)\b", re.I)
_PLOT = re.compile(
    r"\b(plot|visuali[sz]e|graph)\s+(?!the meaning\b)"
    r"|\b(create|make|generate|produce|draw|save|show)\b[^.?!\n]{0,60}\b(chart|plot|graph|histogram|heatmap|visuali[sz]ation)\b",
    re.I,
)
_IMAGE_PATH = re.compile(
    r"\b(?:as|to|into|named|called|save|create)\s+[`'\" ]*([\w./\\-]+\.(?:png|jpe?g|svg|gif|webp))\b", re.I
)


def error_tail(stderr: str, limit: int = 900) -> str:
    """The traceback itself: library warnings printed before it (e.g. ConvergenceWarning) push the
    real error out of view, and the model then "fixes" the warning instead of the bug."""
    start = stderr.rfind("Traceback (most recent call last)")
    return (stderr[start:] if start >= 0 else stderr)[-limit:]


def pure_arithmetic(task: str) -> str:
    """The expression when the task is nothing but plain arithmetic (needs no model call)."""
    task = task.translate(str.maketrans({"−": "-", "×": "*", "÷": "/"}))
    m = _ARITH.search(task)
    if not m or len(task) > 80:
        return ""
    rest = _ARITH.sub(" ", task, count=1).lower()
    words = set(re.findall(r"[a-z]+", rest)) - {"what", "is", "calculate", "compute", "evaluate", "the", "value", "of", "please", "whats", "s"}
    return m.group(1).replace(",", "") if not words and not re.search(r"\d", rest) else ""


def data_preview(project_id: str, task: str) -> tuple[str, list[str]]:
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
    return "\n".join(lines), list(dict.fromkeys(_DATA.findall(task) + names))


class ComputeAgent:
    name = "compute"
    label = "Compute agent"
    description = "Run math or Python: calculations, simulations, data analysis, plots."

    async def run(self, rt: Runtime, task: str) -> Handoff:
        work = task.split("\n\nRequest context:", 1)[0]
        expressions = [pure_arithmetic(clause) for clause in re.split(r"\s+(?:and|then)\s+|;\s*", work, flags=re.I)]
        if all(expressions) and rt.has_tool("eval"):
            results = []
            for expr in expressions:
                res = await rt.execute("eval", {"expression": expr})
                if not res.ok:
                    break
                value = (res.data or {}).get("result")
                results.append(f"{expr} = {value}")
            if len(results) == len(expressions):
                rt.board.computed.extend(results)
                rt.board.compute_exact = True
                return Handoff(True, "Calculated the requested expressions.")
        command = _COMMAND.search(work)
        script = _SCRIPT.search(work)
        if (command or script) and rt.has_tool("bash"):
            cmd = command.group(1) if command else f"python {script.group(1)}"
            res = await rt.execute("bash", {"command": cmd})
            data = res.data if isinstance(res.data, dict) else {}
            output = ((data.get("stdout") or "") + (data.get("stderr") or ""))[:1800].strip()
            if res.ok:
                rt.board.computed.append(f"$ {cmd}\n{output}")
            else:
                rt.board.problems.append(f"The command failed: {(res.error or output)[:200]}")
            return Handoff(res.ok, "Command finished." if res.ok else "Command failed.", output)
        if not rt.has_tool("run_python"):
            rt.board.problems.append("Python execution is disabled for this project.")
            return Handoff(False, "Python execution is disabled.")

        requested_image = _IMAGE_PATH.search(work) if _PLOT.search(work) else None
        plot_requested = bool(_PLOT.search(work))
        skills = await rt.skill_text(task, limit=1)
        system = COMPUTE_SYSTEM + (f"\n\n{skills}" if skills else "")
        input_context = "\n".join([task, *rt.board.attachments])
        preview, inputs = await asyncio.to_thread(data_preview, rt.project_id, input_context)
        inputs = list(dict.fromkeys([*rt.board.attachments, *inputs]))
        context = "\n".join(x for x in (rt.board.for_compute(), preview) if x)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": f"{context}\n\nTASK: {task}".strip()}]
        schemas = rt.schemas("run_python")
        last_error = ""
        feedback = []
        for attempt in range(3):
            out = await rt.llm(messages + feedback, max_new_tokens=1500, temperature=0.1, tools=schemas, prefill=call_prefill("run_python", "code"))
            code = parse_prefilled_call(out, "run_python").get("code", "")
            data = {}
            try:
                if not code:
                    raise SyntaxError("Return a complete run_python call with a code parameter and closing tags.")
                ast.parse(code)
            except SyntaxError as exc:
                last_error = str(exc)
            else:
                args = {"code": code}
                if plot_requested:
                    args.update(output_type="image", inputs=inputs)
                    if requested_image:
                        args["output_path"] = requested_image.group(1)
                res = await rt.execute("run_python", args)
                data = res.data if isinstance(res.data, dict) else {}
                stdout = (data.get("stdout") or "").strip()
                images = [Path(i.get("path") or i.get("name", "")).as_posix() for i in data.get("images", [])]
                files = images + list(data.get("files", []))
                if res.ok and (images if plot_requested else (stdout or files)):
                    rt.board.built += [f for f in files if f not in rt.board.built]
                    saved = f"Saved {', '.join(files)}." if files else ""
                    summary = "\n".join(filter(None, (stdout[:1800], saved)))
                    rt.board.computed.append(summary)
                    return Handoff(True, saved or "Computed results.", summary)
                last_error = error_tail(res.error or data.get("stderr") or ("the requested image was not produced" if plot_requested else "the program produced no results"))
            goal = f"Save {requested_image.group(1) if requested_image else 'an image of the requested plot'}." if plot_requested else "Print the requested results."
            problem = json.dumps({"exit_code": data.get("exit_code"), "error": last_error, "fix": f"Return a complete corrected program. {goal}"})
            feedback = tool_turns("run_python", {"code": code}, problem)
            if attempt < 2:
                await rt.step(f"Code failed, retrying (attempt {attempt + 2} of 3)", stage="verify")
        rt.board.problems.append(f"The computation failed after 3 attempts. Last error: {last_error.strip()[-220:]}")
        return Handoff(False, "Computation failed.")
