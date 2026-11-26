"""Build agent: creates and edits project files (slides, web pages, scripts, documents, reports)."""
from __future__ import annotations

import asyncio
import json
import re

from app.agent.prompts import BUILD_FILE_SYSTEM, BUILD_SCRIPT_SYSTEM
from app.agent.runtime import Handoff, Runtime, call_prefill, tool_turns
from app.agent.tool_parser import parse_tool_calls
from app.config import get_settings
from app.services.artifact_storage import WorkspaceError, resolve_in_workspace, workspace_dir

settings = get_settings()

SLIDES = re.compile(r"\b(pptx|power ?point|slides?|slide ?deck|presentation|keynote)\b", re.I)
# "report" as the thing to create, not the verb ("report accuracy for each model")
REPORT = re.compile(r"\b(write|create|make|generate|prepare|produce|draft|compile)\b[^.?!\n]{0,40}\b(report|white ?paper|write-?up)\b|\b(report|write-?up)\s+(on|about)\b", re.I)
TO_PDF = re.compile(r"\b(convert|export|turn)\b.*\bpdf\b", re.I)
LIST = re.compile(r"\b(what|which|list|show)\b.*\bfiles?\b", re.I)
DELETE = re.compile(r"^\W*(please\s+)?(delete|remove|erase)\b", re.I)
EDIT = re.compile(r"\b(edit|update|modify|change|fix|append|add to|rewrite)\b", re.I)
_PATH = re.compile(r"[\w\-./]+\.(?:pptx|md|html?|py|txt|csv|json|tex|js|ts|css|ya?ml|ipynb|sh)\b", re.I)


_SLIDE_COUNT = re.compile(r"\b(\d{1,2})[- ]slides?\b", re.I)


_KINDS = (
    (re.compile(r"\b(html|web ?pages?|web ?sites?|landing pages?)\b", re.I), "web", ".html"),
    (re.compile(r"\b(python|script)\b", re.I), "code", ".py"),
    (re.compile(r"\bcsv\b", re.I), "data", ".csv"),
    (re.compile(r"\bjson\b", re.I), "data", ".json"),
    (re.compile(r"\b(latex|tex)\b", re.I), "notes", ".tex"),
)
_NAME_STOP = set(
    "write make create build generate prepare draft save produce design python html web webpage page website site landing "
    "script file document markdown csv json latex tex the for that with and about from into this your our you please simple short "
    "small new all can".split()
)


def default_path(project_id: str, task: str) -> str:
    """The path comes from the task, because a small model copies example file names from its prompt."""
    folder, ext = next(((f, e) for rx, f, e in _KINDS if rx.search(task)), ("notes", ".md"))
    words = [w for w in re.findall(r"[a-z0-9]+", task.lower()) if len(w) > 2 and w not in _NAME_STOP][:3]
    stem = ("_" if ext == ".py" else "-").join(words) or "output"
    return unique_path(project_id, f"{folder}/{stem}{ext}")


def unique_path(project_id: str, rel: str) -> str:
    """Never overwrite an existing file when creating a new one."""
    stem, dot, ext = rel.rpartition(".")
    candidate, n = rel, 2
    while resolve_in_workspace(project_id, candidate)[0].exists():
        candidate, n = f"{stem}-{n}{dot}{ext}", n + 1
    return candidate


# "delete the slides" names a kind of file, not a path
_FILE_KINDS = (
    ("slides", re.compile(r"\b(slides?|deck|presentation|pptx|power ?point)\b", re.I), lambda f: f.endswith(".pptx")),
    ("report", re.compile(r"\breports?\b", re.I), lambda f: "report" in f.rsplit("/", 1)[-1]),
    ("chart or image", re.compile(r"\b(charts?|plots?|figures?|graphs?|images?|pictures?)\b", re.I), lambda f: f.endswith((".png", ".jpg", ".jpeg", ".svg"))),
    ("notes", re.compile(r"\b(notes?|markdown)\b", re.I), lambda f: f.endswith(".md")),
    ("script", re.compile(r"\b(scripts?|code|python)\b", re.I), lambda f: f.endswith(".py")),
    ("data", re.compile(r"\b(data|csv|spreadsheets?)\b", re.I), lambda f: f.endswith((".csv", ".tsv", ".xlsx"))),
    ("web page", re.compile(r"\b(web ?pages?|websites?|html)\b", re.I), lambda f: f.endswith(".html")),
)


def files_matching(project_id: str, task: str) -> tuple[list[str], str]:
    """Workspace files the task refers to by name or by kind, and the kind's name ("slides")."""
    ws = workspace_dir(project_id)
    files = [p.relative_to(ws).as_posix() for p in ws.rglob("*") if p.is_file() and ".stemos" not in p.parts]
    low = task.lower()
    named = [f for f in files if f.rsplit("/", 1)[-1].lower() in low]
    if named:
        return named, ""
    kinds = [(label, test) for label, rx, test in _FILE_KINDS if rx.search(task)]
    if not kinds:
        return [], ""
    return [f for f in files if any(test(f.lower()) for _, test in kinds)], kinds[0][0]


def slide_count(project_id: str, rel: str) -> int:
    try:
        from pptx import Presentation

        full, _ = resolve_in_workspace(project_id, rel)
        return len(Presentation(str(full)).slides)
    except Exception:
        return 0


def describe_file(project_id: str, rel: str) -> str:
    """A description read from the real file, so the reply can never misreport what was built."""
    try:
        full, rel = resolve_in_workspace(project_id, rel)
        if full.suffix.lower() == ".pptx":
            from pptx import Presentation

            slides = Presentation(str(full)).slides
            titles = [s.shapes.title.text.strip() for s in slides if s.shapes.title is not None and s.shapes.title.text.strip()]
            return f"**{rel}**: {len(slides)} slides ({', '.join(titles[:12])})"
        text = full.read_text(encoding="utf-8", errors="replace")
        return f"**{rel}**: {len(text.splitlines())} lines"
    except Exception:
        return f"**{rel}**"


class BuildAgent:
    name = "build"
    label = "Build agent"
    description = "Create or edit project files: slides (.pptx), web pages, scripts, documents, reports."

    async def run(self, rt: Runtime, task: str) -> Handoff:
        if DELETE.search(task) and not re.search(r"\bfrom\b", task, re.I):  # "remove outliers from x.csv" edits, not deletes
            named = _PATH.search(task)
            targets, kind = ([named.group(0)], "") if named else await asyncio.to_thread(files_matching, rt.project_id, task)
            if len(targets) != 1:
                if targets:
                    rt.board.exact = f"Several files match: {', '.join(targets)}. Tell me which one to delete."
                elif kind:
                    rt.board.exact = f"There is no {kind} file in this project, so nothing was deleted."
                else:
                    rt.board.exact = "I couldn't tell which file to delete. Name it, for example slides/deck.pptx."
                return Handoff(False, "The file to delete is unclear.")
            res = await rt.execute("delete_file", {"path": targets[0], "confirmed": True})
            rt.board.exact = f"Deleted **{targets[0]}**." if res.ok else f"Could not delete {targets[0]}: {res.error}"
            return Handoff(res.ok, f"Deleted {targets[0]}." if res.ok else res.error[:120])
        if LIST.search(task) and not SLIDES.search(task):
            res = await rt.execute("list_files", {"path": ""})
            files = [f["path"] for f in (res.data or {}).get("files", []) if not f["path"].endswith("/") and f.get("type", "file") != "dir"]
            # the answer is the listing itself: a model would describe files it has not opened
            rt.board.exact = ("Files in this project:\n" + "\n".join(f"- {f}" for f in files)) if files else "This project has no files yet."
            return Handoff(res.ok, f"Listed {len(files)} files.")
        if TO_PDF.search(task):
            md = next((p for p in _PATH.findall(task) if p.lower().endswith(".md")), "report.md")
            res = await rt.execute("convert_report_to_pdf", {"input_path": md})
            if not res.ok:
                rt.board.problems.append(res.error)
                return Handoff(False, res.error[:120])
            rt.board.built.append(f"{md[:-3]}.pdf (converting in the background)")
            return Handoff(True, f"Converting {md} to PDF.")
        if SLIDES.search(task):
            return await self._script(rt, task, ".pptx")
        if REPORT.search(task):
            return await self._report(rt, task)
        return await self._file(rt, task)

    async def _report(self, rt: Runtime, task: str) -> Handoff:
        if rt.board.rounds and not rt.board.facts:  # research ran for this report and found nothing
            return Handoff(False, "No evidence to write the report from.")
        title =re.sub(r"^(please\s+)?(write|create|make|generate|prepare)\s+(me\s+)?(an?\s+)?(detailed\s+|full\s+|short\s+)?(research\s+)?report\s+(on|about)\s+", "", task.strip(), flags=re.I)
        title = re.sub(r"\(?https?://\S+\)?|\.\.\.|…", "", title)  # a rewritten request can carry a link
        title = re.sub(r"^(the\s+)?(paper|article|source|study)\s+", "", title.strip(), flags=re.I)
        title = title.rstrip(".").strip()[:90]
        title = (title[0].upper() + title[1:]) if title else "Research Report"
        res = await rt.execute("generate_report", {"title": title, "output_path": "report.md"})
        if not res.ok:
            rt.board.problems.append(res.error)
            return Handoff(False, res.error[:140])
        info = json.loads(res.content)
        rt.board.built.append(info["report"])
        detail = f"**{info['report']}**: report, {info.get('words')} words, {info.get('sources_cited')} cited sources"
        if info.get("pdf"):
            rt.board.built.append(info["pdf"]["path"])
            detail += f" (PDF version `{info['pdf']['path']}` is being generated)"
        rt.board.deliveries.append(detail)
        return Handoff(True, f"Created {info['report']} ({info.get('words')} words).")

    async def _script(self, rt: Runtime, task: str, ext: str) -> Handoff:
        """Files that need a library (e.g. .pptx): the model writes a program, the sandbox runs it."""
        if not rt.has_tool("run_python"):
            rt.board.problems.append("Python execution is disabled, so the file could not be created.")
            return Handoff(False, "Python execution is disabled.")
        # the file type picks the skill: "slides about a bar chart" must not get the plotting skill
        kind = "PowerPoint presentation slides pptx" if ext == ".pptx" else ext
        skills = await rt.skill_text(f"{kind}. {task[:300]}", limit=1, max_tokens=1000)
        system = BUILD_SCRIPT_SYSTEM + (f"\n\n{skills}" if skills else "")
        content = rt.board.for_build()
        figures = [f for f in rt.board.built if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        extra = f"\n\nFigures you must insert: {', '.join(figures)}" if figures else "\n\nNo figures exist: use text only."
        user = f"{('CONTENT:' + chr(10) + content + chr(10) + chr(10)) if content else ''}TASK: {task[:2500]}{extra}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        schemas = rt.schemas("run_python")
        last_error = ""
        for attempt in range(3):
            out = await rt.llm(messages, max_new_tokens=settings.llm_max_new_tokens, temperature=0.2, tools=schemas, prefill=call_prefill("run_python", "code"))
            calls = parse_tool_calls(out)
            code = (calls[0].arguments.get("code") or "").strip() if calls else ""
            if not code:
                last_error = "no code was produced"
                continue
            res = await rt.execute("run_python", {"code": code})
            data = res.data if isinstance(res.data, dict) else {}
            files = list(data.get("files", [])) + [i.get("path", "") for i in data.get("images", [])]
            made = [f for f in files if f.lower().endswith(ext)]
            wanted = _SLIDE_COUNT.search(task)
            actual = await asyncio.to_thread(slide_count, rt.project_id, made[0]) if res.ok and made and wanted else 0
            if actual and actual != int(wanted.group(1)):
                if attempt < 2:
                    n = int(wanted.group(1))
                    change = f"add {n - actual} more slide{'s' if n - actual > 1 else ''}" if actual < n else f"remove {actual - n} slide{'s' if actual - n > 1 else ''}"
                    feedback = json.dumps({"problem": f"The deck has {actual} slides but the task asked for exactly {n}.", "fix": f"Rewrite the complete program and {change}: call add_slide exactly {n} times."})
                    messages += tool_turns("run_python", {"code": code}, feedback)
                    await rt.step(f"Deck has {actual} slides instead of {wanted.group(1)}, rebuilding", stage="verify")
                    continue
                rt.board.problems.append(f"The deck has {actual} slides, not the {wanted.group(1)} that were asked for.")
            if res.ok and made:
                rt.board.built += [f for f in files if f and f not in rt.board.built]
                for f in made:
                    rt.board.deliveries.append(await asyncio.to_thread(describe_file, rt.project_id, f))
                return Handoff(True, f"Created {', '.join(made)}.")
            last_error = (data.get("stderr") or res.error or f"the program ran but saved no {ext} file")[-700:]
            feedback = json.dumps({"exit_code": data.get("exit_code"), "error": last_error, "fix": f"Rewrite the complete program so it saves a {ext} file."})
            messages += tool_turns("run_python", {"code": code}, feedback)
            if attempt < 2:
                await rt.step(f"Build failed, retrying (attempt {attempt + 2} of 3)", stage="verify")
        rt.board.problems.append(f"The {ext} file could not be created: {last_error[:300]}")
        return Handoff(False, f"Could not create the {ext} file.")

    async def _file(self, rt: Runtime, task: str) -> Handoff:
        """Text files: prefill the native file-creation call so the file is really written."""
        named = _PATH.search(task)
        existing = ""
        path = named.group(0) if named else ""
        edit = False
        if path:
            try:
                full, rel = resolve_in_workspace(rt.project_id, path)
                if full.exists() and EDIT.search(task):
                    edit = True
                    existing = (await asyncio.to_thread(full.read_text, "utf-8", "replace"))[:6000]
                    path = rel
                else:
                    path = unique_path(rt.project_id, rel)
            except WorkspaceError as exc:
                rt.board.problems.append(str(exc))
                return Handoff(False, str(exc))
        else:
            path = default_path(rt.project_id, task)
        tool = "write_file" if edit else "create_file"
        if not rt.has_tool(tool):
            rt.board.problems.append(f"The {tool} tool is disabled for this project.")
            return Handoff(False, f"{tool} is disabled.")
        skills = await rt.skill_text(task, limit=1)
        system = BUILD_FILE_SYSTEM + (f"\n\n{skills}" if skills else "")
        content = rt.board.for_build()
        user = "\n\n".join(x for x in (f"CONTENT:\n{content}" if content else "", f"CURRENT FILE {path}:\n{existing}" if existing else "", f"TASK: {task[:2500]}") if x)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prefill = call_prefill(tool, "path") + f"{path}\n</parameter>\n<parameter=content>\n"
        last_error = ""
        for attempt in range(2):
            out = await rt.llm(messages, max_new_tokens=settings.llm_max_new_tokens, temperature=0.2, tools=rt.schemas(tool), prefill=prefill)
            calls = parse_tool_calls(out)
            args = dict(calls[0].arguments) if calls else {}
            if not args.get("path") or not args.get("content"):
                last_error = "no file content was produced"
                continue
            if edit:
                args["overwrite"] = True
            res = await rt.execute(tool, args)
            if res.ok:
                made = (res.data or {}).get("path", args["path"])
                rt.board.built.append(made)
                rt.board.deliveries.append(await asyncio.to_thread(describe_file, rt.project_id, made))
                return Handoff(True, f"Created {made}.")
            last_error = res.error
            messages += tool_turns(tool, {"path": args["path"], "content": args["content"][:400]}, json.dumps({"error": res.error}))
        rt.board.problems.append(f"The file could not be created: {last_error[:300]}")
        return Handoff(False, "Could not create the file.")
