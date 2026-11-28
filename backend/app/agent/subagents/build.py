"""Build agent: creates and edits project files (slides, web pages, scripts, documents, reports)."""
from __future__ import annotations

import asyncio
import base64
import json
import re

from app.agent.prompts import BUILD_FILE_SYSTEM
from app.agent.runtime import Handoff, Runtime, call_prefill, tool_turns
from app.agent.tool_parser import parse_prefilled_call
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


_SLIDE_COUNT = re.compile(r"\b(\d+)[- ]slides?\b", re.I)
_PRESENTATION_PATH = re.compile(r"[\w\-./]+\.pptx\b", re.I)
_REPORT_PATH = re.compile(r"\b(?:as|to|into|named|called)\s+[`'\" ]*([\w\-./]+\.md)\b", re.I)
_SLIDE_SYSTEM = """Draft one PowerPoint slide with a short title and 2-4 concise bullets.
Use the requested topic and source material. Use this format:
Title: How evaporation works
- Energetic molecules escape the liquid surface.
- Warmer liquids evaporate faster.
End slide"""
_SLIDE_TITLE = re.compile(r"^(?:#+\s*)?\**Title:\s*\**\s*(.+)", re.I)
_SLIDE_BULLET = re.compile(r"^(?:[-*•]|\d+[.)])\s+(.+)")


def read_slide(raw: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines or not (heading := _SLIDE_TITLE.match(lines[0])):
        return None
    title = heading.group(1).strip(" *")
    if not 0 < len(title) <= 80:
        return None
    bullets = [match.group(1).strip() for line in lines[1:] if (match := _SLIDE_BULLET.match(line))]
    bullets = [bullet for bullet in bullets if 0 < len(bullet) <= 160][:6]
    if not bullets:
        return None
    return {"title": title, "bullets": bullets}


def presentation_program(path: str, slides: list[dict[str, Any]], figures: list[str]) -> str:
    payload = base64.b64encode(json.dumps({"path": path, "slides": slides, "figures": figures}).encode("utf-8"))
    return f"""
import base64
import json
from pathlib import Path
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.util import Inches, Pt

deck = Presentation()
deck.slide_width = Inches(13.333)
deck.slide_height = Inches(7.5)
data = json.loads(base64.b64decode({payload!r}))
slides = data["slides"]
figures = data["figures"]
for index, item in enumerate(slides):
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = item["title"]
    slide.shapes.title.width = Inches(12)
    content = slide.placeholders[1]
    content.width = Inches(12)
    text = content.text_frame
    text.word_wrap = True
    text.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    picture = figures[index - (len(slides) - len(figures))] if index >= len(slides) - len(figures) else None
    if picture:
        content.width = Inches(6.1)
    for position, bullet in enumerate(item["bullets"]):
        paragraph = text.add_paragraph() if position else text.paragraphs[0]
        paragraph.text = bullet
        paragraph.font.size = Pt(18 if picture else 24)
    if picture:
        shape = slide.shapes.add_picture(picture, Inches(7.05), Inches(2.1), width=Inches(5.6))
        if shape.height > Inches(4.6):
            shape.width = int(shape.width * Inches(4.6) / shape.height)
            shape.height = Inches(4.6)
output = Path(data["path"])
output.parent.mkdir(parents=True, exist_ok=True)
deck.save(str(output))
"""


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
            if not res.ok:
                rt.board.problems.append(f"The project files could not be listed: {res.error}")
                return Handoff(False, "Could not list files.")
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
            pdf = (res.data or {}).get("path") or f"{md[:-3]}.pdf"
            rt.board.exact = f"PDF conversion for **{pdf}** is in progress."
            return Handoff(True, f"Converting {md} to PDF.")
        if SLIDES.search(task):
            return await self._presentation(rt, task)
        if REPORT.search(task):
            return await self._report(rt, task)
        return await self._file(rt, task)

    async def _report(self, rt: Runtime, task: str) -> Handoff:
        if rt.board.research_attempted and not rt.board.facts:
            return Handoff(False, "No evidence to write the report from.")
        title =re.sub(r"^(please\s+)?(write|create|make|generate|prepare)\s+(me\s+)?(an?\s+)?(detailed\s+|full\s+|short\s+)?(research\s+)?report\s+(on|about)\s+", "", task.strip(), flags=re.I)
        title = re.sub(r"\(?https?://\S+\)?|\.\.\.|…", "", title)  # a rewritten request can carry a link
        title = re.sub(r"^(the\s+)?(paper|article|source|study)\s+", "", title.strip(), flags=re.I)
        title = title.rstrip(".").strip()[:90]
        title = (title[0].upper() + title[1:]) if title else "Research Report"
        named = _REPORT_PATH.search(task)
        res = await rt.execute("generate_report", {"title": title, "output_path": named.group(1) if named else "report.md"})
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

    async def _presentation(self, rt: Runtime, task: str) -> Handoff:
        """Have the model draft slide content and render the validated outline with application code."""
        if not rt.has_tool("run_python"):
            rt.board.problems.append("Python execution is disabled, so the file could not be created.")
            return Handoff(False, "Python execution is disabled.")
        wanted = _SLIDE_COUNT.search(task)
        count = int(wanted.group(1)) if wanted else 5
        if not 1 <= count <= 100:
            rt.board.problems.append("Choose a slide count from 1 to 100.")
            return Handoff(False, "Invalid slide count.")
        named = _PRESENTATION_PATH.search(task)
        requested_path = named.group() if named else "slides/presentation.pptx"
        try:
            _, rel = resolve_in_workspace(rt.project_id, requested_path)
            path = unique_path(rt.project_id, rel)
        except WorkspaceError as exc:
            rt.board.problems.append(str(exc))
            return Handoff(False, "Invalid PowerPoint path.")
        content = rt.board.for_build()
        slides = []
        for number in range(1, count + 1):
            titles = [slide["title"] for slide in slides]
            user = f"TASK: {task}\n\n"
            if content:
                user += f"SOURCE MATERIAL:\n{content}\n\n"
            if titles:
                user += "TOPICS ALREADY COVERED: " + ", ".join(titles) + "\n\n"
            focus = ("Introduce the topic" if number == 1 else "End with the main takeaway" if number == count else
                     ("Explain a core concept", "Describe how it works", "Give a concrete example")[(number - 2) % 3])
            user += f"Write slide {number} of {count}. {focus}, following any specific slide order in the task."
            for attempt in range(3):
                feedback = "\nAdd new points and begin each bullet with a hyphen." if attempt else ""
                out = await rt.llm([{"role": "system", "content": _SLIDE_SYSTEM}, {"role": "user", "content": user + feedback}],
                                   max_new_tokens=160, temperature=0.2, prefill="Title: ", stop=("\nEnd slide",))
                slide = read_slide(out)
                if slide and all(slide["bullets"] != existing["bullets"] for existing in slides):
                    if slide["title"].casefold() in {title.casefold() for title in titles}:
                        slide["title"] = f"{slide['title']} ({number})"
                    slides.append(slide)
                    break
                if attempt < 2:
                    await rt.step(f"Rewriting slide {number} of {count}", stage="verify")
            else:
                rt.board.problems.append(f"Slide {number} of {count} could not be validated.")
                return Handoff(False, "Could not validate the slide outline.")
        figures = [f for f in [*rt.board.built, *rt.board.attachments] if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        figures = list(dict.fromkeys(figures))[:count]
        res = await rt.execute("run_python", {"code": presentation_program(path, slides, figures), "output_type": "presentation",
                                             "output_path": path, "slide_count": count, "inputs": figures})
        if not res.ok or path not in (res.data or {}).get("files", []):
            rt.board.problems.append(f"The PowerPoint could not be created: {res.error or 'the requested deck is missing'}")
            return Handoff(False, "Could not create the PowerPoint.")
        rt.board.built.append(path)
        rt.board.deliveries.append(await asyncio.to_thread(describe_file, rt.project_id, path))
        return Handoff(True, f"Created {path} with {count} slides.")

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
            parsed = parse_prefilled_call(out, tool)
            if parsed.get("path") != path or not parsed.get("content"):
                last_error = "The model did not return a complete response for the requested path."
                messages += tool_turns(tool, {"path": path}, json.dumps({"error": last_error, "fix": "Return the full file content and close the tool call."}))
                continue
            args = {"path": path, "content": parsed["content"]}
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
