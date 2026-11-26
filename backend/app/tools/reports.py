"""generate_report / convert_report_to_pdf tools."""
from __future__ import annotations

import asyncio
import os
from pathlib import PurePosixPath
from typing import Any

from app.database.repositories import projects as repo
from app.database.session import db_session
from app.services import pdf_converter
from app.services.artifact_storage import WorkspaceError, resolve_in_workspace, workspace_dir
from app.services.report_generator import write_report
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.common import register_file_artifact
from app.tools.registry import compact_json, registry


def _project_evidence(project_id: str) -> tuple[list[dict], list[dict]]:
    with db_session() as db:
        evidence = repo.recent_evidence(db, project_id, limit_runs=4)
        sources = []
        for i, s in enumerate(repo.list_sources(db, project_id)[:40], start=1):
            sources.append({"ref": i, "source_key": s.source_key, "source_type": s.source_type, "paper_id": s.paper_id, "title": s.title, "url": s.source_url, "meta": s.meta or {}})
    # remap evidence refs to the project-level numbering by source key
    by_key = {s["source_key"]: s["ref"] for s in sources}
    for ev in evidence:
        if ev.get("source_key") in by_key:
            ev["ref"] = by_key[ev["source_key"]]
    return evidence, sources


async def generate_report(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    title = str(args.get("title") or "Research Report").strip()
    out = str(args.get("output_path") or "report.md")
    if not out.endswith(".md"):
        out += ".md"
    try:
        full, rel = resolve_in_workspace(ctx.project_id, out)
    except WorkspaceError as exc:
        return ToolResult.failure(str(exc))

    if ctx.evidence:
        evidence = [e for e in ctx.evidence if e.get("relevant", True)]
        sources = sorted(ctx.sources.values(), key=lambda s: s["ref"])
        used = {e.get("ref") for e in evidence}
        sources = [s for s in sources if s["ref"] in used]
    else:
        evidence, sources = await asyncio.to_thread(_project_evidence, ctx.project_id)
    if not evidence and not sources:
        return ToolResult.failure("No research evidence in this project yet. Research the topic first (search_arxiv / search_web), then generate the report.")

    report_dir = PurePosixPath(rel).parent
    figures = []
    for a in ctx.artifacts:
        if a["type"] == "image":
            relp = os.path.relpath(a["path"], str(report_dir) if str(report_dir) != "." else ".").replace("\\", "/")
            figures.append({"path": relp, "filename": a["filename"], "caption": (a.get("meta") or {}).get("title") or a["filename"].rsplit(".", 1)[0].replace("_", " ")})

    await ctx.emit({"type": "research_step", "stage": "report", "text": f"Writing report “{title}” from {len(evidence)} evidence items"})
    markdown = await write_report(
        title=title,
        question=ctx.shared.get("question", title),
        evidence=evidence,
        sources=sources,
        figures=figures,
        preferences_text=ctx.shared.get("preferences_text", ""),
        graph_context=ctx.shared.get("graph_context", ""),
        focus=str(args.get("focus", "") or ""),
        cancel_event=ctx.cancel_event,
    )
    full.parent.mkdir(parents=True, exist_ok=True)
    if full.exists():
        backup = workspace_dir(ctx.project_id) / ".stemos" / "backups" / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(full.read_bytes())
    await asyncio.to_thread(full.write_text, markdown, "utf-8")
    art = await register_file_artifact(ctx, rel, artifact_type="report", meta={"title": title, "sources": len(sources)})

    pdf_info = None
    if args.get("convert_pdf", True):
        pdf_rel = rel[:-3] + ".pdf"
        pdf_info = pdf_converter.start_conversion(ctx.project_id, rel, pdf_rel)
        ctx.artifacts.append(pdf_info)
        await ctx.emit({"type": "artifact", "artifact": pdf_info})
    return ToolResult(
        content=compact_json({"report": rel, "artifact_url": art["url"], "words": len(markdown.split()), "sources_cited": len(sources), "pdf": {"path": pdf_info["path"], "status": pdf_info["status"]} if pdf_info else None}),
        data={"markdown": markdown, "path": rel},
        artifacts=[art] + ([pdf_info] if pdf_info else []),
    )


async def convert_report_to_pdf(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    inp = str(args.get("input_path") or "report.md")
    out = str(args.get("output_path") or (inp[:-3] + ".pdf" if inp.endswith(".md") else inp + ".pdf"))
    try:
        full_in, rel_in = resolve_in_workspace(ctx.project_id, inp)
        _, rel_out = resolve_in_workspace(ctx.project_id, out)
    except WorkspaceError as exc:
        return ToolResult.failure(str(exc))
    if not full_in.exists():
        return ToolResult.failure(f"Markdown file '{rel_in}' not found. Generate the report first.")
    info = pdf_converter.start_conversion(ctx.project_id, rel_in, rel_out)
    ctx.artifacts.append(info)
    await ctx.emit({"type": "artifact", "artifact": info})
    return ToolResult(content=compact_json({"pdf": rel_out, "status": "converting in background", "artifact_url": info["url"]}), data=info, artifacts=[info])


registry.register(ToolSpec(
    name="generate_report",
    description="Write a formal Markdown research report (report.md) from the evidence gathered in this project, with inline citations and an automatic reference list; PDF conversion starts automatically.",
    parameters={"properties": {
        "title": {"type": "string", "description": "Report title."},
        "output_path": {"type": "string", "description": "Markdown path in the workspace (default report.md)."},
        "focus": {"type": "string", "description": "Optional emphasis, e.g. 'compare sampling methods'."},
        "convert_pdf": {"type": "boolean", "description": "Also create a PDF (default true)."},
    }, "required": ["title"]},
    handler=generate_report, category="output", read_only=False, timeout=900,
    when_to_use="The user asks for a report, write-up, literature review or document — AFTER researching the topic.",
    when_not_to_use="Normal questions — answer directly in Markdown instead.",
    returns='JSON {"report", "artifact_url", "words", "sources_cited", "pdf":{"path","status"}}.',
    example_args={"title": "Transformer Architecture Research", "output_path": "report.md"},
    example_result='{"report": "report.md", "artifact_url": "/api/artifacts/77aa", "words": 1840, "sources_cited": 7, "pdf": {"path": "report.pdf", "status": "pending"}}',
    interpretation="Tell the user the report and PDF are in the project artifacts and summarise the key findings in a few bullet points.",
))
registry.register(ToolSpec(
    name="convert_report_to_pdf",
    description="Convert an existing Markdown file in the workspace to PDF in a separate background process (equations, tables, code and images preserved).",
    parameters={"properties": {
        "input_path": {"type": "string", "description": "Markdown file, default report.md."},
        "output_path": {"type": "string", "description": "PDF path, default <input>.pdf."},
    }, "required": ["input_path"]},
    handler=convert_report_to_pdf, category="output", read_only=False, timeout=30,
    when_to_use="The user asks for a PDF of an existing Markdown document.",
    when_not_to_use="Right after generate_report (it already converts).",
    returns='JSON {"pdf", "status", "artifact_url"}.',
    example_args={"input_path": "report.md", "output_path": "report.pdf"},
    example_result='{"pdf": "report.pdf", "status": "converting in background", "artifact_url": "/api/artifacts/88bb"}',
    interpretation="Conversion continues in the background; its status is shown in the Reports panel.",
))
