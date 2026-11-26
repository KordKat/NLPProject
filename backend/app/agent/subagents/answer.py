"""Answer agent: the only agent that talks to the user. Streams the final Markdown reply."""
from __future__ import annotations

import asyncio
import difflib
import re

from app.agent.model_engine import GenerationParams, strip_turn_markers
from app.agent.prompts import WRITER_SYSTEM
from app.agent.runtime import Handoff, Runtime, tool_turns
from app.agent.tool_parser import split_reasoning, strip_tool_markup, visible_answer
from app.config import get_settings
from app.retrieval.evidence import attribute_answer
from app.services.artifact_storage import WorkspaceError, resolve_in_workspace, workspace_dir

settings = get_settings()
_FILE = re.compile(r"(?<![\w/:.\-])((?:[\w\-]+/)*[\w\-]+\.(?:pptx|docx|xlsx|csv|tsv|md|html|py|ipynb|png|jpe?g|svg|pdf|json|txt|tex))\b", re.I)


def missing_files(rt: Runtime, text: str) -> list[str]:
    """Workspace files the answer names that do not exist."""
    known = {a.get("path") for a in rt.ctx.artifacts} | {a.get("filename") for a in rt.ctx.artifacts}
    request = rt.board.request
    missing: list[str] = []
    for m in _FILE.finditer(text):
        name = m.group(1)
        if text[max(0, m.start(1) - 8):m.start(1)].lower().endswith(("http://", "https://", "www.")) or name in known or name in request:
            continue
        try:
            full, rel = resolve_in_workspace(rt.project_id, name)
        except WorkspaceError:
            continue
        if full.exists() or any(str(k).endswith("/" + name) for k in known if k) or rel in missing:
            continue
        if "/" not in name and next(workspace_dir(rt.project_id).rglob(name), None) is not None:
            continue
        missing.append(rel)
    return missing[:5]


def repeats_previous(rt: Runtime, answer: str) -> bool:
    """A near-copy of an earlier answer to a DIFFERENT request."""
    if len(answer) < 200:
        return False
    current = set(re.findall(r"[a-z0-9]{3,}", rt.board.request.lower()))
    for req, prev in rt.prev_exchanges:
        if len(prev) < 200:
            continue
        earlier = set(re.findall(r"[a-z0-9]{3,}", req.lower()))
        if current and len(current & earlier) / max(1, len(current | earlier)) >= 0.7:
            continue
        if difflib.SequenceMatcher(None, answer[:2000], prev[:2000], autojunk=False).ratio() >= 0.7:
            return True
    return False


def drop_repeats(text: str, previous: str, threshold: float = 0.6) -> str:
    """Removes sentences that restate `previous`. Code blocks are left alone."""
    words = lambda s: set(re.findall(r"[a-z0-9]{3,}", s.lower()))  # noqa: E731
    seen = [w for w in (words(s) for s in re.split(r"(?<=[.!?])\s+", previous)) if len(w) >= 3]
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        lines = []
        for line in parts[i].split("\n"):
            kept = [s for s in re.split(r"(?<=[.!?])\s+", line) if not (len(words(s)) >= 3 and any(len(words(s) & p) / len(words(s) | p) >= threshold for p in seen))]
            if kept or not line.strip():
                lines.append(" ".join(kept))
        parts[i] = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()


class AnswerAgent:
    name = "answer"
    label = "Answer agent"
    description = "Write the reply to the user. Use it alone for greetings, rewrites and well-known concepts. Use it last after other agents."

    def _messages(self, rt: Runtime, user_content: str) -> list[dict[str, str]]:
        system = WRITER_SYSTEM
        if rt.preferences:
            system += f"\n\nUser preferences:\n{rt.preferences}"
        if rt.project_line:
            system += f"\n\n{rt.project_line}"
        messages = [{"role": "system", "content": system}, *rt.history, {"role": "user", "content": user_content}]
        results = rt.board.for_answer()
        if results:
            messages += tool_turns("get_results", {"for": "this request"}, results)
        return messages

    async def _stream(self, rt: Runtime, messages: list[dict[str, str]]) -> tuple[str, str]:
        params = GenerationParams(max_new_tokens=settings.llm_max_new_tokens, temperature=settings.llm_temperature, top_p=settings.llm_top_p)
        prompt, n = rt.engine.fit_prompt(messages, None, False, params.max_new_tokens)
        await rt.emit({"type": "status", "text": "Writing answer", "prompt_tokens": n})
        raw, sent = "", ""
        async for delta in rt.engine.stream(prompt, params, rt.ctx.cancel_event):
            raw += delta
            _, answer = split_reasoning(raw)
            visible, _ = visible_answer(answer)  # a stray tool tag ends the visible text
            if visible != sent and visible.startswith(sent):
                await rt.emit({"type": "text_delta", "text": visible[len(sent):]})
                sent = visible
        _, answer = split_reasoning(strip_turn_markers(raw), final=True)
        return strip_tool_markup(visible_answer(answer + "\n")[0]), sent

    async def deliver(self, rt: Runtime) -> tuple[Handoff, str]:
        """Verified output needs no second rewrite (which loses citations and adds claims): a built
        file is described from the file itself, and a research paragraph is the answer."""
        board = rt.board
        parts = [board.exact] if board.exact else []
        if board.deliveries:
            parts.append("Created:\n" + "\n".join(f"- {d}" for d in board.deliveries))
        if board.brief:
            parts.append(f"**Key findings:** {board.brief}" if board.deliveries else board.brief)
        if board.computed:
            parts.append("**Computed results:**\n```text\n" + "\n".join(board.computed)[:1200] + "\n```")
        parts += [f"> **Note:** {p}" for p in board.problems]
        text = "\n\n".join(parts)
        await rt.emit({"type": "text_delta", "text": text})
        return Handoff(True, "Delivered."), text

    async def run(self, rt: Runtime, user_content: str, extends: str = "") -> tuple[Handoff, str]:
        """`extends` is the answer being continued: anything restating it is removed."""
        messages = self._messages(rt, user_content)
        text, sent = await self._stream(rt, messages)
        if not text.strip():  # the model ended its turn immediately: nudge it once
            messages.append({"role": "user", "content": "Write the answer now."})
            text, sent = await self._stream(rt, messages)
        if extends:
            text = drop_repeats(text, extends) or "The previous answer already covers this. Tell me which part you want in more detail."
        elif repeats_previous(rt, text):
            await rt.step("Answer repeated an earlier reply, rewriting", stage="verify")
            await rt.emit({"type": "clear_answer"})
            messages += [{"role": "assistant", "content": text[:1200]}, {"role": "user", "content": f"That repeats an earlier answer. Answer this request instead: {rt.board.request[:500]}"}]
            text, sent = await self._stream(rt, messages)
        # figures are attached below the answer: drop inline copies of them
        text = re.sub(r"!\[[^\]]*\]\(([^)\s]+)\)\s*", lambda m: "" if any(m.group(1).endswith(f) or f.endswith(m.group(1)) for f in rt.board.built) else m.group(0), text).strip()
        missing = await asyncio.to_thread(missing_files, rt, text)
        if missing:
            text += f"\n\n> **Note:** these files were mentioned but do not exist: {', '.join(missing)}."
        for problem in rt.board.problems:  # failures are always reported, whatever the model wrote
            text += f"\n\n> **Note:** {problem}"
        final = await asyncio.to_thread(attribute_answer, text, rt.board.facts, rt.cite, rt.cited_numbers() | {0})
        final = final.replace("[0]", "").strip()
        if final != sent.strip():
            await rt.emit({"type": "replace_answer", "text": final})
        return Handoff(True, "Answer delivered."), final
