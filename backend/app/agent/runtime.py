"""Shared run state: the blackboard the main agent routes data through, and the services sub-agents use.

Sub-agents never talk to each other. Each one returns a `Handoff`: a one-line `note` is all the main
agent reads, and the full `data` goes on the blackboard for the agents that need it next.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from app.agent.model_engine import GenerationParams, KeawEngine
from app.agent.tool_parser import format_tool_call
from app.config import get_settings
from app.retrieval.evidence import Fact
from app.services.skills import skill_store
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import registry

settings = get_settings()

Emit = Callable[[dict[str, Any]], Awaitable[None]]
Execute = Callable[..., Awaitable[ToolResult]]


def call_prefill(tool: str, first_param: str) -> str:
    """Start of a native tool call. KeawGPT reliably completes a call it did not have to start."""
    return f"\n<tools_call>\n<function={tool}>\n<parameter={first_param}>\n"


@dataclass
class Handoff:
    ok: bool
    note: str  # one line for the main agent
    data: str = ""  # full result, stored on the blackboard


@dataclass
class Blackboard:
    request: str
    brief: str = ""  # the single cited research paragraph
    facts: list[Fact] = field(default_factory=list)
    graph_notes: str = ""
    computed: list[str] = field(default_factory=list)
    compute_exact: bool = False
    built: list[str] = field(default_factory=list)
    deliveries: list[str] = field(default_factory=list)  # verified descriptions of finished files
    exact: str = ""  # a complete answer read directly from stored data
    earlier: str = ""  # the previous answer, when this request refers to it ("summarize this")
    reference: str = ""  # resolved topic of a follow-up
    attachments: list[str] = field(default_factory=list)
    memory: str = ""
    documents: list[str] = field(default_factory=list)  # attached / named project files
    problems: list[str] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    research_attempted: bool = False
    cited: list[str] = field(default_factory=list)  # source keys, in citation order

    def for_compute(self) -> str:
        parts = [f"Earlier answer the request refers to:\n{self.earlier[:1500]}"] if self.earlier else []
        if self.reference:
            parts.append(f"Follow-up topic: {self.reference}")
        if self.brief:
            parts.append(f"Research notes:\n{self.brief}")
        return "\n\n".join(parts)

    def for_build(self) -> str:
        parts = list(self.documents)
        if self.reference:
            parts.append(f"Follow-up topic: {self.reference}")
        if self.earlier:
            parts.append(f"Earlier answer the request refers to (use it as the content):\n{self.earlier[:2000]}")
        if self.memory:
            parts.append(f"Earlier project work:\n{self.memory[:900]}")
        if self.brief:
            parts.append(f"Research notes (keep the [n] citations):\n{self.brief}")
        if self.computed:
            parts.append("Computed results:\n" + "\n".join(self.computed)[:1500])
        return "\n\n".join(parts)

    def for_answer(self) -> str:
        parts = list(self.documents)
        if self.reference:
            parts.append(f"Follow-up topic: {self.reference}")
        if self.memory:
            parts.append(f"From this project's memory:\n{self.memory}")
        if self.brief:
            parts.append(f"Research findings:\n{self.brief}")
        if self.graph_notes:
            parts.append(f"Related concepts from the knowledge graph:\n{self.graph_notes}")
        if self.computed:
            parts.append("Computed results:\n" + "\n".join(self.computed)[:2500])
        if self.built:
            parts.append("Files created (they are attached; name them and say what they contain, do not paste their content):\n" + "\n".join(f"- {f}" for f in self.built))
        if self.problems:
            parts.append("Problems:\n" + "\n".join(f"- {p}" for p in self.problems))
        return "\n\n".join(parts)


class Runtime:
    """What every sub-agent gets: the model, tools, blackboard and only the user context it needs."""

    def __init__(self, *, engine: KeawEngine, ctx: ToolContext, board: Blackboard, emit: Emit, execute: Execute, user_id: str, project_id: str, enabled_tools: Optional[list[str]], research_rounds: int):
        self.engine = engine
        self.ctx = ctx
        self.board = board
        self.emit = emit
        self.execute = execute
        self.user_id = user_id
        self.project_id = project_id
        self.enabled_tools = enabled_tools
        self.research_rounds = research_rounds
        self.preferences = ""
        self.project_line = ""
        self.history: list[dict[str, str]] = []
        self.prev_exchanges: list[tuple[str, str]] = []
        self.hidden_skills: list[str] = []

    async def step(self, text: str, stage: str = "agent", **extra: Any) -> None:
        await self.emit({"type": "research_step", "stage": stage, "text": text, **extra})

    def has_tool(self, name: str) -> bool:
        return registry.get(name) is not None and (self.enabled_tools is None or name in self.enabled_tools)

    def schemas(self, *names: str) -> list[dict[str, Any]]:
        return registry.schemas([n for n in names if self.has_tool(n)])

    async def llm(self, messages: list[dict[str, str]], *, max_new_tokens: int, temperature: float = 0.1, prefill: str = "", tools: Optional[list[dict]] = None, stop: tuple[str, ...] = ()) -> str:
        return await self.engine.complete(
            messages,
            GenerationParams(max_new_tokens=max_new_tokens, temperature=temperature, stop=stop),
            tools=tools,
            cancel_event=self.ctx.cancel_event,
            prefill=prefill,
        )

    async def skill_text(self, task: str, limit: int = 2, max_tokens: int = 900) -> str:
        """Full text of only the skills relevant to this agent's task."""
        skills = await asyncio.to_thread(skill_store.retrieve, self.user_id, task[:1500], self.hidden_skills, None, limit)
        if not skills:
            return ""
        used = self.ctx.shared.setdefault("skills_used", [])
        for s in skills:
            if s.name not in used:
                used.append(s.name)
        await self.step("Using skills: " + ", ".join(s.title for s in skills), stage="skills")
        return "\n\n".join(f"Instructions ({s.name}):\n{self.engine.truncate_to_tokens(s.body, max_tokens)}" for s in skills)

    # ------------------------------------------------------------------ citations
    def cite(self, key: str) -> int:
        """Citation number of a source. Cited sources are numbered 1..k in order of first use."""
        sources = self.ctx.sources
        if key not in sources:
            return 0
        if key not in self.board.cited:
            self.board.cited.append(key)
            rest = sorted((k for k in sources if k not in self.board.cited), key=lambda k: sources[k]["ref"])
            for i, k in enumerate(self.board.cited + rest, 1):
                sources[k]["ref"] = i
        return sources[key]["ref"]

    def cited_sources(self) -> list[dict[str, Any]]:
        return [self.ctx.sources[k] for k in self.board.cited if k in self.ctx.sources]


def tool_turns(tool: str, args: dict[str, Any], result: str) -> list[dict[str, str]]:
    """A finished call in the model's native transcript format."""
    return [{"role": "assistant", "content": format_tool_call(tool, args)}, {"role": "tool", "content": result}]
