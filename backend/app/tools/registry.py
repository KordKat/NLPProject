"""Central tool registry + executor.

New tools are added by calling `registry.register(ToolSpec(...))` from any module listed in
EXTRA_TOOLS_MODULES (or in BUILTIN_MODULES) — the orchestrator never needs to change.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
from typing import Any, Iterable, Optional

from app.agent.tool_parser import coerce_arguments, format_tool_call
from app.config import get_settings
from app.tools.base import ToolContext, ToolResult, ToolSpec

log = logging.getLogger("stemos.tools")

BUILTIN_MODULES = [
    "app.tools.arxiv",
    "app.tools.web",
    "app.tools.graph",
    "app.tools.calculator",
    "app.tools.bash",
    "app.tools.python",
    "app.tools.files",
    "app.tools.reports",
    "app.tools.memory",
    "app.tools.skills",
]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._loaded = False

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            log.debug("re-registering tool %s", spec.name)
        self._tools[spec.name] = spec
        return spec

    def load(self) -> None:
        if self._loaded:
            return
        extra = [m.strip() for m in get_settings().extra_tools_modules.split(",") if m.strip()]
        for mod in BUILTIN_MODULES + extra:
            importlib.import_module(mod)
        self._loaded = True

    def get(self, name: str) -> Optional[ToolSpec]:
        self.load()
        return self._tools.get(name)

    def names(self) -> list[str]:
        self.load()
        return list(self._tools)

    def specs(self, names: Optional[Iterable[str]] = None) -> list[ToolSpec]:
        self.load()
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def schemas(self, names: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
        return [s.schema() for s in self.specs(names)]

    def icl_block(self, names: Optional[Iterable[str]] = None) -> str:
        """Compact tool manual with one worked example per tool (injected in the system prompt)."""
        parts = []
        for s in self.specs(names):
            parts.append(
                f"### {s.name}\n"
                f"What: {s.description}\n"
                f"Use when: {s.when_to_use}\n"
                f"Do NOT use when: {s.when_not_to_use}\n"
                f"Returns: {s.returns}\n"
                f"Example call:\n{format_tool_call(s.name, s.example_args)}\n"
                f"Example result: {s.example_result}\n"
                f"How to read it: {s.interpretation}"
            )
        return "\n\n".join(parts)

    async def execute(self, ctx: ToolContext, name: str, raw_args: dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            return ToolResult.failure(f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}")
        args = coerce_arguments(raw_args, spec.parameters)
        missing = [r for r in spec.parameters.get("required", []) if r not in args or args[r] in ("", None)]
        if missing:
            return ToolResult.failure(f"Missing required parameter(s) for {name}: {', '.join(missing)}")
        unknown = set(args) - set(spec.parameters.get("properties", {}))
        for k in unknown:
            args.pop(k, None)
        try:
            return await asyncio.wait_for(spec.handler(ctx, args), timeout=spec.timeout)
        except asyncio.TimeoutError:
            return ToolResult.failure(f"Tool {name} timed out after {int(spec.timeout)}s")
        except Exception as exc:  # tools must never crash the agent
            log.exception("tool %s failed", name)
            return ToolResult.failure(f"{type(exc).__name__}: {exc}")


registry = ToolRegistry()


def compact_json(obj: Any, limit: int = 6000) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + " …(truncated)"
