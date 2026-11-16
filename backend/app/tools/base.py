"""Tool abstractions shared by the registry, executor and orchestrator."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class ToolContext:
    user_id: str
    project_id: str
    run_id: str
    emit: Callable[[dict[str, Any]], Awaitable[None]]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # accumulated during a run
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    content: str  # what the model sees (compact, structured text / JSON)
    data: Any = None  # structured data for the UI / orchestrator
    ok: bool = True
    error: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def failure(message: str, **data: Any) -> "ToolResult":
        return ToolResult(content=json.dumps({"error": message, **data}, ensure_ascii=False), ok=False, error=message, data=data or None)

    def preview(self, n: int = 700) -> str:
        return self.content if len(self.content) <= n else self.content[:n] + " …"


Handler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    when_to_use: str
    when_not_to_use: str
    returns: str
    example_args: dict[str, Any]
    example_result: str
    interpretation: str
    category: str = "general"  # research | compute | files | graph | output
    timeout: float = 120.0
    # returns a human-readable reason when the call needs explicit user approval, else None
    needs_approval: Optional[Callable[[dict[str, Any]], Optional[str]]] = None
    read_only: bool = True

    def schema(self) -> dict[str, Any]:
        """Function schema in the shape KeawGPT's chat template renders (tools | tojson)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {**self.parameters, "type": "object", "title": "parameters"},
            },
        }
