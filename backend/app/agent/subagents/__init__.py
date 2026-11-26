"""Sub-agents. Each has one job, its own short prompt and only the tools it needs.

To add one: create a class with `name`, `label`, `description` and `async run(rt, task) -> Handoff`,
then add it to SUB_AGENTS. The main agent offers it automatically.
"""
from app.agent.subagents.answer import AnswerAgent
from app.agent.subagents.build import BuildAgent
from app.agent.subagents.compute import ComputeAgent
from app.agent.subagents.recall import RecallAgent
from app.agent.subagents.research import ResearchAgent

SUB_AGENTS = {a.name: a for a in (ResearchAgent(), ComputeAgent(), BuildAgent(), RecallAgent())}
ANSWER_AGENT = AnswerAgent()

__all__ = ["SUB_AGENTS", "ANSWER_AGENT"]
