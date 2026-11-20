"""use_skill — load the instructions of one skill from the user's skill library on demand."""
from __future__ import annotations

import asyncio
from typing import Any

from app.services.skills import hidden_skills_for, skill_store
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.registry import compact_json, registry


async def use_skill(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    name = str(args["name"]).strip()
    hidden = await asyncio.to_thread(hidden_skills_for, ctx.user_id)
    skill = await asyncio.to_thread(skill_store.get, ctx.user_id, name, hidden)
    if skill is None or skill.hidden:
        names = [n for n, _ in await asyncio.to_thread(skill_store.catalog, ctx.user_id, hidden)]
        return ToolResult.failure(f"No skill named '{name}'. Available skills: {', '.join(names) or 'none'}")
    used = ctx.shared.setdefault("skills_used", [])
    if skill.name in used:
        # already in the system prompt (RELEVANT SKILLS) or loaded earlier in this run
        return ToolResult(
            content=compact_json({"skill": skill.name, "status": "already loaded", "note": "These instructions are already in your context under RELEVANT SKILLS. Do not load them again — continue with the task now (e.g. call run_python)."}),
            data={"name": skill.name, "already_loaded": True},
        )
    used.append(skill.name)
    await ctx.emit({"type": "research_step", "stage": "skills", "text": f"Loaded skill: {skill.title}"})
    return ToolResult(content=compact_json({"skill": skill.name, "title": skill.title, "instructions": skill.body}, 9000), data={"name": skill.name})


registry.register(
    ToolSpec(
        name="use_skill",
        description="Load the full instructions of a skill listed under OTHER AVAILABLE SKILLS (task-specific how-to guidance, e.g. building slides, data analysis, frontend design).",
        parameters={"properties": {"name": {"type": "string", "description": "Exact skill name from the skills list."}}, "required": ["name"]},
        handler=use_skill,
        category="skills",
        timeout=30,
        when_to_use="The task needs a capability described by a listed skill whose instructions are not already in RELEVANT SKILLS.",
        when_not_to_use="The skill is already shown under RELEVANT SKILLS, or no listed skill fits the task.",
        returns='JSON {"skill", "title", "instructions"}.',
        example_args={"name": "pptx-skill"},
        example_result='{"skill": "pptx-skill", "title": "PowerPoint presentations with python-pptx", "instructions": "# PowerPoint presentations ... Build decks by calling run_python with a python-pptx script ..."}',
        interpretation="Follow the returned instructions for the rest of the task.",
    )
)
