"""Regression checks for routing, stage outcomes, and artifact publication."""
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image
from pptx import Presentation

from app.agent import context_manager, orchestrator
from app.agent.router import route_by_rules, split_task
from app.agent.runtime import Blackboard, Handoff
from app.agent.subagents.answer import AnswerAgent
from app.agent.subagents.build import BuildAgent, presentation_program, read_slide
from app.agent.subagents.compute import ComputeAgent
from app.agent.subagents.research import ResearchAgent
from app.agent.tool_parser import parse_prefilled_call, parse_tool_calls
from app.retrieval.evidence import Fact, attribute_answer, extractive_paragraph
from app.services.sandbox import ExecResult
from app.tools import python as python_tool
from app.tools.base import ToolContext, ToolResult


class RoutingTests(unittest.TestCase):
    def test_explicit_deliverable_and_data_routes(self):
        self.assertEqual(route_by_rules("Make this into a 5-slide PowerPoint", True).agents, ["build"])
        self.assertEqual(route_by_rules("Explain my uploaded data.csv").agents, ["compute"])
        self.assertEqual(route_by_rules("7 - 3").agents, ["compute"])
        self.assertTrue(route_by_rules("Summarize that", True).direct)

    def test_task_split_keeps_both_calculations_and_context(self):
        request = "Research solar cells and calculate 2 + 2 and calculate 5 * 6"
        self.assertEqual(split_task("compute", request),
                         "calculate 2 + 2 and calculate 5 * 6\n\nRequest context: Research solar cells")

    def test_latest_unanswered_turn_does_not_take_an_older_reply(self):
        messages = [
            SimpleNamespace(role="user", content="Old request", meta={}),
            SimpleNamespace(role="assistant", content="Old answer", meta={"status": "completed"}),
            SimpleNamespace(role="user", content="New request", meta={"attachments": ["sample.csv"]}),
            SimpleNamespace(role="user", content="continue", meta={}),
        ]
        with patch.object(context_manager, "db_session", return_value=nullcontext(None)), patch.object(
            context_manager.repo, "recent_messages", return_value=messages
        ):
            turn = context_manager.previous_turn("project")
        self.assertEqual((turn.request, turn.answer, turn.status, turn.attachments), ("New request", "", "failed", ["sample.csv"]))

    def test_explicit_web_and_unavailable_arxiv(self):
        rt = SimpleNamespace(has_tool=lambda name: name in {"search_web", "search_arxiv"})
        research = ResearchAgent()
        with patch("app.agent.subagents.research.get_arxiv_index", return_value=SimpleNamespace(available=True)):
            self.assertEqual(research._route(rt, ["solar cells"], "Search the web for solar cells", True),
                             [{"tool": "web", "query": "solar cells"}])
        with patch("app.agent.subagents.research.get_arxiv_index", return_value=SimpleNamespace(available=False)):
            self.assertEqual(research._route(rt, ["solar cells"], "Find studies on solar cells", True),
                             [{"tool": "web", "query": "solar cells"}])


class OutputTests(unittest.TestCase):
    def test_incomplete_and_unexpected_tool_calls_are_rejected(self):
        start = "<tools_call><function=run_python><parameter=code>print(42)"
        self.assertEqual(parse_prefilled_call(start, "run_python"), {})
        complete = start + "</parameter></function></tools_call>"
        self.assertEqual(parse_prefilled_call(complete, "run_python"), {"code": "print(42)"})
        self.assertEqual(parse_prefilled_call(complete, "create_file"), {})
        self.assertEqual(parse_tool_calls('<tool_call>{"name":"run_python","arguments":17}</tool_call>'), [])

    def test_citations_require_the_matching_claim(self):
        fact = Fact("The intervention decreased mortality by 20%.", "source", 1.0, np.ones(1))
        reversed_claim = "The intervention increased mortality by 20% [1][2]."
        self.assertEqual(attribute_answer(reversed_claim, [fact], lambda _: 1),
                         "The intervention increased mortality by 20%.")
        self.assertEqual(attribute_answer(fact.text, [fact], lambda _: 1),
                         "The intervention decreased mortality by 20% [1].")
        self.assertEqual(extractive_paragraph([fact], lambda _: 1),
                         "The intervention decreased mortality by 20% [1].")

    def test_outline_validation_and_real_powerpoint_render(self):
        first = "Title: One\n- First point\nEnd slide"
        second = "Title: Two\n- Second point\nEnd slide"
        self.assertIsNone(read_slide("Title: One"))
        slides = [read_slide(first), read_slide(second)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "slides" / "deck.pptx"
            exec(presentation_program(str(path), slides, []), {})
            self.assertEqual([slide.shapes.title.text for slide in Presentation(str(path)).slides], ["One", "Two"])
            self.assertEqual(python_tool.check_output(path, "presentation", 2), "")
            self.assertIn("Expected 3", python_tool.check_output(path, "presentation", 3))
            path.write_bytes(b"invalid")
            self.assertIn("unreadable", python_tool.check_output(path, "presentation", 2))

    def test_outline_accepts_five_short_bullets_from_the_model(self):
        raw = "Title: Topic\n- One\n- Two\n- Three\n- Four\n- Five\nEnd slide"
        self.assertEqual(len(read_slide(raw)["bullets"]), 5)
        overflow = "Title: Topic\n" + "\n".join(f"- Point {i}" for i in range(8))
        self.assertEqual(len(read_slide(overflow)["bullets"]), 6)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_arithmetic_can_deliver_without_an_available_model(self):
        active = orchestrator.ActiveRun("run", "project", "user")
        project = SimpleNamespace(id="project")
        with patch.object(orchestrator, "get_engine", return_value=SimpleNamespace(available=False)):
            orch = orchestrator.Orchestrator(project=project, user_id="user", message="7 - 3",
                                             research_mode="auto", attachments=[], enabled_tools=["eval"], emit=AsyncMock(), active=active)
        orch.rt.has_tool = lambda name: name == "eval"
        orch.rt.execute = AsyncMock(return_value=ToolResult("4", data={"result": 4}))
        with patch.object(orch, "_update_run"), patch.object(context_manager, "history_messages", return_value=[]), patch.object(
            context_manager, "previous_exchanges", return_value=[]
        ), patch.object(orchestrator.personalization, "preferences_text", return_value=""), patch.object(
            orchestrator.project_memory, "project_line", return_value=""
        ), patch.object(orchestrator, "hidden_skills_for", return_value=[]):
            result = await orch.run()
        self.assertEqual(result["status"], "completed")
        self.assertIn("7 - 3 = 4", result["text"])
        orch.rt.execute.assert_awaited_once_with("eval", {"expression": "7 - 3"})

    async def test_plot_accepts_only_a_published_image(self):
        task = "Plot y=x and save as figures/line.png"
        rt = SimpleNamespace(board=Blackboard(task), project_id="project", has_tool=lambda _: True,
                             step=AsyncMock(), skill_text=AsyncMock(return_value=""), schemas=lambda *_: [],
                             llm=AsyncMock(return_value="<tools_call><function=run_python><parameter=code>print(42)</parameter></function></tools_call>"),
                             execute=AsyncMock(return_value=ToolResult("42", data={"stdout": "42", "images": []})))
        with patch("app.agent.subagents.compute.data_preview", return_value=("", [])):
            outcome = await ComputeAgent().run(rt, task)
        self.assertFalse(outcome.ok)
        self.assertEqual(rt.execute.await_count, 3)
        self.assertEqual(rt.execute.await_args.args[1]["output_path"], "figures/line.png")
        self.assertFalse(rt.board.computed)

    async def test_checked_run_publishes_only_validated_artifacts(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            dest = workspace / "figure.png"
            dest.write_bytes(b"previous output")

            async def bad_run(*_args, cwd, **_kwargs):
                (cwd / "figure.png").write_bytes(b"corrupt")
                return ExecResult(0, "", "", False, "test")

            async def good_run(*_args, cwd, **_kwargs):
                Image.new("RGB", (4, 4), color="red").save(cwd / "figure.png")
                return ExecResult(0, "", "", False, "test")

            ctx = ToolContext("user", "project", "run", AsyncMock(), threading.Event())
            args = {"code": "print(42)", "output_type": "image", "output_path": "figure.png"}
            with patch.object(python_tool, "workspace_dir", return_value=workspace), patch.object(
                python_tool, "resolve_in_workspace", side_effect=lambda _, rel: (workspace / rel, rel)
            ), patch.object(python_tool, "register_file_artifact", new_callable=AsyncMock) as register, patch.object(
                python_tool.sandbox, "run_python", side_effect=bad_run
            ) as run:
                failed = await python_tool.run_python_tool(ctx, args)
                self.assertFalse(failed.ok)
                self.assertEqual(dest.read_bytes(), b"previous output")
                register.assert_not_awaited()
                run.side_effect = good_run
                register.return_value = {"path": "figure.png", "type": "image", "filename": "figure.png", "url": "/image"}
                succeeded = await python_tool.run_python_tool(ctx, args)
                self.assertTrue(succeeded.ok)
                self.assertEqual(succeeded.data["images"][0]["path"], "figure.png")
                self.assertEqual(python_tool.check_output(dest, "image", None), "")
                self.assertEqual(list((workspace / ".stemos" / "attempts").iterdir()), [])

    async def test_failed_listing_and_empty_answer_are_failures(self):
        rt = SimpleNamespace(board=Blackboard("List files"), execute=AsyncMock(return_value=ToolResult.failure("Tool disabled")))
        self.assertFalse((await BuildAgent().run(rt, rt.board.request)).ok)
        self.assertFalse(rt.board.exact)

        agent = AnswerAgent()
        rt = SimpleNamespace(board=Blackboard("Explain"), preferences="", project_line="", history=[], emit=AsyncMock())
        with patch.object(agent, "_stream", new=AsyncMock(return_value=("", ""))):
            handoff, text = await agent.run(rt, "Explain")
        self.assertFalse(handoff.ok)
        self.assertEqual(text, "")

    async def test_run_persists_and_emits_failed_and_partial_status(self):
        project = SimpleNamespace(id="project")
        active = orchestrator.ActiveRun("run", "project", "user")
        for expected in ("failed", "partial"):
            emit = AsyncMock()
            with patch.object(orchestrator, "get_engine", return_value=SimpleNamespace(available=False)):
                orch = orchestrator.Orchestrator(project=project, user_id="user", message="Plot y=x",
                                                 research_mode="auto", attachments=[], enabled_tools=[], emit=emit, active=active)
            orch.outcomes["compute"] = Handoff(False, "Plot failed.")
            if expected == "partial":
                orch.board.brief = "Retrieved findings."
            orch.run = AsyncMock(return_value=orch._result("Plot failed."))
            with patch.object(orch, "_finish_run") as finish, patch.object(
                orchestrator, "Orchestrator", return_value=orch
            ), patch.object(orchestrator, "_save_assistant", return_value="message"):
                await orchestrator.execute_run(project=project, user_id="user", message=orch.message,
                                               research_mode="auto", attachments=[], enabled_tools=[], emit=emit, active=active)
            self.assertEqual(finish.call_args.args[0], expected)
            self.assertEqual(emit.await_args.args[0]["type"], "done")
            self.assertEqual(emit.await_args.args[0]["status"], expected)
