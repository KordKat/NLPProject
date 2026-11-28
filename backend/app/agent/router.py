"""The main agent's decisions.

Two tiers, cheapest first:
1. Explicit signals in the request (asks for a file, a plot, recent facts, earlier work) give the
   ordered list of agents with no model call.
2. Anything else is one KeawGPT forward pass (~0.3 s): the native tool call is prefilled and the model
   only picks the agent. Search runs only when one of the two tiers decides the request needs it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agent.model_engine import KeawEngine
from app.agent.prompts import MAIN_AGENT, STANDALONE_SHOTS, STANDALONE_SYSTEM
from app.agent.runtime import tool_turns
from app.agent.subagents import ANSWER_AGENT, SUB_AGENTS

CHOICE_PREFILL = "\n<tools_call>\n<function="

_CHAT = re.compile(r"^\W*(hi|hey|hello|yo|sup|thanks|thank you|thx|ok|okay|cool|nice|great|good (morning|afternoon|evening|night)|bye|goodbye|how are you|who are you|what can you do)\b[\w\s,!.?']{0,24}$", re.I)
_REWRITE = re.compile(r"^\W*(rewrite|rephrase|shorten|summari[sz]e|simplify|translate|make|explain)\b.{0,30}\b(that|this|it|above|your answer)\b", re.I)
CONTINUATION = re.compile(r"^\W*(continue|go on|keep going|carry on|proceed|go ahead|do it|try again|retry|yes(,? please)?(,? do it)?|ok(ay)?,? (do it|continue|go ahead))\W*$", re.I)

_RECALL = re.compile(
    r"\b(we|our)\b[^.?!]{0,40}\b(found|concluded|discussed|collected|decided|saved|covered|did)\b"
    r"|\bdid we\b[^.?!]{0,30}\b(decide|conclude|discuss|find|collect|save|cover|choose|pick)\b"
    r"|\b(first|second|third|fourth|last|previous) (paper|source|result|method|article|study)\b"
    r"|\b(continue|resume)\b[^.?!]{0,25}\b(previous|earlier|last)\b"
    r"|\b(in|from) this project\b"
    r"|\b(remember|record) that\b|\badd\b[^.?!]{0,40}\bknowledge graph\b",
    re.I,
)
_RESEARCH = re.compile(
    r"\b(latest|newest|recent|recently|currently|nowadays|this (year|month|week)|news|state[- ]of[- ]the[- ]art|20(2[3-9]|3\d))\b"
    r"|\bcurrent (world |state|record|status|version|leader|best|price)"
    r"|\b(search|look up|find|browse)\b[^.?!]{0,60}\b(web|internet|online|google|papers?|articles?|studies|sources?)\b"
    r"|\b(web ?search|google|arxiv|literature|peer[- ]reviewed)\b"
    r"|\b(with|include|add|give)\b[^.?!]{0,20}\b(citations?|references|sources)\b|\bcite\b"
    r"|\bpapers?\b[^.?!]{0,25}\b(report|reported|show|showed|claim|claimed|propose|proposed|found)\b"
    r"|(^|\b(please|and|then)\s+)(research|investigate)\b"
    r"|\b(according to|what does the research|what do studies)\b"
    r"|\bwho (won|is the current)\b"
    r"|\b(photos?|pictures?|images?) of\b",
    re.I,
)
_COMPUTE = re.compile(
    r"(?<![\w.])\d[\d.,]*\s*(\*\*|[+*/^×÷])\s*\(?\d"
    r"|\b(plot|graph of|histogram|scatter ?plot|heatmap|bar chart|line chart|visuali[sz]e|simulate|simulation|monte carlo)\b"
    r"|\b(integral|integrate|derivative|differentiate|eigenvalues?|determinant|inverse of|solve)\b[^.?!]{0,50}[\d=^\[]"
    r"|\b(calculate|compute|evaluate)\b[^.?!]{0,50}\d"
    r"|\b(analy[sz]e|statistics|regression|t-test|anova|correlation)\b[^.?!]{0,50}\.(csv|tsv|xlsx|json)\b"
    r"|\b(analy[sz]e|test)\b[^.?!]{0,30}\b(dataset|data file|csv)\b"
    r"|\b(run|execute)\b[^.?!]{0,30}(`|\.py\b)"
    # machine learning is computed, not built: "train … and report accuracy" asks for numbers, not a report file
    r"|\b(train|fit|evaluate|benchmark|cross-?validate)\w*\b[^.?!]{0,60}\b(models?|classifiers?|regressors?|dataset|data|accuracy|sklearn|scikit)\b"
    r"|\b(sklearn|scikit-learn|xgboost|lightgbm)\b",
    re.I,
)
_SUBTRACTION = re.compile(r"^\s*(?:what is\s+)?[-−]?\d+(?:\.\d+)?\s*[-−]\s*[-−]?\d+(?:\.\d+)?\s*[?=]?\s*$", re.I)
_DATA_REQUEST = re.compile(
    r"\b(explain|summari[sz]e|inspect|describe|analy[sz]e|profile|compare)\b[\s\S]*\.(csv|tsv|xlsx|json|parquet|npy)\b",
    re.I,
)
_FILE_NOUN = r"(pptx|power ?point|slides?|slide ?deck|presentation|landing page|web ?page|website|html|script|notebook|file|document|\.(?:py|md|csv|json|txt|tex|html|docx))"
_BUILD = re.compile(
    rf"\b(make|create|build|generate|write|design|produce|prepare|save|draft|turn|convert|export)\b[^.?!\n]{{0,60}}{_FILE_NOUN}"
    rf"|{_FILE_NOUN}[^.?!\n]{{0,30}}\b(save|saved)\b"
    r"|\b(what|which|list|show)\b[^.?!]{0,20}\bfiles?\b"
    r"|\b(delete|remove|rename|edit|update)\b[^.?!]{0,30}\.\w{2,5}\b"
    r"|^\W*(please\s+)?(delete|remove|erase)\b[^.?!]{0,30}\b(files?|slides?|deck|presentation|report|charts?|figures?|plots?|notes|script)\b"
    r"|\b(convert|export)\b[^.?!]{0,30}\bpdf\b",
    re.I,
)
# "write a function that…" is answered in chat: code only becomes a file when the user asks for one
_SNIPPET = re.compile(r"^\W*(please\s+)?(write|show me|give me)(\s+(an?|the|some))?(\s+[\w+#.\-]+){0,2}?\s+(function|class|method|snippet|regex|regexp|one-liner|sql query|code)\b", re.I)
_WANTS_FILE = re.compile(r"\b(save|saved|store)\b|\b(to|in|as|into) (a |the )?file\b|\.(py|js|ts|md|html|txt|csv|json)\b|\b(presentation|slides?|pptx|report|web ?page|website)\b", re.I)
_CLAUSE = re.compile(
    r"(?:[.;]\s+|,?\s+(?:and then|then|and)\s+)"
    r"(?=(?:please\s+)?(?:research|investigate|find|search|look up|plot|graph|draw|compute|calculate|evaluate|simulate|visuali[sz]e|solve|analy[sz]e)\b)",
    re.I,
)
_FRESH = re.compile(r"\b(latest|newest|recent|new papers?|web|online|internet|search)\b", re.I)
_LIST_FILES = re.compile(r"\b(what|which|list|show)\b[^.?!]{0,20}\bfiles?\b", re.I)
_REPORT = re.compile(r"\b(write|create|make|generate|prepare|produce|draft)\b[^.?!\n]{0,40}\b(report|white ?paper|literature review)\b", re.I)


@dataclass
class Route:
    agents: list[str] = field(default_factory=list)  # ordered, from explicit signals
    direct: bool = False  # answer immediately (greeting, rewrite of the last answer)
    explicit_research: bool = False


def route_by_rules(text: str, has_history: bool = False) -> Route:
    head = text.strip()
    if _CHAT.match(head):
        return Route(direct=True)
    if _SNIPPET.match(head) and not _WANTS_FILE.search(head):
        return Route(direct=True)
    agents: list[str] = []
    if _RECALL.search(head) and not _LIST_FILES.search(head):
        agents.append("recall")
    # a question about earlier work is answered from memory unless it also asks for something new
    explicit_research = bool(_RESEARCH.search(head))
    if (explicit_research or _REPORT.search(head)) and ("recall" not in agents or _FRESH.search(head)):
        agents.append("research")
    if _COMPUTE.search(head) or _SUBTRACTION.fullmatch(head) or _DATA_REQUEST.search(head):
        agents.append("compute")
    if _BUILD.search(head) or _REPORT.search(head):
        agents.append("build")
    return Route(agents=agents, direct=not agents and has_history and bool(_REWRITE.match(head)), explicit_research=explicit_research)


def agent_schema(agent: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": agent.name, "description": agent.description, "parameters": {
        "properties": {"task": {"type": "string", "description": "What this agent must do, in one sentence."}},
        "required": ["task"], "title": "parameters", "type": "object"}, "strict": True}}


async def choose_next(engine: KeawEngine, request: str, done: list[tuple[str, str]], options: list[str]) -> tuple[str, dict[str, float]]:
    """One forward pass: which agent should run next?"""
    agents = {**SUB_AGENTS, ANSWER_AGENT.name: ANSWER_AGENT}
    tools = [agent_schema(agents[o]) for o in options]
    messages = [{"role": "system", "content": MAIN_AGENT}, {"role": "user", "content": request[:1500]}]
    for name, note in done:
        messages += tool_turns(name, {"task": request[:120]}, note)
    probs = await engine.choose(messages, options, tools=tools, prefill=CHOICE_PREFILL)
    return max(probs, key=probs.get), probs


REFERS_BACK = re.compile(r"\b(this(?!\s+(project|conversation|chat|workspace|session))|that|it|its|these|those|them|they|above|same|previous|earlier|last one)\b", re.I)
_WORDS = re.compile(r"[a-z]{4,}")
_SHOT_WORDS = {w for conv, _, rewritten in STANDALONE_SHOTS for w in _WORDS.findall((conv + " " + rewritten).lower())}


def needs_standalone(text: str, has_history: bool) -> bool:
    """A short follow-up that points at the previous turn ("summarize this", "a report on it")."""
    if not has_history or len(text) > 300 or CONTINUATION.match(text) or not REFERS_BACK.search(text):
        return False
    return not route_by_rules(text, has_history=True).direct  # greetings and "rewrite that" use the history directly


async def standalone(rt: Any, request: str, prev_request: str, prev_answer: str) -> str:
    """Rewrites a follow-up into a request every sub-agent understands on its own."""
    def conversation(user: str, assistant: str) -> str:
        return f"User: {user[:300]}\nAssistant: {assistant[:700]}"

    messages = [{"role": "system", "content": STANDALONE_SYSTEM}]
    for conv, latest, rewritten in STANDALONE_SHOTS:
        messages += [{"role": "user", "content": f"CONVERSATION:\n{conv}\nLATEST: {latest}"}, {"role": "assistant", "content": rewritten}]
    messages.append({"role": "user", "content": f"CONVERSATION:\n{conversation(prev_request, prev_answer)}\nLATEST: {request}"})
    out = await rt.llm(messages, max_new_tokens=80, temperature=0.0)
    line = next((l.strip().strip('"') for l in out.strip().splitlines() if l.strip()), "")
    # keep the original when the rewrite is unusable or dropped a number the user gave ("5 slides")
    if not 8 <= len(line) <= 400 or not set(re.findall(r"\d+", request)) <= set(re.findall(r"\d+", line)):
        return request
    # with nothing to refer to, a small model fills in the example's topic ("Attention Is All You Need")
    novel = set(_WORDS.findall(line.lower())) - set(_WORDS.findall(f"{request} {prev_request} {prev_answer}".lower()))
    if len(novel & _SHOT_WORDS) >= 2:
        return request
    return line


def split_task(agent: str, request: str) -> str:
    """The clause of a multi-part request that belongs to this agent, or "" if it cannot be separated."""
    signals = {"research": (_RESEARCH,), "compute": (_COMPUTE, _SUBTRACTION, _DATA_REQUEST)}.get(agent, ())
    clauses = [c.strip(" .;,") for c in _CLAUSE.split(request) if c.strip(" .;,")]
    if not signals or len(clauses) < 2:
        return ""
    matching = [c for c in clauses if any(signal.search(c) for signal in signals)]
    other = [c for c in clauses if c not in matching]
    if not matching or not other:
        return ""
    return " and ".join(matching) + "\n\nRequest context: " + " and ".join(other)


async def write_task(rt: Any, agent: str, request: str, shared: bool = False) -> str:
    """Put the agent's clauses first while keeping the rest available as context."""
    if agent in ("build", "recall"):
        return request
    if shared and (part := split_task(agent, request)):
        return part
    return request
