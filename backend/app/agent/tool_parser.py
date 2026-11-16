"""KeawGPT tool-call protocol: parsing + streaming-safe visible-text extraction.

KeawGPT's chat template (chat_template.jinja in the model repo) instructs the model to emit:

    <tools_call>
    <function=example_function_name>
    <parameter=example_parameter_1>
    value_1
    </parameter>
    </function>
    </tools_call>

The parser below accepts that format (plus the `<tool_call>` spelling and `:` separators seen in the
reference implementation), multiple <function> blocks in one call, and a missing closing tag when
generation was cut by a stop string. A Qwen-style JSON body is accepted as a fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

TOOL_BLOCK_RE = re.compile(r"<tools?_call>(.*?)(?:</tools?_call>|$)", re.DOTALL)
FUNCTION_RE = re.compile(r"<function[=:]\s*([A-Za-z0-9_\-\.]+)\s*>(.*?)(?:</function\s*>|(?=<function[=:])|$)", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter[=:]\s*([A-Za-z0-9_\-]+)\s*>(.*?)(?:</parameter\s*>|(?=<parameter[=:])|(?=</function)|$)", re.DOTALL)

TOOL_START_MARKERS = ("<tools_call", "<tool_call", "<function=")
REASONING_OPEN_RE = re.compile(r"^\s*<(reasoning|think)>", re.DOTALL)
REASONING_CLOSE_RE = re.compile(r"</(reasoning|think)>")
_ALL_TAGS = ["<reasoning>", "</reasoning>", "<think>", "</think>", "<tools_call>", "<tool_call>", "<function="]
_TAG_PREFIXES = {t[:i] for t in _ALL_TAGS for i in range(1, len(t))}


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def hold_partial_tag(text: str) -> str:
    """If the text ends with an unfinished tag prefix (split across tokens), hold it back."""
    for i in range(min(len(text), 14), 0, -1):
        if text[-i:] in _TAG_PREFIXES:
            return text[:-i]
    return text


def split_reasoning(raw: str, final: bool = False) -> tuple[str, str]:
    """Return (reasoning, answer). The generation prompt already ends in `<reasoning>`, so the opening
    tag is usually absent from the generated text; `</reasoning>` marks the start of the answer."""
    m_open = REASONING_OPEN_RE.match(raw)
    m_close = REASONING_CLOSE_RE.search(raw)
    if m_close:
        reasoning = raw[(m_open.end() if m_open else 0): m_close.start()]
        answer = raw[m_close.end():]
        if not final:
            answer = hold_partial_tag(answer)
        return reasoning.strip(), answer.lstrip("\n ")
    if m_open:
        reasoning = raw[m_open.end():]
        return (reasoning if final else hold_partial_tag(reasoning)).lstrip("\n"), ""
    return "", raw if final else hold_partial_tag(raw)


def visible_answer(answer: str) -> tuple[str, bool]:
    """Cut the answer at the first tool-call marker. Returns (visible_text, tool_call_started)."""
    idx = -1
    for marker in TOOL_START_MARKERS:
        i = answer.find(marker)
        if i != -1 and (idx == -1 or i < idx):
            idx = i
    if idx != -1:
        return answer[:idx], True
    return hold_partial_tag(answer), False


def _strip_value(v: str) -> str:
    v = v.strip("\n")
    # values are typically on their own lines; strip one level of surrounding whitespace
    return v.strip()


def _parse_json_body(body: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    try:
        obj = json.loads(body.strip())
    except (json.JSONDecodeError, ValueError):
        return calls
    items = obj if isinstance(obj, list) else [obj]
    for it in items:
        if isinstance(it, dict) and "name" in it:
            args = it.get("arguments") or it.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            calls.append(ToolCall(name=str(it["name"]), arguments=dict(args), raw=body))
    return calls


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    blocks = [m.group(1) for m in TOOL_BLOCK_RE.finditer(text)]
    if not blocks and "<function=" in text:
        blocks = [text[text.find("<function="):]]
    for block in blocks:
        found = False
        for fm in FUNCTION_RE.finditer(block):
            found = True
            name = fm.group(1).strip()
            body = fm.group(2)
            args = {pm.group(1).strip(): _strip_value(pm.group(2)) for pm in PARAMETER_RE.finditer(body)}
            calls.append(ToolCall(name=name, arguments=args, raw=fm.group(0)))
        if not found:
            calls.extend(_parse_json_body(block))
    return calls


def coerce_arguments(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Convert string parameter values to the JSON-schema types declared by the tool."""
    props = (schema or {}).get("properties", {})
    out: dict[str, Any] = {}
    for key, val in args.items():
        spec = props.get(key, {})
        typ = spec.get("type")
        if not isinstance(val, str):
            out[key] = val
            continue
        s = val.strip()
        try:
            if typ == "integer":
                out[key] = int(float(s))
            elif typ == "number":
                out[key] = float(s)
            elif typ == "boolean":
                out[key] = s.lower() in ("true", "1", "yes", "y")
            elif typ in ("array", "object"):
                try:
                    out[key] = json.loads(s)
                except json.JSONDecodeError:
                    if typ == "array":
                        out[key] = [x.strip().strip("\"'") for x in s.strip("[]").split(",") if x.strip()]
                    else:
                        raise
            else:
                out[key] = val.strip("\n") if key in ("content", "code") else s
        except (ValueError, json.JSONDecodeError):
            out[key] = s
    return out


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Serialize a call in KeawGPT's native format (used in history and ICL examples)."""
    lines = ["<tools_call>", f"<function={name}>"]
    for k, v in arguments.items():
        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"<parameter={k}>\n{v}\n</parameter>")
    lines += ["</function>", "</tools_call>"]
    return "\n".join(lines)


def extract_json(text: str) -> Optional[Any]:
    """Robustly pull the first JSON object/array out of model output."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = t.find(open_ch)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(t)):
                c = t[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = t[start: i + 1]
                        try:
                            return json.loads(candidate)
                        except (json.JSONDecodeError, ValueError):
                            # tolerate trailing commas
                            try:
                                return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                            except (json.JSONDecodeError, ValueError):
                                break
            start = t.find(open_ch, start + 1)
    return None
