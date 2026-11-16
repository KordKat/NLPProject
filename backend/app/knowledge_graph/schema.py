"""Graph schema: entity types, relationship whitelist, key normalisation.

Every entity node carries the shared label `:Entity`, a type label (e.g. `:Concept`) and a unique
`uid = "<Type>:<normalised name>"` (papers use `Paper:<arxiv id>`).

Scoping / isolation
* Knowledge extracted from public papers is *shared* (`scope = "global"`) — science is common ground.
* Knowledge from a user's own notes or prompts is *private*: relationship `scope = "project"` with
  `project_id`; private-only entities carry `private_project = <project id>`.
* `(:Project {id, owner_id})-[:EXPLORED]->(:Entity)` records what each project touched; it is how
  the project graph view is assembled. Retrieval only ever returns global relationships or the
  caller's own project relationships.
"""
from __future__ import annotations

import re
import unicodedata

ENTITY_TYPES = [
    "Paper",
    "Author",
    "Concept",
    "ScientificField",
    "Method",
    "Algorithm",
    "Dataset",
    "ResearchProblem",
    "Equation",
    "Model",
    "Note",
]

RELATIONSHIPS = {
    "RELATED_TO": "generic, scientifically meaningful association",
    "CITES": "paper cites paper",
    "AUTHORED_BY": "paper authored by author",
    "INTRODUCES": "paper introduces a concept/method/model/dataset",
    "USES_METHOD": "paper/model/method uses a method or algorithm",
    "USES_DATASET": "paper/method evaluated on a dataset",
    "SOLVES": "method/model addresses a research problem",
    "EXTENDS": "method/model extends or generalises another",
    "CONTRADICTS": "finding contradicts another",
    "DERIVED_FROM": "equation/method derived from another",
    "BELONGS_TO": "concept/method belongs to a scientific field",
    "MENTIONS": "paper discusses a concept (weak, provenance only)",
}

_ALIASES = {
    "USES": "USES_METHOD",
    "USE": "USES_METHOD",
    "USED_BY": "USES_METHOD",
    "PROPOSES": "INTRODUCES",
    "INTRODUCED": "INTRODUCES",
    "IS_PART_OF": "BELONGS_TO",
    "PART_OF": "BELONGS_TO",
    "IS_A": "BELONGS_TO",
    "ADDRESSES": "SOLVES",
    "EVALUATED_ON": "USES_DATASET",
    "BASED_ON": "DERIVED_FROM",
    "IMPROVES": "EXTENDS",
    "RELATED": "RELATED_TO",
}


def normalize_relationship(rel: str) -> str | None:
    r = re.sub(r"[^A-Z_]", "", (rel or "").upper().replace(" ", "_").replace("-", "_"))
    r = _ALIASES.get(r, r)
    return r if r in RELATIONSHIPS else None


def normalize_type(t: str) -> str:
    t = (t or "").strip()
    for et in ENTITY_TYPES:
        if et.lower() == t.lower().replace(" ", "").replace("_", ""):
            return et
    low = t.lower()
    if "field" in low or "domain" in low:
        return "ScientificField"
    if "data" in low:
        return "Dataset"
    if "algo" in low:
        return "Algorithm"
    if "problem" in low or "task" in low:
        return "ResearchProblem"
    if "equation" in low or "formula" in low:
        return "Equation"
    if "model" in low or "architecture" in low:
        return "Model"
    if "method" in low or "technique" in low:
        return "Method"
    return "Concept"


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKC", name or "").strip().lower()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[^\w\s\-\+\.\(\)/']", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .-")
    # light plural folding for concept keys
    if len(s) > 4 and s.endswith("s") and not s.endswith(("ss", "is", "us", "ys", "ics")) and " " not in s[-3:]:
        s = s[:-1]
    return s[:200]


def make_uid(entity_type: str, name: str) -> str:
    if entity_type == "Paper":
        return f"Paper:{name.strip()}"
    return f"{entity_type}:{normalize_name(name)}"
