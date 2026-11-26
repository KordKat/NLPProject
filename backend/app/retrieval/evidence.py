"""From retrieved documents to verifiable facts, and from generated text back to sources.

KeawGPT rewrites well but does not cite reliably, so the two jobs are split:
1. `select_facts`  - pick the sentences most relevant to the question (embeddings, no LLM).
2. the model rewrites those facts into one paragraph, without seeing any source labels.
3. `attribute`     - match every generated sentence to the fact that supports it, attach [n], and in
                     strict mode drop sentences no source supports (including changed numbers).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from app.retrieval.embedding import get_encoder

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[$])")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*%?")
_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_CITE = re.compile(r"\s*\[\d{1,3}\]")
_JUNK = re.compile(r"cookie|subscribe|sign in|log in|privacy policy|all rights reserved|javascript|click here|https?://", re.I)
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "which", "what", "how", "why", "does", "has", "have", "can", "not",
    "these", "those", "their", "its", "into", "also", "such", "find", "search", "look", "recent", "latest", "new", "newest", "paper", "papers",
    "research", "study", "studies", "explain", "compare", "work", "works", "use", "used", "using", "about", "give", "show", "tell", "write", "make",
    "short", "detailed", "report", "summary", "between", "other", "some", "any", "there", "when", "where", "who", "whom", "will", "would", "should",
}
# bibliography entries, acknowledgements and extraction debris are not findings
_NOT_FINDING = re.compile(
    r"\*\s*\[|\[\s*\d+\s*\]\s*[a-z]\.|\bet al\b|\bPhys\.|\bRev\.|\bJ\.\s|\bpp\.|\bvol\.|\bibid\b|\(\d{4}\)\s*\d+|\b\d+\s*\(\d{4}\)|\barXiv:"
    r"|\b(acknowledg|thank|grateful|dedicat|my (parent|thesis|family|advisor|supervisor)|funded by|supported by|grant no)"
    r"|\b(previous|next|this|following) (chapter|section|lecture)s?\b|\bas we saw\b|\babstract page\b",
    re.I,
)

_FUNCTION_WORDS = set(
    "the a an of in on at to by for from with as and or but that which who is are was were be been has have had it its this these those "
    "their they we can may will not than into through between while when because".split()
)
# questions whose answer must contain a figure: a paragraph without one has not answered them
_NEEDS_VALUE = re.compile(r"\b(version|how many|how much|what (year|date|percentage|score|accuracy|number|value)|melting point|boiling point|half-life)\b", re.I)
# "X is the latest version" is a claim of its own: a real number in the source does not support it
_CURRENCY = re.compile(r"\b(latest|newest|most recent|current|currently)\b", re.I)
# "Surface codes (30, 31, 32)" or "shown [4]": reference marks copied from the source. "SU(2)" stays.
_REF_MARKS = re.compile(r"\s+(?:\(\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*\)|\[\s*\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*\s*\])")
_LABEL = re.compile(r"^(Source|Note|Abstract|Summary|Description|Overview)\s*:\s*", re.I)
# page navigation and course blurbs are written as commands to the reader
_PROMO = re.compile(r"^(Learn|Find|Discover|Explore|Read|Get|See|Browse|Join|Check|Download|Watch|Try|Sign|Subscribe|Click|Visit)\b")
# the model saying the facts do not contain the answer
_NO_ANSWER = re.compile(
    r"\b(not|n't|no)\b[^.]{0,40}\b(stated|mentioned|specified|provided|given|included|available)\b[^.]{0,30}\b(facts?|text|information|context|sources?)\b"
    r"|\b(facts?|text|sources?)\b[^.]{0,30}\b(do not|does not|don't|doesn't)\b[^.]{0,20}\b(state|mention|specify|provide|contain|say)\b",
    re.I,
)
_META = re.compile(r"\b(provided|given|above|listed) (facts?|text|information|context)\b|\bthe facts\b", re.I)
# a paragraph cannot open on a word that refers to a sentence that is not there
_DANGLING = re.compile(r"^(Their|They|It|Its|This|These|Those|Such)\b")
_FIRST_PERSON = re.compile(r"\b(we|our)\b", re.I)
_CONNECTOR = re.compile(r"^(However|Additionally|Moreover|Furthermore|In addition|Also|Thus|Therefore|Consequently),?\s+(\w)")

STRICT_SIM = 0.52  # a summary sentence must be this close to a fact to be kept
SOFT_SIM = 0.62  # an answer sentence must be this close to a fact to receive a citation


@dataclass
class Fact:
    text: str
    source_key: str
    score: float
    vec: np.ndarray


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(re.sub(r"\s+", " ", text or "")) if s.strip()]


def _is_prose(s: str) -> bool:
    """Real sentences are full of function words; menus, tables of contents and link captions are not."""
    words = re.findall(r"[a-z]+", s.lower())
    return bool(words) and sum(w in _FUNCTION_WORDS for w in words) / len(words) >= 0.12


def _is_title(s: str) -> bool:
    """Page and paper titles with author names: mostly Capitalized Words."""
    words = re.findall(r"[A-Za-z][\w\-]*", s)
    return len(words) >= 5 and sum(w[0].isupper() for w in words) / len(words) >= 0.6


def _usable(s: str) -> bool:
    s = _LABEL.sub("", s)
    if not s or _PROMO.match(s) or "*" in s or _is_title(s):
        return False
    if not 40 <= len(s) <= 360 or not s[0].isupper() or s.endswith((":", "..", "…")) or _JUNK.search(s) or _NOT_FINDING.search(s):
        return False
    return sum(ch.isalpha() or ch == " " for ch in s) / len(s) >= 0.8 and _is_prose(s)


def clean_fact(sentence: str) -> str:
    return _REF_MARKS.sub("", _LABEL.sub("", sentence))


def states_no_answer(paragraph: str, question: str = "") -> bool:
    """True when the rewrite says the facts lack the answer, or a question asking for a figure got none."""
    sentences = split_sentences(paragraph)
    if not sentences or _NO_ANSWER.search(sentences[0]):
        return True
    if not _NEEDS_VALUE.search(question):
        return False
    return not _NUM.search(_CITE.sub("", paragraph)) or bool(_CURRENCY.search(question) and not _CURRENCY.search(paragraph))


def _stems(terms: set[str]) -> set[str]:
    """Crude stems, so network/networks and entangled/entanglement count as the same term."""
    return {t[:5] for t in terms}


def _on_topic(sentence: str, q_stems: set[str]) -> bool:
    """Embeddings alone call "efficiency curves of a detector" relevant to "solar cell efficiency":
    a fact must also share at least two of the question's key terms (or its only one)."""
    return len(_stems(_terms(sentence)) & q_stems) >= min(2, len(q_stems))


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    enc = get_encoder()
    return np.vstack([enc.encode(texts[i: i + batch]) for i in range(0, len(texts), batch)])


def select_facts(question: str, candidates: Iterable[dict[str, Any]], *, expected: str = "", per_doc: int = 3, limit: int = 12, min_score: float = 0.33) -> list[Fact]:
    """The sentences that best answer the question, at most `per_doc` from each document.

    A question and its answer are worded differently, so sentences are also compared with `expected`,
    a guess at what the answer looks like. The guess only ranks real sentences and is never shown."""
    q_vec = embed([question])[0]
    q_stems = _stems(_terms(question))
    # an explanation uses words the question does not ("forget gate", "cell state"), so the expected
    # answer widens the vocabulary, but a sentence must still share a term with the question itself
    topic = q_stems | _stems(_terms(expected))
    pool: list[tuple[str, str]] = []
    for c in candidates:
        text = " ".join([c.get("abstract") or ""] + [ch.get("text", "") for ch in c.get("chunks") or []])
        sents = [clean_fact(s) for s in dict.fromkeys(split_sentences(text)) if _usable(s) and _on_topic(s, topic) and _stems(_terms(s)) & q_stems]
        # a cheap lexical pass keeps the embedding step small on long pages
        sents = sorted(sents, key=lambda s: len(topic & _stems(_terms(s))), reverse=True)[:20]
        pool += [(s, c["source_key"]) for s in sents]
    if not pool:
        return []
    vecs = embed([s for s, _ in pool])
    sims = vecs @ q_vec
    rank = sims if not expected else 0.4 * sims + 0.6 * (vecs @ embed([expected])[0])
    if _NEEDS_VALUE.search(question):  # the answer is a figure: prefer sentences that contain one
        rank = rank + 0.1 * np.array([bool(_NUM.search(s)) for s, _ in pool], dtype=float)
    taken: dict[str, int] = {}
    facts: list[Fact] = []
    floor = float(rank.max()) - 0.2  # far below the best match is padding, not evidence
    for i in np.argsort(-rank).tolist():
        if len(facts) >= limit or rank[i] < floor:
            break
        s, key = pool[i]
        if sims[i] < min_score or taken.get(key, 0) >= per_doc or any(float(vecs[i] @ f.vec) > 0.9 for f in facts):
            continue
        taken[key] = taken.get(key, 0) + 1
        facts.append(Fact(s, key, round(float(rank[i]), 4), vecs[i]))
    return facts


def uncovered(queries: list[str], facts: list[Fact], threshold: float = 0.42) -> list[str]:
    """Queries no collected fact answers: these are the gaps worth another retrieval round."""
    if not queries:
        return []
    if not facts:
        return list(queries)
    sims = embed(queries) @ np.vstack([f.vec for f in facts]).T
    return [q for q, row in zip(queries, sims) if float(row.max()) < threshold]


def _numbers_supported(sentence: str, fact_text: str) -> bool:
    """The sentence's figures, and any claim that something is the latest, must be in the source."""
    have = set(_NUM.findall(fact_text))
    if _CURRENCY.search(sentence) and not _CURRENCY.search(fact_text):
        return False
    return all(n in have for n in _NUM.findall(sentence))


def _with_citation(sentence: str, numbers: list[int]) -> str:
    tag = "".join(f"[{n}]" for n in numbers)
    if sentence[-1:] in ".!?":
        return f"{sentence[:-1]} {tag}{sentence[-1]}"
    return f"{sentence} {tag}"


def _best_keys(row: np.ndarray, facts: list[Fact], threshold: float) -> list[int]:
    """Indexes of the best fact plus a second one from another source when nearly as close."""
    order = np.argsort(-row).tolist()
    best = order[0]
    if row[best] < threshold:
        return []
    out = [best]
    for j in order[1:]:
        if row[j] < max(threshold, row[best] - 0.05):
            break
        if facts[j].source_key != facts[best].source_key:
            out.append(j)
            break
    return out


def attribute_paragraph(paragraph: str, facts: list[Fact], cite: Callable[[str], int]) -> str:
    """Strict: keep only sentences a fact supports, each ending with its [n]. If too little survives
    for the amount of evidence, the best unused facts are added verbatim."""
    sentences = [s for s in split_sentences(_CITE.sub("", paragraph)) if len(s) > 15 and not _META.search(s) and not _NO_ANSWER.search(s)]
    if not sentences or not facts:
        return ""
    fact_vecs = np.vstack([f.vec for f in facts])
    sentence_vecs = embed(sentences)
    kept: list[tuple[str, np.ndarray, list[int]]] = []
    for sentence, vec, row in zip(sentences, sentence_vecs, sentence_vecs @ fact_vecs.T):
        hits = _best_keys(row, facts, STRICT_SIM)
        if hits and _numbers_supported(sentence, " ".join(facts[j].text for j in hits)):
            kept.append((sentence, vec, hits))
    while kept and _DANGLING.match(kept[0][0]):
        kept.pop(0)
    wanted = min(3, len(facts) // 2)
    for j, fact in enumerate(facts):
        if len(kept) >= wanted:
            break
        if _FIRST_PERSON.search(fact.text):  # "we examine…" reads wrongly when quoted as the answer
            continue
        if (kept or not _DANGLING.match(fact.text)) and not any(j in hits or float(vec @ fact.vec) > 0.8 for _, vec, hits in kept):
            kept.append((fact.text, fact.vec, [j]))
    if kept:
        first = _CONNECTOR.sub(lambda m: m.group(2).upper(), kept[0][0])
        kept[0] = (first, kept[0][1], kept[0][2])
    return " ".join(_with_citation(s, sorted({cite(facts[j].source_key) for j in hits})) for s, _, hits in kept)


def extractive_paragraph(facts: list[Fact], cite: Callable[[str], int], n: int = 3) -> str:
    """Fallback when the rewrite is unusable: the top facts verbatim, cited."""
    chosen = [f for f in facts if not _DANGLING.match(f.text) and not _FIRST_PERSON.search(f.text)][:n] or facts[:n]
    return " ".join(_with_citation(f.text, [cite(f.source_key)]) for f in chosen)


_SKIP_LINE = re.compile(r"^\s*(#|\||```|\$\$|>)")
_PROTECTED = re.compile(r"(```.*?```|`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]*\$)", re.S)
_FAKE_CITE = re.compile(r"\s*(?<![\w\])$])\[(\d{1,3})\](\([^)\s]*\))?")


def clean_citations(text: str, valid: set[int]) -> str:
    """Remove [n] that match no real source, leaving code and math (x[1], A[2]) untouched."""
    parts = _PROTECTED.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = _FAKE_CITE.sub(lambda m: m.group(0) if int(m.group(1)) in valid and not m.group(2) else "", parts[i])
    return "".join(parts)



def attribute_answer(text: str, facts: list[Fact], cite: Callable[[str], int], valid: set[int]) -> str:
    """Soft: never removes content. Drops [n] that do not exist and adds [n] to prose sentences a
    fact clearly supports; sentences from the model's own knowledge stay uncited."""
    text = clean_citations(text, valid)
    if not facts:
        return text
    fact_vecs = np.vstack([f.vec for f in facts])
    out: list[str] = []
    in_block = False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "$$")):
            in_block = not in_block if line.count("```") + line.count("$$") == 1 else in_block
            out.append(line)
            continue
        if in_block or _SKIP_LINE.match(line) or len(line) < 40:
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        parts = split_sentences(line)
        need = [i for i, s in enumerate(parts) if len(s) > 30 and not _CITE.search(s)]
        if need:
            sims = embed([parts[i] for i in need]) @ fact_vecs.T
            for i, row in zip(need, sims):
                hits = _best_keys(row, facts, SOFT_SIM)
                if hits and _numbers_supported(parts[i], " ".join(facts[j].text for j in hits)):
                    parts[i] = _with_citation(parts[i], sorted({cite(facts[j].source_key) for j in hits}))
        out.append(indent + " ".join(parts))
    return "\n".join(out)
