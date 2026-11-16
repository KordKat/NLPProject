"""Chunk + paper metadata access for the FAISS corpus.

documents.jsonl holds one chunk per line; `document_offsets.bin` maps vector id → byte offset
(little-endian uint64), exactly as in search_faiss.py. Chunk metadata carries the arXiv id
(`metadata.id`), `created` date and chunk indices — but no title/authors. We therefore build a small
SQLite sidecar (paper id → title, byline, abstract, date, chunk-id range) once, by streaming the
JSONL file. The sidecar lives in storage/cache (the index directory may be read-only) and is keyed by
the index manifest so it is rebuilt automatically when the corpus changes. This scales to the full
production corpus (it is a single sequential pass, can be prebuilt with
`python -m app.retrieval.paper_store`), and never requires rebuilding the FAISS index.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings

log = logging.getLogger("stemos.papers")
settings = get_settings()

AFFILIATION_WORDS = re.compile(
    r"universit|institut|department|dept\.|laborator|school|college|cent(er|re)|@|e-?mail|faculty|academy|"
    r"corporation|inc\.|ltd|research|national|street|avenue|\d{4,}",
    re.I,
)
NAME_LINE = re.compile(r"^[A-Z][A-Za-z\.\-'’` ]+(?:(?:,| and |&)\s*[A-Z][A-Za-z\.\-'’` ]+)*\s*\**$")
FOOTNOTE_CUT = re.compile(r"\d{2,}[A-Z].*$")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_front_matter(text: str) -> dict[str, Any]:
    """Heuristically extract title / byline / abstract from the first chunk of a paper."""
    lines = [l.rstrip() for l in text.splitlines()]
    title = ""
    title_end = 0
    for i, line in enumerate(lines[:15]):
        if line.startswith("#") and not re.match(r"^#+\s*abstract", line, re.I):
            title = line.lstrip("#").strip()
            title_end = i
            # continuation lines of a wrapped title
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith("#"):
                    break
                cut = FOOTNOTE_CUT.sub("", nxt).strip()
                was_cut = cut != nxt
                if len(cut) > 90 or AFFILIATION_WORDS.search(cut) or NAME_LINE.match(cut) and len(cut.split()) <= 4 and not was_cut:
                    break
                title = f"{title} {cut}"
                title_end = j
                if was_cut:
                    break
            break
    title = FOOTNOTE_CUT.sub("", title).strip() or ""

    abstract = ""
    abs_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*(#+\s*)?abstract\b", line, re.I):
            abs_idx = i
            break
    if abs_idx is not None:
        first = re.sub(r"^\s*(#+\s*)?abstract[:.\s]*", "", lines[abs_idx], flags=re.I)
        body = [first] if first else []
        for line in lines[abs_idx + 1:]:
            if line.startswith("#"):
                break
            body.append(line)
        abstract = _clean(" ".join(body))[:2000]

    byline_lines: list[str] = []
    stop = abs_idx if abs_idx is not None else min(len(lines), title_end + 8)
    for line in lines[title_end + 1: stop]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        byline_lines.append(s)
    authors: list[str] = []
    for s in byline_lines[:6]:
        if AFFILIATION_WORDS.search(s) or len(s) > 140 or any(ch.isdigit() for ch in s):
            continue
        if NAME_LINE.match(s):
            for part in re.split(r",| and |&", s):
                p = part.strip(" *")
                if 2 <= len(p.split()) <= 5:
                    authors.append(p)
        if len(authors) >= 12:
            break
    return {
        "title": _clean(title)[:400],
        "authors": authors[:12],
        "byline": _clean(" · ".join(byline_lines[:4]))[:400],
        "abstract": abstract,
    }


class PaperStore:
    def __init__(self, faiss_dir: Path):
        self.dir = faiss_dir
        self.docs_path = faiss_dir / "documents.jsonl"
        self.offsets_path = faiss_dir / "document_offsets.bin"
        self.manifest: dict[str, Any] = {}
        mpath = faiss_dir / "manifest.json"
        if mpath.exists():
            self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        key = hashlib.sha1(json.dumps(self.manifest, sort_keys=True).encode() + str(faiss_dir).encode()).hexdigest()[:12]
        self.meta_db = settings.cache_path / f"paper_meta_{key}.sqlite"
        self._local = threading.local()
        self.status = "missing" if not self.docs_path.exists() else ("ready" if self._is_built() else "building")
        self._build_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ chunks
    def read_document(self, doc_id: int) -> dict[str, Any]:
        with self.offsets_path.open("rb") as f:
            f.seek(doc_id * 8)
            raw = f.read(8)
        if len(raw) != 8:
            raise IndexError(f"No document offset for id {doc_id}")
        offset = struct.unpack("<Q", raw)[0]
        with self.docs_path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line.decode("utf-8", "replace"))

    # ------------------------------------------------------------------ sidecar
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.meta_db), check_same_thread=False)
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def _is_built(self) -> bool:
        if not self.meta_db.exists():
            return False
        try:
            c = sqlite3.connect(str(self.meta_db))
            row = c.execute("SELECT value FROM info WHERE key='complete'").fetchone()
            c.close()
            return bool(row and row[0] == "1")
        except sqlite3.Error:
            return False

    def ensure_built_async(self) -> None:
        if self.status != "building" or self._build_thread is not None:
            return
        self._build_thread = threading.Thread(target=self.build, daemon=True, name="paper-meta-build")
        self._build_thread.start()

    def build(self) -> None:
        log.info("Building paper metadata sidecar %s from %s", self.meta_db, self.docs_path)
        tmp = self.meta_db.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        c = sqlite3.connect(str(tmp))
        c.execute("CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT)")
        c.execute(
            "CREATE TABLE papers (paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, byline TEXT, abstract TEXT,"
            " created TEXT, license TEXT, first_doc INTEGER, last_doc INTEGER, n_chunks INTEGER)"
        )
        papers: dict[str, dict[str, Any]] = {}
        batch: list[tuple] = []

        def flush_done(keep: set[str]) -> None:
            for pid in [p for p in papers if p not in keep]:
                d = papers.pop(pid)
                batch.append((pid, d["title"], json.dumps(d["authors"]), d["byline"], d["abstract"], d["created"], d["license"], d["first"], d["last"], d["n"]))
            if len(batch) >= 2000:
                c.executemany("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
                batch.clear()

        try:
            with self.docs_path.open("rb") as f:
                current: Optional[str] = None
                for raw in f:
                    try:
                        doc = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    md = doc.get("metadata", {}) or {}
                    pid = str(md.get("id", ""))
                    if not pid:
                        continue
                    did = int(doc.get("id", -1))
                    entry = papers.get(pid)
                    if entry is None:
                        fm = parse_front_matter(doc.get("text", ""))
                        entry = {
                            **fm,
                            "created": str(md.get("created", "")),
                            "license": str(md.get("license", "")),
                            "first": did,
                            "last": did,
                            "n": 0,
                        }
                        papers[pid] = entry
                    elif not entry["abstract"] and entry["n"] < 3:
                        fm = parse_front_matter(doc.get("text", ""))
                        if fm["abstract"]:
                            entry["abstract"] = fm["abstract"]
                    entry["first"] = min(entry["first"], did)
                    entry["last"] = max(entry["last"], did)
                    entry["n"] += 1
                    if pid != current:
                        # papers are written contiguously; flush finished ones periodically
                        if len(papers) > 64:
                            flush_done({pid})
                        current = pid
            flush_done(set())
            if batch:
                c.executemany("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
            c.execute("CREATE INDEX ix_created ON papers(created)")
            c.execute("INSERT INTO info VALUES ('complete','1')")
            c.commit()
            c.close()
            tmp.replace(self.meta_db)
            self.status = "ready"
            log.info("Paper metadata sidecar ready")
        except Exception:
            c.close()
            self.status = "error"
            log.exception("Paper metadata build failed")

    def paper(self, paper_id: str) -> Optional[dict[str, Any]]:
        if self.status != "ready":
            return None
        row = self._conn().execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["authors"] = json.loads(d.get("authors") or "[]")
        return d

    def paper_or_fallback(self, paper_id: str, chunk: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Metadata for a paper; falls back to parsing the given chunk while the sidecar builds."""
        meta = self.paper(paper_id)
        if meta:
            return meta
        md = (chunk or {}).get("metadata", {}) if chunk else {}
        fm = parse_front_matter(chunk.get("text", "")) if chunk and md.get("chunk_index", 1) == 0 else {"title": "", "authors": [], "byline": "", "abstract": ""}
        return {"paper_id": paper_id, **fm, "created": str(md.get("created", "")), "license": str(md.get("license", "")), "first_doc": None, "last_doc": None, "n_chunks": None}

    def first_chunk_text(self, paper_id: str, max_chars: int = 2500) -> str:
        meta = self.paper(paper_id)
        if not meta or meta.get("first_doc") is None:
            return ""
        try:
            return self.read_document(int(meta["first_doc"])).get("text", "")[:max_chars]
        except (IndexError, OSError, json.JSONDecodeError):
            return ""

    def search_titles(self, terms: list[str], limit: int = 5) -> list[dict[str, Any]]:
        if self.status != "ready" or not terms:
            return []
        conds = " OR ".join(["lower(title) LIKE ?"] * len(terms))
        rows = self._conn().execute(f"SELECT * FROM papers WHERE {conds} LIMIT ?", [f"%{t.lower()}%" for t in terms] + [limit]).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["authors"] = json.loads(d.get("authors") or "[]")
            out.append(d)
        return out


if __name__ == "__main__":  # prebuild the sidecar: python -m app.retrieval.paper_store
    logging.basicConfig(level=logging.INFO)
    store = PaperStore(settings.faiss_path)
    store.build()
