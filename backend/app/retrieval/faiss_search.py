"""FAISS ArXiv search, compatible with the existing index produced by build_faiss.py / search_faiss.py.

Index: IndexFlatIP over L2-normalised all-MiniLM-L6-v2 embeddings; vector id == documents.jsonl id.
We over-fetch chunks, group them by arXiv paper id, apply a minimum-similarity threshold (so we never
pad results with unrelated papers) and optional metadata filters, then resolve paper metadata.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from app.config import get_settings
from app.retrieval.embedding import get_encoder
from app.retrieval.paper_store import PaperStore

log = logging.getLogger("stemos.faiss")
settings = get_settings()


class ArxivIndex:
    def __init__(self) -> None:
        self.dir = settings.faiss_path
        self.index = None
        self.store: Optional[PaperStore] = None
        self.status = "not_loaded"
        self.error = ""
        self.manifest: dict[str, Any] = {}
        self._cache: "OrderedDict[str, list[dict]]" = OrderedDict()
        self._cache_lock = threading.Lock()

    def load(self) -> None:
        try:
            import faiss

            idx_path = self.dir / "faiss.index"
            if not idx_path.exists():
                raise FileNotFoundError(f"FAISS index not found at {idx_path}")
            mpath = self.dir / "manifest.json"
            if mpath.exists():
                self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
                mid = self.manifest.get("model_id")
                if mid and mid != settings.embedding_model:
                    log.warning(
                        "Index manifest model_id=%s differs from EMBEDDING_MODEL=%s — using the manifest model for consistency",
                        mid, settings.embedding_model,
                    )
                    settings.embedding_model = mid
            # memory-map large indexes instead of loading them into RAM
            size = idx_path.stat().st_size
            flags = faiss.IO_FLAG_MMAP if size > 2 * 1024**3 else 0
            self.index = faiss.read_index(str(idx_path), flags) if flags else faiss.read_index(str(idx_path))
            dim = int(self.manifest.get("embedding_dimension", self.index.d))
            if dim != self.index.d:
                raise ValueError(f"Manifest dimension {dim} != index dimension {self.index.d}")
            self.store = PaperStore(self.dir)
            self.store.ensure_built_async()
            # warm the encoder and verify dimensions
            enc = get_encoder()
            v = enc.encode(["warmup"])
            if v.shape[1] != self.index.d:
                raise ValueError(f"Encoder dimension {v.shape[1]} != index dimension {self.index.d}")
            self.status = "ready"
            log.info("FAISS index loaded: %d vectors, dim %d", self.index.ntotal, self.index.d)
        except Exception as exc:
            self.status = "error"
            self.error = str(exc)
            log.exception("FAISS load failed")

    @property
    def available(self) -> bool:
        return self.status == "ready"

    def info(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "vectors": int(self.index.ntotal) if self.index is not None else 0,
            "papers_seen": self.manifest.get("papers_seen"),
            "model": settings.embedding_model,
            "metadata_sidecar": self.store.status if self.store else "n/a",
            "corpus_note": "Demo subset of arXiv — not the full collection." if (self.manifest.get("papers_seen") or 0) < 100000 else "",
        }

    def _cache_get(self, key: str) -> Optional[list[dict]]:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key: str, val: list[dict]) -> None:
        with self._cache_lock:
            self._cache[key] = val
            while len(self._cache) > 256:
                self._cache.popitem(last=False)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        date_from: str = "",
        date_to: str = "",
        exclude_ids: Optional[list[str]] = None,
        chunks_per_paper: int = 2,
    ) -> list[dict[str, Any]]:
        if not self.available or self.index is None or self.store is None:
            raise RuntimeError(f"ArXiv index unavailable: {self.error or self.status}")
        top_k = max(1, min(int(top_k), 20))
        min_score = settings.arxiv_min_score if min_score is None else float(min_score)
        key = json.dumps([query, top_k, min_score, date_from, date_to, sorted(exclude_ids or []), chunks_per_paper])
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        qv = get_encoder().encode_one(query)
        n_fetch = min(int(self.index.ntotal), max(top_k * settings.arxiv_overfetch, 40))
        scores, ids = self.index.search(qv, n_fetch)
        exclude = set(exclude_ids or [])
        papers: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for score, doc_id in zip(scores[0].tolist(), ids[0].tolist()):
            if doc_id < 0 or score < min_score:
                continue
            try:
                doc = self.store.read_document(int(doc_id))
            except (IndexError, OSError, json.JSONDecodeError) as exc:
                log.warning("missing document %s: %s", doc_id, exc)
                continue
            md = doc.get("metadata", {}) or {}
            pid = str(md.get("id", ""))
            if not pid or pid in exclude:
                continue
            created = str(md.get("created", ""))
            if date_from and created and created[:10] < date_from:
                continue
            if date_to and created and created[:10] > date_to:
                continue
            entry = papers.get(pid)
            if entry is None:
                if len(papers) >= top_k:
                    continue
                meta = self.store.paper_or_fallback(pid, doc)
                entry = {
                    "paper_id": pid,
                    "arxiv_id": pid,
                    "title": meta.get("title") or f"arXiv:{pid}",
                    "authors": meta.get("authors", []),
                    "byline": meta.get("byline", ""),
                    "published": created[:10],
                    "abstract": meta.get("abstract", ""),
                    "license": md.get("license", ""),
                    "url": f"https://arxiv.org/abs/{pid}",
                    "score": round(float(score), 4),
                    "chunks": [],
                }
                papers[pid] = entry
            if len(entry["chunks"]) < chunks_per_paper:
                entry["chunks"].append({"doc_id": int(doc_id), "score": round(float(score), 4), "text": doc.get("text", "")[:1600]})
        results = list(papers.values())
        self._cache_put(key, results)
        return results

    def get_paper(self, paper_id: str) -> Optional[dict[str, Any]]:
        """Resolve a paper by arXiv id (used for graph-discovered papers in hybrid retrieval)."""
        if self.store is None:
            return None
        meta = self.store.paper(paper_id)
        if not meta:
            return None
        text = self.store.first_chunk_text(paper_id, 1600)
        return {
            "paper_id": paper_id,
            "arxiv_id": paper_id,
            "title": meta.get("title") or f"arXiv:{paper_id}",
            "authors": meta.get("authors", []),
            "byline": meta.get("byline", ""),
            "published": (meta.get("created") or "")[:10],
            "abstract": meta.get("abstract", ""),
            "url": f"https://arxiv.org/abs/{paper_id}",
            "score": 0.0,
            "chunks": [{"doc_id": meta.get("first_doc"), "score": 0.0, "text": text}] if text else [],
        }

    def chunks_for_paper(self, paper_id: str, query: str, n: int = 2) -> list[dict[str, Any]]:
        """Best-matching chunks of a specific paper for a query (re-scores the paper's chunk range)."""
        if self.store is None or self.index is None:
            return []
        meta = self.store.paper(paper_id)
        if not meta or meta.get("first_doc") is None:
            return []
        first, last = int(meta["first_doc"]), int(meta["last_doc"])
        if last - first > 400:
            last = first + 400
        try:
            vecs = np.vstack([self.index.reconstruct(i) for i in range(first, last + 1)]).astype(np.float32)
        except Exception:
            return []
        qv = get_encoder().encode_one(query)
        sims = (vecs @ qv.T).ravel()
        order = np.argsort(-sims)[:n]
        out = []
        for o in order:
            did = first + int(o)
            try:
                out.append({"doc_id": did, "score": round(float(sims[o]), 4), "text": self.store.read_document(did).get("text", "")[:1600]})
            except (IndexError, OSError):
                pass
        return out


_index: Optional[ArxivIndex] = None


def get_arxiv_index() -> ArxivIndex:
    global _index
    if _index is None:
        _index = ArxivIndex()
    return _index
