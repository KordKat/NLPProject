"""Query encoder — must match the encoder used to build the FAISS index.

The index manifest (manifest.json) records `model_id = sentence-transformers/all-MiniLM-L6-v2`,
384-dim, L2-normalised embeddings searched with inner product (== cosine).

This reproduces the model's sentence-transformers pipeline exactly (modules.json of the model repo:
Transformer → mean Pooling → Normalize, max_seq_length 256) using `transformers` directly, so the
query vectors are identical to `SentenceTransformer(...).encode(..., normalize_embeddings=True)` as in
search_faiss.py — without importing sentence-transformers (its scikit-learn dependency ships native
DLLs that some Windows Application Control policies block).
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Optional

import numpy as np

from app.config import get_settings

log = logging.getLogger("stemos.embedding")
settings = get_settings()

MAX_SEQ_LENGTH = 256  # sentence-transformers config for all-MiniLM-L6-v2


class QueryEncoder:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.dimension: Optional[int] = None

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    dev = self.device
                    if dev == "auto":
                        dev = "cuda" if torch.cuda.is_available() else "cpu"
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    model = AutoModel.from_pretrained(self.model_name).to(dev)
                    model.eval()
                    self._dev = dev
                    self._model = model
                    self.dimension = int(model.config.hidden_size)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        model = self._ensure()
        with self._lock, torch.inference_mode():
            batch = self._tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(self._dev)
            out = model(**batch).last_hidden_state  # (B, T, H)
            mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # mean pooling
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)  # Normalize module
        return np.ascontiguousarray(pooled.float().cpu().numpy().astype(np.float32))

    @lru_cache(maxsize=2048)
    def encode_cached(self, text: str) -> bytes:
        return self.encode([text])[0].tobytes()

    def encode_one(self, text: str) -> np.ndarray:
        return np.frombuffer(self.encode_cached(text), dtype=np.float32).reshape(1, -1)


_encoder: Optional[QueryEncoder] = None


def get_encoder() -> QueryEncoder:
    global _encoder
    if _encoder is None:
        _encoder = QueryEncoder(settings.embedding_model, settings.embedding_device)
    return _encoder
