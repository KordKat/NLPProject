#!/usr/bin/env python3
"""Query the FAISS index created by build_faiss.py."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def read_document(docs_path: Path, offsets_path: Path, doc_id: int) -> dict:
    with offsets_path.open("rb") as offsets_file:
        offsets_file.seek(doc_id * 8)
        raw = offsets_file.read(8)
    if len(raw) != 8:
        raise IndexError(f"No document offset found for ID {doc_id}")
    offset = struct.unpack("<Q", raw)[0]
    with docs_path.open("rb") as docs_file:
        docs_file.seek(offset)
        line = docs_file.readline()
    return json.loads(line.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Text query")
    parser.add_argument("--output-dir", default="faiss_output")
    parser.add_argument("--model", default="KordAI/III-BERT-EN-STEM-AllNLI")
    parser.add_argument("--device", default="auto")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    index = faiss.read_index(str(out / "faiss.index"))
    model = SentenceTransformer(args.model, device=None if args.device == "auto" else args.device)
    query_vector = model.encode(
        [args.query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    scores, ids = index.search(query_vector, args.k)

    docs_path = out / "documents.jsonl"
    offsets_path = out / "document_offsets.bin"
    for rank, (score, doc_id) in enumerate(zip(scores[0], ids[0]), start=1):
        if doc_id < 0:
            continue
        doc = read_document(docs_path, offsets_path, int(doc_id))
        metadata = doc.get("metadata", {})
        print(f"\n[{rank}] id={doc_id} score={score:.4f}")
        print(f"source={metadata.get('dataset')} split={metadata.get('split')} source_id={metadata.get('id')}")
        print(f"metadata={json.dumps(metadata, ensure_ascii=False, default=str)}")
        print(doc.get("text", ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
