"""
Chunking / KB loading (blueprint §9).

The knowledge base is authored as already-chunked rule entries (one JSON object
per line) so each chunk has a stable id that doubles as its citation id. In a real
system you'd chunk raw regulation PDFs by article/clause here; the Chunk shape is
the same either way, so that upgrade is localized.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KB = Path("data/knowledge_base/rules.jsonl")


@dataclass
class Chunk:
    chunk_id: str          # 'doc-validity-001' → the citation id
    text: str
    breadcrumb: str = ""
    parent_id: str | None = None
    meta: dict = field(default_factory=dict)   # applies_to, topic, source, version, ...


def load_kb(path: str | Path = DEFAULT_KB) -> list[Chunk]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"knowledge base not found: {path}")
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        meta = {k: o[k] for k in ("applies_to", "topic", "source", "version",
                                  "effective_date", "language") if k in o}
        chunks.append(Chunk(chunk_id=o["id"], text=o["text"],
                            breadcrumb=o.get("breadcrumb", ""),
                            parent_id=o.get("parent_id"), meta=meta))
    return chunks
