"""
Retrieval (blueprint §9). Hybrid dense + lexical, fused with RRF.

For the demo this uses an in-memory index (no Postgres needed). The production
path is pgvector for dense + tsvector for lexical, fused with the SAME RRF below —
so moving to Postgres is a swap of the two search functions, not a rewrite.
Write the fusion yourself: it's ~15 lines and you must understand the ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from evident.rag.chunk import Chunk, load_kb
from evident.rag.embed import embed, cosine_similarity, resolve_backend

_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Combine several ranked id-lists into one. score(d) = Σ 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


class KBIndex:
    """Tiny in-memory hybrid index over the compliance knowledge base."""

    def __init__(self, chunks: list[Chunk], backend: str | None = None):
        self.chunks = chunks
        self.backend = resolve_backend(backend)
        self.by_id = {c.chunk_id: c for c in chunks}
        self._emb = {c.chunk_id: e for c, e in
                     zip(chunks, embed([c.text for c in chunks], backend=self.backend))}
        self._tok = {c.chunk_id: _tokens(c.text) for c in chunks}

    @classmethod
    def from_kb(cls, path=None, backend: str | None = None) -> "KBIndex":
        return cls(load_kb(path) if path else load_kb(), backend=backend)

    def _candidates(self, applies_to: str | None) -> list[Chunk]:
        if not applies_to:
            return self.chunks
        keep = [c for c in self.chunks
                if not c.meta.get("applies_to") or applies_to in c.meta["applies_to"]]
        return keep or self.chunks          # never filter everything away

    def search(self, query: str, applies_to: str | None = None, top_k: int = 5) -> list[Hit]:
        cands = self._candidates(applies_to)
        qvec = embed([query], backend=self.backend)[0]
        qtok = _tokens(query)

        dense = sorted(cands, key=lambda c: cosine_similarity(qvec, self._emb[c.chunk_id]),
                       reverse=True)
        lexical = sorted(cands, key=lambda c: len(qtok & self._tok[c.chunk_id]), reverse=True)

        dense_ids = [c.chunk_id for c in dense]
        lexical_ids = [c.chunk_id for c in lexical]
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids])

        hits = []
        for cid in fused[:top_k]:
            hits.append(Hit(
                chunk=self.by_id[cid],
                score=round(cosine_similarity(qvec, self._emb[cid]), 3),
                dense_rank=dense_ids.index(cid) + 1,
                lexical_rank=lexical_ids.index(cid) + 1,
            ))
        return hits
