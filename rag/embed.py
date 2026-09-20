"""
Embeddings (blueprint §9, §22 module ③).

Pluggable backends so RAG runs on any machine:
  • "sentence-transformers" (default when installed) — real semantic embeddings,
      multilingual, pip-only (no Ollama). `pip install -e ".[rag]"`.
  • "ollama" — bge-m3 via the Ollama REST API (no torch, needs the daemon).
  • "hashing" — a zero-dependency deterministic bag-of-words hashing embedding.
      Lower quality, but means the notebook and tests ALWAYS run. Used as the
      automatic fallback when sentence-transformers isn't installed.

For learning, `cosine_similarity` is implemented from scratch (retrieve.py uses it).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from typing import Sequence

_ST_MODEL = os.getenv("EMBED_ST_MODEL",
                      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_HASH_DIM = 512
_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def resolve_backend(backend: str | None) -> str:
    if backend:
        return backend
    env = os.getenv("EMBED_BACKEND")
    if env:
        return env
    try:                       # prefer real embeddings when available
        import sentence_transformers  # noqa: F401
        return "sentence-transformers"
    except ImportError:
        return "hashing"


_warned = False


def embed(texts: list[str], backend: str | None = None) -> list[list[float]]:
    b = resolve_backend(backend)
    if b == "sentence-transformers":
        try:
            return _embed_st(texts)
        except Exception as e:      # offline / model download blocked → stay usable
            global _warned
            if not _warned:
                import sys
                print(f"[evident] sentence-transformers unavailable ({e.__class__.__name__}); "
                      f"falling back to the hashing embedding.", file=sys.stderr)
                _warned = True
            return [_embed_hash(t) for t in texts]
    if b == "ollama":
        return [_embed_ollama(t) for t in texts]
    if b == "hashing":
        return [_embed_hash(t) for t in texts]
    raise ValueError(f"unknown embed backend: {b!r}")


# --- sentence-transformers ---
_ST = None


def _embed_st(texts: list[str]) -> list[list[float]]:  # pragma: no cover - heavy dep
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(_ST_MODEL)
    return _ST.encode(texts, normalize_embeddings=True).tolist()


# --- ollama ---
def _embed_ollama(text: str) -> list[float]:  # pragma: no cover - needs daemon
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/embeddings"
    body = json.dumps({"model": os.getenv("EMBED_MODEL", "bge-m3"), "prompt": text}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310 (local)
        v = json.loads(r.read())["embedding"]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# --- hashing (zero-dep fallback / teaching baseline) ---
def _embed_hash(text: str, dim: int = _HASH_DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]
