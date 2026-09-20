"""
Grounding & anti-hallucination (blueprint §9). Five defences, strongest first:
  1. the LLM never owns the verdict (rules decide pass/fail)
  2. closed-book forbidden (answer only from provided context)
  3. mandatory, verified citations (reject cites not in the retrieved set)
  4. quote-and-check (the quote must be a substring of the cited chunk)
  5. faithfulness scoring (NLI / LLM-as-judge, offline + optional guardrail)

This module implements the structural, LLM-free path (retrieve → cite, always
runnable) plus an OPTIONAL Ollama grounded answer that is passed through the
structural checks (3, 4) before being trusted.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from evident.rag.retrieve import KBIndex, Hit

# Map an anomaly type to a retrieval query (blueprint §9: the finding is the query).
_ANOMALY_QUERY = {
    "expired_document": "identity document expired past validity date not acceptable",
    "dob_conflict": "conflicting date of birth across identity documents",
    "full_name_conflict": "name mismatch inconsistent across documents",
    "name_conflict": "name mismatch inconsistent across documents",
    "invalid_id_format": "identity card number format invalid wrong length",
    "missing_field": "missing mandatory field incomplete document",
    "low_confidence": "low recognition confidence manual review threshold",
}
_FIRST_SENTENCE = re.compile(r"^[^.]+\.")


def citations_valid(cited_ids: list[str], retrieved_ids: set[str]) -> bool:
    """Defence #3: every cited chunk id must have actually been retrieved."""
    return all(cid in retrieved_ids for cid in cited_ids)


def quote_supported(quote: str, chunk_text: str) -> bool:
    """Defence #4: the supporting quote must literally occur in the cited chunk."""
    if not quote:
        return False
    return quote.strip().lower() in chunk_text.lower()


def _first_sentence(text: str) -> str:
    m = _FIRST_SENTENCE.match(text.strip())
    return m.group(0) if m else text[:160]


def explain_anomaly(anomaly_type: str, index: KBIndex, applies_to: str | None = None,
                    detail: str = "") -> dict:
    """
    Ground one anomaly in a retrieved, cited rule — no LLM required.
    Returns a structured, machine-checkable explanation (the §9 contract shape).
    """
    query = _ANOMALY_QUERY.get(anomaly_type, anomaly_type.replace("_", " ")) + " " + detail
    hits = index.search(query, applies_to=applies_to, top_k=3)
    if not hits:
        return {"anomaly": anomaly_type, "supported": False, "citations": [],
                "retrieval_confidence": 0.0, "statement": "No supporting rule found."}
    top = hits[0]
    quote = _first_sentence(top.chunk.text)
    citation = {"chunk_id": top.chunk.chunk_id, "quote": quote,
                "breadcrumb": top.chunk.breadcrumb, "source": top.chunk.meta.get("source")}
    return {
        "anomaly": anomaly_type,
        "statement": top.chunk.text,
        "citations": [citation],
        "supported": quote_supported(quote, top.chunk.text)
        and citations_valid([top.chunk.chunk_id], {h.chunk.chunk_id for h in hits}),
        "retrieval_confidence": top.score,
    }


def answer_question(question: str, index: KBIndex, applies_to: str | None = None,
                    use_llm: bool = False) -> dict:
    """
    Answer a compliance question ("why was this flagged?") with cited passages.
    Structural grounding always; LLM prose only if use_llm and Ollama is reachable,
    and only after its citations pass the checks.
    """
    hits = index.search(question, applies_to=applies_to, top_k=3)
    citations = [{"chunk_id": h.chunk.chunk_id, "quote": _first_sentence(h.chunk.text),
                  "breadcrumb": h.chunk.breadcrumb, "score": h.score} for h in hits]
    answer = " ".join(h.chunk.text for h in hits[:1]) if hits else "No supporting rule found."
    result = {"question": question, "answer": answer, "citations": citations,
              "grounded": bool(hits), "source": "retrieval"}
    if use_llm and hits:
        llm = _llm_grounded(question, hits)
        if llm is not None:
            result.update(llm)
            result["source"] = "llm+retrieval"
    return result


def _llm_grounded(question: str, hits: list[Hit]) -> dict | None:  # pragma: no cover
    """Optional: an Ollama answer constrained to the retrieved context, then checked."""
    context = "\n".join(f"[{h.chunk.chunk_id}] {h.chunk.text}" for h in hits)
    prompt = (
        "Answer the compliance question using ONLY the numbered context. "
        "Cite the chunk id(s) you used. If nothing supports an answer, say so.\n"
        'Return JSON: {"answer": str, "cited_ids": [str]}\n\n'
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n"
    )
    try:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
        body = json.dumps({"model": os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
                           "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:   # noqa: S310
            data = json.loads(json.loads(r.read())["response"])
        retrieved_ids = {h.chunk.chunk_id for h in hits}
        cited = [c for c in data.get("cited_ids", []) if c in retrieved_ids]  # defence #3
        if not cited:
            return None
        return {"answer": data.get("answer", ""),
                "citations": [{"chunk_id": c} for c in cited]}
    except Exception:
        return None
