"""
Cross-document consistency engine (blueprint §12).

Normalise first, then compare. Every value comparison is deterministic; an LLM is
used elsewhere only for language equivalence and narration, never to decide whether
two values match. Name similarity uses difflib (stdlib) so there is no hidden
dependency and the threshold is explicit and testable.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

NAME_MATCH_THRESHOLD = 0.88  # tune against the synthetic set; keep it explicit


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


@dataclass(frozen=True)
class FieldConsistency:
    field: str
    status: str          # "match" | "conflict" | "single_source" | "missing"
    values: dict         # doc_id -> value
    detail: str = ""


def _collect(documents: list[dict], field: str) -> dict:
    """Map doc_id -> printed value for a field, skipping docs that lack it."""
    out = {}
    for d in documents:
        v = str(d.get("fields", {}).get(field, "")).strip()
        if v:
            out[d["doc_id"]] = v
    return out


def check_names(documents: list[dict]) -> FieldConsistency:
    values = _collect(documents, "full_name")
    if len(values) < 2:
        return FieldConsistency("full_name", "single_source", values)
    names = list(values.values())
    worst = min(name_similarity(names[0], n) for n in names[1:])
    status = "match" if worst >= NAME_MATCH_THRESHOLD else "conflict"
    return FieldConsistency("full_name", status, values, f"min_similarity={worst:.2f}")


def check_dob(documents: list[dict]) -> FieldConsistency:
    values = _collect(documents, "dob")
    if len(values) < 2:
        return FieldConsistency("dob", "single_source", values)
    status = "match" if len(set(values.values())) == 1 else "conflict"
    return FieldConsistency("dob", status, values)


def consistency_matrix(documents: list[dict]) -> list[FieldConsistency]:
    """Return the per-field verdicts an analyst sees (blueprint §12 matrix)."""
    return [check_names(documents), check_dob(documents)]
