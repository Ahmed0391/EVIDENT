"""
Field extraction (blueprint §6, §9, §22 module ②).

An LLM *chain*, not an agent: control flow (prompt → validate → repair) lives here,
not in the model. Two backends so you get a real number before setting up Ollama:

  • "regex"  (default) — a layout/regex BASELINE. No LLM, runs immediately. Always
      have a baseline before you reach for a model: it tells you whether the LLM is
      actually earning its keep.
  • "llm"    — schema-constrained generation via Ollama, with a validate→repair
      retry loop. Extraction confidence is derived from OCR-agreement, NOT from the
      model's self-reported number (which is unreliable).

Both return a plain ``{field_name: value}`` dict for the given doc_type.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from difflib import SequenceMatcher

# Fields we try to extract per document type (mirrors the synthetic schema).
FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "national_id": ("full_name", "id_number", "dob", "expiry"),
    "passport": ("full_name", "passport_number", "nationality", "dob", "expiry"),
    "proof_of_address": ("full_name", "address", "issue_date", "provider"),
}

_DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_CIN_RE = re.compile(r"\b\d{8}\b")
_PASSPORT_RE = re.compile(r"\b[A-Z]\d{7}\b")

# Label keywords (as printed on the synthetic docs) → the field they precede.
_LABELS = {
    "full_name": ("name", "nom"),
    "address": ("adresse", "address"),
    "provider": ("emetteur", "émetteur", "provider"),
}


def extract_fields(ocr_text: str, doc_type: str, backend: str | None = None) -> dict:
    backend = backend or os.getenv("EXTRACT_BACKEND", "regex")
    if backend == "regex":
        return _regex_extract(ocr_text, doc_type)
    if backend == "llm":
        return _llm_extract(ocr_text, doc_type)
    raise ValueError(f"unknown extract backend: {backend!r} (use 'regex' or 'llm')")


def extraction_confidence(value: str, ocr_text: str) -> float:
    """How well the extracted value is supported by the OCR text (0..1)."""
    if not value:
        return 0.0
    v = value.lower().strip()
    if v in ocr_text.lower():
        return 1.0
    return SequenceMatcher(None, v, ocr_text.lower()).ratio()


# --------------------------------------------------------------------------- #
# Baseline: layout + regex
# --------------------------------------------------------------------------- #

def _after_label(lines: list[str], keywords: tuple[str, ...]) -> str:
    """Return the value that follows a label line (same line, else the next one)."""
    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in keywords):
            # value on the same line after the label?
            after = re.split(r"[:/]", line)[-1].strip()
            if after and not any(k in after.lower() for k in keywords) and len(after) > 1:
                return after
            # otherwise the next non-empty line
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return nxt.strip()
    return ""


_NAME_RE = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]+(?: [A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]+)+$")


def _sorted_dates(ocr_text: str) -> list[str]:
    """All dd/mm/yyyy dates, sorted chronologically (robust to OCR line order)."""
    from evident.pipeline.rules import parse_date
    found = _DATE_RE.findall(ocr_text)
    parsed = [(parse_date(d), d) for d in found]
    parsed = [(p, raw) for p, raw in parsed if p is not None]
    parsed.sort(key=lambda t: t[0])
    return [raw for _, raw in parsed]


def _name_fallback(lines: list[str]) -> str:
    """When the label anchor fails (garbled OCR), pick the first name-shaped line."""
    for ln in lines:
        s = ln.strip()
        if _NAME_RE.match(s) and not any(k in s.lower() for k in
                                         ("nom", "name", "passe", "carte", "identit",
                                          "national", "adresse", "domicile")):
            return s
    return ""


def _regex_extract(ocr_text: str, doc_type: str) -> dict:
    lines = [ln for ln in ocr_text.splitlines() if ln.strip()]
    dates = _sorted_dates(ocr_text)          # earliest → latest
    out: dict[str, str] = {}

    for f in FIELDS_BY_TYPE[doc_type]:
        if f == "id_number":
            m = _CIN_RE.findall(ocr_text)
            out[f] = m[0] if m else ""
        elif f == "passport_number":
            m = _PASSPORT_RE.findall(ocr_text)
            out[f] = m[0] if m else ""
        elif f == "nationality":
            out[f] = "TUN" if "tun" in ocr_text.lower() else ""
        elif f == "dob":
            out[f] = dates[0] if dates else ""            # birth = earliest date
        elif f == "expiry":
            out[f] = dates[-1] if len(dates) > 1 else ""  # expiry = latest date
        elif f == "issue_date":
            out[f] = dates[0] if dates else ""
        elif f == "full_name":
            out[f] = _after_label(lines, _LABELS["full_name"]) or _name_fallback(lines)
        elif f in _LABELS:
            out[f] = _after_label(lines, _LABELS[f])
        else:
            out[f] = ""
    return out


# --------------------------------------------------------------------------- #
# LLM: schema-constrained generation via Ollama, with a repair retry
# --------------------------------------------------------------------------- #

def _ollama(prompt: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
    body = json.dumps({
        "model": os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
        "prompt": prompt,
        "stream": False,
        "format": "json",          # Ollama constrains output to valid JSON
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:          # noqa: S310 (local)
        return json.loads(r.read())["response"]


def _prompt(ocr_text: str, doc_type: str, error: str = "") -> str:
    fields = ", ".join(FIELDS_BY_TYPE[doc_type])
    repair = f"\nYour previous answer was invalid: {error}\nReturn ONLY valid JSON." if error else ""
    return (
        "You extract identity-document fields from noisy OCR text. "
        "Use ONLY what appears in the text; if a field is absent, use an empty string. "
        "Do not invent values.\n"
        f"Document type: {doc_type}\n"
        f"Return a JSON object with EXACTLY these keys: {fields}\n"
        f"Dates must stay in dd/mm/yyyy as printed.{repair}\n\n"
        f"OCR TEXT:\n\"\"\"\n{ocr_text}\n\"\"\"\n"
    )


def _llm_extract(ocr_text: str, doc_type: str) -> dict:  # pragma: no cover - needs Ollama
    expected = set(FIELDS_BY_TYPE[doc_type])
    error = ""
    for _ in range(2):  # one shot + one repair
        try:
            raw = _ollama(_prompt(ocr_text, doc_type, error))
            data = json.loads(raw)
            if isinstance(data, dict) and expected.issubset(data.keys()):
                return {k: str(data.get(k, "") or "") for k in FIELDS_BY_TYPE[doc_type]}
            error = f"missing keys; need {sorted(expected)}"
        except (json.JSONDecodeError, KeyError, OSError) as e:
            error = str(e)
    # graceful fallback so the pipeline never crashes on a bad model response
    return {k: "" for k in FIELDS_BY_TYPE[doc_type]}
