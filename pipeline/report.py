"""
Evidence-linked report (blueprint §6 F6, §10).

The original emitted `Status = Rejected`. Evident emits a report where every
statement points back to the evidence that produced it: the extracted value + its
OCR/extraction confidence, and each anomaly + the rule id (and, from Phase 3, the
cited regulation). This module builds a structured dict and a readable markdown
rendering. PDF export can wrap the markdown later.
"""
from __future__ import annotations

from evident.pipeline.orchestrator import CaseResult

_STATUS_LABEL = {
    "auto_cleared": "Conforme (auto-cleared)",
    "needs_review": "À vérifier (human review)",
    "processing": "Processing",
}


def build_report(result: CaseResult) -> dict:
    """Structured, machine-readable report."""
    return {
        "case_id": result.case_id,
        "status": result.status,
        "status_label": _STATUS_LABEL.get(result.status, result.status),
        "confidence": result.confidence,
        "n_documents": len(result.documents),
        "n_anomalies": len(result.anomalies),
        "anomalies": [
            {"type": a.type, "severity": a.severity, "detail": a.detail,
             "rule_id": a.rule_id, "doc_id": a.doc_id,
             "citations": a.citations, "rule_statement": a.rule_statement}
            for a in result.anomalies
        ],
        "fields": [
            {"doc_id": f.doc_id, "field": f.field, "value": f.value,
             "ocr_confidence": f.ocr_confidence,
             "extraction_confidence": f.extraction_confidence,
             "confidence": f.confidence, "bbox": f.bbox}
            for f in result.findings
        ],
        "consistency": [
            {"field": c.field, "status": c.status, "values": c.values, "detail": c.detail}
            for c in result.consistency
        ],
    }


def render_markdown(result: CaseResult) -> str:
    """Human-readable report — every claim traceable to its evidence."""
    r = build_report(result)
    lines = [
        f"# KYC analysis — case {r['case_id']}",
        "",
        f"**Recommendation:** {r['status_label']}  ·  "
        f"**Confidence:** {r['confidence']:.2f}  ·  "
        f"_advisory only — a human makes the binding decision_",
        "",
        f"## Anomalies ({r['n_anomalies']})",
    ]
    if r["anomalies"]:
        for a in r["anomalies"]:
            where = f" · doc `{a['doc_id']}`" if a["doc_id"] else ""
            rule = f" · rule `{a['rule_id']}`" if a["rule_id"] else ""
            lines.append(f"- **[{a['severity']}] {a['type']}** — {a['detail']}{where}{rule}")
            for c in a.get("citations", []):
                src = c.get("source") or c.get("breadcrumb") or ""
                lines.append(f"    - 📎 _cited:_ `{c['chunk_id']}` {src} — \"{c['quote']}\"")
    else:
        lines.append("- none detected")

    lines += ["", "## Cross-document consistency"]
    for c in r["consistency"]:
        lines.append(f"- **{c['field']}**: {c['status']} — {c['values']}")

    lines += ["", "## Extracted fields (with evidence)", "",
              "| doc | field | value | OCR | extract | conf |",
              "|-----|-------|-------|-----|---------|------|"]
    for f in r["fields"]:
        lines.append(
            f"| {f['doc_id']} | {f['field']} | {f['value'] or '—'} | "
            f"{f['ocr_confidence']:.2f} | {f['extraction_confidence']:.2f} | "
            f"{f['confidence']:.2f} |"
        )
    return "\n".join(lines)
