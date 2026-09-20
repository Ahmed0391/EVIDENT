"""
Pipeline orchestrator (blueprint §6, §7, §8) — Phase 2.

A DETERMINISTIC state machine, NOT an agent. It runs the fixed sequence
OCR → extract → deterministic checks → cross-doc → confidence → route, and
assembles one traceable CaseResult where every finding carries its evidence.
(RAG and the risk agent slot in at Phases 3–4; the shape already leaves room.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evident.config import settings
from evident.pipeline import rules, crossdoc
from evident.pipeline.extract import FIELDS_BY_TYPE, extract_fields, extraction_confidence
from evident.pipeline.confidence import field_confidence, aggregate_confidence
from evident.pipeline.ocr import run_ocr, OcrResult

# Anomalies that force human review regardless of confidence (blueprint §7).
HARD_ANOMALY_TYPES = {
    "expired_document", "invalid_id_format", "missing_field",
    "name_conflict", "dob_conflict",
}


@dataclass
class FieldFinding:
    doc_id: str
    doc_type: str
    field: str
    value: str
    ocr_confidence: float
    extraction_confidence: float
    confidence: float
    bbox: list[float] | None = None      # evidence #1: value → pixels


@dataclass
class Anomaly:
    type: str
    severity: str
    detail: str
    rule_id: str | None = None
    doc_id: str | None = None
    citations: list = field(default_factory=list)   # RAG evidence (blueprint §9)
    rule_statement: str | None = None


@dataclass
class CaseResult:
    case_id: str
    status: str = "processing"
    confidence: float | None = None
    findings: list[FieldFinding] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    consistency: list = field(default_factory=list)     # crossdoc.FieldConsistency
    documents: list[dict] = field(default_factory=list)  # {doc_id, doc_type, fields, ocr_text, ocr_conf}
    agent_rationale: str | None = None                  # set when agent=True
    agent_trajectory: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)      # stage → ms (observability, §15)


def route(confidence: float, has_hard_anomaly: bool) -> str:
    """The branch that makes Evident Evident (blueprint §7). Advisory only."""
    if has_hard_anomaly:
        return "needs_review"
    if confidence >= settings.review_threshold:
        return "auto_cleared"
    return "needs_review"


def _ocr_conf_for(value: str, ocr: OcrResult) -> float | tuple[float, list[float] | None]:
    """Average OCR confidence of the tokens that make up `value` (+ a bbox)."""
    if not value:
        return 0.0, None
    words = value.lower().split()
    matched = [t for t in ocr.tokens if t.text.lower().strip(",.;:") in words]
    if not matched:
        return ocr.mean_confidence, None
    conf = sum(t.confidence for t in matched) / len(matched)
    xs = [c for t in matched for c in (t.bbox[0], t.bbox[2])]
    ys = [c for t in matched for c in (t.bbox[1], t.bbox[3])]
    return conf, [min(xs), min(ys), max(xs), max(ys)]


def run_pipeline(case, ocr_backend: str | None = None,
                 extract_backend: str | None = None,
                 ground: bool = False, kb_index=None,
                 agent: bool = False, agent_policy: str | None = None,
                 confidence_model=None, history: list | None = None) -> CaseResult:
    """
    `case` is anything with `.case_id` and `.documents` (a list of objects with
    `.doc_id`, `.doc_type` and an image path via `case.image_path(doc)`), e.g. a
    LoadedCase from pipeline.cases. Returns a fully assembled CaseResult.

    ground=True attaches a retrieved, cited compliance rule to each anomaly (RAG, §9).
    agent=True lets the risk agent decide the routing + rationale (§8); it implies
    grounding. confidence_model, if given, replaces the rule-based aggregate with a
    calibrated probability (§10). All opt-in so a plain run stays fast.
    """
    import time
    t = {"ocr_ms": 0.0, "extract_ms": 0.0}
    t0_all = time.perf_counter()
    result = CaseResult(case_id=case.case_id)
    extracted_docs: list[dict] = []
    all_field_confs: list[float] = []

    # --- per-document: OCR → extract → per-field confidence + rules ---
    for doc in case.documents:
        _t = time.perf_counter()
        ocr = run_ocr(case.image_path(doc), backend=ocr_backend)
        t["ocr_ms"] += (time.perf_counter() - _t) * 1000
        _t = time.perf_counter()
        pred = extract_fields(ocr.text, doc.doc_type, backend=extract_backend)
        t["extract_ms"] += (time.perf_counter() - _t) * 1000

        for f in FIELDS_BY_TYPE[doc.doc_type]:
            value = pred.get(f, "")
            oc, bbox = _ocr_conf_for(value, ocr)
            ec = extraction_confidence(value, ocr.text)
            conf = field_confidence(oc, ec)
            result.findings.append(FieldFinding(
                doc_id=doc.doc_id, doc_type=doc.doc_type, field=f, value=value,
                ocr_confidence=round(oc, 3), extraction_confidence=round(ec, 3),
                confidence=round(conf, 3), bbox=bbox,
            ))
            if value:                       # only count present fields toward confidence
                all_field_confs.append(conf)

        extracted_docs.append({"doc_id": doc.doc_id, "doc_type": doc.doc_type,
                               "fields": pred, "ocr_text": ocr.text,
                               "ocr_conf": round(ocr.mean_confidence, 3)})
        _document_rules(doc.doc_type, pred, doc.doc_id, result)

    result.documents = extracted_docs

    # --- cross-document consistency ---
    result.consistency = crossdoc.consistency_matrix(extracted_docs)
    for fc in result.consistency:
        if fc.status == "conflict":
            result.anomalies.append(Anomaly(
                type=f"{fc.field}_conflict", severity="high",
                detail=f"{fc.field} disagrees across documents: {fc.values}",
                rule_id=f"crossdoc.{fc.field}",
            ))

    # --- ground each anomaly in a cited rule (RAG §9); agent implies grounding ---
    if ground or agent:
        _t = time.perf_counter()
        _ground_anomalies(result, kb_index)
        t["retrieval_ms"] = (time.perf_counter() - _t) * 1000

    # --- confidence (rule-based, or calibrated model if supplied §10) ---
    n_hard = sum(1 for a in result.anomalies if a.type in HARD_ANOMALY_TYPES)
    result.confidence = round(
        aggregate_confidence(all_field_confs, n_anomalies=len(result.anomalies),
                             retrieval_faithful=True), 3)
    if confidence_model is not None:
        from evident.pipeline.confidence import case_features
        result.confidence = round(
            confidence_model.predict(case_features(result, all_field_confs)), 3)

    # --- routing: the agent decides (§8), or the deterministic gate ---
    if agent:
        from evident.agent.risk_agent import run_risk_agent
        _t = time.perf_counter()
        decision = run_risk_agent(result, kb_index, policy=agent_policy, history=history)
        t["agent_ms"] = (time.perf_counter() - _t) * 1000
        result.status = "auto_cleared" if decision.recommendation == "auto_clear" else "needs_review"
        result.agent_rationale = decision.rationale
        result.agent_trajectory = decision.tool_calls
    else:
        result.status = route(result.confidence, has_hard_anomaly=n_hard > 0)

    t["total_ms"] = (time.perf_counter() - t0_all) * 1000
    result.timings_ms = {k: round(v, 1) for k, v in t.items()}
    return result


def _ground_anomalies(result: CaseResult, kb_index=None) -> None:
    """Attach a retrieved, cited rule to each anomaly. Imported lazily so RAG deps
    are only touched when grounding is requested."""
    from evident.rag.retrieve import KBIndex
    from evident.rag.ground import explain_anomaly
    index = kb_index or KBIndex.from_kb()
    doc_type_by_id = {d["doc_id"]: d["doc_type"] for d in result.documents}
    for a in result.anomalies:
        applies = doc_type_by_id.get(a.doc_id) if a.doc_id else None
        exp = explain_anomaly(a.type, index, applies_to=applies, detail=a.detail)
        a.citations = exp["citations"]
        a.rule_statement = exp["statement"] if exp["supported"] else None


def _document_rules(doc_type: str, fields: dict, doc_id: str, result: CaseResult) -> None:
    # required fields
    for r in rules.check_required_fields(doc_type, fields, FIELDS_BY_TYPE[doc_type]):
        if not r.passed:
            result.anomalies.append(Anomaly("missing_field", "medium", r.detail,
                                            rule_id=r.rule_id, doc_id=doc_id))
    # expiry
    if fields.get("expiry"):
        r = rules.check_not_expired(fields["expiry"])
        if not r.passed:
            result.anomalies.append(Anomaly("expired_document", "high", r.detail,
                                            rule_id=r.rule_id, doc_id=doc_id))
    # id format
    id_field = "id_number" if doc_type == "national_id" else (
        "passport_number" if doc_type == "passport" else None)
    if id_field and fields.get(id_field):
        r = rules.check_id_format(doc_type, fields[id_field])
        if not r.passed:
            result.anomalies.append(Anomaly("invalid_id_format", "high", r.detail,
                                            rule_id=r.rule_id, doc_id=doc_id))
