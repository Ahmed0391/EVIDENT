"""
FastAPI entrypoint (blueprint §13) — Phases 2–5.

Ties the whole system together over in-memory stores (no Postgres needed to run):
auth (JWT + RBAC), async analysis (BackgroundTasks + a job store), the risk agent
+ calibrated confidence, RAG Q&A, the human-in-the-loop review/feedback flywheel,
and observability at /metrics.
Run:  uvicorn evident.api.main:app --reload   →  http://localhost:8000/
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import RedirectResponse, PlainTextResponse

from evident import __version__
from evident.api import store, auth
from evident.api.deps import current_user, require_role
from evident.api.schemas import CaseSummary, CaseStatus, ReviewAction
from evident.pipeline import cases as case_loader
from evident.pipeline.orchestrator import run_pipeline
from evident.pipeline.report import build_report
from evident.obs import metrics

app = FastAPI(
    title="Evident",
    version=__version__,
    description="Explainable KYC document intelligence — every decision traceable to evidence.",
)

# Load the calibrated confidence model if it's been trained (blueprint §10).
_CONF_MODEL = None
_model_path = Path("models/confidence.json")
if _model_path.exists():
    from evident.pipeline.confidence import ConfidenceModel
    _CONF_MODEL = ConfidenceModel.load(_model_path)

_KB_INDEX = None


def _kb_index():
    global _KB_INDEX
    if _KB_INDEX is None:
        from evident.rag.retrieve import KBIndex
        _KB_INDEX = KBIndex.from_kb()
    return _KB_INDEX


# --------------------------------------------------------------------------- #
# Basics + auth
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__, "calibrated": _CONF_MODEL is not None}


@app.post("/auth/login")
def login(username: str, password: str) -> dict:
    user = auth.authenticate(username, password)
    if user is None:
        raise HTTPException(401, "invalid credentials")
    return {"access_token": auth.create_token(user["sub"], user["role"]),
            "token_type": "bearer", "role": user["role"]}


@app.get("/metrics")
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render_prometheus())


@app.get("/metrics/summary")
def metrics_summary() -> dict:
    return metrics.snapshot()


# --------------------------------------------------------------------------- #
# Cases (async analysis)
# --------------------------------------------------------------------------- #

@app.get("/cases")
def list_synthetic_cases() -> dict:
    try:
        return {"cases": case_loader.list_cases(), "review_queue": store.review_queue()}
    except Exception:
        return {"cases": [], "review_queue": []}


def _run_analysis(case_id: str, job_id: str, ground: bool, agent: bool) -> None:
    try:
        case = case_loader.load_case(case_id)
        result = run_pipeline(case, ground=ground, agent=agent,
                              confidence_model=_CONF_MODEL, kb_index=_kb_index() if (ground or agent) else None)
        metrics.record_case(result)
        store.put(case_id, build_report(result))
        store.set_job(job_id, status="done")
    except Exception as e:                              # pragma: no cover
        store.set_job(job_id, status="failed", error=str(e))


@app.post("/cases/{case_id}/analyze", status_code=202)
def analyze(case_id: str, background: BackgroundTasks,
            ground: bool = True, agent: bool = True) -> dict:
    """Kick off analysis asynchronously; poll the returned job, then GET the case."""
    if not (Path("data/synthetic/labels") / f"{case_id}.json").exists() and \
       not (Path("data/demo/labels") / f"{case_id}.json").exists():
        raise HTTPException(404, f"unknown case {case_id}")
    job_id = store.new_job(case_id)
    store.set_job(job_id, status="running")
    background.add_task(_run_analysis, case_id, job_id, ground, agent)
    return {"job_id": job_id, "status": "running",
            "poll": f"/jobs/{job_id}", "then": f"/cases/{case_id}"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/cases/{case_id}", response_model=CaseSummary)
def get_case(case_id: str) -> CaseSummary:
    if not store.exists(case_id):
        raise HTTPException(404, f"case {case_id} not analyzed yet — POST /cases/{case_id}/analyze")
    return _summary(case_id)


@app.get("/cases/{case_id}/findings")
def get_findings(case_id: str) -> dict:
    rep = _require(case_id)
    return {"fields": rep["fields"], "consistency": rep["consistency"],
            "anomalies": rep["anomalies"]}


@app.get("/cases/{case_id}/report")
def get_report(case_id: str) -> dict:
    return _require(case_id)


@app.post("/cases/{case_id}/qa")
def qa(case_id: str, question: str, use_llm: bool = False) -> dict:
    from evident.rag.ground import answer_question
    return answer_question(question, _kb_index(), use_llm=use_llm)


# --------------------------------------------------------------------------- #
# Human-in-the-loop (blueprint §11) — analyst role required
# --------------------------------------------------------------------------- #

@app.post("/cases/{case_id}/review", response_model=CaseSummary)
def review(case_id: str, action: ReviewAction,
           user: dict = Depends(require_role("analyst", "admin"))) -> CaseSummary:
    rep = _require(case_id)
    rep["review"] = {"action": action.action, "edited_fields": action.edited_fields,
                     "reason": action.reason, "analyst": user["sub"]}
    rep["status"] = "reviewed"
    store.put(case_id, rep)
    store.add_review({"case_id": case_id, "analyst": user["sub"], "action": action.action,
                      "edited_fields": action.edited_fields, "reason": action.reason})
    metrics.inc("human_reviews_total")
    if action.action != "approve":
        metrics.inc("human_overrides_total")
    # corrections feed the flywheel (extraction eval / calibration / RAG eval)
    for field, value in (action.edited_fields or {}).items():
        store.add_feedback({"case_id": case_id, "field": field, "corrected_value": value,
                            "label_kind": "extraction"})
    return _summary(case_id)


@app.get("/feedback")
def list_feedback(user: dict = Depends(require_role("admin"))) -> dict:
    return {"feedback": store.feedback()}


# --------------------------------------------------------------------------- #

def _require(case_id: str) -> dict:
    rep = store.get(case_id)
    if rep is None:
        raise HTTPException(404, f"case {case_id} not analyzed yet")
    return rep


def _summary(case_id: str) -> CaseSummary:
    rep = store.get(case_id)
    status = rep["status"] if rep["status"] in CaseStatus._value2member_map_ else "processing"
    return CaseSummary(case_id=case_id, status=CaseStatus(status),
                       confidence=rep.get("confidence"),
                       n_findings=len(rep.get("fields", [])),
                       n_anomalies=rep.get("n_anomalies", 0))
