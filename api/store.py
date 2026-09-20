"""
Tiny in-memory stores (Phase 2–5).

Keeps the app runnable with zero infrastructure. Postgres persistence (the schema
in db/schema.sql) is a later swap; these interfaces are deliberately small.
Holds analyzed cases, async job state, and the human-review / feedback records
(the flywheel, blueprint §11).
"""
from __future__ import annotations

import uuid
from threading import Lock

_CASES: dict[str, dict] = {}
_JOBS: dict[str, dict] = {}
_REVIEWS: list[dict] = []
_FEEDBACK: list[dict] = []
_LOCK = Lock()


# --- analyzed case reports ---
def put(case_id: str, payload: dict) -> None:
    with _LOCK:
        _CASES[case_id] = payload


def get(case_id: str) -> dict | None:
    return _CASES.get(case_id)


def exists(case_id: str) -> bool:
    return case_id in _CASES


def review_queue() -> list[str]:
    return [cid for cid, r in _CASES.items() if r.get("status") == "needs_review"]


# --- async jobs ---
def new_job(case_id: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = {"job_id": job_id, "case_id": case_id, "status": "queued"}
    return job_id


def set_job(job_id: str, **fields) -> None:
    with _LOCK:
        _JOBS.get(job_id, {}).update(fields)


def get_job(job_id: str) -> dict | None:
    return _JOBS.get(job_id)


# --- human review + feedback (the flywheel) ---
def add_review(record: dict) -> None:
    with _LOCK:
        _REVIEWS.append(record)


def add_feedback(record: dict) -> None:
    with _LOCK:
        _FEEDBACK.append(record)


def feedback() -> list[dict]:
    return list(_FEEDBACK)
