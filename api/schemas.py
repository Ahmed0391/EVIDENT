"""Pydantic request/response models. The Finding is the reusable unit (blueprint §13)."""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class DocType(str, Enum):
    national_id = "national_id"
    passport = "passport"
    proof_of_address = "proof_of_address"


class Evidence(BaseModel):
    """One item of the evidence ledger (blueprint §10). Kept typed and separate."""
    kind: str = Field(..., description="document|ocr|model|rule|retrieval|llm")
    source: str = Field(..., description="e.g. doc_id, rule_id, chunk_id")
    detail: str = ""
    bbox: list[float] | None = None
    score: float | None = None


class Citation(BaseModel):
    chunk_id: str
    quote: str


class Finding(BaseModel):
    field: str
    value: str | None = None
    status: str = "info"                       # info|match|conflict|anomaly
    evidence: list[Evidence] = []
    citations: list[Citation] = []
    ocr_confidence: float | None = None
    extraction_confidence: float | None = None
    confidence: float | None = None            # decomposed/calibrated aggregate


class CaseStatus(str, Enum):
    created = "created"
    processing = "processing"
    auto_cleared = "auto_cleared"
    needs_review = "needs_review"
    reviewed = "reviewed"


class CaseSummary(BaseModel):
    case_id: str
    status: CaseStatus
    confidence: float | None = None
    n_findings: int = 0
    n_anomalies: int = 0


class ReviewAction(BaseModel):
    action: str                                # approve|reject|request_correction
    edited_fields: dict[str, str] = {}
    reason: str | None = None
