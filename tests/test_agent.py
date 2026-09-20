"""Risk agent tests (rule policy — deterministic, no LLM/network needed)."""
import os

os.environ.setdefault("EMBED_BACKEND", "hashing")  # keep retrieval offline/deterministic

from evident.pipeline.orchestrator import CaseResult, Anomaly
from evident.agent.risk_agent import run_risk_agent


def _clean_case():
    r = CaseResult(case_id="c_clean", confidence=0.9)
    r.documents = [{"doc_id": "d1", "doc_type": "national_id"}]
    return r


def _expired_case():
    r = CaseResult(case_id="c_exp", confidence=0.74)
    r.documents = [{"doc_id": "d1", "doc_type": "national_id"}]
    r.anomalies = [Anomaly("expired_document", "high", "expired", doc_id="d1")]
    return r


def test_agent_auto_clears_clean_case():
    d = run_risk_agent(_clean_case(), policy="rule")
    assert d.recommendation == "auto_clear"
    assert d.tool_calls and d.tool_calls[-1]["tool"] == "finish"


def test_agent_escalates_hard_anomaly():
    d = run_risk_agent(_expired_case(), policy="rule")
    assert d.recommendation == "escalate"
    tools_used = [t["tool"] for t in d.tool_calls]
    assert "retrieve_regulations" in tools_used      # it grounded the anomaly first
    assert "escalate_to_human" in tools_used
    assert d.citations                                # grounding produced a citation


def test_agent_records_trajectory():
    d = run_risk_agent(_expired_case(), policy="rule")
    assert isinstance(d.tool_calls, list) and len(d.tool_calls) >= 2
