"""
The risk agent's tool belt (blueprint §8).

Tools may be deterministic — an agent's tools don't have to be AI. Agency lives in
*which* tools the agent calls and *when it stops*, not in the tools themselves.
Each tool takes plain inputs and returns a plain dict, so the agent loop (and its
trajectory log) stays simple and inspectable.
"""
from __future__ import annotations


def get_rule_findings(result) -> dict:
    """Deterministic rule + cross-doc results already computed for the case."""
    return {
        "n_anomalies": len(result.anomalies),
        "anomaly_types": [a.type for a in result.anomalies],
        "hard": [a.type for a in result.anomalies
                 if a.severity == "high" or a.type.endswith("_conflict")
                 or a.type in ("expired_document", "invalid_id_format", "missing_field")],
    }


def retrieve_regulations(anomaly_type: str, index, applies_to: str | None = None,
                         detail: str = "") -> dict:
    """Ground an anomaly in a retrieved, cited rule (RAG)."""
    from evident.rag.ground import explain_anomaly
    return explain_anomaly(anomaly_type, index, applies_to=applies_to, detail=detail)


def lookup_similar_cases(result, history: list | None = None) -> dict:
    """
    How were similar past cases decided? Similarity = overlap of anomaly types.
    `history` is a list of {"anomaly_types": [...], "decision": "..."} (empty by default).
    """
    history = history or []
    types = set(a.type for a in result.anomalies)
    similar = [h for h in history if types & set(h.get("anomaly_types", []))]
    escalated = sum(1 for h in similar if h.get("decision") == "needs_review")
    return {"n_similar": len(similar), "n_escalated": escalated}


def compute_confidence(result) -> dict:
    """Return the case's (decomposed) confidence — already computed by the pipeline."""
    return {"confidence": result.confidence}


def escalate_to_human(result, reason: str) -> dict:
    """Route to the human review queue with a reason (advisory only)."""
    return {"action": "escalate", "reason": reason}
