"""
Risk & Compliance agent (blueprint §8) — the ONE true agent in Evident.

It runs a reason → act → observe loop: from what it has seen so far it chooses the
next tool to call, and stops when it can produce a grounded, cited recommendation
(or decides to escalate). The loop, the tool belt, the stop conditions and the
recorded trajectory are the real, defensible parts — and they're what makes this an
agent rather than a fixed chain.

The *policy* (which tool next) is pluggable:
  • "rule" (default) — a deterministic policy, so the agent runs anywhere with no LLM.
  • "llm"  — an Ollama model chooses the next action from the tool menu (the real
      agentic version); falls back to the rule policy if Ollama isn't reachable.

The recommendation is ADVISORY ONLY — a human makes the binding decision (§11).
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field

from evident.config import settings
from evident.agent import tools

_TOOLS = ("retrieve_regulations", "lookup_similar_cases", "compute_confidence",
          "escalate_to_human", "finish")


@dataclass
class AgentDecision:
    recommendation: str                 # auto_clear | escalate  (advisory)
    rationale: str
    confidence: float
    citations: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)   # trajectory, for agent eval


def run_risk_agent(result, index=None, policy: str | None = None,
                   history: list | None = None, max_steps: int = 8) -> AgentDecision:
    policy = policy or os.getenv("AGENT_POLICY", "rule")
    if index is None:
        from evident.rag.retrieve import KBIndex
        index = KBIndex.from_kb()

    doc_type_by_id = {d["doc_id"]: d["doc_type"] for d in result.documents}
    state = {
        "findings": tools.get_rule_findings(result),
        "grounded": {},        # anomaly_type -> citation
        "similar": None,
        "confidence": None,
    }
    trajectory: list[dict] = []
    citations: list = []

    for _ in range(max_steps):
        action = (_llm_policy(state) if policy == "llm" else None) or _rule_policy(state)
        name = action["tool"]
        trajectory.append(action)

        if name == "retrieve_regulations":
            a = next(a for a in result.anomalies if a.type not in state["grounded"])
            applies = doc_type_by_id.get(a.doc_id) if a.doc_id else None
            obs = tools.retrieve_regulations(a.type, index, applies_to=applies, detail=a.detail)
            state["grounded"][a.type] = obs.get("citations", [])
            if obs.get("citations"):
                citations.extend(obs["citations"])
        elif name == "lookup_similar_cases":
            state["similar"] = tools.lookup_similar_cases(result, history)
        elif name == "compute_confidence":
            state["confidence"] = tools.compute_confidence(result)["confidence"]
        elif name == "escalate_to_human":
            return AgentDecision("escalate", action.get("reason", "flagged for review"),
                                 result.confidence, citations, trajectory)
        elif name == "finish":
            rec = action.get("recommendation", "auto_clear")
            return AgentDecision(rec, action.get("reason", "no blocking findings"),
                                 result.confidence, citations, trajectory)

    return AgentDecision("escalate", "step budget exhausted — routing to human",
                         result.confidence, citations, trajectory)


def _rule_policy(state: dict) -> dict:
    """Deterministic next-action policy (runs with no LLM)."""
    findings = state["findings"]
    # 1) ground every anomaly in a cited rule
    if len(state["grounded"]) < findings["n_anomalies"]:
        return {"tool": "retrieve_regulations", "why": "ground an un-cited anomaly"}
    # 2) if there are anomalies, consult prior similar cases once
    if findings["n_anomalies"] and state["similar"] is None:
        return {"tool": "lookup_similar_cases", "why": "check how similar cases went"}
    # 3) make sure we have confidence
    if state["confidence"] is None:
        return {"tool": "compute_confidence", "why": "need the confidence signal"}
    # 4) decide
    if findings["hard"]:
        return {"tool": "escalate_to_human",
                "reason": f"hard anomaly: {', '.join(sorted(set(findings['hard'])))}"}
    if state["confidence"] < settings.review_threshold:
        return {"tool": "escalate_to_human",
                "reason": f"confidence {state['confidence']:.2f} below "
                          f"threshold {settings.review_threshold}"}
    return {"tool": "finish", "recommendation": "auto_clear",
            "reason": "no anomalies and confidence above threshold"}


def _llm_policy(state: dict) -> dict | None:  # pragma: no cover - needs Ollama
    """Optional: let an Ollama model choose the next tool. Falls back to rule policy."""
    prompt = (
        "You are a KYC risk agent. Given the state, choose the NEXT tool to call.\n"
        f"Tools: {list(_TOOLS)}\n"
        "Rules of thumb: ground every anomaly (retrieve_regulations) before deciding; "
        "escalate_to_human on any hard anomaly or low confidence; otherwise finish "
        "with recommendation auto_clear.\n"
        'Return JSON: {"tool": str, "reason": str, "recommendation": str?}\n\n'
        f"STATE: {json.dumps(state, default=list)}\n"
    )
    try:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
        body = json.dumps({"model": os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
                           "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310
            action = json.loads(json.loads(r.read())["response"])
        return action if action.get("tool") in _TOOLS else None
    except Exception:
        return None
