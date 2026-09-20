"""
Lightweight in-process metrics (blueprint §15) — counters + histograms, no
Prometheus client dependency. `render_prometheus()` emits the text exposition
format so a real Prometheus could scrape /metrics later; `snapshot()` powers the
dashboard. This is deliberately tiny: observability without an MLOps platform.
"""
from __future__ import annotations

from collections import defaultdict
from threading import Lock

_counters: dict[str, float] = defaultdict(float)
_hist: dict[str, list[float]] = defaultdict(list)
_lock = Lock()


def inc(name: str, value: float = 1.0) -> None:
    with _lock:
        _counters[name] += value


def observe(name: str, value: float) -> None:
    with _lock:
        _hist[name].append(value)


def record_case(result) -> None:
    """Record metrics from a finished CaseResult (called after each analysis)."""
    inc("cases_total")
    inc(f"cases_status_{result.status}")
    inc("anomalies_total", len(result.anomalies))
    for k, v in (result.timings_ms or {}).items():
        observe(k, v)
    if result.confidence is not None:
        observe("confidence", result.confidence)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    s = sorted(values)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"count": len(s), "mean": round(sum(s) / len(s), 2),
            "p50": round(p(0.5), 2), "p95": round(p(0.95), 2)}


def snapshot() -> dict:
    with _lock:
        return {"counters": dict(_counters),
                "histograms": {k: _stats(v) for k, v in _hist.items()}}


def render_prometheus() -> str:
    lines = []
    snap = snapshot()
    for k, v in snap["counters"].items():
        lines.append(f"evident_{k} {v}")
    for k, st in snap["histograms"].items():
        for stat, val in st.items():
            lines.append(f"evident_{k}_{stat} {val}")
    return "\n".join(lines) + "\n"
