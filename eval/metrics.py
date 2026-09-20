"""
Evaluation metrics — implemented from scratch on purpose (blueprint §22).

These are trivial to code and essential to understand, so we don't hide them
behind a library. Retrieval: Recall@k, MRR. Uncertainty: Expected Calibration
Error (ECE). Add NDCG here later if/when you have graded relevance.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved."""
    rel = set(relevant)
    if not rel:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & rel) / len(rel)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item (0 if none retrieved)."""
    rel = set(relevant)
    for i, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def mrr(rankings: Sequence[tuple[Sequence[str], Iterable[str]]]) -> float:
    """Mean reciprocal rank over many (retrieved, relevant) pairs."""
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in rankings) / len(rankings)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """
    ECE: weighted average gap between confidence and accuracy across bins.
    0 == perfectly calibrated. Pair this with a reliability diagram when reporting.
    """
    if not confidences:
        return 0.0
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    n = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences)
               if (c > lo or (b == 0 and c == 0.0)) and c <= hi]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        ece += (len(idx) / n) * abs(avg_conf - acc)
    return ece


def reliability_curve(confidences, correct, n_bins=10):
    """Return per-bin (mean_confidence, accuracy, count) for a reliability diagram."""
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences)
               if (c > lo or (b == 0 and c == 0.0)) and c <= hi]
        if not idx:
            continue
        mc = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        out.append((mc, acc, len(idx)))
    return out
