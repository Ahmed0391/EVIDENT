"""
Confidence composition (blueprint §10).

MVP: a transparent rule-based aggregate — never a single mystery number. The case
aggregate is the weakest link over critical fields, penalised per anomaly, and
gated to ~0 if retrieval faithfulness fails.
Strong: replace `aggregate_confidence` with a calibrated classifier trained on
human-confirm/override labels, and report ECE (see eval/metrics.py).
"""
from __future__ import annotations


def field_confidence(ocr_conf: float, extraction_conf: float) -> float:
    """Per-field confidence: both signals must be high (weakest link)."""
    return min(max(ocr_conf, 0.0), max(extraction_conf, 0.0))


def aggregate_confidence(
    field_confs: list[float],
    n_anomalies: int,
    retrieval_faithful: bool = True,
    anomaly_penalty: float = 0.15,
) -> float:
    """
    Transparent MVP aggregate = P(a human would confirm) proxy.
    Weakest critical field, minus a penalty per anomaly, gated by faithfulness.
    """
    if not retrieval_faithful:
        return 0.0
    if not field_confs:
        return 0.0
    base = min(field_confs)
    score = base - anomaly_penalty * n_anomalies
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Strong version: a CALIBRATED confidence model (blueprint §10).
# "Confidence" is defined as P(a human analyst confirms the automated decision),
# trained on features and evaluated with ECE + a reliability diagram — not a vibe.
# Implemented from scratch (logistic regression + standardization) to learn it.
# --------------------------------------------------------------------------- #

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

FEATURE_NAMES = ("min_field_conf", "mean_field_conf", "n_anomalies",
                 "n_hard", "n_docs", "mean_ocr_conf")


def case_features(result, field_confs: list[float] | None = None) -> list[float]:
    """Extract the fixed feature vector for the calibrator from a CaseResult."""
    if field_confs is None:
        field_confs = [f.confidence for f in result.findings if f.value]
    hard = {"expired_document", "invalid_id_format", "missing_field",
            "name_conflict", "full_name_conflict", "dob_conflict"}
    ocr = [d.get("ocr_conf", 0.0) for d in result.documents] or [0.0]
    return [
        min(field_confs) if field_confs else 0.0,
        sum(field_confs) / len(field_confs) if field_confs else 0.0,
        float(len(result.anomalies)),
        float(sum(1 for a in result.anomalies if a.type in hard)),
        float(len(result.documents)),
        sum(ocr) / len(ocr),
    ]


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class ConfidenceModel:
    weights: list[float]
    bias: float
    mu: list[float]
    sigma: list[float]
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def predict(self, features: list[float]) -> float:
        z = self.bias
        for w, x, m, s in zip(self.weights, features, self.mu, self.sigma):
            z += w * ((x - m) / (s or 1.0))
        return _sigmoid(z)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_calibrator(X: list[list[float]], y: list[int],
                   epochs: int = 800, lr: float = 0.3) -> ConfidenceModel:
    """Standardize features, then fit logistic regression by gradient descent."""
    n, d = len(X), len(X[0])
    mu = [sum(row[j] for row in X) / n for j in range(d)]
    sigma = [(sum((row[j] - mu[j]) ** 2 for row in X) / n) ** 0.5 or 1.0 for j in range(d)]
    Xs = [[(row[j] - mu[j]) / sigma[j] for j in range(d)] for row in X]

    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(Xs, y):
            p = _sigmoid(b + sum(w[j] * xi[j] for j in range(d)))
            err = p - yi
            for j in range(d):
                gw[j] += err * xi[j]
            gb += err
        w = [w[j] - lr * gw[j] / n for j in range(d)]
        b -= lr * gb / n
    return ConfidenceModel(weights=w, bias=b, mu=mu, sigma=sigma)
