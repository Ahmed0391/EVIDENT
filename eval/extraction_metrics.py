"""
Extraction metrics (blueprint §19). Field-level exact-match + micro P/R/F1.

Scoring per document, over the fields we expect for its type:
  • gold present  = ground-truth value is non-empty
  • pred present  = predicted value is non-empty
  • correct       = both present AND equal after normalisation
Micro-averaged across all fields of all documents:
  precision = correct / pred_present
  recall    = correct / gold_present
  f1        = harmonic mean
Exact-match accuracy = correct / gold_present  (same as recall here, reported
separately because it's the number people intuitively ask for).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


def normalize_value(value: str) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


@dataclass
class Counts:
    correct: int = 0
    gold_present: int = 0
    pred_present: int = 0
    per_field: dict = field(default_factory=dict)   # field -> [correct, gold_present]

    def add(self, field_name: str, pred: str, gold: str) -> None:
        gp = bool(normalize_value(gold))
        pp = bool(normalize_value(pred))
        ok = gp and normalize_value(pred) == normalize_value(gold)
        self.gold_present += gp
        self.pred_present += pp
        self.correct += ok
        c, g = self.per_field.get(field_name, (0, 0))
        self.per_field[field_name] = (c + ok, g + gp)

    def merge(self, other: "Counts") -> None:
        self.correct += other.correct
        self.gold_present += other.gold_present
        self.pred_present += other.pred_present
        for k, (c, g) in other.per_field.items():
            pc, pg = self.per_field.get(k, (0, 0))
            self.per_field[k] = (pc + c, pg + g)


def prf1(counts: Counts) -> dict:
    p = counts.correct / counts.pred_present if counts.pred_present else 0.0
    r = counts.correct / counts.gold_present if counts.gold_present else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "exact_match": r, "n_gold": counts.gold_present}


def score_document(pred: dict, gold: dict, fields: tuple[str, ...]) -> Counts:
    c = Counts()
    for f in fields:
        c.add(f, pred.get(f, ""), gold.get(f, ""))
    return c
