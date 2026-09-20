"""
Case loading helpers (Phase 2).

A "case" is a customer file: several documents that should describe one person.
This module assembles a case from the synthetic dataset so the orchestrator can
run the full pipeline on it. Ground-truth fields are carried along for display /
evaluation only — the pipeline never reads them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CaseDocument:
    doc_id: str
    doc_type: str
    image: str                       # path relative to the data dir
    gold_fields: dict = field(default_factory=dict)
    difficulty: str = "clean"


@dataclass
class LoadedCase:
    case_id: str
    documents: list[CaseDocument]
    gold_anomalies: list = field(default_factory=list)
    data_dir: Path | None = None

    def image_path(self, doc: CaseDocument) -> str:
        return str((self.data_dir or Path(".")) / doc.image)


def list_cases(data_dir: str | Path = "data/synthetic") -> list[str]:
    labels = Path(data_dir) / "labels"
    return sorted(p.stem for p in labels.glob("*.json"))


def load_case(case_id: str, data_dir: str | Path = "data/synthetic") -> LoadedCase:
    data_dir = Path(data_dir)
    label = data_dir / "labels" / f"{case_id}.json"
    if not label.exists():
        raise FileNotFoundError(
            f"No case {case_id} in {data_dir}. Run: python data/generate.py --n 50 --seed 42"
        )
    obj = json.loads(label.read_text(encoding="utf-8"))
    docs = [
        CaseDocument(
            doc_id=d["doc_id"], doc_type=d["doc_type"], image=d["image_path"],
            gold_fields=d.get("fields", {}), difficulty=d.get("difficulty", "clean"),
        )
        for d in obj["documents"]
    ]
    return LoadedCase(case_id=obj["case_id"], documents=docs,
                      gold_anomalies=obj.get("anomalies", []), data_dir=data_dir)
