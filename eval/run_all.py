"""
Run the whole evaluation suite (blueprint §19) and print a one-screen summary.

  python eval/run_all.py

Runs extraction (needs OCR + a dataset), RAG retrieval, and reads the confidence
calibration report (train it first with scripts/train_confidence.py). Each writes
its JSON under eval/results/, which eval/check_thresholds.py then gates on.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=False)


def main() -> None:
    py = sys.executable
    _run([py, "eval/run_extraction.py", "--data", "data/synthetic"])
    _run([py, "eval/run_rag.py"])

    print("\n================ SUMMARY ================")
    for name, file in [("extraction", "eval/results/extraction.json"),
                       ("rag", "eval/results/rag.json"),
                       ("confidence", "eval/results/confidence.json")]:
        p = ROOT / file
        if not p.exists():
            print(f"{name:<11}: (not found)")
            continue
        d = json.loads(p.read_text())
        if name == "extraction":
            print(f"{name:<11}: F1={d['overall']['f1']:.3f}  exact={d['overall']['exact_match']:.3f}")
        elif name == "rag":
            print(f"{name:<11}: Recall@{d['k']}={d['recall_at_k']:.3f}  MRR={d['mrr']:.3f}  ({d['embed_backend']})")
        elif name == "confidence":
            print(f"{name:<11}: ECE raw={d['ece_raw']:.3f} → calibrated={d['ece_calibrated']:.3f}")
    print("========================================")
    print("gate: python eval/check_thresholds.py")


if __name__ == "__main__":
    main()
