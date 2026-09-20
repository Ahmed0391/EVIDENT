"""
Evaluation-regression gate (blueprint §15) — the twist that makes CI test *model
quality*, not just code. Reads the eval result JSONs and fails (exit 1) if any
metric is below its floor. Wire this into GitHub Actions after the eval runs.

  python eval/check_thresholds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Floors — raise these as the system improves; a drop below fails the build.
THRESHOLDS = {
    "eval/results/extraction.json": [("overall.f1", 0.75)],
    "eval/results/rag.json": [("recall_at_k", 0.60), ("mrr", 0.60)],
    "eval/results/confidence.json": [("ece_calibrated", 0.20, "<=")],
}


def _get(obj: dict, path: str):
    for key in path.split("."):
        obj = obj[key]
    return obj


def main() -> int:
    failures = []
    for file, checks in THRESHOLDS.items():
        p = Path(file)
        if not p.exists():
            print(f"· skip {file} (not found — run the eval first)")
            continue
        data = json.loads(p.read_text())
        for check in checks:
            metric, floor = check[0], check[1]
            op = check[2] if len(check) > 2 else ">="
            val = _get(data, metric)
            ok = val <= floor if op == "<=" else val >= floor
            flag = "OK " if ok else "FAIL"
            print(f"  [{flag}] {file}:{metric} = {val:.3f}  ({op} {floor})")
            if not ok:
                failures.append(f"{file}:{metric}={val:.3f} {op} {floor}")
    if failures:
        print("\nEVAL GATE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\n✓ eval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
