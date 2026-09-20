"""
Train the calibrated confidence model (blueprint §10).

Runs the pipeline over a dataset, builds a feature vector per case, and labels each
case by whether the automated decision was CORRECT (matches the ground-truth
decision derived from injected anomalies). Fits a logistic-regression calibrator so
that the output means: P(a human analyst confirms the automated decision).

Reports ECE for the raw aggregate vs the calibrated model — the headline that shows
the number became honest. Saves the model and a report the notebook plots.

  python scripts/train_confidence.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from evident.pipeline.cases import list_cases, load_case
from evident.pipeline.orchestrator import run_pipeline
from evident.pipeline.confidence import case_features, fit_calibrator

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("evmetrics", _ROOT / "eval" / "metrics.py")
_m = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules[_spec.name] = _m
_spec.loader.exec_module(_m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--model-out", type=Path, default=Path("models/confidence.json"))
    ap.add_argument("--report-out", type=Path, default=Path("eval/results/confidence.json"))
    ap.add_argument("--ocr-backend", default=None)
    ap.add_argument("--extract-backend", default=None)
    args = ap.parse_args()

    ids = list_cases(args.data)
    X, y, raw = [], [], []
    for i, cid in enumerate(ids, 1):
        case = load_case(cid, args.data)
        result = run_pipeline(case, ocr_backend=args.ocr_backend,
                              extract_backend=args.extract_backend)
        correct_status = "needs_review" if case.gold_anomalies else "auto_cleared"
        X.append(case_features(result))
        y.append(int(result.status == correct_status))
        raw.append(result.confidence)
        print(f"\r  processed {i}/{len(ids)} cases…", end="", flush=True)
    print()

    model = fit_calibrator(X, y)
    cal = [model.predict(x) for x in X]

    ece_raw = _m.expected_calibration_error(raw, [bool(v) for v in y])
    ece_cal = _m.expected_calibration_error(cal, [bool(v) for v in y])
    acc = sum(y) / len(y)

    model.save(args.model_out)
    report = {
        "n_cases": len(ids),
        "decision_accuracy": acc,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "reliability_raw": _m.reliability_curve(raw, [bool(v) for v in y]),
        "reliability_calibrated": _m.reliability_curve(cal, [bool(v) for v in y]),
        "feature_names": model.feature_names,
        "weights": model.weights,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== Confidence calibration ===")
    print(f"cases={len(ids)}  decision_accuracy={acc:.3f}")
    print(f"ECE  raw={ece_raw:.3f}  →  calibrated={ece_cal:.3f}"
          f"   ({'improved' if ece_cal <= ece_raw else 'worse'})")
    print(f"✓ model  → {args.model_out}")
    print(f"✓ report → {args.report_out}")


if __name__ == "__main__":
    main()
