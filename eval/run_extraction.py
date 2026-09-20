"""
Extraction evaluation runner (blueprint §19) — your first real accuracy number.

Reads the synthetic manifest, runs OCR → extraction on each document, and scores
the predicted fields against the free ground truth. Prints overall + per-field +
per-difficulty metrics, and writes eval/results/extraction.json.

Examples:
  python eval/run_extraction.py                          # regex baseline, all docs
  python eval/run_extraction.py --limit 30
  python eval/run_extraction.py --extract-backend llm    # needs Ollama running
  python eval/run_extraction.py --ocr-backend easyocr

Run the baseline first to see whether the LLM actually beats it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evident.pipeline.ocr import run_ocr
from evident.pipeline.extract import extract_fields, FIELDS_BY_TYPE

# eval/ is on sys.path when this file is run as a script
from extraction_metrics import Counts, prf1, score_document


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--limit", type=int, default=0, help="cap number of documents (0 = all)")
    ap.add_argument("--ocr-backend", default=None)
    ap.add_argument("--extract-backend", default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/results/extraction.json"))
    args = ap.parse_args()

    manifest = args.data / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"No manifest at {manifest}. Run: python data/generate.py --n 50 --seed 42")

    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    overall = Counts()
    by_difficulty: dict[str, Counts] = {}
    ocr_confs: list[float] = []
    t0 = time.perf_counter()

    for i, row in enumerate(rows, 1):
        image = args.data / row["image"]
        gold = row["fields"]
        doc_type = row["doc_type"]
        ocr = run_ocr(str(image), backend=args.ocr_backend)
        ocr_confs.append(ocr.mean_confidence)
        pred = extract_fields(ocr.text, doc_type, backend=args.extract_backend)

        c = score_document(pred, gold, FIELDS_BY_TYPE[doc_type])
        overall.merge(c)
        by_difficulty.setdefault(row["difficulty"], Counts()).merge(c)
        print(f"\r  scored {i}/{len(rows)} documents…", end="", flush=True)

    dt = time.perf_counter() - t0
    print()

    summary = prf1(overall)
    per_field = {
        f: {"exact_match": (c / g if g else 0.0), "n": g}
        for f, (c, g) in sorted(overall.per_field.items())
    }
    per_diff = {d: prf1(c) for d, c in sorted(by_difficulty.items())}
    mean_ocr = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0

    # ---- report ----
    print("\n=== Extraction evaluation ===")
    print(f"docs={len(rows)}  ocr={args.ocr_backend or 'tesseract'}  "
          f"extract={args.extract_backend or 'regex'}  "
          f"{len(rows)/dt:.1f} docs/s  mean_OCR_conf={mean_ocr:.2f}")
    print(f"\nOVERALL  P={summary['precision']:.3f}  R={summary['recall']:.3f}  "
          f"F1={summary['f1']:.3f}  exact-match={summary['exact_match']:.3f}")
    print("\nby field:")
    for f, m in per_field.items():
        print(f"  {f:<16} exact-match={m['exact_match']:.3f}  (n={m['n']})")
    print("\nby difficulty:")
    for d, m in per_diff.items():
        print(f"  {d:<8} F1={m['f1']:.3f}  exact-match={m['exact_match']:.3f}  (n={m['n_gold']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"ocr": args.ocr_backend or "tesseract",
                   "extract": args.extract_backend or "regex", "n_docs": len(rows)},
        "overall": summary, "per_field": per_field, "per_difficulty": per_diff,
        "mean_ocr_confidence": mean_ocr, "docs_per_sec": len(rows) / dt,
    }, indent=2), encoding="utf-8")
    print(f"\n✓ wrote {args.out}")


if __name__ == "__main__":
    main()
