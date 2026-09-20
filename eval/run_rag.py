"""
RAG retrieval evaluation (blueprint §9, §19).

Scores retrieval against a small gold set (question → relevant chunk ids) with the
from-scratch metrics in metrics.py: Recall@k and MRR. Faithfulness / citation
correctness of generated answers is a separate, LLM-dependent eval (added later).

  python eval/run_rag.py                      # auto embedding backend
  python eval/run_rag.py --backend hashing    # force the zero-dep baseline
  python eval/run_rag.py --k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evident.rag.retrieve import KBIndex
from evident.rag.embed import resolve_backend

from metrics import recall_at_k, mrr   # eval/ is on sys.path when run as a script


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("eval/rag_gold.jsonl"))
    ap.add_argument("--backend", default=None, help="embedding backend (default: auto)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("eval/results/rag.json"))
    args = ap.parse_args()

    gold = [json.loads(l) for l in args.gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    backend = resolve_backend(args.backend)
    index = KBIndex.from_kb(backend=backend)

    rankings = []
    recalls = []
    for q in gold:
        hits = index.search(q["question"], top_k=args.k)
        retrieved = [h.chunk.chunk_id for h in hits]
        rankings.append((retrieved, q["relevant"]))
        recalls.append(recall_at_k(retrieved, q["relevant"], args.k))

    recall = sum(recalls) / len(recalls)
    mrr_score = mrr(rankings)
    print(f"\n=== RAG retrieval eval ===")
    print(f"queries={len(gold)}  embed_backend={backend}  k={args.k}")
    print(f"Recall@{args.k} = {recall:.3f}")
    print(f"MRR         = {mrr_score:.3f}")

    import json as _json
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_json.dumps(
        {"queries": len(gold), "embed_backend": backend, "k": args.k,
         "recall_at_k": recall, "mrr": mrr_score}, indent=2), encoding="utf-8")
    if backend == "hashing":
        print("\n(note: 'hashing' is the zero-dependency fallback — install "
              "sentence-transformers via `.[rag]` for real semantic retrieval.)")


if __name__ == "__main__":
    main()
