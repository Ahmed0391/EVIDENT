# Evaluation

The section that turns a demo into a result (blueprint §19). Metrics live in
`metrics.py`, implemented from scratch on purpose.

Suites to build out, one per layer:
- `ocr/`         — CER/WER, field-level accuracy, accuracy-by-difficulty
- `extraction/`  — per-field precision/recall/F1, exact-match (normalised)
- `anomaly/`     — precision/recall/F1 + confusion, weighted false-negative rate
- `rag/`         — Recall@k, MRR (+ faithfulness, citation correctness)
- `confidence/`  — ECE + reliability diagram

Ground truth comes free from `data/generate.py` (fields = extraction labels,
injected anomalies = anomaly labels). Author ~50–100 questions with gold chunks
for the RAG suite.

**Do not** measure: BLEU/ROUGE on the report, MTEB in the abstract, or five
redundant IR metrics. Recall@k + MRR + faithfulness is enough.
