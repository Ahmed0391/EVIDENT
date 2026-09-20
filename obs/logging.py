"""
Observability (blueprint §15) — lightweight on purpose. Structured JSON logs with
a trace_id per case, plus a place to record stage timings. No MLOps platform.

Phase 5 adds Prometheus counters/histograms at /metrics; this gives you structured
logging from day one.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "msg": record.getMessage(), "logger": record.name}
        for key in ("trace_id", "stage", "ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "evident") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timed(logger: logging.Logger, stage: str, trace_id: str = "-"):
    """Log how long a pipeline stage took (ocr_ms, llm_ms, retrieval_ms ...)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        logger.info(f"{stage} done", extra={"stage": stage, "ms": round(ms, 1), "trace_id": trace_id})
