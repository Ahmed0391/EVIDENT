"""
End-to-end pipeline test (Phase 2). Skips cleanly when OCR isn't available, so
`pytest` stays green on a machine without Tesseract installed.
"""
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pytesseract")
# Honor TESSERACT_CMD (Windows installs often aren't on PATH) as well as PATH.
_TESS = shutil.which("tesseract") or (
    os.getenv("TESSERACT_CMD") if os.path.exists(os.getenv("TESSERACT_CMD", "")) else None
)
if _TESS is None:
    pytest.skip("tesseract not found (PATH or TESSERACT_CMD)", allow_module_level=True)

from evident.pipeline.cases import list_cases, load_case  # noqa: E402
from evident.pipeline.orchestrator import run_pipeline, CaseResult  # noqa: E402
from evident.pipeline.report import build_report, render_markdown  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_DATA = next((d for d in ("data/demo", "data/synthetic") if (_ROOT / d / "labels").exists()), None)
if _DATA is None:
    pytest.skip("no dataset — run data/generate.py first", allow_module_level=True)
_DATA = str(_ROOT / _DATA)


def test_pipeline_runs_end_to_end():
    case_id = list_cases(_DATA)[0]
    result = run_pipeline(load_case(case_id, _DATA))
    assert isinstance(result, CaseResult)
    assert result.findings, "should extract at least some fields"
    assert result.status in {"auto_cleared", "needs_review"}
    assert 0.0 <= result.confidence <= 1.0


def test_report_shape():
    result = run_pipeline(load_case(list_cases(_DATA)[0], _DATA))
    rep = build_report(result)
    assert {"case_id", "status", "confidence", "anomalies", "fields"} <= rep.keys()
    assert isinstance(render_markdown(result), str)


def test_expired_document_is_flagged():
    """If the dataset has an expired-doc case, the pipeline must route it to review."""
    import json
    cases_file = Path(_DATA) / "cases.jsonl"
    expired = [json.loads(l)["case_id"] for l in cases_file.read_text().splitlines()
               if "expired_document" in json.loads(l)["anomaly_types"]]
    if not expired:
        pytest.skip("no expired-document case in this dataset")
    result = run_pipeline(load_case(expired[0], _DATA))
    assert result.status == "needs_review"
    assert any(a.type == "expired_document" for a in result.anomalies)
