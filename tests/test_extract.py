import importlib.util
import sys
from pathlib import Path

from evident.pipeline import extract

_spec = importlib.util.spec_from_file_location(
    "evident_extraction_metrics",
    Path(__file__).resolve().parents[1] / "eval" / "extraction_metrics.py",
)
em = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = em          # so dataclass type resolution can find the module
_spec.loader.exec_module(em)


# ---- the regex baseline (no OCR / no LLM needed) ----

def test_regex_extract_national_id():
    text = "N° CIN\n12345678\nNom / Name\nAhmed Saidi\nNé(e) le / DOB\n30/10/2003\nValable jusqu'au\n28/03/2030"
    out = extract.extract_fields(text, "national_id", backend="regex")
    assert out["id_number"] == "12345678"
    assert out["dob"] == "30/10/2003"
    assert out["expiry"] == "28/03/2030"
    assert "Ahmed" in out["full_name"]


def test_regex_extract_passport_number():
    text = "N° Passport\nA1234567\nNationalité\nTUN\nDOB\n01/01/1990"
    out = extract.extract_fields(text, "passport", backend="regex")
    assert out["passport_number"] == "A1234567"
    assert out["nationality"] == "TUN"


def test_extraction_confidence():
    assert extract.extraction_confidence("Ahmed Saidi", "... ahmed saidi ...") == 1.0
    assert extract.extraction_confidence("", "anything") == 0.0
    assert 0.0 < extract.extraction_confidence("Ahmad Saidi", "ahmed saidi") < 1.0


# ---- extraction metrics ----

def test_score_document_exact_match():
    gold = {"full_name": "Ahmed Saidi", "id_number": "12345678", "dob": "30/10/2003", "expiry": ""}
    pred = {"full_name": "ahmed  saidi", "id_number": "12345678", "dob": "30/10/2003", "expiry": ""}
    c = em.score_document(pred, gold, ("full_name", "id_number", "dob", "expiry"))
    m = em.prf1(c)
    # 3 gold-present fields, all correct (name matches after normalisation)
    assert m["n_gold"] == 3
    assert m["exact_match"] == 1.0


def test_normalize_value():
    assert em.normalize_value("  Ahmed   SAÏDI ") == "ahmed saidi"
