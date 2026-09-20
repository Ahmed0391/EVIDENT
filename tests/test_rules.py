from datetime import date

from evident.pipeline import rules


def test_expired_document_detected():
    r = rules.check_not_expired("01/01/2020", today=date(2026, 1, 1))
    assert r.passed is False and r.rule_id == "document.expired"


def test_valid_document_passes():
    r = rules.check_not_expired("01/01/2030", today=date(2026, 1, 1))
    assert r.passed is True


def test_cin_format():
    assert rules.check_id_format("national_id", "12345678").passed is True
    assert rules.check_id_format("national_id", "123456").passed is False  # too short


def test_passport_format():
    assert rules.check_id_format("passport", "A1234567").passed is True
    assert rules.check_id_format("passport", "12345678").passed is False


def test_missing_required_field():
    fields = {"full_name": "Ahmed Saidi", "id_number": "12345678", "dob": "30/10/2003", "expiry": ""}
    results = rules.check_required_fields("national_id", fields,
                                         ("full_name", "id_number", "dob", "expiry"))
    missing = [r for r in results if not r.passed]
    assert len(missing) == 1 and missing[0].rule_id.endswith("expiry")


def test_parse_date_formats():
    assert rules.parse_date("2003-10-30") == date(2003, 10, 30)
    assert rules.parse_date("30/10/2003") == date(2003, 10, 30)
    assert rules.parse_date("not a date") is None
