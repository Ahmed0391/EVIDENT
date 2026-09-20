"""
Deterministic rule engine (blueprint §7, §12).

Everything here is EXACT and testable — no LLM, ever. Date logic, ID formats and
required-field checks are pure functions of their inputs, which is precisely why
they must not be delegated to a probabilistic model. Each check returns a
``RuleResult`` so the confidence layer and the report can cite it by id.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

_DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y")

CIN_RE = re.compile(r"^\d{8}$")               # Tunisian CIN: exactly 8 digits
PASSPORT_RE = re.compile(r"^[A-Z]\d{7}$")     # 1 letter + 7 digits


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    passed: bool
    detail: str = ""


def parse_date(value: str) -> date | None:
    """Parse a date string in any accepted format; return None if unparseable."""
    if not value:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def check_not_expired(expiry: str, today: date | None = None) -> RuleResult:
    today = today or date.today()
    d = parse_date(expiry)
    if d is None:
        return RuleResult("expiry.unparseable", False, f"cannot parse expiry {expiry!r}")
    return RuleResult(
        "document.expired",
        passed=d >= today,
        detail=f"expiry {d.isoformat()} {'>=' if d >= today else '<'} today {today.isoformat()}",
    )


def check_dob_plausible(dob: str, today: date | None = None) -> RuleResult:
    today = today or date.today()
    d = parse_date(dob)
    if d is None:
        return RuleResult("dob.unparseable", False, f"cannot parse dob {dob!r}")
    age = (today - d).days / 365.25
    return RuleResult("dob.plausible", passed=0 < age < 120, detail=f"age≈{age:.0f}")


def check_id_format(doc_type: str, value: str) -> RuleResult:
    if doc_type == "national_id":
        return RuleResult("id_format.cin", bool(CIN_RE.match(value or "")),
                          "expected 8 digits")
    if doc_type == "passport":
        return RuleResult("id_format.passport", bool(PASSPORT_RE.match(value or "")),
                          "expected letter + 7 digits")
    return RuleResult("id_format.na", True, "no format rule for this doc type")


def check_required_fields(doc_type: str, fields: dict, required: tuple[str, ...]) -> list[RuleResult]:
    out = []
    for key in required:
        present = bool(str(fields.get(key, "")).strip())
        out.append(RuleResult(f"required.{doc_type}.{key}", present,
                              "present" if present else "missing/illegible"))
    return out
