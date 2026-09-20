"""
Evident — synthetic KYC dataset generator.

Design principle (see the blueprint, §18): generate the STRUCTURED DATA FIRST,
render the document image SECOND. Because we invent every field before drawing it,
the ground-truth labels come for free — which is exactly what makes the whole
system evaluable later.

For each case we create 2–3 documents (national ID / passport / proof of address),
optionally inject *labelled* anomalies (expired doc, name mismatch across docs,
missing field, invalid ID format, conflicting date of birth), and optionally
degrade the image (blur / rotate / noise) so OCR has something to struggle with.

Two kinds of ground truth are written, and they are NOT the same thing:
  • fields      -> what is actually PRINTED on the image  (OCR / extraction target)
  • anomalies   -> case-level facts about inconsistencies  (anomaly-detection target)

Outputs (under --out, default data/synthetic/):
  images/<case_id>/<doc_id>.png     the rendered document
  labels/<case_id>.json             full ground truth for the case
  manifest.jsonl                    one line per DOCUMENT (extraction eval)
  cases.jsonl                       one line per CASE     (anomaly eval)

Usage:
  python data/generate.py --n 50 --seed 42
  python data/generate.py --n 200 --out data/synthetic --anomaly-rate 0.5

Dependencies: faker, pillow  (see pyproject.toml).
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path

try:
    from faker import Faker
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: pip install faker pillow") from e
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DOC_TYPES = ("national_id", "passport", "proof_of_address")

# The anomalies we know how to inject, each with a stable label id.
ANOMALIES = (
    "expired_document",
    "name_mismatch",
    "conflicting_dob",
    "missing_field",
    "invalid_id_format",
)

DIFFICULTIES = ("clean", "easy", "medium", "hard")

# Fields required per document type (drives the missing-field check).
REQUIRED_FIELDS = {
    "national_id": ("full_name", "id_number", "dob", "expiry"),
    "passport": ("full_name", "passport_number", "nationality", "dob", "expiry"),
    "proof_of_address": ("full_name", "address", "issue_date", "provider"),
}


# --------------------------------------------------------------------------- #
# Fonts (best-effort; falls back to PIL's bitmap font if DejaVu is absent)
# --------------------------------------------------------------------------- #

def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Document:
    doc_id: str
    doc_type: str
    fields: dict          # what is printed on the image == extraction ground truth
    difficulty: str
    image_path: str = ""


@dataclass
class Case:
    case_id: str
    person: dict          # the canonical person (before any perturbation)
    documents: list       # list[Document]
    anomalies: list = field(default_factory=list)   # anomaly-detection ground truth


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace — used to perturb names subtly."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def _fmt(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _cin_number(rng: random.Random) -> str:
    # Tunisian CIN: 8 digits.
    return "".join(str(rng.randint(0, 9)) for _ in range(8))


def _passport_number(rng: random.Random) -> str:
    # One letter + 7 digits, e.g. "A1234567".
    return rng.choice("ABCDEFGHJKLMNP") + "".join(str(rng.randint(0, 9)) for _ in range(7))


def _typo(name: str, rng: random.Random) -> str:
    """Produce a *near* variant of a name (transliteration / spelling drift)."""
    swaps = {"Mohamed": "Mohammed", "Ahmed": "Ahmad", "Youssef": "Yousef",
             "Aymen": "Aymane", "Sarra": "Sara", "Khalil": "Khaleel"}
    for k, v in swaps.items():
        if k in name:
            return name.replace(k, v)
    # otherwise duplicate or drop a letter somewhere in the surname
    parts = name.split()
    i = rng.randrange(len(parts[-1]) - 1) if len(parts[-1]) > 2 else 0
    surname = parts[-1]
    parts[-1] = surname[:i] + surname[i] + surname[i:]  # duplicate a char
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Case construction (data first!)
# --------------------------------------------------------------------------- #

def build_case(idx: int, fake: Faker, rng: random.Random, anomaly_rate: float,
               force_difficulty: str | None = None) -> Case:
    case_id = f"case_{idx:05d}"
    today = date.today()

    first = fake.first_name()
    last = fake.last_name()
    full_name = f"{first} {last}"
    dob = fake.date_between(start_date="-60y", end_date="-19y")

    person = {
        "first_name": first,
        "last_name": last,
        "full_name": full_name,
        "dob": _fmt(dob),
        "nationality": "TUN",
        "address": fake.address().replace("\n", ", "),
    }

    # Decide which anomalies (if any) this case carries.
    injected: list[dict] = []
    if rng.random() < anomaly_rate:
        n = rng.choices([1, 2], weights=[0.75, 0.25])[0]
        injected_types = rng.sample(ANOMALIES, k=n)
    else:
        injected_types = []

    # Base valid values.
    id_number = _cin_number(rng)
    passport_number = _passport_number(rng)
    issue = fake.date_between(start_date="-8y", end_date="-1y")
    expiry = date(issue.year + 10, issue.month, min(issue.day, 28))

    # Start from clean printed values for each doc.
    id_fields = {
        "full_name": full_name, "id_number": id_number,
        "dob": _fmt(dob), "expiry": _fmt(expiry),
    }
    pp_fields = {
        "full_name": full_name, "passport_number": passport_number,
        "nationality": "TUN", "dob": _fmt(dob), "expiry": _fmt(expiry),
    }
    poa_fields = {
        "full_name": full_name, "address": person["address"],
        "issue_date": _fmt(fake.date_between(start_date="-3m", end_date="today")),
        "provider": rng.choice(["STEG", "SONEDE", "Ooredoo", "Topnet", "Orange TN"]),
    }

    # Apply the chosen anomalies by *perturbing what gets printed*.
    for a in injected_types:
        if a == "expired_document":
            past_expiry = today - timedelta(days=rng.randint(30, 900))
            target = rng.choice([id_fields, pp_fields])
            target["expiry"] = _fmt(past_expiry)
            injected.append({"type": a, "detail": "printed expiry is in the past"})
        elif a == "name_mismatch":
            poa_fields["full_name"] = _typo(full_name, rng)
            injected.append({"type": a,
                             "detail": f"proof-of-address name '{poa_fields['full_name']}' "
                                       f"!= id name '{full_name}'"})
        elif a == "conflicting_dob":
            alt = dob + timedelta(days=rng.choice([-1, 1]) * rng.randint(365, 3650))
            pp_fields["dob"] = _fmt(alt)
            injected.append({"type": a, "detail": "passport DOB differs from national ID"})
        elif a == "missing_field":
            target, key = rng.choice([
                (id_fields, "expiry"), (pp_fields, "nationality"), (poa_fields, "provider"),
            ])
            target[key] = ""  # printed as blank
            injected.append({"type": a, "detail": f"field '{key}' missing/illegible"})
        elif a == "invalid_id_format":
            id_fields["id_number"] = id_number[:6]  # too short → fails format check
            injected.append({"type": a, "detail": "CIN number is not 8 digits"})

    difficulty = force_difficulty or rng.choices(DIFFICULTIES, weights=[0.25, 0.35, 0.25, 0.15])[0]

    docs = [
        Document(f"{case_id}_id", "national_id", id_fields, difficulty),
        Document(f"{case_id}_pp", "passport", pp_fields, difficulty),
        Document(f"{case_id}_poa", "proof_of_address", poa_fields, difficulty),
    ]
    # Occasionally drop the passport to create single-source / missing-doc cases.
    if rng.random() < 0.15:
        docs = [docs[0], docs[2]]

    return Case(case_id=case_id, person=person, documents=docs, anomalies=injected)


# --------------------------------------------------------------------------- #
# Rendering (image second)
# --------------------------------------------------------------------------- #

_LABELS = {
    "national_id": "RÉPUBLIQUE — CARTE D'IDENTITÉ NATIONALE",
    "passport": "PASSEPORT / PASSPORT",
    "proof_of_address": "JUSTIFICATIF DE DOMICILE",
}
_FIELD_LABELS = {
    "full_name": "Nom / Name", "id_number": "N° CIN", "passport_number": "N° Passport",
    "nationality": "Nationalité", "dob": "Né(e) le / DOB", "expiry": "Valable jusqu'au",
    "address": "Adresse", "issue_date": "Date du document", "provider": "Émetteur",
}


def render(doc: Document, out_dir: Path, rng: random.Random) -> None:
    W, H = 720, 460
    bg = (247, 246, 240) if doc.doc_type != "passport" else (238, 242, 248)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    title_f = _load_font(22, bold=True)
    label_f = _load_font(14)
    value_f = _load_font(19, bold=True)

    # header band
    d.rectangle([0, 0, W, 56], fill=(15, 92, 94))
    d.text((20, 16), _LABELS[doc.doc_type], font=title_f, fill=(255, 255, 255))

    # a fake photo box for id/passport
    if doc.doc_type in ("national_id", "passport"):
        d.rectangle([W - 160, 80, W - 30, 250], outline=(120, 120, 120), width=2)
        d.text((W - 150, 155), "PHOTO", font=label_f, fill=(150, 150, 150))

    y = 90
    for key in REQUIRED_FIELDS[doc.doc_type]:
        val = doc.fields.get(key, "")
        d.text((28, y), _FIELD_LABELS.get(key, key), font=label_f, fill=(90, 90, 90))
        d.text((28, y + 18), str(val) if val else "—", font=value_f, fill=(20, 20, 20))
        y += 60

    # MRZ-ish strip for passports
    if doc.doc_type == "passport":
        mrz = "P<TUN" + _norm(doc.fields["full_name"]).replace(" ", "<<").upper()
        d.rectangle([0, H - 46, W, H], fill=(225, 228, 233))
        d.text((16, H - 38), (mrz + "<" * 40)[:46], font=_load_font(15), fill=(40, 40, 40))
        d.text((16, H - 20), (doc.fields.get("passport_number", "") + "<" * 40)[:46],
               font=_load_font(15), fill=(40, 40, 40))

    img = _degrade(img, doc.difficulty, rng)

    case_dir = out_dir / "images" / doc.doc_id.rsplit("_", 1)[0]
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{doc.doc_id}.png"
    img.save(path)
    doc.image_path = str(path.relative_to(out_dir))


def _degrade(img: Image.Image, difficulty: str, rng: random.Random) -> Image.Image:
    if difficulty == "clean":
        return img
    if difficulty in ("medium", "hard"):
        img = img.rotate(rng.uniform(-3.5, 3.5), expand=False,
                         fillcolor=(247, 246, 240), resample=Image.BICUBIC)
    blur = {"easy": 0.4, "medium": 0.9, "hard": 1.6}[difficulty]
    img = img.filter(ImageFilter.GaussianBlur(blur))
    if difficulty in ("medium", "hard"):
        # additive gaussian noise
        import struct
        px = img.load()
        sigma = 10 if difficulty == "medium" else 22
        for _ in range(int(img.width * img.height * (0.12 if difficulty == "hard" else 0.05))):
            x, y = rng.randrange(img.width), rng.randrange(img.height)
            r, g, b = px[x, y]
            n = int(rng.gauss(0, sigma))
            px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)),
                        max(0, min(255, b + n)))
    return img


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic KYC dataset with ground truth.")
    ap.add_argument("--n", type=int, default=50, help="number of cases")
    ap.add_argument("--seed", type=int, default=42, help="rng seed (reproducible)")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"), help="output dir")
    ap.add_argument("--anomaly-rate", type=float, default=0.5,
                    help="fraction of cases that carry at least one anomaly")
    ap.add_argument("--locale", type=str, default="fr_FR", help="Faker locale")
    ap.add_argument("--difficulty", type=str, default=None, choices=DIFFICULTIES,
                    help="force all documents to this difficulty (default: random mix)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fake = Faker(args.locale)
    Faker.seed(args.seed)

    out = args.out
    # Clean prior output so regenerating is reproducible (no stale cases linger).
    import shutil
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for f in ("manifest.jsonl", "cases.jsonl"):
        (out / f).unlink(missing_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    manifest = open(out / "manifest.jsonl", "w", encoding="utf-8")
    cases_f = open(out / "cases.jsonl", "w", encoding="utf-8")

    n_docs = 0
    n_anom = 0
    for i in range(args.n):
        case = build_case(i, fake, rng, args.anomaly_rate, force_difficulty=args.difficulty)
        for doc in case.documents:
            render(doc, out, rng)
            manifest.write(json.dumps({
                "case_id": case.case_id, "doc_id": doc.doc_id, "doc_type": doc.doc_type,
                "image": doc.image_path, "difficulty": doc.difficulty,
                "fields": doc.fields,
            }, ensure_ascii=False) + "\n")
            n_docs += 1

        # per-case ground truth file
        (out / "labels" / f"{case.case_id}.json").write_text(
            json.dumps({
                "case_id": case.case_id, "person": case.person,
                "documents": [asdict(d) for d in case.documents],
                "anomalies": case.anomalies,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        cases_f.write(json.dumps({
            "case_id": case.case_id,
            "n_documents": len(case.documents),
            "anomaly_types": [a["type"] for a in case.anomalies],
            "has_anomaly": bool(case.anomalies),
        }, ensure_ascii=False) + "\n")
        n_anom += bool(case.anomalies)

    manifest.close()
    cases_f.close()

    print(f"✓ wrote {args.n} cases · {n_docs} documents · "
          f"{n_anom} cases with anomalies ({n_anom / args.n:.0%})")
    print(f"  images   → {out/'images'}")
    print(f"  labels   → {out/'labels'}")
    print(f"  manifest → {out/'manifest.jsonl'}  (extraction eval)")
    print(f"  cases    → {out/'cases.jsonl'}      (anomaly eval)")


if __name__ == "__main__":
    main()
