"""
OCR service (blueprint §6, §22 module ①).

A backend service wrapping a learned model — NOT an agent. Returns text, per-word
confidence and bounding boxes (evidence #1 and #2 in the ledger).

Two pluggable backends so this runs on any machine:
  • "tesseract" (default) — light + fast, but needs the Tesseract binary installed.
      Windows: install from https://github.com/UB-Mannheim/tesseract/wiki , then either
      add it to PATH or set the env var TESSERACT_CMD to tesseract.exe's full path.
      Then: pip install pytesseract
  • "easyocr" — no system binary (pure pip), downloads models on first run, pulls
      torch (heavy) and is slower on CPU. Good fallback if you can't install Tesseract.
      Then: pip install easyocr

Pick via the `backend` argument or the OCR_BACKEND env var.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class OcrToken:
    text: str
    confidence: float          # 0..1
    bbox: list[float]          # [x0, y0, x1, y1]


@dataclass
class OcrResult:
    text: str
    tokens: list[OcrToken] = field(default_factory=list)
    mean_confidence: float = 0.0


def run_ocr(image_path: str, backend: str | None = None, lang: str = "fra+eng") -> OcrResult:
    backend = backend or os.getenv("OCR_BACKEND", "tesseract")
    if backend == "tesseract":
        return _tesseract(image_path, lang)
    if backend == "easyocr":
        return _easyocr(image_path)
    raise ValueError(f"unknown OCR backend: {backend!r} (use 'tesseract' or 'easyocr')")


# --------------------------------------------------------------------------- #
# Tesseract
# --------------------------------------------------------------------------- #

def _tesseract(image_path: str, lang: str) -> OcrResult:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Tesseract backend needs: pip install pytesseract pillow  (and the Tesseract "
            "binary installed). See this module's docstring for Windows steps."
        ) from e

    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    img = Image.open(image_path)
    # Fall back to English-only if the French language pack isn't installed.
    try:
        data = pytesseract.image_to_data(img, lang=lang,
                                         output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        data = pytesseract.image_to_data(img, lang="eng",
                                         output_type=pytesseract.Output.DICT)

    tokens: list[OcrToken] = []
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:        # -1 == no text in that box
            continue
        c = conf / 100.0
        tokens.append(OcrToken(
            text=word, confidence=c,
            bbox=[float(data["left"][i]), float(data["top"][i]),
                  float(data["left"][i] + data["width"][i]),
                  float(data["top"][i] + data["height"][i])],
        ))
        confs.append(c)

    text = _reflow(data)
    return OcrResult(text=text, tokens=tokens,
                     mean_confidence=sum(confs) / len(confs) if confs else 0.0)


def _reflow(data: dict) -> str:
    """Rebuild readable lines from Tesseract's word-level output."""
    lines: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    return "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))


# --------------------------------------------------------------------------- #
# EasyOCR
# --------------------------------------------------------------------------- #

def _easyocr(image_path: str) -> OcrResult:  # pragma: no cover - optional/heavy
    try:
        import easyocr
    except ImportError as e:
        raise SystemExit("EasyOCR backend needs: pip install easyocr") from e

    reader = _easyocr_reader()
    results = reader.readtext(image_path)   # [(bbox, text, conf), ...]
    tokens, confs, parts = [], [], []
    for box, txt, conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        tokens.append(OcrToken(text=txt, confidence=float(conf),
                               bbox=[min(xs), min(ys), max(xs), max(ys)]))
        confs.append(float(conf))
        parts.append(txt)
    return OcrResult(text="\n".join(parts), tokens=tokens,
                     mean_confidence=sum(confs) / len(confs) if confs else 0.0)


_READER = None


def _easyocr_reader():  # pragma: no cover
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["fr", "en"], gpu=False)
    return _READER
