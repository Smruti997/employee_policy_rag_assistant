from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import OcrUnavailableError
from app.data import pdf_reader

DOCS = Path(__file__).resolve().parents[2] / "app" / "docs"
TEXT_PDF = DOCS / "hr" / "leave_policy.pdf"
SCANNED_PDF = DOCS / "exec" / "strategic_plan.pdf"


def test_text_pdf_uses_native_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page with a real text layer must not invoke OCR."""

    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("OCR must not run for text-based PDFs")

    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_string", fail)

    pages = pdf_reader.extract_pages(TEXT_PDF)

    assert pages and sum(len(p.text) for p in pages) > 0


def test_scanned_pdf_falls_back_to_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_string", lambda *_a, **_k: "Recovered scan text.")

    pages = pdf_reader.extract_pages(SCANNED_PDF)

    assert pages[0].text == "Recovered scan text."


def test_missing_tesseract_raises_instead_of_silently_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The previous bare `except` returned "" and hid a broken OCR setup."""

    import pytesseract

    def missing(*_args: object, **_kwargs: object) -> str:
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", missing)

    with pytest.raises(OcrUnavailableError):
        pdf_reader.extract_pages(SCANNED_PDF)


def test_structure_headings_splits_inline_numbered_heading() -> None:
    text = pdf_reader._structure_headings(
        "1. Purpose\nThe committee reviews pay.\n"
        "2. Membership The committee comprises four independent directors of the board."
    )
    assert "2. Membership\nThe committee comprises" in text


def test_leave_policy_preserves_numbered_heading_lines() -> None:
    pages = pdf_reader.extract_pages(TEXT_PDF)
    lines = pages[0].text.splitlines()
    assert "1. Overview" in lines
    assert "3. Sick Leave" in lines
