"""PDF text extraction with OCR fallback for scanned pages."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
import pytesseract

if tesseract_cmd := os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

from app.core.exceptions import OcrUnavailableError


logger = logging.getLogger(__name__)

_MIN_TEXT_CHARS = 50  # Below this threshold a page is treated as scanned
_OCR_ZOOM = 2.0  # 2x zoom for better OCR accuracy

# Numbered heading jammed onto the same line as its body:
# "2. Membership The committee comprises..."
_INLINE_NUMBERED_HEADING = re.compile(
    r"^(?P<heading>\d+\.\s+[A-Z][A-Za-z0-9 &/'(),.-]{0,80})"
    r"\s+(?P<body>[A-Z].{20,})$"
)


@dataclass(frozen=True, slots=True)
class PageText:
    page: int     # 1-indexed
    text: str


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract text from every page of *pdf_path*.

    Uses PyMuPDF native extraction first; falls back to Tesseract OCR for
    pages that contain fewer than *_MIN_TEXT_CHARS* characters. Heading
    lines are preserved (and split off inline bodies) so the chunker can
    segment on section boundaries.

    Raises:
        OcrUnavailableError: a page needed OCR but Tesseract is not installed.
            The whole document fails rather than being indexed with missing
            pages, so a broken OCR setup cannot silently produce empty results.
    """
    import fitz  # PyMuPDF

    pages: list[PageText] = []
    doc = fitz.open(str(pdf_path))

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = _normalize_whitespace(page.get_text("text"))

            if len(text) < _MIN_TEXT_CHARS:
                text = _ocr_page(page, page_index + 1)
                if len(text) < _MIN_TEXT_CHARS:
                    logger.warning(
                        "%s page %d yielded almost no text (%d chars) even after "
                        "OCR; the scan may be blank or too low quality.",
                        pdf_path.name,
                        page_index + 1,
                        len(text),
                    )

            pages.append(
                PageText(page=page_index + 1, text=_structure_headings(text))
            )
    finally:
        doc.close()

    return pages


def _ocr_page(page: object, page_number: int) -> str:
    """Render *page* to an image and run Tesseract OCR on it.

    Raises:
        OcrUnavailableError: Tesseract is not installed or not on PATH.
        RuntimeError: Tesseract ran but failed on this page.
    """
    import io

    import fitz
    import pytesseract
    from PIL import Image

    mat = fitz.Matrix(_OCR_ZOOM, _OCR_ZOOM)
    pix = page.get_pixmap(matrix=mat)  # type: ignore[attr-defined]
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    try:
        return _normalize_whitespace(pytesseract.image_to_string(image))
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Tesseract is required to read scanned pages but was not found on PATH. "
            "Install it (macOS: `brew install tesseract`) and restart the process."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc


def _structure_headings(text: str) -> str:
    """Keep recognized headings on their own line.

    Native PyMuPDF output already isolates numbered headings. OCR and some
    PDF layouts glue a heading onto the following sentence; split those so
    the chunker can treat them as section boundaries.
    """
    if not text:
        return ""

    structured: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if structured and structured[-1] != "":
                structured.append("")
            continue
        match = _INLINE_NUMBERED_HEADING.match(line)
        if match:
            structured.append(match.group("heading"))
            structured.append(match.group("body"))
            continue
        structured.append(line)
    return "\n".join(structured).strip()


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
