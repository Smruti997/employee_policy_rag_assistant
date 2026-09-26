"""Section-aware chunking: one chunk per heading, with bounded overflow."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.data.pdf_reader import PageText


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str       # sha256(source_file:page:chunk_index)
    text: str
    source_file: str
    department: str
    access_level: int
    page: int
    chunk_index: int


# Sentence boundary: period / exclamation / question mark followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Paragraph boundary: two or more newlines
_PARAGRAPH_END = re.compile(r"\n{2,}")

# Numbered policy headings ("1. Overview", "6. Non-Reimbursable Items") and
# scanned-plan headings ("Priority 1: Accelerate international expansion...").
_NUMBERED_HEADING = re.compile(
    r"^\d+\.\s+[A-Z][A-Za-z0-9 &/'(),.-]{0,80}$"
)
_PRIORITY_HEADING = re.compile(
    r"^(?P<heading>Priority\s+\d+\s*:)(?P<body>.*)$",
    re.IGNORECASE,
)


def chunk_pages(
    pages: list[PageText],
    source_file: str,
    department: str,
    access_level: int,
    chunk_size: int = 450,
    overlap: int = 75,
) -> list[Chunk]:
    """Split *pages* into heading-bounded chunks.

    Strategy:
    - Detect numbered headings (``1. Overview``) and ``Priority N:`` lines.
    - The document title / unheaded preamble is its own opening section.
    - Each heading starts a new chunk; overlap never crosses sections.
    - Oversized sections split on paragraph then sentence boundaries, with
      bounded word overlap *inside* that section. Continuation chunks repeat
      the heading so retrieval still has the section label.
    - Documents with no recognized headings fall back to bounded sentence
      windows of *chunk_size* words with *overlap* carry-over.
    """
    sections = _split_into_sections(pages)
    if not sections:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0
    for heading, body_lines, start_page in sections:
        section_chunks, chunk_index = _chunk_section(
            heading=heading,
            body_lines=body_lines,
            start_page=start_page,
            source_file=source_file,
            department=department,
            access_level=access_level,
            chunk_size=chunk_size,
            overlap=overlap,
            chunk_index=chunk_index,
        )
        chunks.extend(section_chunks)
    return chunks


@dataclass(frozen=True, slots=True)
class _Line:
    page: int
    text: str


def _split_into_sections(pages: list[PageText]) -> list[tuple[str, list[_Line], int]]:
    """Return (heading, body_lines, start_page) for each section.

    Heading is empty for the title/preamble. Body lines keep page numbers.
    A ``Priority N:`` line may carry body text after the colon; that remainder
    stays in the new section rather than becoming the heading.
    """
    lines: list[_Line] = []
    for pt in pages:
        for raw in pt.text.splitlines():
            text = raw.strip()
            if text:
                lines.append(_Line(page=pt.page, text=text))

    if not lines:
        return []

    sections: list[tuple[str, list[_Line], int]] = []
    heading = ""
    body: list[_Line] = []
    start_page = lines[0].page

    def flush() -> None:
        if heading or body:
            sections.append((heading, list(body), start_page))

    for line in lines:
        priority = _PRIORITY_HEADING.match(line.text)
        if priority:
            flush()
            heading = re.sub(r"\s+", " ", priority.group("heading")).strip()
            rest = priority.group("body").strip()
            body = [_Line(page=line.page, text=rest)] if rest else []
            start_page = line.page
            continue
        if _NUMBERED_HEADING.match(line.text):
            flush()
            heading = line.text
            body = []
            start_page = line.page
            continue
        if not heading and not body:
            start_page = line.page
        body.append(line)

    flush()
    return sections


def _chunk_section(
    heading: str,
    body_lines: list[_Line],
    start_page: int,
    source_file: str,
    department: str,
    access_level: int,
    chunk_size: int,
    overlap: int,
    chunk_index: int,
) -> tuple[list[Chunk], int]:
    body_text = " ".join(line.text for line in body_lines).strip()
    if heading and not body_text:
        chunk = _make_chunk(
            heading, source_file, department, access_level, start_page, chunk_index
        )
        return [chunk], chunk_index + 1
    if not heading and not body_text:
        return [], chunk_index

    prefix = f"{heading}\n" if heading else ""
    prefix_words = heading.split() if heading else []
    # Room left for body after repeating the heading in continuation chunks.
    body_budget = max(chunk_size - len(prefix_words), 1)

    words = body_text.split()
    if len(prefix_words) + len(words) <= chunk_size:
        text = f"{prefix}{body_text}".strip()
        chunk = _make_chunk(
            text, source_file, department, access_level, start_page, chunk_index
        )
        return [chunk], chunk_index + 1

    sentences = _sentences_from_body(body_text)
    chunks: list[Chunk] = []
    buf: list[str] = []

    def emit(part: list[str]) -> None:
        nonlocal chunk_index
        body = " ".join(part).strip()
        text = f"{prefix}{body}".strip() if heading else body
        chunks.append(
            _make_chunk(
                text, source_file, department, access_level, start_page, chunk_index
            )
        )
        chunk_index += 1

    for sent in sentences:
        sent_words = sent.split()
        if buf and len(buf) + len(sent_words) > body_budget:
            emit(buf)
            overlap_words = buf[-overlap:] if overlap else []
            buf = overlap_words + sent_words
        else:
            buf.extend(sent_words)

    if buf:
        emit(buf)
    return chunks, chunk_index


def _sentences_from_body(body_text: str) -> list[str]:
    sentences: list[str] = []
    for para in _PARAGRAPH_END.split(body_text):
        para = para.strip()
        if not para:
            continue
        for sent in _SENTENCE_END.split(para):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences or [body_text]


def _make_chunk(
    text: str,
    source_file: str,
    department: str,
    access_level: int,
    page: int,
    chunk_index: int,
) -> Chunk:
    raw = f"{source_file}:{page}:{chunk_index}"
    chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_file=source_file,
        department=department,
        access_level=access_level,
        page=page,
        chunk_index=chunk_index,
    )
