from __future__ import annotations

import asyncio

import pytest
from unittest.mock import MagicMock

from app.auth.rbac import AccessScope
from app.data.chunker import chunk_pages
from app.data.document_metadata import get_metadata
from app.data.pdf_reader import PageText
from app.data.retriever import Retriever


def test_document_metadata_lookup():
    meta = get_metadata("leave_policy.pdf")
    assert meta.department == "hr"
    assert meta.access_level == 1


def test_chunk_pages():
    pages = [
        PageText(
            page=1,
            text="First sentence here. Second sentence here. Third sentence here.",
        )
    ]
    chunks = chunk_pages(
        pages=pages,
        source_file="leave_policy.pdf",
        department="hr",
        access_level=1,
        chunk_size=10,
        overlap=2,
    )
    assert len(chunks) >= 1
    assert chunks[0].source_file == "leave_policy.pdf"
    assert chunks[0].department == "hr"
    assert chunks[0].access_level == 1


def test_chunk_pages_splits_on_numbered_headings() -> None:
    pages = [
        PageText(
            page=1,
            text=(
                "Acme Corp - Leave Policy\n"
                "1. Overview\n"
                "This leave policy applies to all full-time employees.\n"
                "2. Annual Leave Entitlement\n"
                "Full-time employees are entitled to 22 working days."
            ),
        )
    ]
    chunks = chunk_pages(
        pages=pages,
        source_file="leave_policy.pdf",
        department="hr",
        access_level=1,
    )
    texts = [c.text for c in chunks]
    assert texts[0] == "Acme Corp - Leave Policy"
    assert texts[1].startswith("1. Overview\n")
    assert "22 working days" not in texts[1]
    assert texts[2].startswith("2. Annual Leave Entitlement\n")
    assert all(c.department == "hr" and c.access_level == 1 for c in chunks)


def test_chunk_pages_splits_inline_numbered_heading() -> None:
    from app.data.pdf_reader import _structure_headings

    structured = _structure_headings(
        "1. Purpose\n"
        "The Code of Conduct sets out the standards.\n"
        "2. Respectful Workplace Harassment is not tolerated anywhere in the company."
    )
    pages = [PageText(page=1, text=structured)]
    chunks = chunk_pages(
        pages=pages,
        source_file="code_of_conduct.pdf",
        department="hr",
        access_level=1,
    )
    assert any(c.text.startswith("1. Purpose\n") for c in chunks)
    respectful = next(c for c in chunks if c.text.startswith("2. Respectful Workplace"))
    assert "Harassment is not tolerated" in respectful.text
    assert "Code of Conduct" not in respectful.text


def test_chunk_pages_splits_priority_headings() -> None:
    pages = [
        PageText(
            page=1,
            text=(
                "Strategic Plan 2026-2028 (Scanned Copy)\n"
                "Acme Corp Strategic Plan 2026 to 2028.\n"
                "Priority 1: Accelerate international expansion in the APAC region.\n"
                "Priority 2: Shift the product portfolio toward recurring revenue.\n"
                "Priority 3: Reduce unit cost of delivery by 15 percent.\n"
                "Priority 4: Build an AI-first operating model.\n"
                "Board approval for this plan was secured on 15 February 2026."
            ),
        )
    ]
    chunks = chunk_pages(
        pages=pages,
        source_file="strategic_plan.pdf",
        department="exec",
        access_level=3,
    )
    headings = [c.text.split("\n", 1)[0] for c in chunks]
    assert headings[0] == "Strategic Plan 2026-2028 (Scanned Copy) Acme Corp Strategic Plan 2026 to 2028."
    assert headings[1] == "Priority 1:"
    assert headings[2] == "Priority 2:"
    assert headings[3] == "Priority 3:"
    assert headings[4] == "Priority 4:"
    assert "Board approval" in chunks[4].text
    assert "APAC" not in chunks[2].text


def test_chunk_pages_keeps_overlap_inside_oversized_section() -> None:
    body = " ".join(f"Sentence {i} continues here." for i in range(20))
    pages = [
        PageText(page=1, text=f"1. Overview\n{body}\n2. Next\nShort body."),
        PageText(page=2, text="More of section two stays with Next."),
    ]
    chunks = chunk_pages(
        pages=pages,
        source_file="leave_policy.pdf",
        department="hr",
        access_level=1,
        chunk_size=20,
        overlap=4,
    )
    overview = [c for c in chunks if c.text.startswith("1. Overview")]
    nxt = [c for c in chunks if c.text.startswith("2. Next")]
    assert len(overview) >= 2
    assert all("2. Next" not in c.text for c in overview)
    assert nxt[0].page == 1
    assert "More of section two" in nxt[0].text


def test_retriever_builds_rbac_filter():
    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    retriever = Retriever(store=mock_store, embedder=mock_embedder)
    scope = AccessScope(departments=("company", "hr"), maximum_access_level=2)

    # search is async: it offloads embedding and the store query off the loop.
    asyncio.run(retriever.search("sick leave", scope))

    mock_store.search.assert_called_once()
    args, kwargs = mock_store.search.call_args
    assert args[0] == [0.1, 0.2, 0.3]
    rbac_filter = args[1]
    assert "must" in rbac_filter
