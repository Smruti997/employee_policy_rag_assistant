from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.data.document_metadata import DOCUMENT_METADATA
from app.data.ingestion import _open_store, prepare_documents


def test_prepare_documents_fails_before_reingest_if_pdf_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        prepare_documents(tmp_path)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Missing required PDF(s)" in err
    assert "leave_policy.pdf" in err


def test_open_store_uses_explicit_path(tmp_path: Path) -> None:
    args = SimpleNamespace(qdrant_path=str(tmp_path / "qdrant"), qdrant_url=None)
    store = _open_store(args)
    assert store.location == args.qdrant_path


def test_source_docs_cover_every_mapped_file() -> None:
    docs = Path(__file__).resolve().parents[2] / "app" / "docs"
    found = {p.name for p in docs.rglob("*.pdf")}
    assert set(DOCUMENT_METADATA) <= found
