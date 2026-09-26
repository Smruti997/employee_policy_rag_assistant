"""Ingestion CLI: PDFs → chunks → embeddings → Qdrant.

Usage:
    python -m app.data.ingestion --source-dir app/docs --qdrant-path ./app/qdrant_db --recreate-collection
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import OcrUnavailableError
from app.data.chunker import Chunk, chunk_pages
from app.data.document_metadata import DOCUMENT_METADATA, get_metadata
from app.data.embeddings import default_embedder
from app.data.pdf_reader import extract_pages
from app.data.qdrant_store import QdrantStore


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    path: Path
    filename: str
    department: str
    access_level: int
    chunks: list[Chunk]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_dir = "app/docs" if Path("app/docs").exists() else "docs"
    parser = argparse.ArgumentParser(description="Ingest policy PDFs into Qdrant.")
    parser.add_argument("--source-dir", default=default_dir, help="Root docs folder")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    parser.add_argument("--qdrant-url", default=None, help="Qdrant server URL")
    parser.add_argument("--qdrant-path", default=None, help="Local Qdrant DB directory path")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    return parser.parse_args(argv)


def _open_store(args: argparse.Namespace) -> QdrantStore:
    if args.qdrant_path:
        return QdrantStore(path=args.qdrant_path)
    if args.qdrant_url:
        return QdrantStore(url=args.qdrant_url)
    return QdrantStore()


def prepare_documents(source_dir: Path) -> list[PreparedDocument]:
    """Extract and chunk every mapped PDF before touching Qdrant.

    Raises SystemExit on a missing mapped file, OCR failure, or empty extract
    so --recreate-collection cannot drop a good collection and then fail.
    """
    pdf_files = sorted(source_dir.rglob("*.pdf"))
    found = {path.name: path for path in pdf_files}
    missing = sorted(set(DOCUMENT_METADATA) - set(found))
    if missing:
        print(
            f"[ERROR] Missing required PDF(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    prepared: list[PreparedDocument] = []
    for filename, meta in DOCUMENT_METADATA.items():
        pdf_path = found[filename]
        print(f"[INFO] Processing {pdf_path} — dept={meta.department} level={meta.access_level}")
        try:
            pages = extract_pages(pdf_path)
        except OcrUnavailableError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"       {len(pages)} page(s) extracted")

        chunks = chunk_pages(
            pages=pages,
            source_file=filename,
            department=meta.department,
            access_level=meta.access_level,
        )
        print(f"       {len(chunks)} chunk(s) generated")
        if not chunks:
            print(f"[ERROR] No text extracted from {pdf_path}", file=sys.stderr)
            sys.exit(1)

        prepared.append(
            PreparedDocument(
                path=pdf_path,
                filename=filename,
                department=meta.department,
                access_level=meta.access_level,
                chunks=chunks,
            )
        )

    extras = sorted(name for name in found if name not in DOCUMENT_METADATA)
    for name in extras:
        print(f"[WARN] Skipping unknown file: {found[name]} (not in document_metadata)")
    return prepared


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    source_dir = Path(args.source_dir)

    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(list(source_dir.rglob('*.pdf')))} PDF file(s)")
    prepared = prepare_documents(source_dir)

    embedder = default_embedder()
    store = _open_store(args)
    print(f"[INFO] Embedder dimension: {embedder.dimension}")
    print(f"[INFO] Qdrant backend: {store.location}")
    store.create_collection(dimension=embedder.dimension, recreate=args.recreate_collection)
    print(f"[INFO] Qdrant collection ready at {store.location}")

    for doc in prepared:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(doc.chunks), args.batch_size):
            batch = doc.chunks[i : i + args.batch_size]
            all_embeddings.extend(embedder.embed([c.text for c in batch]))
        store.upsert_chunks(doc.chunks, all_embeddings)
        print(f"       Upserted {doc.filename} ({len(doc.chunks)} chunks) ✓")

    print(f"[INFO] Ingestion complete. {sum(len(d.chunks) for d in prepared)} chunk(s) indexed.")


if __name__ == "__main__":
    run()
