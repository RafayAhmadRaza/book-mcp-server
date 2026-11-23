#!/usr/bin/env python3
"""Ingest a book into one of the Milvus collections.

Examples
--------
    python scripts/ingest_book.py --book "The Hobbit" --source ~/books/hobbit.epub
    python scripts/ingest_book.py --book "My Manual" --source manual.pdf --ocr --collection ocr

The collection defaults to the extracted-text one. Use ``--collection ocr``
(or ``--ocr``) for scanned PDFs that need Tesseract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import sys
from pathlib import Path

from book_mcp import config
from book_mcp.ingest import chunk_text, read_book
from book_mcp.store import ensure_collection, insert_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", required=True, help="title stored with every chunk")
    parser.add_argument("--source", required=True, type=Path, help=".txt, .md, .pdf or .epub file")
    parser.add_argument(
        "--collection",
        choices=["books", "ocr"],
        default=None,
        help="which collection to fill (default: books, or ocr when --ocr is used)",
    )
    parser.add_argument("--ocr", action="store_true", help="OCR a scanned PDF with Tesseract")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    source = args.source.expanduser()
    if not source.is_file():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1

    collection = args.collection or ("ocr" if args.ocr else "books")
    if collection == "ocr":
        collection = config.OCR_COLLECTION
    else:
        collection = config.TEXT_COLLECTION

    print(f"Reading {source} ...")
    text, label = read_book(source, ocr=args.ocr)
    if not text.strip():
        print("No text extracted; nothing to ingest.", file=sys.stderr)
        return 1

    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    print(f"Chunks: {len(chunks)} (source={label}, collection={collection})")

    ensure_collection(collection)
    written = insert_chunks(collection, chunks, book=args.book, source=label)
    print(f"Inserted {written} chunks into {collection}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
