#!/usr/bin/env python3
"""Query the book collections without an MCP client (useful for smoke tests)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from book_mcp import config
from book_mcp.store import search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--collection", choices=["books", "ocr"], default="books")
    parser.add_argument("--book", default=None, help="restrict to one book title")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    collection = config.OCR_COLLECTION if args.collection == "ocr" else config.TEXT_COLLECTION
    hits = search(collection, args.query, limit=args.limit, book=args.book)

    if not hits:
        print("No results.")
        return

    for hit in hits:
        print("\n---")
        print(f"Book:   {hit['book']}")
        print(f"Source: {hit['source']}")
        print(f"Text:   {hit['text']}")


if __name__ == "__main__":
    main()
