#!/usr/bin/env python3
"""Create the two Milvus Lite collections used by the Book MCP server."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from book_mcp import config
from book_mcp.store import ensure_collection, get_embedding_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="embedding dimension (defaults to the embedding model's size)",
    )
    args = parser.parse_args()

    dim = args.dim or config.EMBEDDING_DIM
    if args.dim is None:
        try:
            dim = int(get_embedding_fn().dim())
        except Exception:
            # Fall back to the configured default if the model is unavailable.
            pass

    for collection in (config.TEXT_COLLECTION, config.OCR_COLLECTION):
        ensure_collection(collection, dim=dim)
        print(f"{collection}: ready (dim={dim})")


if __name__ == "__main__":
    main()
