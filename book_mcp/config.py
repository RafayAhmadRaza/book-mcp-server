"""Configuration for the Book MCP server.

Every path can be overridden with an environment variable so the server can be
pointed at a real library without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (the directory containing this package).
ROOT = Path(__file__).resolve().parent.parent

try:  # optional convenience for local development
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (ROOT / path)


# Milvus Lite database. Never commit this file.
DB_PATH = _path_from_env("BOOK_MCP_DB", ROOT / "data" / "books.db")

# Directory searched by ``ocr_book_page`` and the ``book://`` resource.
BOOKS_DIR = _path_from_env("BOOKS_DIR", ROOT / "books")

TEXT_COLLECTION = os.getenv("BOOK_COLLECTION", "books_collection")
OCR_COLLECTION = os.getenv("OCR_COLLECTION", "ocr_books_collection")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# The default ONNX embedding function (paraphrase-albert-small-v2) is 768-d.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
