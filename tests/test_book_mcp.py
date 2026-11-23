from pathlib import Path

import pytest

from book_mcp import config
from book_mcp.ingest import chunk_text, resolve_book_path
from book_mcp.store import escape_filter_value


def test_escape_filter_value():
    assert escape_filter_value('He said "hi"') == 'He said \\"hi\\"'


def test_chunk_text_respects_size():
    text = " ".join(["sentence"] * 200)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_resolve_book_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BOOKS_DIR", tmp_path)
    with pytest.raises(ValueError):
        resolve_book_path("../secrets.pdf")


def test_resolve_book_path_accepts_plain_name(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BOOKS_DIR", tmp_path)
    assert resolve_book_path("book.pdf") == Path(tmp_path / "book.pdf").resolve()
