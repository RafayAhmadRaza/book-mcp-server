"""An MCP server for searching a personal library.

Two collections are supported:

* ``books_collection``     - text extracted from EPUB/PDF/TXT books
* ``ocr_books_collection`` - text recovered from scanned PDFs with OCR

Both live in a local Milvus Lite database and are queried with the same
default ONNX embedding function.
"""

from __future__ import annotations

import logging

from mcp.server import MCPServer

from book_mcp import config
from book_mcp.ingest import ocr_pdf_page, resolve_book_path
from book_mcp.store import list_books as _list_books
from book_mcp.store import search as _search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("book_mcp")

mcp = MCPServer("Book Search")


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "No relevant passages found."
    return "\n---\n".join(
        f"Book: {hit['book']}\nSource: {hit['source']}\nText: {hit['text']}\n"
        for hit in hits
    )


def _safe_search(collection: str, query: str, limit: int, book: str | None = None) -> str:
    try:
        return _format_hits(_search(collection, query, limit=limit, book=book))
    except ValueError as error:
        return str(error)
    except Exception as error:  # keep tool failures readable for the model
        logger.exception("search failed")
        return f"Search failed: {error}"


def _safe_list(collection: str) -> str:
    try:
        books = _list_books(collection)
    except Exception as error:
        logger.exception("listing failed")
        return f"Could not list books: {error}"

    if not books:
        return "No books found in the database."
    return "\n".join(books)


@mcp.tool()
def list_books() -> str:
    """List all books indexed from extracted text (EPUB/PDF/TXT)."""
    return _safe_list(config.TEXT_COLLECTION)


@mcp.tool()
def list_ocr_books() -> str:
    """List all books indexed from OCR'd scans."""
    return _safe_list(config.OCR_COLLECTION)


@mcp.tool()
def search_book(query: str, limit: int = 5) -> str:
    """Semantic search across the extracted-text book collection."""
    return _safe_search(config.TEXT_COLLECTION, query, limit)


@mcp.tool()
def search_ocr_book(query: str, limit: int = 5) -> str:
    """Semantic search across the OCR'd book collection."""
    return _safe_search(config.OCR_COLLECTION, query, limit)


@mcp.tool()
def search_book_in(book: str, query: str, limit: int = 5) -> str:
    """Search for relevant passages inside one specific extracted-text book."""
    return _safe_search(config.TEXT_COLLECTION, query, limit, book=book)


@mcp.tool()
def search_book_in_ocr(book: str, query: str, limit: int = 5) -> str:
    """Search for relevant passages inside one specific OCR'd book."""
    return _safe_search(config.OCR_COLLECTION, query, limit, book=book)


@mcp.tool()
def ocr_book_page(book: str, page_number: int) -> str:
    """OCR a single page from a PDF stored in BOOKS_DIR."""
    try:
        path = resolve_book_path(book)
        if not path.is_file():
            return f"Book not found: {book}"

        text = ocr_pdf_page(path, page_number)
        if not text.strip():
            return f"No text could be extracted from page {page_number}."

        return f"Book: {book}\nPage: {page_number}\n\n{text}"
    except Exception as error:
        return f"OCR failed: {error}"


@mcp.tool()
def get_quiz_context(book: str, topic: str, count: int = 5) -> str:
    """Retrieve passages from an OCR'd book for writing quiz questions."""
    count = max(1, min(count, 10))
    query = (
        f"Important information, events, characters, facts, "
        f"and details about {topic}"
    )

    try:
        hits = _search(config.OCR_COLLECTION, query, limit=count, book=book)
    except Exception as error:
        return f"Quiz context lookup failed: {error}"

    if not hits:
        return f"No relevant passages found in '{book}' about '{topic}'."

    passages = "\n---\n".join(
        f"Passage {index}\nSource: {hit['source']}\nText: {hit['text']}\n"
        for index, hit in enumerate(hits, start=1)
    )
    return f"Quiz context for '{book}' — topic: {topic}\n\n{passages}"


@mcp.prompt()
def study_book(book: str, goal: str) -> str:
    return f"""
    Create a study plan for {book}.
    The student's goal is:
    {goal}

    Use the available resources to identify relevant chapters.
    """


@mcp.prompt()
def summarize_book(book: str) -> str:
    return f"""
    Summarize the book {book}.

    Use the available book resources and search tools.
    Focus on:
    - main concepts
    - prerequisites
    - practical applications
    - difficulty
    """


@mcp.resource("book://{book_name}", mime_type="application/pdf")
def get_book(book_name: str) -> bytes:
    """Return the raw PDF for a book in BOOKS_DIR."""
    return resolve_book_path(book_name).read_bytes()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
