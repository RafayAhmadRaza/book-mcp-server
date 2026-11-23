"""Text extraction, OCR and chunking used to fill the two collections."""

from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pypdf import PdfReader

from . import config


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def read_text_file(path: Path) -> tuple[str, str]:
    """Read a plain-text book. Returns ``(text, source_label)``."""
    return path.read_text(encoding="utf-8", errors="replace"), "text"


def read_pdf_text(path: Path) -> tuple[str, str]:
    """Read the embedded text layer of a PDF (no OCR)."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages), "pdf-text"


def read_epub(path: Path) -> tuple[str, str]:
    """Read an EPUB through ebooklib/BeautifulSoup."""
    from bs4 import BeautifulSoup
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(path))
    chapters: list[str] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        text = soup.get_text(" ", strip=True)
        if text:
            chapters.append(text)
    return "\n\n".join(chapters), "epub"


def read_book(path: Path, ocr: bool = False) -> tuple[str, str]:
    """Extract text from .txt/.md/.pdf/.epub, optionally OCR-ing a PDF."""
    suffix = path.suffix.lower()

    if ocr:
        if suffix != ".pdf":
            raise ValueError("OCR mode only supports PDF files.")
        return ocr_pdf(path), "ocr"

    if suffix == ".pdf":
        return read_pdf_text(path)
    if suffix == ".epub":
        return read_epub(path)
    if suffix in {".txt", ".md", ".markdown", ""}:
        return read_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix or '(none)'}")


def ocr_pdf(path: Path, dpi: int = 200) -> str:
    """OCR every page of a PDF with PyMuPDF + Tesseract."""
    import pytesseract

    document = fitz.open(str(path))
    pages: list[str] = []
    try:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(dpi=dpi)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image)
            pages.append(f"\n===== PAGE {page_number} =====\n\n{text}")
    finally:
        document.close()

    return "\n".join(pages)


def ocr_pdf_page(path: Path, page_number: int, dpi: int = 200) -> str:
    """OCR a single, 1-based page of a PDF."""
    import pytesseract

    document = fitz.open(str(path))
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError(f"Invalid page number. The book has {len(document)} pages.")

        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image)
    finally:
        document.close()


def resolve_book_path(name: str) -> Path:
    """Resolve a book name inside BOOKS_DIR, rejecting path traversal."""
    candidate = (config.BOOKS_DIR / name).resolve()
    books_dir = config.BOOKS_DIR.resolve()

    if books_dir not in candidate.parents and candidate != books_dir:
        raise ValueError("Book must live inside the configured BOOKS_DIR.")

    return candidate
