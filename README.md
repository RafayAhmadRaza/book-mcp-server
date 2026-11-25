# Book MCP Server

An MCP server that lets an LLM search your personal library locally. It combines
two pipelines that used to live in separate scripts:

- **Extracted text** — EPUB, PDF text layers and plain text, chunked and stored
  in `books_collection`.
- **OCR** — scanned PDF pages rendered with PyMuPDF and read with Tesseract,
  stored in `ocr_books_collection`.

Both collections live in a local [Milvus Lite](https://milvus.io/docs/milvus_lite.md)
database and are queried with the default ONNX embedding function from
`milvus-model` (paraphrase-albert-small-v2, 768-d).

Nothing leaves your machine: the database is a single local file and the
embedding model runs on CPU through ONNX Runtime.

## Tools exposed to the model

| Tool | What it does |
| --- | --- |
| `list_books` | List titles in the extracted-text collection |
| `list_ocr_books` | List titles in the OCR collection |
| `search_book` | Semantic search across extracted text |
| `search_ocr_book` | Semantic search across OCR'd books |
| `search_book_in` | Search inside one extracted-text book |
| `search_book_in_ocr` | Search inside one OCR'd book |
| `ocr_book_page` | OCR a single page of a PDF from `BOOKS_DIR` |
| `get_quiz_context` | Pull passages from an OCR'd book for quiz writing |

It also ships two prompts (`study_book`, `summarize_book`) and a
`book://{book_name}` resource that returns the raw PDF.

## Layout

```
book_mcp/
  config.py    environment-driven settings
  store.py     Milvus client, collection creation, insert/search helpers
  ingest.py    text extraction, OCR and chunking
  server.py    MCP tools, prompts and resources
scripts/
  create_collections.py
  ingest_book.py
  search.py
tests/
```

## Setup

Python 3.10+ and Tesseract are required. On Arch:

```bash
sudo pacman -S tesseract tesseract-data-eng
```

On Debian/Ubuntu:

```bash
sudo apt install tesseract-ocr
```

Create an environment and install dependencies with `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ingesting books

Collections are created automatically on first insert. To create them up front:

```bash
python scripts/create_collections.py
```

Add an EPUB/PDF/TXT book:

```bash
python scripts/ingest_book.py \
  --book "The Hobbit" \
  --source ~/books/the-hobbit.epub
```

Add a scanned PDF with OCR (this is the step that needs Tesseract):

```bash
python scripts/ingest_book.py \
  --book "My Field Guide" \
  --source ~/books/field-guide.pdf \
  --ocr
```

`--ocr` writes to the OCR collection by default; `--collection books|ocr`
overrides that. Chunk size and overlap default to 1000/200 characters and can
be set per run or through `CHUNK_SIZE` / `CHUNK_OVERLAP`.

## Running the server

```bash
python server.py
```

The server speaks MCP over stdio. Example client configuration:

```json
{
  "mcpServers": {
    "books": {
      "command": "python",
      "args": ["/absolute/path/to/book-mcp-server/server.py"],
      "env": {
        "BOOKS_DIR": "/home/you/books",
        "BOOK_MCP_DB": "/home/you/.local/share/book-mcp/books.db"
      }
    }
  }
}
```

After installing the package, `book-mcp` is available as a console script.

## Configuration

Copy `.env.example` to `.env` or export the variables directly.

| Variable | Default | Meaning |
| --- | --- | --- |
| `BOOK_MCP_DB` | `data/books.db` | Milvus Lite database file |
| `BOOKS_DIR` | `books/` | PDFs available to `ocr_book_page` and `book://` |
| `BOOK_COLLECTION` | `books_collection` | Extracted-text collection name |
| `OCR_COLLECTION` | `ocr_books_collection` | OCR collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `EMBEDDING_DIM` | `768` | Embedding dimension |

## Querying without an MCP client

```bash
python scripts/search.py "What is the book about?" --collection ocr --book "My Field Guide"
```

## Tests

```bash
uv pip install pytest
pytest
```
