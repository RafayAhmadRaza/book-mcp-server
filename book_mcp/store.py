"""Milvus storage helpers shared by the MCP server and the ingest CLI."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from milvus_model import DefaultEmbeddingFunction
from pymilvus import DataType, MilvusClient

from . import config


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(str(config.DB_PATH))


@lru_cache(maxsize=1)
def get_embedding_fn() -> DefaultEmbeddingFunction:
    # Downloads the ONNX model on first use, not at import time.
    return DefaultEmbeddingFunction()


def escape_filter_value(value: str) -> str:
    """Escape a value used inside a Milvus expression string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def has_collection(name: str) -> bool:
    return get_client().has_collection(name)


def ensure_collection(name: str, dim: int | None = None) -> None:
    """Create the collection if it does not exist yet."""
    client = get_client()
    if client.has_collection(name):
        return

    dim = dim or config.EMBEDDING_DIM

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("book", DataType.VARCHAR, max_length=1000)
    schema.add_field("source", DataType.VARCHAR, max_length=1000)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )


def _next_id(collection: str) -> int:
    stats = get_client().get_collection_stats(collection)
    return int(stats.get("row_count", 0))


def insert_chunks(
    collection: str,
    chunks: Iterable[str],
    book: str,
    source: str,
) -> int:
    """Embed and insert text chunks, returning the number of rows written."""
    chunks = [chunk for chunk in chunks if chunk.strip()]
    if not chunks:
        return 0

    ensure_collection(collection)

    vectors = get_embedding_fn().encode_documents(chunks)
    start_id = _next_id(collection)

    rows = [
        {
            "id": start_id + offset,
            "vector": vector,
            "text": chunk,
            "book": book,
            "source": source,
        }
        for offset, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    get_client().insert(collection_name=collection, data=rows)
    return len(rows)


def search(
    collection: str,
    query: str,
    limit: int = 5,
    book: str | None = None,
) -> list[dict]:
    """Run a semantic search and return the raw entity dictionaries."""
    client = get_client()
    if not client.has_collection(collection):
        raise ValueError(
            f"Collection '{collection}' does not exist yet. "
            "Run the ingest script first."
        )

    client.load_collection(collection)

    results = client.search(
        collection_name=collection,
        data=get_embedding_fn().encode_queries([query]),
        limit=limit,
        filter=f'book == "{escape_filter_value(book)}"' if book else "",
        output_fields=["text", "book", "source"],
    )

    if not results or not results[0]:
        return []

    return [
        {
            "text": hit["entity"]["text"],
            "book": hit["entity"]["book"],
            "source": hit["entity"]["source"],
        }
        for hit in results[0]
    ]


def list_books(collection: str) -> list[str]:
    client = get_client()
    if not client.has_collection(collection):
        return []

    client.load_collection(collection)

    rows = client.query(
        collection_name=collection,
        filter="",
        output_fields=["book"],
        limit=16384,
    )

    return sorted({row["book"] for row in rows if row.get("book")})
