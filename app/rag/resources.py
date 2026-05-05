from __future__ import annotations

import asyncio
import os
from typing import Any, Optional


_embeddings: Any | None = None
_embeddings_lock = asyncio.Lock()

_vector_stores: dict[str, object] = {}
_vector_store_locks: dict[str, asyncio.Lock] = {}

DEFAULT_STORE_KEY = "__default__"


def planning_collection_name() -> str:
    return os.getenv(
        "PGVECTOR_COLLECTION_PLANNING",
        "planning_documents__multilingual_e5_base_ghr1__planning_hierarchical_parent_context",
    )


def _store_key(collection_name: Optional[str]) -> str:
    return collection_name or DEFAULT_STORE_KEY


async def get_embeddings() -> Any:
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    async with _embeddings_lock:
        if _embeddings is None:
            from .embedder import build_embeddings

            _embeddings = build_embeddings()
        return _embeddings


async def initialize_vector_store(collection_name: Optional[str] = None):
    key = _store_key(collection_name)
    existing = _vector_stores.get(key)
    if existing is not None:
        return existing

    lock = _vector_store_locks.setdefault(key, asyncio.Lock())
    async with lock:
        existing = _vector_stores.get(key)
        if existing is not None:
            return existing

        embeddings = await get_embeddings()
        from .retriever import build_pgvector_store

        vector_store = build_pgvector_store(embeddings, collection_name=collection_name)
        await vector_store.__apost_init__()
        _vector_stores[key] = vector_store
        return vector_store


async def initialize_listing_vector_store():
    return await initialize_vector_store()


async def initialize_planning_vector_store():
    return await initialize_vector_store(planning_collection_name())


def get_initialized_vector_store(collection_name: Optional[str] = None):
    key = _store_key(collection_name)
    vector_store = _vector_stores.get(key)
    if vector_store is None:
        raise RuntimeError("Vector store not initialized. Await initialize_vector_store() first.")
    return vector_store


def get_initialized_listing_vector_store():
    return get_initialized_vector_store()


def get_initialized_planning_vector_store():
    return get_initialized_vector_store(planning_collection_name())
