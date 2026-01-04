from __future__ import annotations

import os
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever

def build_pgvector_store(embeddings) -> PGVector:
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing")

    collection_name = os.getenv("PGVECTOR_COLLECTION", "post_embeddings")

    return PGVector(
        connection=db_url,
        embeddings=embeddings,
        collection_name=collection_name,
        use_jsonb=True,
        async_mode=True,
        create_extension=False,
    )

def build_retriever(vs: PGVector, k: int = 12, filters: dict | None = None) -> VectorStoreRetriever:
    # filters map vào metadata JSONB
    # ví dụ: {"city":"Hà Nội","district":"Cầu Giấy","type":"apartment"}
    search_kwargs = {"k": k}
    if filters:
        search_kwargs["filter"] = filters
    return vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
