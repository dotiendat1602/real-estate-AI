from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from sqlalchemy import text

from ..rag.embedder import build_embeddings
from ..rag.retriever import build_pgvector_store
from ..utils.chunking import build_splitter
from ..db.pgvector import AsyncSessionLocal

router = APIRouter()

_embeddings = build_embeddings()
_vs = None
_splitter = build_splitter()

async def initialize_vector_store():
    """Khởi tạo async vector store"""
    global _vs
    if _vs is None:
        _vs = build_pgvector_store(_embeddings)
        await _vs.__apost_init__()
    return _vs

def get_vector_store():
    global _vs
    if _vs is None:
        raise RuntimeError("Vector store not initialized.")
    return _vs

class IngestPost(BaseModel):
    postId: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class IngestRequest(BaseModel):
    posts: list[IngestPost] = Field(default_factory=list)

class UpdatePostRequest(BaseModel):
    postId: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

@router.get("/ingest/posts")
async def list_ingested_posts(limit: int = 50, offset: int = 0):
    """
    Liệt kê post đã ingest:
    - postId
    - số chunks
    - embedding_id (chunkIndex = 0)
    - sample metadata
    """
    async with AsyncSessionLocal() as session:
        q = text("""
            WITH agg AS (
              SELECT
                (cmetadata->>'postId')::int AS post_id,
                COUNT(*) AS chunks
              FROM langchain_pg_embedding
              WHERE cmetadata ? 'postId'
              GROUP BY (cmetadata->>'postId')::int
            ),
            sample AS (
              SELECT DISTINCT ON ((cmetadata->>'postId')::int)
                id AS embedding_id,
                (cmetadata->>'postId')::int AS post_id,
                cmetadata AS sample_metadata
              FROM langchain_pg_embedding
              WHERE cmetadata->>'chunkIndex' = '0'
              ORDER BY (cmetadata->>'postId')::int, id
            )
            SELECT
              agg.post_id,
              agg.chunks,
              sample.embedding_id,
              sample.sample_metadata
            FROM agg
            LEFT JOIN sample ON sample.post_id = agg.post_id
            ORDER BY agg.post_id DESC
            LIMIT :limit OFFSET :offset;
        """)
        res = await session.execute(q, {"limit": limit, "offset": offset})
        rows = [dict(r._mapping) for r in res.fetchall()]

    return {"ok": True, "items": rows, "limit": limit, "offset": offset}

@router.get("/ingest/posts/{post_id}")
async def get_ingested_post(post_id: int, limit: int = 50, offset: int = 0):
    """
    Xem tất cả chunks embeddings của 1 post.
    Trả chunkIndex + metadata. (Nếu có cột document/text thì bạn bật thêm ở SELECT)
    """
    async with AsyncSessionLocal() as session:
        q = text("""
            SELECT
              id,
              (cmetadata->>'chunkIndex')::int AS chunk_index,
              cmetadata
              -- , document  -- bật nếu bảng có cột này
            FROM langchain_pg_embedding
            WHERE cmetadata->>'postId' = :post_id
            ORDER BY chunk_index ASC
            LIMIT :limit OFFSET :offset;
        """)
        res = await session.execute(q, {"post_id": str(post_id), "limit": limit, "offset": offset})
        rows = [dict(r._mapping) for r in res.fetchall()]

        q2 = text("""
            SELECT COUNT(*) AS total
            FROM langchain_pg_embedding
            WHERE cmetadata->>'postId' = :post_id;
        """)
        total = (await session.execute(q2, {"post_id": str(post_id)})).scalar_one()

    return {"ok": True, "postId": post_id, "totalChunks": total, "items": rows}

@router.post("/ingest/posts")
async def ingest_posts(req: IngestRequest):
    """Ingest multiple posts - thêm embeddings mới"""
    if not req.posts:
        return {"ok": True, "ingestedChunks": 0}

    vs = get_vector_store()
    
    docs: list[Document] = []
    for p in req.posts:
        text = (p.content or "").strip()
        if not text:
            continue

        chunks = _splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            md = dict(p.metadata or {})
            md.update({
                "postId": p.postId,
                "chunkIndex": idx,
            })
            docs.append(Document(page_content=chunk, metadata=md))

    if not docs:
        return {"ok": True, "ingestedChunks": 0}

    ids = await vs.aadd_documents(docs)
    return {"ok": True, "ingestedChunks": len(ids)}

@router.put("/ingest/posts/{post_id}")
async def update_post_embeddings(post_id: int, req: UpdatePostRequest):
    """
    Update embeddings của một post.
    Workflow: Xóa embeddings cũ → Tạo embeddings mới
    """
    if req.postId != post_id:
        raise HTTPException(
            status_code=400, 
            detail=f"postId mismatch: URL={post_id}, body={req.postId}"
        )
    
    # 1. Xóa embeddings cũ
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'postId' = :post_id"),
            {"post_id": str(post_id)}
        )
        deleted_count = result.rowcount
        await session.commit()
    
    # 2. Tạo embeddings mới
    text = (req.content or "").strip()
    if not text:
        return {
            "ok": True, 
            "postId": post_id,
            "deletedChunks": deleted_count,
            "ingestedChunks": 0
        }
    
    vs = get_vector_store()
    chunks = _splitter.split_text(text)
    
    docs: list[Document] = []
    for idx, chunk in enumerate(chunks):
        md = dict(req.metadata or {})
        md.update({
            "postId": post_id,
            "chunkIndex": idx,
        })
        docs.append(Document(page_content=chunk, metadata=md))
    
    ids = await vs.aadd_documents(docs)
    
    return {
        "ok": True,
        "postId": post_id,
        "deletedChunks": deleted_count,
        "ingestedChunks": len(ids)
    }

@router.delete("/ingest/posts/{post_id}")
async def delete_post_embeddings(post_id: int):
    """
    Xóa tất cả embeddings của một post.
    Được gọi khi post bị xóa (soft delete) hoặc reject.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'postId' = :post_id"),
            {"post_id": str(post_id)}
        )
        deleted_count = result.rowcount
        await session.commit()
    
    return {
        "ok": True,
        "postId": post_id,
        "deletedChunks": deleted_count
    }
