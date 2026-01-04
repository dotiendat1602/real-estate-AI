from __future__ import annotations

from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from app.rag.embedder import build_embeddings
from app.rag.retriever import build_pgvector_store
from app.utils.chunking import build_splitter

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

@router.post("/ingest/posts")
async def ingest_posts(req: IngestRequest):
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
