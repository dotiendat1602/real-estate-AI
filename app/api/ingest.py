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
_vs = build_pgvector_store(_embeddings)
_splitter = build_splitter()

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

    # add_documents sync; chạy trong threadpool nếu bạn muốn async “đúng nghĩa”
    ids = _vs.add_documents(docs)
    return {"ok": True, "ingestedChunks": len(ids)}
