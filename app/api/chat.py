from __future__ import annotations

import os
from typing import Any, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_openai import ChatOpenAI

from ..rag.embedder import build_embeddings
from ..rag.retriever import build_pgvector_store, build_retriever
from ..rag.filter_extractor import extract_filters_from_query
from ..rag.chain import RagChain
from ..rag.message_history import MessageHistoryManager
from ..db.pgvector import get_db

router = APIRouter()

_embeddings = build_embeddings()
_vs = None

async def initialize_vector_store():
    """Khởi tạo async vector store - gọi từ startup event"""
    global _vs
    if _vs is None:
        _vs = build_pgvector_store(_embeddings)
        # Trigger async init
        await _vs.__apost_init__()
    return _vs

def get_vector_store():
    """Lấy vector store đã được khởi tạo"""
    global _vs
    if _vs is None:
        raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
    return _vs

def build_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )

class ChatRequest(BaseModel):
    userId: Optional[int] = None
    sessionId: Optional[int] = None
    message: str = Field(min_length=1)
    topK: int = Field(default=int(os.getenv("TOP_K_DEFAULT", "12")), ge=1, le=50)

class ChatResponse(BaseModel):
    sessionId: int
    answer: str
    citations: list[dict[str, Any]]
    extractedFilters: dict[str, Any] = Field(default_factory=dict)

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    if not req.userId:
        raise ValueError("userId is required for chat history")
    
    history_manager = MessageHistoryManager(db)
    session_id = await history_manager.get_or_create_session(req.userId, req.sessionId)
    
    # Lấy lịch sử 6 tin nhắn gần nhất
    history = await history_manager.get_messages(session_id, limit=6)
    
    vs = get_vector_store()

    llm = build_llm()
    
    filters = await extract_filters_from_query(req.message, llm)
    print(f"Extracted filters: {filters}")

    retriever = build_retriever(vs, k=req.topK, filters=filters)
    print(f"Using topK={req.topK} for retrieval.")
    print(f"retriever: {retriever}")
    chain = RagChain(llm=llm, retriever=retriever)

    result = await chain.run(req.message, history=history)
    
    await history_manager.add_message(session_id, "user", req.message)
    await history_manager.add_message(session_id, "assistant", result.answer)
    
    return ChatResponse(
        sessionId=session_id,
        answer=result.answer,
        citations=result.citations,
        extractedFilters=filters
    )
