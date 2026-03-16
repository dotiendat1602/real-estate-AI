from __future__ import annotations

import os
from typing import Any, Optional
import re
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


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    parts = re.split(r"[^a-zA-Z0-9_\-\u00C0-\u1EF9]+", text.lower())
    return {p for p in parts if len(p) >= 3}


def _rerank_citations(message: str, citations: list[dict[str, Any]], planning_contexts: list[Any]) -> list[dict[str, Any]]:
    if not citations:
        return citations

    q_terms = _tokenize(message)
    planning_property_ids = {ctx.propertyId for ctx in planning_contexts}

    def score(c: dict[str, Any]) -> float:
        snippet = (c.get("snippet") or "")
        text_terms = _tokenize(snippet)
        overlap = len(q_terms.intersection(text_terms))

        semantic_score = float(c.get("score") or 0.0)
        planning_bonus = 2.0 if c.get("propertyId") in planning_property_ids else 0.0
        return overlap * 1.5 + semantic_score + planning_bonus

    return sorted(citations, key=score, reverse=True)

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
    class PlanningContext(BaseModel):
        propertyId: int
        planningStatus: str
        riskLevel: Optional[str] = None
        landUseCurrent: Optional[str] = None
        landUsePlanned: Optional[str] = None
        dossierCode: Optional[str] = None
        dossierName: Optional[str] = None
        checkedAt: Optional[str] = None
        reportSummaries: list[dict[str, Any]] = Field(default_factory=list)

    userId: Optional[int] = None
    sessionId: Optional[int] = None
    message: str = Field(min_length=1)
    topK: int = Field(default=int(os.getenv("TOP_K_DEFAULT", "12")), ge=1, le=50)
    planningContexts: list[PlanningContext] = Field(default_factory=list)

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

    extra_context = ""
    if req.planningContexts:
        lines: list[str] = ["=== PLANNING REPORT CONTEXT (BACKEND STORED) ==="]
        for ctx in req.planningContexts:
            docs = ctx.reportSummaries or []
            doc_lines = []
            for item in docs[:8]:
                title = item.get("title") or "Tai lieu quy hoach"
                fmt = item.get("format") or "unknown"
                doc_lines.append(f"- {title} ({fmt})")

            docs_text = "\n".join(doc_lines) if doc_lines else "- Khong co tai lieu"
            lines.append(
                "\n".join([
                    f"Property #{ctx.propertyId}",
                    f"Planning status: {ctx.planningStatus}",
                    f"Risk level: {ctx.riskLevel or 'UNKNOWN'}",
                    f"Land use current: {ctx.landUseCurrent or 'N/A'}",
                    f"Land use planned: {ctx.landUsePlanned or 'N/A'}",
                    f"Dossier: {(ctx.dossierCode or 'N/A')} - {(ctx.dossierName or 'N/A')}",
                    f"Checked at: {ctx.checkedAt or 'N/A'}",
                    "Report summaries:",
                    docs_text,
                ])
            )

        extra_context = "\n\n".join(lines)

    result = await chain.run(req.message, history=history, extra_context=extra_context)
    reranked_citations = _rerank_citations(req.message, result.citations, req.planningContexts)
    
    await history_manager.add_message(session_id, "user", req.message)
    await history_manager.add_message(session_id, "assistant", result.answer)
    
    return ChatResponse(
        sessionId=session_id,
        answer=result.answer,
        citations=reranked_citations,
        extractedFilters=filters
    )
