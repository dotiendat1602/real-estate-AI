from __future__ import annotations

import os
from typing import Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from app.rag.embedder import build_embeddings
from app.rag.retriever import build_pgvector_store, build_retriever
from app.rag.chain import RagChain

router = APIRouter()

_embeddings = build_embeddings()
_vs = build_pgvector_store(_embeddings)

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
    message: str = Field(min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    topK: int = Field(default=int(os.getenv("TOP_K_DEFAULT", "12")), ge=1, le=50)

class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    retriever = build_retriever(_vs, k=req.topK, filters=req.filters)

    llm = build_llm()
    chain = RagChain(llm=llm, retriever=retriever)

    result = await chain.run(req.message)
    return ChatResponse(answer=result.answer, citations=result.citations)
