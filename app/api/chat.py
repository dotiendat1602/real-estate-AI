from __future__ import annotations

import os
from typing import Any, Optional
import re
import unicodedata
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document

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
_planning_vs = None

_PLANNING_KEYWORDS = [
    "quy hoach",
    "quy hoạch",
    "ke hoach su dung dat",
    "kế hoạch sử dụng đất",
    "khsd",
    "thong tin quy hoach",
    "thông tin quy hoạch",
    "loai dat",
    "loại đất",
    "quy hoach do thi",
    "quy hoạch đô thị",
]

_HANOI_DISTRICT_ALIASES: dict[str, list[str]] = {
    "Cầu Giấy": ["cau giay", "cầu giấy", "quan cau giay", "quận cầu giấy"],
    "Ba Đình": ["ba dinh", "ba đình", "quan ba dinh", "quận ba đình"],
    "Đống Đa": ["dong da", "đống đa", "quan dong da", "quận đống đa"],
    "Hoàn Kiếm": ["hoan kiem", "hoàn kiếm", "quan hoan kiem", "quận hoàn kiếm"],
    "Hai Bà Trưng": ["hai ba trung", "hai bà trưng", "quan hai ba trung", "quận hai bà trưng"],
    "Thanh Xuân": ["thanh xuan", "thanh xuân", "quan thanh xuan", "quận thanh xuân"],
    "Hoàng Mai": ["hoang mai", "hoàng mai", "quan hoang mai", "quận hoàng mai"],
    "Long Biên": ["long bien", "long biên", "quan long bien", "quận long biên"],
    "Tây Hồ": ["tay ho", "tây hồ", "quan tay ho", "quận tây hồ"],
    "Hà Đông": ["ha dong", "hà đông", "quan ha dong", "quận hà đông"],
    "Nam Từ Liêm": ["nam tu liem", "nam từ liêm", "quan nam tu liem", "quận nam từ liêm"],
    "Bắc Từ Liêm": ["bac tu liem", "bắc từ liêm", "quan bac tu liem", "quận bắc từ liêm"],
}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _normalize_nl(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = _strip_accents(lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _has_planning_intent(message: str) -> bool:
    normalized = _normalize_nl(message)
    return any(_normalize_nl(keyword) in normalized for keyword in _PLANNING_KEYWORDS)


def _extract_district_from_message(message: str) -> Optional[str]:
    normalized = _normalize_nl(message)
    if not normalized:
        return None

    for canonical, aliases in _HANOI_DISTRICT_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_nl(alias)
            if alias_norm and alias_norm in normalized:
                return canonical
    return None


def _extract_plan_year_from_message(message: str) -> Optional[int]:
    for match in re.findall(r"\b(20\d{2})\b", message or ""):
        try:
            year = int(match)
            if 2000 <= year <= 2100:
                return year
        except Exception:
            continue
    return None


def _district_candidates(district: Optional[str]) -> list[str]:
    if not district:
        return []

    values = [district.strip(), _strip_accents(district).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


async def _retrieve_planning_docs_for_nl_query(planning_vs, message: str, top_k: int) -> list[Document]:
    district = _extract_district_from_message(message)
    plan_year = _extract_plan_year_from_message(message)

    district_values = _district_candidates(district)

    base_candidates: list[dict[str, Any]] = []
    if district_values and plan_year is not None:
        base_candidates.append(
            {
                "documentScope": "planning",
                "district": {"$in": district_values},
                "planYear": plan_year,
            }
        )
    if district_values:
        base_candidates.append(
            {
                "documentScope": "planning",
                "district": {"$in": district_values},
            }
        )
    if plan_year is not None:
        base_candidates.append(
            {
                "documentScope": "planning",
                "planYear": plan_year,
            }
        )

    base_candidates.append({"documentScope": "planning"})

    for base_filter in base_candidates:
        planning_retriever = build_retriever(
            planning_vs,
            k=max(6, min(top_k, 12)),
            filters={"chunkTypes": ["text", "table"]},
            base_filter=base_filter,
        )
        planning_docs: list[Document] = await planning_retriever.ainvoke(message)
        if planning_docs:
            print(f"[PlanningNL] Retrieved {len(planning_docs)} docs with base_filter={base_filter}")
            return planning_docs

    print("[PlanningNL] No planning documents found for NL query")
    return []


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


def _build_planning_citations(docs: list[Document]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for d in docs:
        md = d.metadata or {}
        doc_id = str(md.get("planningDocumentId") or "")
        chunk_type = md.get("chunkType") or "text"
        global_chunk_index = md.get("globalChunkIndex")
        chunk_index = md.get("chunkIndex")
        page_number = md.get("pageNumber")
        key = f"{doc_id}:{chunk_type}:{global_chunk_index}:{chunk_index}:{page_number}"
        if key in seen_ids:
            continue
        seen_ids.add(key)

        out.append({
            "postId": None,
            "propertyId": md.get("propertyId"),
            "planningDocumentId": md.get("planningDocumentId"),
            "documentScope": md.get("documentScope"),
            "documentType": md.get("documentType"),
            "dossierCode": md.get("dossierCode"),
            "city": md.get("city"),
            "district": md.get("district"),
            "planYear": md.get("planYear"),
            "title": md.get("title"),
            "sourceUrl": md.get("sourceUrl"),
            "format": md.get("format"),
            "chunkType": chunk_type,
            "chunkIndex": chunk_index,
            "globalChunkIndex": global_chunk_index,
            "pageNumber": page_number,
            "lineStart": md.get("lineStart"),
            "lineEnd": md.get("lineEnd"),
            "sourceLocator": md.get("sourceLocator"),
            "chunker": md.get("chunker"),
            "snippet": (d.page_content or "")[:300],
        })

    return out

async def initialize_vector_store():
    """Khởi tạo async vector store - gọi từ startup event"""
    global _vs, _planning_vs
    if _vs is None:
        _vs = build_pgvector_store(_embeddings)
        # Trigger async init
        await _vs.__apost_init__()
    if _planning_vs is None:
        planning_collection = os.getenv("PGVECTOR_COLLECTION_PLANNING", "planning_documents")
        _planning_vs = build_pgvector_store(_embeddings, collection_name=planning_collection)
        await _planning_vs.__apost_init__()
    return _vs

def get_vector_store():
    """Lấy vector store đã được khởi tạo"""
    global _vs
    if _vs is None:
        raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
    return _vs


def get_planning_vector_store():
    global _planning_vs
    if _planning_vs is None:
        raise RuntimeError("Planning vector store not initialized. Call initialize_vector_store() first.")
    return _planning_vs

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
    planning_citations: list[dict[str, Any]] = []
    planning_docs: list[Document] = []
    planning_vs = get_planning_vector_store()

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

        planning_property_ids = [ctx.propertyId for ctx in req.planningContexts]
        planning_retriever = build_retriever(
            planning_vs,
            k=max(6, min(req.topK, 12)),
            filters={"chunkTypes": ["text", "table"]},
            base_filter={
                "documentScope": "planning",
                "propertyId": {"$in": planning_property_ids},
            },
        )
        planning_docs = await planning_retriever.ainvoke(req.message)
        if planning_docs:
            planning_text = "\n\n".join(d.page_content for d in planning_docs if d.page_content)
            extra_context = f"{extra_context}\n\n=== PLANNING VECTOR CONTEXT ===\n{planning_text}" if extra_context else planning_text
            planning_citations = _build_planning_citations(planning_docs)

    elif _has_planning_intent(req.message):
        district = _extract_district_from_message(req.message)
        plan_year = _extract_plan_year_from_message(req.message)
        planning_docs = await _retrieve_planning_docs_for_nl_query(planning_vs, req.message, req.topK)

        if planning_docs:
            planning_text = "\n\n".join(d.page_content for d in planning_docs if d.page_content)
            header = [
                "=== PLANNING VECTOR CONTEXT (AUTO FROM NATURAL LANGUAGE QUERY) ===",
                f"District hint: {district or 'N/A'}",
                f"Plan year hint: {plan_year if plan_year is not None else 'N/A'}",
            ]
            planning_context = "\n".join(header + [planning_text])
            extra_context = f"{extra_context}\n\n{planning_context}" if extra_context else planning_context
            planning_citations = _build_planning_citations(planning_docs)

    result = await chain.run(req.message, history=history, extra_context=extra_context)
    merged_citations = result.citations + planning_citations
    reranked_citations = _rerank_citations(req.message, merged_citations, req.planningContexts)
    
    await history_manager.add_message(session_id, "user", req.message)
    await history_manager.add_message(session_id, "assistant", result.answer)
    
    return ChatResponse(
        sessionId=session_id,
        answer=result.answer,
        citations=reranked_citations,
        extractedFilters=filters
    )
