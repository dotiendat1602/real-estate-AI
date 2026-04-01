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
    "quy hoach su dung dat",
    "quy hoạch sử dụng đất",
    "ke hoach su dung dat",
    "kế hoạch sử dụng đất",
    "ke hoach",
    "kế hoạch",
    "khsd",
    "khsdd",
    "khsdđ",
    "thong tin quy hoach",
    "thông tin quy hoạch",
    "loai dat",
    "loại đất",
    "thua dat",
    "thửa đất",
    "to ban do",
    "tờ bản đồ",
    "muc dich su dung dat",
    "mục đích sử dụng đất",
    "quy hoach do thi",
    "quy hoạch đô thị",
]

_PLANNING_STRUCTURAL_TERMS = [
    "thua dat",
    "to ban do",
    "quy hoach",
    "ke hoach su dung dat",
    "muc dich su dung dat",
    "loai dat",
    "dat o",
    "hanh lang",
    "chi gioi",
]

_PLANNING_DISTRICT_STOPWORDS = {
    "ke",
    "hoach",
    "su",
    "dung",
    "dat",
    "nam",
    "quan",
    "huyen",
    "thi",
    "xa",
    "thanh",
    "pho",
}

_PLANNING_FACT_MARKERS = [
    "ma ho so",
    "dossier",
    "dossier code",
    "dossiercode",
    "ap dung cho nam nao",
    "nam nao",
    "thuoc khu vuc nao",
    "khu vuc nao",
    "quan nao",
    "huyen nao",
    "la gi",
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
    if any(_normalize_nl(keyword) in normalized for keyword in _PLANNING_KEYWORDS):
        return True

    has_structural_term = any(term in normalized for term in _PLANNING_STRUCTURAL_TERMS)
    if not has_structural_term:
        return False

    # If user mentions land/planning structural terms plus either district hint
    # or plan year, treat it as planning intent even without exact keyword phrases.
    has_district_hint = _extract_district_from_message(message) is not None
    has_plan_year = _extract_plan_year_from_message(message) is not None
    return has_district_hint or has_plan_year


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


def _extract_district_from_history(history_messages: Optional[list[dict[str, Any]]]) -> Optional[str]:
    if not history_messages:
        return None

    for item in reversed(history_messages):
        role = str(item.get("role") or "").strip().lower()
        if role != "user":
            continue
        district = _extract_district_from_message(str(item.get("content") or ""))
        if district:
            return district

    for item in reversed(history_messages):
        district = _extract_district_from_message(str(item.get("content") or ""))
        if district:
            return district
    return None


def _extract_plan_year_from_history(history_messages: Optional[list[dict[str, Any]]]) -> Optional[int]:
    if not history_messages:
        return None

    for item in reversed(history_messages):
        role = str(item.get("role") or "").strip().lower()
        if role != "user":
            continue
        year = _extract_plan_year_from_message(str(item.get("content") or ""))
        if year is not None:
            return year

    for item in reversed(history_messages):
        year = _extract_plan_year_from_message(str(item.get("content") or ""))
        if year is not None:
            return year
    return None


def _is_planning_fact_query(message: str) -> bool:
    normalized = _normalize_nl(message)
    if not normalized:
        return False
    if not _has_planning_intent(message):
        return False
    return any(marker in normalized for marker in _PLANNING_FACT_MARKERS)


def _district_tokens(district: Optional[str]) -> list[str]:
    if not district:
        return []

    canonical = district.strip()
    variants: list[str] = [canonical]
    variants.extend(_HANOI_DISTRICT_ALIASES.get(canonical, []))

    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = _normalize_nl(item)
        for token in normalized.split():
            if len(token) < 2 or token in _PLANNING_DISTRICT_STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def _planning_doc_haystack(doc: Document) -> str:
    md = doc.metadata or {}
    parts = [
        str(md.get("district") or ""),
        str(md.get("title") or ""),
        str(md.get("dossierCode") or ""),
        str(md.get("city") or ""),
        (doc.page_content or "")[:450],
    ]
    return _normalize_nl(" ".join(parts))


def _doc_matches_district(doc: Document, district: Optional[str]) -> bool:
    if not district:
        return True

    tokens = _district_tokens(district)
    if not tokens:
        return True

    haystack = _planning_doc_haystack(doc)
    return all(token in haystack for token in tokens)


def _doc_matches_plan_year(doc: Document, plan_year: Optional[int]) -> bool:
    if plan_year is None:
        return True

    md = doc.metadata or {}
    raw_year = md.get("planYear")
    if raw_year is not None:
        try:
            return int(raw_year) == int(plan_year)
        except Exception:
            pass

    return str(plan_year) in _planning_doc_haystack(doc)


def _planning_doc_score(doc: Document, message: str, district: Optional[str], plan_year: Optional[int]) -> float:
    md = doc.metadata or {}
    msg_norm = _normalize_nl(message)
    score = 0.0

    if _doc_matches_district(doc, district):
        score += 120.0
    elif district:
        score -= 80.0

    if _doc_matches_plan_year(doc, plan_year):
        score += 24.0
    elif plan_year is not None:
        score -= 16.0

    chunk_type = str(md.get("chunkType") or "").lower().strip()
    if chunk_type == "text":
        score += 2.0

    if "ma ho so" in msg_norm or "dossier" in msg_norm:
        if md.get("dossierCode"):
            score += 14.0

    if "nam nao" in msg_norm or "ap dung" in msg_norm:
        if str(plan_year or "") and str(plan_year) in _planning_doc_haystack(doc):
            score += 8.0

    q_terms = _tokenize(message)
    t_terms = _tokenize(
        " ".join(
            [
                str(md.get("title") or ""),
                str(md.get("district") or ""),
                (doc.page_content or "")[:240],
            ]
        )
    )
    score += min(len(q_terms.intersection(t_terms)), 10) * 0.4
    return score


def _district_code_fragment(district: Optional[str]) -> str:
    if not district:
        return ""
    ascii_text = _strip_accents(district)
    parts = re.split(r"[^A-Za-z0-9]+", ascii_text)
    cleaned = [p for p in parts if p]
    return "".join(token[:1].upper() + token[1:] for token in cleaned)


def _planning_query_candidates(message: str, district: Optional[str], plan_year: Optional[int]) -> list[str]:
    candidates: list[str] = [message]

    if district and plan_year is not None:
        candidates.append(f"Kế hoạch sử dụng đất năm {plan_year} quận {district}")
        candidates.append(f"Quyết định về việc phê duyệt Kế hoạch sử dụng đất năm {plan_year} quận {district}")
    elif district:
        candidates.append(f"Kế hoạch sử dụng đất quận {district}")

    if district and plan_year is not None:
        district_code = _district_code_fragment(district)
        if district_code:
            candidates.append(f"HN-{district_code}-KH{plan_year}")

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_nl(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(candidate)
    return out


def _select_relevant_content_lines(
    text: str,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    max_lines: int,
) -> list[str]:
    if max_lines <= 0:
        return []

    if not text:
        return []

    q_terms = _tokenize(message)
    district_tokens = set(_district_tokens(district))
    year_token = str(plan_year) if plan_year is not None else ""

    scored: list[tuple[float, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue

        norm = _normalize_nl(line)
        if not norm:
            continue

        line_terms = _tokenize(line)
        score = 0.0
        score += min(len(q_terms.intersection(line_terms)), 8) * 1.6

        if district_tokens and all(token in norm for token in district_tokens):
            score += 5.0

        if year_token and year_token in norm:
            score += 3.5

        if score > 0:
            scored.append((score, line))

    scored.sort(key=lambda item: item[0], reverse=True)

    out: list[str] = []
    seen_norm: set[str] = set()
    for _, line in scored:
        normalized = _normalize_nl(line)
        if normalized in seen_norm:
            continue
        seen_norm.add(normalized)
        out.append(line)
        if len(out) >= max_lines:
            break
    return out


def _compact_planning_doc(
    doc: Document,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    fact_query: bool,
) -> Document:
    md = doc.metadata or {}

    lines: list[str] = ["[document_scope=planning]"]
    if md.get("city"):
        lines.append(f"[city={md.get('city')}]")
    if md.get("district"):
        lines.append(f"[district={md.get('district')}]")
    if md.get("planYear") is not None:
        lines.append(f"[plan_year={md.get('planYear')}]")
    if md.get("dossierCode"):
        lines.append(f"[dossier_code={md.get('dossierCode')}]")
    if md.get("title"):
        lines.append(f"[title={md.get('title')}]")

    excerpt_max_lines = 0 if fact_query else 4
    excerpt_lines = _select_relevant_content_lines(
        doc.page_content or "",
        message,
        district,
        plan_year,
        max_lines=excerpt_max_lines,
    )
    lines.extend(excerpt_lines)

    compact_content = "\n".join(lines)
    return Document(page_content=compact_content, metadata=md)


def _compact_planning_docs(
    docs: list[Document],
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    fact_query: bool,
) -> list[Document]:
    out: list[Document] = []
    seen: set[str] = set()

    for doc in docs:
        compacted = _compact_planning_doc(doc, message, district, plan_year, fact_query=fact_query)
        key = _normalize_nl(compacted.page_content)
        if key in seen:
            continue
        seen.add(key)
        out.append(compacted)

    return out


def _dedupe_planning_docs(docs: list[Document]) -> list[Document]:
    out: list[Document] = []
    seen: set[str] = set()

    for doc in docs:
        md = doc.metadata or {}
        key = "|".join(
            [
                str(md.get("planningDocumentId") or ""),
                str(md.get("chunkType") or ""),
                str(md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex") or ""),
                str(md.get("pageNumber") or ""),
                (doc.page_content or "")[:80],
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)

    return out


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


async def _retrieve_planning_docs_for_nl_query(
    planning_vs,
    message: str,
    top_k: int,
    history_messages: Optional[list[dict[str, Any]]] = None,
) -> list[Document]:
    district = _extract_district_from_message(message) or _extract_district_from_history(history_messages)
    plan_year = _extract_plan_year_from_message(message)
    if plan_year is None:
        plan_year = _extract_plan_year_from_history(history_messages)

    fact_query = _is_planning_fact_query(message)
    final_k = max(2, min(top_k, 4 if fact_query else 8))
    probe_k = max(12, min(36, final_k * 4))
    chunk_types = ["text"] if fact_query else ["text", "table"]

    query_candidates = _planning_query_candidates(message, district, plan_year)

    base_candidates: list[dict[str, Any]] = []
    if plan_year is not None:
        base_candidates.append({"documentScope": "planning", "planYear": plan_year})
    base_candidates.append({"documentScope": "planning"})

    best_fallback: list[Document] = []

    for base_filter in base_candidates:
        planning_retriever = build_retriever(
            planning_vs,
            k=probe_k,
            filters={"chunkTypes": chunk_types},
            base_filter=base_filter,
        )

        docs: list[Document] = []
        for query_text in query_candidates:
            pulled = await planning_retriever.ainvoke(query_text)
            if pulled:
                docs.extend(pulled)

        if not docs:
            continue

        deduped = _dedupe_planning_docs(docs)
        ranked = sorted(
            deduped,
            key=lambda d: _planning_doc_score(d, message, district, plan_year),
            reverse=True,
        )

        strict = [d for d in ranked if _doc_matches_district(d, district) and _doc_matches_plan_year(d, plan_year)]
        if strict:
            selected = strict[:final_k]
            selected = _compact_planning_docs(
                selected,
                message,
                district,
                plan_year,
                fact_query=fact_query,
            )
            print(
                "[PlanningNL] Retrieved "
                f"{len(selected)} strict docs (district={district or 'N/A'}, year={plan_year or 'N/A'}) "
                f"with base_filter={base_filter}"
            )
            return selected

        relaxed = [d for d in ranked if _doc_matches_district(d, district)]
        if relaxed:
            selected = relaxed[:final_k]
            selected = _compact_planning_docs(
                selected,
                message,
                district,
                plan_year,
                fact_query=fact_query,
            )
            print(
                "[PlanningNL] Retrieved "
                f"{len(selected)} district-matched docs (district={district or 'N/A'}, year={plan_year or 'N/A'}) "
                f"with base_filter={base_filter}"
            )
            return selected

        if not best_fallback:
            best_fallback = ranked[:final_k]

    if best_fallback:
        compact_fallback = _compact_planning_docs(
            best_fallback,
            message,
            district,
            plan_year,
            fact_query=fact_query,
        )
        print(
            "[PlanningNL] Falling back to broad planning docs "
            f"(district={district or 'N/A'}, year={plan_year or 'N/A'})"
        )
        return compact_fallback

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
    
    llm = build_llm()
    
    filters = await extract_filters_from_query(req.message, llm)
    print(f"Extracted filters: {filters}")

    extra_context = ""
    planning_citations: list[dict[str, Any]] = []
    planning_docs: list[Document] = []
    use_planning_mode = bool(req.planningContexts) or _has_planning_intent(req.message)

    if use_planning_mode:
        print("Planning mode enabled for retrieval.")
        planning_vs = get_planning_vector_store()
        planning_retriever = build_retriever(
            planning_vs,
            k=max(6, min(req.topK, 12)),
            filters={"chunkTypes": ["text", "table"]},
            base_filter={"documentScope": "planning"},
        )
        retriever = planning_retriever
    else:
        vs = get_vector_store()
        retriever = build_retriever(vs, k=req.topK, filters=filters)

    print(f"Using topK={req.topK} for retrieval.")
    print(f"retriever: {retriever}")
    chain = RagChain(llm=llm, retriever=retriever)

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

    elif use_planning_mode:
        district = _extract_district_from_message(req.message)
        plan_year = _extract_plan_year_from_message(req.message)
        planning_docs = await _retrieve_planning_docs_for_nl_query(
            planning_vs,
            req.message,
            req.topK,
            history_messages=history,
        )

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
