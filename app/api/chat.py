from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Optional
from fastapi import APIRouter, Depends
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..rag.llm import build_llm
from ..rag.citation_utils import (
    dedupe_citations as _dedupe_citations,
    rerank_citations as _rerank_citations,
)
from ..rag.listing_fallback import ListingFallbackRetriever
from ..rag.resources import (
    initialize_listing_vector_store,
    initialize_planning_vector_store,
)
from ..rag.retriever import build_retriever
from ..rag.filter_extractor import extract_filters_from_query_with_usage
from ..rag.llm_usage import sum_token_usage
from ..rag.planning_retrieval import (
    _extract_district_from_message,
    _extract_plan_year_from_message,
    _has_planning_intent,
    _retrieve_planning_docs_for_nl_query,
)
from ..rag.message_history import MessageHistoryManager
from ..rag.static_retriever import StaticDocumentsRetriever as _StaticDocumentsRetriever
from ..utils.text import normalize_vietnamese_search_text as _normalize_nl
from ..db.pgvector import get_db

router = APIRouter()
_logger = logging.getLogger(__name__)

_POST_ID_PATTERNS = (
    re.compile(r"/posts/(\d+)", re.IGNORECASE),
    re.compile(r"\bLISTING_ID:\s*(\d+)", re.IGNORECASE),
    re.compile(
        r"\b(?:căn|can|bất\s*động\s*sản|bat\s*dong\s*san|bđs|bds|post|tin)\s*#?\s*(\d{2,})\b",
        re.IGNORECASE,
    ),
)

_FOLLOW_UP_ANCHOR_TERMS = (
    "căn này",
    "can nay",
    "can nha nay",
    "nha nay",
    "ngoi nha nay",
    "can ho nay",
    "mat bang nay",
    "lo dat nay",
    "bđs này",
    "bds nay",
    "bất động sản này",
    "bat dong san nay",
    "tin nay",
    "chủ nhà",
    "chu nha",
    "liên lạc",
    "lien lac",
    "liên hệ",
    "lien he",
    "gần",
    "gan",
    "quy hoạch",
    "quy hoach",
    "hay có",
    "hay co",
)

_FOLLOW_UP_ANCHOR_PATTERN = re.compile(
    r"\b(?:can(?:\s+nha|\s+ho)?|nha|ngoi\s+nha|bat\s+dong\s+san|bds|tin|mat\s+bang|lo\s+dat)\s+"
    r"(?:nay|do|kia)\b",
    re.IGNORECASE,
)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
            elif item is not None:
                parts.append(str(item))
        return " ".join(parts)
    return str(content or "")


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    if role:
        return str(role)
    name = message.__class__.__name__.lower()
    if "human" in name:
        return "human"
    if "ai" in name or "assistant" in name:
        return "ai"
    return ""


def _extract_post_id_from_text(text_value: str) -> int | None:
    for pattern in _POST_ID_PATTERNS:
        match = pattern.search(text_value or "")
        if not match:
            continue
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            continue

    normalized = _normalize_nl(text_value or "")
    normalized_match = re.search(
        r"\b(?:can(?:\s+nha|\s+ho)?|nha|bat\s+dong\s+san|bds|post|tin)\s*#?\s*(\d{2,})\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if normalized_match:
        try:
            return int(normalized_match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _extract_contextual_post_id(
    message: str,
    history: list[Any],
    fallback_post_id: int | None = None,
) -> int | None:
    explicit = _extract_post_id_from_text(message)
    if explicit is not None:
        return explicit

    normalized = _normalize_nl(message or "")
    has_follow_up_anchor = any(term in normalized for term in _FOLLOW_UP_ANCHOR_TERMS) or bool(
        _FOLLOW_UP_ANCHOR_PATTERN.search(normalized)
    )
    if not has_follow_up_anchor:
        return None

    for preferred_role in ("human", "ai", "assistant"):
        for history_message in reversed(history or []):
            role = _message_role(history_message)
            if preferred_role == "human" and role not in {"human", "user"}:
                continue
            if preferred_role in {"ai", "assistant"} and role not in {"ai", "assistant"}:
                continue
            post_id = _extract_post_id_from_text(_message_text(history_message))
            if post_id is not None:
                return post_id
    return fallback_post_id


def _clean_cell(value: Any) -> str:
    return str(value or "").strip()


async def _load_post_context_document(db: AsyncSession, post_id: int) -> Document | None:
    row = (
        await db.execute(
            text(
                """
                SELECT
                  post.id AS post_id,
                  p.id AS property_id,
                  post.post_type,
                  post.post_title,
                  p.description,
                  p.price,
                  p.area,
                  p."bedroomNumber" AS bedrooms,
                  p."toiletNumber" AS toilets,
                  p."floorNumber" AS floor_number,
                  p.lat,
                  p.lon,
                  p.location,
                  pr.name AS province,
                  d.name AS district,
                  w.name AS ward,
                  planning.status AS planning_status,
                  planning.risk_level,
                  planning.confidence_level,
                  planning.explanation,
                  planning.ky_quy_hoach,
                  planning.ma_ho_so,
                  planning.ten_ho_so,
                  planning.ten_loai_dat_ht,
                  planning.ten_loai_dat_qh
                FROM posts post
                JOIN properties p ON p.id = post.property_id
                LEFT JOIN provinces pr ON pr.id = p.province_id
                LEFT JOIN districts d ON d.id = p.district_id
                LEFT JOIN wards w ON w.id = p.ward_id
                LEFT JOIN LATERAL (
                  SELECT
                    pm.status,
                    pm.risk_level,
                    pm.confidence_level,
                    pm.explanation,
                    pcl.ky_quy_hoach,
                    pz.ma_ho_so,
                    pd.ten_ho_so,
                    pz.ten_loai_dat_ht,
                    pz.ten_loai_dat_qh
                  FROM property_planning_matches pm
                  LEFT JOIN planning_coordinate_lookups pcl ON pcl.id = pm.lookup_id
                  LEFT JOIN planning_zones pz ON pz.id = pm.zone_id
                  LEFT JOIN planning_dossiers pd ON pd.ma_ho_so = pz.ma_ho_so
                  WHERE pm.property_id = p.id
                  ORDER BY pm.matched_at DESC
                  LIMIT 1
                ) planning ON TRUE
                WHERE post.id = :post_id
                  AND post.deleted_at IS NULL
                  AND post.post_status = 'APPROVED'
                  AND p.deleted_at IS NULL
                  AND p.status = 'ACTIVE'
                LIMIT 1
                """
            ),
            {"post_id": int(post_id)},
        )
    ).first()

    if not row:
        return None

    r = row._mapping
    location = ", ".join(
        part
        for part in (
            _clean_cell(r.get("location")),
            _clean_cell(r.get("ward")),
            _clean_cell(r.get("district")),
            _clean_cell(r.get("province")),
        )
        if part
    )
    planning_lines = [
        "--- ĐỐI CHIẾU QUY HOẠCH ---",
        f"Trạng thái: {_clean_cell(r.get('planning_status')) or 'Chưa có bản ghi đối chiếu quy hoạch cho bất động sản này'}",
    ]
    if r.get("risk_level"):
        planning_lines.append(f"Mức rủi ro: {r.get('risk_level')}")
    if r.get("confidence_level"):
        planning_lines.append(f"Độ tin cậy: {r.get('confidence_level')}")
    if r.get("ky_quy_hoach"):
        planning_lines.append(f"Kỳ quy hoạch: {r.get('ky_quy_hoach')}")
    if r.get("ma_ho_so") or r.get("ten_ho_so"):
        planning_lines.append(f"Hồ sơ: {_clean_cell(r.get('ma_ho_so'))} {_clean_cell(r.get('ten_ho_so'))}".strip())
    if r.get("ten_loai_dat_ht") or r.get("ten_loai_dat_qh"):
        planning_lines.append(
            f"Loại đất hiện trạng/quy hoạch: {_clean_cell(r.get('ten_loai_dat_ht')) or 'N/A'} / {_clean_cell(r.get('ten_loai_dat_qh')) or 'N/A'}"
        )
    if r.get("explanation"):
        planning_lines.append(f"Giải thích: {r.get('explanation')}")

    content = "\n".join(
        [
            f"LISTING_ID: {r.get('post_id')}",
            "--- BẤT ĐỘNG SẢN ĐANG ĐƯỢC HỎI ---",
            f"Tiêu đề: {_clean_cell(r.get('post_title'))}",
            f"Loại: {'Cần bán' if r.get('post_type') == 'SALE' else 'Cho thuê' if r.get('post_type') == 'RENT' else 'Khác'}",
            f"Vị trí: {location or 'N/A'}",
            f"Tọa độ: lat={r.get('lat') or 'N/A'}, lon={r.get('lon') or 'N/A'}",
            f"Giá: {r.get('price') or 'N/A'}",
            f"Diện tích: {r.get('area') or 'N/A'} m²",
            f"Số phòng ngủ: {r.get('bedrooms') or 'N/A'}",
            f"Số phòng vệ sinh: {r.get('toilets') or 'N/A'}",
            _clean_cell(r.get("description")),
            "",
            *planning_lines,
        ]
    ).strip()

    return Document(
        page_content=content,
        metadata={
            "postId": r.get("post_id"),
            "propertyId": r.get("property_id"),
            "postType": r.get("post_type"),
            "postStatus": "APPROVED",
            "title": _clean_cell(r.get("post_title")),
            "postTitle": _clean_cell(r.get("post_title")),
            "sourceUrl": f"/posts/{r.get('post_id')}",
            "city": _clean_cell(r.get("province")),
            "district": _clean_cell(r.get("district")),
            "ward": _clean_cell(r.get("ward")),
            "location": _clean_cell(r.get("location")),
            "price": float(r["price"]) if r.get("price") is not None else None,
            "area": float(r["area"]) if r.get("area") is not None else None,
            "bedrooms": r.get("bedrooms"),
            "floorNumber": r.get("floor_number"),
            "retrievalSource": "db_exact_post_context",
            "planningStatus": r.get("planning_status"),
            "planningRiskLevel": r.get("risk_level"),
            "planningConfidenceLevel": r.get("confidence_level"),
            "planningExplanation": r.get("explanation"),
            "planningPeriod": r.get("ky_quy_hoach"),
            "planningDossierCode": r.get("ma_ho_so"),
            "planningDossierName": _clean_cell(r.get("ten_ho_so")),
            "planningCurrentLandType": r.get("ten_loai_dat_ht"),
            "planningPlannedLandType": r.get("ten_loai_dat_qh"),
        },
    )

async def _with_timeout(coro, *, seconds: float, label: str, fallback=None):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        _logger.warning("%s exceeded %.1fs; using fallback.", label, seconds)
        return fallback


def _filter_citations_to_answered_listings(
    answer: str,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not citations:
        return citations

    mentioned_ids = {
        int(match)
        for match in re.findall(
            r"(?:/posts/|BẤT\s+ĐỘNG\s+SẢN\s+|BAT\s+DONG\s+SAN\s+)(\d+)",
            answer or "",
            flags=re.IGNORECASE,
        )
    }

    normalized_answer = _normalize_nl(answer or "")
    if not mentioned_ids:
        for citation in citations:
            post_id = citation.get("postId")
            title = str(citation.get("postTitle") or citation.get("title") or "").strip()
            normalized_title = _normalize_nl(title)
            if post_id and normalized_title and len(normalized_title) >= 18 and normalized_title[:45] in normalized_answer:
                try:
                    mentioned_ids.add(int(post_id))
                except (TypeError, ValueError):
                    pass

    if not mentioned_ids:
        return citations

    filtered: list[dict[str, Any]] = []
    for citation in citations:
        post_id = citation.get("postId")
        if post_id is None:
            filtered.append(citation)
            continue
        try:
            if int(post_id) in mentioned_ids:
                filtered.append(citation)
        except (TypeError, ValueError):
            continue

    return filtered or citations


def _filter_citations_to_contextual_post(
    citations: list[dict[str, Any]],
    contextual_post_id: int | None,
    *,
    include_planning: bool,
) -> list[dict[str, Any]]:
    if contextual_post_id is None or not citations:
        return citations

    filtered: list[dict[str, Any]] = []
    for citation in citations:
        post_id = citation.get("postId")
        planning_document_id = citation.get("planningDocumentId")

        if post_id is None:
            if include_planning and planning_document_id is not None:
                filtered.append(citation)
            continue

        try:
            if int(post_id) == int(contextual_post_id):
                filtered.append(citation)
        except (TypeError, ValueError):
            continue

    return filtered or citations


def _citation_listing_overlap_score(answer: str, citation: dict[str, Any]) -> int:
    normalized_answer = _normalize_nl(answer or "")
    if not normalized_answer:
        return 0

    score = 0
    title = _normalize_nl(str(citation.get("postTitle") or citation.get("title") or ""))
    if title:
        title_tokens = [token for token in title.split() if len(token) >= 3]
        score += sum(2 for token in title_tokens[:20] if token in normalized_answer)

    for field in ("location", "ward", "district", "city", "categoryName"):
        value = _normalize_nl(str(citation.get(field) or ""))
        if value and value in normalized_answer:
            score += 4

    for field in ("area", "bedrooms", "floorNumber"):
        value = citation.get(field)
        if value in (None, ""):
            continue
        raw_number = str(value).split(".")[0]
        if raw_number and re.search(rf"\b{re.escape(raw_number)}\b", normalized_answer):
            score += 2

    price = citation.get("price")
    if price not in (None, ""):
        try:
            price_value = float(price)
            if price_value >= 1_000_000 and str(int(price_value / 1_000_000)) in normalized_answer:
                score += 2
        except (TypeError, ValueError):
            pass

    return score


def _looks_like_single_listing_answer(question: str, answer: str) -> bool:
    normalized_question = _normalize_nl(question or "")
    normalized_answer = _normalize_nl(answer or "")
    if not normalized_answer:
        return False

    link_count = len(re.findall(r"/posts/\d+", answer or "", flags=re.IGNORECASE))
    if link_count > 1:
        return False

    multi_listing_markers = (
        "cac can",
        "nhung can",
        "mot so can",
        "danh sach",
        "lua chon",
        "phuong an",
    )
    if any(marker in normalized_answer for marker in multi_listing_markers):
        return False

    if any(marker in normalized_answer for marker in ("co mot can", "mot can", "hien tai co mot")):
        return True

    return "mot can" in normalized_question or link_count == 1


def _filter_citations_to_single_answered_listing(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not citations or not _looks_like_single_listing_answer(question, answer):
        return citations

    listing_citations = [citation for citation in citations if citation.get("postId") is not None]
    if len(listing_citations) <= 1:
        return citations

    best = max(
        enumerate(listing_citations),
        key=lambda item: (_citation_listing_overlap_score(answer, item[1]), -item[0]),
    )[1]

    filtered: list[dict[str, Any]] = []
    for citation in citations:
        if citation.get("postId") is None:
            filtered.append(citation)
            continue
        if citation is best:
            filtered.append(citation)
    return filtered or [best]


async def initialize_vector_store():
    """Khởi tạo async vector store - gọi từ startup event"""
    listing_vs = await initialize_listing_vector_store()
    await initialize_planning_vector_store()
    return listing_vs

class ChatRequest(BaseModel):
    userId: Optional[int] = None
    sessionId: Optional[int] = None
    postId: Optional[int] = None
    message: str = Field(min_length=1)

class ChatResponse(BaseModel):
    sessionId: int
    answer: str
    citations: list[dict[str, Any]]
    extractedFilters: dict[str, Any] = Field(default_factory=dict)
    tokenUsage: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    if not req.userId:
        raise ValueError("userId is required for chat history")

    from ..rag.chain import RagChain, build_retrieval_query
    
    history_manager = MessageHistoryManager(db)
    session_id = await history_manager.get_or_create_session(req.userId, req.sessionId)
    
    # Lấy lịch sử 6 tin nhắn gần nhất
    history = await history_manager.get_messages(session_id, limit=6)
    retrieval_message = build_retrieval_query(req.message, history)
    top_k = max(1, min(int(os.getenv("TOP_K_DEFAULT", "16")), 50))
    contextual_post_id = _extract_contextual_post_id(req.message, history, req.postId)
    
    llm = build_llm()

    use_planning_mode = _has_planning_intent(req.message)
    filters: dict[str, Any] = {}
    filter_token_usage: dict[str, Any] = {}
    if not use_planning_mode:
        filter_query = retrieval_message if _normalize_nl(retrieval_message) != _normalize_nl(req.message) else req.message
        filters, filter_token_usage = await extract_filters_from_query_with_usage(filter_query, llm)
        _logger.debug("Extracted listing filters: %s", filters)

    exact_post_context_doc: Document | None = None
    if contextual_post_id is not None:
        filters["postId"] = contextual_post_id
        exact_post_context_doc = await _load_post_context_document(db, contextual_post_id)
        if exact_post_context_doc:
            md = exact_post_context_doc.metadata or {}
            location_hint = ", ".join(
                str(part)
                for part in (md.get("location"), md.get("ward"), md.get("district"), md.get("city"))
                if part
            )
            retrieval_message = "\n".join(
                part
                for part in (
                    retrieval_message,
                    f"Bất động sản đang được hỏi: postId={contextual_post_id}",
                    f"Vị trí bất động sản: {location_hint}" if location_hint else "",
                )
                if part
            )

    extra_context = ""
    planning_docs: list[Document] = []
    planning_vs = None

    if use_planning_mode:
        planning_vs = await initialize_planning_vector_store()
        retriever = _StaticDocumentsRetriever([])
    else:
        vs = await initialize_listing_vector_store()
        retriever = ListingFallbackRetriever(
            build_retriever(
                vs,
                k=top_k,
                filters=filters,
                base_filter={"postStatus": "APPROVED"},
            ),
            query=retrieval_message,
            filters=filters,
            k=top_k,
        )

    if use_planning_mode:
        planning_docs = await _with_timeout(
            _retrieve_planning_docs_for_nl_query(
                planning_vs,
                retrieval_message,
                top_k,
                history_messages=history,
            ),
            seconds=float(os.getenv("CHAT_PLANNING_RETRIEVAL_TIMEOUT_SECONDS", "25")),
            label="planning NL retrieval",
            fallback=[],
        )

        if planning_docs:
            district = _extract_district_from_message(retrieval_message) or _extract_district_from_message(req.message)
            plan_year = _extract_plan_year_from_message(retrieval_message) or _extract_plan_year_from_message(req.message)
            header = [
                "=== PLANNING ANSWER GUIDANCE ===",
                f"District hint: {district or 'N/A'}",
                f"Plan year hint: {plan_year if plan_year is not None else 'N/A'}",
                (
                    "If the question asks whether the target listing is affected by planning, "
                    "use the target listing's parcel-level planning status first. "
                    "Treat retrieved planning documents as district-level support unless they explicitly name the target address."
                ),
            ]
            planning_guidance = "\n".join(header)
            extra_context = f"{extra_context}\n\n{planning_guidance}" if extra_context else planning_guidance

    if use_planning_mode:
        if exact_post_context_doc:
            planning_docs = [exact_post_context_doc, *planning_docs]
        retriever = _StaticDocumentsRetriever(planning_docs)

    chain = RagChain(llm=llm, retriever=retriever)
    context_max_docs = int(
        os.getenv(
            "RAG_PLANNING_CONTEXT_MAX_DOCS" if use_planning_mode else "RAG_LISTING_CONTEXT_MAX_DOCS",
            os.getenv("RAG_CONTEXT_MAX_DOCS", "4"),
        )
    )
    context_max_chars_per_doc = int(
        os.getenv(
            "RAG_PLANNING_CONTEXT_MAX_CHARS_PER_DOC" if use_planning_mode else "RAG_LISTING_CONTEXT_MAX_CHARS_PER_DOC",
            os.getenv("RAG_CONTEXT_MAX_CHARS_PER_DOC", "1400"),
        )
    )
    result = await chain.run(
        req.message,
        history=history,
        extra_context=extra_context,
        max_docs=context_max_docs,
        max_chars_per_doc=context_max_chars_per_doc,
    )
    merged_citations = _filter_citations_to_contextual_post(
        result.citations,
        contextual_post_id,
        include_planning=use_planning_mode,
    )
    merged_citations = _filter_citations_to_answered_listings(
        result.answer,
        merged_citations,
    )
    merged_citations = _dedupe_citations(merged_citations)
    if not use_planning_mode:
        merged_citations = _filter_citations_to_single_answered_listing(
            req.message,
            result.answer,
            merged_citations,
        )
    reranked_citations = _rerank_citations(merged_citations)
    
    await history_manager.add_message(session_id, "user", req.message)
    await history_manager.add_message(session_id, "assistant", result.answer)

    answer_token_usage = result.token_usage or {}
    
    return ChatResponse(
        sessionId=session_id,
        answer=result.answer,
        citations=reranked_citations,
        extractedFilters=filters,
        tokenUsage={
            "filter_extraction": filter_token_usage,
            "answer_generation": answer_token_usage,
            "total": sum_token_usage([filter_token_usage, answer_token_usage]),
        },
        timings=result.timings or {},
    )
