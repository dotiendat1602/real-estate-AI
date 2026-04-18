from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional
import re
import unicodedata
from collections import OrderedDict
from functools import lru_cache
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI

from ..rag.embedder import build_embeddings
from ..rag.retriever import build_pgvector_store, build_retriever, lexical_search_documents
from ..rag.filter_extractor import extract_filters_from_query
from ..rag.chain import RagChain, build_retrieval_query
from ..rag.planning_pipeline import (
    choose_better_planning_fallback,
    is_recovery_grouping_query,
)
from ..planning.features import (
    planning_admin_unit_header_hits as _planning_admin_unit_header_hits,
    planning_doc_haystack as _planning_doc_haystack,
    planning_has_admin_unit_evidence as _planning_has_admin_unit_evidence,
    planning_has_direct_admin_unit_count_phrase as _planning_has_direct_admin_unit_count_phrase,
    planning_has_direct_natural_area_phrase as _planning_has_direct_natural_area_phrase,
    planning_has_explicit_project_row as _planning_has_explicit_project_row,
    planning_has_natural_area_admin_evidence as _planning_has_natural_area_admin_evidence,
    planning_is_toc_like_chunk as _is_planning_toc_like_chunk,
    planning_land_change_label_hits as _planning_land_change_label_hits,
    planning_registered_plan_evidence_hits as _planning_registered_plan_evidence_hits,
    strip_planning_metadata_lines as _strip_planning_metadata_lines,
)
from ..planning.docs import (
    dedupe_planning_docs as _dedupe_planning_docs,
    planning_chunk_type as _planning_chunk_type,
    planning_doc_identity as _planning_doc_identity,
)
from ..planning.metadata import canonicalize_planning_district
from ..planning.profiles import (
    build_planning_query_profile,
)
from ..planning.augmenters import (
    augment_planning_continuation_neighbors as _augment_planning_continuation_neighbors_base,
    augment_planning_intent_evidence as _augment_planning_intent_evidence_base,
    augment_planning_land_change_fact_docs as _augment_planning_land_change_fact_docs_base,
    augment_planning_land_recovery_evidence as _augment_planning_land_recovery_evidence_base,
    augment_planning_table_neighbors as _augment_planning_table_neighbors_base,
    augment_planning_text_neighbors as _augment_planning_text_neighbors_base,
)
from ..planning.query_builders import (
    planning_intent_rescue_queries as _planning_intent_rescue_queries_base,
    planning_query_candidates as _planning_query_candidates_base,
)
from ..planning.ranker import (
    planning_doc_score as _planning_doc_score_base,
    planning_query_terms as _planning_query_terms,
    planning_specialized_evidence_score as _planning_specialized_evidence_score_base,
)
from ..planning.selector import select_ranked_planning_docs as _select_ranked_planning_docs_base
from ..planning.specialized import force_planning_specialized_evidence as _force_planning_specialized_evidence_base
from ..rag.message_history import MessageHistoryManager
from ..db.pgvector import get_db

router = APIRouter()

_embeddings = build_embeddings()
_vs = None
_planning_vs = None

_PLANNING_KEYWORDS = [
    "quy hoach",
    "quy hoach su dung dat",
    "ke hoach su dung dat",
    "ke hoach",
    "khsd",
    "khsdd",
    "thong tin quy hoach",
    "loai dat",
    "thua dat",
    "to ban do",
    "muc dich su dung dat",
    "quy hoach do thi",
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
    "thu hoi dat",
    "dat nong nghiep",
    "dat phi nong nghiep",
    "dat chua su dung",
    "dua dat chua su dung vao su dung",
    "dien tich tu nhien",
    "don vi hanh chinh",
    "luat dat dai",
]

_PLANNING_LAND_ADMIN_TERMS = [
    "thu hoi",
    "dat nong nghiep",
    "dat phi nong nghiep",
    "dat chua su dung",
    "dien tich tu nhien",
    "don vi hanh chinh",
    "cong trinh",
    "du an",
    "luat dat dai",
]

_PLANNING_REASON_CONTEXT_TERMS = [
    "quan ly dat dai",
    "giai phap",
    "ha tang",
    "thoat nuoc",
    "giao thong",
    "dich vu",
    "nha o",
    "do thi hoa",
    "dan so co hoc",
    "bai giua",
    "song hong",
    "giai phong mat bang",
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

_DISTRICT_PREFIX_PATTERN = re.compile(r"^(quan|huyen|thi\s+xa|thi\s+tran)\s+")

def _canonical_district_name(seed: str) -> str:
    return canonicalize_planning_district(seed, title=seed, dossier_code="") or seed


_DISTRICT_MATCH_ALIASES: dict[str, tuple[str, ...]] = {
    _canonical_district_name("Ba Dinh"): ("ba dinh", "quan ba dinh"),
    _canonical_district_name("Bac Tu Liem"): ("bac tu liem", "quan bac tu liem"),
    _canonical_district_name("Cau Giay"): ("cau giay", "quan cau giay"),
    _canonical_district_name("Dong Da"): ("dong da", "quan dong da"),
    _canonical_district_name("Hai Ba Trung"): ("hai ba trung", "quan hai ba trung"),
    _canonical_district_name("Ha Dong"): ("ha dong", "quan ha dong"),
    _canonical_district_name("Hoan Kiem"): ("hoan kiem", "quan hoan kiem"),
    _canonical_district_name("Hoang Mai"): ("hoang mai", "quan hoang mai"),
    _canonical_district_name("Long Bien"): ("long bien", "quan long bien"),
    _canonical_district_name("Nam Tu Liem"): ("nam tu liem", "quan nam tu liem"),
    _canonical_district_name("Son Tay"): ("son tay", "thi xa son tay"),
    _canonical_district_name("Tay Ho"): ("tay ho", "quan tay ho"),
    _canonical_district_name("Thanh Xuan"): ("thanh xuan", "quan thanh xuan"),
}

_WARD_DISTRICT_HINTS: dict[str, tuple[str, ...]] = {
    _canonical_district_name("Hoang Mai"): ("yen so",),
    _canonical_district_name("Cau Giay"): ("mai dich",),
    _canonical_district_name("Hoan Kiem"): ("chuong duong",),
}


_PLANNING_MAX_QUERY_CANDIDATES = 12
_PLANNING_MAX_FACT_SUBQUERIES = 8
_PLANNING_QUERY_CACHE_SIZE = 256

_PLANNING_QUERY_CACHE: OrderedDict[str, tuple[list[Document], list[Document]]] = OrderedDict()

_MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00c4", "\u00c6", "\u00e2\u20ac", "\u2013", "\u2014", "\ufffd")


def _repair_mojibake(text: str) -> str:
    raw = str(text or "")
    if not raw or not any(marker in raw for marker in _MOJIBAKE_MARKERS):
        return raw

    candidates = [raw]
    for codec in ("latin1", "cp1252"):
        current = raw
        for _ in range(2):
            try:
                repaired = current.encode(codec).decode("utf-8")
            except Exception:
                break
            candidates.append(repaired)
            current = repaired

    def _score(value: str) -> tuple[int, int]:
        marker_hits = sum(value.count(marker) for marker in _MOJIBAKE_MARKERS)
        replacement_hits = value.count("\ufffd")
        return marker_hits + replacement_hits * 2, len(value)

    return min(candidates, key=_score)


def _strip_accents(text: str) -> str:
    normalized = _repair_mojibake(text or "").replace("\u0111", "d").replace("\u0110", "D")
    decomposed = unicodedata.normalize("NFD", normalized)
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

    has_district_hint = _extract_district_from_message(message) is not None
    has_plan_year = _extract_plan_year_from_message(message) is not None
    has_admin_unit_hint = bool(re.search(r"\b(phuong|xa|thi tran)\b", normalized))

    has_land_admin_term = any(term in normalized for term in _PLANNING_LAND_ADMIN_TERMS)
    if has_land_admin_term and (has_district_hint or has_plan_year or has_admin_unit_hint):
        return True

    has_reason_context_term = any(term in normalized for term in _PLANNING_REASON_CONTEXT_TERMS)
    if has_reason_context_term and (has_district_hint or has_plan_year or has_admin_unit_hint):
        return True

    has_structural_term = any(term in normalized for term in _PLANNING_STRUCTURAL_TERMS)
    if not has_structural_term:
        return False

    # If user mentions land/planning structural terms plus either district hint
    # or plan year, treat it as planning intent even without exact keyword phrases.
    return has_district_hint or has_plan_year or has_admin_unit_hint


@lru_cache(maxsize=512)
def _planning_query_profile(message: str):
    return build_planning_query_profile(
        message,
        planning_intent=_has_planning_intent(message),
    )


def _extract_district_from_message(message: str) -> Optional[str]:
    normalized = _normalize_nl(message)
    if not normalized:
        return None

    padded = f" {normalized} "

    for canonical, aliases in _DISTRICT_MATCH_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_nl(alias)
            if alias_norm and f" {alias_norm} " in padded:
                return canonical

    for canonical, ward_aliases in _WARD_DISTRICT_HINTS.items():
        for alias in ward_aliases:
            alias_norm = _normalize_nl(alias)
            if alias_norm and f" {alias_norm} " in padded:
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


def _history_role(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("role") or "").strip().lower()

    msg_type = str(getattr(item, "type", "") or "").strip().lower()
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"

    role = str(getattr(item, "role", "") or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "ai"}:
        return "assistant"
    return role


def _history_content(item: Any) -> str:
    if isinstance(item, dict):
        content = item.get("content")
    else:
        content = getattr(item, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return " ".join(chunks)

    return str(content or "")


def _extract_district_from_history(history_messages: Optional[list[Any]]) -> Optional[str]:
    if not history_messages:
        return None

    for item in reversed(history_messages):
        role = _history_role(item)
        if role != "user":
            continue
        district = _extract_district_from_message(_history_content(item))
        if district:
            return district

    for item in reversed(history_messages):
        district = _extract_district_from_message(_history_content(item))
        if district:
            return district
    return None


def _extract_plan_year_from_history(history_messages: Optional[list[Any]]) -> Optional[int]:
    if not history_messages:
        return None

    for item in reversed(history_messages):
        role = _history_role(item)
        if role != "user":
            continue
        year = _extract_plan_year_from_message(_history_content(item))
        if year is not None:
            return year

    for item in reversed(history_messages):
        year = _extract_plan_year_from_message(_history_content(item))
        if year is not None:
            return year
    return None


def _is_planning_fact_query(message: str) -> bool:
    return _planning_query_profile(message).fact_query


def _is_planning_project_listing_query(message: str) -> bool:
    return _planning_query_profile(message).project_listing


def _is_planning_land_change_query(message: str) -> bool:
    return _planning_query_profile(message).land_change


def _district_tokens(district: Optional[str]) -> list[str]:
    if not district:
        return []

    canonical = canonicalize_planning_district(district, title=district, dossier_code="") or district.strip()
    variants: list[str] = [canonical]
    variants.extend(_DISTRICT_MATCH_ALIASES.get(canonical, ()))

    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = _normalize_nl(item)
        for token in normalized.split():
            if len(token) < 3 or token in _PLANNING_DISTRICT_STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def _district_aliases_for_matching(district: Optional[str]) -> list[str]:
    if not district:
        return []

    canonical = canonicalize_planning_district(district, title=district, dossier_code="") or district.strip()
    candidates = [canonical, _strip_accents(canonical)]
    candidates.extend(_DISTRICT_MATCH_ALIASES.get(canonical, ()))

    out: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        normalized = _normalize_nl(value)
        if not normalized:
            continue

        variants = [normalized, _DISTRICT_PREFIX_PATTERN.sub("", normalized).strip()]
        for variant in variants:
            if len(variant) < 3 or variant in seen:
                continue
            seen.add(variant)
            out.append(variant)

    return out


def _doc_matches_district(doc: Document, district: Optional[str]) -> bool:
    if not district:
        return True

    md = doc.metadata or {}
    expected = canonicalize_planning_district(
        district,
        title=district,
        dossier_code=str(md.get("dossierCode") or ""),
    )
    doc_district = canonicalize_planning_district(
        str(md.get("district") or "") or str(md.get("districtRaw") or ""),
        title=str(md.get("title") or ""),
        dossier_code=str(md.get("dossierCode") or ""),
    )
    if expected and doc_district and expected == doc_district:
        return True

    haystack = _planning_doc_haystack(doc)
    padded_haystack = f" {haystack} "

    aliases = _district_aliases_for_matching(expected or district)
    if aliases and any(f" {alias} " in padded_haystack for alias in aliases):
        return True

    # Fallback when phrase matching fails on sparse chunks.
    tokens = _district_tokens(expected or district)
    if len(tokens) >= 2:
        return all(f" {token} " in padded_haystack for token in tokens)

    return False


def _doc_matches_plan_year(doc: Document, plan_year: Optional[int]) -> bool:
    if plan_year is None:
        return True

    md = doc.metadata or {}
    raw_year = md.get("planYear")
    if raw_year is not None:
        try:
            if int(raw_year) == int(plan_year):
                return True
        except Exception:
            pass

    return str(plan_year) in _planning_doc_haystack(doc)


def _planning_doc_score(doc: Document, message: str, district: Optional[str], plan_year: Optional[int]) -> float:
    return _planning_doc_score_base(
        doc,
        message,
        district,
        plan_year,
        doc_matches_district=_doc_matches_district,
        doc_matches_plan_year=_doc_matches_plan_year,
    )


def _planning_specialized_evidence_score(
    doc: Document,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
) -> Optional[float]:
    return _planning_specialized_evidence_score_base(
        doc,
        message,
        district,
        plan_year,
        doc_matches_district=_doc_matches_district,
        doc_matches_plan_year=_doc_matches_plan_year,
    )


def _planning_intent_rescue_queries(message: str, district: Optional[str], plan_year: Optional[int]) -> list[str]:
    return _planning_intent_rescue_queries_base(message, district, plan_year)


def _planning_query_candidates(message: str, district: Optional[str], plan_year: Optional[int]) -> list[str]:
    return _planning_query_candidates_base(
        message,
        district,
        plan_year,
        max_query_candidates=_PLANNING_MAX_QUERY_CANDIDATES,
        max_fact_subqueries=_PLANNING_MAX_FACT_SUBQUERIES,
    )


def _select_ranked_planning_docs(
    ranked_docs: list[Document],
    message: str,
    limit: int,
    district: Optional[str] = None,
    plan_year: Optional[int] = None,
) -> list[Document]:
    return _select_ranked_planning_docs_base(
        ranked_docs,
        message,
        limit,
        district,
        plan_year,
        doc_matches_district=_doc_matches_district,
        doc_matches_plan_year=_doc_matches_plan_year,
        planning_doc_score=_planning_doc_score,
    )


async def _augment_planning_text_neighbors(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    limit: int,
) -> list[Document]:
    return await _augment_planning_text_neighbors_base(
        planning_vs,
        message,
        district,
        plan_year,
        selected_docs,
        limit,
        planning_doc_score=_planning_doc_score,
    )


async def _augment_planning_continuation_neighbors(
    planning_vs,
    message: str,
    selected_docs: list[Document],
    limit: int,
) -> list[Document]:
    return await _augment_planning_continuation_neighbors_base(
        planning_vs,
        message,
        selected_docs,
        limit,
    )


async def _augment_planning_land_change_fact_docs(
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    limit: int,
) -> list[Document]:
    return await _augment_planning_land_change_fact_docs_base(
        message,
        district,
        plan_year,
        selected_docs,
        limit,
        planning_doc_score=_planning_doc_score,
        load_planning_document_docs=_load_planning_document_docs_sync,
    )


async def _augment_planning_table_neighbors(
    planning_vs,
    message: str,
    selected_docs: list[Document],
    limit: int,
) -> list[Document]:
    return await _augment_planning_table_neighbors_base(
        planning_vs,
        message,
        selected_docs,
        limit,
    )


async def _augment_planning_intent_evidence(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    pool_docs: list[Document],
    limit: int,
) -> list[Document]:
    return await _augment_planning_intent_evidence_base(
        planning_vs,
        message,
        district,
        plan_year,
        selected_docs,
        pool_docs,
        limit,
        planning_doc_score=_planning_doc_score,
        planning_specialized_evidence_score=_planning_specialized_evidence_score,
        planning_intent_rescue_queries=_planning_intent_rescue_queries,
    )


async def _augment_planning_land_recovery_evidence(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    pool_docs: list[Document],
    limit: int,
) -> list[Document]:
    return await _augment_planning_land_recovery_evidence_base(
        planning_vs,
        message,
        district,
        plan_year,
        selected_docs,
        pool_docs,
        limit,
        planning_doc_score=_planning_doc_score,
    )


async def _force_planning_specialized_evidence(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    limit: int,
) -> list[Document]:
    return await _force_planning_specialized_evidence_base(
        planning_vs,
        message,
        district,
        plan_year,
        selected_docs,
        limit,
        planning_doc_score=_planning_doc_score,
        planning_specialized_evidence_score=_planning_specialized_evidence_score,
        load_planning_document_docs=_load_planning_document_docs_sync,
        load_admin_overview_sql_rescue_docs=_load_admin_overview_sql_rescue_docs,
    )


def _rebalance_planning_chunk_mix(
    docs: list[Document],
    *,
    limit: int,
    fact_query: bool,
) -> list[Document]:
    if not docs or limit <= 0:
        return []

    text_docs: list[Document] = []
    table_docs: list[Document] = []
    other_docs: list[Document] = []
    seen: set[str] = set()

    for doc in docs:
        identity = _planning_doc_identity(doc)
        if identity in seen:
            continue
        seen.add(identity)

        chunk_type = _planning_chunk_type(doc)
        if chunk_type == "text":
            text_docs.append(doc)
        elif chunk_type == "table":
            table_docs.append(doc)
        else:
            other_docs.append(doc)

    if not table_docs or not text_docs:
        return [*text_docs, *table_docs, *other_docs][:limit]

    if fact_query:
        table_target = min(len(table_docs), max(1, min(limit // 2, 3)))
    else:
        table_target = min(len(table_docs), max(1, min(limit // 3, 2)))
    text_target = min(len(text_docs), max(1, limit - table_target))

    merged: list[Document] = []
    text_idx = 0
    table_idx = 0
    start_with_text = len(text_docs) >= len(table_docs)

    while len(merged) < min(limit, text_target + table_target):
        picked = False
        if start_with_text and text_idx < text_target:
            merged.append(text_docs[text_idx])
            text_idx += 1
            picked = True
        if len(merged) >= min(limit, text_target + table_target):
            break
        if table_idx < table_target:
            merged.append(table_docs[table_idx])
            table_idx += 1
            picked = True
        if len(merged) >= min(limit, text_target + table_target):
            break
        if not start_with_text and text_idx < text_target:
            merged.append(text_docs[text_idx])
            text_idx += 1
            picked = True
        if not picked:
            break
        start_with_text = True

    for doc in [*text_docs[text_idx:], *table_docs[table_idx:], *other_docs]:
        if len(merged) >= limit:
            break
        merged.append(doc)

    return merged[:limit]


def _select_relevant_content_lines(
    text: str,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    max_lines: int,
) -> list[str]:
    if max_lines <= 0 or not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    query_terms = _planning_query_terms(message, max_terms=20)
    district_aliases = set(_district_aliases_for_matching(district))
    year_token = str(plan_year) if plan_year is not None else ""

    scored: list[tuple[float, int, str]] = []
    for idx, line in enumerate(lines):
        normalized = _normalize_nl(line)
        score = 0.0
        if idx < 3:
            score += 0.4
        if query_terms:
            score += sum(1 for term in query_terms if term in normalized) * 1.2
        if district_aliases:
            score += sum(1 for alias in district_aliases if alias in normalized) * 1.4
        if year_token and year_token in normalized:
            score += 1.0
        if re.search(r"\b\d+(?:[\.,]\d+)?\b", normalized):
            score += 0.35
        if _planning_has_explicit_project_row(normalized):
            score += 1.4
        if planning_marker_hits := _planning_registered_plan_evidence_hits(normalized):
            score += min(planning_marker_hits, 3) * 0.8
        if _planning_land_change_label_hits(normalized) > 0:
            score += 0.8
        if any(marker in normalized for marker in ("tong so", "tong cong", "dien tich", "thu hoi", "quyet dinh", "nghi quyet")):
            score += 0.6
        scored.append((score, idx, line))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    chosen = sorted(scored[: min(max_lines, len(scored))], key=lambda item: item[1])
    return [line for _, _, line in chosen]


def _compact_planning_doc(
    doc: Document,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    fact_query: bool,
) -> Document:
    md = doc.metadata or {}
    raw_content = doc.page_content or ""
    raw_body = _strip_planning_metadata_lines(raw_content)
    if not _normalize_nl(raw_body):
        raw_body = raw_content

    header_lines: list[str] = []
    for key, label in (
        ("city", "city"),
        ("district", "district"),
        ("planYear", "plan_year"),
        ("dossierCode", "dossier_code"),
        ("title", "title"),
    ):
        value = md.get(key)
        if value not in (None, ""):
            header_lines.append(f"[{label}={value}]")

    chunk_type = md.get("chunkType")
    if chunk_type not in (None, ""):
        header_lines.append(
            "document_type:unknown | "
            f"district:{md.get('district') or ''} | "
            f"plan_year:{md.get('planYear') or ''} | "
            f"chunk_type:{chunk_type} | "
            f"title:{md.get('title') or ''}"
        )

    if len(raw_body) <= (2200 if fact_query else 1200):
        compact_body = raw_body.strip()
    else:
        excerpt_lines = _select_relevant_content_lines(
            raw_body,
            message,
            district,
            plan_year,
            max_lines=28 if fact_query else 12,
        )
        compact_body = "\n".join(excerpt_lines).strip() or raw_body[: (2200 if fact_query else 1200)].strip()

    parts = [*header_lines, compact_body] if compact_body else header_lines
    return Document(page_content="\n".join(part for part in parts if part).strip(), metadata=md)


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
        compacted_body = _strip_planning_metadata_lines(compacted.page_content or "")
        if not _normalize_nl(compacted_body):
            continue
        md = compacted.metadata or {}
        chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
        key = "|".join(
            [
                str(md.get("planningDocumentId") or ""),
                str(md.get("chunkType") or ""),
                str(chunk_idx or ""),
                _normalize_nl(compacted.page_content)[:280],
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(compacted)

    return out


def _planning_query_cache_key(
    query_text: str,
    base_filter: dict[str, Any],
    probe_k: int,
    lexical_k: int,
    use_lexical: bool,
) -> str:
    payload = {
        "query": _normalize_nl(query_text),
        "base": base_filter,
        "probe_k": probe_k,
        "lexical_k": lexical_k,
        "use_lexical": use_lexical,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _planning_query_cache_get(key: str) -> tuple[list[Document], list[Document]] | None:
    cached = _PLANNING_QUERY_CACHE.get(key)
    if cached is None:
        return None
    _PLANNING_QUERY_CACHE.move_to_end(key)
    return cached


def _planning_query_cache_put(key: str, value: tuple[list[Document], list[Document]]) -> None:
    _PLANNING_QUERY_CACHE[key] = value
    _PLANNING_QUERY_CACHE.move_to_end(key)
    while len(_PLANNING_QUERY_CACHE) > _PLANNING_QUERY_CACHE_SIZE:
        _PLANNING_QUERY_CACHE.popitem(last=False)


def _planning_sync_database_url() -> str | None:
    raw = (os.getenv("DATABASE_URL", "") or "").strip()
    if not raw:
        return None
    if raw.startswith("postgresql+asyncpg://"):
        return "postgresql://" + raw[len("postgresql+asyncpg://") :]
    return raw


def _load_planning_document_docs_sync(
    planning_document_id: int,
    plan_year: Optional[int],
    *,
    chunk_types: tuple[str, ...] = ("text", "table"),
    limit: int = 2500,
) -> list[Document]:
    conninfo = _planning_sync_database_url()
    if not conninfo:
        return []

    try:
        import psycopg
    except Exception:
        return []

    docs: list[Document] = []
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                year_value = str(plan_year) if plan_year is not None else None
                query = """
                    select document, cmetadata
                    from ai.langchain_pg_embedding
                    where collection_id = (
                        select uuid from ai.langchain_pg_collection where name = %s
                    )
                      and cmetadata->>'planningDocumentId' = %s
                """
                params: list[Any] = [
                    os.getenv("PGVECTOR_COLLECTION_PLANNING", "planning_documents"),
                    str(planning_document_id),
                ]
                if year_value is not None:
                    query += " and cmetadata->>'planYear' = %s"
                    params.append(year_value)
                if chunk_types:
                    query += " and coalesce(cmetadata->>'chunkType', 'text') = any(%s)"
                    params.append(list(chunk_types))
                query += """
                    order by cast(coalesce(cmetadata->>'globalChunkIndex', cmetadata->>'chunkIndex', '0') as int)
                    limit %s
                """
                params.append(max(1, limit))
                cur.execute(query, tuple(params))
                for document, metadata in cur.fetchall():
                    md = dict(metadata or {})
                    docs.append(Document(page_content=str(document or ""), metadata=md))
    except Exception:
        return []

    return docs


def _load_admin_overview_sql_rescue_docs_sync(
    planning_document_id: int,
    plan_year: Optional[int],
    limit: int = 6,
) -> list[Document]:
    conninfo = _planning_sync_database_url()
    if not conninfo:
        return []

    try:
        import psycopg
    except Exception:
        return []

    scored_docs: list[tuple[float, int, int, Document]] = []
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                year_value = str(plan_year) if plan_year is not None else None
                query = """
                    select document, cmetadata
                    from ai.langchain_pg_embedding
                    where collection_id = (
                        select uuid from ai.langchain_pg_collection where name = %s
                    )
                      and cmetadata->>'planningDocumentId' = %s
                """
                params: list[Any] = [
                    os.getenv("PGVECTOR_COLLECTION_PLANNING", "planning_documents"),
                    str(planning_document_id),
                ]
                if year_value is not None:
                    query += " and cmetadata->>'planYear' = %s"
                    params.append(year_value)
                query += """
                      and coalesce(cmetadata->>'chunkType', 'text') in ('text', 'table')
                    order by cast(coalesce(cmetadata->>'globalChunkIndex', cmetadata->>'chunkIndex', '0') as int)
                    limit 2500
                """
                cur.execute(
                    query,
                    tuple(params),
                )
                for document, metadata in cur.fetchall():
                    md = dict(metadata or {})
                    doc = Document(page_content=str(document or ""), metadata=md)
                    haystack = _planning_doc_haystack(doc)
                    if not haystack or _is_planning_toc_like_chunk(doc.page_content or "", haystack):
                        continue

                    admin_unit_hits = _planning_admin_unit_header_hits(haystack)
                    has_natural_area = any(
                        marker in haystack
                        for marker in ("dien tich tu nhien", "tong dien tich tu nhien", "co dien tich tu nhien")
                    )
                    has_admin_units = _planning_has_admin_unit_evidence(haystack)
                    if not has_natural_area and not has_admin_units:
                        continue

                    score = 0.0
                    if has_natural_area:
                        score += 3.0
                    if has_admin_units:
                        score += 3.0
                    if _planning_has_natural_area_admin_evidence(haystack):
                        score += 6.0
                    if _planning_has_direct_natural_area_phrase(haystack):
                        score += 3.2
                    if _planning_has_direct_admin_unit_count_phrase(haystack):
                        score += 3.0
                    if re.search(r"\b\d+(?:[\.,]\d+)?\s*ha\b", haystack):
                        score += 1.4
                    if re.search(r"\b(?:gom|co)\s+\d+\s+don\s+vi\s+hanh\s+chinh\b", haystack):
                        score += 2.2
                    if re.search(r"\b\d+\s+phuong\b", haystack) and re.search(r"\b\d+\s+xa\b", haystack):
                        score += 1.8
                    score += min(admin_unit_hits, 15) * 0.18

                    raw_chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
                    try:
                        chunk_idx = int(raw_chunk_idx) if raw_chunk_idx is not None else 0
                    except (TypeError, ValueError):
                        chunk_idx = 0

                    chunk_type = str(md.get("chunkType") or "").lower().strip()
                    type_bias = 0 if chunk_type == "text" else 1
                    scored_docs.append((score, type_bias, chunk_idx, doc))
    except Exception:
        return []

    scored_docs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [doc for _, _, _, doc in scored_docs[: max(1, limit)]]


async def _load_admin_overview_sql_rescue_docs(
    planning_document_id: int,
    plan_year: Optional[int],
    limit: int = 6,
) -> list[Document]:
    return await asyncio.to_thread(
        _load_admin_overview_sql_rescue_docs_sync,
        planning_document_id,
        plan_year,
        limit,
    )


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / float(k + rank + 1)


def _planning_scope_min_docs(scope: str, final_k: int, fact_query: bool) -> int:
    # Prevent thin planning contexts from returning too early.
    default_threshold = max(4, final_k // 2)
    if scope == "strict" and fact_query:
        default_threshold = max(default_threshold, min(final_k, 6))

    env_key = "RAG_PLANNING_STRICT_MIN_DOCS" if scope == "strict" else "RAG_PLANNING_DISTRICT_MIN_DOCS"
    raw_override = (os.getenv(env_key, "") or "").strip()
    if raw_override:
        try:
            default_threshold = int(raw_override)
        except ValueError:
            pass

    return max(1, min(final_k, default_threshold))


def _planning_debug_enabled() -> bool:
    raw = (os.getenv("RAG_PLANNING_DEBUG", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on", "debug"}


def _planning_doc_debug_fields(doc: Document) -> dict[str, Any]:
    md = doc.metadata or {}
    chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
    return {
        "planningDocumentId": md.get("planningDocumentId"),
        "chunkType": md.get("chunkType"),
        "chunkIndex": chunk_idx,
        "pageNumber": md.get("pageNumber"),
        "planYear": md.get("planYear"),
        "district": md.get("district"),
        "sourceLocator": md.get("sourceLocator"),
    }


def _planning_debug_log(event: str, payload: Any) -> None:
    if not _planning_debug_enabled():
        return

    try:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(payload)
    print(f"[PlanningDebug] {event}: {serialized}")


def _planning_debug_doc_list(docs: list[Document], *, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, doc in enumerate(docs[: max(0, limit)]):
        fields = _planning_doc_debug_fields(doc)
        fields["rank"] = rank + 1
        fields["identity"] = _planning_doc_identity(doc)
        fields["preview"] = _normalize_nl((doc.page_content or ""))[:120]
        out.append(fields)
    return out


async def _retrieve_planning_docs_for_nl_query(
    planning_vs,
    message: str,
    top_k: int,
    history_messages: Optional[list[dict[str, Any]]] = None,
    force_planning_document_id: Optional[int] = None,
) -> list[Document]:
    district = _extract_district_from_message(message) or _extract_district_from_history(history_messages)
    plan_year = _extract_plan_year_from_message(message)
    if plan_year is None:
        plan_year = _extract_plan_year_from_history(history_messages)

    fact_query = _is_planning_fact_query(message)
    final_k = max(4, min(top_k, 14 if fact_query else 10))
    strict_min_docs = _planning_scope_min_docs("strict", final_k, fact_query)
    district_min_docs = _planning_scope_min_docs("district", final_k, fact_query)
    probe_k = max(20, min(120, final_k * 8))
    lexical_k = max(12, min(40, probe_k // 3))
    chunk_types = ["text", "table"]

    query_candidates = _planning_query_candidates(message, district, plan_year)
    debug_enabled = _planning_debug_enabled()
    recovery_grouping_query = is_recovery_grouping_query(_normalize_nl(message))
    project_listing_query = _is_planning_project_listing_query(message)

    if debug_enabled:
        _planning_debug_log(
            "query_context",
            {
                "message": message,
                "district": district,
                "planYear": plan_year,
                "factQuery": fact_query,
                "topK": top_k,
                "finalK": final_k,
                "strictMinDocs": strict_min_docs,
                "districtMinDocs": district_min_docs,
                "probeK": probe_k,
                "lexicalK": lexical_k,
                "queryCandidates": query_candidates,
                "recoveryGroupingQuery": recovery_grouping_query,
                "projectListingQuery": project_listing_query,
            },
        )

    district_canonical = canonicalize_planning_district(district, title=district, dossier_code="") if district else None
    raw_base_candidates: list[dict[str, Any]] = []

    if district_canonical:
        if plan_year is not None:
            raw_base_candidates.append(
                {
                    "documentScope": "planning",
                    "district": district_canonical,
                    "planYear": plan_year,
                }
            )
        raw_base_candidates.append(
            {
                "documentScope": "planning",
                "district": district_canonical,
            }
        )

    if force_planning_document_id is not None:
        if plan_year is not None:
            raw_base_candidates.append(
                {
                    "documentScope": "planning",
                    "planningDocumentId": int(force_planning_document_id),
                    "planYear": plan_year,
                }
            )
        raw_base_candidates.append(
            {
                "documentScope": "planning",
                "planningDocumentId": int(force_planning_document_id),
            }
        )

    if plan_year is not None:
        raw_base_candidates.append({"documentScope": "planning", "planYear": plan_year})
    raw_base_candidates.append({"documentScope": "planning"})

    base_candidates: list[dict[str, Any]] = []
    seen_base_filters: set[str] = set()
    for candidate in raw_base_candidates:
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if key in seen_base_filters:
            continue
        seen_base_filters.add(key)
        base_candidates.append(candidate)

    if debug_enabled:
        _planning_debug_log("base_filter_candidates", base_candidates)

    best_fallback: list[Document] = []
    best_fallback_score = float("-inf")

    def _fallback_doc_score(doc: Document) -> float:
        return _planning_doc_score(doc, message, district, plan_year)

    for base_filter in base_candidates:
        if debug_enabled:
            _planning_debug_log("base_filter_start", base_filter)

        planning_retriever = build_retriever(
            planning_vs,
            k=probe_k,
            filters={"chunkTypes": chunk_types},
            base_filter=base_filter,
        )

        docs_by_identity: dict[str, Document] = {}
        vector_rrf_boost: dict[str, float] = {}
        lexical_rrf_boost: dict[str, float] = {}

        for query_text in query_candidates:
            cache_key = _planning_query_cache_key(
                query_text,
                base_filter,
                probe_k,
                lexical_k,
                True,
            )
            cached_result = _planning_query_cache_get(cache_key)
            used_cache = cached_result is not None

            if cached_result is not None:
                vector_docs, lexical_docs = cached_result
            else:
                vector_task = asyncio.create_task(planning_retriever.ainvoke(query_text))
                lexical_task = asyncio.create_task(
                    lexical_search_documents(
                        planning_vs,
                        query_text,
                        k=lexical_k,
                        filters={"chunkTypes": chunk_types},
                        base_filter=base_filter,
                    )
                )

                vector_docs = await vector_task
                try:
                    lexical_docs = await lexical_task
                except Exception:
                    lexical_docs = []

                _planning_query_cache_put(cache_key, (vector_docs, lexical_docs))

            if debug_enabled:
                _planning_debug_log(
                    "candidate_query_hits",
                    {
                        "baseFilter": base_filter,
                        "query": query_text,
                        "cache": used_cache,
                        "vectorCount": len(vector_docs),
                        "lexicalCount": len(lexical_docs),
                        "vectorTop": _planning_debug_doc_list(vector_docs, limit=4),
                        "lexicalTop": _planning_debug_doc_list(lexical_docs, limit=4),
                    },
                )

            for rank, doc in enumerate(vector_docs):
                identity = _planning_doc_identity(doc)
                vector_rrf_boost[identity] = vector_rrf_boost.get(identity, 0.0) + _rrf_score(rank)
                if identity not in docs_by_identity:
                    docs_by_identity[identity] = doc

            for rank, doc in enumerate(lexical_docs):
                identity = _planning_doc_identity(doc)
                lexical_rrf_boost[identity] = lexical_rrf_boost.get(identity, 0.0) + _rrf_score(rank)
                if identity not in docs_by_identity:
                    docs_by_identity[identity] = doc

            if len(docs_by_identity) >= probe_k * 2:
                break

        docs = list(docs_by_identity.values())
        if not docs:
            if debug_enabled:
                _planning_debug_log("base_filter_empty", {"baseFilter": base_filter})
            continue

        deduped = _dedupe_planning_docs(docs)
        ranked_with_scores: list[tuple[Document, float, float, float, float, float, str]] = []
        for d in deduped:
            identity = _planning_doc_identity(d)
            planning_score = _planning_doc_score(d, message, district, plan_year)
            vector_rrf_score = vector_rrf_boost.get(identity, 0.0) * 18.0
            lexical_rrf_score = lexical_rrf_boost.get(identity, 0.0) * 42.0
            rrf_score = vector_rrf_score + lexical_rrf_score
            total_score = planning_score + rrf_score
            ranked_with_scores.append((d, total_score, planning_score, rrf_score, lexical_rrf_score, vector_rrf_score, identity))

        ranked_with_scores.sort(key=lambda item: item[1], reverse=True)
        ranked = [item[0] for item in ranked_with_scores]

        if debug_enabled:
            top_ranked: list[dict[str, Any]] = []
            for rank, (doc, total_score, planning_score, rrf_score, lexical_rrf_score, vector_rrf_score, identity) in enumerate(
                ranked_with_scores[: min(24, len(ranked_with_scores))],
                start=1,
            ):
                fields = _planning_doc_debug_fields(doc)
                fields.update(
                    {
                        "rank": rank,
                        "identity": identity,
                        "score": round(total_score, 6),
                        "planningScore": round(planning_score, 6),
                        "rrfScore": round(rrf_score, 6),
                        "lexicalRrfScore": round(lexical_rrf_score, 6),
                        "vectorRrfScore": round(vector_rrf_score, 6),
                        "districtMatch": _doc_matches_district(doc, district),
                        "yearMatch": _doc_matches_plan_year(doc, plan_year),
                        "tocLike": _is_planning_toc_like_chunk(doc.page_content or "", _planning_doc_haystack(doc)),
                    }
                )
                top_ranked.append(fields)

            _planning_debug_log(
                "base_filter_ranked",
                {
                    "baseFilter": base_filter,
                    "rawUniqueCount": len(docs),
                    "dedupedCount": len(deduped),
                    "rankedCount": len(ranked),
                    "topRanked": top_ranked,
                },
            )

        strict = [d for d in ranked if _doc_matches_district(d, district) and _doc_matches_plan_year(d, plan_year)]
        relaxed = [d for d in ranked if _doc_matches_district(d, district)]

        if debug_enabled:
            _planning_debug_log(
                "scope_pool_sizes",
                {
                    "baseFilter": base_filter,
                    "strict": len(strict),
                    "district": len(relaxed),
                    "broad": len(ranked),
                },
            )

        for scope, pool in (("strict", strict), ("district", relaxed), ("broad", ranked)):
            if not pool:
                continue

            if debug_enabled:
                _planning_debug_log(
                    "scope_pool_top",
                    {
                        "scope": scope,
                        "baseFilter": base_filter,
                        "poolSize": len(pool),
                        "poolTop": _planning_debug_doc_list(pool, limit=8),
                    },
                )

            selected = _select_ranked_planning_docs(
                pool,
                message,
                final_k,
                district=district,
                plan_year=plan_year,
            )
            if not selected:
                continue

            if debug_enabled:
                _planning_debug_log(
                    "selected_initial",
                    {
                        "scope": scope,
                        "baseFilter": base_filter,
                        "selectedCount": len(selected),
                        "selected": _planning_debug_doc_list(selected, limit=12),
                    },
                )

            selected = await _augment_planning_intent_evidence(
                planning_vs,
                message,
                district,
                plan_year,
                selected,
                pool,
                final_k,
            )

            selected = await _augment_planning_land_recovery_evidence(
                planning_vs,
                message,
                district,
                plan_year,
                selected,
                pool,
                final_k,
            )

            selected = await _force_planning_specialized_evidence(
                planning_vs,
                message,
                district,
                plan_year,
                selected,
                final_k,
            )

            selected = await _augment_planning_continuation_neighbors(
                planning_vs,
                message,
                selected,
                final_k,
            )

            selected = await _augment_planning_land_change_fact_docs(
                message,
                district,
                plan_year,
                selected,
                final_k,
            )

            if debug_enabled:
                _planning_debug_log(
                    "selected_after_intent_rescue",
                    {
                        "scope": scope,
                        "baseFilter": base_filter,
                        "selectedCount": len(selected),
                        "selected": _planning_debug_doc_list(selected, limit=12),
                    },
                )

            if fact_query and selected:
                selected = await _augment_planning_text_neighbors(
                    planning_vs,
                    message,
                    district,
                    plan_year,
                    selected,
                    final_k,
                )

                if debug_enabled:
                    _planning_debug_log(
                        "selected_after_text_neighbor_augment",
                        {
                            "scope": scope,
                            "baseFilter": base_filter,
                            "selectedCount": len(selected),
                            "selected": _planning_debug_doc_list(selected, limit=12),
                        },
                    )

                if _is_planning_land_change_query(message):
                    selected = await _augment_planning_land_change_fact_docs(
                        message,
                        district,
                        plan_year,
                        selected,
                        final_k,
                    )

            if (recovery_grouping_query or project_listing_query) and selected:
                selected = await _augment_planning_table_neighbors(
                    planning_vs,
                    message,
                    selected,
                    final_k,
                )

            if recovery_grouping_query and selected:
                selected = await _augment_recovery_grouping_with_neighbors(
                    planning_vs,
                    selected,
                    final_k,
                )

            if debug_enabled:
                _planning_debug_log(
                    "selected_after_neighbor_augment",
                    {
                        "scope": scope,
                        "baseFilter": base_filter,
                        "selectedCount": len(selected),
                        "selected": _planning_debug_doc_list(selected, limit=12),
                    },
                )

            selected = _rebalance_planning_chunk_mix(
                selected,
                limit=final_k,
                fact_query=fact_query,
            )

            if debug_enabled:
                _planning_debug_log(
                    "selected_after_mix_rebalance",
                    {
                        "scope": scope,
                        "baseFilter": base_filter,
                        "selectedCount": len(selected),
                        "selected": _planning_debug_doc_list(selected, limit=12),
                    },
                )

            selected = _compact_planning_docs(
                selected,
                message,
                district,
                plan_year,
                fact_query=fact_query,
            )

            selected = _rebalance_planning_chunk_mix(
                selected,
                limit=final_k,
                fact_query=fact_query,
            )

            if debug_enabled:
                compact_snapshot: list[dict[str, Any]] = []
                for rank, doc in enumerate(selected[:12], start=1):
                    fields = _planning_doc_debug_fields(doc)
                    fields.update(
                        {
                            "rank": rank,
                            "contentChars": len(doc.page_content or ""),
                            "contentPreview": _normalize_nl((doc.page_content or ""))[:180],
                        }
                    )
                    compact_snapshot.append(fields)

                _planning_debug_log(
                    "selected_compacted",
                    {
                        "scope": scope,
                        "baseFilter": base_filter,
                        "selectedCount": len(selected),
                        "selected": compact_snapshot,
                    },
                )

            if scope == "strict":
                if len(selected) >= strict_min_docs:
                    print(
                        "[PlanningNL] Retrieved "
                        f"{len(selected)} strict docs (year={plan_year or 'N/A'}) "
                        f"with base_filter={base_filter}"
                    )
                    if debug_enabled:
                        _planning_debug_log(
                            "strict_scope_return",
                            {
                                "baseFilter": base_filter,
                                "selectedCount": len(selected),
                                "minRequired": strict_min_docs,
                            },
                        )
                    return selected

                print(
                    "[PlanningNL] Strict scope too sparse "
                    f"({len(selected)} docs < {strict_min_docs}); trying broader scope "
                    f"(year={plan_year or 'N/A'})"
                )
                continue

            best_fallback, best_fallback_score = choose_better_planning_fallback(
                best_fallback,
                best_fallback_score,
                selected,
                _fallback_doc_score,
            )

            if scope == "district" and len(selected) >= district_min_docs:
                print(
                    "[PlanningNL] Retrieved "
                    f"{len(selected)} district-matched docs (year={plan_year or 'N/A'}) "
                    f"with base_filter={base_filter}"
                )
                if debug_enabled:
                    _planning_debug_log(
                        "district_scope_return",
                        {
                            "baseFilter": base_filter,
                            "selectedCount": len(selected),
                            "minRequired": district_min_docs,
                        },
                    )
                return selected

    if best_fallback:
        compact_fallback = _compact_planning_docs(
            best_fallback,
            message,
            district,
            plan_year,
            fact_query=fact_query,
        )
        compact_fallback = _rebalance_planning_chunk_mix(
            compact_fallback,
            limit=final_k,
            fact_query=fact_query,
        )
        if debug_enabled:
            _planning_debug_log(
                "broad_fallback_return",
                {
                    "fallbackCount": len(compact_fallback),
                    "fallbackScore": best_fallback_score,
                    "fallbackTop": _planning_debug_doc_list(compact_fallback, limit=12),
                },
            )
        print(
            "[PlanningNL] Falling back to broad planning docs "
            f"(year={plan_year or 'N/A'})"
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
    retrieval_message = build_retrieval_query(req.message, history)
    
    llm = build_llm()
    
    filter_query = retrieval_message if _normalize_nl(retrieval_message) != _normalize_nl(req.message) else req.message
    filters = await extract_filters_from_query(filter_query, llm)
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
        planning_docs = await planning_retriever.ainvoke(retrieval_message)
        if planning_docs:
            planning_text = "\n\n".join(d.page_content for d in planning_docs if d.page_content)
            extra_context = f"{extra_context}\n\n=== PLANNING VECTOR CONTEXT ===\n{planning_text}" if extra_context else planning_text
            planning_citations = _build_planning_citations(planning_docs)

    elif use_planning_mode:
        district = _extract_district_from_message(req.message)
        plan_year = _extract_plan_year_from_message(req.message)
        planning_docs = await _retrieve_planning_docs_for_nl_query(
            planning_vs,
            retrieval_message,
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
