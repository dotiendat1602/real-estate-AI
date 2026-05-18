from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional
import re
from collections import OrderedDict
from functools import lru_cache
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document

from ..rag.llm import build_llm
from ..rag.citation_utils import (
    build_planning_citations as _build_planning_citations,
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
from ..rag.planning_pipeline import choose_better_planning_fallback
from ..planning.features import (
    planning_doc_haystack as _planning_doc_haystack,
    planning_has_explicit_project_row as _planning_has_explicit_project_row,
    planning_land_change_label_hits as _planning_land_change_label_hits,
    planning_registered_plan_evidence_hits as _planning_registered_plan_evidence_hits,
    strip_planning_metadata_lines as _strip_planning_metadata_lines,
)
from ..planning.docs import (
    dedupe_planning_docs as _dedupe_planning_docs,
    planning_chunk_type as _planning_chunk_type,
    planning_doc_identity as _planning_doc_identity,
)
from ..planning.metadata import PLANNING_DISTRICT_ALIASES, canonicalize_planning_district
from ..planning.profiles import (
    PLANNING_KEYWORDS as _PLANNING_KEYWORDS,
    PLANNING_LAND_ADMIN_TERMS as _PLANNING_LAND_ADMIN_TERMS,
    PLANNING_REASON_CONTEXT_TERMS as _PLANNING_REASON_CONTEXT_TERMS,
    PLANNING_STRUCTURAL_TERMS as _PLANNING_STRUCTURAL_TERMS,
    build_planning_query_profile,
)
from ..planning.query_builders import planning_query_candidates as _planning_query_candidates_base
from ..planning.query_builders import district_code_fragment as _district_code_fragment
from ..planning.ranker import (
    planning_query_terms as _planning_query_terms,
)
from ..planning.selector import select_ranked_planning_docs as _select_ranked_planning_docs_base
from ..rag.message_history import MessageHistoryManager
from ..rag.static_retriever import StaticDocumentsRetriever as _StaticDocumentsRetriever
from ..utils.text import (
    normalize_vietnamese_search_text as _normalize_nl,
    strip_vietnamese_accents as _strip_accents,
)
from ..db.pgvector import get_db

router = APIRouter()
_logger = logging.getLogger(__name__)

async def _with_timeout(coro, *, seconds: float, label: str, fallback=None):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        _logger.warning("%s exceeded %.1fs; using fallback.", label, seconds)
        return fallback


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

_DISTRICT_MATCH_ALIASES: dict[str, tuple[str, ...]] = PLANNING_DISTRICT_ALIASES

_WARD_DISTRICT_HINTS: dict[str, tuple[str, ...]] = {
    "Hoàng Mai": ("yen so",),
    "Cầu Giấy": ("mai dich",),
    "Hoàn Kiếm": ("chuong duong",),
}


_PLANNING_MAX_QUERY_CANDIDATES = 12
_PLANNING_MAX_FACT_SUBQUERIES = 8
_PLANNING_QUERY_CACHE_SIZE = 256

_PLANNING_QUERY_CACHE: OrderedDict[str, list[Document]] = OrderedDict()

def _has_planning_intent(message: str) -> bool:
    normalized = _normalize_nl(message)
    if any(_normalize_nl(keyword) in normalized for keyword in _PLANNING_KEYWORDS):
        return True

    has_district_hint = _extract_district_from_message(message) is not None
    has_plan_year = _extract_plan_year_from_message(message) is not None
    has_admin_unit_hint = bool(re.search(r"\b(phuong|xa|thi tran)\b", normalized))

    strong_land_admin_terms = tuple(term for term in _PLANNING_LAND_ADMIN_TERMS if term not in {"cong trinh", "du an"})
    has_land_admin_term = any(term in normalized for term in strong_land_admin_terms)
    if has_land_admin_term and (has_district_hint or has_plan_year or has_admin_unit_hint):
        return True

    has_project_term = any(term in normalized for term in ("cong trinh", "du an"))
    if has_project_term and has_plan_year and (has_district_hint or has_admin_unit_hint):
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

    ranked_lines: list[tuple[tuple[int, int, int, int, int, int, int], int, str]] = []
    for idx, line in enumerate(lines):
        normalized = _normalize_nl(line)
        key = (
            1 if idx < 3 else 0,
            sum(1 for term in query_terms if term in normalized) if query_terms else 0,
            sum(1 for alias in district_aliases if alias in normalized) if district_aliases else 0,
            1 if year_token and year_token in normalized else 0,
            1 if re.search(r"\b\d+(?:[\.,]\d+)?\b", normalized) else 0,
            1 if _planning_has_explicit_project_row(normalized) else 0,
            _planning_registered_plan_evidence_hits(normalized) + _planning_land_change_label_hits(normalized),
        )
        ranked_lines.append((key, idx, line))

    ranked_lines.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    chosen = sorted(ranked_lines[: min(max_lines, len(ranked_lines))], key=lambda item: item[1])
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
) -> str:
    payload = {
        "query": _normalize_nl(query_text),
        "base": base_filter,
        "probe_k": probe_k,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _planning_query_cache_get(key: str) -> list[Document] | None:
    cached = _PLANNING_QUERY_CACHE.get(key)
    if cached is None:
        return None
    _PLANNING_QUERY_CACHE.move_to_end(key)
    return cached


def _planning_query_cache_put(key: str, value: list[Document]) -> None:
    _PLANNING_QUERY_CACHE[key] = value
    _PLANNING_QUERY_CACHE.move_to_end(key)
    while len(_PLANNING_QUERY_CACHE) > _PLANNING_QUERY_CACHE_SIZE:
        _PLANNING_QUERY_CACHE.popitem(last=False)


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
    final_k = max(4, min(top_k, 16 if fact_query else 12))
    strict_min_docs = _planning_scope_min_docs("strict", final_k, fact_query)
    district_min_docs = _planning_scope_min_docs("district", final_k, fact_query)
    probe_k = max(20, min(120, final_k * 8))
    chunk_types = ["text", "table"]

    query_candidates = _planning_query_candidates(message, district, plan_year)

    district_canonical = canonicalize_planning_district(district, title=district, dossier_code="") if district else None
    raw_base_candidates: list[dict[str, Any]] = []

    if district_canonical:
        if plan_year is not None:
            district_code = _district_code_fragment(district_canonical)
            if district_code:
                raw_base_candidates.append(
                    {
                        "documentScope": "planning",
                        "dossierCode": f"HN-{district_code}-KH{plan_year}",
                    }
                )
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

    best_fallback: list[Document] = []

    for base_filter in base_candidates:
        planning_retriever = build_retriever(
            planning_vs,
            k=probe_k,
            filters={"chunkTypes": chunk_types},
            base_filter=base_filter,
            mode_override="vector",
        )

        docs_by_identity: dict[str, Document] = {}
        vector_rrf_boost: dict[str, float] = {}

        for query_text in query_candidates:
            cache_key = _planning_query_cache_key(
                query_text,
                base_filter,
                probe_k,
            )
            cached_result = _planning_query_cache_get(cache_key)

            if cached_result is not None:
                vector_docs = cached_result
            else:
                vector_docs = await planning_retriever.ainvoke(query_text)
                _planning_query_cache_put(cache_key, vector_docs)

            for rank, doc in enumerate(vector_docs):
                identity = _planning_doc_identity(doc)
                vector_rrf_boost[identity] = vector_rrf_boost.get(identity, 0.0) + _rrf_score(rank)
                if identity not in docs_by_identity:
                    docs_by_identity[identity] = doc

            if len(docs_by_identity) >= probe_k * 2:
                break

        docs = list(docs_by_identity.values())
        if not docs:
            continue

        deduped = _dedupe_planning_docs(docs)
        ranked_with_scores: list[tuple[Document, float, str]] = []
        for d in deduped:
            identity = _planning_doc_identity(d)
            vector_rrf_score = vector_rrf_boost.get(identity, 0.0)
            ranked_with_scores.append((d, vector_rrf_score, identity))

        ranked_with_scores.sort(key=lambda item: item[1], reverse=True)
        ranked = [item[0] for item in ranked_with_scores]

        strict = [d for d in ranked if _doc_matches_district(d, district) and _doc_matches_plan_year(d, plan_year)]
        relaxed = [d for d in ranked if _doc_matches_district(d, district)]

        for scope, pool in (("strict", strict), ("district", relaxed), ("broad", ranked)):
            if not pool:
                continue

            selected = _select_ranked_planning_docs(
                pool,
                message,
                final_k,
                district=district,
                plan_year=plan_year,
            )
            if not selected:
                continue

            selected = _rebalance_planning_chunk_mix(
                selected,
                limit=final_k,
                fact_query=fact_query,
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

            if scope == "strict":
                if len(selected) >= strict_min_docs:
                    _logger.info(
                        "Planning NL retrieved %s strict docs (year=%s) with base_filter=%s",
                        len(selected),
                        plan_year or "N/A",
                        base_filter,
                    )
                    return selected

                _logger.info(
                    "Planning NL strict scope too sparse (%s docs < %s); trying broader scope (year=%s)",
                    len(selected),
                    strict_min_docs,
                    plan_year or "N/A",
                )
                continue

            best_fallback = choose_better_planning_fallback(
                best_fallback,
                selected,
            )

            if scope == "district" and len(selected) >= district_min_docs:
                _logger.info(
                    "Planning NL retrieved %s district-matched docs (year=%s) with base_filter=%s",
                    len(selected),
                    plan_year or "N/A",
                    base_filter,
                )
                return selected

    if best_fallback:
        compact_fallback = _rebalance_planning_chunk_mix(
            best_fallback,
            limit=final_k,
            fact_query=fact_query,
        )
        _logger.info("Planning NL falling back to broad planning docs (year=%s)", plan_year or "N/A")
        return compact_fallback

    _logger.info("Planning NL found no documents")
    return []

async def initialize_vector_store():
    """Khởi tạo async vector store - gọi từ startup event"""
    listing_vs = await initialize_listing_vector_store()
    await initialize_planning_vector_store()
    return listing_vs

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
    topK: int = Field(default=int(os.getenv("TOP_K_DEFAULT", "16")), ge=1, le=50)
    planningContexts: list[PlanningContext] = Field(default_factory=list)

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
    
    llm = build_llm()

    use_planning_mode = bool(req.planningContexts) or _has_planning_intent(req.message)
    filters: dict[str, Any] = {}
    filter_token_usage: dict[str, Any] = {}
    if not use_planning_mode:
        filter_query = retrieval_message if _normalize_nl(retrieval_message) != _normalize_nl(req.message) else req.message
        filters, filter_token_usage = await extract_filters_from_query_with_usage(filter_query, llm)
        _logger.debug("Extracted listing filters: %s", filters)

    extra_context = ""
    planning_citations: list[dict[str, Any]] = []
    planning_docs: list[Document] = []
    planning_vs = None

    if use_planning_mode:
        planning_vs = await initialize_planning_vector_store()
        retriever = _StaticDocumentsRetriever([])
    else:
        vs = await initialize_listing_vector_store()
        retriever = ListingFallbackRetriever(
            build_retriever(vs, k=req.topK, filters=filters),
            query=req.message,
            filters=filters,
            k=req.topK,
        )

    if req.planningContexts:
        if planning_vs is None:
            planning_vs = await initialize_planning_vector_store()
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
            k=max(6, min(req.topK, 16)),
            filters={"chunkTypes": ["text", "table"]},
            base_filter={
                "documentScope": "planning",
                "propertyId": {"$in": planning_property_ids},
            },
            mode_override="vector",
        )
        planning_docs = await _with_timeout(
            planning_retriever.ainvoke(retrieval_message),
            seconds=float(os.getenv("CHAT_PLANNING_RETRIEVAL_TIMEOUT_SECONDS", "25")),
            label="planning context retrieval",
            fallback=[],
        )
        if planning_docs:
            planning_text = "\n\n".join(d.page_content for d in planning_docs if d.page_content)
            extra_context = f"{extra_context}\n\n=== PLANNING VECTOR CONTEXT ===\n{planning_text}" if extra_context else planning_text
            planning_citations = _build_planning_citations(planning_docs)

    elif use_planning_mode:
        district = _extract_district_from_message(req.message)
        plan_year = _extract_plan_year_from_message(req.message)
        planning_docs = await _with_timeout(
            _retrieve_planning_docs_for_nl_query(
                planning_vs,
                retrieval_message,
                req.topK,
                history_messages=history,
            ),
            seconds=float(os.getenv("CHAT_PLANNING_RETRIEVAL_TIMEOUT_SECONDS", "25")),
            label="planning NL retrieval",
            fallback=[],
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

    if use_planning_mode:
        retriever = _StaticDocumentsRetriever(planning_docs)

    chain = RagChain(llm=llm, retriever=retriever)
    result = await chain.run(req.message, history=history, extra_context=extra_context)
    merged_citations = result.citations + planning_citations
    reranked_citations = _rerank_citations(req.message, merged_citations, req.planningContexts)
    
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
