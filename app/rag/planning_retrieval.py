from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from langchain_core.documents import Document

from ..planning.features import (
    planning_doc_haystack as _planning_doc_haystack,
    strip_planning_metadata_lines as _strip_planning_metadata_lines,
)
from ..planning.docs import (
    dedupe_planning_docs as _dedupe_planning_docs,
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
from ..planning.query_builders import district_code_fragment as _district_code_fragment
from ..planning.query_builders import planning_query_candidates as _planning_query_candidates_base
from ..planning.selector import select_ranked_planning_docs as _select_ranked_planning_docs_base
from ..utils.text import (
    normalize_vietnamese_search_text as _normalize_nl,
    strip_vietnamese_accents as _strip_accents,
)
from .planning_pipeline import choose_better_planning_fallback
from .retriever import build_retriever

_logger = logging.getLogger(__name__)

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
_PLANNING_QUERY_CACHE_SIZE = 256

_PLANNING_QUERY_CACHE: OrderedDict[str, list[Document]] = OrderedDict()


@dataclass(frozen=True)
class _PlanningRetrievalContext:
    district: str | None
    plan_year: int | None
    fact_query: bool
    final_k: int
    strict_min_docs: int
    district_min_docs: int
    probe_k: int
    query_candidates: list[str]


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


def _compact_planning_doc(
    doc: Document,
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
        compact_body = raw_body[: (2200 if fact_query else 1200)].strip()

    parts = [*header_lines, compact_body] if compact_body else header_lines
    return Document(page_content="\n".join(part for part in parts if part).strip(), metadata=md)


def _compact_planning_docs(
    docs: list[Document],
    *,
    fact_query: bool,
) -> list[Document]:
    out: list[Document] = []
    seen: set[str] = set()

    for doc in docs:
        compacted = _compact_planning_doc(doc, fact_query=fact_query)
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


def _resolve_planning_retrieval_context(
    message: str,
    top_k: int,
    history_messages: Optional[list[dict[str, Any]]],
) -> _PlanningRetrievalContext:
    district = _extract_district_from_message(message) or _extract_district_from_history(history_messages)
    plan_year = _extract_plan_year_from_message(message)
    if plan_year is None:
        plan_year = _extract_plan_year_from_history(history_messages)

    fact_query = _is_planning_fact_query(message)
    final_k = max(4, min(top_k, 16 if fact_query else 12))
    return _PlanningRetrievalContext(
        district=district,
        plan_year=plan_year,
        fact_query=fact_query,
        final_k=final_k,
        strict_min_docs=_planning_scope_min_docs("strict", final_k, fact_query),
        district_min_docs=_planning_scope_min_docs("district", final_k, fact_query),
        probe_k=max(20, min(120, final_k * 8)),
        query_candidates=_planning_query_candidates(message, district, plan_year),
    )


def _planning_base_filter_candidates(
    district: str | None,
    plan_year: int | None,
    force_planning_document_id: Optional[int],
) -> list[dict[str, Any]]:
    district_canonical = canonicalize_planning_district(district, title=district, dossier_code="") if district else None
    raw_candidates: list[dict[str, Any]] = []

    if district_canonical:
        if plan_year is not None:
            district_code = _district_code_fragment(district_canonical)
            if district_code:
                raw_candidates.append(
                    {
                        "documentScope": "planning",
                        "dossierCode": f"HN-{district_code}-KH{plan_year}",
                    }
                )
            raw_candidates.append(
                {
                    "documentScope": "planning",
                    "district": district_canonical,
                    "planYear": plan_year,
                }
            )
        raw_candidates.append(
            {
                "documentScope": "planning",
                "district": district_canonical,
            }
        )

    if force_planning_document_id is not None:
        if plan_year is not None:
            raw_candidates.append(
                {
                    "documentScope": "planning",
                    "planningDocumentId": int(force_planning_document_id),
                    "planYear": plan_year,
                }
            )
        raw_candidates.append(
            {
                "documentScope": "planning",
                "planningDocumentId": int(force_planning_document_id),
            }
        )

    if plan_year is not None:
        raw_candidates.append({"documentScope": "planning", "planYear": plan_year})
    raw_candidates.append({"documentScope": "planning"})

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


async def _rank_planning_docs_for_filter(
    planning_vs,
    base_filter: dict[str, Any],
    query_candidates: list[str],
    probe_k: int,
) -> list[Document]:
    planning_retriever = build_retriever(
        planning_vs,
        k=probe_k,
        filters={"chunkTypes": ["text", "table"]},
        base_filter=base_filter,
        mode_override="vector",
    )

    docs_by_identity: dict[str, Document] = {}
    vector_rrf_boost: dict[str, float] = {}

    for query_text in query_candidates:
        cache_key = _planning_query_cache_key(query_text, base_filter, probe_k)
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

    deduped = _dedupe_planning_docs(list(docs_by_identity.values()))
    return sorted(
        deduped,
        key=lambda doc: vector_rrf_boost.get(_planning_doc_identity(doc), 0.0),
        reverse=True,
    )


def _planning_scope_pools(
    ranked_docs: list[Document],
    district: str | None,
    plan_year: int | None,
) -> tuple[tuple[str, list[Document]], ...]:
    strict = [doc for doc in ranked_docs if _doc_matches_district(doc, district) and _doc_matches_plan_year(doc, plan_year)]
    district_matched = [doc for doc in ranked_docs if _doc_matches_district(doc, district)]
    return (
        ("strict", strict),
        ("district", district_matched),
        ("broad", ranked_docs),
    )


async def _retrieve_planning_docs_for_nl_query(
    planning_vs,
    message: str,
    top_k: int,
    history_messages: Optional[list[dict[str, Any]]] = None,
    force_planning_document_id: Optional[int] = None,
) -> list[Document]:
    ctx = _resolve_planning_retrieval_context(message, top_k, history_messages)
    base_candidates = _planning_base_filter_candidates(
        ctx.district,
        ctx.plan_year,
        force_planning_document_id,
    )
    best_fallback: list[Document] = []

    for base_filter in base_candidates:
        ranked = await _rank_planning_docs_for_filter(
            planning_vs,
            base_filter,
            ctx.query_candidates,
            ctx.probe_k,
        )
        if not ranked:
            continue

        for scope, pool in _planning_scope_pools(ranked, ctx.district, ctx.plan_year):
            if not pool:
                continue

            selected = _select_ranked_planning_docs(
                pool,
                message,
                ctx.final_k,
                district=ctx.district,
                plan_year=ctx.plan_year,
            )
            if not selected:
                continue

            selected = _compact_planning_docs(
                selected,
                fact_query=ctx.fact_query,
            )

            if scope == "strict":
                if len(selected) >= ctx.strict_min_docs:
                    _logger.info(
                        "Planning NL retrieved %s strict docs (year=%s) with base_filter=%s",
                        len(selected),
                        ctx.plan_year or "N/A",
                        base_filter,
                    )
                    return selected

                _logger.info(
                    "Planning NL strict scope too sparse (%s docs < %s); trying broader scope (year=%s)",
                    len(selected),
                    ctx.strict_min_docs,
                    ctx.plan_year or "N/A",
                )
                continue

            best_fallback = choose_better_planning_fallback(
                best_fallback,
                selected,
            )

            if scope == "district" and len(selected) >= ctx.district_min_docs:
                _logger.info(
                    "Planning NL retrieved %s district-matched docs (year=%s) with base_filter=%s",
                    len(selected),
                    ctx.plan_year or "N/A",
                    base_filter,
                )
                return selected

    if best_fallback:
        _logger.info("Planning NL falling back to broad planning docs (year=%s)", ctx.plan_year or "N/A")
        return best_fallback[: ctx.final_k]

    _logger.info("Planning NL found no documents")
    return []
