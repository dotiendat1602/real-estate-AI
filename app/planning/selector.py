from __future__ import annotations

from typing import Callable, Optional

from langchain_core.documents import Document

from .features import planning_is_heading_or_incomplete_chunk, planning_is_toc_like_chunk
from .profiles import build_planning_query_profile

DistrictMatcher = Callable[[Document, Optional[str]], bool]
YearMatcher = Callable[[Document, Optional[int]], bool]


def select_ranked_planning_docs(
    ranked_docs: list[Document],
    message: str,
    limit: int,
    district: Optional[str] = None,
    plan_year: Optional[int] = None,
    *,
    doc_matches_district: DistrictMatcher,
    doc_matches_plan_year: YearMatcher,
) -> list[Document]:
    """Select from an already ranked planning pool without adding another score layer."""
    if not ranked_docs or limit <= 0:
        return []

    profile = build_planning_query_profile(message, planning_intent=True)
    prefer_text_evidence = bool(
        profile.explanatory_query
        or profile.focus_area_reason
        or profile.sector_land_demand
        or profile.drainage_transport
        or profile.project_delay_reason
    )

    def _identity(doc: Document) -> str:
        md = doc.metadata or {}
        chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
        return "|".join(
            [
                str(md.get("planningDocumentId") or ""),
                str(md.get("chunkType") or ""),
                str(chunk_idx or ""),
            ]
        )

    def _chunk_type(doc: Document) -> str:
        return str((doc.metadata or {}).get("chunkType") or "").lower().strip()

    def _scope_key(item: tuple[int, Document]) -> tuple[bool, bool, bool, int]:
        index, doc = item
        district_mismatch = district is not None and not doc_matches_district(doc, district)
        year_mismatch = plan_year is not None and not doc_matches_plan_year(doc, plan_year)
        weak_chunk = planning_is_heading_or_incomplete_chunk(doc)
        return (district_mismatch, year_mismatch, weak_chunk, index)

    ordered: list[Document] = []
    seen: set[str] = set()
    for _, doc in sorted(enumerate(ranked_docs), key=_scope_key):
        if planning_is_toc_like_chunk(doc.page_content or ""):
            continue
        key = _identity(doc)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(doc)

    if not ordered:
        return []

    has_table = any(_chunk_type(doc) == "table" for doc in ordered)
    if not has_table:
        return ordered[:limit]

    if limit == 1:
        table_target = 1
    else:
        table_target = min(max(1, limit // 2), limit - 1)
    if prefer_text_evidence:
        table_target = min(table_target, max(1, limit // 4))
    text_target = max(0, limit - table_target)

    selected: list[Document] = []
    selected_keys: set[str] = set()

    def _push(doc: Document) -> bool:
        key = _identity(doc)
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(doc)
        return True

    table_count = 0
    text_count = 0
    for doc in ordered:
        if len(selected) >= limit:
            break
        chunk_type = _chunk_type(doc)
        if chunk_type == "table" and table_count < table_target:
            if _push(doc):
                table_count += 1
        elif chunk_type == "text" and text_count < text_target:
            if _push(doc):
                text_count += 1

    for doc in ordered:
        if len(selected) >= limit:
            break
        _push(doc)

    return selected
