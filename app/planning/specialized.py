from __future__ import annotations

import asyncio
import re
from typing import Callable, Optional

from langchain_core.documents import Document

from ..rag.retriever import lexical_search_documents
from .docs import planning_doc_identity, planning_doc_pid_idx
from .features import (
    planning_admin_unit_header_hits,
    planning_continuation_signal,
    planning_doc_content_norm,
    planning_doc_haystack,
    planning_has_admin_unit_evidence,
    planning_has_direct_admin_unit_count_phrase,
    planning_has_direct_natural_area_phrase,
    planning_has_explicit_project_row,
    planning_has_land_pair_evidence,
    planning_has_natural_area_admin_evidence,
    planning_has_registered_resolution_count_evidence,
    planning_has_unused_zero_evidence,
    planning_is_heading_or_incomplete_chunk,
    planning_is_toc_like_chunk,
    planning_land_change_label_hits,
    planning_registered_plan_evidence_hits,
    strip_planning_metadata_lines,
)
from .profiles import (
    PLANNING_IMPLEMENTATION_CARRY_FORWARD_MARKERS,
    build_planning_query_profile,
    planning_marker_hits,
)
from .query_builders import planning_intent_rescue_queries, planning_specialized_limit
from .ranker import planning_rescue_query_score

PlanningDocScoreFn = Callable[[Document, str, Optional[str], Optional[int]], float]
PlanningSpecializedEvidenceScoreFn = Callable[[Document, str, Optional[str], Optional[int]], float | None]
LoadPlanningDocsFn = Callable[..., list[Document]]
LoadAdminOverviewDocsFn = Callable[[int, Optional[int], int], object]


async def force_planning_specialized_evidence(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    limit: int,
    *,
    planning_doc_score: PlanningDocScoreFn,
    planning_specialized_evidence_score: PlanningSpecializedEvidenceScoreFn,
    load_planning_document_docs: LoadPlanningDocsFn,
    load_admin_overview_sql_rescue_docs: LoadAdminOverviewDocsFn,
) -> list[Document]:
    if not selected_docs or limit <= 0:
        return selected_docs

    profile = build_planning_query_profile(message, planning_intent=True)
    specialized_limit = planning_specialized_limit(message)
    if specialized_limit <= 0:
        return selected_docs

    candidate_pids: list[int] = []
    for doc in selected_docs:
        planning_document_id, _ = planning_doc_pid_idx(doc)
        if planning_document_id is None or planning_document_id in candidate_pids:
            continue
        candidate_pids.append(planning_document_id)
        if len(candidate_pids) >= 2:
            break

    if not candidate_pids:
        return selected_docs

    rescue_queries = planning_intent_rescue_queries(message, district, plan_year)
    scored_by_identity: dict[str, tuple[float, Document]] = {}
    for doc in selected_docs:
        identity = planning_doc_identity(doc)
        specialized_score = planning_specialized_evidence_score(doc, message, district, plan_year)
        if specialized_score is None:
            continue
        scored_by_identity[identity] = (specialized_score, doc)

    if profile.land_recovery:
        for doc in selected_docs:
            if not planning_is_heading_or_incomplete_chunk(doc, project_listing_query=profile.project_listing):
                continue
            planning_document_id, chunk_index = planning_doc_pid_idx(doc)
            if planning_document_id is None or chunk_index is None:
                continue
            neighbor_index = chunk_index + 1
            try:
                neighbor_docs = await lexical_search_documents(
                    planning_vs,
                    query="",
                    k=2,
                    filters={"chunkTypes": ["text"], "globalChunkIndex": {"$in": [neighbor_index]}},
                    base_filter={"documentScope": "planning", "planningDocumentId": planning_document_id},
                    allow_empty_terms=True,
                )
            except Exception:
                continue

            for neighbor in neighbor_docs:
                identity = planning_doc_identity(neighbor)
                specialized_score = planning_specialized_evidence_score(neighbor, message, district, plan_year)
                if specialized_score is None:
                    haystack = planning_doc_haystack(neighbor)
                    area_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?\s*ha\b", haystack))
                    if area_hits < 2 or re.search(r"\bchuyen\s+(?:muc\s+dich|doi)\b", haystack):
                        continue
                    specialized_score = planning_doc_score(neighbor, message, district, plan_year) + 3.2
                previous = scored_by_identity.get(identity)
                if previous is None or specialized_score > previous[0]:
                    scored_by_identity[identity] = (specialized_score + 2.2, neighbor)

    if profile.city_level_listing:
        for doc in selected_docs:
            haystack = planning_doc_haystack(doc)
            if not any(marker in haystack for marker in ("44 yet kieu", "ga c10", "ga s12")):
                continue
            planning_document_id, chunk_index = planning_doc_pid_idx(doc)
            if planning_document_id is None or chunk_index is None or chunk_index <= 0:
                continue
            try:
                neighbor_docs = await lexical_search_documents(
                    planning_vs,
                    query="",
                    k=2,
                    filters={"chunkTypes": ["text"], "globalChunkIndex": {"$in": [chunk_index - 1]}},
                    base_filter={"documentScope": "planning", "planningDocumentId": planning_document_id},
                    allow_empty_terms=True,
                )
            except Exception:
                continue

            for neighbor in neighbor_docs:
                haystack = planning_doc_haystack(neighbor)
                if "cap thanh pho" not in haystack:
                    continue
                identity = planning_doc_identity(neighbor)
                score = planning_doc_score(neighbor, message, district, plan_year) + 4.2
                previous = scored_by_identity.get(identity)
                if previous is None or score > previous[0]:
                    scored_by_identity[identity] = (score, neighbor)

    if profile.admin_overview:
        admin_queries = [
            "co dien tich tu nhien gom don vi hanh chinh",
            "co dien tich tu nhien",
            "don vi hanh chinh cap phuong cap xa",
            "tong dien tich tu nhien don vi hanh chinh",
        ]
        for planning_document_id in candidate_pids:
            base_filter = {"documentScope": "planning", "planningDocumentId": planning_document_id}
            if plan_year is not None:
                base_filter["planYear"] = plan_year
            for query_text in admin_queries:
                try:
                    candidates = await lexical_search_documents(
                        planning_vs,
                        query=query_text,
                        k=8,
                        filters={"chunkTypes": ["text", "table"]},
                        base_filter=base_filter,
                    )
                except Exception:
                    continue

                for candidate in candidates:
                    specialized_score = planning_specialized_evidence_score(candidate, message, district, plan_year)
                    if specialized_score is None:
                        continue
                    identity = planning_doc_identity(candidate)
                    total_score = specialized_score + planning_rescue_query_score(candidate, query_text) + 1.4
                    previous = scored_by_identity.get(identity)
                    if previous is None or total_score > previous[0]:
                        scored_by_identity[identity] = (total_score, candidate)

            sql_candidates = await load_admin_overview_sql_rescue_docs(planning_document_id, plan_year, 6)
            for candidate in sql_candidates:
                specialized_score = planning_specialized_evidence_score(candidate, message, district, plan_year)
                if specialized_score is None:
                    continue
                identity = planning_doc_identity(candidate)
                total_score = specialized_score + 3.0
                previous = scored_by_identity.get(identity)
                if previous is None or total_score > previous[0]:
                    scored_by_identity[identity] = (total_score, candidate)

        for doc in selected_docs:
            planning_document_id, chunk_index = planning_doc_pid_idx(doc)
            if planning_document_id is None or chunk_index is None:
                continue
            candidate_indices = [idx for idx in (chunk_index - 2, chunk_index - 1, chunk_index + 1, chunk_index + 2) if idx >= 0]
            if not candidate_indices:
                continue
            try:
                neighbor_docs = await lexical_search_documents(
                    planning_vs,
                    query="",
                    k=max(4, len(candidate_indices)),
                    filters={"chunkTypes": ["text", "table"], "globalChunkIndex": {"$in": candidate_indices}},
                    base_filter={"documentScope": "planning", "planningDocumentId": planning_document_id},
                    allow_empty_terms=True,
                )
            except Exception:
                continue

            for neighbor in neighbor_docs:
                haystack = planning_doc_haystack(neighbor)
                if not (
                    any(marker in haystack for marker in ("dien tich tu nhien", "tong dien tich tu nhien", "co dien tich tu nhien"))
                    or planning_has_admin_unit_evidence(haystack)
                ):
                    continue
                identity = planning_doc_identity(neighbor)
                score = planning_doc_score(neighbor, message, district, plan_year) + 4.0
                if planning_has_natural_area_admin_evidence(haystack):
                    score += 2.4
                previous = scored_by_identity.get(identity)
                if previous is None or score > previous[0]:
                    scored_by_identity[identity] = (score, neighbor)

    if profile.registered_plan_composition:
        registered_queries = [
            "danh muc cac cong trinh du an dua vao ke hoach su dung dat la",
            "danh muc cac cong trinh du an thu hoi dat nam 2025",
            "danh muc cac du an chuyen muc dich dat trong lua nghi quyet",
            "nghi quyet hdnd thong qua la du an voi dien tich",
        ]
        for planning_document_id in candidate_pids:
            base_filter = {"documentScope": "planning", "planningDocumentId": planning_document_id}
            if plan_year is not None:
                base_filter["planYear"] = plan_year
            for query_text in registered_queries:
                try:
                    candidates = await lexical_search_documents(
                        planning_vs,
                        query=query_text,
                        k=8,
                        filters={"chunkTypes": ["text", "table"]},
                        base_filter=base_filter,
                    )
                except Exception:
                    continue

                for candidate in candidates:
                    haystack = planning_doc_haystack(candidate)
                    if not (
                        planning_registered_plan_evidence_hits(haystack) > 0
                        or "dua vao ke hoach su dung dat" in haystack
                        or "nghi quyet" in haystack
                        or planning_has_registered_resolution_count_evidence(haystack)
                    ):
                        continue
                    specialized_score = planning_specialized_evidence_score(candidate, message, district, plan_year)
                    if specialized_score is None:
                        continue
                    identity = planning_doc_identity(candidate)
                    total_score = specialized_score + planning_rescue_query_score(candidate, query_text) + 2.2
                    previous = scored_by_identity.get(identity)
                    if previous is None or total_score > previous[0]:
                        scored_by_identity[identity] = (total_score, candidate)

            sql_candidates = await asyncio.to_thread(
                load_planning_document_docs,
                planning_document_id,
                plan_year,
                chunk_types=("text", "table"),
                limit=2500,
            )
            for candidate in sql_candidates:
                haystack = planning_doc_haystack(candidate)
                if planning_is_toc_like_chunk(candidate.page_content or "", haystack):
                    continue
                continuation_score = planning_continuation_signal(
                    candidate,
                    registered_plan_composition_query=profile.registered_plan_composition,
                    land_change_query=profile.land_change,
                )
                has_resolution_count = planning_has_registered_resolution_count_evidence(haystack)
                if continuation_score <= 0.0 and planning_registered_plan_evidence_hits(haystack) == 0 and not has_resolution_count:
                    continue
                if continuation_score <= 0.0 and not any(
                    marker in haystack for marker in ("dua vao ke hoach su dung dat", "nghi quyet", "thu hoi dat nam 2025")
                ) and not has_resolution_count:
                    continue
                if has_resolution_count and not any(marker in haystack for marker in ("nghi quyet", "hdnd", "hoi dong nhan dan")):
                    continue
                identity = planning_doc_identity(candidate)
                total_score = planning_doc_score(candidate, message, district, plan_year) + continuation_score * 3.2
                total_score += planning_rescue_query_score(candidate, message) + 1.6
                if has_resolution_count:
                    total_score += 3.0
                previous = scored_by_identity.get(identity)
                if previous is None or total_score > previous[0]:
                    scored_by_identity[identity] = (total_score, candidate)

    if profile.land_change:
        land_change_queries = [
            "hien trang dat nong nghiep nam 2024 den nam 2025",
            "dat phi nong nghiep nam 2024 den nam 2025",
            "dat chua su dung nam 2024 den nam 2025",
            "2025 2024 tang giam dat nong nghiep dat phi nong nghiep",
        ]
        for planning_document_id in candidate_pids:
            base_filter = {"documentScope": "planning", "planningDocumentId": planning_document_id}
            if plan_year is not None:
                base_filter["planYear"] = plan_year
            for query_text in land_change_queries:
                try:
                    candidates = await lexical_search_documents(
                        planning_vs,
                        query=query_text,
                        k=8,
                        filters={"chunkTypes": ["text", "table"]},
                        base_filter=base_filter,
                    )
                except Exception:
                    continue

                for candidate in candidates:
                    raw_content = strip_planning_metadata_lines(candidate.page_content or "")
                    content_norm = planning_doc_content_norm(candidate)
                    land_change_hits = planning_land_change_label_hits(content_norm)
                    has_year_pair = "2024" in content_norm and "2025" in content_norm
                    has_unused_zero = planning_has_unused_zero_evidence(content_norm)
                    has_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat nong nghiep")
                    has_non_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat phi nong nghiep")
                    if land_change_hits == 0 and not has_unused_zero and not (has_agri_pair or has_non_agri_pair):
                        continue
                    if not has_year_pair and land_change_hits < 2 and not has_unused_zero and not (has_agri_pair or has_non_agri_pair):
                        continue
                    specialized_score = planning_specialized_evidence_score(candidate, message, district, plan_year)
                    if specialized_score is None:
                        specialized_score = planning_doc_score(candidate, message, district, plan_year)
                    identity = planning_doc_identity(candidate)
                    total_score = specialized_score + planning_rescue_query_score(candidate, query_text) + 2.0
                    if not has_year_pair:
                        total_score += 1.4
                    if has_agri_pair or has_non_agri_pair:
                        total_score += 1.8
                    previous = scored_by_identity.get(identity)
                    if previous is None or total_score > previous[0]:
                        scored_by_identity[identity] = (total_score, candidate)

            sql_candidates = await asyncio.to_thread(
                load_planning_document_docs,
                planning_document_id,
                plan_year,
                chunk_types=("text", "table"),
                limit=2500,
            )
            for candidate in sql_candidates:
                haystack = planning_doc_haystack(candidate)
                if planning_is_toc_like_chunk(candidate.page_content or "", haystack):
                    continue
                raw_content = strip_planning_metadata_lines(candidate.page_content or "")
                content_norm = planning_doc_content_norm(candidate)
                land_change_hits = planning_land_change_label_hits(content_norm)
                has_year_pair = "2024" in content_norm and "2025" in content_norm
                has_unused_zero = planning_has_unused_zero_evidence(content_norm)
                has_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat nong nghiep")
                has_non_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat phi nong nghiep")
                if not has_year_pair and land_change_hits < 2 and not has_unused_zero and not (has_agri_pair or has_non_agri_pair):
                    continue
                identity = planning_doc_identity(candidate)
                total_score = planning_doc_score(candidate, message, district, plan_year) + 1.8
                if has_year_pair:
                    total_score += 2.0
                if land_change_hits >= 2:
                    total_score += 1.4
                if has_unused_zero:
                    total_score += 1.8
                if has_agri_pair or has_non_agri_pair:
                    total_score += 1.8
                previous = scored_by_identity.get(identity)
                if previous is None or total_score > previous[0]:
                    scored_by_identity[identity] = (total_score, candidate)

    if profile.registered_plan_composition or profile.project_delay_reason:
        for doc in selected_docs:
            planning_document_id, chunk_index = planning_doc_pid_idx(doc)
            if planning_document_id is None or chunk_index is None:
                continue
            haystack = planning_doc_haystack(doc)
            candidate_indices: list[int] = []
            if profile.registered_plan_composition and (
                planning_is_heading_or_incomplete_chunk(doc, project_listing_query=profile.project_listing)
                or any(marker in haystack for marker in ("tong so cong trinh du an dang ky thuc hien", "danh muc cong trinh du an trong ke hoach su dung dat", "trong do"))
            ):
                candidate_indices.extend([chunk_index + 1, chunk_index + 2])
            if profile.project_delay_reason and any(marker in haystack for marker in ("chua to chuc", "chuyen tiep", "ket qua thuc hien", "chua thuc hien")):
                candidate_indices.extend([chunk_index + 1, chunk_index + 2])
            candidate_indices = [idx for idx in sorted(set(candidate_indices)) if idx >= 0]
            if not candidate_indices:
                continue
            try:
                neighbor_docs = await lexical_search_documents(
                    planning_vs,
                    query="",
                    k=max(4, len(candidate_indices) * 2),
                    filters={"chunkTypes": ["text"], "globalChunkIndex": {"$in": candidate_indices}},
                    base_filter={"documentScope": "planning", "planningDocumentId": planning_document_id},
                    allow_empty_terms=True,
                )
            except Exception:
                continue

            for neighbor in neighbor_docs:
                specialized_score = planning_specialized_evidence_score(neighbor, message, district, plan_year)
                if specialized_score is None:
                    continue
                identity = planning_doc_identity(neighbor)
                total_score = specialized_score + 2.8
                previous = scored_by_identity.get(identity)
                if previous is None or total_score > previous[0]:
                    scored_by_identity[identity] = (total_score, neighbor)

    query_limit = 4 if profile.new_registration_unique else 6
    if profile.registered_plan_composition or profile.project_delay_reason:
        query_limit = max(query_limit, 8)
    fetch_k = max(8, min(16, specialized_limit * 6))
    for planning_document_id in candidate_pids:
        base_filters: list[dict[str, object]] = []
        if plan_year is not None:
            base_filters.append({"documentScope": "planning", "planningDocumentId": planning_document_id, "planYear": plan_year})
        base_filters.append({"documentScope": "planning", "planningDocumentId": planning_document_id})

        for base_filter in base_filters:
            for query_text in rescue_queries[:query_limit]:
                try:
                    candidates = await lexical_search_documents(
                        planning_vs,
                        query=query_text,
                        k=fetch_k,
                        filters={"chunkTypes": ["text", "table"]},
                        base_filter=base_filter,
                    )
                except Exception:
                    continue

                for candidate in candidates:
                    specialized_score = planning_specialized_evidence_score(candidate, message, district, plan_year)
                    if specialized_score is None:
                        continue
                    identity = planning_doc_identity(candidate)
                    total_score = specialized_score + planning_rescue_query_score(candidate, query_text)
                    previous = scored_by_identity.get(identity)
                    if previous is None or total_score > previous[0]:
                        scored_by_identity[identity] = (total_score, candidate)

    if not scored_by_identity:
        return selected_docs

    prioritized = sorted(scored_by_identity.items(), key=lambda item: item[1][0], reverse=True)
    prioritized_docs: list[Document] = []
    prioritized_ids: set[str] = set()

    if profile.article67:
        anchor_doc: Optional[Document] = None
        tang_le_doc: Optional[Document] = None
        bo_cong_an_doc: Optional[Document] = None
        for _, (_, doc) in prioritized:
            haystack = planning_doc_haystack(doc)
            chunk_type = str((doc.metadata or {}).get("chunkType") or "").lower().strip()
            article67_entities = sum(
                1 for marker in ("nha tang le quoc gia", "tran thanh tong", "bo cong an", "tran binh trong", "tran nhan tong") if marker in haystack
            )
            if anchor_doc is None and ("khoan 4 dieu 67" in haystack or "dieu 67" in haystack):
                anchor_doc = doc
            is_project_row = (planning_has_explicit_project_row(haystack) and article67_entities > 0) or (chunk_type == "table" and article67_entities > 0)
            if tang_le_doc is None and is_project_row and any(marker in haystack for marker in ("nha tang le quoc gia", "tran thanh tong")):
                tang_le_doc = doc
            if bo_cong_an_doc is None and is_project_row and any(marker in haystack for marker in ("bo cong an", "tran binh trong", "tran nhan tong")):
                bo_cong_an_doc = doc
            if anchor_doc is not None and tang_le_doc is not None and bo_cong_an_doc is not None:
                break

        for doc in [anchor_doc, tang_le_doc, bo_cong_an_doc]:
            if doc is None:
                continue
            identity = planning_doc_identity(doc)
            if identity in prioritized_ids:
                continue
            prioritized_ids.add(identity)
            prioritized_docs.append(doc)

    if profile.admin_overview:
        direct_natural_area_doc: Optional[Document] = None
        fallback_natural_area_doc: Optional[Document] = None
        direct_count_doc: Optional[Document] = None
        admin_units_doc: Optional[Document] = None
        for _, (_, doc) in prioritized:
            haystack = planning_doc_haystack(doc)
            if direct_natural_area_doc is None and planning_has_direct_natural_area_phrase(haystack):
                direct_natural_area_doc = doc
            if fallback_natural_area_doc is None and any(marker in haystack for marker in ("dien tich tu nhien", "tong dien tich tu nhien", "co dien tich tu nhien")):
                fallback_natural_area_doc = doc
            if direct_count_doc is None and planning_has_direct_admin_unit_count_phrase(haystack):
                direct_count_doc = doc
            if admin_units_doc is None and planning_has_admin_unit_evidence(haystack):
                admin_units_doc = doc
            if direct_natural_area_doc is not None and direct_count_doc is not None and admin_units_doc is not None:
                break

        for doc in [direct_count_doc, direct_natural_area_doc, admin_units_doc, fallback_natural_area_doc]:
            if doc is None:
                continue
            identity = planning_doc_identity(doc)
            if identity in prioritized_ids:
                continue
            prioritized_ids.add(identity)
            prioritized_docs.append(doc)

    if profile.land_change:
        paired_totals_doc: Optional[Document] = None
        agricultural_pair_doc: Optional[Document] = None
        non_agricultural_pair_doc: Optional[Document] = None
        unused_pair_doc: Optional[Document] = None
        current_status_doc: Optional[Document] = None
        unused_status_doc: Optional[Document] = None
        for _, (_, doc) in prioritized:
            haystack = planning_doc_haystack(doc)
            raw_content = strip_planning_metadata_lines(doc.page_content or "")
            content_norm = planning_doc_content_norm(doc)
            land_change_hits = planning_land_change_label_hits(content_norm)
            has_unused_zero = planning_has_unused_zero_evidence(content_norm)
            has_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat nong nghiep")
            has_non_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat phi nong nghiep")
            if current_status_doc is None and land_change_hits >= 2 and ("trong do" in content_norm or "hien trang nam 2024" in content_norm):
                current_status_doc = doc
            if unused_status_doc is None and has_unused_zero:
                unused_status_doc = doc
            if paired_totals_doc is None and (has_agri_pair or has_non_agri_pair) and land_change_hits >= 1:
                paired_totals_doc = doc
            if agricultural_pair_doc is None and has_agri_pair:
                agricultural_pair_doc = doc
            if non_agricultural_pair_doc is None and has_non_agri_pair:
                non_agricultural_pair_doc = doc
            if unused_pair_doc is None and ("dat chua su dung" in haystack or has_unused_zero):
                unused_pair_doc = doc
            if paired_totals_doc is not None and agricultural_pair_doc is not None and non_agricultural_pair_doc is not None and unused_pair_doc is not None:
                break

        for doc in [paired_totals_doc, agricultural_pair_doc, non_agricultural_pair_doc, current_status_doc, unused_pair_doc, unused_status_doc]:
            if doc is None:
                continue
            identity = planning_doc_identity(doc)
            if identity in prioritized_ids:
                continue
            prioritized_ids.add(identity)
            prioritized_docs.append(doc)

    if profile.registered_plan_composition:
        total_registered_doc: Optional[Document] = None
        resolution_count_doc: Optional[Document] = None
        resolution_area_doc: Optional[Document] = None
        added_plan_doc: Optional[Document] = None
        breakdown_doc: Optional[Document] = None
        for _, (_, doc) in prioritized:
            haystack = planning_doc_haystack(doc)
            registered_hits = planning_registered_plan_evidence_hits(haystack)
            has_total_registered = registered_hits > 0 and re.search(r"\b\d+\s+(?:danh\s+muc|cong\s+trinh|du\s+an)\b", haystack) is not None
            if total_registered_doc is None and has_total_registered:
                total_registered_doc = doc
            if resolution_count_doc is None and planning_has_registered_resolution_count_evidence(haystack):
                resolution_count_doc = doc
            if resolution_area_doc is None and any(marker in haystack for marker in ("thu hoi dat nam 2025", "chuyen muc dich dat trong lua", "nghi quyet")) and re.search(r"\b\d+(?:[\.,]\d+)?\s*ha\b", haystack):
                resolution_area_doc = doc
            if added_plan_doc is None and any(marker in haystack for marker in ("dua vao ke hoach su dung dat la", "dua vao ke hoach su dung dat")):
                added_plan_doc = doc
            if breakdown_doc is None and planning_continuation_signal(
                doc,
                registered_plan_composition_query=profile.registered_plan_composition,
                land_change_query=profile.land_change,
            ) > 0.0 and re.search(r"\b\d+(?:[\.,]\d+)?\s*ha\b", haystack):
                breakdown_doc = doc
            if total_registered_doc is not None and resolution_count_doc is not None and resolution_area_doc is not None and added_plan_doc is not None and breakdown_doc is not None:
                break

        for doc in [total_registered_doc, resolution_count_doc, resolution_area_doc, added_plan_doc, breakdown_doc]:
            if doc is None:
                continue
            identity = planning_doc_identity(doc)
            if identity in prioritized_ids:
                continue
            prioritized_ids.add(identity)
            prioritized_docs.append(doc)

    if profile.implementation_carry_forward:
        approved_doc: Optional[Document] = None
        implemented_doc: Optional[Document] = None
        carry_forward_doc: Optional[Document] = None
        for _, (_, doc) in prioritized:
            haystack = planning_doc_haystack(doc)
            if approved_doc is None and "ubnd thanh pho phe duyet" in haystack:
                approved_doc = doc
            if implemented_doc is None and any(marker in haystack for marker in ("da co quyet dinh giao dat", "da thuc hien thu hoi dat giai phong mat bang", "da cam moc gioi", "da thuc hien")):
                implemented_doc = doc
            if carry_forward_doc is None and any(marker in haystack for marker in ("chuyen tiep sang nam ke hoach 2025", "chua to chuc thuc hien", "chua to chuc")):
                carry_forward_doc = doc
            if approved_doc is not None and implemented_doc is not None and carry_forward_doc is not None:
                break

        for doc in [approved_doc, implemented_doc, carry_forward_doc]:
            if doc is None:
                continue
            identity = planning_doc_identity(doc)
            if identity in prioritized_ids:
                continue
            prioritized_ids.add(identity)
            prioritized_docs.append(doc)

    filtered_prioritized = prioritized
    if profile.registered_plan_composition:
        filtered_prioritized = [
            item
            for item in prioritized
            if planning_registered_plan_evidence_hits(planning_doc_haystack(item[1][1])) > 0
            or any(marker in planning_doc_haystack(item[1][1]) for marker in ("dua vao ke hoach su dung dat", "nghi quyet", "thu hoi dat nam 2025"))
        ]
    elif profile.land_change:
        filtered_prioritized = [
            item
            for item in prioritized
            if (
                ("2024" in planning_doc_content_norm(item[1][1]) and "2025" in planning_doc_content_norm(item[1][1]))
                or planning_land_change_label_hits(planning_doc_content_norm(item[1][1])) >= 2
                or planning_has_unused_zero_evidence(planning_doc_content_norm(item[1][1]))
                or planning_has_land_pair_evidence(strip_planning_metadata_lines(item[1][1].page_content or "") or planning_doc_content_norm(item[1][1]), "dat nong nghiep")
                or planning_has_land_pair_evidence(strip_planning_metadata_lines(item[1][1].page_content or "") or planning_doc_content_norm(item[1][1]), "dat phi nong nghiep")
            )
        ]
    elif profile.implementation_carry_forward:
        filtered_prioritized = [
            item
            for item in prioritized
            if planning_marker_hits(planning_doc_haystack(item[1][1]), PLANNING_IMPLEMENTATION_CARRY_FORWARD_MARKERS) > 0
            or any(marker in planning_doc_haystack(item[1][1]) for marker in ("da co quyet dinh giao dat", "da thuc hien thu hoi dat giai phong mat bang"))
        ]

    for identity, (_, doc) in filtered_prioritized or prioritized:
        if len(prioritized_docs) >= specialized_limit:
            break
        if identity in prioritized_ids:
            continue
        prioritized_ids.add(identity)
        prioritized_docs.append(doc)

    if not prioritized_docs:
        return selected_docs

    merged: list[Document] = []
    seen_merged: set[str] = set()
    for doc in [*prioritized_docs, *selected_docs]:
        identity = planning_doc_identity(doc)
        if identity in seen_merged:
            continue
        seen_merged.add(identity)
        merged.append(doc)
        if len(merged) >= limit:
            break

    return merged
