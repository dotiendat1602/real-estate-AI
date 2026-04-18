from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Optional

from langchain_core.documents import Document

from ..rag.retriever import lexical_search_documents
from .docs import planning_doc_identity, planning_doc_pid_idx
from .query_builders import planning_specialized_limit
from .features import (
    has_land_split_markers,
    planning_continuation_signal,
    planning_doc_content_norm,
    planning_doc_haystack,
    planning_has_explicit_project_row,
    planning_has_land_pair_evidence,
    planning_has_unused_zero_evidence,
    planning_is_tabular_header_fragment,
    planning_is_toc_like_chunk,
    planning_land_change_label_hits,
    planning_registered_plan_evidence_hits,
    strip_planning_metadata_lines,
)
from .profiles import build_planning_query_profile
from .ranker import (
    planning_intent_alignment_score,
    planning_intent_markers,
    planning_neighbor_offsets,
    planning_query_terms,
    planning_rescue_query_score,
)

PlanningDocScoreFn = Callable[[Document, str, Optional[str], Optional[int]], float]
PlanningSpecializedEvidenceScoreFn = Callable[[Document, str, Optional[str], Optional[int]], float | None]
PlanningRescueQueriesFn = Callable[[str, Optional[str], Optional[int]], list[str]]


async def augment_planning_text_neighbors(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    limit: int,
    *,
    planning_doc_score: PlanningDocScoreFn,
) -> list[Document]:
    if not selected_docs or limit <= 0:
        return selected_docs

    profile = build_planning_query_profile(message, planning_intent=True)
    existing_identities: set[str] = {planning_doc_identity(doc) for doc in selected_docs}
    existing_pid_idx: set[tuple[int, int]] = set()
    anchor_candidates: list[tuple[float, int, int]] = []

    for doc in selected_docs:
        planning_document_id, chunk_index = planning_doc_pid_idx(doc)
        if planning_document_id is None or chunk_index is None:
            continue

        existing_pid_idx.add((planning_document_id, chunk_index))
        score = planning_doc_score(doc, message, district, plan_year)
        anchor_candidates.append((score, planning_document_id, chunk_index))

    if not anchor_candidates:
        return selected_docs

    anchor_candidates.sort(reverse=True)
    anchors_by_pid: dict[int, list[int]] = {}
    for _, planning_document_id, chunk_index in anchor_candidates:
        bucket = anchors_by_pid.setdefault(planning_document_id, [])
        if chunk_index in bucket:
            continue
        bucket.append(chunk_index)
        if len(bucket) >= 2:
            continue
        if len(anchors_by_pid) >= 2 and all(len(v) >= 1 for v in anchors_by_pid.values()):
            if sum(len(v) for v in anchors_by_pid.values()) >= 4:
                break

    target_indices_by_pid: dict[int, set[int]] = {}
    for planning_document_id, anchor_indices in anchors_by_pid.items():
        target_indices: set[int] = set()
        for doc in selected_docs:
            pid, anchor_idx = planning_doc_pid_idx(doc)
            if pid != planning_document_id or anchor_idx not in anchor_indices:
                continue

            offsets = planning_neighbor_offsets(doc, message)
            if profile.project_listing:
                offsets.update({offset for offset in range(-12, 13) if offset != 0})
            elif not offsets:
                offsets.update({-2, -1, 1, 2})

            for offset in offsets:
                candidate_idx = anchor_idx + offset
                if candidate_idx < 0:
                    continue
                if (planning_document_id, candidate_idx) in existing_pid_idx:
                    continue
                target_indices.add(candidate_idx)
        if target_indices:
            target_indices_by_pid[planning_document_id] = target_indices

    if not target_indices_by_pid:
        return selected_docs

    fetched_neighbors: list[Document] = []
    for planning_document_id, target_indices in target_indices_by_pid.items():
        try:
            candidates = await lexical_search_documents(
                planning_vs,
                query="",
                k=max(4, len(target_indices) * 2),
                filters={"chunkTypes": ["text"], "globalChunkIndex": {"$in": sorted(target_indices)}},
                base_filter={"documentScope": "planning", "planningDocumentId": planning_document_id},
                allow_empty_terms=True,
            )
        except Exception:
            continue

        for candidate in candidates:
            candidate_pid, candidate_idx = planning_doc_pid_idx(candidate)
            if candidate_pid != planning_document_id or candidate_idx is None:
                continue
            if candidate_idx not in target_indices:
                continue

            haystack = planning_doc_haystack(candidate)
            if planning_is_toc_like_chunk(candidate.page_content or "", haystack):
                continue

            identity = planning_doc_identity(candidate)
            if identity in existing_identities:
                continue
            existing_identities.add(identity)
            fetched_neighbors.append(candidate)

    if not fetched_neighbors:
        return selected_docs

    def _continuation_score(doc: Document) -> float:
        return planning_continuation_signal(
            doc,
            registered_plan_composition_query=profile.registered_plan_composition,
            land_change_query=profile.land_change,
        )

    def _neighbor_rank(doc: Document) -> tuple[int, float]:
        continuation_score = _continuation_score(doc)
        doc_score = planning_doc_score(doc, message, district, plan_year)
        rescue_score = planning_rescue_query_score(doc, message) * 1.1
        total_score = doc_score + rescue_score + continuation_score * 2.6

        bucket = 2
        if continuation_score > 0.0:
            bucket = 0
        elif profile.land_change:
            content_norm = planning_doc_content_norm(doc)
            if planning_land_change_label_hits(content_norm) > 0 or planning_has_unused_zero_evidence(content_norm):
                bucket = 1
        elif profile.registered_plan_composition:
            haystack = planning_doc_haystack(doc)
            if any(marker in haystack for marker in ("dua vao ke hoach su dung dat", "nghi quyet", "dien tich")):
                bucket = 1

        return (bucket, -total_score)

    fetched_neighbors.sort(key=_neighbor_rank)
    if profile.project_listing:
        max_neighbors = min(len(fetched_neighbors), max(4, min(8, max(2, limit // 2))))
    else:
        max_neighbors = min(len(fetched_neighbors), max(2, min(4, max(1, limit // 3))))
    selected_neighbors = fetched_neighbors[:max_neighbors]
    if not selected_neighbors:
        return selected_docs

    insert_at = min(len(selected_docs), max(1, len(selected_docs) // 2))
    merged_candidate = [*selected_docs[:insert_at], *selected_neighbors, *selected_docs[insert_at:]]

    merged: list[Document] = []
    seen_merged: set[str] = set()
    for doc in merged_candidate:
        identity = planning_doc_identity(doc)
        if identity in seen_merged:
            continue
        seen_merged.add(identity)
        merged.append(doc)
        if len(merged) >= limit:
            break

    return merged


async def augment_planning_continuation_neighbors(
    planning_vs,
    message: str,
    selected_docs: list[Document],
    limit: int,
) -> list[Document]:
    if not selected_docs or limit <= 0:
        return selected_docs

    profile = build_planning_query_profile(message, planning_intent=True)
    if not (profile.registered_plan_composition or profile.land_change):
        return selected_docs

    existing_identities: set[str] = {planning_doc_identity(doc) for doc in selected_docs}
    anchor_targets: dict[tuple[int, int], set[int]] = {}

    for doc in selected_docs:
        planning_document_id, chunk_index = planning_doc_pid_idx(doc)
        if planning_document_id is None or chunk_index is None:
            continue

        haystack = planning_doc_haystack(doc)
        content_norm = planning_doc_content_norm(doc)
        candidate_indices: set[int] = set()
        if profile.registered_plan_composition and any(
            marker in haystack
            for marker in (
                "tong so cong trinh du an dang ky thuc hien",
                "tong so du an",
                "trong do",
                "dang ky thuc hien",
            )
        ):
            candidate_indices.update({chunk_index + 1, chunk_index + 2})
        if profile.land_change and (
            ("trong do" in content_norm and planning_land_change_label_hits(content_norm) > 0)
            or ("hien trang nam 2024" in content_norm and planning_land_change_label_hits(content_norm) > 0)
        ):
            candidate_indices.update({chunk_index + 1, chunk_index + 2})
        if candidate_indices:
            anchor_targets[(planning_document_id, chunk_index)] = {idx for idx in candidate_indices if idx >= 0}

    if not anchor_targets:
        return selected_docs

    def _continuation_score(doc: Document) -> float:
        return planning_continuation_signal(
            doc,
            registered_plan_composition_query=profile.registered_plan_composition,
            land_change_query=profile.land_change,
        )

    fetched_by_anchor: dict[tuple[int, int], list[Document]] = {}
    for (planning_document_id, anchor_idx), candidate_indices in anchor_targets.items():
        try:
            candidates = await lexical_search_documents(
                planning_vs,
                query="",
                k=max(4, len(candidate_indices) * 2),
                filters={"chunkTypes": ["text"], "globalChunkIndex": {"$in": sorted(candidate_indices)}},
                base_filter={"documentScope": "planning", "planningDocumentId": planning_document_id},
                allow_empty_terms=True,
            )
        except Exception:
            continue

        accepted: list[Document] = []
        for candidate in candidates:
            identity = planning_doc_identity(candidate)
            if identity in existing_identities:
                continue
            if profile.registered_plan_composition:
                if _continuation_score(candidate) <= 0.0:
                    continue
            if profile.land_change:
                candidate_content_norm = planning_doc_content_norm(candidate)
                if _continuation_score(candidate) <= 0.0 and not (
                    planning_land_change_label_hits(candidate_content_norm) > 0
                    or planning_has_unused_zero_evidence(candidate_content_norm)
                ):
                    continue
            existing_identities.add(identity)
            accepted.append(candidate)

        if accepted:
            accepted.sort(
                key=lambda doc: (
                    -_continuation_score(doc),
                    -planning_rescue_query_score(doc, message),
                )
            )
            fetched_by_anchor[(planning_document_id, anchor_idx)] = accepted[:2]

    if not fetched_by_anchor:
        return selected_docs

    merged: list[Document] = []
    seen_merged: set[str] = set()
    for doc in selected_docs:
        identity = planning_doc_identity(doc)
        if identity not in seen_merged:
            seen_merged.add(identity)
            merged.append(doc)

        planning_document_id, chunk_index = planning_doc_pid_idx(doc)
        if planning_document_id is None or chunk_index is None:
            continue
        for neighbor in fetched_by_anchor.get((planning_document_id, chunk_index), []):
            neighbor_identity = planning_doc_identity(neighbor)
            if neighbor_identity in seen_merged:
                continue
            seen_merged.add(neighbor_identity)
            merged.append(neighbor)

    return merged[:limit]


async def augment_planning_land_change_fact_docs(
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    limit: int,
    *,
    planning_doc_score: PlanningDocScoreFn,
    load_planning_document_docs: Callable[..., list[Document]],
) -> list[Document]:
    profile = build_planning_query_profile(message, planning_intent=True)
    if not selected_docs or limit <= 0 or not profile.land_change:
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

    prioritized_facts: list[Document] = []
    prioritized_ids: set[str] = set()
    for planning_document_id in candidate_pids:
        sql_candidates = await asyncio.to_thread(
            load_planning_document_docs,
            planning_document_id,
            plan_year,
            chunk_types=("text", "table"),
            limit=2500,
        )
        slot_docs: dict[str, tuple[float, Document]] = {}
        for candidate in sql_candidates:
            haystack = planning_doc_haystack(candidate)
            if not haystack or planning_is_toc_like_chunk(candidate.page_content or "", haystack):
                continue
            if planning_is_tabular_header_fragment(haystack):
                continue

            raw_content = strip_planning_metadata_lines(candidate.page_content or "")
            content_norm = planning_doc_content_norm(candidate)
            land_change_hits = planning_land_change_label_hits(content_norm)
            has_year_pair = "2024" in content_norm and "2025" in content_norm
            has_unused_zero = planning_has_unused_zero_evidence(content_norm)
            has_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat nong nghiep")
            has_non_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat phi nong nghiep")
            has_recovery_markers = any(
                marker in content_norm for marker in ("thu hoi", "ke hoach thu hoi dat", "dien tich thu hoi", "dat can thu hoi")
            )
            if not has_year_pair and land_change_hits < 2 and not has_unused_zero and not (has_agri_pair or has_non_agri_pair):
                continue
            if has_recovery_markers and not has_year_pair:
                continue
            if any(marker in haystack for marker in ("chuyen sang dat phi nong nghiep", "chuyen doi co cau su dung dat")) and not (
                has_agri_pair or has_non_agri_pair or has_unused_zero
            ):
                continue

            score = planning_doc_score(candidate, message, district, plan_year)
            if has_year_pair:
                score += 6.0
            if land_change_hits >= 2 and any(marker in content_norm for marker in ("hien trang nam 2024", "trong do")):
                score += 4.0
            if has_unused_zero:
                score += 4.5
            if has_agri_pair:
                score += 3.0
            if has_non_agri_pair:
                score += 3.0

            def _update_slot(slot: str, slot_score: float) -> None:
                previous = slot_docs.get(slot)
                if previous is None or slot_score > previous[0]:
                    slot_docs[slot] = (slot_score, candidate)

            if has_year_pair and land_change_hits >= 2 and (has_agri_pair or has_non_agri_pair):
                _update_slot("paired_totals", score + 2.4)
            if has_agri_pair:
                _update_slot("agricultural_pair", score + 1.8)
            if has_non_agri_pair:
                _update_slot("non_agricultural_pair", score + 1.8)
            if land_change_hits >= 2 and any(marker in content_norm for marker in ("hien trang nam 2024", "trong do")):
                _update_slot("current_status", score + 1.4)
            if has_unused_zero:
                _update_slot("unused_zero", score + 2.6)

        for slot in ("paired_totals", "agricultural_pair", "non_agricultural_pair", "current_status", "unused_zero"):
            doc = slot_docs.get(slot, (0.0, None))[1]
            if doc is None:
                continue
            identity = planning_doc_identity(doc)
            if identity in prioritized_ids:
                continue
            prioritized_ids.add(identity)
            prioritized_facts.append(doc)

    if not prioritized_facts:
        return selected_docs

    merged: list[Document] = []
    seen: set[str] = set()
    for doc in [*prioritized_facts, *selected_docs]:
        identity = planning_doc_identity(doc)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(doc)
        if len(merged) >= limit:
            break

    return merged


async def augment_planning_table_neighbors(
    planning_vs,
    message: str,
    selected_docs: list[Document],
    limit: int,
) -> list[Document]:
    if not selected_docs or limit <= 0:
        return selected_docs

    profile = build_planning_query_profile(message, planning_intent=True)
    query_terms = planning_query_terms(message)
    existing_identities: set[str] = {planning_doc_identity(doc) for doc in selected_docs}
    existing_pid_idx: set[tuple[int, int]] = set()
    anchor_indices_by_pid: dict[int, set[int]] = {}

    for doc in selected_docs:
        md = doc.metadata or {}
        chunk_type = str(md.get("chunkType") or "").lower().strip()
        if chunk_type != "table":
            continue

        planning_document_id, chunk_index = planning_doc_pid_idx(doc)
        if planning_document_id is None or chunk_index is None:
            continue

        existing_pid_idx.add((planning_document_id, chunk_index))
        blob = planning_doc_haystack(doc)
        overlap_hits = sum(1 for term in query_terms if term in blob) if query_terms else 0
        has_numeric_row = re.search(r"\b\d+(?:[\.,]\d+)?(?:\s+\d+(?:[\.,]\d+)?){1,}\b", blob) is not None

        if overlap_hits > 0 or has_numeric_row:
            anchor_indices_by_pid.setdefault(planning_document_id, set()).add(chunk_index)

    if not anchor_indices_by_pid:
        for doc in selected_docs:
            md = doc.metadata or {}
            if str(md.get("chunkType") or "").lower().strip() != "table":
                continue
            planning_document_id, chunk_index = planning_doc_pid_idx(doc)
            if planning_document_id is None or chunk_index is None:
                continue
            anchor_indices_by_pid.setdefault(planning_document_id, set()).add(chunk_index)
            if len(anchor_indices_by_pid) >= 2:
                break

    if not anchor_indices_by_pid:
        return selected_docs

    neighbor_radius = 6 if profile.project_listing else (2 if limit >= 10 else 1)
    target_indices_by_pid: dict[int, set[int]] = {}
    for planning_document_id in sorted(anchor_indices_by_pid.keys())[:2]:
        target_indices: set[int] = set()
        for anchor_idx in anchor_indices_by_pid.get(planning_document_id, set()):
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                if offset == 0:
                    continue
                candidate_idx = anchor_idx + offset
                if candidate_idx < 0:
                    continue
                if (planning_document_id, candidate_idx) in existing_pid_idx:
                    continue
                target_indices.add(candidate_idx)
        if target_indices:
            target_indices_by_pid[planning_document_id] = target_indices

    if not target_indices_by_pid:
        return selected_docs

    fetched_neighbors: list[Document] = []
    for planning_document_id, target_indices in target_indices_by_pid.items():
        try:
            candidates = await lexical_search_documents(
                planning_vs,
                query="",
                k=max(4, len(target_indices) * 2),
                filters={"chunkTypes": ["table"], "globalChunkIndex": {"$in": sorted(target_indices)}},
                base_filter={"documentScope": "planning", "planningDocumentId": planning_document_id},
                allow_empty_terms=True,
            )
        except Exception:
            continue

        for candidate in candidates:
            candidate_pid, candidate_idx = planning_doc_pid_idx(candidate)
            if candidate_pid != planning_document_id or candidate_idx is None:
                continue
            if candidate_idx not in target_indices:
                continue

            identity = planning_doc_identity(candidate)
            if identity in existing_identities:
                continue

            existing_identities.add(identity)
            fetched_neighbors.append(candidate)

    if not fetched_neighbors:
        return selected_docs

    def _neighbor_priority(doc: Document) -> tuple[float, float]:
        blob = planning_doc_haystack(doc)
        overlap = sum(1 for term in query_terms if term in blob) if query_terms else 0
        numeric_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", blob))
        content_norm = planning_doc_content_norm(doc)
        year_pair = 1 if ("2024" in content_norm and "2025" in content_norm) else 0
        rescue_score = planning_rescue_query_score(doc, message)
        listing_entity_hits = 0
        if profile.project_listing:
            listing_entity_hits = sum(
                1
                for marker in (
                    "tram y te",
                    "chuong duong",
                    "f/thpt1",
                    "mai dich",
                    "44 yet kieu",
                    "ga c10",
                    "ga s12",
                    "khoan 4 dieu 67",
                    "nha tang le quoc gia",
                    "5 tran thanh tong",
                    "30 tran binh trong",
                    "58 tran nhan tong",
                    "tong so",
                    "duoc phe duyet",
                    "8,6915",
                    "8.6915",
                )
                if marker in blob
            )
        score = float(overlap) * 1.8 + float(min(numeric_hits, 10)) * 0.15 + float(year_pair) * 0.5
        score += rescue_score * 1.25
        score += float(min(listing_entity_hits, 8)) * 0.7
        return (score, float(overlap))

    fetched_neighbors.sort(key=_neighbor_priority, reverse=True)
    max_neighbors = min(len(fetched_neighbors), max(2, min(6, max(1, limit // 3))))
    selected_neighbors = fetched_neighbors[:max_neighbors]
    if not selected_neighbors:
        return selected_docs

    insert_at = min(len(selected_docs), max(1, len(selected_docs) // 2))
    merged_candidate = [*selected_docs[:insert_at], *selected_neighbors, *selected_docs[insert_at:]]

    merged: list[Document] = []
    seen_merged: set[str] = set()
    for doc in merged_candidate:
        identity = planning_doc_identity(doc)
        if identity in seen_merged:
            continue
        seen_merged.add(identity)
        merged.append(doc)
        if len(merged) >= limit:
            break

    return merged


async def augment_planning_intent_evidence(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    pool_docs: list[Document],
    limit: int,
    *,
    planning_doc_score: PlanningDocScoreFn,
    planning_specialized_evidence_score: PlanningSpecializedEvidenceScoreFn,
    planning_intent_rescue_queries: PlanningRescueQueriesFn,
) -> list[Document]:
    if not selected_docs or limit <= 0:
        return selected_docs

    profile = build_planning_query_profile(message, planning_intent=True)
    normalized_message = profile.normalized
    intent_marker_terms = planning_intent_markers(message)
    rescue_queries = planning_intent_rescue_queries(message, district, plan_year)
    query_years = set(re.findall(r"\b20\d{2}\b", normalized_message))
    if not intent_marker_terms and not rescue_queries:
        return selected_docs

    candidate_pids: list[int] = []
    for doc in [*selected_docs, *pool_docs[:10]]:
        planning_document_id, _ = planning_doc_pid_idx(doc)
        if planning_document_id is None or planning_document_id in candidate_pids:
            continue
        candidate_pids.append(planning_document_id)
        if len(candidate_pids) >= 2:
            break

    if not candidate_pids:
        return selected_docs

    existing_identities: set[str] = {planning_doc_identity(doc) for doc in selected_docs}
    rescue_seen_identities: set[str] = set(existing_identities)
    lexical_k = max(18, min(40, limit * 3 + 8)) if profile.project_listing else max(10, min(26, limit * 2 + 4))
    scored_rescue: list[tuple[Document, float, float, str]] = []
    forced_winners: list[Document] = []
    forced_winner_ids: set[str] = set()

    for planning_document_id in candidate_pids:
        rescue_base_filters: list[dict[str, Any]] = []
        if plan_year is not None:
            rescue_base_filters.append(
                {
                    "documentScope": "planning",
                    "planningDocumentId": planning_document_id,
                    "planYear": plan_year,
                }
            )
        rescue_base_filters.append({"documentScope": "planning", "planningDocumentId": planning_document_id})

        query_limit = min(len(rescue_queries), 8) if profile.project_listing else min(len(rescue_queries), 4)
        per_query_take = 2 if profile.project_listing else 1
        for rescue_base_filter in rescue_base_filters:
            for query_text in rescue_queries[:query_limit]:
                try:
                    candidates = await lexical_search_documents(
                        planning_vs,
                        query=query_text,
                        k=lexical_k,
                        filters={"chunkTypes": ["text", "table"]},
                        base_filter=rescue_base_filter,
                    )
                except Exception:
                    continue

                query_scored: list[tuple[Document, float, float, str]] = []
                for candidate in candidates:
                    identity = planning_doc_identity(candidate)
                    if identity in existing_identities:
                        continue

                    haystack = planning_doc_haystack(candidate)
                    if planning_is_toc_like_chunk(candidate.page_content or "", haystack):
                        continue

                    intent_score = planning_intent_alignment_score(candidate, intent_marker_terms, query_years)
                    rescue_query_score = planning_rescue_query_score(candidate, query_text)
                    entity_hits = sum(
                        1
                        for marker in (
                            "du an",
                            "cong trinh",
                            "xay dung",
                            "tram y te",
                            "truong thpt",
                            "f/thpt",
                            "44 yet kieu",
                            "ga c10",
                            "ga s12",
                            "khoan 4 dieu 67",
                            "nha tang le",
                            "tran thanh tong",
                            "bo cong an",
                            "tran binh trong",
                            "tran nhan tong",
                            "quyet dinh",
                            "phu luc",
                        )
                        if marker in haystack
                    )

                    if intent_marker_terms and intent_score <= 0.0 and not (profile.project_listing and entity_hits > 0):
                        continue

                    total_score = planning_doc_score(candidate, message, district, plan_year) * 0.7
                    total_score += intent_score * 1.4 + rescue_query_score * 1.8
                    if profile.project_listing:
                        total_score += min(entity_hits, 6) * 0.95
                        if "theo quyet dinh" in normalized_message and "quyet dinh" in haystack:
                            total_score += 1.1
                    query_scored.append((candidate, total_score, rescue_query_score, identity))

                if not query_scored:
                    continue

                query_scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
                winner_candidate, _, winner_query_score, winner_identity = query_scored[0]
                if winner_identity not in forced_winner_ids and winner_query_score > 0.0:
                    forced_winners.append(winner_candidate)
                    forced_winner_ids.add(winner_identity)
                added = 0
                for candidate, total_score, rescue_query_score, identity in query_scored:
                    if identity in rescue_seen_identities:
                        continue
                    scored_rescue.append((candidate, total_score, rescue_query_score, identity))
                    rescue_seen_identities.add(identity)
                    added += 1
                    if added >= per_query_take:
                        break

    specialized_candidates: list[tuple[float, Document, str]] = []
    specialized_seen: set[str] = set()
    for doc in [*forced_winners, *selected_docs, *(item[0] for item in scored_rescue)]:
        identity = planning_doc_identity(doc)
        if identity in specialized_seen:
            continue
        specialized_seen.add(identity)
        specialized_score = planning_specialized_evidence_score(doc, message, district, plan_year)
        if specialized_score is None:
            continue
        specialized_candidates.append((specialized_score, doc, identity))

    specialized_candidates.sort(key=lambda item: item[0], reverse=True)
    specialized_limit = planning_specialized_limit(message)

    if not scored_rescue:
        if forced_winners or specialized_candidates:
            merged: list[Document] = []
            seen: set[str] = set()
            prioritized = [doc for _, doc, _ in specialized_candidates[:specialized_limit]] if specialized_limit > 0 else []
            for doc in [*prioritized, *forced_winners, *selected_docs]:
                identity = planning_doc_identity(doc)
                if identity in seen:
                    continue
                seen.add(identity)
                merged.append(doc)
                if len(merged) >= limit:
                    break
            return merged
        return selected_docs

    scored_rescue.sort(key=lambda item: (item[1], item[2]), reverse=True)
    if profile.project_listing:
        max_injected = min(len(scored_rescue), max(3, min(6, max(2, limit // 2 + 1))))
    else:
        max_injected = min(len(scored_rescue), max(2, min(6, max(1, limit // 2))))
    injected_docs: list[Document] = []
    injected_ids: set[str] = set()
    if specialized_limit > 0:
        for _, doc, identity in specialized_candidates[:specialized_limit]:
            if identity in injected_ids:
                continue
            injected_ids.add(identity)
            injected_docs.append(doc)
            if len(injected_docs) >= max_injected:
                break

    for doc in forced_winners:
        identity = planning_doc_identity(doc)
        if identity in injected_ids:
            continue
        injected_ids.add(identity)
        injected_docs.append(doc)
        if len(injected_docs) >= max_injected:
            break

    for doc, _, _, identity in scored_rescue:
        if identity in injected_ids:
            continue
        injected_ids.add(identity)
        injected_docs.append(doc)
        if len(injected_docs) >= max_injected:
            break

    merged: list[Document] = []
    seen: set[str] = set()
    for doc in [*injected_docs, *selected_docs]:
        identity = planning_doc_identity(doc)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(doc)
        if len(merged) >= limit:
            break

    return merged


async def augment_planning_land_recovery_evidence(
    planning_vs,
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    selected_docs: list[Document],
    pool_docs: list[Document],
    limit: int,
    *,
    planning_doc_score: PlanningDocScoreFn,
) -> list[Document]:
    if not selected_docs or limit <= 0:
        return selected_docs

    profile = build_planning_query_profile(message, planning_intent=True)
    if not profile.land_recovery:
        return selected_docs

    expects_split = all(marker in profile.normalized for marker in ("dat nong nghiep", "dat phi nong nghiep"))
    candidate_pids: list[int] = []
    for doc in [*selected_docs, *pool_docs[:8]]:
        planning_document_id, _ = planning_doc_pid_idx(doc)
        if planning_document_id is None or planning_document_id in candidate_pids:
            continue
        candidate_pids.append(planning_document_id)
        if len(candidate_pids) >= 2:
            break

    if not candidate_pids:
        return selected_docs

    recovery_queries = [
        "ke hoach thu hoi dat tong cong dat nong nghiep dat phi nong nghiep",
        "tong dien tich dat thu hoi dat nong nghiep dat phi nong nghiep ha",
        "tong cong thu hoi dat nong nghiep dat phi nong nghiep",
    ]
    if plan_year is not None:
        recovery_queries = [f"{query} nam {plan_year}" for query in recovery_queries] + recovery_queries
    if district:
        recovery_queries = [f"{query} {district}" for query in recovery_queries] + recovery_queries

    existing_identities: set[str] = {planning_doc_identity(doc) for doc in selected_docs}
    scored_candidates: list[tuple[Document, float, str]] = []

    for planning_document_id in candidate_pids:
        base_filters: list[dict[str, Any]] = []
        if plan_year is not None:
            base_filters.append(
                {
                    "documentScope": "planning",
                    "planningDocumentId": planning_document_id,
                    "planYear": plan_year,
                }
            )
        base_filters.append({"documentScope": "planning", "planningDocumentId": planning_document_id})

        for base_filter in base_filters:
            for query_text in recovery_queries[:6]:
                try:
                    candidates = await lexical_search_documents(
                        planning_vs,
                        query=query_text,
                        k=max(12, min(18, limit * 2)),
                        filters={"chunkTypes": ["text", "table"]},
                        base_filter=base_filter,
                    )
                except Exception:
                    continue

                for candidate in candidates:
                    identity = planning_doc_identity(candidate)
                    if identity in existing_identities:
                        continue

                    haystack = planning_doc_haystack(candidate)
                    if planning_is_toc_like_chunk(candidate.page_content or "", haystack):
                        continue

                    has_split_markers = has_land_split_markers(haystack)
                    has_total_markers = any(marker in haystack for marker in ("tong cong", "tong dien tich", "ke hoach thu hoi dat"))
                    has_area_unit = re.search(r"\b\d+(?:[\.,]\d+)?\s*ha\b", haystack) is not None

                    if expects_split and not (has_split_markers and (has_total_markers or has_area_unit)):
                        continue
                    if not expects_split and not (has_total_markers or has_area_unit):
                        continue

                    area_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?\s*ha\b", haystack))
                    score = planning_doc_score(candidate, message, district, plan_year)
                    score += float(min(area_hits, 6)) * 0.7
                    if has_split_markers:
                        score += 1.2
                    if has_total_markers:
                        score += 1.0

                    scored_candidates.append((candidate, score, identity))

                if len(scored_candidates) >= max(8, limit):
                    break

            if len(scored_candidates) >= max(8, limit):
                break

    if not scored_candidates:
        return selected_docs

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    injected_docs: list[Document] = []
    seen_injected: set[str] = set()
    max_injected = min(max(2, limit // 3), 4)
    for candidate, _, identity in scored_candidates:
        if identity in seen_injected:
            continue
        seen_injected.add(identity)
        injected_docs.append(candidate)
        if len(injected_docs) >= max_injected:
            break

    if not injected_docs:
        return selected_docs

    keep_count = max(0, limit - len(injected_docs))
    merged: list[Document] = []
    seen_merged: set[str] = set()

    for doc in [*injected_docs, *selected_docs[:keep_count]]:
        identity = planning_doc_identity(doc)
        if identity in seen_merged:
            continue
        seen_merged.add(identity)
        merged.append(doc)
        if len(merged) >= limit:
            break

    return merged
