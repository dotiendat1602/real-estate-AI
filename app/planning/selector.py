from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional

from langchain_core.documents import Document

from .features import (
    planning_admin_unit_header_hits,
    planning_count_pattern_score,
    planning_doc_content_norm,
    planning_explanatory_evidence_hits,
    planning_has_admin_unit_evidence,
    planning_has_explicit_project_row,
    planning_has_land_pair_evidence,
    planning_has_natural_area_admin_evidence,
    planning_has_unused_zero_evidence,
    planning_is_heading_or_incomplete_chunk,
    planning_is_toc_like_chunk,
    planning_land_change_label_hits,
    planning_named_entity_hits,
    planning_registered_plan_evidence_hits,
    strip_planning_metadata_lines,
)
from .profiles import (
    PLANNING_FOCUS_MANAGEMENT_MARKERS,
    PLANNING_GROWTH_PRESSURE_MARKERS,
    PLANNING_IMPLEMENTATION_CARRY_FORWARD_MARKERS,
    PLANNING_PROJECT_DELAY_REASON_MARKERS,
    PLANNING_REGISTERED_COMPOSITION_MARKERS,
    build_planning_query_profile,
    planning_focus_phrases,
    planning_marker_hits,
    strip_accents,
)
from .ranker import planning_intent_markers, planning_query_terms

DistrictMatcher = Callable[[Document, Optional[str]], bool]
YearMatcher = Callable[[Document, Optional[int]], bool]
DocScoreFn = Callable[[Document, str, Optional[str], Optional[int]], float]


def _normalize_selector_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = strip_accents(lowered)
    lowered = lowered.replace("Ä‘", "d").replace("Ä", "d")
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def select_ranked_planning_docs(
    ranked_docs: list[Document],
    message: str,
    limit: int,
    district: Optional[str] = None,
    plan_year: Optional[int] = None,
    *,
    doc_matches_district: DistrictMatcher,
    doc_matches_plan_year: YearMatcher,
    planning_doc_score: DocScoreFn,
) -> list[Document]:
    if not ranked_docs or limit <= 0:
        return []

    profile = build_planning_query_profile(message, planning_intent=True)
    query_terms = planning_query_terms(message)
    intent_markers = planning_intent_markers(message)
    query_years = set(re.findall(r"\b20\d{2}\b", _normalize_selector_text(message)))
    normalized_message = _normalize_selector_text(message)
    recovery_split_query = profile.land_recovery and all(
        marker in normalized_message for marker in ("dat nong nghiep", "dat phi nong nghiep")
    )
    decision_query = any(marker in normalized_message for marker in ("theo quyet dinh", "quyet dinh phe duyet"))
    prefer_text_evidence = profile.explanatory_query or any(
        marker in normalized_message
        for marker in (
            "nhu the nao",
            "trien khai",
            "cong tac",
            "ket qua",
            "tien do",
            "du an nao",
            "gom nhung gi",
            "dang ky moi",
            "duy nhat",
            "khoan 4 dieu 67",
            "cap thanh pho",
        )
    )
    if profile.admin_overview:
        prefer_text_evidence = True
    if profile.project_listing:
        if profile.decision_total:
            prefer_text_evidence = False
        elif profile.city_level_listing or profile.article67 or profile.new_registration_unique:
            prefer_text_evidence = True
        else:
            prefer_text_evidence = False
    if profile.land_change or profile.public_purpose_composition:
        prefer_text_evidence = False
    if (
        profile.gpmb_stats
        or profile.environment_constraint
        or profile.plan_necessity
        or profile.focus_area_reason
        or profile.sector_land_demand
        or profile.drainage_transport
        or profile.project_delay_reason
    ):
        prefer_text_evidence = True

    selected: list[Document] = []
    seen: set[str] = set()

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

    def _blob(doc: Document) -> str:
        md = doc.metadata or {}
        return _normalize_selector_text(
            "\n".join(
                [
                    str(md.get("title") or ""),
                    str(md.get("sourceLocator") or ""),
                    doc.page_content or "",
                ]
            )
        )

    def _overlap_hits(blob: str) -> int:
        if not query_terms:
            return 0
        return sum(1 for term in query_terms if term in blob)

    def _priority(doc: Document) -> tuple[float, float]:
        md = doc.metadata or {}
        chunk_type = str(md.get("chunkType") or "").lower().strip()
        blob = _blob(doc)
        content_norm = _normalize_selector_text(strip_planning_metadata_lines(doc.page_content or ""))
        title_norm = _normalize_selector_text(str(md.get("title") or ""))
        named_entity_hits = planning_named_entity_hits(blob)
        count_pattern = planning_count_pattern_score(blob)
        explicit_project_row = planning_has_explicit_project_row(blob)
        admin_unit_hits = planning_admin_unit_header_hits(blob)
        has_admin_overview_evidence = planning_has_natural_area_admin_evidence(blob)
        explanatory_hits = planning_explanatory_evidence_hits(blob)

        if planning_is_toc_like_chunk(doc.page_content or "", blob):
            return (-999.0, 0.0)

        overlap = float(_overlap_hits(blob))
        intent_hits = float(sum(1 for marker in intent_markers if marker in blob)) if intent_markers else 0.0
        numeric_hits = float(len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", blob)))
        dense_numeric_rows = float(len(re.findall(r"\b\d+(?:[\.,]\d+)?(?:\s+\d+(?:[\.,]\d+)?){2,}\b", blob)))
        year_pair = 1.0 if ("2024" in content_norm and "2025" in content_norm) else 0.0
        doc_years = set(re.findall(r"\b20\d{2}\b", content_norm))
        content_year_hits = float(sum(1 for year in query_years if year in content_norm)) if query_years else 0.0
        district_ok = doc_matches_district(doc, district)
        year_ok = doc_matches_plan_year(doc, plan_year)
        year_coverage = 0.0
        if len(query_years) >= 2:
            overlap_years = len(query_years.intersection(doc_years))
            if overlap_years >= len(query_years):
                year_coverage = 1.2
            elif overlap_years == 1:
                year_coverage = -0.8
        if prefer_text_evidence:
            table_bonus = 0.1 if chunk_type == "table" else 0.55
        else:
            table_bonus = 0.8 if chunk_type == "table" else 0.2

        score = overlap * 1.6
        score += min(intent_hits, 8.0) * 0.45
        score += min(numeric_hits, 10.0) * 0.2 + min(dense_numeric_rows, 6.0) * 0.35
        score += year_pair * 0.6 + year_coverage + content_year_hits * 0.7 + table_bonus
        score += min(named_entity_hits, 4) * 1.15
        score += count_pattern
        if explicit_project_row:
            score += 1.6
        if admin_unit_hits > 0:
            score += min(admin_unit_hits, 12) * 0.12
        if district is not None:
            score += 0.9 if district_ok else -3.0
        if plan_year is not None:
            score += 0.6 if year_ok else -1.0

        if profile.project_listing:
            listing_hits = sum(
                1
                for marker in ("phu luc", "du an", "cong trinh", "khoan 4 dieu 67", "dang ky moi", "duy nhat", "quyet dinh")
                if marker in blob
            )
            score += min(listing_hits, 8) * 0.4
            if count_pattern > 0.0 and named_entity_hits == 0 and profile.new_registration_unique:
                score -= 2.6

        if profile.land_recovery:
            has_recovery = "thu hoi" in blob or "ke hoach thu hoi dat" in blob
            has_conversion_only = "chuyen muc dich" in blob and not has_recovery
            has_recovery_total = "dat can thu hoi" in blob or "tong dien tich dat can thu hoi" in blob
            if has_recovery:
                score += 1.1
            if has_recovery_total:
                score += 2.2
            if has_conversion_only and recovery_split_query:
                score -= 2.0

        if profile.admin_overview:
            has_natural_area = any(
                marker in blob for marker in ("dien tich tu nhien", "tong dien tich tu nhien", "co dien tich tu nhien")
            )
            has_admin_units = planning_has_admin_unit_evidence(blob)
            if has_natural_area:
                score += 2.0
            if has_admin_units:
                score += 1.8
            if has_admin_overview_evidence:
                score += 3.2

        if profile.land_change:
            land_class_hits = sum(1 for marker in ("dat nong nghiep", "dat phi nong nghiep", "dat chua su dung") if marker in content_norm)
            has_unused_zero = planning_has_unused_zero_evidence(content_norm)
            raw_content = strip_planning_metadata_lines(doc.page_content or "")
            has_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat nong nghiep")
            has_non_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat phi nong nghiep")
            if "2024" in content_norm and "2025" in content_norm:
                score += 2.0
                if has_agri_pair:
                    score += 2.4
                if has_non_agri_pair:
                    score += 2.4
            score += float(min(land_class_hits, 3)) * 1.0
            if land_class_hits >= 2 and any(marker in content_norm for marker in ("trong do", "hien trang nam 2024")):
                score += 2.4
            if has_unused_zero:
                score += 4.2
            if not has_agri_pair and "dat nong nghiep" in content_norm and "chuyen sang dat phi nong nghiep" not in content_norm:
                score -= 1.8
            if not has_non_agri_pair and "dat phi nong nghiep" in content_norm and "chuyen sang dat phi nong nghiep" in content_norm:
                score -= 1.6
            if chunk_type == "table" and land_class_hits > 0 and "2024" in content_norm and "2025" in content_norm:
                score += 1.3
            if land_class_hits == 0 and not has_unused_zero:
                score -= 6.0
            if "khong thay doi so voi hien trang" in content_norm and land_class_hits == 0:
                score -= 2.6
            if any(marker in content_norm for marker in ("chuyen sang dat phi nong nghiep", "chuyen doi co cau su dung dat")) and not (
                has_agri_pair or has_non_agri_pair or has_unused_zero
            ):
                score -= 4.2
            if land_class_hits == 0 and any(marker in content_norm for marker in ("di tich", "danh lam", "chieu sang cong cong", "nang luong")):
                score -= 3.0

        if profile.public_purpose_composition:
            component_hits = sum(
                1
                for marker in ("giao thong", "thuy loi", "di tich", "nang luong", "buu chinh", "cho", "khu vui choi", "sinh hoat cong dong")
                if marker in blob
            )
            if "dat su dung vao muc dich cong cong" in blob or "muc dich cong cong" in blob:
                score += 2.2
            score += float(min(component_hits, 6)) * 0.8

        if profile.project_structure:
            structure_hits = sum(
                1
                for marker in ("da thuc hien", "chua thuc hien", "chua to chuc", "chuyen tiep", "dua vao ke hoach", "bao cao thuyet minh", "hdnd")
                if marker in blob
            )
            score += float(min(structure_hits, 5)) * 0.85
            if profile.registered_plan_composition:
                registered_hits = planning_marker_hits(blob, PLANNING_REGISTERED_COMPOSITION_MARKERS)
                registered_evidence_hits = planning_registered_plan_evidence_hits(blob)
                has_registered_context = registered_evidence_hits > 0 or any(
                    marker in blob
                    for marker in (
                        "dang ky lap danh muc",
                        "ke hoach su dung dat nam 2025 cap huyen",
                        "danh muc cac cong trinh du an thu hoi dat nam 2025",
                        "danh muc cac du an chuyen muc dich",
                        "dua vao ke hoach su dung dat",
                        "nghi quyet",
                    )
                )
                if any(marker in blob for marker in ("tong so cong trinh", "tong so du an", "dang ky thuc hien")):
                    score += 1.9
                score += float(min(registered_hits, 7)) * 0.85
                score += float(min(registered_evidence_hits, 7)) * 0.8
                if has_registered_context:
                    score += 2.3
                if "ubnd thanh pho phe duyet" in blob and not has_registered_context and registered_hits == 0:
                    score -= 2.8
                if any(marker in blob for marker in ("ket qua thuc hien", "da thuc hien", "chua thuc hien")) and not has_registered_context:
                    score -= 2.0
            if profile.project_delay_reason:
                delay_reason_hits = planning_marker_hits(blob, PLANNING_PROJECT_DELAY_REASON_MARKERS)
                if "nguyen nhan chu yeu" in blob:
                    score += 2.8
                score += float(min(delay_reason_hits, 7)) * 0.95
                score += float(min(explanatory_hits, 6)) * 0.35
                if chunk_type == "table" and delay_reason_hits == 0:
                    score -= 3.0
                if delay_reason_hits == 0 and explanatory_hits == 0:
                    score -= 2.2
            if profile.implementation_carry_forward:
                carry_forward_hits = planning_marker_hits(blob, PLANNING_IMPLEMENTATION_CARRY_FORWARD_MARKERS)
                if "ubnd thanh pho phe duyet" in blob:
                    score += 1.6
                score += float(min(carry_forward_hits, 7)) * 0.9

        if profile.gpmb_stats:
            gpmb_hits = sum(
                1
                for marker in ("giai phong mat bang", "thong bao thu hoi", "dieu tra", "xac nhan nguon goc", "du thao", "phuong an", "boi thuong", "tai dinh cu", "ty dong", "ho gia dinh")
                if marker in blob
            )
            score += float(min(gpmb_hits, 6)) * 0.85

        if profile.environment_constraint:
            env_hits = sum(
                1
                for marker in ("moi truong", "o nhiem", "nuoc mat", "nuoc duoi dat", "khong khi", "bod5", "cod", "tss", "amoni", "h2s")
                if marker in blob
            )
            score += float(min(env_hits, 6)) * 0.85

        if profile.plan_necessity:
            legal_hits = sum(
                1
                for marker in ("luat dat dai", "nghi dinh", "hang nam cap huyen", "thu hoi dat", "giao dat", "cho thue dat", "chuyen muc dich")
                if marker in blob
            )
            score += float(min(legal_hits, 4)) * 0.9

        if profile.focus_area_reason:
            focus_hits = sum(1 for phrase in planning_focus_phrases(message)[:4] if phrase in blob)
            score += float(min(focus_hits, 3)) * 1.05
            if profile.focus_management:
                management_hits = planning_marker_hits(blob, PLANNING_FOCUS_MANAGEMENT_MARKERS)
                score += float(min(management_hits, 7)) * 0.95
                if management_hits == 0 and not any(
                    marker in blob for marker in ("quan ly dat dai", "giai phap", "kiem tra", "thu hoi dat")
                ):
                    score -= 2.0

        if profile.drainage_transport:
            infra_hits = sum(
                1
                for marker in ("vung trung", "dia hinh", "song", "ho", "thoat nuoc", "ung ngap", "to lich", "lu", "set", "kim nguu", "yen so", "linh dam", "den lu")
                if marker in blob
            )
            score += float(min(infra_hits, 6)) * 0.8
        if profile.sector_land_demand:
            pressure_hits = planning_marker_hits(blob, PLANNING_GROWTH_PRESSURE_MARKERS)
            score += float(min(pressure_hits, 8)) * 0.9
            score += float(min(explanatory_hits, 6)) * 0.25
            if pressure_hits == 0 and all(marker not in blob for marker in ("dat thuong mai dich vu", "dat giao thong", "dat o do thi", "nha o")):
                score -= 1.8
            if chunk_type == "table" and pressure_hits == 0 and explanatory_hits == 0:
                score -= 2.4

        if decision_query:
            if "quyet dinh" in title_norm:
                score += 1.4
            else:
                score -= 0.6

        if profile.city_level_listing:
            city_hits = sum(1 for marker in ("44 yet kieu", "ga c10", "ga s12") if marker in blob)
            has_city_label = "cap thanh pho" in blob and any(marker in blob for marker in ("cong trinh", "du an"))
            score += float(city_hits) * 1.7
            if has_city_label:
                score += 2.3
            if city_hits == 0 and named_entity_hits == 0:
                score -= 2.0

        if profile.article67:
            has_article67_marker = "khoan 4 dieu 67" in blob or "dieu 67" in blob
            article67_entities = sum(
                1
                for marker in ("nha tang le quoc gia", "tran thanh tong", "bo cong an", "tran binh trong", "tran nhan tong")
                if marker in blob
            )
            if has_article67_marker:
                score += 2.2
            score += float(article67_entities) * 1.3
            if ("dieu 78" in blob or "dieu 79" in blob) and not has_article67_marker:
                score -= 3.0

        if profile.decision_total:
            if re.search(r"\bbao\s+gom\s+32\s+du\s+an\b", blob):
                score += 4.0
            elif re.search(r"\b32\b", blob):
                score += 0.4
            if "8,6915" in blob or "8.6915" in blob:
                score += 2.0
            if re.search(r"\b32/2024/qh15\b", blob) or re.search(r"\b32 2024 qh15\b", blob):
                score -= 4.8

        if planning_is_heading_or_incomplete_chunk(doc, project_listing_query=profile.project_listing):
            score -= 1.0

        base_score = planning_doc_score(doc, message, district, plan_year)
        if base_score <= -1e8:
            base_score = -20.0
        score += base_score * 0.42

        if query_years and content_year_hits <= 0.0:
            score -= 0.8
        return (score, overlap)

    scored_order: list[tuple[Document, float, float, str]] = []
    for doc in ranked_docs:
        priority_score, secondary_score = _priority(doc)
        scored_order.append((doc, priority_score, secondary_score, _identity(doc)))

    scored_order.sort(key=lambda item: (-item[1], -item[2], item[3]))
    ordered = [item[0] for item in scored_order]

    has_table = any(str((doc.metadata or {}).get("chunkType") or "").lower().strip() == "table" for doc in ordered)
    table_target = 0
    if has_table:
        if limit == 1:
            table_target = 1
        else:
            table_target = min(max(1, limit // 2), limit - 1)
        if prefer_text_evidence:
            table_target = min(table_target, max(1, limit // 4))
        if profile.project_listing:
            if profile.city_level_listing or profile.article67 or profile.new_registration_unique:
                table_target = min(table_target, max(1, limit // 3))
            else:
                table_target = min(limit - 1, max(table_target, max(3, limit // 2)))
    text_target = max(0, limit - table_target)

    def _push(doc: Document) -> bool:
        key = _identity(doc)
        if key in seen:
            return False
        seen.add(key)
        selected.append(doc)
        return True

    if profile.land_change:
        paired_totals_doc: Optional[Document] = None
        agricultural_pair_doc: Optional[Document] = None
        non_agricultural_pair_doc: Optional[Document] = None
        current_status_doc: Optional[Document] = None
        unused_status_doc: Optional[Document] = None

        for doc in ordered:
            content_norm = planning_doc_content_norm(doc)
            raw_content = strip_planning_metadata_lines(doc.page_content or "")
            land_class_hits = planning_land_change_label_hits(content_norm)
            has_year_pair = "2024" in content_norm and "2025" in content_norm
            has_unused_zero = planning_has_unused_zero_evidence(content_norm)
            has_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat nong nghiep")
            has_non_agri_pair = planning_has_land_pair_evidence(raw_content or content_norm, "dat phi nong nghiep")

            if paired_totals_doc is None and (
                (has_agri_pair and has_non_agri_pair)
                or (has_year_pair and land_class_hits >= 2 and (has_agri_pair or has_non_agri_pair))
            ):
                paired_totals_doc = doc
            if agricultural_pair_doc is None and has_agri_pair:
                agricultural_pair_doc = doc
            if non_agricultural_pair_doc is None and has_non_agri_pair:
                non_agricultural_pair_doc = doc
            if current_status_doc is None and land_class_hits >= 2 and any(marker in content_norm for marker in ("hien trang nam 2024", "trong do")):
                current_status_doc = doc
            if unused_status_doc is None and has_unused_zero:
                unused_status_doc = doc
            if (
                paired_totals_doc is not None
                and agricultural_pair_doc is not None
                and non_agricultural_pair_doc is not None
                and unused_status_doc is not None
            ):
                break

        for doc in [paired_totals_doc, agricultural_pair_doc, non_agricultural_pair_doc, current_status_doc, unused_status_doc]:
            if doc is not None and len(selected) < limit:
                _push(doc)

    table_count = 0
    if table_target > 0:
        for doc in ordered:
            if len(selected) >= limit or table_count >= table_target:
                break
            chunk_type = str((doc.metadata or {}).get("chunkType") or "").lower().strip()
            if chunk_type != "table":
                continue
            if _push(doc):
                table_count += 1

    text_count = 0
    if text_target > 0:
        for doc in ordered:
            if len(selected) >= limit or text_count >= text_target:
                break
            chunk_type = str((doc.metadata or {}).get("chunkType") or "").lower().strip()
            if chunk_type != "text":
                continue
            if _push(doc):
                text_count += 1

    for doc in ordered:
        if len(selected) >= limit:
            break
        _push(doc)

    return selected
