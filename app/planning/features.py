from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document

from ..utils.text import normalize_text as _normalize_text
from .metadata import canonicalize_planning_district
from .profiles import PLANNING_REGISTERED_PLAN_EVIDENCE_MARKERS

PLANNING_ENTITY_MARKERS = (
    "tram y te",
    "truong thpt",
    "tru so bo cong an",
    "44 yet kieu",
    "ga c10",
    "ga s12",
    "nha tang le quoc gia",
    "tran thanh tong",
    "bo cong an",
    "tran binh trong",
    "tran nhan tong",
    "f/thpt1",
    "mai dich",
    "chuong duong",
)

_PLANNING_EXPLANATORY_EVIDENCE_MARKERS = (
    "nguyen nhan",
    "ly do",
    "boi",
    "do do",
    "vi vay",
    "muc tieu",
    "dinh huong",
    "yeu cau",
    "co so",
    "can cu",
    "giai phap",
    "khuyen nghi",
    "tac dong",
    "anh huong",
    "he qua",
    "rui ro",
    "uu tien",
    "thuc hien",
    "trien khai",
    "quy trinh",
    "trinh tu",
)




def is_planning_metadata_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False

    if stripped.startswith("[") and stripped.endswith("]") and "=" in stripped:
        return True

    if "|" not in stripped and ":" not in stripped:
        return False

    normalized = _normalize_text(stripped)
    if not normalized:
        return False

    metadata_keys = (
        "document scope",
        "document type",
        "district",
        "plan year",
        "chunk type",
        "title",
        "dossier code",
        "city",
    )
    key_hits = sum(1 for key in metadata_keys if key in normalized)
    if key_hits < 2:
        return False

    if "|" in stripped:
        return stripped.count("|") >= 2

    return key_hits >= 3 and len(normalized.split()) <= 48


def strip_planning_metadata_lines(content: str) -> str:
    if not content:
        return ""

    cleaned: list[str] = []
    for idx, line in enumerate(content.splitlines()):
        if idx < 4 and is_planning_metadata_line(line):
            continue
        cleaned.append(line)

    cleaned_content = "\n".join(cleaned).strip()
    if cleaned_content:
        return cleaned_content
    return content.strip()


def planning_doc_haystack(doc: Document) -> str:
    md = doc.metadata or {}
    content = strip_planning_metadata_lines(doc.page_content or "")
    district_value = canonicalize_planning_district(
        str(md.get("district") or ""),
        title=str(md.get("title") or ""),
        dossier_code=str(md.get("dossierCode") or ""),
    )
    parts = [
        district_value or str(md.get("district") or ""),
        str(md.get("districtRaw") or ""),
        str(md.get("title") or ""),
        str(md.get("sectionHeading") or ""),
        str(md.get("hierarchyPath") or ""),
        str(md.get("dossierCode") or ""),
        str(md.get("city") or ""),
        content[:1600],
    ]
    return _normalize_text(" ".join(parts))


def planning_named_entity_hits(normalized_text: str) -> int:
    if not normalized_text:
        return 0
    return sum(1 for marker in PLANNING_ENTITY_MARKERS if marker in normalized_text)


def planning_has_explicit_project_row(normalized_text: str) -> bool:
    if not normalized_text:
        return False

    return (
        re.search(r"\b(?:xay\s+dung|dau\s+tu|mo\s+rong)\b", normalized_text) is not None
        and (
            "du an" in normalized_text
            or "cong trinh" in normalized_text
            or planning_named_entity_hits(normalized_text) > 0
        )
    )


def planning_admin_unit_header_hits(normalized_text: str) -> int:
    if not normalized_text:
        return 0

    hits = 0
    seen: set[str] = set()
    for match in re.finditer(r"(?:phuong|xa)\s+[a-z0-9]+(?:\s+[a-z0-9]+){0,2}", normalized_text):
        candidate = re.sub(r"\s+", " ", match.group(0)).strip()
        if candidate in {"xa hoi", "phuong an", "phuong phap", "phuong huong"}:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        hits += 1
    return hits


def planning_has_direct_admin_unit_count_phrase(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    return re.search(r"(?:gom|co)\s+\d+\s+don\s+vi\s+hanh", normalized_text) is not None


def planning_has_direct_natural_area_phrase(normalized_text: str) -> bool:
    if not normalized_text:
        return False

    patterns = (
        r"tong\s+dien\s+tich\s+t?\s*u?\s*nhien(?:\s+nam\s+[\d\s]{4,6})?(?:\s+[a-z\s]{0,40})?\s+la\s+[\d\s\.,]+\s*ha",
        r"co\s+dien\s+tich\s+[a-z\s]{0,12}nhien\s+[\d\s\.,]+\s*ha",
    )
    return any(re.search(pattern, normalized_text) is not None for pattern in patterns)


def planning_explanatory_evidence_hits(haystack: str) -> int:
    return sum(1 for marker in _PLANNING_EXPLANATORY_EVIDENCE_MARKERS if marker in haystack)


def planning_registered_plan_evidence_hits(haystack: str) -> int:
    return sum(1 for marker in PLANNING_REGISTERED_PLAN_EVIDENCE_MARKERS if marker in haystack)


def planning_land_change_label_hits(normalized_text: str) -> int:
    if not normalized_text:
        return 0
    return sum(1 for marker in ("dat nong nghiep", "dat phi nong nghiep", "dat chua su dung") if marker in normalized_text)


def planning_is_heading_or_incomplete_chunk(doc: Document) -> bool:
    content = strip_planning_metadata_lines(doc.page_content or "")
    normalized = _normalize_text(content)
    if not normalized:
        return False
    if planning_has_explicit_project_row(normalized) and planning_named_entity_hits(normalized) > 0:
        return False

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return False

    if len(lines) <= 2 and not re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:ha|m2|%|ty)\b", normalized):
        if any(marker in normalized for marker in ("tong dien tich", "danh muc", "thu hoi", "du an", "cong trinh")):
            return True

    if normalized.endswith(":"):
        return True
    if any(normalized.endswith(marker) for marker in (" la", " gom", " cu the", " trong do")):
        return True
    if "dat can thu hoi" in normalized and "ha" not in normalized:
        return True
    return False


def planning_is_toc_like_chunk(raw_content: str, normalized_content: Optional[str] = None) -> bool:
    if not raw_content:
        return False

    normalized = normalized_content or _normalize_text(raw_content)
    if not normalized:
        return False
    if planning_has_explicit_project_row(normalized) or planning_named_entity_hits(normalized) > 0:
        return False

    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
    dotted_line_hits = len(re.findall(r"\.{2,}\s*\d+\s*$", raw_content, flags=re.MULTILINE))
    heading_ref_hits = len(re.findall(r"\b(?:bang|bieu|muc|chuong|phan|phu\s+luc)\s*\d+\b", normalized))
    numeric_evidence_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?\s*(?:ha|m2|%|ty|ty\s+dong)\b", normalized))
    toc_entry_hits = len(
        re.findall(
            r"(?:^|\s)(?:\d+\s*)?(?:bang|bieu|muc|chuong|phan|phu\s+luc)\s*[a-z0-9/\-\.]*",
            normalized,
        )
    )
    dense_numeric_row_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?(?:\s+\d+(?:[\.,]\d+)?){2,}\b", normalized))
    has_toc_keyword = "muc luc" in normalized or "danh muc" in normalized
    has_substantive_verbs = any(
        marker in normalized
        for marker in ("thu hoi", "chuyen muc dich", "giai phong", "dau gia", "thuc hien", "trien khai")
    )

    if dotted_line_hits >= 2 and numeric_evidence_hits == 0:
        return True
    if has_toc_keyword and heading_ref_hits >= 2 and numeric_evidence_hits == 0:
        return True
    if toc_entry_hits >= 2 and dense_numeric_row_hits == 0 and not has_substantive_verbs:
        return True
    if heading_ref_hits >= 2 and len(lines) <= 10 and numeric_evidence_hits == 0:
        return True
    if heading_ref_hits >= 4 and numeric_evidence_hits == 0 and not has_substantive_verbs:
        return True

    return False
