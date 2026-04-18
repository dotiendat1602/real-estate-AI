from __future__ import annotations

import re
import unicodedata
from typing import Optional

from langchain_core.documents import Document

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


def _strip_accents(text: str) -> str:
    normalized = (text or "").replace("Ä‘", "d").replace("Ä", "D")
    decomposed = unicodedata.normalize("NFD", normalized)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _normalize_text(text: str) -> str:
    lowered = _strip_accents((text or "").lower().strip())
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


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
        str(md.get("dossierCode") or ""),
        str(md.get("city") or ""),
        content[:1600],
    ]
    return _normalize_text(" ".join(parts))


def planning_doc_content_norm(doc: Document) -> str:
    return _normalize_text(strip_planning_metadata_lines(doc.page_content or ""))


def has_land_split_markers(normalized_text: str) -> bool:
    if not normalized_text:
        return False

    has_agri = re.search(r"\bdat\s+nong\s+nghi\s*ep\b", normalized_text) is not None or "dat nong nghiep" in normalized_text
    has_non_agri = (
        re.search(r"\bdat\s+phi\s+nong\s+nghi\s*ep\b", normalized_text) is not None
        or "dat phi nong nghiep" in normalized_text
    )
    return has_agri and has_non_agri


def planning_chunk_type_hint(doc: Document, content_norm: Optional[str] = None) -> str:
    md = doc.metadata or {}
    raw_chunk = _normalize_text(str(md.get("chunkType") or ""))
    if raw_chunk in {"table", "text"}:
        return raw_chunk

    if any(md.get(key) for key in ("tableId", "table_id", "tableIndex", "table_index", "tableName", "table_name")):
        return "table"

    raw_content = (doc.page_content or "")[:2400]
    if re.search(r"^\s*\|[^\n]*\|[^\n]*\|\s*$", raw_content, re.MULTILINE):
        return "table"
    if raw_content.count("\t") >= 4:
        return "table"

    normalized = content_norm or _normalize_text(raw_content)
    unit_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?\s*(?:ha|m2|%|ty|ty\s+dong)\b", normalized))
    header_hits = sum(
        1
        for marker in ("stt", "chi tieu", "ma dat", "loai dat", "tong cong", "don vi", "dien tich")
        if marker in normalized
    )
    if unit_hits >= 2 and header_hits >= 2:
        return "table"

    if re.search(r"\b(?:bang|bieu)\s*(?:so)?\s*\d+\b", normalized) and unit_hits >= 1 and header_hits >= 1:
        return "table"

    if raw_chunk:
        if "table" in raw_chunk or "bang" in raw_chunk:
            return "table"
        if "text" in raw_chunk:
            return "text"

    return "text"


def planning_count_pattern_score(normalized_text: str) -> float:
    if not normalized_text:
        return 0.0

    score = 0.0
    if re.search(r"\bbao\s+gom\s+\d+\s+du\s+an\b", normalized_text):
        score += 3.2
    if re.search(r"\btong\s+(?:so|cong)[^\n]{0,80}(?:du\s+an|cong\s+trinh)\b", normalized_text):
        score += 2.4
    if re.search(r"\b\d+(?:[\.,]\d+)?\s*ha\b", normalized_text):
        score += 0.9
    return score


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


def planning_has_admin_unit_evidence(normalized_text: str) -> bool:
    if not normalized_text:
        return False

    if any(marker in normalized_text for marker in ("don vi hanh chinh", "cap phuong", "cap xa")):
        return True

    admin_unit_hits = planning_admin_unit_header_hits(normalized_text)
    has_header_context = (
        re.search(r"\b(?:bang|bieu)\s+\d+\b", normalized_text) is not None
        and any(marker in normalized_text for marker in ("chi tieu", "phan bo", "phan theo"))
    )
    return admin_unit_hits >= 8 and has_header_context


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


def planning_has_natural_area_admin_evidence(normalized_text: str) -> bool:
    if not normalized_text:
        return False

    has_natural_area = any(
        marker in normalized_text
        for marker in (
            "dien tich tu nhien",
            "tong dien tich tu nhien",
            "co dien tich tu nhien",
        )
    )
    has_admin_units = planning_has_admin_unit_evidence(normalized_text)
    return has_natural_area and has_admin_units


def planning_explanatory_evidence_hits(haystack: str) -> int:
    return sum(1 for marker in _PLANNING_EXPLANATORY_EVIDENCE_MARKERS if marker in haystack)


def planning_registered_plan_evidence_hits(haystack: str) -> int:
    return sum(1 for marker in PLANNING_REGISTERED_PLAN_EVIDENCE_MARKERS if marker in haystack)


def planning_has_registered_resolution_count_evidence(haystack: str) -> bool:
    if not haystack:
        return False
    if not any(marker in haystack for marker in ("nghi quyet", "hdnd", "hoi dong nhan dan")):
        return False
    return re.search(r"\b\d{1,3}\s+(?:du\s*an|cong\s*trinh|danh\s*muc)\b", haystack) is not None


def planning_land_change_label_hits(normalized_text: str) -> int:
    if not normalized_text:
        return 0
    return sum(1 for marker in ("dat nong nghiep", "dat phi nong nghiep", "dat chua su dung") if marker in normalized_text)


def planning_has_land_pair_evidence(text: str, land_label: str) -> bool:
    if not text:
        return False

    normalized_text = _normalize_text(text)
    if land_label not in normalized_text:
        code_pattern = {
            "dat nong nghiep": "nnp",
            "dat phi nong nghiep": "pnn",
            "dat chua su dung": "csd",
        }.get(land_label)
        if code_pattern is None or code_pattern not in normalized_text:
            return False

    escaped_label = re.escape(land_label)
    code_pattern = {
        "dat nong nghiep": "nnp",
        "dat phi nong nghiep": "pnn",
        "dat chua su dung": "csd",
    }.get(land_label, "[a-z]{2,4}")
    number_pattern = r"\d+(?:[\s\.,]\d+)*"
    text_patterns = (
        rf"{escaped_label}\s+nam\s+2024[^\.:\n]{{0,140}}({number_pattern})\s*ha[^\.:\n]{{0,140}}nam\s+2025[^\.:\n]{{0,140}}({number_pattern})",
        rf"{escaped_label}\s+uoc\s+tinh\s+nam\s+2024[^\.:\n]{{0,140}}({number_pattern})\s*ha[^\.:\n]{{0,140}}nam\s+2025[^\.:\n]{{0,140}}({number_pattern})",
    )
    for pattern in text_patterns:
        if re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None:
            return True

    def _to_number(value: str) -> float | None:
        raw = (value or "").strip()
        if not raw:
            return None
        compact = raw.replace(" ", "").replace(".", "").replace(",", ".")
        try:
            return float(compact)
        except ValueError:
            return None

    table_match = re.search(
        rf"\b{code_pattern}\b\s+(\d[\d\.,]*)\s+\d[\d\.,]*\s+(\d[\d\.,]*)\s+\d[\d\.,]*",
        text,
        flags=re.IGNORECASE,
    )
    if table_match is None:
        table_match = re.search(
            rf"\b{code_pattern}\b\s+(\d+(?:\s+\d+)*)\s+\d+(?:\s+\d+)*\s+(\d+(?:\s+\d+)*)\s+\d+(?:\s+\d+)*",
            normalized_text,
            flags=re.IGNORECASE,
        )
    if table_match is None:
        return False

    first_value = _to_number(table_match.group(1))
    second_value = _to_number(table_match.group(2))
    return bool(first_value and second_value and first_value >= 100.0 and second_value >= 100.0)


def planning_has_unused_zero_evidence(normalized_text: str) -> bool:
    if not normalized_text or "dat chua su dung" not in normalized_text:
        return False
    return (
        "khong con dien tich" in normalized_text
        or "khong con" in normalized_text
        or re.search(r"\b0+(?:[\.,]0+)?\s*ha\b", normalized_text) is not None
    )


def planning_is_heading_or_incomplete_chunk(doc: Document, *, project_listing_query: bool = False) -> bool:
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
    if project_listing_query and planning_count_pattern_score(normalized) > 0 and planning_named_entity_hits(normalized) == 0:
        return True

    return False


def planning_continuation_signal(
    doc: Document,
    *,
    registered_plan_composition_query: bool = False,
    land_change_query: bool = False,
) -> float:
    content = strip_planning_metadata_lines(doc.page_content or "")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return 0.0

    first_line = lines[0]
    first_norm = _normalize_text(first_line)
    body_norm = _normalize_text(content)
    if not first_norm or not body_norm:
        return 0.0

    if re.match(r"^(?:chuong|phan|muc|bang|bieu|tt|stt|\d+(?:\.\d+)+)\b", first_norm):
        return 0.0

    starts_like_fragment = (
        first_line[:1] in {"+", "-", "(", ")"}
        or first_norm.startswith(("trong do", "tich ", "dien tich "))
        or (re.match(r"^[a-z]", first_norm) is not None and len(first_norm.split()) <= 10)
    )
    numeric_payload = len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", body_norm))
    has_area_value = re.search(r"\b\d+(?:[\.,]\d+)?\s*ha\b", body_norm) is not None
    if not starts_like_fragment or (numeric_payload < 2 and not has_area_value):
        return 0.0

    score = 1.8
    if registered_plan_composition_query and any(
        marker in body_norm for marker in ("dua vao ke hoach su dung dat", "du an", "dien tich", "nghi quyet")
    ):
        score += 2.0
    if land_change_query:
        land_change_label_hits = planning_land_change_label_hits(body_norm)
        if land_change_label_hits > 0:
            score += 2.2
        if planning_has_unused_zero_evidence(body_norm):
            score += 1.4
    return score


def planning_is_tabular_header_fragment(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    if not lines or len(lines) > 10:
        return False

    header_hits = sum(
        1
        for marker in ("2025/2024", "2024/2025", "tang (+)", "giam (-)", "dien tich", "co cau", "tt", "stt")
        if marker in normalized_text
    )
    numeric_hits = len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", normalized_text))
    return header_hits >= 4 and numeric_hits <= 2 and planning_land_change_label_hits(normalized_text) == 0


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
