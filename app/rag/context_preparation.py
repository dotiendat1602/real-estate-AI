from __future__ import annotations

from typing import Any
import re

from langchain_core.documents import Document

from ..planning.profiles import (
    build_planning_query_profile,
)
from .query_intents import (
    BUSINESS_MARKERS as _BUSINESS_MARKERS,
    INVESTMENT_MARKERS as _INVESTMENT_MARKERS,
    STUDY_WORK_MARKERS as _STUDY_WORK_MARKERS,
    SUITABILITY_MARKERS as _SUITABILITY_MARKERS,
    build_query_intents as _build_query_intents,
    is_planning_fact_question as _is_planning_fact_question,
)
from .text_utils import normalize_text as _normalize_text, sanitize_llm_text
from .listing_processing import (
    extract_query_terms as _extract_query_terms,
    structured_highlights as _structured_highlights,
)

_CONTEXT_DOC_HEADER_PREFIXES = (
    "=== BAT DONG SAN",
    "=== PLANNING",
    "--- THÔNG TIN CHI TIẾT ---",
    "--- DAC DIEM ---",
    "--- VI TRI ---",
)

_CONTEXT_LINE_HINTS = (
    "phong ngu",
    "phong ve sinh",
    "dien tich",
    "noi that",
    "mat tien",
    "ngo",
    "o to",
    "thang may",
    "tien ich",
    "phu hop",
    "quy hoach",
    "dossier",
    "ma ho so",
    "plan year",
)


_STUDY_WORK_NOISE_MARKERS = (
    "ban cong",
    "san phoi",
    "giuong",
    "tu quan ao",
    "tu lanh",
    "may giat",
    "binh nong lanh",
    "cho ngay doi dien",
)




_ADDRESS_FRAGMENT_MARKERS = (
    "duong",
    "phuong",
    "quan",
    "huyen",
    "ha noi",
    "ngo",
    "so ",
)

_LISTING_BOILERPLATE_MARKERS = (
    "cho thue nha rieng tai",
    "cho thue nha rieng",
    "chinh chu",
    "uy tin",
    "thong tin chi tiet",
    "danh muc",
    "loai:",
)


_DIRECTION_LABELS = {
    "dong": "Dong",
    "tay": "Tay",
    "nam": "Nam",
    "bac": "Bac",
    "dong nam": "Dong Nam",
    "dong bac": "Dong Bac",
    "tay nam": "Tay Nam",
    "tay bac": "Tay Bac",
}

def _is_address_like_line(normalized_line: str) -> bool:
    marker_hits = sum(1 for marker in _ADDRESS_FRAGMENT_MARKERS if marker in normalized_line)
    return marker_hits >= 3 and len(normalized_line) >= 35


def _is_listing_boilerplate_line(normalized_line: str) -> bool:
    return any(marker in normalized_line for marker in _LISTING_BOILERPLATE_MARKERS)


def _dedupe_repeated_blocks(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        key = _normalize_text(block)[:220]
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return "\n\n".join(out)


def _split_long_line(line: str) -> list[str]:
    prepared = line
    for marker in (
        "Giá thuê",
        "Giá:",
        "Liên hệ",
        "Ưu điểm",
        "Thông tin nhà",
        "Tiện ích",
        "Diện tích",
        "Kết cấu",
        "Gồm",
        "Tổng cộng",
        "Nhà phù hợp",
        "Loại:",
        "Danh mục:",
    ):
        prepared = prepared.replace(marker, f"\n{marker}")

    parts = re.split(r"\n+|(?<=[\.!?;])\s+", prepared)
    return [part.strip(" -") for part in parts if part and part.strip()]


def _line_relevance_key(
    line: str,
    query_terms: set[str],
    number_terms: set[str],
    intents: dict[str, bool],
) -> tuple[int, int, int, int, int, int, int]:
    normalized_line = _normalize_text(line)
    if not normalized_line:
        return (-1, 0, 0, 0, 0, 0, 0)

    has_header = any(normalized_line.startswith(_normalize_text(prefix)) for prefix in _CONTEXT_DOC_HEADER_PREFIXES)
    term_hits = sum(1 for term in query_terms if term in normalized_line)
    number_hits = sum(1 for number in number_terms if number in line)
    hint_hits = sum(1 for hint in _CONTEXT_LINE_HINTS if hint in normalized_line)
    intent_hits = 0

    if intents.get("suitability_query"):
        intent_hits += sum(1 for marker in _SUITABILITY_MARKERS if marker in normalized_line)
    if intents.get("study_work_query"):
        intent_hits += sum(1 for marker in _STUDY_WORK_MARKERS if marker in normalized_line)
    if intents.get("explanatory_query"):
        intent_hits += sum(
            1
            for marker in (
                "phu hop",
                "duoc xem",
                "thuan tien",
                "dap ung",
                "uu diem",
                "han che",
                "vi tri",
                "tien ich",
                "noi that",
                "gia dinh",
                "kinh doanh",
                "dau tu",
            )
            if marker in normalized_line
        )

    noise = int("lien he" in normalized_line or re.search(r"\b0\d{8,10}\b", normalized_line) is not None)
    noise += int(not intents.get("needs_price") and (normalized_line.startswith("gia") or " gia:" in normalized_line))
    noise += int(normalized_line.startswith("loai") or normalized_line.startswith("danh muc"))
    noise += int(normalized_line.startswith("---") and not any(term in normalized_line for term in query_terms))
    noise += int(not intents.get("needs_direction") and (normalized_line.startswith("huong") or " huong:" in normalized_line))
    noise += int(intents.get("suitability_query") and not intents.get("needs_area") and normalized_line.startswith("dien tich"))
    noise += int(_is_listing_boilerplate_line(normalized_line))
    noise += int(not intents.get("needs_location") and _is_address_like_line(normalized_line))

    has_business_marker = any(marker in normalized_line for marker in _BUSINESS_MARKERS)
    business_signal = 0
    if has_business_marker:
        if intents.get("business_query") or intents.get("investment_query"):
            business_signal = 1
        elif intents.get("suitability_query"):
            business_signal = 1
        else:
            noise += 1

    if intents.get("study_work_query") and any(marker in normalized_line for marker in _STUDY_WORK_NOISE_MARKERS):
        noise += 1

    return (-noise, int(has_header), term_hits, number_hits, intent_hits, hint_hits, business_signal)


def _compact_doc_content(question: str, content: str, max_chars: int, max_lines: int = 8) -> str:
    if not content:
        return ""

    prepared = content.replace("<br/>", "\n").replace("<br>", "\n")
    prepared = _dedupe_repeated_blocks(prepared)
    if len(prepared) <= max_chars and prepared.count("\n") <= max_lines:
        return prepared

    raw_lines = [line.strip() for line in prepared.splitlines() if line.strip()]
    lines: list[str] = []
    for raw_line in raw_lines:
        if len(raw_line) > 200:
            lines.extend(_split_long_line(raw_line))
        else:
            lines.append(raw_line)

    # Remove near-duplicate fragments after sentence splitting.
    deduped_lines: list[str] = []
    deduped_seen: set[str] = set()
    for line in lines:
        key = _normalize_text(line)
        if not key or key in deduped_seen:
            continue
        deduped_seen.add(key)
        deduped_lines.append(line)
    lines = deduped_lines

    if not lines:
        return prepared[:max_chars]

    query_terms, number_terms = _extract_query_terms(question)
    intents = _build_query_intents(question)

    selected: list[str] = []
    selected_norm: set[str] = set()
    used_chars = 0

    # Keep the first section header when available so the model retains listing identity.
    for line in lines:
        if any(line.startswith(prefix) for prefix in _CONTEXT_DOC_HEADER_PREFIXES):
            key = _normalize_text(line)
            if key and key not in selected_norm:
                selected.append(line)
                selected_norm.add(key)
                used_chars += len(line) + 1
            break

    ranked_lines: list[tuple[tuple[int, int, int, int, int, int, int], int, str]] = []
    for idx, line in enumerate(lines):
        key = _line_relevance_key(line, query_terms, number_terms, intents)
        ranked_lines.append((key, idx, line))

    ranked_lines.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    for key, _, line in ranked_lines:
        if len(selected) >= max_lines:
            break
        if selected and key[0] < 0 and not any(key[2:]):
            continue

        normalized = _normalize_text(line)
        if not normalized or normalized in selected_norm:
            continue

        next_chars = used_chars + len(line) + 1
        if next_chars > max_chars:
            continue

        selected.append(line)
        selected_norm.add(normalized)
        used_chars = next_chars

    if not selected:
        # Safe fallback: keep a compact prefix if scoring filtered everything.
        return prepared[:max_chars]

    compact = "\n".join(selected).strip()
    return compact[:max_chars]


def _doc_identity(doc: Document) -> str:
    md = doc.metadata or {}
    return "|".join(
        [
            str(md.get("postId") or ""),
            str(md.get("propertyId") or ""),
            str(md.get("planningDocumentId") or ""),
            str(md.get("chunkType") or ""),
            str(md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex") or ""),
            _normalize_text((doc.page_content or "")[:120]),
        ]
    )


def _is_planning_context_doc(doc: Document) -> bool:
    md = doc.metadata or {}
    return md.get("planningDocumentId") is not None and md.get("postId") is None


def _format_price_vnd(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return f"{number:,} VND".replace(",", ".")


def _format_numeric_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return str(int(number))
    return str(round(number, 2))


def _extract_price_from_text(text: str) -> int | None:
    normalized = _normalize_text(text)

    patterns = (
        r"(?:gia|thue|ban|chi|hon)\s*[:\-]?\s*(\d+(?:[\.,]\d+)?)\s*ty",
        r"(\d+(?:[\.,]\d+)?)\s*ty",
        r"(?:gia|thue|ban|chi|hon)\s*[:\-]?\s*(\d+(?:[\.,]\d+)?)\s*trieu",
        r"(\d+(?:[\.,]\d+)?)\s*trieu",
    )

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        raw = match.group(1).replace(",", ".")
        try:
            amount = float(raw)
        except ValueError:
            continue
        if "ty" in pattern:
            return int(amount * 1_000_000_000)
        return int(amount * 1_000_000)

    return None


def _extract_area_from_text(text: str) -> float | None:
    normalized = _normalize_text(text)
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(?:m2|m²)", normalized)
    if not match:
        return None
    raw = match.group(1).replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _extract_bedrooms_from_text(text: str) -> int | None:
    normalized = _normalize_text(text)
    for pattern in (
        r"(\d+)\s*(?:phong ngu|pn)\b",
        r"\b(?:phong ngu|pn)\s*(\d+)\b",
    ):
        match = re.search(pattern, normalized)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _extract_bathrooms_from_text(text: str) -> int | None:
    normalized = _normalize_text(text)
    for pattern in (
        r"(\d+)\s*(?:phong ve sinh|wc|vs)\b",
        r"\b(?:phong ve sinh|wc|vs)\s*(\d+)\b",
    ):
        match = re.search(pattern, normalized)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def _extract_min_rental_period_from_text(text: str) -> str | None:
    normalized = _normalize_text(text)
    for pattern in (
        r"thoi gian (?:cho )?thue toi thieu\s*(\d+)\s*(thang|nam)",
        r"toi thieu\s*(\d+)\s*(thang|nam)",
    ):
        match = re.search(pattern, normalized)
        if match:
            amount = match.group(1)
            unit = match.group(2)
            return f"{amount} {unit}"
    return None


def _extract_monthly_cashflow_from_text(text: str) -> int | None:
    normalized = _normalize_text(text)
    match = re.search(r"(?:dong tien|cashflow|doanh thu)\s*(\d+(?:[\.,]\d+)?)\s*(trieu|tr|ty)\b", normalized)
    if not match:
        return None

    raw = match.group(1).replace(",", ".")
    unit = match.group(2)
    try:
        value = float(raw)
    except ValueError:
        return None

    if unit == "ty":
        return int(value * 1_000_000_000)
    return int(value * 1_000_000)


def _extract_district_from_text(text: str) -> str | None:
    if not text:
        return None

    match = re.search(r"(?:[Qq]uận|[Hh]uyện)\s+([^,;\n]+)", text)
    if not match:
        return None

    value = match.group(1).strip()
    value = re.split(r"\b(?:Hà Nội|Ho Chi Minh|Hồ Chí Minh)\b", value, maxsplit=1)[0].strip()
    if not value:
        return None

    tokens = value.split()
    if len(tokens) > 5:
        value = " ".join(tokens[:5])
    return value


def _extract_direction_from_text(text: str) -> str | None:
    normalized = _normalize_text(text)
    pattern = (
        r"(?:"
        r"huong(?:\s+cua\s+chinh|\s+ban\s+cong|\s+nha)?"
        r"|maindoordirection"
        r"|balconydirection"
        r"|direction"
        r")\s*(?:[:=]\s*)?"
        r"(dong\s+nam|dong\s+bac|tay\s+nam|tay\s+bac|dong|tay|nam|bac)\b"
    )
    match = re.search(pattern, normalized)
    if match:
        key = re.sub(r"\s+", " ", match.group(1)).strip()
        if key in _DIRECTION_LABELS:
            return _DIRECTION_LABELS[key]
    return None


def _humanize_post_type(value: str) -> str | None:
    normalized = _normalize_text(value)
    if not normalized:
        return None
    if normalized in {"rent", "cho thue", "thue"}:
        return "Cho thue"
    if normalized in {"sale", "can ban", "ban"}:
        return "Can ban"
    return value.strip() or None








def _merge_context_snippets(primary: str, secondary: str, max_chars: int) -> str:
    base = (primary or "").strip()
    addon = (secondary or "").strip()
    if not addon:
        return base
    if not base:
        return addon[:max_chars]

    merged_lines = [line for line in base.splitlines() if line.strip()]
    seen_norm = {_normalize_text(line) for line in merged_lines if _normalize_text(line)}

    for line in addon.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = _normalize_text(stripped)
        if not normalized or normalized in seen_norm:
            continue

        candidate = "\n".join([*merged_lines, stripped]).strip()
        if len(candidate) > max_chars:
            break

        merged_lines.append(stripped)
        seen_norm.add(normalized)

    return "\n".join(merged_lines).strip()


def _build_structured_listing_context(question: str, doc: Document) -> str | None:
    md = doc.metadata or {}
    post_id = md.get("postId")
    planning_document_id = md.get("planningDocumentId")
    if post_id is None and planning_document_id is None:
        return None

    # Planning documents should keep factual content lines instead of listing-style field projection.
    if planning_document_id is not None and post_id is None:
        return None

    text_blob = "\n".join(
        [
            str(md.get("title") or ""),
            str(md.get("variant") or ""),
            str(md.get("extra") or ""),
            str(md.get("address") or ""),
            str(md.get("district") or ""),
            str(md.get("mainDoorDirection") or ""),
            str(doc.page_content or ""),
        ]
    )

    lines: list[str] = []
    if post_id is not None:
        lines.append(f"=== BAT DONG SAN {post_id} ===")
    else:
        lines.append(f"=== PLANNING {planning_document_id} ===")

    title = str(md.get("title") or "").strip()
    if title:
        lines.append(f"- Tieu de: {title}")

    post_type = _humanize_post_type(str(md.get("postType") or ""))
    if post_type:
        lines.append(f"- Loai tin: {post_type}")

    district = str(md.get("district") or "").strip() or (_extract_district_from_text(text_blob) or "")
    if district:
        lines.append(f"- Quan/Huyen: {district}")

    address = str(md.get("address") or "").strip()
    if address:
        lines.append(f"- Vi tri: {address}")

    price = _format_price_vnd(md.get("price"))
    if not price:
        price = _format_price_vnd(_extract_price_from_text(text_blob))
    if price:
        lines.append(f"- Gia: {price}")

    area_val = md.get("area")
    if area_val is None:
        area_val = _extract_area_from_text(text_blob)
    area = _format_numeric_value(area_val)
    if area:
        lines.append(f"- Dien tich: {area} m2")

    bedrooms_val = md.get("bedrooms")
    if bedrooms_val is None:
        bedrooms_val = _extract_bedrooms_from_text(text_blob)
    bedrooms = _format_numeric_value(bedrooms_val)
    if bedrooms:
        lines.append(f"- So phong ngu: {bedrooms}")

    bathrooms_val = md.get("bathrooms")
    if bathrooms_val is None:
        bathrooms_val = _extract_bathrooms_from_text(text_blob)
    bathrooms = _format_numeric_value(bathrooms_val)
    if bathrooms:
        lines.append(f"- So phong ve sinh: {bathrooms}")

    furnishing = str(md.get("furnishing") or "").strip()
    if furnishing:
        lines.append(f"- Noi that: {furnishing}")

    direction = str(md.get("mainDoorDirection") or "").strip() or str(md.get("direction") or "").strip()
    if not direction:
        direction = _extract_direction_from_text(text_blob) or ""
    if direction:
        lines.append(f"- Huong: {direction}")

    min_rental_period = str(md.get("minRentalPeriod") or "").strip()
    if not min_rental_period:
        min_rental_period = _extract_min_rental_period_from_text(text_blob) or ""
    if min_rental_period:
        lines.append(f"- Thoi gian thue toi thieu: {min_rental_period}")

    cashflow = _format_price_vnd(md.get("monthlyCashflow"))
    if not cashflow:
        cashflow = _format_price_vnd(_extract_monthly_cashflow_from_text(text_blob))
    if cashflow:
        lines.append(f"- Dong tien: {cashflow}/thang")

    highlights = _structured_highlights(question, md.get("highlights"), prefer_broad=True)
    if highlights:
        lines.append(f"- Diem noi bat: {highlights}")

    raw_evidence = _compact_doc_content(question, doc.page_content or "", max_chars=1100, max_lines=16)
    if raw_evidence:
        lines.append("--- CHI TIET ---")
        lines.append(raw_evidence)

    # Keep structured path only when there is enough evidence.
    if len(lines) < 4:
        return None

    return "\n".join(lines)

def _planning_context_lines(context_text: str) -> list[str]:
    out: list[str] = []
    for raw_line in (context_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _normalize_text(line).startswith("tom tat "):
            continue
        if line.startswith("[") or line.startswith("==="):
            continue
        if line.lower().startswith(("document_type:", "district:", "plan_year:", "chunk_type:", "title:")):
            continue
        out.append(line)
    return out


def _planning_contract_markers(question: str) -> set[str]:
    profile = build_planning_query_profile(question)
    terms, _ = _extract_query_terms(question, max_terms=16)

    stopwords = {
        "nhu",
        "the",
        "nao",
        "trong",
        "cua",
        "theo",
        "sau",
        "khi",
        "duoc",
        "phan",
        "nhom",
        "bao",
        "nhieu",
        "nam",
        "quan",
        "huyen",
    }

    markers: set[str] = {
        "tong so",
        "tong cong",
        "dien tich",
        "du an",
        "cong trinh",
        "ha",
        "thu hoi",
        "chuyen muc dich",
        "quyet dinh",
        "nghi quyet",
        "bao cao thuyet minh",
    }

    if profile.project_structure or profile.implementation_carry_forward:
        markers.update({"da thuc hien", "chua thuc hien", "chuyen tiep", "chua to chuc", "dua vao ke hoach"})
    if profile.project_delay_reason:
        markers.update({"nguyen nhan", "thu tuc phe duyet", "bao cao kinh te ky thuat"})

    normalized_question = _normalize_text(question)
    if any(marker in normalized_question for marker in ("kiem tra", "giam sat", "cap nhat", "danh muc", "nhiem vu")):
        markers.update(
            {
                "kiem tra",
                "giam sat",
                "thuc hien ke hoach",
                "cap nhat",
                "danh muc",
                "bo sung danh muc",
                "du dieu kien",
                "phe duyet bo sung",
            }
        )

    markers.update(re.findall(r"\b20\d{2}\b", normalized_question))

    for term in terms:
        if len(term) < 4 or term in stopwords:
            continue
        markers.add(term)

    return markers


def _planning_evidence_source_label(doc: Document) -> str:
    md = doc.metadata or {}
    chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
    return (
        f"pid={md.get('planningDocumentId') or '?'}"
        f",chunk={md.get('chunkType') or '?'}"
        f",idx={chunk_idx if chunk_idx is not None else '?'}"
    )


def _build_planning_evidence_contract(question: str, docs: list[Document], max_facts: int = 12) -> str | None:
    if not docs or not _is_planning_fact_question(question):
        return None

    markers = _planning_contract_markers(question)
    if not markers:
        return None

    candidates: list[tuple[tuple[int, int, int, int], int, int, str, str]] = []
    for doc_rank, doc in enumerate(docs):
        lines = _planning_context_lines(doc.page_content or "")
        if not lines:
            continue

        source_label = _planning_evidence_source_label(doc)
        for line_idx, line in enumerate(lines):
            normalized = _normalize_text(line)
            if len(normalized) < 18:
                continue

            marker_hits = sum(1 for marker in markers if marker in normalized)
            if marker_hits <= 0:
                continue

            has_number = int(re.search(r"\b\d+(?:[\.,]\d+)?\b", normalized) is not None)
            has_core_fact = int(
                any(marker in normalized for marker in ("tong so", "tong cong", "dien tich", "du an", "cong trinh", "ha"))
            )
            concise = int(len(normalized) <= 260)
            candidates.append(((marker_hits, has_number, has_core_fact, concise), doc_rank, line_idx, line, source_label))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (tuple(-value for value in item[0]), item[1], item[2]))
    selected: list[tuple[str, str]] = []
    seen_lines: set[str] = set()

    for _, _, _, line, source_label in candidates:
        norm = _normalize_text(line)
        if norm in seen_lines:
            continue
        seen_lines.add(norm)
        selected.append((line.strip(), source_label))
        if len(selected) >= max(4, max_facts):
            break

    if len(selected) < 2:
        return None

    rows = [
        "EVIDENCE CONTRACT",
        "- Use only facts [F#] below for planning and numeric statements.",
        "- If a detail explicitly requested by the user is missing here, state it is unavailable in retrieved context.",
        "- Do not mention missing extra details that the user did not request.",
    ]

    for idx, (line, source_label) in enumerate(selected, start=1):
        compact_line = re.sub(r"\s+", " ", line).strip()
        if len(compact_line) > 260:
            compact_line = compact_line[:260].rstrip()
        rows.append(f"[F{idx}] {compact_line} (src: {source_label})")

    return "\n".join(rows)


def prepare_docs_for_context(
    question: str,
    docs: list[Document],
    max_docs: int = 4,
    max_chars_per_doc: int = 1400,
) -> list[Document]:
    if not docs:
        return []

    max_docs = max(1, int(max_docs))
    max_chars_per_doc = max(400, int(max_chars_per_doc))

    deduped: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        key = _doc_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    planning_only = bool(deduped) and all(_is_planning_context_doc(doc) for doc in deduped)
    planning_fact_query = planning_only and _is_planning_fact_question(question)

    if planning_only:
        # Planning docs have already been ranked/compacted in planning retrieval.
        selected = deduped[:max_docs]
    else:
        # Keep retriever order as the document ranking source; this avoids a
        # second keyword-based rerank layer inside context preparation.
        selected = deduped[:max_docs]

    # Keep listing context rich by default to reduce intent-keyword coupling.
    needs_rich_listing_context = not planning_only

    prepared: list[Document] = []
    seen_compacted_keys: set[str] = set()
    structured_post_ids: set[str] = set()

    for doc in selected:
        if _is_planning_context_doc(doc):
            compacted = (doc.page_content or "").strip()
            planning_char_limit = max(max_chars_per_doc, 2600)
            if len(compacted) > planning_char_limit:
                compacted = compacted[:planning_char_limit].rstrip()
        else:
            md = doc.metadata or {}
            structured = _build_structured_listing_context(question, doc)
            post_id_key = str(md.get("postId") or "")
            use_structured = bool(structured)
            if use_structured and post_id_key and post_id_key in structured_post_ids:
                use_structured = False

            if use_structured:
                compacted = structured or ""
                if post_id_key:
                    structured_post_ids.add(post_id_key)

                if needs_rich_listing_context:
                    raw_excerpt_max_lines = 16
                    raw_excerpt = _compact_doc_content(
                        question,
                        doc.page_content or "",
                        max_chars=max_chars_per_doc,
                        max_lines=raw_excerpt_max_lines,
                    )
                    compacted = _merge_context_snippets(compacted, raw_excerpt, max_chars=max_chars_per_doc)
            else:
                compacted = _compact_doc_content(question, doc.page_content or "", max_chars=max_chars_per_doc)

        if not compacted:
            continue

        compacted = sanitize_llm_text(compacted)
        if not compacted:
            continue

        if _is_planning_context_doc(doc):
            md = doc.metadata or {}
            planning_chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
            compacted_key = "|".join(
                [
                    "planning",
                    str(md.get("planningDocumentId") or ""),
                    str(md.get("chunkType") or ""),
                    str(planning_chunk_idx or ""),
                ]
            )
        else:
            compacted_key = _normalize_text(compacted)[:260]

        if compacted_key and compacted_key in seen_compacted_keys:
            continue
        if compacted_key:
            seen_compacted_keys.add(compacted_key)

        prepared.append(Document(page_content=compacted, metadata=doc.metadata))

    if prepared:
        if planning_only and planning_fact_query:
            contract = _build_planning_evidence_contract(question, prepared)
            if contract:
                prepared = [
                    Document(
                        page_content=contract,
                        metadata={
                            "documentScope": "planning",
                            "isPlanningEvidenceContract": True,
                        },
                    ),
                    *prepared,
                ]
        return prepared

    # Final fallback keeps at least one document so generation/evaluation remains grounded.
    fallback = selected[0]
    return [Document(page_content=(fallback.page_content or "")[:max_chars_per_doc], metadata=fallback.metadata)]


def build_citations(docs: list[Document]) -> list[dict[str, Any]]:
    out = []
    seen_post_ids = set()
    
    for d in docs:
        md = d.metadata or {}
        if md.get("isPlanningEvidenceContract"):
            continue
        post_id = md.get("postId")
        
        # Deduplicate by postId (avoid showing same post multiple times)
        if post_id and post_id in seen_post_ids:
            continue
        
        if post_id:
            seen_post_ids.add(post_id)
        
        out.append({
            "postId": post_id,
            "propertyId": md.get("propertyId"),
            "planningDocumentId": md.get("planningDocumentId"),
            "title": md.get("title"),
            "postTitle": md.get("postTitle"),
            "sourceUrl": md.get("sourceUrl"),
            "format": md.get("format"),
            "documentScope": md.get("documentScope"),
            "documentType": md.get("documentType"),
            "dossierCode": md.get("dossierCode"),
            "planYear": md.get("planYear"),
            "chunkType": md.get("chunkType"),
            "chunkIndex": md.get("chunkIndex"),
            "globalChunkIndex": md.get("globalChunkIndex"),
            "pageNumber": md.get("pageNumber"),
            "lineStart": md.get("lineStart"),
            "lineEnd": md.get("lineEnd"),
            "sourceLocator": md.get("sourceLocator"),
            "chunker": md.get("chunker"),
            "postType": md.get("postType"),
            "categoryName": md.get("categoryName"),
            "city": md.get("city"),
            "district": md.get("district"),
            "ward": md.get("ward"),
            "location": md.get("location"),
            "price": md.get("price"),
            "area": md.get("area"),
            "bedrooms": md.get("bedrooms"),
            "amenities": md.get("amenities", []),
            "snippet": (d.page_content or "")[:300],
        })
    
    return out


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        sanitize_llm_text(doc.page_content)
        for doc in docs
        if doc.page_content
    )
