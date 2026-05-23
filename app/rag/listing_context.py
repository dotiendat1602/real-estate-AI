from __future__ import annotations

from collections.abc import Callable
from typing import Any
import re

from langchain_core.documents import Document

from .text_utils import normalize_text as _normalize_text
from .listing_processing import structured_highlights as _structured_highlights

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








def merge_context_snippets(primary: str, secondary: str, max_chars: int) -> str:
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


def build_structured_listing_context(
    question: str,
    doc: Document,
    raw_evidence_builder: Callable[[str, Document], str] | None = None,
) -> str | None:
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
        lines.append(f"LISTING_ID: {post_id}")
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

    raw_evidence = raw_evidence_builder(question, doc) if raw_evidence_builder else ""
    if raw_evidence:
        lines.append("--- CHI TIET ---")
        lines.append(raw_evidence)

    # Keep structured path only when there is enough evidence.
    if len(lines) < 4:
        return None

    return "\n".join(lines)
