from __future__ import annotations

from collections.abc import Callable
from typing import Any
import re

from langchain_core.documents import Document

from .text_utils import normalize_text as _normalize_text
from .listing_processing import structured_highlights as _structured_highlights
from ..utils.text import repair_mojibake

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

_UTILITY_GROUP_MARKERS = {
    "education": (
        "education",
        "giao duc",
        "truong",
        "truong hoc",
        "mau giao",
        "tieu hoc",
        "thcs",
        "thpt",
        "dai hoc",
        "hoc vien",
    ),
    "shopping": (
        "commercial shopping",
        "shopping",
        "thuong mai",
        "sieu thi",
        "trung tam thuong mai",
        "mall",
        "aeon",
        "winmart",
        "vinmart",
    ),
    "park": ("park plaza", "park", "cong vien", "quang truong"),
    "healthcare": ("healthcare", "y te", "benh vien", "phong kham"),
    "transport": ("transport", "giao thong", "xe buyt", "bus", "metro", "ben xe"),
    "finance": ("financial", "tai chinh", "ngan hang", "atm"),
    "dining": ("dining", "an uong", "nha hang", "cafe"),
    "entertainment": ("entertainment", "giai tri", "rap phim"),
    "sports": ("sports", "the thao", "san bong"),
    "parking": ("parking", "bai do xe"),
}

_UTILITY_GROUP_LABELS = {
    "education": "truong hoc",
    "shopping": "sieu thi/trung tam thuong mai",
    "park": "cong vien",
    "healthcare": "y te",
    "transport": "giao thong",
    "finance": "tai chinh",
    "dining": "an uong",
    "entertainment": "giai tri",
    "sports": "the thao",
    "parking": "bai do xe",
}


def _clean_text(value: Any) -> str:
    return repair_mojibake(str(value or "")).strip()


def _normalized_repaired(value: Any) -> str:
    return _normalize_text(repair_mojibake(str(value or "")))


def _utility_groups_for_text(value: Any) -> set[str]:
    normalized = _normalized_repaired(value)
    if not normalized:
        return set()
    return {
        group
        for group, markers in _UTILITY_GROUP_MARKERS.items()
        if any(marker in normalized for marker in markers)
    }


def _requested_utility_groups(question: str) -> set[str]:
    requested = _utility_groups_for_text(question)
    normalized = _normalized_repaired(question)
    if not requested:
        return set()

    # Only treat utility group words as restrictive when the user is asking
    # about nearby facilities, not when the same words appear in a listing title.
    has_nearby_signal = any(
        marker in normalized
        for marker in ("gan", "xung quanh", "tien ich", "cach", "quanh day", "lan can")
    )
    if not has_nearby_signal:
        return set()
    return requested


def _format_distance_meters(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    if number >= 1000:
        km = number / 1000
        return f"cach {km:.1f} km".replace(".0", "")
    return f"cach {int(round(number))} m"


def _format_utility_item(item: dict[str, Any]) -> str:
    name = _clean_text(item.get("name"))
    category_label = _clean_text(item.get("categoryLabel") or item.get("category"))
    parts = [name]
    if category_label:
        parts.append(category_label)
    distance = _format_distance_meters(item.get("distanceM"))
    if distance:
        parts.append(distance)
    travel_time = item.get("travelTimeS")
    try:
        if travel_time:
            parts.append(f"khoang {max(1, int((int(travel_time) + 59) / 60))} phut")
    except (TypeError, ValueError):
        pass
    location = _clean_text(item.get("location"))
    if location:
        parts.append(location)
    return " - ".join(part for part in parts if part)


def _dedupe_utility_lines(lines: list[str], limit: int = 6) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    seen_names: set[str] = set()
    for line in lines:
        cleaned = _clean_text(line).strip(" -" + "\u2022")
        if not cleaned:
            continue
        key = _normalized_repaired(cleaned)
        name_key = _normalized_repaired(re.split(r"\s*(?:-|:|\()", cleaned, maxsplit=1)[0])
        if not key or key in seen or (name_key and name_key in seen_names):
            continue
        seen.add(key)
        if name_key:
            seen_names.add(name_key)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _nearby_utility_lines(question: str, md: dict[str, Any], content: str) -> tuple[list[str], set[str]]:
    requested_groups = _requested_utility_groups(question)
    if not requested_groups:
        return [], set()

    lines: list[str] = []
    found_groups: set[str] = set()

    utilities_top = md.get("utilitiesTop")
    if isinstance(utilities_top, list):
        for item in utilities_top:
            if not isinstance(item, dict):
                continue
            searchable = " ".join(
                _clean_text(item.get(field))
                for field in ("name", "category", "categoryLabel", "location", "note")
            )
            matched = _utility_groups_for_text(searchable) & requested_groups
            if not matched:
                continue
            found_groups.update(matched)
            lines.append(_format_utility_item(item))

    nearby_utilities = md.get("nearbyUtilities")
    if isinstance(nearby_utilities, str):
        for line in nearby_utilities.splitlines():
            matched = _utility_groups_for_text(line) & requested_groups
            if matched:
                found_groups.update(matched)
                lines.append(line)

    for raw_line in (content or "").replace("<br/>", "\n").replace("<br>", "\n").splitlines():
        matched = _utility_groups_for_text(raw_line) & requested_groups
        if not matched:
            continue
        found_groups.update(matched)
        lines.append(raw_line)

    return _dedupe_utility_lines(lines), found_groups


def _filter_raw_evidence_for_utility_question(question: str, raw_evidence: str) -> str:
    requested_groups = _requested_utility_groups(question)
    if not requested_groups or not raw_evidence:
        return raw_evidence

    kept: list[str] = []
    for raw_line in raw_evidence.splitlines():
        matched_groups = _utility_groups_for_text(raw_line)
        if matched_groups and not (matched_groups & requested_groups):
            continue
        kept.append(raw_line)

    return "\n".join(kept).strip()

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

    floor_number = _format_numeric_value(md.get("floorNumber"))
    if floor_number:
        lines.append(f"- So tang: {floor_number}")

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

    planning_status = str(md.get("planningStatus") or "").strip()
    if planning_status:
        planning_parts = [f"trang thai {planning_status}"]
        risk_level = str(md.get("planningRiskLevel") or "").strip()
        confidence_level = str(md.get("planningConfidenceLevel") or "").strip()
        planning_period = str(md.get("planningPeriod") or "").strip()
        dossier_code = str(md.get("planningDossierCode") or "").strip()
        dossier_name = str(md.get("planningDossierName") or "").strip()
        current_land_type = str(md.get("planningCurrentLandType") or "").strip()
        planned_land_type = str(md.get("planningPlannedLandType") or "").strip()
        explanation = str(md.get("planningExplanation") or "").strip()
        if risk_level:
            planning_parts.append(f"rui ro {risk_level}")
        if confidence_level:
            planning_parts.append(f"do tin cay {confidence_level}")
        if planning_period:
            planning_parts.append(f"ky {planning_period}")
        if dossier_code or dossier_name:
            planning_parts.append(f"ho so {dossier_code} {dossier_name}".strip())
        if current_land_type or planned_land_type:
            planning_parts.append(
                f"loai dat hien trang/quy hoach: {current_land_type or 'N/A'} / {planned_land_type or 'N/A'}"
            )
        if explanation:
            planning_parts.append(f"ghi chu {explanation}")
        lines.append(f"- Doi chieu quy hoach cua can nay: {'; '.join(planning_parts)}")

    highlights = _structured_highlights(question, md.get("highlights"), prefer_broad=True)
    if highlights:
        lines.append(f"- Diem noi bat: {highlights}")

    nearby_utility_lines, found_utility_groups = _nearby_utility_lines(question, md, doc.page_content or "")
    requested_utility_groups = _requested_utility_groups(question)
    if nearby_utility_lines:
        lines.append(f"- Tien ich xung quanh lien quan: {'; '.join(nearby_utility_lines)}")
    missing_utility_groups = requested_utility_groups - found_utility_groups
    if missing_utility_groups:
        missing_labels = ", ".join(_UTILITY_GROUP_LABELS.get(group, group) for group in sorted(missing_utility_groups))
        lines.append(f"- Chua co thong tin ve {missing_labels} trong du lieu tien ich xung quanh.")

    raw_evidence = raw_evidence_builder(question, doc) if raw_evidence_builder else ""
    raw_evidence = _filter_raw_evidence_for_utility_question(question, raw_evidence)
    if raw_evidence:
        lines.append("--- CHI TIET ---")
        lines.append(raw_evidence)

    # Keep structured path only when there is enough evidence.
    if len(lines) < 4:
        return None

    return "\n".join(lines)
