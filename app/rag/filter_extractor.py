from __future__ import annotations

from typing import Any
import re

from pydantic import BaseModel, Field

from ..utils.text import normalize_vietnamese_search_text as _normalize_text
from .llm_usage import extract_token_usage, message_content_to_text


class PropertyFilters(BaseModel):
    """Extracted filters from user query."""

    city: str | None = Field(default=None, description="City, for example Ha Noi or TP.HCM")
    district: str | None = Field(default=None, description="District")
    postType: str | None = Field(default=None, description="SALE or RENT or OTHER")
    priceMin: int | None = Field(default=None, description="Minimum price in VND")
    priceMax: int | None = Field(default=None, description="Maximum price in VND")
    areaMin: int | None = Field(default=None, description="Minimum area in square meters")
    areaMax: int | None = Field(default=None, description="Maximum area in square meters")
    bedrooms: int | None = Field(default=None, description="Number of bedrooms")
    floorNumber: int | None = Field(default=None, description="Exact number of floors")
    floorMin: int | None = Field(default=None, description="Minimum number of floors")
    floorMax: int | None = Field(default=None, description="Maximum number of floors")


FILTER_EXTRACTION_PROMPT = """Extract real estate search filters from the user's question in Vietnamese.

User question: {question}

Extract these filters if mentioned:
- city: city/province, for example Ha Noi or TP.HCM
- district: district
- postType: SALE (ban/mua), RENT (cho thue/thue), or OTHER
- priceMin, priceMax: price, converted to VND. 1 ty = 1,000,000,000 and 1 trieu = 1,000,000
- areaMin, areaMax: area in square meters
- bedrooms: number of bedrooms
- floorNumber, floorMin, floorMax: number of floors. If the user says "khoang/tam/gan 4 tang", use floorMin=3 and floorMax=5. If exact, use floorNumber.

Return ONLY the filters that are explicitly mentioned. Use null for unmentioned fields.

{format_instructions}
"""

_ADDRESS_MARKERS = (
    "duong",
    "pho",
    "ngo",
    "ngach",
    "hem",
    "so",
    "dia chi",
)

_POST_TYPE_ALLOWED = {"SALE", "RENT", "OTHER"}


def _sanitize_text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _looks_like_address_fragment(value: str) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return False
    if any(marker in normalized for marker in _ADDRESS_MARKERS):
        return True
    # Values like "54/98" or "281 Truong Dinh" should not be used as district filters.
    if re.search(r"\d", normalized):
        return True
    return False


def _to_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _floor_filters_from_query(question: str) -> dict[str, int]:
    normalized = _normalize_text(question or "")
    if not normalized:
        return {}

    range_match = re.search(
        r"(?:tu|khoang)\s*(\d{1,2})\s*(?:den|-|toi)\s*(\d{1,2})\s*(?:tang|lau|floor)s?\b",
        normalized,
    )
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        if 0 < low <= 100 and 0 < high <= 100:
            return {"floorMin": min(low, high), "floorMax": max(low, high)}

    match = re.search(r"\b(\d{1,2})\s*(?:tang|lau|floor)s?\b", normalized)
    if not match:
        return {}

    floor = int(match.group(1))
    if not 0 < floor <= 100:
        return {}

    nearby_prefix = normalized[max(0, match.start() - 24) : match.start()]
    if any(marker in nearby_prefix for marker in ("khoang", "tam", "gan", "xap xi", "do")):
        return {"floorMin": max(1, floor - 1), "floorMax": floor + 1}

    return {"floorNumber": floor}


def _sanitize_extracted_filters(filters: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    city = _sanitize_text_value(filters.get("city"))
    if city and not _looks_like_address_fragment(city):
        cleaned["city"] = city

    district = _sanitize_text_value(filters.get("district"))
    if district and not _looks_like_address_fragment(district):
        cleaned["district"] = district

    post_type_raw = _sanitize_text_value(filters.get("postType"))
    if post_type_raw:
        normalized_post_type = post_type_raw.upper()
        if normalized_post_type in _POST_TYPE_ALLOWED:
            cleaned["postType"] = normalized_post_type

    bedrooms = _to_positive_int(filters.get("bedrooms"))
    if bedrooms is not None and bedrooms <= 20:
        cleaned["bedrooms"] = bedrooms

    floor_number = _to_positive_int(filters.get("floorNumber"))
    if floor_number is not None and floor_number <= 100:
        cleaned["floorNumber"] = floor_number

    for field in ("priceMin", "priceMax", "areaMin", "areaMax", "floorMin", "floorMax"):
        value = _to_positive_int(filters.get(field))
        if value is not None:
            cleaned[field] = value

    if "priceMin" in cleaned and "priceMax" in cleaned and cleaned["priceMin"] > cleaned["priceMax"]:
        cleaned["priceMin"], cleaned["priceMax"] = cleaned["priceMax"], cleaned["priceMin"]

    if "areaMin" in cleaned and "areaMax" in cleaned and cleaned["areaMin"] > cleaned["areaMax"]:
        cleaned["areaMin"], cleaned["areaMax"] = cleaned["areaMax"], cleaned["areaMin"]

    if "floorMin" in cleaned and "floorMax" in cleaned and cleaned["floorMin"] > cleaned["floorMax"]:
        cleaned["floorMin"], cleaned["floorMax"] = cleaned["floorMax"], cleaned["floorMin"]

    if "floorNumber" in cleaned and ("floorMin" in cleaned or "floorMax" in cleaned):
        cleaned.pop("floorNumber", None)

    return cleaned


async def extract_filters_from_query_with_usage(question: str, llm) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Use LLM to extract structured filters from natural language query.
    """
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    parser = JsonOutputParser(pydantic_object=PropertyFilters)
    filter_prompt = ChatPromptTemplate.from_template(FILTER_EXTRACTION_PROMPT)

    try:
        prompt_value = await filter_prompt.ainvoke(
            {
                "question": question,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        ai_message = await llm.ainvoke(prompt_value)
        result = parser.parse(message_content_to_text(getattr(ai_message, "content", "")))

        filters = {k: v for k, v in result.items() if v is not None}
        filters = {**filters, **_floor_filters_from_query(question)}
        return _sanitize_extracted_filters(filters), extract_token_usage(ai_message)
    except Exception as e:
        print(f"Error extracting filters: {e}")
        return _sanitize_extracted_filters(_floor_filters_from_query(question)), {}
