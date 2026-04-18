from __future__ import annotations
from typing import Any
import re
import unicodedata

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class PropertyFilters(BaseModel):
    """Extracted filters from user query"""
    city: str | None = Field(default=None, description="Thành phố (Hà Nội, TP.HCM, Đà Nẵng...)")
    district: str | None = Field(default=None, description="Quận/Huyện")
    postType: str | None = Field(default=None, description="SALE or RENT or OTHER")
    priceMin: int | None = Field(default=None, description="Giá tối thiểu (VNĐ)")
    priceMax: int | None = Field(default=None, description="Giá tối đa (VNĐ)")
    areaMin: int | None = Field(default=None, description="Diện tích tối thiểu (m²)")
    areaMax: int | None = Field(default=None, description="Diện tích tối đa (m²)")
    bedrooms: int | None = Field(default=None, description="Số phòng ngủ")

FILTER_EXTRACTION_PROMPT = """Extract real estate search filters from the user's question in Vietnamese.

User question: {question}

Extract these filters if mentioned:
- city: Thành phố (Hà Nội, TP.HCM, Đà Nẵng...)
- district: Quận/Huyện
- postType: SALE (bán), RENT (cho thuê), or OTHER
- priceMin, priceMax: Giá (convert to VNĐ: tỷ = 1,000,000,000, triệu = 1,000,000)
- areaMin, areaMax: Diện tích (m²)
- bedrooms: Số phòng ngủ

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


def _normalize_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


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

    for field in ("priceMin", "priceMax", "areaMin", "areaMax"):
        value = _to_positive_int(filters.get(field))
        if value is not None:
            cleaned[field] = value

    if "priceMin" in cleaned and "priceMax" in cleaned and cleaned["priceMin"] > cleaned["priceMax"]:
        cleaned["priceMin"], cleaned["priceMax"] = cleaned["priceMax"], cleaned["priceMin"]

    if "areaMin" in cleaned and "areaMax" in cleaned and cleaned["areaMin"] > cleaned["areaMax"]:
        cleaned["areaMin"], cleaned["areaMax"] = cleaned["areaMax"], cleaned["areaMin"]

    return cleaned

async def extract_filters_from_query(question: str, llm) -> dict[str, Any]:
    """
    Use LLM to extract structured filters from natural language query
    
    Example:
    Input: "Tôi muốn tìm căn hộ 3 phòng ngủ ở Hà Nội giá dưới 5 tỷ"
    Output: {"city": "Hà Nội", "bedrooms": 3, "priceMax": 5000000000, "postType": "SALE"}
    """
    parser = JsonOutputParser(pydantic_object=PropertyFilters)
    
    prompt = ChatPromptTemplate.from_template(FILTER_EXTRACTION_PROMPT)
    
    chain = prompt | llm | parser
    
    try:
        result = await chain.ainvoke({
            "question": question,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Remove None values
        filters = {k: v for k, v in result.items() if v is not None}
        return _sanitize_extracted_filters(filters)
    except Exception as e:
        print(f"Error extracting filters: {e}")
        return {}