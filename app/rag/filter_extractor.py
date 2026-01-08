from __future__ import annotations
from typing import Any
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
        return filters
    except Exception as e:
        print(f"Error extracting filters: {e}")
        return {}