from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import text

from ..db.pgvector import AsyncSessionLocal


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    lowered = (value or "").lower()
    return any(needle in lowered for needle in needles)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _format_price(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} tỷ".rstrip("0").rstrip(".")
    if number >= 1_000_000:
        return f"{number / 1_000_000:.0f} triệu"
    return f"{number:,.0f} VNĐ"


def _append_text_filter(
    clauses: list[str],
    params: dict[str, Any],
    *,
    key: str,
    value: Any,
    columns: tuple[str, ...],
) -> None:
    raw = _clean(value)
    if not raw:
        return
    params[key] = f"%{raw.lower()}%"
    clauses.append("(" + " OR ".join(f"lower(coalesce({col}, '')) LIKE :{key}" for col in columns) + ")")


def _has_structured_listing_intent(query: str, filters: dict[str, Any]) -> bool:
    if filters:
        return True
    query_lc = (query or "").lower()
    return _contains_any(
        query_lc,
        (
            "bài đăng",
            "bai dang",
            "căn hộ",
            "can ho",
            "chung cư",
            "chung cu",
            "nhà",
            "nha",
            "bán",
            "ban",
            "mua",
            "cho thuê",
            "cho thue",
            "thuê",
            "thue",
            "giá",
            "gia",
            "phòng ngủ",
            "phong ngu",
            "hà nội",
            "ha noi",
            "hà đông",
            "ha dong",
        ),
    )


async def search_listing_documents(
    query: str,
    filters: dict[str, Any] | None,
    *,
    k: int = 8,
    _disable_city_hint: bool = False,
) -> list[Document]:
    filters = dict(filters or {})
    clauses = [
        "post.deleted_at IS NULL",
        "(post.post_status = 'APPROVED' OR post.source = 'BATDONGSAN')",
        "p.deleted_at IS NULL",
        "p.status = 'ACTIVE'",
    ]
    params: dict[str, Any] = {"limit": max(1, min(int(k), 20))}

    post_type = _clean(filters.get("postType")).upper()
    query_lc = (query or "").lower()
    if post_type in {"SALE", "RENT", "OTHER"}:
        clauses.append("post.post_type = :post_type")
        params["post_type"] = post_type
    elif _contains_any(query_lc, ("cho thuê", "thue", "thuê")):
        clauses.append("post.post_type = 'RENT'")
    elif _contains_any(query_lc, ("bán", "ban", "mua")):
        clauses.append("post.post_type = 'SALE'")

    has_price_intent = (
        filters.get("priceMin") is not None
        or filters.get("priceMax") is not None
        or _contains_any(query_lc, ("gia", "giá", "ty", "tỷ", "trieu", "triệu", "re", "rẻ"))
    )
    if has_price_intent:
        clauses.append("p.price IS NOT NULL AND p.price > 0")

    if filters.get("priceMin") is not None:
        clauses.append("p.price >= :price_min")
        params["price_min"] = filters["priceMin"]
    if filters.get("priceMax") is not None:
        clauses.append("p.price <= :price_max")
        params["price_max"] = filters["priceMax"]

    if filters.get("areaMin") is not None:
        clauses.append("p.area >= :area_min")
        params["area_min"] = filters["areaMin"]
    if filters.get("areaMax") is not None:
        clauses.append("p.area <= :area_max")
        params["area_max"] = filters["areaMax"]

    if filters.get("bedrooms") is not None:
        clauses.append('p."bedroomNumber" >= :bedrooms')
        params["bedrooms"] = filters["bedrooms"]

    _append_text_filter(
        clauses,
        params,
        key="city",
        value=filters.get("city")
        or (None if _disable_city_hint else ("Hà Nội" if _contains_any(query_lc, ("hà nội", "ha noi")) else None)),
        columns=("pr.name", "p.location", "post.post_title", "post.post_content"),
    )
    _append_text_filter(
        clauses,
        params,
        key="district",
        value=filters.get("district") or ("Hà Đông" if _contains_any(query_lc, ("hà đông", "ha dong")) else None),
        columns=("d.name", "p.location", "post.post_title", "post.post_content"),
    )

    if _contains_any(query_lc, ("căn hộ", "can ho", "chung cư", "chung cu")):
        clauses.append("(lower(coalesce(cat.category_name, '')) LIKE '%căn hộ%' OR lower(coalesce(cat.category_name, '')) LIKE '%chung cư%' OR lower(coalesce(post.post_title, '')) LIKE '%căn hộ%')")

    if _contains_any(query_lc, ("thang máy", "thang may", "elevator")):
        clauses.append(
            "("
            "lower(coalesce(amenities.amenity_names, '')) LIKE '%thang máy%' "
            "OR lower(coalesce(post.post_title, '')) LIKE '%thang máy%' "
            "OR lower(coalesce(post.post_content, '')) LIKE '%thang máy%' "
            "OR lower(coalesce(p.description, '')) LIKE '%thang máy%' "
            "OR lower(coalesce(p.location, '')) LIKE '%thang máy%'"
            ")"
        )

    order_parts: list[str] = []
    if filters.get("bedrooms") is not None:
        order_parts.append('CASE WHEN p."bedroomNumber" = :bedrooms THEN 0 ELSE 1 END')
    if _contains_any(query_lc, ("giá tốt", "gia tot", "rẻ", "re")):
        order_parts.append("p.price ASC NULLS LAST")
    order_parts.append("post.created_at DESC")
    order_clause = ", ".join(order_parts)

    sql = text(
        f"""
        WITH amenities AS (
          SELECT
            pa.property_id,
            string_agg(a.name, ', ' ORDER BY a.name) AS amenity_names
          FROM property_amenities pa
          JOIN amenities a ON a.id = pa.amenity_id
          WHERE pa.deleted_at IS NULL AND a.deleted_at IS NULL
          GROUP BY pa.property_id
        )
        SELECT
          post.id AS post_id,
          p.id AS property_id,
          post.post_type,
          post.post_title,
          post.post_content,
          p.description,
          p.price,
          p.area,
          p."bedroomNumber" AS bedrooms,
          p."toiletNumber" AS toilets,
          p.furniture_status AS furniture_status,
          p.location,
          pr.name AS province,
          d.name AS district,
          w.name AS ward,
          cat.category_name,
          amenities.amenity_names
        FROM posts post
        JOIN properties p ON p.id = post.property_id
        LEFT JOIN provinces pr ON pr.id = p.province_id
        LEFT JOIN districts d ON d.id = p.district_id
        LEFT JOIN wards w ON w.id = p.ward_id
        LEFT JOIN categories cat ON cat.category_id = p.category_id
        LEFT JOIN amenities ON amenities.property_id = p.id
        WHERE {" AND ".join(clauses)}
        ORDER BY {order_clause}
        LIMIT :limit
        """
    )

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(sql, params)).fetchall()

    if not rows and (filters.get("city") or _contains_any(query_lc, ("hà nội", "ha noi"))) and not _disable_city_hint:
        relaxed_filters = dict(filters)
        relaxed_filters.pop("city", None)
        return await search_listing_documents(
            query,
            relaxed_filters,
            k=k,
            _disable_city_hint=True,
        )

    docs: list[Document] = []
    for row in rows:
        r = dict(row._mapping)
        location_parts = [r.get("location"), r.get("ward"), r.get("district"), r.get("province")]
        location = ", ".join(part for part in location_parts if part)
        content = "\n".join(
            [
                f"=== BẤT ĐỘNG SẢN {r.get('post_id')} ===",
                f"Loại: {'Cần bán' if r.get('post_type') == 'SALE' else 'Cho thuê' if r.get('post_type') == 'RENT' else 'Khác'}",
                f"Danh mục: {r.get('category_name') or 'N/A'}",
                "",
                "--- THÔNG TIN CHI TIẾT ---",
                _clean(r.get("post_title")),
                _clean(r.get("post_content"))[:900],
                _clean(r.get("description"))[:500],
                "",
                "--- ĐẶC ĐIỂM ---",
                f"Giá: {_format_price(r.get('price'))}",
                f"Diện tích: {r.get('area') or 'N/A'} m²",
                f"Số phòng ngủ: {r.get('bedrooms') or 'N/A'}",
                f"Số phòng vệ sinh: {r.get('toilets') or 'N/A'}",
                f"Nội thất: {r.get('furniture_status') or 'N/A'}",
                "",
                "--- VỊ TRÍ ---",
                location or "N/A",
                "",
                "--- TIỆN ÍCH ---",
                r.get("amenity_names") or "N/A",
            ]
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "postId": r.get("post_id"),
                    "propertyId": r.get("property_id"),
                    "postType": r.get("post_type"),
                    "postTitle": r.get("post_title"),
                    "title": r.get("post_title"),
                    "city": r.get("province"),
                    "district": r.get("district"),
                    "ward": r.get("ward"),
                    "location": r.get("location"),
                    "price": float(r["price"]) if r.get("price") is not None else None,
                    "area": float(r["area"]) if r.get("area") is not None else None,
                    "bedrooms": r.get("bedrooms"),
                    "categoryName": r.get("category_name"),
                    "snippet": content[:300],
                    "retrievalSource": "db_listing_fallback",
                },
            )
        )

    return docs


class ListingFallbackRetriever:
    def __init__(self, primary: Any, *, query: str, filters: dict[str, Any] | None, k: int) -> None:
        self.primary = primary
        self.query = query
        self.filters = dict(filters or {})
        self.k = max(1, int(k))

    async def ainvoke(self, query: str) -> list[Document]:
        docs = await self.primary.ainvoke(query)
        use_structured_fallback = _has_structured_listing_intent(self.query or query, self.filters)
        if len(docs) >= min(3, self.k) and not use_structured_fallback:
            return docs

        fallback_docs = await search_listing_documents(self.query or query, self.filters, k=self.k)
        if not fallback_docs:
            return docs

        seen: set[str] = set()
        merged: list[Document] = []
        candidates = [*fallback_docs, *docs] if use_structured_fallback else [*docs, *fallback_docs]
        for doc in candidates:
            md = doc.metadata or {}
            key = str(md.get("postId") or md.get("propertyId") or doc.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= self.k:
                break
        return merged
