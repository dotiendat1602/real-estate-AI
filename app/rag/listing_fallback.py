from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from sqlalchemy import text

from ..db.pgvector import AsyncSessionLocal
from ..utils.text import normalize_vietnamese_search_text, repair_mojibake


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    lowered = repair_mojibake(value or "").lower()
    normalized = normalize_vietnamese_search_text(lowered)
    return any(
        repair_mojibake(needle).lower() in lowered
        or normalize_vietnamese_search_text(needle) in normalized
        for needle in needles
    )


def _clean(value: Any) -> str:
    return repair_mojibake(str(value or "")).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.strip("{}").split(",") if item.strip()]
    return [value]


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


async def _filter_approved_listing_docs(docs: list[Document]) -> list[Document]:
    post_ids: set[int] = set()
    for doc in docs:
        post_id = (doc.metadata or {}).get("postId")
        try:
            if post_id is not None:
                post_ids.add(int(post_id))
        except (TypeError, ValueError):
            continue

    if not post_ids:
        return docs

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id
                    FROM posts
                    WHERE id = ANY(:post_ids)
                      AND deleted_at IS NULL
                      AND post_status = 'APPROVED'
                    """
                ),
                {"post_ids": list(post_ids)},
            )
        ).fetchall()

    approved_ids = {int(row._mapping["id"]) for row in rows}
    filtered: list[Document] = []
    for doc in docs:
        post_id = (doc.metadata or {}).get("postId")
        if post_id is None:
            filtered.append(doc)
            continue
        try:
            if int(post_id) in approved_ids:
                filtered.append(doc)
        except (TypeError, ValueError):
            continue
    return filtered


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
    return _contains_any(
        query or "",
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
            "tang",
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
        "post.post_status = 'APPROVED'",
        "p.deleted_at IS NULL",
        "p.status = 'ACTIVE'",
    ]
    params: dict[str, Any] = {"limit": max(1, min(int(k), 20))}

    post_id = filters.get("postId")
    if post_id is not None:
        try:
            params["post_id"] = int(post_id)
            clauses.append("post.id = :post_id")
        except (TypeError, ValueError):
            pass

    post_type = _clean(filters.get("postType")).upper()
    query_lc = (query or "").lower()
    if post_type in {"SALE", "RENT", "OTHER"}:
        clauses.append("post.post_type = :post_type")
        params["post_type"] = post_type
    elif _contains_any(query_lc, ("cho thuê", "cho thue", "thuê", "thue")):
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

    if filters.get("floorNumber") is not None:
        clauses.append('p."floorNumber" = :floor_number')
        params["floor_number"] = filters["floorNumber"]
    if filters.get("floorMin") is not None:
        clauses.append('p."floorNumber" >= :floor_min')
        params["floor_min"] = filters["floorMin"]
    if filters.get("floorMax") is not None:
        clauses.append('p."floorNumber" <= :floor_max')
        params["floor_max"] = filters["floorMax"]

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
        clauses.append(
            "("
            "lower(coalesce(cat.category_name, '')) LIKE '%căn hộ%' "
            "OR lower(coalesce(cat.category_name, '')) LIKE '%chung cư%' "
            "OR lower(coalesce(post.post_title, '')) LIKE '%căn hộ%'"
            ")"
        )

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
    if filters.get("floorNumber") is not None:
        order_parts.append('CASE WHEN p."floorNumber" = :floor_number THEN 0 ELSE 1 END')
    elif filters.get("floorMin") is not None and filters.get("floorMax") is not None:
        target_floor = int((int(filters["floorMin"]) + int(filters["floorMax"])) / 2)
        params["target_floor"] = target_floor
        order_parts.append('ABS(p."floorNumber" - :target_floor) ASC NULLS LAST')
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
        ),
        nearby_utilities AS (
          SELECT
            pu.property_id,
            string_agg(
              CONCAT(
                u.utility_category::text,
                ': ',
                COALESCE(u.utility_name, ''),
                CASE
                  WHEN pu.distance_m IS NOT NULL THEN CONCAT(' (cách ', ROUND(pu.distance_m::numeric, 0)::text, 'm)')
                  ELSE ''
                END,
                CASE
                  WHEN pu.travel_time_s IS NOT NULL THEN CONCAT(', khoảng ', CEIL(pu.travel_time_s / 60.0)::int::text, ' phút')
                  ELSE ''
                END,
                CASE
                  WHEN u.location IS NOT NULL AND u.location <> '' THEN CONCAT(', ', u.location)
                  ELSE ''
                END,
                CASE
                  WHEN pu.note IS NOT NULL AND pu.note <> '' THEN CONCAT(', ', pu.note)
                  ELSE ''
                END
              ),
              E'\n' ORDER BY pu.distance_m NULLS LAST, u.utility_name
            ) AS utility_details,
            array_agg(DISTINCT u.utility_category::text) AS utility_categories
          FROM property_utilities pu
          JOIN utilities u ON u.id = pu.utility_id
          WHERE pu.deleted_at IS NULL AND u.deleted_at IS NULL
          GROUP BY pu.property_id
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
          p."floorNumber" AS floor_number,
          p.furniture_status AS furniture_status,
          p.location,
          pr.name AS province,
          d.name AS district,
          w.name AS ward,
          cat.category_name,
          amenities.amenity_names,
          nearby_utilities.utility_details,
          nearby_utilities.utility_categories,
          creator.name AS owner_name,
          creator.phone AS owner_phone,
          creator.email AS owner_email
        FROM posts post
        JOIN properties p ON p.id = post.property_id
        LEFT JOIN users creator ON creator.id = post.created_by_id
        LEFT JOIN provinces pr ON pr.id = p.province_id
        LEFT JOIN districts d ON d.id = p.district_id
        LEFT JOIN wards w ON w.id = p.ward_id
        LEFT JOIN categories cat ON cat.category_id = p.category_id
        LEFT JOIN amenities ON amenities.property_id = p.id
        LEFT JOIN nearby_utilities ON nearby_utilities.property_id = p.id
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
        location = ", ".join(_clean(part) for part in location_parts if part)
        content = "\n".join(
            [
                f"LISTING_ID: {r.get('post_id')}",
                f"Loại: {'Cần bán' if r.get('post_type') == 'SALE' else 'Cho thuê' if r.get('post_type') == 'RENT' else 'Khác'}",
                f"Danh mục: {_clean(r.get('category_name')) or 'N/A'}",
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
                f"Nội thất: {_clean(r.get('furniture_status')) or 'N/A'}",
                "",
                "--- VỊ TRÍ ---",
                location or "N/A",
                "",
                "--- TIỆN ÍCH ---",
                _clean(r.get("amenity_names")) or "N/A",
                "",
                "--- TIỆN ÍCH XUNG QUANH ---",
                _clean(r.get("utility_details")) or "N/A",
                "",
                "--- LIÊN HỆ NGƯỜI ĐĂNG ---",
                f"Tên: {_clean(r.get('owner_name')) or 'N/A'}",
                f"Số điện thoại: {_clean(r.get('owner_phone')) or 'N/A'}",
                f"Email: {_clean(r.get('owner_email')) or 'N/A'}",
            ]
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "postId": r.get("post_id"),
                    "propertyId": r.get("property_id"),
                    "postType": r.get("post_type"),
                    "postStatus": "APPROVED",
                    "postTitle": _clean(r.get("post_title")),
                    "title": _clean(r.get("post_title")),
                    "sourceUrl": f"/posts/{r.get('post_id')}",
                    "city": _clean(r.get("province")),
                    "district": _clean(r.get("district")),
                    "ward": _clean(r.get("ward")),
                    "location": _clean(r.get("location")),
                    "price": float(r["price"]) if r.get("price") is not None else None,
                    "area": float(r["area"]) if r.get("area") is not None else None,
                    "bedrooms": r.get("bedrooms"),
                    "floorNumber": r.get("floor_number"),
                    "categoryName": _clean(r.get("category_name")),
                    "utilityCategories": _as_list(r.get("utility_categories")),
                    "nearbyUtilities": _clean(r.get("utility_details")),
                    "ownerName": _clean(r.get("owner_name")),
                    "ownerPhone": _clean(r.get("owner_phone")),
                    "ownerEmail": _clean(r.get("owner_email")),
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
        docs = await _filter_approved_listing_docs(await self.primary.ainvoke(query))
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
