from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.db.pgvector import AsyncSessionLocal  # noqa: E402
from app.rag.resources import initialize_listing_vector_store  # noqa: E402
from app.utils.chunking import build_splitter  # noqa: E402


UTILITY_LABELS = {
    "COMMERCIAL_SHOPPING": "Thương mại/siêu thị",
    "HEALTHCARE": "Y tế",
    "EDUCATION": "Giáo dục",
    "TRANSPORT": "Giao thông",
    "PARK_PLAZA": "Công viên/quảng trường",
    "FINANCIAL": "Tài chính",
    "GOVERNMENT": "Cơ quan hành chính",
    "ENTERTAINMENT": "Giải trí",
    "DINING": "Ăn uống",
    "SPORTS": "Thể thao",
    "RELIGIOUS": "Tôn giáo",
    "FUEL": "Nhiên liệu",
    "ACCOMMODATION": "Lưu trú",
    "PARKING": "Bãi đỗ xe",
    "OTHER": "Khác",
}


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)).strip()))
    except Exception:
        return default


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_price(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "N/A"
    return f"{number:,.0f}".replace(",", ".")


def _utility_line(item: dict[str, Any]) -> str:
    name = _clean(item.get("name"))
    category = _clean(item.get("category"))
    parts = [f"{name} ({UTILITY_LABELS.get(category, category or 'Khác')})"]
    distance = _num(item.get("distanceM"))
    if distance is not None:
        parts.append(f"cách {distance:,.0f}m".replace(",", "."))
    travel_time = item.get("travelTimeS")
    if travel_time:
        parts.append(f"khoảng {int((int(travel_time) + 59) / 60)} phút")
    if item.get("location"):
        parts.append(_clean(item.get("location")))
    if item.get("note"):
        parts.append(_clean(item.get("note")))
    if item.get("isPrimary"):
        parts.append("tiện ích nổi bật")
    return f"• {' - '.join(parts)}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _build_content(row: dict[str, Any]) -> str:
    amenities = [_clean(item) for item in _as_list(row.get("amenities")) if _clean(item)]
    utilities = [item for item in _as_list(row.get("utilities")) if isinstance(item, dict) and _clean(item.get("name"))]
    utilities = utilities[:30]

    lines = [
        f"LISTING_ID: {row['post_id']}",
        f"Loại: {'Cần bán' if row.get('post_type') == 'SALE' else 'Cho thuê' if row.get('post_type') == 'RENT' else 'Khác'}",
        f"Danh mục: {_clean(row.get('category_name')) or 'N/A'}",
        "",
        "--- THÔNG TIN CHI TIẾT ---",
        _clean(row.get("post_title")),
        _clean(row.get("post_content")),
        _clean(row.get("description")),
        "",
        "--- ĐẶC ĐIỂM ---",
        f"Giá: {_format_price(row.get('price'))} VNĐ",
        f"Diện tích: {row.get('area') or 'N/A'} m²",
        f"Số phòng ngủ: {row.get('bedrooms') or 'N/A'}",
        f"Số phòng vệ sinh: {row.get('toilets') or 'N/A'}",
        f"Hướng: {_clean(row.get('orientation')) or 'N/A'}",
        f"Tình trạng nội thất: {_clean(row.get('furniture_status')) or 'N/A'}",
        "",
        "--- VỊ TRÍ ---",
        _clean(row.get("location")),
        f"Phường/Xã: {_clean(row.get('ward'))}" if row.get("ward") else "",
        f"Quận/Huyện: {_clean(row.get('district'))}" if row.get("district") else "",
        f"Thành phố: {_clean(row.get('province'))}" if row.get("province") else "",
    ]

    if amenities:
        lines.extend(["", "--- TIỆN ÍCH NỘI KHU ---", *[f"• {item}" for item in amenities]])
    if utilities:
        lines.extend(["", "--- TIỆN ÍCH XUNG QUANH ---", *[_utility_line(item) for item in utilities]])

    lines.extend(
        [
            "",
            "--- LIÊN HỆ NGƯỜI ĐĂNG ---",
            f"Tên: {_clean(row.get('owner_name')) or 'N/A'}",
            f"Số điện thoại: {_clean(row.get('owner_phone')) or 'N/A'}",
            f"Email: {_clean(row.get('owner_email')) or 'N/A'}",
        ]
    )

    return "\n".join(line for line in lines if line is not None).strip()


def _build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    utilities = [item for item in _as_list(row.get("utilities")) if isinstance(item, dict) and _clean(item.get("name"))]
    utility_tags = sorted({_clean(item.get("name")) for item in utilities if _clean(item.get("name"))})
    utility_categories = sorted({_clean(item.get("category")) for item in utilities if _clean(item.get("category"))})
    amenities = [_clean(item) for item in _as_list(row.get("amenities")) if _clean(item)]

    return {
        "postId": row.get("post_id"),
        "propertyId": row.get("property_id"),
        "postType": row.get("post_type"),
        "postStatus": "APPROVED",
        "title": _clean(row.get("post_title")),
        "postTitle": _clean(row.get("post_title")),
        "sourceUrl": f"/posts/{row.get('post_id')}",
        "city": _clean(row.get("province")) or None,
        "district": _clean(row.get("district")) or None,
        "ward": _clean(row.get("ward")) or None,
        "location": _clean(row.get("location")) or None,
        "price": _num(row.get("price")),
        "area": _num(row.get("area")),
        "bedrooms": row.get("bedrooms"),
        "categoryName": _clean(row.get("category_name")) or None,
        "amenities": amenities,
        "utilityTags": utility_tags,
        "utilityCategories": utility_categories,
        "utilitiesTop": [
            {
                "name": _clean(item.get("name")),
                "category": _clean(item.get("category")) or None,
                "categoryLabel": UTILITY_LABELS.get(_clean(item.get("category")), _clean(item.get("category")) or "Khác"),
                "distanceM": _num(item.get("distanceM")),
                "travelTimeS": item.get("travelTimeS"),
                "location": _clean(item.get("location")) or None,
                "note": _clean(item.get("note")) or None,
                "isPrimary": bool(item.get("isPrimary")),
            }
            for item in utilities[:12]
        ],
        "ownerName": _clean(row.get("owner_name")) or None,
        "ownerPhone": _clean(row.get("owner_phone")) or None,
        "ownerEmail": _clean(row.get("owner_email")) or None,
    }


async def _fetch_posts(limit: int | None, offset: int) -> list[dict[str, Any]]:
    limit_clause = "LIMIT :limit" if limit else ""
    params: dict[str, Any] = {"offset": offset}
    if limit:
        params["limit"] = limit

    sql = text(
        f"""
        WITH amenities_agg AS (
          SELECT
            pa.property_id,
            array_agg(a.name ORDER BY a.name) AS amenities
          FROM property_amenities pa
          JOIN amenities a ON a.id = pa.amenity_id
          WHERE pa.deleted_at IS NULL AND a.deleted_at IS NULL
          GROUP BY pa.property_id
        ),
        utilities_agg AS (
          SELECT
            pu.property_id,
            jsonb_agg(
              jsonb_build_object(
                'name', u.utility_name,
                'category', u.utility_category::text,
                'distanceM', pu.distance_m,
                'travelTimeS', pu.travel_time_s,
                'location', u.location,
                'note', pu.note,
                'isPrimary', COALESCE(pu.is_primary, false)
              )
              ORDER BY pu.distance_m NULLS LAST, u.utility_name
            ) AS utilities
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
          p.orientation,
          p.furniture_status,
          p.location,
          pr.name AS province,
          d.name AS district,
          w.name AS ward,
          cat.category_name,
          amenities_agg.amenities,
          utilities_agg.utilities,
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
        LEFT JOIN amenities_agg ON amenities_agg.property_id = p.id
        LEFT JOIN utilities_agg ON utilities_agg.property_id = p.id
        WHERE post.deleted_at IS NULL
          AND post.post_status = 'APPROVED'
          AND p.deleted_at IS NULL
          AND p.status = 'ACTIVE'
        ORDER BY post.id ASC
        {limit_clause} OFFSET :offset
        """
    )
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(sql, params)).fetchall()
    return [dict(row._mapping) for row in rows]


async def _count_posts() -> int:
    sql = text(
        """
        SELECT COUNT(*)
        FROM posts post
        JOIN properties p ON p.id = post.property_id
        WHERE post.deleted_at IS NULL
          AND post.post_status = 'APPROVED'
          AND p.deleted_at IS NULL
          AND p.status = 'ACTIVE'
        """
    )
    async with AsyncSessionLocal() as session:
        return int((await session.execute(sql)).scalar_one())


async def _delete_old_embeddings(post_ids: list[int]) -> int:
    if not post_ids:
        return 0
    collection_name = os.getenv("PGVECTOR_COLLECTION", "post_embeddings__multilingual_e5_base_ghr1")
    sql = text(
        """
        DELETE FROM langchain_pg_embedding embedding
        USING langchain_pg_collection collection
        WHERE embedding.collection_id = collection.uuid
          AND collection.name = :collection_name
          AND embedding.cmetadata->>'postId' = ANY(:post_ids)
        """
    )
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql, {"collection_name": collection_name, "post_ids": [str(pid) for pid in post_ids]})
        await session.commit()
        return int(result.rowcount or 0)


async def reingest(batch_size: int, limit: int | None, dry_run: bool) -> None:
    total = await _count_posts()
    target = min(total, limit) if limit else total
    print(f"Approved active posts: {total}. Target reingest: {target}. Dry run: {dry_run}")
    if dry_run or target <= 0:
        return

    splitter = build_splitter(
        chunk_size=_int_env("LISTING_CHUNK_SIZE", 900),
        chunk_overlap=_int_env("LISTING_CHUNK_OVERLAP", 120),
    )
    vector_store = await initialize_listing_vector_store()

    processed = 0
    deleted_total = 0
    chunks_total = 0
    offset = 0

    while processed < target:
        remaining = target - processed
        rows = await _fetch_posts(limit=min(batch_size, remaining), offset=offset)
        if not rows:
            break

        post_ids = [int(row["post_id"]) for row in rows]
        deleted_total += await _delete_old_embeddings(post_ids)

        docs: list[Document] = []
        for row in rows:
            content = _build_content(row)
            metadata = _build_metadata(row)
            for chunk_index, chunk in enumerate(splitter.split_text(content)):
                md = dict(metadata)
                md["chunkIndex"] = chunk_index
                docs.append(Document(page_content=chunk, metadata=md))

        if docs:
            ids = await vector_store.aadd_documents(docs)
            chunks_total += len(ids)

        processed += len(rows)
        offset += len(rows)
        print(f"Processed {processed}/{target} posts; deleted {deleted_total} old chunks; added {chunks_total} chunks")

    print(f"Done. Reingested posts={processed}, deletedChunks={deleted_total}, addedChunks={chunks_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ingest approved listing posts into the listing vector store.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(reingest(batch_size=max(1, args.batch_size), limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
