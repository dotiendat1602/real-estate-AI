from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from evaluation.runners.generate_goldens_from_db import (
    _assign_ids,
    _collect_planning_docs,
    _collect_posts,
    _dedupe_by_input,
    _dedupe_conversation_by_turns,
    _filter_out_id_style_conversations,
    _filter_out_id_style_single,
    _friendly_city,
    _friendly_district,
    _load_metadata_rows,
)


def _parse_metadata_rows(raw_rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_rows:
        if not row:
            continue
        raw = row[0]
        if not raw or raw in seen:
            continue
        seen.add(raw)
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
    return out


def _load_balanced_metadata_rows(sample_limit: int) -> list[dict[str, Any]]:
    # First pass reuses the existing loader to keep compatibility with current env setup.
    primary_rows = _load_metadata_rows(sample_limit=sample_limit)
    if len(_collect_planning_docs(primary_rows)) >= 5 and len(_collect_posts(primary_rows)) >= 10:
        return primary_rows

    load_dotenv()
    pgvector_url = os.getenv("PGVECTOR_URL")
    if not pgvector_url:
        return primary_rows

    extra_limit = max(sample_limit, 2000)
    engine = create_engine(pgvector_url)
    with engine.connect() as conn:
        random_rows = conn.execute(
            text("SELECT cmetadata::text FROM langchain_pg_embedding ORDER BY random() LIMIT :limit"),
            {"limit": extra_limit},
        ).fetchall()
        planning_rows = conn.execute(
            text(
                "SELECT cmetadata::text FROM langchain_pg_embedding "
                "WHERE cmetadata::text LIKE '%\"planningDocumentId\"%' LIMIT :limit"
            ),
            {"limit": extra_limit},
        ).fetchall()
        post_rows = conn.execute(
            text(
                "SELECT cmetadata::text FROM langchain_pg_embedding "
                "WHERE cmetadata::text LIKE '%\"postId\"%' LIMIT :limit"
            ),
            {"limit": extra_limit},
        ).fetchall()

    merged = _parse_metadata_rows([(json.dumps(row, ensure_ascii=False),) for row in primary_rows])
    merged.extend(_parse_metadata_rows(random_rows))
    merged.extend(_parse_metadata_rows(planning_rows))
    merged.extend(_parse_metadata_rows(post_rows))

    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in merged:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
    return deduped


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _format_money(value: Any) -> str:
    val = _to_float(value)
    if val is None:
        return "N/A"
    if val >= 1_000_000_000:
        return f"{val/1_000_000_000:.2f} tỷ"
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f} triệu"
    return f"{val:.0f}"


def _unit_price(price: Any, area: Any) -> float | None:
    p = _to_float(price)
    a = _to_float(area)
    if p is None or a is None or a <= 0:
        return None
    return p / a


def _format_location_parts(*parts: Any) -> str | None:
    cleaned = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    if not cleaned:
        return None
    return ", ".join(cleaned)


def _post_reference_explicit(post: dict[str, Any]) -> str:
    category = post.get("categoryName") or "bất động sản"
    location = _format_location_parts(post.get("ward"), post.get("district"), post.get("city"))

    descriptors: list[str] = []
    if post.get("area") is not None:
        descriptors.append(f"diện tích {post['area']} m2")
    if post.get("price") is not None:
        descriptors.append(f"giá {post['price']}")

    suffix = f" ({', '.join(descriptors)})" if descriptors else ""
    if location:
        return f"{category} tại {location}{suffix}"
    return f"{category} thiếu thông tin vị trí cụ thể{suffix}"


def _is_fake_post(post: dict[str, Any]) -> bool:
    ward = _normalize_key(post.get("ward"))
    city = _normalize_key(post.get("city"))
    return "ben nghe" in ward and "ho chi minh" in city


def _build_planning_index(planning_docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for doc in planning_docs:
        city_key = _normalize_key(_friendly_city(doc) or doc.get("city"))
        if not city_key:
            continue
        index.setdefault(city_key, []).append(doc)

    for city_docs in index.values():
        city_docs.sort(key=lambda d: d.get("planningDocumentId") or 0)

    return index


def _pick_planning_bundle_for_post(
    post: dict[str, Any],
    all_planning_docs: list[dict[str, Any]],
    planning_by_city: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None] | None:
    post_city_key = _normalize_key(post.get("city"))
    city_docs = planning_by_city.get(post_city_key) or [] if post_city_key else []
    candidate_docs = city_docs if len(city_docs) >= 2 else all_planning_docs
    if len(candidate_docs) < 2:
        return None

    post_district_key = _normalize_key(post.get("district"))

    def district_key(doc: dict[str, Any]) -> str:
        return _normalize_key(_friendly_district(doc) or doc.get("district"))

    base_index = 0
    post_id = post.get("postId")
    if isinstance(post_id, int) and candidate_docs:
        base_index = post_id % len(candidate_docs)

    same_district = [d for d in candidate_docs if post_district_key and district_key(d) == post_district_key]
    primary = same_district[0] if same_district else candidate_docs[base_index]

    primary_dk = district_key(primary)

    secondary = None
    for step in range(1, len(candidate_docs) + 1):
        doc = candidate_docs[(base_index + step) % len(candidate_docs)]
        if doc.get("planningDocumentId") == primary.get("planningDocumentId"):
            continue
        if district_key(doc) != primary_dk:
            secondary = doc
            break

    if secondary is None:
        for step in range(1, len(candidate_docs) + 1):
            doc = candidate_docs[(base_index + step) % len(candidate_docs)]
            if doc.get("planningDocumentId") != primary.get("planningDocumentId"):
                secondary = doc
                break

    if secondary is None:
        return None

    secondary_dk = district_key(secondary)
    tertiary = None
    for step in range(2, len(candidate_docs) + 2):
        doc = candidate_docs[(base_index + step) % len(candidate_docs)]
        doc_id = doc.get("planningDocumentId")
        if doc_id in {primary.get("planningDocumentId"), secondary.get("planningDocumentId")}:
            continue
        dk = district_key(doc)
        if dk and dk not in {primary_dk, secondary_dk}:
            tertiary = doc
            break

    return primary, secondary, tertiary


def _planning_summary(doc: dict[str, Any], alias: str) -> str:
    district = _friendly_district(doc) or doc.get("district") or "N/A"
    city = _friendly_city(doc) or doc.get("city") or "N/A"
    return (
        f"{alias}: planningDocumentId={doc.get('planningDocumentId')}, "
        f"planYear={doc.get('planYear') or 'N/A'}, dossierCode={doc.get('dossierCode') or 'N/A'}, "
        f"district={district}, city={city}, title={doc.get('title') or 'N/A'}"
    )


def _post_summary(post: dict[str, Any]) -> str:
    return (
        f"post: postId={post.get('postId')}, postType={post.get('postType') or 'N/A'}, "
        f"categoryName={post.get('categoryName') or 'N/A'}, city={post.get('city') or 'N/A'}, "
        f"district={post.get('district') or 'N/A'}, ward={post.get('ward') or 'N/A'}, "
        f"price={post.get('price') if post.get('price') is not None else 'N/A'}, "
        f"area={post.get('area') if post.get('area') is not None else 'N/A'}, "
        f"bedrooms={post.get('bedrooms') if post.get('bedrooms') is not None else 'N/A'}"
    )


def _context_bundle(post: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any] | None) -> list[str]:
    context = [_planning_summary(p1, "planningA"), _planning_summary(p2, "planningB")]
    if p3 is not None:
        context.append(_planning_summary(p3, "planningC"))
    context.append(_post_summary(post))
    return context


def build_single_turn_hard_why_goldens(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    planning_docs = _collect_planning_docs(rows)
    posts = [p for p in _collect_posts(rows) if not _is_fake_post(p)]
    planning_by_city = _build_planning_index(planning_docs)

    candidates: list[dict[str, Any]] = []

    for post in posts:
        bundle = _pick_planning_bundle_for_post(post, planning_docs, planning_by_city)
        if bundle is None:
            continue
        p1, p2, p3 = bundle

        ref = _post_reference_explicit(post)
        p1_district = _friendly_district(p1) or p1.get("district") or "khu vực A"
        p2_district = _friendly_district(p2) or p2.get("district") or "khu vực B"
        p3_district = _friendly_district(p3) if p3 else None
        p1_title = p1.get("title") or "tài liệu quy hoạch A"
        p2_title = p2.get("title") or "tài liệu quy hoạch B"

        price_txt = _format_money(post.get("price"))
        area_val = post.get("area")
        bed_val = post.get("bedrooms")
        unit = _unit_price(post.get("price"), post.get("area"))
        unit_txt = f"{unit:,.0f}" if unit is not None else "N/A"

        context = _context_bundle(post, p1, p2, p3)

        candidates.append(
            {
                "difficulty": "hard",
                "question_type": "why",
                "domain": "cross_planning_real_estate",
                "input": (
                    f"Vì sao khi phân tích tin {ref}, mình nên ưu tiên đối chiếu "
                    f"quy hoạch của {p1_district} thay vì {p2_district}?"
                ),
                "expected_output_outline": [
                    "So sánh mức độ khớp location giữa tin bất động sản và từng tài liệu quy hoạch",
                    "Giải thích rủi ro suy luận sai khi dùng tài liệu khác quận/huyện",
                    "Kết luận cần ưu tiên tài liệu khớp district/city/planYear trước khi nhận định",
                ],
                "context": context,
                "target_metadata": {
                    "postId": post.get("postId"),
                    "planningDocumentIds": [
                        p1.get("planningDocumentId"),
                        p2.get("planningDocumentId"),
                        p3.get("planningDocumentId") if p3 else None,
                    ],
                },
            }
        )

        candidates.append(
            {
                "difficulty": "hard",
                "question_type": "why",
                "domain": "cross_planning_real_estate",
                "input": (
                    f"Vì sao kết luận cho tin {ref} có thể bị lệch nếu chỉ bám vào văn bản \"{p2_title}\" "
                    f"mà không kiểm tra chéo với \"{p1_title}\"?"
                ),
                "expected_output_outline": [
                    "Nêu logic kiểm tra chéo giữa hai văn bản quy hoạch khác khu vực",
                    "Làm rõ vai trò của district/city/dossierCode trong việc tránh nhầm tài liệu",
                    "Đưa ra lý do vì sao cần xác nhận tính liên quan trước khi đưa khuyến nghị",
                ],
                "context": context,
                "target_metadata": {
                    "postId": post.get("postId"),
                    "planningDocumentIds": [
                        p1.get("planningDocumentId"),
                        p2.get("planningDocumentId"),
                        p3.get("planningDocumentId") if p3 else None,
                    ],
                },
            }
        )

        candidates.append(
            {
                "difficulty": "hard",
                "question_type": "why",
                "domain": "cross_planning_real_estate",
                "input": (
                    f"Vì sao với tin {ref} (giá {price_txt}, diện tích {area_val if area_val is not None else 'N/A'} m2, "
                    f"{bed_val if bed_val is not None else 'N/A'} phòng ngủ), vẫn cần kiểm tra đồng thời planYear và dossierCode "
                    f"từ nhiều tài liệu quy hoạch trước khi đánh giá mức độ phù hợp?"
                ),
                "expected_output_outline": [
                    "Giải thích vì sao dữ liệu giá/diện tích/phòng ngủ chưa đủ để kết luận quy hoạch",
                    "Nêu bước xác minh planYear + dossierCode trên tài liệu đúng khu vực",
                    "Thể hiện lập luận thận trọng khi dữ liệu nguồn mâu thuẫn hoặc khác quận",
                ],
                "context": context,
                "target_metadata": {
                    "postId": post.get("postId"),
                    "estimatedUnitPrice": unit_txt,
                    "planningDocumentIds": [
                        p1.get("planningDocumentId"),
                        p2.get("planningDocumentId"),
                        p3.get("planningDocumentId") if p3 else None,
                    ],
                },
            }
        )

        if p3_district:
            candidates.append(
                {
                    "difficulty": "hard",
                    "question_type": "why",
                    "domain": "cross_planning_real_estate",
                    "input": (
                        f"Vì sao trong bài toán của tin {ref}, việc so sánh chéo 3 nguồn quy hoạch "
                        f"({p1_district}, {p2_district}, {p3_district}) sẽ cho kết luận đáng tin hơn chỉ dùng 1 nguồn?"
                    ),
                    "expected_output_outline": [
                        "Phân tích lợi ích của đối chiếu đa nguồn để phát hiện sai lệch khu vực/năm",
                        "Nêu cách ưu tiên nguồn có mức độ khớp cao nhất với metadata của tin",
                        "Kết luận theo logic kiểm chứng thay vì suy đoán một nguồn",
                    ],
                    "context": context,
                    "target_metadata": {
                        "postId": post.get("postId"),
                        "planningDocumentIds": [
                            p1.get("planningDocumentId"),
                            p2.get("planningDocumentId"),
                            p3.get("planningDocumentId"),
                        ],
                    },
                }
            )

    unique_cases = _dedupe_by_input(_filter_out_id_style_single(candidates))
    if len(unique_cases) < count:
        raise RuntimeError(
            f"Chỉ tạo được {len(unique_cases)} câu single-turn hard/why từ dữ liệu hiện có, ít hơn yêu cầu {count}."
        )

    return _assign_ids(unique_cases[:count], "ST_HARD_WHY")


def build_conversation_hard_why_goldens(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    planning_docs = _collect_planning_docs(rows)
    posts = [p for p in _collect_posts(rows) if not _is_fake_post(p)]
    planning_by_city = _build_planning_index(planning_docs)

    candidates: list[dict[str, Any]] = []

    for post in posts:
        bundle = _pick_planning_bundle_for_post(post, planning_docs, planning_by_city)
        if bundle is None:
            continue
        p1, p2, p3 = bundle

        ref = _post_reference_explicit(post)
        p1_district = _friendly_district(p1) or p1.get("district") or "khu vực A"
        p2_district = _friendly_district(p2) or p2.get("district") or "khu vực B"
        p3_district = _friendly_district(p3) if p3 else None
        p1_title = p1.get("title") or "tài liệu quy hoạch A"
        p2_title = p2.get("title") or "tài liệu quy hoạch B"

        context = _context_bundle(post, p1, p2, p3)

        candidates.append(
            {
                "difficulty": "hard",
                "question_type": "why",
                "domain": "cross_planning_real_estate",
                "scenario": "Người dùng hỏi lý do ưu tiên tài liệu quy hoạch đúng khu vực, rồi hỏi thêm lý do cần kiểm chứng đa tiêu chí trước khi kết luận.",
                "expected_outcome": "Bot lập luận rõ vì sao phải ưu tiên tài liệu khớp location và vì sao cần đối chiếu planYear/dossierCode trước khi đưa nhận định cho tin bất động sản.",
                "turns": [
                    {
                        "role": "user",
                        "content": (
                            f"Vì sao khi thẩm định tin {ref}, mình phải ưu tiên quy hoạch của {p1_district} "
                            f"hơn tài liệu của {p2_district}?"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Vậy vì sao chỉ khớp quận thôi vẫn chưa đủ, mà còn phải kiểm tra thêm planYear và dossierCode "
                            f"trước khi đưa khuyến nghị?"
                        ),
                    },
                ],
                "context": context,
            }
        )

        candidates.append(
            {
                "difficulty": "hard",
                "question_type": "why",
                "domain": "cross_planning_real_estate",
                "scenario": "Người dùng hỏi vì sao không thể suy luận từ một văn bản quy hoạch đơn lẻ, sau đó yêu cầu giải thích logic kiểm tra chéo nhiều nguồn.",
                "expected_outcome": "Bot nêu được rủi ro của suy luận một nguồn và giải thích cách so sánh chéo nhiều tài liệu cùng với metadata tin bất động sản.",
                "turns": [
                    {
                        "role": "user",
                        "content": (
                            f"Vì sao nếu chỉ đọc mỗi \"{p2_title}\" thì kết luận cho tin {ref} có thể bị thiên lệch?"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Nếu phải kiểm tra chéo với \"{p1_title}\""
                            f"{f' và một tài liệu ở {p3_district}' if p3_district else ''}, vì sao mức độ tin cậy lại cao hơn?"
                        ),
                    },
                ],
                "context": context,
            }
        )

        candidates.append(
            {
                "difficulty": "hard",
                "question_type": "why",
                "domain": "cross_planning_real_estate",
                "scenario": "Người dùng hỏi vì sao thông tin giá/diện tích chưa đủ để kết luận, rồi hỏi tiếp vì sao cần gắn với nhiều văn bản quy hoạch để suy luận logic.",
                "expected_outcome": "Bot giải thích rõ giới hạn của metadata bất động sản đơn lẻ và lý do phải kết hợp nhiều tài liệu quy hoạch trước khi đánh giá.",
                "turns": [
                    {
                        "role": "user",
                        "content": (
                            f"Vì sao chỉ nhìn giá, diện tích và số phòng ngủ của tin {ref} vẫn chưa đủ để đánh giá rủi ro quy hoạch?"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Vậy vì sao cần ghép thêm ít nhất hai văn bản quy hoạch ({p1_district}, {p2_district}) "
                            f"để suy luận một cách logic hơn?"
                        ),
                    },
                ],
                "context": context,
            }
        )

    unique_cases = _dedupe_conversation_by_turns(_filter_out_id_style_conversations(candidates))
    if len(unique_cases) < count:
        raise RuntimeError(
            f"Chỉ tạo được {len(unique_cases)} câu conversational hard/why từ dữ liệu hiện có, ít hơn yêu cầu {count}."
        )

    return _assign_ids(unique_cases[:count], "MT_HARD_WHY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard/why goldens from current DB embeddings.")
    parser.add_argument("--single-count", type=int, default=50, help="Number of hard single-turn why goldens")
    parser.add_argument(
        "--conversation-count",
        type=int,
        default=50,
        help="Number of hard conversational why goldens",
    )
    parser.add_argument("--sample-limit", type=int, default=10000, help="Max embedding rows to inspect")
    parser.add_argument(
        "--single-output",
        default="single_turn_goldens_hard_why.json",
        help="Output filename under evaluation/datasets for single-turn set",
    )
    parser.add_argument(
        "--conversation-output",
        default="conversational_goldens_hard_why.json",
        help="Output filename under evaluation/datasets for conversational set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    single_path = repo_root / "evaluation" / "datasets" / args.single_output
    conv_path = repo_root / "evaluation" / "datasets" / args.conversation_output

    rows = _load_balanced_metadata_rows(sample_limit=args.sample_limit)
    single_turn = build_single_turn_hard_why_goldens(rows, count=args.single_count)
    conv = build_conversation_hard_why_goldens(rows, count=args.conversation_count)

    single_path.write_text(json.dumps(single_turn, ensure_ascii=False, indent=2), encoding="utf-8")
    conv_path.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")

    planning_count = len(_collect_planning_docs(rows))
    post_count = len(_collect_posts(rows))
    print(f"Found {planning_count} planning docs and {post_count} posts from embeddings metadata")
    print(f"Generated {len(single_turn)} hard/why single-turn goldens -> {single_path}")
    print(f"Generated {len(conv)} hard/why conversational goldens -> {conv_path}")


if __name__ == "__main__":
    main()
