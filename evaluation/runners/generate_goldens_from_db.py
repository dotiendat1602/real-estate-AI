from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _load_metadata_rows(sample_limit: int) -> list[dict[str, Any]]:
    load_dotenv()
    pgvector_url = os.getenv("PGVECTOR_URL")
    if not pgvector_url:
        raise RuntimeError("PGVECTOR_URL is missing")

    engine = create_engine(pgvector_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                WITH planning_docs AS (
                    SELECT DISTINCT ON ((cmetadata::jsonb->>'planningDocumentId'))
                        cmetadata::text AS cmeta
                    FROM langchain_pg_embedding
                    WHERE (cmetadata::jsonb->>'planningDocumentId') IS NOT NULL
                    ORDER BY (cmetadata::jsonb->>'planningDocumentId')
                ),
                posts AS (
                    SELECT DISTINCT ON ((cmetadata::jsonb->>'postId'))
                        cmetadata::text AS cmeta
                    FROM langchain_pg_embedding
                    WHERE (cmetadata::jsonb->>'postId') IS NOT NULL
                    ORDER BY (cmetadata::jsonb->>'postId')
                ),
                merged AS (
                    SELECT cmeta FROM planning_docs
                    UNION ALL
                    SELECT cmeta FROM posts
                )
                SELECT cmeta FROM merged LIMIT :limit
                """
            ),
            {"limit": sample_limit},
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            out.append(json.loads(row[0]))
        except Exception:
            continue
    return out


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _unique_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        k = str(value)
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out


def _collect_planning_docs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    planning_rows = _unique_by_key(
        [r for r in rows if r.get("planningDocumentId") is not None],
        "planningDocumentId",
    )

    docs: list[dict[str, Any]] = []
    for row in planning_rows:
        doc_id = _to_int(row.get("planningDocumentId"))
        if doc_id is None:
            continue
        docs.append(
            {
                "planningDocumentId": doc_id,
                "planYear": _to_int(row.get("planYear")),
                "city": _normalize_text(row.get("city")),
                "district": _normalize_text(row.get("district")),
                "title": _normalize_text(row.get("title")),
                "dossierCode": _normalize_text(row.get("dossierCode")),
                "documentType": _normalize_text(row.get("documentType")),
            }
        )

    docs.sort(key=lambda x: x["planningDocumentId"])
    return docs


def _collect_posts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    post_rows = _unique_by_key([r for r in rows if r.get("postId") is not None], "postId")

    posts: list[dict[str, Any]] = []
    for row in post_rows:
        post_id = _to_int(row.get("postId"))
        if post_id is None:
            continue
        posts.append(
            {
                "postId": post_id,
                "postType": _normalize_text(row.get("postType")),
                "categoryName": _normalize_text(row.get("categoryName")),
                "city": _normalize_text(row.get("city")),
                "district": _normalize_text(row.get("district")),
                "ward": _normalize_text(row.get("ward")),
                "price": row.get("price"),
                "area": row.get("area"),
                "bedrooms": row.get("bedrooms"),
            }
        )

    posts.sort(key=lambda x: x["postId"])
    return posts


def _assign_ids(cases: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(cases, start=1):
        item = dict(item)
        item["id"] = f"{prefix}_{idx:03d}"
        out.append(item)
    return out


def _dedupe_by_input(cases: list[dict[str, Any]], key: str = "input") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        value = str(case.get(key) or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(case)
    return out


def _dedupe_conversation_by_turns(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        turns = case.get("turns") or []
        signature = json.dumps(turns, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(case)
    return out


def _format_location(*parts: str | None) -> str | None:
    cleaned = [str(p).strip() for p in parts if p and str(p).strip()]
    if not cleaned:
        return None
    return ", ".join(cleaned)


_DISTRICT_FRAGMENT_RE = re.compile(r"(quận|huyện|thị xã|thị trấn)\s+[^,;\-()]+", re.IGNORECASE)
_CITY_FRAGMENT_RE = re.compile(r"thành phố\s+[^,;\-()]+", re.IGNORECASE)


def _extract_fragment(pattern: re.Pattern[str], *values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = pattern.search(value)
        if match:
            return match.group(0).strip()
    return None


def _friendly_district(doc: dict[str, Any]) -> str | None:
    return _extract_fragment(_DISTRICT_FRAGMENT_RE, doc.get("district"), doc.get("title")) or _normalize_text(
        doc.get("district")
    )


def _friendly_city(doc: dict[str, Any]) -> str | None:
    extracted_city = _extract_fragment(_CITY_FRAGMENT_RE, doc.get("city"), doc.get("district"), doc.get("title"))
    return extracted_city or _normalize_text(doc.get("city"))


def _planning_subject(doc: dict[str, Any]) -> str:
    district = _friendly_district(doc)
    city = _friendly_city(doc)
    title = doc.get("title")
    year = doc.get("planYear")

    location = _format_location(district, city)
    if location and year is not None:
        return f"kế hoạch sử dụng đất năm {year} tại {location}"
    if location:
        return f"quy hoạch sử dụng đất tại {location}"
    if title:
        return f'văn bản "{title}"'
    return "văn bản quy hoạch sử dụng đất"


def _post_reference(post: dict[str, Any]) -> str:
    category = post.get("categoryName") or "bất động sản"
    location = _format_location(post.get("ward"), post.get("district"), post.get("city")) or "khu vực này"

    descriptors: list[str] = []
    if post.get("area") is not None:
        descriptors.append(f"diện tích {post['area']} m2")
    if post.get("price") is not None:
        descriptors.append(f"giá {post['price']}")

    suffix = f" ({', '.join(descriptors)})" if descriptors else ""
    return f"{category} ở {location}{suffix}"


def _contains_id_style_text(value: str) -> bool:
    normalized = f" {value.lower()} "
    return " id " in normalized or "planningdocumentid" in normalized or "postid" in normalized


def _sentence_case(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def _filter_out_id_style_single(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if not _contains_id_style_text(str(c.get("input") or ""))]


def _filter_out_id_style_conversations(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in cases:
        turns = case.get("turns") or []
        has_id_style = any(_contains_id_style_text(str(turn.get("content") or "")) for turn in turns)
        if has_id_style:
            continue
        out.append(case)
    return out


def build_single_turn_easy_goldens(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    planning_docs = _collect_planning_docs(rows)
    posts = _collect_posts(rows)

    candidates: list[dict[str, Any]] = []

    for doc in planning_docs:
        doc_id = doc["planningDocumentId"]
        year = doc.get("planYear")
        city = _friendly_city(doc)
        district = _friendly_district(doc)
        title = doc.get("title")
        dossier = doc.get("dossierCode")
        doc_type = doc.get("documentType")
        location = _format_location(district, city)
        subject = _planning_subject(doc)
        context_city = doc.get("city") or city or "N/A"
        context_district = doc.get("district") or district or "N/A"

        if title:
            year_question = f'Văn bản "{title}" áp dụng cho năm nào?'
            dossier_question = f'Mã hồ sơ của văn bản "{title}" là gì?'
            district_question = f'Văn bản "{title}" đang nói về quận/huyện nào?'
            city_question = f'Văn bản "{title}" thuộc thành phố nào?'
            doc_type_question = f'Văn bản "{title}" thuộc loại tài liệu gì?'
        else:
            year_question = f"{_sentence_case(subject)} áp dụng cho năm nào?"
            dossier_question = f"Mã hồ sơ của {subject} là gì?"
            if location:
                district_question = f"Kế hoạch sử dụng đất tại {location} đang nói về quận/huyện nào?"
            else:
                district_question = "Văn bản quy hoạch sử dụng đất đang nói về quận/huyện nào?"

            if district and year is not None:
                city_question = f"Kế hoạch sử dụng đất năm {year} tại {district} thuộc thành phố nào?"
            elif district:
                city_question = f"Khu vực {district} thuộc thành phố nào theo metadata quy hoạch?"
            else:
                city_question = "Văn bản quy hoạch sử dụng đất thuộc thành phố nào?"

            doc_type_question = f"Văn bản của {subject} thuộc loại tài liệu gì?"

        if year is not None:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "input": year_question,
                    "expected_output_outline": [
                        f"Nêu đúng planYear: {year}",
                        "Không suy đoán nếu thiếu dữ liệu",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, planYear={year}, district={context_district}, city={context_city}, title={title or 'N/A'}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

        if dossier:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "input": dossier_question,
                    "expected_output_outline": [
                        f"Nêu đúng dossierCode: {dossier}",
                        "Trả lời ngắn gọn, đúng metadata",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, dossierCode={dossier}, district={context_district}, city={context_city}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

        if district:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "input": district_question,
                    "expected_output_outline": [
                        f"Nêu đúng district: {district}",
                        "Không thêm địa danh ngoài dữ liệu",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, district={context_district}, city={context_city}, title={title or 'N/A'}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

        if city:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "input": city_question,
                    "expected_output_outline": [
                        f"Nêu đúng city: {city}",
                        "Trả lời một ý chính, không suy đoán",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, city={context_city}, district={context_district}, title={title or 'N/A'}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

        if doc_type:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "input": doc_type_question,
                    "expected_output_outline": [
                        f"Nêu đúng documentType: {doc_type}",
                        "Nếu thiếu thì nói rõ thiếu dữ liệu",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, documentType={doc_type}, district={context_district}, city={context_city}, title={title or 'N/A'}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

    for post in posts:
        post_id = post["postId"]
        post_type = post.get("postType")
        category = post.get("categoryName")
        city = post.get("city")
        district = post.get("district")
        ward = post.get("ward")
        price = post.get("price")
        area = post.get("area")
        bedrooms = post.get("bedrooms")
        location = _format_location(ward, district, city) or "khu vực này"
        reference = _post_reference(post)
        category_label = category or "bất động sản"

        if post_type:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "input": f"Tin {reference} là dạng SALE hay RENT vậy?",
                    "expected_output_outline": [
                        f"Nêu đúng postType: {post_type}",
                        "Không suy đoán nếu thiếu dữ liệu",
                    ],
                    "context": [f"postId={post_id}, postType={post_type}"],
                    "target_metadata": {"postId": post_id},
                }
            )

        if price is not None:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "input": f"Mức giá của tin {category_label} tại {location} là bao nhiêu?",
                    "expected_output_outline": [
                        f"Nêu đúng price: {price}",
                        "Trả lời ngắn gọn, không thêm nhận định",
                    ],
                    "context": [f"postId={post_id}, price={price}"],
                    "target_metadata": {"postId": post_id},
                }
            )

        if area is not None:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "input": f"Diện tích căn {category_label} ở {location} là bao nhiêu m2?",
                    "expected_output_outline": [
                        f"Nêu đúng area: {area}",
                        "Không suy đoán khi thiếu field",
                    ],
                    "context": [f"postId={post_id}, area={area}"],
                    "target_metadata": {"postId": post_id},
                }
            )

        if bedrooms is not None:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "input": f"Bất động sản tại {location} có bao nhiêu phòng ngủ?",
                    "expected_output_outline": [
                        f"Nêu đúng bedrooms: {bedrooms}",
                        "Không tự suy đoán nếu không có dữ liệu",
                    ],
                    "context": [f"postId={post_id}, bedrooms={bedrooms}"],
                    "target_metadata": {"postId": post_id},
                }
            )

        if category:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "input": f"Tin tại {location} thuộc danh mục bất động sản nào?",
                    "expected_output_outline": [
                        f"Nêu đúng categoryName: {category}",
                        "Bám đúng metadata, không diễn giải thêm",
                    ],
                    "context": [f"postId={post_id}, categoryName={category}"],
                    "target_metadata": {"postId": post_id},
                }
            )

        if city or district or ward:
            location_parts = [x for x in [ward, district, city] if x]
            location = ", ".join(location_parts) if location_parts else "N/A"
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "input": f"Vị trí cụ thể của tin {category_label} ở đâu?",
                    "expected_output_outline": [
                        f"Nêu đúng location theo metadata: {location}",
                        "Không bổ sung địa chỉ ngoài dữ liệu",
                    ],
                    "context": [
                        f"postId={post_id}, ward={ward or 'N/A'}, district={district or 'N/A'}, city={city or 'N/A'}"
                    ],
                    "target_metadata": {"postId": post_id},
                }
            )

    unique_cases = _dedupe_by_input(_filter_out_id_style_single(candidates))
    if not unique_cases:
        raise RuntimeError("Không thể tạo câu hỏi single-turn từ dữ liệu embeddings.")

    if len(unique_cases) < count:
        print(
            "[Generator] Chỉ tạo được "
            f"{len(unique_cases)} câu single-turn easy từ dữ liệu hiện có; "
            f"sẽ dùng toàn bộ thay vì yêu cầu {count}."
        )
        count = len(unique_cases)

    selected = unique_cases[:count]
    return _assign_ids(selected, "ST_EASY")


def build_conversation_easy_goldens(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    planning_docs = _collect_planning_docs(rows)
    posts = _collect_posts(rows)

    candidates: list[dict[str, Any]] = []

    for doc in planning_docs:
        doc_id = doc["planningDocumentId"]
        year = doc.get("planYear")
        dossier = doc.get("dossierCode")
        district = _friendly_district(doc)
        city = _friendly_city(doc)
        district_label = district or "khu vực này"
        city_label = city or "địa phương đó"
        title = doc.get("title") or "N/A"
        subject = _planning_subject(doc)

        if year is not None and dossier:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "scenario": "Người dùng hỏi tự nhiên về quy hoạch theo khu vực rồi hỏi tiếp mã hồ sơ.",
                    "expected_outcome": "Bot trả đúng planYear và dossierCode theo đúng tài liệu đã truy xuất.",
                    "turns": [
                        {
                            "role": "user",
                            "content": f"Cho mình hỏi thông tin quy hoạch sử dụng đất ở {district_label}, kế hoạch này áp dụng năm nào?",
                        },
                        {"role": "user", "content": "Cho mình xin luôn mã hồ sơ của kế hoạch đó nhé."},
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, planYear={year}, dossierCode={dossier}, district={doc.get('district') or district or 'N/A'}, city={doc.get('city') or city or 'N/A'}, title={title}"
                    ],
                }
            )

        if year is not None and title != "N/A":
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "scenario": "Người dùng nêu tên tài liệu trước, sau đó hỏi năm áp dụng.",
                    "expected_outcome": "Bot giữ đúng tài liệu theo title và trả lời đúng khu vực + planYear.",
                    "turns": [
                        {"role": "user", "content": f"Mình đang xem tài liệu \"{title}\", tài liệu này nói về khu vực nào vậy?"},
                        {"role": "user", "content": "Kế hoạch trong tài liệu đó áp dụng cho năm nào?"},
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, planYear={year}, district={doc.get('district') or district or 'N/A'}, city={doc.get('city') or city or 'N/A'}, title={title}"
                    ],
                }
            )

        if year is not None:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "scenario": "Người dùng hỏi theo địa bàn hành chính rồi follow-up về năm kế hoạch.",
                    "expected_outcome": "Bot trả đúng city/district và đúng planYear dựa trên cùng một nguồn.",
                    "turns": [
                        {
                            "role": "user",
                            "content": f"Thông tin quy hoạch của {district_label} thuộc thành phố nào vậy?",
                        },
                        {"role": "user", "content": "Vậy kế hoạch này áp dụng cho năm nào?"},
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, planYear={year}, district={doc.get('district') or district or 'N/A'}, city={doc.get('city') or city_label}, title={title}"
                    ],
                }
            )

        if doc.get("documentType"):
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "land_use_planning",
                    "scenario": "Người dùng hỏi loại văn bản rồi hỏi tiếp mã hồ sơ trong cùng ngữ cảnh.",
                    "expected_outcome": "Bot trả đúng documentType và dossierCode (nếu có), bám cùng tài liệu.",
                    "turns": [
                        {"role": "user", "content": f"{_sentence_case(subject)} thuộc loại văn bản gì vậy?"},
                        {
                            "role": "user",
                            "content": "Nếu có thì cho mình luôn mã hồ sơ của văn bản này.",
                        },
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, documentType={doc.get('documentType')}, dossierCode={dossier or 'N/A'}, district={doc.get('district') or district or 'N/A'}, city={doc.get('city') or city or 'N/A'}"
                    ],
                }
            )

    for post in posts:
        post_id = post["postId"]
        post_type = post.get("postType")
        price = post.get("price")
        area = post.get("area")
        bedrooms = post.get("bedrooms")
        category = post.get("categoryName") or "N/A"
        city = post.get("city") or "N/A"
        district = post.get("district") or "N/A"
        ward = post.get("ward") or "N/A"
        reference = _post_reference(post)

        if price is not None and post_type:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "scenario": "Người dùng hỏi giá trước rồi hỏi loại giao dịch của cùng một tin.",
                    "expected_outcome": "Bot trả đúng price và postType theo cùng bản ghi metadata.",
                    "turns": [
                        {"role": "user", "content": f"Tin {reference} đang có mức giá bao nhiêu vậy?"},
                        {"role": "user", "content": "Tin này là SALE hay RENT?"},
                    ],
                    "context": [
                        f"postId={post_id}, price={price}, postType={post_type}, area={area}, bedrooms={bedrooms}, categoryName={category}"
                    ],
                }
            )

        if area is not None and bedrooms is not None:
            candidates.append(
                {
                    "difficulty": "easy",
                    "question_type": "what",
                    "domain": "real_estate",
                    "scenario": "Người dùng hỏi diện tích rồi hỏi tiếp số phòng ngủ.",
                    "expected_outcome": "Bot trả đúng area và bedrooms của cùng tin, không lẫn dữ liệu.",
                    "turns": [
                        {"role": "user", "content": f"Tin {reference} có diện tích bao nhiêu m2?"},
                        {"role": "user", "content": "Còn số phòng ngủ thì sao?"},
                    ],
                    "context": [f"postId={post_id}, area={area}, bedrooms={bedrooms}, categoryName={category}"],
                }
            )

        candidates.append(
            {
                "difficulty": "easy",
                "question_type": "what",
                "domain": "real_estate",
                "scenario": "Người dùng hỏi danh mục trước rồi hỏi vị trí cụ thể của cùng tin.",
                "expected_outcome": "Bot trả đúng category và location theo city/district/ward metadata.",
                "turns": [
                    {"role": "user", "content": f"Tin {reference} thuộc danh mục gì?"},
                    {"role": "user", "content": "Tin này nằm ở khu vực nào?"},
                ],
                "context": [
                    f"postId={post_id}, categoryName={category}, city={city}, district={district}, ward={ward}, postType={post_type or 'N/A'}"
                ],
            }
        )

    unique_cases = _dedupe_conversation_by_turns(_filter_out_id_style_conversations(candidates))
    if not unique_cases:
        raise RuntimeError("Không thể tạo câu hỏi conversational từ dữ liệu embeddings.")

    if len(unique_cases) < count:
        raise RuntimeError(
            f"Chỉ tạo được {len(unique_cases)} câu conversational easy từ dữ liệu hiện có, ít hơn yêu cầu {count}."
        )

    selected = unique_cases[:count]
    return _assign_ids(selected, "MT_EASY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate easy goldens from current DB embeddings.")
    parser.add_argument("--single-easy-count", type=int, default=50, help="Number of easy single-turn goldens")
    parser.add_argument(
        "--conversation-easy-count",
        type=int,
        default=50,
        help="Number of easy conversational goldens",
    )
    parser.add_argument("--sample-limit", type=int, default=10000, help="Max embedding rows to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    single_path = repo_root / "evaluation" / "datasets" / "single_turn_goldens.json"
    conv_path = repo_root / "evaluation" / "datasets" / "conversational_goldens.json"

    rows = _load_metadata_rows(sample_limit=args.sample_limit)
    single_turn = build_single_turn_easy_goldens(rows, count=args.single_easy_count)
    conv = build_conversation_easy_goldens(rows, count=args.conversation_easy_count)

    single_path.write_text(json.dumps(single_turn, ensure_ascii=False, indent=2), encoding="utf-8")
    conv_path.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")

    planning_count = len(_collect_planning_docs(rows))
    post_count = len(_collect_posts(rows))
    print(f"Found {planning_count} planning docs and {post_count} posts from embeddings metadata")
    print(f"Generated {len(single_turn)} easy single-turn goldens -> {single_path}")
    print(f"Generated {len(conv)} easy conversational goldens -> {conv_path}")


if __name__ == "__main__":
    main()
