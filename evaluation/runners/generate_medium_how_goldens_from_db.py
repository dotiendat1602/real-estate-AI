from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.runners.generate_goldens_from_db import (
    _assign_ids,
    _collect_planning_docs,
    _collect_posts,
    _dedupe_by_input,
    _dedupe_conversation_by_turns,
    _filter_out_id_style_conversations,
    _filter_out_id_style_single,
    _format_location,
    _friendly_city,
    _friendly_district,
    _load_metadata_rows,
    _planning_subject,
    _post_reference,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _unit_price_text(price: Any, area: Any) -> str | None:
    price_val = _safe_float(price)
    area_val = _safe_float(area)
    if price_val is None or area_val is None or area_val <= 0:
        return None
    unit = price_val / area_val
    return f"{unit:.2f}"


def build_single_turn_medium_how_goldens(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
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
        subject = _planning_subject(doc)

        context_city = doc.get("city") or city or "N/A"
        context_district = doc.get("district") or district or "N/A"
        location = _format_location(district, city) or "khu vực này"

        if year is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "land_use_planning",
                    "input": f"Làm thế nào để xác định {subject} đang áp dụng cho năm nào từ dữ liệu hiện có?",
                    "expected_output_outline": [
                        "Nêu cách kiểm tra trường planYear trong metadata/tài liệu đã truy xuất",
                        f"Kết luận năm áp dụng: {year}",
                        "Không suy đoán nếu thiếu dữ liệu",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, planYear={year}, district={context_district}, city={context_city}, title={title or 'N/A'}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

        if year is not None and dossier:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "land_use_planning",
                    "input": f"Nếu muốn tra mã hồ sơ của {subject} một cách chắc chắn thì nên kiểm tra như thế nào?",
                    "expected_output_outline": [
                        "Nêu cách đối chiếu district/city/planYear trước khi đọc dossierCode",
                        f"Mã hồ sơ đúng: {dossier}",
                        "Trả lời ngắn gọn, bám sát metadata",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, dossierCode={dossier}, planYear={year}, district={context_district}, city={context_city}, title={title or 'N/A'}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

        if district and city:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "land_use_planning",
                    "input": f"Làm sao để kiểm chứng tài liệu quy hoạch này thuộc đúng địa bàn nào trong {city}?",
                    "expected_output_outline": [
                        "Nêu cách đọc district/city trong metadata để xác thực địa bàn",
                        f"Khu vực đúng: {district}, {city}",
                        "Không thêm địa danh ngoài dữ liệu",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, district={context_district}, city={context_city}, title={title or 'N/A'}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

        if title and year is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "land_use_planning",
                    "input": f"Khi chỉ có tên văn bản \"{title}\", làm thế nào để kiểm tra năm áp dụng và phạm vi khu vực?",
                    "expected_output_outline": [
                        "Nêu cách dùng title + planYear + district/city để xác minh",
                        f"Kết quả đúng: năm {year}, khu vực {location}",
                        "Không suy luận vượt ngoài metadata",
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, title={title}, planYear={year}, district={context_district}, city={context_city}"
                    ],
                    "target_metadata": {"planningDocumentId": doc_id},
                }
            )

    for post in posts:
        post_id = post["postId"]
        post_type = post.get("postType")
        category = post.get("categoryName") or "bất động sản"
        city = post.get("city")
        district = post.get("district")
        ward = post.get("ward")
        price = post.get("price")
        area = post.get("area")
        bedrooms = post.get("bedrooms")
        reference = _post_reference(post)
        location = _format_location(ward, district, city) or "khu vực này"

        unit_text = _unit_price_text(price, area)
        if price is not None and area is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "real_estate",
                    "input": f"Làm thế nào để ước tính nhanh giá trên mỗi m2 cho tin {reference}?",
                    "expected_output_outline": [
                        "Nêu cách lấy price chia cho area để ước tính đơn giá",
                        f"Dữ liệu đầu vào: price={price}, area={area}",
                        f"Đơn giá tham chiếu (price/area): {unit_text if unit_text is not None else 'không tính được'}",
                    ],
                    "context": [
                        f"postId={post_id}, price={price}, area={area}, city={city or 'N/A'}, district={district or 'N/A'}, ward={ward or 'N/A'}"
                    ],
                    "target_metadata": {"postId": post_id},
                }
            )

        if post_type and category:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "real_estate",
                    "input": f"Làm sao để xác định tin {reference} phù hợp nhu cầu mua hay thuê?",
                    "expected_output_outline": [
                        "Nêu cách đọc postType để phân loại SALE/RENT",
                        f"Kết luận loại tin: {post_type}",
                        f"Danh mục tham chiếu: {category}",
                    ],
                    "context": [
                        f"postId={post_id}, postType={post_type}, categoryName={category}, city={city or 'N/A'}, district={district or 'N/A'}"
                    ],
                    "target_metadata": {"postId": post_id},
                }
            )

        if bedrooms is not None and area is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "real_estate",
                    "input": f"Làm thế nào để kiểm tra nhanh mức độ phù hợp công năng của tin {reference} cho hộ gia đình nhỏ?",
                    "expected_output_outline": [
                        "Nêu cách kết hợp bedrooms + area để đánh giá sơ bộ công năng",
                        f"Dữ liệu chính: bedrooms={bedrooms}, area={area}",
                        "Nếu thiếu dữ liệu khác thì phải nói rõ giới hạn",
                    ],
                    "context": [f"postId={post_id}, bedrooms={bedrooms}, area={area}, categoryName={category}"],
                    "target_metadata": {"postId": post_id},
                }
            )

        if ward or district or city:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "real_estate",
                    "input": f"Nếu muốn kiểm tra vị trí chi tiết của tin {category} này thì nên đọc theo thứ tự như thế nào?",
                    "expected_output_outline": [
                        "Nêu cách đọc theo cấp ward -> district -> city",
                        f"Vị trí theo metadata: {location}",
                        "Không thêm địa chỉ ngoài dữ liệu có sẵn",
                    ],
                    "context": [
                        f"postId={post_id}, ward={ward or 'N/A'}, district={district or 'N/A'}, city={city or 'N/A'}, categoryName={category}"
                    ],
                    "target_metadata": {"postId": post_id},
                }
            )

    unique_cases = _dedupe_by_input(_filter_out_id_style_single(candidates))
    if not unique_cases:
        raise RuntimeError("Không thể tạo câu hỏi single-turn medium/how từ dữ liệu embeddings.")

    if len(unique_cases) < count:
        raise RuntimeError(
            f"Chỉ tạo được {len(unique_cases)} câu single-turn medium/how từ dữ liệu hiện có, ít hơn yêu cầu {count}."
        )

    return _assign_ids(unique_cases[:count], "ST_MEDIUM_HOW")


def build_conversation_medium_how_goldens(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    planning_docs = _collect_planning_docs(rows)
    posts = _collect_posts(rows)

    candidates: list[dict[str, Any]] = []

    for doc in planning_docs:
        doc_id = doc["planningDocumentId"]
        year = doc.get("planYear")
        dossier = doc.get("dossierCode")
        city = _friendly_city(doc)
        district = _friendly_district(doc)
        title = doc.get("title") or "tài liệu quy hoạch"

        context_city = doc.get("city") or city or "N/A"
        context_district = doc.get("district") or district or "N/A"
        district_label = district or "khu vực này"

        if year is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "land_use_planning",
                    "scenario": "Người dùng hỏi cách kiểm tra năm áp dụng, sau đó yêu cầu áp dụng ngay vào tài liệu cụ thể.",
                    "expected_outcome": "Bot nêu được cách kiểm tra planYear và trả đúng năm áp dụng theo cùng tài liệu.",
                    "turns": [
                        {
                            "role": "user",
                            "content": f"Mình muốn tự kiểm tra năm áp dụng của tài liệu quy hoạch ở {district_label}. Làm thế nào cho đúng?",
                        },
                        {
                            "role": "user",
                            "content": f"Bạn áp dụng luôn cách đó cho tài liệu \"{title}\" và cho mình năm cụ thể nhé.",
                        },
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, title={title}, planYear={year}, district={context_district}, city={context_city}, dossierCode={dossier or 'N/A'}"
                    ],
                }
            )

        if dossier and year is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "land_use_planning",
                    "scenario": "Người dùng hỏi quy trình tra mã hồ sơ rồi yêu cầu kết quả cụ thể cho cùng một kế hoạch.",
                    "expected_outcome": "Bot mô tả cách đối chiếu metadata và trả đúng dossierCode.",
                    "turns": [
                        {
                            "role": "user",
                            "content": f"Nếu mình muốn tra mã hồ sơ của kế hoạch sử dụng đất ở {district_label} thì nên làm thế nào để tránh nhầm?",
                        },
                        {
                            "role": "user",
                            "content": f"Ok, vậy áp dụng luôn cho tài liệu \"{title}\" và cho mình mã hồ sơ cụ thể.",
                        },
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, title={title}, dossierCode={dossier}, planYear={year}, district={context_district}, city={context_city}"
                    ],
                }
            )

        if district and city:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "land_use_planning",
                    "scenario": "Người dùng hỏi cách xác thực địa bàn, rồi yêu cầu trả kết quả địa bàn cụ thể.",
                    "expected_outcome": "Bot nêu cách kiểm tra district/city và trả đúng địa bàn tài liệu.",
                    "turns": [
                        {
                            "role": "user",
                            "content": "Làm thế nào để kiểm tra một tài liệu quy hoạch thuộc đúng quận/huyện nào trong Hà Nội khi tên khá giống nhau?",
                        },
                        {
                            "role": "user",
                            "content": f"Bạn áp dụng luôn cho \"{title}\" và trả giúp mình đúng địa bàn cụ thể nhé.",
                        },
                    ],
                    "context": [
                        f"planningDocumentId={doc_id}, title={title}, district={context_district}, city={context_city}, planYear={year if year is not None else 'N/A'}"
                    ],
                }
            )

    for post in posts:
        post_id = post["postId"]
        post_type = post.get("postType")
        category = post.get("categoryName") or "bất động sản"
        city = post.get("city")
        district = post.get("district")
        ward = post.get("ward")
        price = post.get("price")
        area = post.get("area")
        bedrooms = post.get("bedrooms")
        reference = _post_reference(post)
        location = _format_location(ward, district, city) or "khu vực này"
        unit_text = _unit_price_text(price, area)

        if price is not None and area is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "real_estate",
                    "scenario": "Người dùng hỏi cách ước tính đơn giá theo m2, sau đó yêu cầu áp dụng ngay cho tin cụ thể.",
                    "expected_outcome": "Bot nêu đúng cách tính từ price/area và đưa được kết quả tham chiếu cho cùng tin.",
                    "turns": [
                        {
                            "role": "user",
                            "content": "Làm thế nào để ước tính nhanh giá trên m2 của một tin bất động sản chỉ từ dữ liệu cơ bản?",
                        },
                        {
                            "role": "user",
                            "content": f"Bạn áp dụng luôn cho tin {reference} và cho mình kết quả tham chiếu nhé.",
                        },
                    ],
                    "context": [
                        f"postId={post_id}, price={price}, area={area}, estimatedUnitPrice={unit_text if unit_text is not None else 'N/A'}, city={city or 'N/A'}, district={district or 'N/A'}, ward={ward or 'N/A'}"
                    ],
                }
            )

        if post_type and (district or city):
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "real_estate",
                    "scenario": "Người dùng hỏi cách kiểm tra loại giao dịch và vị trí, rồi yêu cầu tóm tắt kết quả cho một tin.",
                    "expected_outcome": "Bot trình bày cách đọc postType/location và trả đúng thông tin theo metadata.",
                    "turns": [
                        {
                            "role": "user",
                            "content": "Làm sao để kiểm tra nhanh một tin là bán hay thuê và nằm ở khu vực nào?",
                        },
                        {
                            "role": "user",
                            "content": f"Bạn áp dụng cho tin {reference} và tóm tắt giúp mình kết quả nhé.",
                        },
                    ],
                    "context": [
                        f"postId={post_id}, postType={post_type}, categoryName={category}, ward={ward or 'N/A'}, district={district or 'N/A'}, city={city or 'N/A'}, location={location}"
                    ],
                }
            )

        if bedrooms is not None and area is not None:
            candidates.append(
                {
                    "difficulty": "medium",
                    "question_type": "how",
                    "domain": "real_estate",
                    "scenario": "Người dùng hỏi cách đánh giá công năng, sau đó yêu cầu áp dụng cho một tin cụ thể.",
                    "expected_outcome": "Bot nêu cách kết hợp bedrooms/area và trả thông tin đúng của tin đang xét.",
                    "turns": [
                        {
                            "role": "user",
                            "content": "Làm thế nào để đánh giá nhanh một tin có phù hợp hộ gia đình nhỏ dựa trên dữ liệu cơ bản?",
                        },
                        {
                            "role": "user",
                            "content": f"Áp dụng thử cho tin {reference} và cho mình nhận định sơ bộ dựa trên dữ liệu nhé.",
                        },
                    ],
                    "context": [
                        f"postId={post_id}, bedrooms={bedrooms}, area={area}, categoryName={category}, ward={ward or 'N/A'}, district={district or 'N/A'}, city={city or 'N/A'}"
                    ],
                }
            )

    unique_cases = _dedupe_conversation_by_turns(_filter_out_id_style_conversations(candidates))
    if not unique_cases:
        raise RuntimeError("Không thể tạo câu hỏi conversational medium/how từ dữ liệu embeddings.")

    if len(unique_cases) < count:
        raise RuntimeError(
            f"Chỉ tạo được {len(unique_cases)} câu conversational medium/how từ dữ liệu hiện có, ít hơn yêu cầu {count}."
        )

    return _assign_ids(unique_cases[:count], "MT_MEDIUM_HOW")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate medium/how goldens from current DB embeddings.")
    parser.add_argument("--single-count", type=int, default=50, help="Number of medium single-turn how goldens")
    parser.add_argument(
        "--conversation-count",
        type=int,
        default=50,
        help="Number of medium conversational how goldens",
    )
    parser.add_argument("--sample-limit", type=int, default=10000, help="Max embedding rows to inspect")
    parser.add_argument(
        "--single-output",
        default="single_turn_goldens_medium_how.json",
        help="Output filename under evaluation/datasets for single-turn set",
    )
    parser.add_argument(
        "--conversation-output",
        default="conversational_goldens_medium_how.json",
        help="Output filename under evaluation/datasets for conversational set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    single_path = repo_root / "evaluation" / "datasets" / args.single_output
    conv_path = repo_root / "evaluation" / "datasets" / args.conversation_output

    rows = _load_metadata_rows(sample_limit=args.sample_limit)
    single_turn = build_single_turn_medium_how_goldens(rows, count=args.single_count)
    conv = build_conversation_medium_how_goldens(rows, count=args.conversation_count)

    single_path.write_text(json.dumps(single_turn, ensure_ascii=False, indent=2), encoding="utf-8")
    conv_path.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")

    planning_count = len(_collect_planning_docs(rows))
    post_count = len(_collect_posts(rows))
    print(f"Found {planning_count} planning docs and {post_count} posts from embeddings metadata")
    print(f"Generated {len(single_turn)} medium/how single-turn goldens -> {single_path}")
    print(f"Generated {len(conv)} medium/how conversational goldens -> {conv_path}")


if __name__ == "__main__":
    main()
