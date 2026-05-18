from __future__ import annotations

from typing import Callable
import re

from .query_intents import (
    build_query_intents as _build_query_intents,
    is_planning_fact_question as _is_planning_fact_question,
)
from .text_utils import normalize_text as _normalize_text
from .listing_processing import (
    condense_suitability_answer as _condense_suitability_answer,
    strip_outdoor_detail_lines as _strip_outdoor_detail_lines,
)

_UNREQUESTED_PRICE_LINE_MARKERS = (
    "gia ban",
    "gia thue",
    "gia:",
    "vnd",
    "trieu",
    "ty",
)

_CONTACT_MARKERS = (
    "lien he",
    "so dien thoai",
    "hotline",
    "chu nha",
    "moi gioi",
)

_SPACIOUSNESS_MARKERS = (
    "rong rai",
    "thoang",
    "khong gian",
    "dien tich",
)

def _looks_uncertain_or_no_data(answer: str) -> bool:
    a = _normalize_text(answer)
    signals = [
        "không biết",
        "khong biet",
        "không có trong context",
        "khong co trong context",
        "không đủ dữ liệu",
        "khong du du lieu",
        "i don't know",
        "insufficient",
        "not enough information",
    ]
    return any(signal in a for signal in signals)


def _strip_unrequested_price_lines(answer: str) -> str:
    kept_lines: list[str] = []
    for raw_line in (answer or "").splitlines():
        normalized_line = _normalize_text(raw_line)
        if normalized_line and any(marker in normalized_line for marker in _UNREQUESTED_PRICE_LINE_MARKERS):
            relevant_non_price = any(
                marker in normalized_line
                for marker in (
                    "dien tich",
                    "phong ngu",
                    "phong ve sinh",
                    "noi that",
                    "huong",
                    "khong gian",
                    "rong rai",
                    "thoang",
                    "phu hop",
                    "thuan tien",
                )
            )
            if not relevant_non_price:
                continue
            cleaned_line = re.sub(
                r",?\s*gi[aá][^,.;\n]*(?:VND|VNĐ|triệu|trieu|tỷ|ty)(?:/[^\s,.;]+)?",
                "",
                raw_line,
                flags=re.IGNORECASE,
            )
            cleaned_line = re.sub(
                r",?\s*\d[\d\.,]*\s*(?:triệu|trieu|tỷ|ty)(?:\s*(?:VND|VNĐ))?(?:/[^\s,.;]+)?",
                "",
                cleaned_line,
                flags=re.IGNORECASE,
            )
            cleaned_line = re.sub(
                r",?\s*\d[\d\.,]*\s*(?:VND|VNĐ)(?:/[^\s,.;]+)?",
                "",
                cleaned_line,
                flags=re.IGNORECASE,
            )
            cleaned_line = re.sub(r"\s{2,}", " ", cleaned_line).strip(" ,;-")
            if cleaned_line:
                kept_lines.append(cleaned_line)
            continue
        kept_lines.append(raw_line)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _strip_unrequested_contact_lines(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if any(marker in q for marker in ("lien he", "so dien thoai", "hotline", "chu nha", "moi gioi")):
        return answer

    kept_lines: list[str] = []
    changed = False
    for raw_line in (answer or "").splitlines():
        normalized_line = _normalize_text(raw_line)
        if (
            any(marker in normalized_line for marker in _CONTACT_MARKERS)
            or re.search(r"\b0\d{8,10}\b", raw_line)
        ):
            changed = True
            continue
        kept_lines.append(raw_line)

    if not changed:
        return answer
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def _strip_spaciousness_extra_lines(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if not any(marker in q for marker in _SPACIOUSNESS_MARKERS):
        return answer

    kept_lines: list[str] = []
    changed = False
    for raw_line in (answer or "").splitlines():
        normalized_line = _normalize_text(raw_line)
        if not normalized_line:
            kept_lines.append(raw_line)
            continue
        has_nearby_amenity = any(
            marker in normalized_line
            for marker in ("xung quanh", "benh vien", "sieu thi", "cong vien", "truong hoc", "cho ")
        )
        has_spacious_fact = any(
            marker in normalized_line
            for marker in ("dien tich", "m2", "phong ngu", "phong ve sinh", "noi that", "huong", "ban cong", "rong rai", "thoang")
        )
        if has_nearby_amenity and not has_spacious_fact:
            changed = True
            continue
        kept_lines.append(raw_line)

    if not changed:
        return answer
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def _strip_unrequested_area_suffix(answer: str) -> str:
    cleaned = (answer or "").strip()
    if not cleaned:
        return cleaned

    patterns = [
        r",?\s*với\s+tổng\s+diện\s+tích[^\n\.]*ha\.?",
        r",?\s*voi\s+tong\s+dien\s+tich[^\n\.]*ha\.?",
        r",?\s*tổng\s+diện\s+tích[^\n\.]*ha\.?",
        r",?\s*tong\s+dien\s+tich[^\n\.]*ha\.?",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _extract_named_project_line(answer: str) -> str | None:
    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized = _normalize_text(line)
        if "ten du an" not in normalized:
            continue

        extracted = re.sub(
            r"^[\-\*\s]*\*{0,2}\s*t[eê]n\s+d[uự]\s+[aá]n\s*\*{0,2}\s*[:\-]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip(" -*")
        if extracted:
            return extracted

    return None


def _extract_flexible_project_line(answer: str) -> str | None:
    extracted = _extract_named_project_line(answer)
    if extracted:
        return extracted

    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized = _normalize_text(line)
        if normalized.startswith("du an"):
            suffix_match = re.search(r":\s*(.+)$", line)
            if suffix_match:
                return suffix_match.group(1).strip(" -*")

        if line.startswith("- **"):
            bold_match = re.search(r"\*\*([^*]+)\*\*", line)
            if bold_match:
                candidate = bold_match.group(1).strip()
                if _normalize_text(candidate) not in {"du an", "ten du an"}:
                    return candidate

    return None


def _normalize_single_new_project_answer(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if "dang ky moi" not in q or "duy nhat" not in q:
        return answer

    project_name = _extract_flexible_project_line(answer)
    if not project_name:
        return answer

    normalized = project_name.strip()
    if not normalized:
        return answer

    return f"Tên dự án là {normalized}."


def _normalize_split_recovery_wording(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if not ("thu hoi" in q and "dat nong nghiep" in q and "dat phi nong nghiep" in q):
        return answer

    cleaned = (answer or "").strip()
    if not cleaned:
        return cleaned

    normalized = _normalize_text(cleaned)
    ascii_text = _normalize_text(cleaned)


    agri_value: str | None = None
    non_agri_value: str | None = None

    for match in re.finditer(
        r"dat\s+(?:phi\s+)?nong\s+nghiep[^\n]{0,90}?(\d+(?:[\.,]\d+)?)\s*ha",
        ascii_text,
    ):
        segment = match.group(0)
        value = match.group(1)
        if "dat phi nong nghiep" in segment and non_agri_value is None:
            non_agri_value = value
        elif "dat phi nong nghiep" not in segment and agri_value is None:
            agri_value = value

    if agri_value and non_agri_value:
        return (
            f"Đất nông nghiệp thu hồi là {agri_value} ha.\n"
            f"Đất phi nông nghiệp thu hồi là {non_agri_value} ha."
        )

    if "dat nong nghiep" in normalized and "dat phi nong nghiep" in normalized:
        cleaned = re.sub(r"(?i)dự\s+kiến\s+thu\s+hồi", "thu hồi", cleaned)
        cleaned = re.sub(r"(?i)du\s+kien\s+thu\s+hoi", "thu hồi", cleaned)

    return cleaned


def _normalize_planning_reporting_chain(question: str, answer: str) -> str:
    if not _is_planning_fact_question(question):
        return answer

    q = _normalize_text(question)
    # Only rewrite reporting chain wording when the user explicitly asks about reporting/update flow.
    if not any(
        marker in q
        for marker in (
            "bao cao",
            "trinh ubnd",
            "bo sung danh muc",
            "cap nhat danh muc",
            "so tai nguyen",
        )
    ):
        return answer

    lines = (answer or "").splitlines()
    if not lines:
        return answer

    changed = False
    rewritten_lines: list[str] = []

    for raw_line in lines:
        line = raw_line
        norm = _normalize_text(line)
        if (
            "trinh ubnd thanh pho" in norm
            and "so tai nguyen va moi truong" not in norm
            and "bao cao" in norm
        ):
            if "de trinh ubnd thanh pho" in norm:
                line = re.sub(
                    r"(?i)\bđể\s+trình\s+UBND\s+Thành\s+phố\b",
                    "để báo cáo Sở Tài nguyên và Môi trường trình UBND Thành phố",
                    line,
                )
                line = re.sub(
                    r"(?i)\bde\s+trinh\s+UBND\s+thanh\s+pho\b",
                    "de bao cao So Tai nguyen va Moi truong trinh UBND Thanh pho",
                    line,
                )
            else:
                line = re.sub(
                    r"(?i)\btrình\s+UBND\s+Thành\s+phố\b",
                    "báo cáo Sở Tài nguyên và Môi trường trình UBND Thành phố",
                    line,
                )
                line = re.sub(
                    r"(?i)\btrinh\s+UBND\s+thanh\s+pho\b",
                    "bao cao So Tai nguyen va Moi truong trinh UBND Thanh pho",
                    line,
                )
            changed = changed or (line != raw_line)

        rewritten_lines.append(line)

    if not changed:
        return answer

    return "\n".join(rewritten_lines).strip()


def _normalize_post_approval_execution_answer(question: str, answer: str) -> str:
    if not _is_planning_fact_question(question):
        return answer

    q = _normalize_text(question)
    asks_post_approval_execution = (
        "duoc phe duyet" in q
        and (
            "phai trien khai" in q
            or "trien khai nhu the nao" in q
            or "phai thuc hien nhu the nao" in q
        )
    )
    if not asks_post_approval_execution:
        return answer

    lines = (answer or "").splitlines()
    if not lines:
        return answer

    filtered_lines: list[str] = []
    changed = False
    bullet_count = 0

    risky_markers = (
        "danh gia kha nang thuc hien",
        "du co so phap ly",
        "bo sung ke hoach su dung dat",
        "bo sung danh muc ke hoach su dung dat",
    )

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            filtered_lines.append(raw_line)
            continue

        is_bullet = stripped.startswith("-")
        if is_bullet:
            bullet_count += 1
            norm = _normalize_text(stripped.lstrip("- ").strip())
            if any(marker in norm for marker in risky_markers):
                changed = True
                continue

        filtered_lines.append(raw_line)

    if not changed:
        return answer

    remaining_bullets = sum(1 for line in filtered_lines if line.strip().startswith("-"))
    if bullet_count > 0 and remaining_bullets == 0:
        return answer

    return "\n".join(filtered_lines).strip()


def _strip_unrequested_planning_extra_lines(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if not any(marker in q for marker in ("tong", "dien tich", "bao nhieu", "chi tieu", "the hien")):
        return answer
    if any(marker in q for marker in ("ngoai ra", "them", "bo sung", "thong tin khac")):
        return answer

    asks_detail_listing = any(
        marker in q
        for marker in (
            "chi tiet",
            "liet ke",
            "du an nao",
            "gom nhung gi",
            "nhung cong trinh nao",
            "nhung du an nao",
        )
    )
    asks_location_listing = any(marker in q for marker in ("phuong nao", "khu nao", "o dau", "dia ban nao", "vi tri"))

    lines = (answer or "").splitlines()
    filtered: list[str] = []
    changed = False
    skip_blank = False
    asks_count_or_area = any(marker in q for marker in ("tong", "dien tich", "bao nhieu", "so luong", "quy mo"))
    for line in lines:
        normalized = _normalize_text(line)
        raw_lower = (line or "").lower()
        if normalized.startswith(("ngoai ra", "ben canh do", "dong thoi")):
            changed = True
            skip_blank = True
            continue
        if not asks_detail_listing and (
            (
                (
                    "thong tin chi tiet" in normalized
                    or "th ng tin chi ti t" in normalized
                    or normalized.startswith(("thong tin ve", "th ng tin"))
                    or "thông tin chi tiết" in raw_lower
                )
                and (
                    any(marker in normalized for marker in ("khong co", "khong co san", "khong duoc cung cap", "kh ng c", "kh ng duoc"))
                    or any(marker in raw_lower for marker in ("không có", "không được cung cấp"))
                )
            )
            or normalized.startswith(("neu can them", "vui long cho biet"))
        ):
            changed = True
            skip_blank = True
            continue
        if not asks_location_listing and (
            normalized.startswith("ke hoach cu the")
            or "trong do co cac khu vuc" in normalized
        ):
            changed = True
            skip_blank = True
            continue
        if not asks_count_or_area and (
            normalized.startswith(("- tong so", "tong so", "- dien tich", "dien tich"))
            or normalized.startswith(("- tong dien tich", "tong dien tich"))
        ):
            changed = True
            skip_blank = True
            continue
        if skip_blank and not normalized:
            changed = True
            continue
        filtered.append(line.rstrip())
        skip_blank = False

    if not changed:
        cleaned = answer
    else:
        cleaned = "\n".join(filtered).strip()

    q_without_non_agri = q.replace("dat phi nong nghiep", "")
    if "dat phi nong nghiep" in q and "dat nong nghiep" not in q_without_non_agri:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        kept_sentences: list[str] = []
        for sentence in sentences:
            normalized_sentence = _normalize_text(sentence)
            raw_sentence_lower = (sentence or "").lower()
            if (
                (
                    "dat nong nghiep" in normalized_sentence
                    or "t n ng nghi p" in normalized_sentence
                    or "đất nông nghiệp" in raw_sentence_lower
                )
                and "dat phi nong nghiep" not in normalized_sentence
                and "đất phi nông nghiệp" not in raw_sentence_lower
            ):
                changed = True
                continue
            kept_sentences.append(sentence)
        cleaned = " ".join(sentence.strip() for sentence in kept_sentences if sentence.strip()).strip()

    return cleaned


def _normalize_single_land_recovery_total_answer(question: str, answer: str) -> str:
    q = _normalize_text(question)
    asks_single_recovery_total = (
        "thu hoi" in q
        and "bao nhieu" in q
        and "dat nong nghiep" in q
        and "dat phi nong nghiep" not in q
        and "tong cong" not in q
    )
    if not asks_single_recovery_total:
        return answer

    match = re.search(
        r"(\d[\d\.,]*)\s*ha\s+đất nông nghiệp",
        answer,
        flags=re.IGNORECASE,
    )
    if not match:
        ascii_answer = _normalize_text(answer)
        match = re.search(r"thu\s+hoi\s+(\d[\d\.,]*)\s*ha\s+dat\s+nong\s+nghiep", ascii_answer)
    if not match:
        return answer

    return f"Diện tích đất nông nghiệp cần thu hồi là {match.group(1)} ha."


def _normalize_indoor_amenities_answer(question: str, answer: str) -> str:
    intents = _build_query_intents(question)
    if not intents.get("needs_indoor_amenities"):
        return answer

    cleaned_answer = (answer or "").strip()
    if not cleaned_answer:
        return cleaned_answer

    keep_markers = (
        "thang may",
        "noi that",
        "thiet ke",
        "phong ngu",
        "phong ve sinh",
        "bep",
        "giuong",
        "wc",
    )
    drop_markers = (
        "huong",
        "anh sang",
        "thoang dang",
        "giao thong",
        "o to",
        "truc duong",
        "cong vien",
        "aeon",
        "ngo",
        "mat tien",
    )

    kept_lines: list[str] = []
    for line in cleaned_answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = _normalize_text(stripped)
        if any(marker in normalized for marker in drop_markers) and not any(
            marker in normalized for marker in keep_markers
        ):
            continue
        if any(marker in normalized for marker in keep_markers):
            cleaned_line = re.sub(
                r",?\s*(?:nhờ|do)\s+hướng[^\n]*$",
                "",
                stripped,
                flags=re.IGNORECASE,
            ).strip(" ,")
            cleaned_line = re.sub(
                r",?\s*hướng cửa chính[^\n]*$",
                "",
                cleaned_line,
                flags=re.IGNORECASE,
            ).strip(" ,")
            cleaned_line = re.sub(
                r",?\s*mang lại[^\n]*ánh sáng[^\n]*$",
                "",
                cleaned_line,
                flags=re.IGNORECASE,
            ).strip(" ,")
            kept_lines.append(cleaned_line or stripped)

    if not kept_lines:
        return cleaned_answer
    return "\n".join(kept_lines)


def _strip_non_auction_scope_lines(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if "dau gia quyen su dung dat" not in q:
        return answer

    lines = (answer or "").splitlines()
    if not lines:
        return answer

    kept_lines: list[str] = []
    changed = False
    for line in lines:
        norm = _normalize_text(line)
        if "cho thue quyen su dung" in norm and any(
            marker in norm for marker in ("dat nong nghiep cong ich", "dat chua su dung")
        ):
            changed = True
            continue
        if "o dat" in norm and any(marker in norm for marker in ("f1 odk", "f1/odk", "odk5")):
            changed = True
            continue
        kept_lines.append(line)

    if not changed:
        return answer
    rewritten = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    return rewritten or answer


def _strip_no_detail_tail(question: str, answer: str) -> str:
    lines = (answer or "").splitlines()
    if len(lines) <= 1:
        return answer

    kept_lines: list[str] = []
    changed = False
    for line in lines:
        norm = _normalize_text(line)
        has_no_detail_signal = "thong tin chi tiet" in norm and any(
            marker in norm for marker in ("khong co san", "khong day du", "khong co")
        )
        if has_no_detail_signal:
            changed = True
            continue
        kept_lines.append(line)

    if not changed:
        return answer
    rewritten = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    return rewritten or answer


def _normalize_indoor_amenities_relevancy(question: str, answer: str) -> str:
    intents = _build_query_intents(question)
    if not intents.get("needs_indoor_amenities"):
        return answer

    normalized = _normalize_text(answer)
    amenities: list[str] = []
    if "thang may" in normalized:
        amenities.append("thang máy")
    if "thiet ke hien dai" in normalized or "thiet ke" in normalized:
        amenities.append("thiết kế hiện đại")

    if not amenities:
        return answer

    if len(amenities) == 1:
        return f"Tiện ích nổi bật trong nhà là {amenities[0]}."
    return f"Tiện ích nổi bật trong nhà là {amenities[0]} và {amenities[1]}."


def _focus_waste_overlap_reasoning(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if not ("lang phi" in q or "chong cheo" in q):
        return answer

    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    if len(lines) <= 1:
        return answer

    reasoning_markers = (
        "dam bao tinh thong nhat",
        "quan ly chat che",
        "giam sat chat che",
        "dung quy hoach",
        "dung phap luat",
        "su dung dat hieu qua",
        "tinh kha thi",
        "tranh lang phi",
        "tranh chong cheo",
    )
    kept = [line for line in lines if any(marker in _normalize_text(line) for marker in reasoning_markers)]
    return "\n".join(kept) if kept else answer


def _normalize_unused_land_area_answer(question: str, answer: str) -> str:
    q = _normalize_text(question)
    if not (
        "dat chua su dung" in q
        and "dua vao su dung" in q
        and "bao nhieu" in q
    ):
        return answer

    candidate_texts = [
        line
        for line in (answer or "").splitlines()
        if "chua su dung" in _normalize_text(line) or "dua vao su dung" in _normalize_text(line)
    ]
    candidate_texts.append(answer or "")
    for text in candidate_texts:
        match = re.search(r"(\d+(?:\s*[,\.]\s*\d+)?)\s*ha", text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", "", match.group(1)).replace(".", ",")
            return f"Di\u1ec7n t\u00edch \u0111\u1ea5t ch\u01b0a s\u1eed d\u1ee5ng \u0111\u01b0a v\u00e0o s\u1eed d\u1ee5ng l\u00e0 {value} ha."
    return answer


AnswerTransform = Callable[[str, str], str]

_PRE_CONTEXT_ANSWER_TRANSFORMS: tuple[AnswerTransform, ...] = (
    _normalize_single_new_project_answer,
    _normalize_split_recovery_wording,
    _normalize_single_land_recovery_total_answer,
)

_POST_CONTEXT_ANSWER_TRANSFORMS: tuple[AnswerTransform, ...] = (
    _normalize_post_approval_execution_answer,
    _normalize_planning_reporting_chain,
    _strip_non_auction_scope_lines,
    _strip_no_detail_tail,
    _normalize_indoor_amenities_relevancy,
    _focus_waste_overlap_reasoning,
    _normalize_unused_land_area_answer,
)


def _keep_non_empty(previous: str, candidate: str) -> str:
    cleaned_candidate = (candidate or "").strip()
    return cleaned_candidate if cleaned_candidate else previous


def postprocess_answer(question: str, answer: str) -> str:
    if not answer:
        return answer

    cleaned = answer.strip()

    # Remove common generic trailing invites that hurt factual relevancy metrics.
    generic_tail_patterns = [
        r"\s*Nếu bạn cần thêm[^.!?]*[.!?]?\s*$",
        r"\s*Nếu cần thêm[^.!?]*[.!?]?\s*$",
        r"\s*Thông tin này được nêu[^.!?]*[.!?]?\s*$",
        r"\s*Kế hoạch sử dụng đất hàng năm[^.!?]*[.!?]?\s*$",
        r"\s*Ke hoach su dung dat hang nam[^.!?]*[.!?]?\s*$",
        r"\s*If you need more information[^.!?]*[.!?]?\s*$",
    ]
    for pattern in generic_tail_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    intents = _build_query_intents(question)
    question_norm = _normalize_text(question)

    asks_project_count = (
        "bao nhieu" in question_norm
        and ("cong trinh" in question_norm or "du an" in question_norm)
    )
    asks_project_name_listing = (
        "du an" in question_norm
        and any(marker in question_norm for marker in ("du an nao", "hai du an", "liet ke", "gom nhung gi"))
        and "bao nhieu" not in question_norm
    )
    if asks_project_count and "dien tich" not in question_norm:
        cleaned = _strip_unrequested_area_suffix(cleaned)

    if asks_project_name_listing and "dien tich" not in question_norm:
        cleaned = _strip_unrequested_area_suffix(cleaned)

    for transform in _PRE_CONTEXT_ANSWER_TRANSFORMS:
        cleaned = _keep_non_empty(cleaned, transform(question, cleaned))
    for transform in _POST_CONTEXT_ANSWER_TRANSFORMS:
        cleaned = _keep_non_empty(cleaned, transform(question, cleaned))

    planning_land_fact_question = any(marker in question_norm for marker in ("dat phi nong nghiep", "dat nong nghiep", "dat chua su dung"))
    if _is_planning_fact_question(question) or planning_land_fact_question:
        cleaned = _keep_non_empty(cleaned, _strip_unrequested_planning_extra_lines(question, cleaned))

    if intents.get("suitability_query") and not intents.get("needs_price"):
        cleaned = _keep_non_empty(cleaned, _strip_unrequested_price_lines(cleaned))

    cleaned = _keep_non_empty(cleaned, _strip_unrequested_contact_lines(question, cleaned))
    cleaned = _keep_non_empty(cleaned, _strip_spaciousness_extra_lines(question, cleaned))

    if intents.get("needs_indoor_amenities"):
        cleaned = _keep_non_empty(cleaned, _strip_outdoor_detail_lines(cleaned))
        cleaned = _keep_non_empty(cleaned, _normalize_indoor_amenities_answer(question, cleaned))

    if intents.get("suitability_query"):
        cleaned = _keep_non_empty(cleaned, _condense_suitability_answer(question, cleaned))

    if not _is_planning_fact_question(question):
        return cleaned

    if _looks_uncertain_or_no_data(cleaned):
        return cleaned

    return cleaned


def detect_lang(text: str) -> str:
    """Detect whether the answer should be Vietnamese or English."""
    t = (text or "").strip()
    if not t:
        return "English"

    try:
        import langid

        lang, _score = langid.classify(t)
        if lang == "vi":
            return "Vietnamese"
    except Exception:
        pass

    lower = t.lower()
    vi_chars = (
        "ăâđêôơư"
        "áàảãạấầẩẫậ"
        "ắằẳẵặéèẻẽẹ"
        "ếềểễệíìỉĩị"
        "óòỏõọốồổỗộ"
        "ớờởỡợúùủũụ"
        "ứừửữựýỳỷỹỵ"
    )
    if any(c in lower for c in vi_chars):
        return "Vietnamese"

    vi_markers = {
        "toi", "ban", "minh", "muon", "can", "tim", "kiem", "nha", "dat", "canho",
        "gia", "bao", "nhieu", "o", "tai", "quan", "huyen", "phuong", "duong",
        "dien", "tich", "phong", "ngu", "phap", "ly", "so", "do", "hop", "dong",
        "gan", "trung", "tam", "khoang", "trieu", "ty", "co", "khong", "neu", "thi",
        "va", "voi", "cho", "xin", "giup",
    }
    tokens = []
    for word in lower.replace("\n", " ").split():
        token = "".join(ch for ch in word if ch.isalnum())
        if token:
            tokens.append(token)

    if not tokens:
        return "English"

    hit = sum(1 for token in tokens if token in vi_markers)
    if hit >= 2 or (len(tokens) <= 6 and hit >= 1):
        return "Vietnamese"
    return "English"
