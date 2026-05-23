from __future__ import annotations

import re

from .query_intents import build_query_intents as _build_query_intents
from .text_utils import normalize_text as _normalize_text

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
        if any(marker in normalized_line for marker in _CONTACT_MARKERS) or re.search(r"\b0\d{8,10}\b", raw_line):
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
            for marker in (
                "dien tich",
                "m2",
                "phong ngu",
                "phong ve sinh",
                "noi that",
                "huong",
                "ban cong",
                "rong rai",
                "thoang",
            )
        )
        if has_nearby_amenity and not has_spacious_fact:
            changed = True
            continue
        kept_lines.append(raw_line)

    if not changed:
        return answer
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def _keep_non_empty(previous: str, candidate: str) -> str:
    cleaned_candidate = (candidate or "").strip()
    return cleaned_candidate if cleaned_candidate else previous


def postprocess_answer(question: str, answer: str) -> str:
    if not answer:
        return answer

    cleaned = answer.strip()
    cleaned = re.sub(
        r"(?im)^\s*=+\s*(?:BẤT\s+ĐỘNG\s+SẢN|BAT\s+DONG\s+SAN)\s+(\d+)\s*=+\s*[-:]*\s*",
        r"[Xem chi tiết](/posts/\1) ",
        cleaned,
    )
    cleaned = re.sub(r"(?im)^\s*LISTING_ID\s*:\s*\d+\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*---\s*[^-\n]+\s*---\s*$", "", cleaned)

    generic_tail_patterns = [
        r"\s*Nếu bạn cần thêm[^.!?]*[.!?]?\s*$",
        r"\s*Nếu cần thêm[^.!?]*[.!?]?\s*$",
        r"\s*Thông tin này được nêu[^.!?]*[.!?]?\s*$",
        r"\s*Kế hoạch sử dụng đất hằng năm[^.!?]*[.!?]?\s*$",
        r"\s*Ke hoach su dung dat hang nam[^.!?]*[.!?]?\s*$",
        r"\s*If you need more information[^.!?]*[.!?]?\s*$",
    ]
    for pattern in generic_tail_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    intents = _build_query_intents(question)
    if intents.get("suitability_query") and not intents.get("needs_price"):
        cleaned = _keep_non_empty(cleaned, _strip_unrequested_price_lines(cleaned))

    cleaned = _keep_non_empty(cleaned, _strip_unrequested_contact_lines(question, cleaned))
    cleaned = _keep_non_empty(cleaned, _strip_spaciousness_extra_lines(question, cleaned))

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
        "toi",
        "ban",
        "minh",
        "muon",
        "can",
        "tim",
        "kiem",
        "nha",
        "dat",
        "canho",
        "gia",
        "bao",
        "nhieu",
        "o",
        "tai",
        "quan",
        "huyen",
        "phuong",
        "duong",
        "dien",
        "tich",
        "phong",
        "ngu",
        "phap",
        "ly",
        "so",
        "do",
        "hop",
        "dong",
        "gan",
        "trung",
        "tam",
        "khoang",
        "trieu",
        "ty",
        "co",
        "khong",
        "neu",
        "thi",
        "va",
        "voi",
        "cho",
        "xin",
        "giup",
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
