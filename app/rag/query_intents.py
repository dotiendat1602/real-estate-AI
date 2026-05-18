from __future__ import annotations

import re

from .text_utils import normalize_text as _normalize_text

PLANNING_FACT_MARKERS = (
    "ke hoach su dung dat",
    "quy hoach",
    "ma ho so",
    "dossier",
    "nam nao",
    "ap dung",
    "khu vuc nao",
    "quan nao",
    "huyen nao",
)

QUERY_LOCATION_HINT_MARKERS = (
    "quan ",
    "huyen ",
    "phuong ",
    "duong ",
    "ngo ",
    "dia chi",
    "ha noi",
    "ho chi minh",
)

SUITABILITY_MARKERS = (
    "phu hop",
    "hop voi",
    "nen chon",
    "co nen",
    "danh gia",
    "nhan xet",
    "loi the",
    "bat loi",
    "uu diem",
    "nhuoc diem",
    "dap ung",
    "thuan tien",
    "hap dan",
    "nhu the nao",
    "ra sao",
)

BUSINESS_MARKERS = (
    "kinh doanh",
    "buon ban",
    "cho thue lai",
    "mat bang",
    "van phong",
    "dong tien",
    "doanh thu",
)

INVESTMENT_MARKERS = (
    "dau tu",
    "sinh loi",
    "tiem nang",
    "tang gia",
    "giu tien",
)

STUDY_WORK_MARKERS = (
    "hoc",
    "lam viec",
    "di lam",
    "truong",
    "dai hoc",
    "van phong",
    "cong ty",
)


def is_planning_fact_question(question: str) -> bool:
    q = _normalize_text(question)
    return any(marker in q for marker in PLANNING_FACT_MARKERS)


def build_query_intents(question: str) -> dict[str, bool]:
    q = _normalize_text(question)
    asks_direction = "huong" in q
    asks_main_door_direction = asks_direction and "cua chinh" in q
    asks_price = (
        any(marker in q for marker in ("gia", "trieu", "ty", "vnd"))
        or "bao nhieu tien" in q
        or any(marker in q for marker in ("ngan sach", "ngan sach thap", "chi phi", "re", "gia re"))
        or ("bao nhieu" in q and any(marker in q for marker in ("ban", "thue")))
        or ("bao" in q and bool(re.search(r"\bnhi\w*\b", q)) and ("gi" in q or "gia" in q))
    )

    return {
        "needs_price": asks_price,
        "needs_area": any(marker in q for marker in ("dien tich", "m2", "met vuong")),
        "needs_direction": asks_direction,
        "needs_main_door_direction": asks_main_door_direction,
        "needs_bedrooms": bool(re.search(r"(?:phong ngu|\d+\s*pn|pn\s*\d+|\bpn\b)", q)),
        "needs_bathrooms": bool(re.search(r"(?:phong ve sinh|toilet|\bwc\b|\bvs\b|\d+\s*wc|\d+\s*vs)", q)),
        "needs_furnishing": any(marker in q for marker in ("noi that", "full noi that", "day du noi that")),
        "needs_min_rental_period": any(marker in q for marker in ("toi thieu", "bao lau", "thoi gian thue")),
        "needs_cashflow": any(marker in q for marker in ("dong tien", "doanh thu", "thu nhap", "cashflow")),
        "needs_indoor_amenities": any(
            marker in q
            for marker in (
                "tien ich trong nha",
                "trong nha",
                "noi that ben trong",
                "diem noi bat gi ve tien ich trong nha",
            )
        ),
        "needs_location": any(
            marker in q
            for marker in ("dia chi", "vi tri", "o dau", "quan nao", "huyen nao", "thanh pho nao")
        ) or any(marker in q for marker in QUERY_LOCATION_HINT_MARKERS),
        "suitability_query": any(marker in q for marker in SUITABILITY_MARKERS),
        "business_query": any(marker in q for marker in BUSINESS_MARKERS),
        "investment_query": any(marker in q for marker in INVESTMENT_MARKERS),
        "study_work_query": any(marker in q for marker in STUDY_WORK_MARKERS),
        "explanatory_query": any(
            marker in q
            for marker in (
                "vi sao",
                "tai sao",
                "ly do",
                "duoc xem la",
                "nhu the nao",
                "ra sao",
            )
        ),
    }
