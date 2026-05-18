from __future__ import annotations

from ..utils.text import normalize_text as _normalize_ranker_text

_PLANNING_QUERY_TERM_STOPWORDS = {
    "bao",
    "nhieu",
    "tong",
    "tongcong",
    "duoc",
    "nhu",
    "the",
    "nao",
    "ra",
    "sao",
    "trong",
    "theo",
    "voi",
    "cua",
    "tren",
    "nam",
    "quan",
    "huyen",
    "thi",
    "xa",
    "thanh",
    "pho",
    "hoach",
    "ke",
    "su",
    "dung",
    "dat",
    "ve",
    "cac",
    "nhung",
    "la",
}

def planning_query_terms(message: str, max_terms: int = 28) -> list[str]:
    normalized = _normalize_ranker_text(message)
    if not normalized:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for token in normalized.split():
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in _PLANNING_QUERY_TERM_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max_terms:
            break

    return out
