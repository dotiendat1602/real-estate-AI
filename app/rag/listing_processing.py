from __future__ import annotations

from typing import Any
import re

from .query_intents import (
    BUSINESS_MARKERS as _BUSINESS_MARKERS,
    INVESTMENT_MARKERS as _INVESTMENT_MARKERS,
    STUDY_WORK_MARKERS as _STUDY_WORK_MARKERS,
    SUITABILITY_MARKERS as _SUITABILITY_MARKERS,
    build_query_intents as _build_query_intents,
)
from .text_utils import normalize_text as _normalize_text


def extract_query_terms(question: str, max_terms: int = 18) -> tuple[set[str], set[str]]:
    normalized = _normalize_text(question)
    terms = [token for token in normalized.split() if len(token) >= 3]
    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
        if len(unique_terms) >= max_terms:
            break

    number_terms = set(re.findall(r"\d+(?:/\d+)?", question or ""))
    return set(unique_terms), number_terms


def structured_highlights(question: str, highlights: Any, *, prefer_broad: bool = False) -> str | None:
    raw = str(highlights or "").strip()
    if not raw:
        return None

    chunks = [item.strip() for item in raw.split("|") if item.strip()]
    if not chunks:
        return None

    question_terms, _ = extract_query_terms(question)
    intents = _build_query_intents(question)

    selected: list[str] = []
    for item in chunks:
        norm = _normalize_text(item)
        if any(term in norm for term in question_terms):
            selected.append(item)
            continue
        if intents.get("suitability_query") and any(marker in norm for marker in _SUITABILITY_MARKERS):
            selected.append(item)

    if not selected:
        if intents.get("suitability_query"):
            if intents.get("study_work_query"):
                selected = [
                    item
                    for item in chunks
                    if any(marker in _normalize_text(item) for marker in _STUDY_WORK_MARKERS)
                ]
            elif intents.get("business_query") or intents.get("investment_query"):
                selected = [
                    item
                    for item in chunks
                    if any(marker in _normalize_text(item) for marker in (*_BUSINESS_MARKERS, *_INVESTMENT_MARKERS))
                ]
        if not selected and prefer_broad:
            selected = [
                item
                for item in chunks
                if any(marker in _normalize_text(item) for marker in ("khong", "không", *_BUSINESS_MARKERS))
            ]
        if not selected:
            selected = chunks[: (4 if prefer_broad else 2)]

    limit = 6 if prefer_broad else 4
    return "; ".join(selected[:limit])
