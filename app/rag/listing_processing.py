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

_SUITABILITY_REASONING_MARKERS = (
    "giup",
    "thuan tien",
    "phu hop",
    "loi the",
    "uu diem",
    "dap ung",
)

_SUITABILITY_DETAIL_FACT_MARKERS = (
    "gia",
    "dien tich",
    "phong ngu",
    "phong ve sinh",
    "huong",
    "noi that",
    "thoi gian thue",
    "dong tien",
)

_LOCATION_FACT_MARKERS = (
    "dia chi",
    "vi tri",
    "quan",
    "huyen",
    "phuong",
    "duong",
    "gan",
)

_OUTDOOR_DETAIL_MARKERS = (
    "mat tien",
    "duong vao",
    "o to",
    "giao thong",
    "vi tri",
    "cong vien",
    "aeon",
    "truc duong",
    "phong ngu",
    "phong ve sinh",
)


def _looks_uncertain_or_no_data(answer: str) -> bool:
    normalized = _normalize_text(answer)
    return any(
        marker in normalized
        for marker in (
            "khong co thong tin",
            "khong tim thay",
            "khong du thong tin",
            "khong biet",
            "chua co du lieu",
            "i do not know",
        )
    )


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


def strip_outdoor_detail_lines(answer: str) -> str:
    lines = (answer or "").splitlines()
    kept: list[str] = []
    for line in lines:
        normalized = _normalize_text(line)
        if normalized and any(marker in normalized for marker in _OUTDOOR_DETAIL_MARKERS):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def condense_suitability_answer(question: str, answer: str) -> str:
    intents = _build_query_intents(question)
    if not intents.get("suitability_query"):
        return answer

    cleaned_answer = (answer or "").strip()
    if not cleaned_answer or _looks_uncertain_or_no_data(cleaned_answer):
        return cleaned_answer

    raw_lines = [line.strip() for line in cleaned_answer.splitlines() if line.strip()]
    lines: list[str] = []
    split_applied = False
    for line in raw_lines:
        if len(raw_lines) <= 2 and len(line) > 140:
            parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip()]
            split_applied = split_applied or len(parts) > 1
            lines.extend(parts or [line])
        else:
            lines.append(line)
    if len(lines) <= 2 and not split_applied:
        return cleaned_answer

    query_terms, _ = extract_query_terms(question, max_terms=24)
    query_terms = {term for term in query_terms if len(term) >= 4}

    requires_specific_fact = any(
        intents.get(key)
        for key in (
            "needs_price",
            "needs_area",
            "needs_direction",
            "needs_bedrooms",
            "needs_bathrooms",
            "needs_furnishing",
            "needs_location",
            "needs_cashflow",
            "needs_min_rental_period",
            "needs_indoor_amenities",
        )
    )
    needs_location_signal = bool(
        intents.get("needs_location")
        or intents.get("study_work_query")
        or intents.get("business_query")
    )

    kept_lines: list[str] = []
    for idx, line in enumerate(lines):
        normalized_line = _normalize_text(line)
        if not normalized_line:
            continue

        is_intro_line = idx == 0 and not line.lstrip().startswith(("-", "*"))
        if is_intro_line:
            kept_lines.append(line)
            continue

        overlap_hits = sum(1 for term in query_terms if term in normalized_line)
        has_suitability_signal = any(marker in normalized_line for marker in _SUITABILITY_MARKERS)
        has_reasoning_signal = any(marker in normalized_line for marker in _SUITABILITY_REASONING_MARKERS)
        has_business_signal = any(marker in normalized_line for marker in _BUSINESS_MARKERS)
        has_investment_signal = any(marker in normalized_line for marker in _INVESTMENT_MARKERS)
        has_study_signal = any(marker in normalized_line for marker in _STUDY_WORK_MARKERS)
        has_detail_fact = any(marker in normalized_line for marker in _SUITABILITY_DETAIL_FACT_MARKERS)
        has_location_fact = any(marker in normalized_line for marker in _LOCATION_FACT_MARKERS)

        if not intents.get("needs_price") and ("gia" in normalized_line or "vnd" in normalized_line):
            if not (overlap_hits or has_reasoning_signal or has_business_signal or has_investment_signal):
                continue

        if not intents.get("needs_area") and normalized_line.startswith("dien tich"):
            if not (overlap_hits or has_reasoning_signal):
                continue

        if not needs_location_signal and has_location_fact:
            if not (overlap_hits or has_reasoning_signal):
                continue

        if has_detail_fact and not requires_specific_fact:
            if not (overlap_hits or has_reasoning_signal or has_business_signal or has_investment_signal):
                continue

        if (
            overlap_hits
            or has_suitability_signal
            or has_reasoning_signal
            or has_business_signal
            or has_investment_signal
            or has_study_signal
        ):
            kept_lines.append(line)
            continue

        if requires_specific_fact and (has_detail_fact or has_location_fact):
            kept_lines.append(line)

    if not kept_lines:
        return cleaned_answer

    deduped_lines: list[str] = []
    seen_norm: set[str] = set()
    for line in kept_lines:
        key = _normalize_text(line)
        if not key or key in seen_norm:
            continue
        seen_norm.add(key)
        deduped_lines.append(line)

    if not deduped_lines:
        return cleaned_answer

    intro: list[str] = []
    body: list[str] = []
    for line in deduped_lines:
        if not intro and not line.lstrip().startswith(("-", "*")):
            intro.append(line)
        else:
            body.append(line)

    body_limit = 4 if requires_specific_fact else 3
    condensed = [*intro, *body[:body_limit]] if intro else body[:body_limit]
    final_answer = "\n".join(condensed).strip()
    return final_answer or cleaned_answer


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
                if any(marker in _normalize_text(item) for marker in ("khong", "không", *(_BUSINESS_MARKERS)))
            ]
        if not selected:
            selected = chunks[: (4 if prefer_broad else 2)]

    limit = 6 if prefer_broad else 4
    return "; ".join(selected[:limit])
