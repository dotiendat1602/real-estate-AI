from __future__ import annotations

import re
from typing import Optional

from ..utils.text import normalize_text as _normalize_text
from .profiles import planning_focus_phrases, strip_accents
from .ranker import planning_query_terms



def district_code_fragment(district: Optional[str]) -> str:
    if not district:
        return ""
    ascii_text = strip_accents(district)
    parts = re.split(r"[^A-Za-z0-9]+", ascii_text)
    cleaned = [part for part in parts if part]
    return "".join(token[:1].upper() + token[1:] for token in cleaned)


def planning_fact_subqueries(message: str) -> list[str]:
    msg_norm = _normalize_text(message)
    if not msg_norm:
        return []

    query_terms = [term for term in planning_query_terms(message, max_terms=14) if term]
    focus_terms = planning_focus_phrases(message)
    year_terms = re.findall(r"\b20\d{2}\b", msg_norm)

    out: list[str] = []
    if query_terms:
        out.append(" ".join(query_terms[:6]))
        if len(query_terms) > 6:
            out.append(" ".join(query_terms[6:12]))

    for phrase in focus_terms[:4]:
        out.append(phrase)

    if year_terms and query_terms:
        out.append(f"{' '.join(query_terms[:4])} {' '.join(year_terms[:2])}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        key = _normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def planning_query_candidates(
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    max_query_candidates: int,
    max_fact_subqueries: int,
) -> list[str]:
    candidates: list[str] = []
    district_label = (district or "").strip()
    fact_subqueries = planning_fact_subqueries(message)[:max_fact_subqueries]

    candidates.append(message)

    if district_label:
        candidates.append(f"{message} {district_label}")
    if plan_year is not None:
        candidates.append(f"{message} nam {plan_year}")
    if district_label and plan_year is not None:
        candidates.append(f"{message} {district_label} nam {plan_year}")

    focus_terms = list(planning_query_terms(message, max_terms=12))
    if focus_terms:
        focus_query = " ".join(focus_terms)
        candidates.append(focus_query)
        if district_label:
            candidates.append(f"{focus_query} {district_label}")
        if plan_year is not None:
            candidates.append(f"{focus_query} {plan_year}")

    candidates.extend(fact_subqueries)
    for subquery in fact_subqueries:
        if district_label:
            candidates.append(f"{subquery} {district_label}")
        if plan_year is not None:
            candidates.append(f"{subquery} nam {plan_year}")

    if district_label and plan_year is not None:
        district_code = district_code_fragment(district_label)
        if district_code:
            candidates.append(f"HN-{district_code}-KH{plan_year}")

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(candidate)
        if len(out) >= max_query_candidates:
            break
    return out
