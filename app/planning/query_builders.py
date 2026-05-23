from __future__ import annotations

import re
from typing import Optional

from ..utils.text import normalize_text as _normalize_text
from .profiles import strip_accents



def district_code_fragment(district: Optional[str]) -> str:
    if not district:
        return ""
    ascii_text = strip_accents(district)
    parts = re.split(r"[^A-Za-z0-9]+", ascii_text)
    cleaned = [part for part in parts if part]
    return "".join(token[:1].upper() + token[1:] for token in cleaned)


def planning_query_candidates(
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    max_query_candidates: int,
) -> list[str]:
    candidates: list[str] = []
    district_label = (district or "").strip()

    candidates.append(message)

    if district_label:
        candidates.append(f"{message} {district_label}")
    if plan_year is not None:
        candidates.append(f"{message} nam {plan_year}")
    if district_label and plan_year is not None:
        candidates.append(f"{message} {district_label} nam {plan_year}")

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
