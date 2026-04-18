from __future__ import annotations

import re
from typing import Callable

from langchain_core.documents import Document


def _has_grouping_project_context(normalized_blob: str) -> bool:
    return re.search(
        r"(?:tong\s+du\s+an|tong\s+so\s+du\s+an|du\s+an\s+thu\s+hoi|du\s+an\s+khong\s+thu\s+hoi|"
        r"theo\s+dieu\s*79|theo\s+dieu\s*78|chuyen\s+tiep|dang\s+ky\s+moi)",
        normalized_blob,
    ) is not None


def _has_large_grouping_project_count(normalized_blob: str) -> bool:
    # OCR table rows often contain standalone counts (e.g., "62 59 3") without "du an" suffix.
    if re.search(r"\b([2-9]\d|1\d{2})\s*du\s+an\b", normalized_blob) is not None:
        return True

    if not _has_grouping_project_context(normalized_blob):
        return False

    return re.search(r"\b([2-9]\d|1\d{2})\b", normalized_blob) is not None


def is_recovery_grouping_query(normalized_message: str) -> bool:
    """Detect queries that split projects into recovery vs non-recovery groups."""
    if not normalized_message:
        return False
    if "thu hoi" not in normalized_message:
        return False
    return any(
        marker in normalized_message
        for marker in (
            "khong thu hoi",
            "khong thuoc truong hop thu hoi",
            "thu hoi dat va khong thu hoi dat",
        )
    )


def recovery_grouping_signal_score(normalized_blob: str) -> float:
    """Score evidence quality for recovery/non-recovery analytical grouping."""
    if not normalized_blob:
        return 0.0

    has_article_7879 = re.search(r"\bdieu\s*78\b|\bdieu\s*79\b", normalized_blob) is not None
    has_article_6162 = re.search(r"\bdieu\s*61\b|\bdieu\s*62\b", normalized_blob) is not None
    has_large_project_count = _has_large_grouping_project_count(normalized_blob)

    has_recovery_group = (
        re.search(r"(?:du\s+an|cong\s+trinh)[^\n]{0,100}thu\s+hoi", normalized_blob) is not None
        or "thuoc truong hop thu hoi" in normalized_blob
    )
    has_non_recovery_group = (
        re.search(r"khong\s+thuoc\s+truong\s+hop\s+thu\s+hoi", normalized_blob) is not None
        or re.search(r"khong\s+thu\s+hoi", normalized_blob) is not None
    )
    has_project_count = re.search(r"\b\d+(?:[\.,]\d+)?\s*du\s+an\b", normalized_blob) is not None
    has_area_count = re.search(r"\b\d+(?:[\.,]\d+)?\s*ha\b", normalized_blob) is not None
    has_toc_like_heading = re.search(r"bang\s*\d+:[^\n]{0,220}\.{2,}\s*\d+", normalized_blob) is not None

    score = 0.0
    if has_article_7879:
        score += 6.0 if has_large_project_count else 1.5
    if has_recovery_group and has_non_recovery_group:
        score += 6.0
    if has_project_count:
        score += 4.0
    if has_area_count:
        score += 2.0
    if has_article_7879 and not has_large_project_count:
        score -= 2.0
    if has_toc_like_heading and not has_large_project_count:
        score -= 6.0
    if has_article_6162 and not has_article_7879:
        score -= 4.0
    return score


def has_min_recovery_grouping_evidence(normalized_blob: str) -> bool:
    """Require both groups and project counts to pass analytical grouping quality gate."""
    if not normalized_blob:
        return False

    has_article_7879 = re.search(r"\bdieu\s*78\b|\bdieu\s*79\b", normalized_blob) is not None
    has_recovery_group = (
        re.search(r"(?:du\s+an|cong\s+trinh)[^\n]{0,100}thu\s+hoi", normalized_blob) is not None
        or "thuoc truong hop thu hoi" in normalized_blob
    )
    has_non_recovery_group = (
        re.search(r"khong\s+thuoc\s+truong\s+hop\s+thu\s+hoi", normalized_blob) is not None
        or re.search(r"khong\s+thu\s+hoi", normalized_blob) is not None
    )
    has_project_count = (
        re.search(r"\b\d+(?:[\.,]\d+)?\s*du\s+an\b", normalized_blob) is not None
        or (_has_grouping_project_context(normalized_blob) and re.search(r"\b\d+(?:[\.,]\d+)?\b", normalized_blob) is not None)
    )
    has_large_project_count = _has_large_grouping_project_count(normalized_blob)

    if not (has_article_7879 and has_recovery_group and has_non_recovery_group and has_project_count and has_large_project_count):
        return False

    return recovery_grouping_signal_score(normalized_blob) >= 8.0


def score_planning_fallback_candidate(
    docs: list[Document],
    doc_score_fn: Callable[[Document], float],
) -> float:
    """Compute a stable fallback quality score from top-ranked docs."""
    if not docs:
        return float("-inf")

    top_docs = docs[: min(4, len(docs))]
    mean_top_score = sum(doc_score_fn(doc) for doc in top_docs) / len(top_docs)
    # Slightly favor richer candidate sets while capping noise contribution.
    density_bonus = min(len(docs), 8) * 0.02
    return mean_top_score + density_bonus


def choose_better_planning_fallback(
    current_docs: list[Document],
    current_score: float,
    candidate_docs: list[Document],
    doc_score_fn: Callable[[Document], float],
) -> tuple[list[Document], float]:
    """Keep the stronger fallback candidate across retrieval passes."""
    if not candidate_docs:
        return current_docs, current_score

    candidate_score = score_planning_fallback_candidate(candidate_docs, doc_score_fn)
    if not current_docs or candidate_score > current_score:
        return list(candidate_docs), candidate_score

    return current_docs, current_score
