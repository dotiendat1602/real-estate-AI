from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .langchain_adapter import ConversationTrace, TurnTrace


_DIRECTIVE_PREFIXES = (
    "neu dung",
    "neu ro",
    "xac dinh dung",
    "xac dinh",
    "tra loi dung",
    "tra loi ro",
    "bot tra loi dung",
    "bot neu dung",
)

_GUIDANCE_PREFIXES = (
    "khong suy doan",
    "khong tra loi",
    "khong bo sung",
    "khong them",
    "khong nham",
    "khong ke nham",
    "khong suy dien",
    "khong tu suy dien",
    "tranh nham",
    "tranh lan man",
    "tra loi ngan gon",
    "tra loi mot y",
    "chi tra loi",
    "chi can",
    "co the",
    "optional",
)

_GUIDANCE_CONTAINS = (
    "neu can",
    "neu muon",
    "khong suy doan",
    "khong suy dien",
    "khong tra loi",
    "khong nham",
    "khong ke nham",
)

_FACT_HINTS = (
    "gia",
    "dien tich",
    "phong ngu",
    "phong ve sinh",
    "huong",
    "noi that",
    "quan",
    "huyen",
    "thanh pho",
    "dossier",
    "planyear",
    "dong tien",
    "du an",
)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = lowered.replace("đ", "d")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _strip_list_prefix(line: str) -> str:
    text = (line or "").strip()
    text = re.sub(r"^[-*•]+\s*", "", text)
    text = re.sub(r"^\d+[\.)]\s*", "", text)
    return text.strip()


def _looks_guidance_line(normalized: str) -> bool:
    if not normalized:
        return False
    if normalized.startswith(_GUIDANCE_PREFIXES):
        return True
    return any(marker in normalized for marker in _GUIDANCE_CONTAINS)


def _is_fact_like_line(line: str, normalized: str) -> bool:
    if not normalized:
        return False
    if re.search(r"\d", line):
        return True
    if re.search(r"\b[A-Z_]{3,}\b", line):
        return True
    return any(marker in normalized for marker in _FACT_HINTS)


def _rephrase_directive_fact(head_normalized: str, tail: str) -> str:
    payload = (tail or "").strip()
    if not payload:
        return payload

    if any(marker in head_normalized for marker in ("gia ban", "gia thue", "gia")):
        return f"Giá: {payload}"
    if "dien tich" in head_normalized:
        return f"Diện tích: {payload}"
    if "phong ngu" in head_normalized:
        return f"Số phòng ngủ: {payload}"
    if any(marker in head_normalized for marker in ("phong ve sinh", "wc", "vs")):
        return f"Số phòng vệ sinh: {payload}"
    if "noi that" in head_normalized:
        return f"Nội thất: {payload}"
    if "huong" in head_normalized:
        return f"Hướng: {payload}"
    if any(marker in head_normalized for marker in ("quan", "huyen", "thanh pho")):
        return f"Quận/Huyện/Thành phố: {payload}"
    if any(marker in head_normalized for marker in ("thoi gian", "toi thieu")):
        return f"Thời gian thuê tối thiểu: {payload}"
    if any(marker in head_normalized for marker in ("so tang", "tang toi da")):
        return f"Số tầng tối đa: {payload}"
    if "dong tien" in head_normalized:
        return f"Dòng tiền: {payload}"

    return payload


def _extract_directive_payload(candidate: str, normalized: str) -> str | None:
    for prefix in _DIRECTIVE_PREFIXES:
        if not normalized.startswith(prefix):
            continue

        words = candidate.split()
        prefix_len = len(prefix.split())
        if len(words) <= prefix_len:
            return None

        payload_words = words[prefix_len:]
        while payload_words and _normalize_text(payload_words[0]) in {"la", "ve", "day", "viec", "rang"}:
            payload_words = payload_words[1:]

        payload = " ".join(payload_words).strip()
        return payload or None

    return None


def _sanitize_expected_output_for_recall(expected_output: str) -> str:
    """
    Keep factual targets while dropping rubric-only instructions that are not retrieval facts.
    """
    lines = [line.strip() for line in (expected_output or "").splitlines() if line.strip()]
    if not lines:
        return expected_output

    sanitized: list[str] = []
    for line in lines:
        candidate = _strip_list_prefix(line)
        if not candidate:
            continue

        normalized = _normalize_text(candidate)
        if _looks_guidance_line(normalized):
            continue

        # Convert directive lines to factual payload when possible.
        if ":" in candidate:
            head, tail = candidate.split(":", 1)
            head_norm = _normalize_text(head)
            if head_norm.startswith(_DIRECTIVE_PREFIXES):
                factual_tail = tail.strip()
                if factual_tail and not _looks_guidance_line(_normalize_text(factual_tail)):
                    sanitized.append(_rephrase_directive_fact(head_norm, factual_tail))
                continue

        # Most directive lines without payload are rubric-style instructions.
        if normalized.startswith(_DIRECTIVE_PREFIXES):
            payload = _extract_directive_payload(candidate, normalized)
            if payload and not _looks_guidance_line(_normalize_text(payload)):
                sanitized.append(payload)
            continue

        if _is_fact_like_line(candidate, normalized):
            sanitized.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for line in sanitized:
        key = _normalize_text(line)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    if not deduped:
        return expected_output

    return "\n".join(deduped)


def build_single_turn_test_case(trace: TurnTrace, golden: dict[str, Any]):
    from deepeval.test_case import LLMTestCase

    expected_output_raw = golden.get("expected_output") or "\n".join(golden.get("expected_output_outline", []))
    expected_output = _sanitize_expected_output_for_recall(str(expected_output_raw or ""))

    return LLMTestCase(
        input=trace.input,
        actual_output=trace.actual_output,
        retrieval_context=trace.retrieval_context,
        context=golden.get("context"),
        expected_output=expected_output,
    )


def build_conversation_test_case(trace: ConversationTrace, golden: dict[str, Any]):
    from deepeval.test_case import ConversationalTestCase, Turn

    turns: list[Turn] = []
    for item in trace.turns:
        turns.append(Turn(role="user", content=item.input))
        turns.append(Turn(role="assistant", content=item.actual_output))

    return ConversationalTestCase(
        scenario=golden.get("scenario", "Conversation quality check"),
        expected_outcome=golden.get("expected_outcome", "Assistant stays consistent and grounded."),
        turns=turns,
        context=golden.get("context"),
        chatbot_role=golden.get(
            "chatbot_role",
            "RAG chatbot for real-estate and land-use-planning assistance",
        ),
    )
