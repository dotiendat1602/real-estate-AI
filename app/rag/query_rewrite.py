from __future__ import annotations

from typing import Any
import re

from .query_intents import QUERY_LOCATION_HINT_MARKERS as _QUERY_LOCATION_HINT_MARKERS
from .text_utils import normalize_text as _normalize_text

_FOLLOW_UP_REFERENCE_MARKERS = (
    " do ",
    " do?",
    " do la",
    " do thi",
    " nay ",
    " nay?",
    " kia",
    " vay",
    " nhu vay",
    "chi tiet nao",
    "nhung chi tiet nao",
    "yeu to nao",
    "nhung diem nao",
    "nhung tien ich nao",
)

_ANCHOR_ENTITY_MARKERS = (
    "nha",
    "can",
    "bat dong san",
    "dat",
    "van phong",
    "mat bang",
    "lo dat",
    "quan",
    "huyen",
    "duong",
    "ngo",
)

def _history_message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "").strip().lower()

    msg_type = str(getattr(message, "type", "") or "").strip().lower()
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"

    role = str(getattr(message, "role", "") or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "ai"}:
        return "assistant"
    return role


def _history_message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return " ".join(chunks).strip()

    return str(content or "").strip()


def _recent_user_messages(history: list[Any] | None, max_messages: int = 4) -> list[str]:
    if not history:
        return []

    out: list[str] = []
    for message in reversed(history):
        if _history_message_role(message) != "user":
            continue
        content = _history_message_content(message)
        if not content:
            continue
        out.append(content)
        if len(out) >= max_messages:
            break

    return list(reversed(out))


def _looks_follow_up_question(question: str) -> bool:
    normalized = _normalize_text(question)
    if not normalized:
        return False

    padded = f" {normalized} "
    if any(marker in padded for marker in _FOLLOW_UP_REFERENCE_MARKERS):
        return True

    if normalized.startswith(("neu ", "vay ", "the ", "con ", "va ")):
        return True

    token_count = len(normalized.split())
    has_location_signal = any(marker in normalized for marker in _QUERY_LOCATION_HINT_MARKERS)
    has_number_signal = bool(re.search(r"\d+(?:/\d+)?", question or ""))
    return token_count <= 10 and not has_location_signal and not has_number_signal


def _is_anchor_rich_message(message: str) -> bool:
    normalized = _normalize_text(message)
    if not normalized:
        return False

    if re.search(r"\d+(?:/\d+)?", message or ""):
        return True

    if any(marker in normalized for marker in _QUERY_LOCATION_HINT_MARKERS):
        return True

    return len(normalized.split()) >= 6 and any(marker in normalized for marker in _ANCHOR_ENTITY_MARKERS)


def build_retrieval_query(question: str, history: list[Any] | None = None) -> str:
    """Rewrite retrieval query for follow-up turns so retrieval stays anchored to prior user context."""
    current = (question or "").strip()
    if not current:
        return current

    if not history or not _looks_follow_up_question(current):
        return current

    user_messages = _recent_user_messages(history, max_messages=4)
    if not user_messages:
        return current

    anchor = ""
    for candidate in reversed(user_messages):
        if _is_anchor_rich_message(candidate):
            anchor = candidate.strip()
            break

    if not anchor:
        anchor = user_messages[-1].strip()

    if not anchor:
        return current

    current_norm = _normalize_text(current)
    anchor_norm = _normalize_text(anchor)
    if anchor_norm and (anchor_norm in current_norm or current_norm in anchor_norm):
        return current

    combined = (
        f"Bat dong san dang duoc nhac toi: {anchor}\n"
        f"Cau hoi hien tai: {current}"
    )
    return combined[:500]
