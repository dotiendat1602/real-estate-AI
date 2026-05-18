from __future__ import annotations

import re
import unicodedata

_MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00c4", "\u00c6", "\u00e2\u20ac", "\u2013", "\u2014", "\ufffd")


def repair_mojibake(text: str) -> str:
    raw = str(text or "")
    if not raw or not any(marker in raw for marker in _MOJIBAKE_MARKERS):
        return raw

    candidates = [raw]
    for codec in ("latin1", "cp1252"):
        current = raw
        for _ in range(2):
            try:
                repaired = current.encode(codec).decode("utf-8")
            except Exception:
                break
            candidates.append(repaired)
            current = repaired

    def _score(value: str) -> tuple[int, int]:
        marker_hits = sum(value.count(marker) for marker in _MOJIBAKE_MARKERS)
        replacement_hits = value.count("\ufffd")
        return marker_hits + replacement_hits * 2, len(value)

    return min(candidates, key=_score)


def strip_vietnamese_accents(text: str) -> str:
    normalized = repair_mojibake(text or "").replace("\u0111", "d").replace("\u0110", "D")
    decomposed = unicodedata.normalize("NFD", normalized)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_vietnamese_search_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = strip_vietnamese_accents(lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def strip_accents(text: str) -> str:
    normalized = (
        text or ""
    ).replace("\u0111", "d").replace("\u0110", "D").replace("\u00c4\u2018", "d").replace(
        "\u00c4\u0090", "D"
    ).replace(
        "\u00c3\u201e\u00e2\u20ac\u02dc", "d"
    )
    decomposed = unicodedata.normalize("NFD", normalized)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    lowered = strip_accents((text or "").lower().strip())
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def sanitize_llm_text(text: str, *, max_len: int | None = None) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    normalized = unicodedata.normalize("NFKC", raw.replace("\x00", " "))
    cleaned_chars: list[str] = []
    for ch in normalized:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            continue
        if code < 32 and ch not in {"\n", "\t"}:
            continue
        if code == 127:
            continue
        cleaned_chars.append(ch)

    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if max_len is not None and len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned
