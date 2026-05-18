from __future__ import annotations

from typing import Any


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    return str(content or "")


def extract_token_usage(message: Any) -> dict[str, Any]:
    usage_metadata = getattr(message, "usage_metadata", None)
    response_metadata = getattr(message, "response_metadata", None) or {}
    response_usage = {}
    if isinstance(response_metadata, dict):
        response_usage = (
            response_metadata.get("token_usage")
            or response_metadata.get("usage")
            or {}
        )

    input_tokens = _coerce_int(
        _dict_get(usage_metadata, "input_tokens")
        or _dict_get(response_usage, "prompt_tokens")
        or _dict_get(response_usage, "input_tokens")
    )
    output_tokens = _coerce_int(
        _dict_get(usage_metadata, "output_tokens")
        or _dict_get(response_usage, "completion_tokens")
        or _dict_get(response_usage, "output_tokens")
    )
    total_tokens = _coerce_int(
        _dict_get(usage_metadata, "total_tokens")
        or _dict_get(response_usage, "total_tokens")
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    details: dict[str, Any] = {}
    input_details = _dict_get(usage_metadata, "input_token_details") or _dict_get(
        response_usage,
        "prompt_tokens_details",
    )
    output_details = _dict_get(usage_metadata, "output_token_details") or _dict_get(
        response_usage,
        "completion_tokens_details",
    )
    if input_details:
        details["input_token_details"] = input_details
    if output_details:
        details["output_token_details"] = output_details

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        **details,
    }


def sum_token_usage(usages: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = _coerce_int(usage.get(key))
            if value is not None:
                totals[key] += value
    return totals


def _dict_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
