from __future__ import annotations

from inspect import signature
from typing import Any


def safe_metric_init(metric_cls, **kwargs):
    """Instantiate metric classes defensively across deepeval versions."""
    try:
        return metric_cls(**kwargs)
    except TypeError:
        accepted = set(signature(metric_cls).parameters.keys())
        reduced = {k: v for k, v in kwargs.items() if k in accepted}
        return metric_cls(**reduced)


def try_get_case_params() -> tuple[Any | None, Any | None]:
    """Return optional LLMTestCaseParams symbols if available in installed version."""
    try:
        from deepeval.test_case import LLMTestCaseParams

        return LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT
    except Exception:
        return None, None


def try_get_case_param(name: str) -> Any | None:
    """Return a specific LLMTestCaseParams symbol if available."""
    try:
        from deepeval.test_case import LLMTestCaseParams

        return getattr(LLMTestCaseParams, name, None)
    except Exception:
        return None
