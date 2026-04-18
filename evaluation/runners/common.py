from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return to_jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown_summary(title: str, sections: list[str]) -> str:
    body = "\n\n".join(sections)
    return f"# {title}\n\n{body}\n"


def apply_deepeval_timeout_env(
    judge_config: dict[str, Any] | None,
    per_attempt_timeout_seconds: float | None = None,
    disable_timeouts: bool = False,
) -> dict[str, Any]:
    """Apply DeepEval timeout envs for long-running judge calls.

    - If disable_timeouts=True: disable DeepEval-enforced timeouts.
    - Else: set per-attempt timeout override.
      When not explicitly provided, auto-pick a safer value for slower judges.
    """
    if disable_timeouts:
        os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "true"
        os.environ.pop("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", None)
        os.environ.pop("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", None)
        return {
            "disable_timeouts": True,
            "per_attempt_timeout_seconds": None,
        }

    # Ensure timeout system is enabled when we provide explicit overrides.
    os.environ.pop("DEEPEVAL_DISABLE_TIMEOUTS", None)

    timeout_value: float
    explicit_timeout = (
        float(per_attempt_timeout_seconds)
        if per_attempt_timeout_seconds is not None
        else 0.0
    )
    if explicit_timeout > 0:
        timeout_value = explicit_timeout
    else:
        judge = judge_config.get("judge", {}) if isinstance(judge_config, dict) else {}
        raw_timeout = judge.get("timeout_seconds", 180)
        try:
            judge_timeout = float(raw_timeout)
        except (TypeError, ValueError):
            judge_timeout = 180.0
        # DeepSeek judge can be slow for detailed metrics; keep a safer floor.
        timeout_value = max(300.0, judge_timeout)

    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = str(timeout_value)
    # Let DeepEval compute per-task budget from per-attempt/retry settings.
    os.environ.pop("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", None)

    return {
        "disable_timeouts": False,
        "per_attempt_timeout_seconds": timeout_value,
    }
