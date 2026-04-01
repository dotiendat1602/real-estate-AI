from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.adapters.langchain_adapter import (
    ConversationTrace,
    LangChainEvalAdapter,
    TurnTrace,
)


async def generate_single_turn_traces(
    adapter: LangChainEvalAdapter,
    goldens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for golden in goldens:
        trace: TurnTrace = await adapter.run_single_turn(golden["input"])
        traces.append({
            "golden": golden,
            "trace": asdict(trace),
        })
    return traces


async def generate_conversation_traces(
    adapter: LangChainEvalAdapter,
    goldens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for golden in goldens:
        trace: ConversationTrace = await adapter.run_conversation(
            conversation_id=golden.get("id", "unknown_conversation"),
            turns=golden.get("turns", []),
        )
        traces.append({
            "golden": golden,
            "trace": {
                "conversation_id": trace.conversation_id,
                "turns": [asdict(turn) for turn in trace.turns],
            },
        })
    return traces


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
