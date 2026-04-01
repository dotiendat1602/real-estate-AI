from __future__ import annotations

import argparse
import asyncio
import os
import uuid
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig

from evaluation.adapters.langchain_adapter import LangChainEvalAdapter, conversation_trace_to_dict
from evaluation.adapters.trace_to_testcase import build_conversation_test_case
from evaluation.metrics.custom_gevals import build_conversational_geval
from evaluation.runners.common import (
    load_json,
    load_yaml,
    now_utc_iso,
    to_jsonable,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline conversational RAG evaluation.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to evaluation folder",
    )
    parser.add_argument(
        "--output-name",
        default="baseline_eval_conversation.json",
        help="Output report filename under evaluation/reports",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, evaluate only the first N conversation cases",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent async metric evaluations to reduce API rate-limit errors",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.2,
        help="Delay between async metric requests in seconds",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run metric evaluation sequentially (more stable, slower)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    eval_root = Path(args.root)

    thresholds = load_yaml(eval_root / "config" / "thresholds.yaml")
    settings = load_yaml(eval_root / "config" / "eval_settings.yaml")
    judge = load_yaml(eval_root / "config" / "judge_model.yaml")
    goldens = load_json(eval_root / "datasets" / "conversational_goldens.json")

    judge_model = judge.get("judge", {}).get("model")
    if judge_model:
        os.environ.setdefault("DEEPEVAL_MODEL", str(judge_model))

    if args.max_cases and args.max_cases > 0:
        goldens = goldens[: args.max_cases]

    adapter = LangChainEvalAdapter(settings=settings)
    traces = []
    test_cases = []

    for golden in goldens:
        trace = await adapter.run_conversation(golden.get("id", "conv_case"), golden.get("turns", []))
        traces.append({"golden": golden, "trace": conversation_trace_to_dict(trace)})
        test_cases.append(build_conversation_test_case(trace, golden))

    conv_metric = build_conversational_geval(thresholds, judge_model=judge_model)
    eval_result = evaluate(
        test_cases=test_cases,
        metrics=[conv_metric],
        async_config=AsyncConfig(
            run_async=not bool(args.sync),
            throttle_value=max(0.0, float(args.throttle_seconds)),
            max_concurrent=max(1, int(args.max_concurrent)),
        ),
    )

    run_id = str(uuid.uuid4())
    report_payload = {
        "run_id": run_id,
        "created_at": now_utc_iso(),
        "phase": "conversation",
        "judge": judge,
        "thresholds": thresholds,
        "settings": settings,
        "summary": {
            "cases": len(test_cases),
            "difficulties": _count_by_field(goldens, "difficulty"),
            "domains": _count_by_field(goldens, "domain"),
        },
        "deepeval_result": to_jsonable(eval_result),
        "traces": traces,
    }

    report_path = eval_root / "reports" / args.output_name
    write_json(report_path, report_payload)


def _count_by_field(items: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field, "unknown"))
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
