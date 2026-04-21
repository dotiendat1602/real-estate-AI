from __future__ import annotations

import argparse
import asyncio
import re
import uuid
import sys
import time
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience dependency
    load_dotenv = None

from evaluation.adapters.langchain_adapter import ConversationTrace, LangChainEvalAdapter, TurnTrace, conversation_trace_to_dict
from evaluation.adapters.trace_to_testcase import build_conversation_test_case
from evaluation.metrics.custom_gevals import build_knowledge_retention_metric
from evaluation.metrics.metric_helpers import (
    build_conversation_scorecard,
    resolve_judge_model_and_apply_env,
)
from evaluation.runners.common import (
    apply_deepeval_timeout_env,
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
        "--dataset",
        default="conversation_goldens_hard_tuned.json",
        help="Dataset filename under evaluation/datasets",
    )
    parser.add_argument(
        "--trace-input",
        default="",
        help="Optional conversation trace checkpoint JSON to score without regenerating traces. Relative paths are resolved under evaluation/reports.",
    )
    parser.add_argument(
        "--resume-trace-input",
        default="",
        help="Optional partial trace checkpoint JSON to reuse while generating missing traces. Relative paths are resolved under evaluation/reports.",
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
        "--trace-max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent conversation trace generation calls before DeepEval scoring",
    )
    parser.add_argument(
        "--trace-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each conversation trace generation case before failing",
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
    parser.add_argument(
        "--per-attempt-timeout-seconds",
        type=float,
        default=0.0,
        help="DeepEval timeout per provider attempt in seconds; <= 0 uses auto value",
    )
    parser.add_argument(
        "--disable-deepeval-timeouts",
        action="store_true",
        help="Disable DeepEval-enforced timeouts (provider SDK timeouts still apply)",
    )
    return parser.parse_args()


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


async def main() -> None:
    _configure_utf8_stdio()
    args = parse_args()
    eval_root = Path(args.root)
    if load_dotenv is not None:
        load_dotenv(eval_root.parent / ".env")

    thresholds = load_yaml(eval_root / "config" / "thresholds.yaml")
    settings = load_yaml(eval_root / "config" / "eval_settings.yaml")
    judge = load_yaml(eval_root / "config" / "judge_model.yaml")
    goldens = load_json(eval_root / "datasets" / args.dataset)

    timeout_config = apply_deepeval_timeout_env(
        judge_config=judge,
        per_attempt_timeout_seconds=float(args.per_attempt_timeout_seconds),
        disable_timeouts=bool(args.disable_deepeval_timeouts),
    )

    if args.max_cases and args.max_cases > 0:
        goldens = goldens[: args.max_cases]

    trace_checkpoint_path = eval_root / "reports" / f"{Path(args.output_name).stem}.traces.json"
    trace_started_at = time.perf_counter()
    trace_max_concurrent = max(1, int(args.trace_max_concurrent))

    if args.trace_input:
        trace_input_path = Path(args.trace_input)
        if not trace_input_path.is_absolute():
            trace_input_path = eval_root / "reports" / trace_input_path
        checkpoint = load_json(trace_input_path)
        trace_records = checkpoint.get("traces") or []
        if args.max_cases and args.max_cases > 0:
            trace_records = trace_records[: args.max_cases]
        traces = trace_records
        goldens = [record["golden"] for record in traces]
        test_cases = [
            build_conversation_test_case(_conversation_trace_from_record(record["trace"]), record["golden"])
            for record in traces
        ]
        print(f"[ConversationTrace] Loaded {len(test_cases)} traces from {trace_input_path}", flush=True)
    else:
        adapter = LangChainEvalAdapter(settings=settings)
        trace_semaphore = asyncio.Semaphore(trace_max_concurrent)
        trace_attempts = max(1, int(args.trace_attempts))
        resumed_by_id: dict[str, dict] = {}
        if args.resume_trace_input:
            resume_trace_path = Path(args.resume_trace_input)
            if not resume_trace_path.is_absolute():
                resume_trace_path = eval_root / "reports" / resume_trace_path
            resume_checkpoint = load_json(resume_trace_path)
            for record in resume_checkpoint.get("traces") or []:
                golden_id = str(record.get("golden", {}).get("id", ""))
                if golden_id:
                    resumed_by_id[golden_id] = record
            print(f"[ConversationTrace] Resuming with {len(resumed_by_id)} traces from {resume_trace_path}", flush=True)

        async def build_trace_case(index: int, golden: dict):
            async with trace_semaphore:
                case_started_at = time.perf_counter()
                print(
                    f"[ConversationTrace] {index + 1}/{len(goldens)} {golden.get('id', 'conv_case')} start",
                    flush=True,
                )
                for attempt in range(1, trace_attempts + 1):
                    try:
                        trace = await adapter.run_conversation(
                            golden.get("id", "conv_case"),
                            golden.get("turns", []),
                            target_metadata=_resolve_target_metadata(golden),
                        )
                        break
                    except Exception as exc:
                        if attempt >= trace_attempts:
                            print(
                                f"[ConversationTrace] {index + 1}/{len(goldens)} {golden.get('id', 'conv_case')} "
                                f"failed after {attempt} attempts: {type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            raise
                        delay = min(60.0, 5.0 * attempt)
                        print(
                            f"[ConversationTrace] {index + 1}/{len(goldens)} {golden.get('id', 'conv_case')} "
                            f"attempt {attempt} failed: {type(exc).__name__}: {exc}; retrying in {delay}s",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
                print(
                    f"[ConversationTrace] {index + 1}/{len(goldens)} {golden.get('id', 'conv_case')} "
                    f"done in {round(time.perf_counter() - case_started_at, 3)}s turns={len(trace.turns)}",
                    flush=True,
                )
                return (
                    index,
                    {"golden": golden, "trace": conversation_trace_to_dict(trace)},
                    build_conversation_test_case(trace, golden),
                )

        trace_records: list[dict | None] = [None] * len(goldens)
        test_case_records = [None] * len(goldens)
        pending_indices: list[int] = []
        for index, golden in enumerate(goldens):
            resumed_record = resumed_by_id.get(str(golden.get("id", "")))
            if resumed_record:
                trace_records[index] = resumed_record
                test_case_records[index] = build_conversation_test_case(
                    _conversation_trace_from_record(resumed_record["trace"]),
                    resumed_record["golden"],
                )
            else:
                pending_indices.append(index)

        def write_trace_checkpoint(completed_count: int, phase: str = "conversation_trace_generation_partial") -> None:
            write_json(
                trace_checkpoint_path,
                {
                    "created_at": now_utc_iso(),
                    "phase": phase,
                    "dataset": args.dataset,
                    "cases": completed_count,
                    "total_cases": len(goldens),
                    "trace_max_concurrent": trace_max_concurrent,
                    "timing": {
                        "trace_generation_seconds_so_far": round(time.perf_counter() - trace_started_at, 3),
                    },
                    "traces": [record for record in trace_records if record is not None],
                },
            )

        if trace_max_concurrent == 1:
            for index in pending_indices:
                golden = goldens[index]
                result_index, trace_record, test_case = await build_trace_case(index, golden)
                trace_records[result_index] = trace_record
                test_case_records[result_index] = test_case
                write_trace_checkpoint(sum(record is not None for record in trace_records))
        else:
            tasks = [asyncio.create_task(build_trace_case(index, goldens[index])) for index in pending_indices]
            for task in asyncio.as_completed(tasks):
                result_index, trace_record, test_case = await task
                trace_records[result_index] = trace_record
                test_case_records[result_index] = test_case
                write_trace_checkpoint(sum(record is not None for record in trace_records))

        traces = [record for record in trace_records if record is not None]
        test_cases = [test_case for test_case in test_case_records if test_case is not None]
        write_json(
            trace_checkpoint_path,
            {
                "created_at": now_utc_iso(),
                "phase": "conversation_trace_generation",
                "dataset": args.dataset,
                "cases": len(test_cases),
                "total_cases": len(goldens),
                "trace_max_concurrent": trace_max_concurrent,
                "timing": {
                    "trace_generation_seconds": round(time.perf_counter() - trace_started_at, 3),
                },
                "traces": traces,
            },
        )

    # Apply judge provider environment only for scoring to avoid impacting runtime QA model.
    judge_model = resolve_judge_model_and_apply_env(judge)

    conv_metric = build_knowledge_retention_metric(thresholds, judge_model=judge_model)
    print(f"[ConversationEval] scoring {len(test_cases)} cases", flush=True)
    eval_result = evaluate(
        test_cases=test_cases,
        metrics=[conv_metric],
        async_config=AsyncConfig(
            run_async=not bool(args.sync),
            throttle_value=max(0.0, float(args.throttle_seconds)),
            max_concurrent=max(1, int(args.max_concurrent)),
        ),
    )
    deepeval_result = to_jsonable(eval_result)
    scorecard = build_conversation_scorecard(
        deepeval_result=deepeval_result,
        thresholds=thresholds,
    )

    run_id = str(uuid.uuid4())
    report_payload = {
        "run_id": run_id,
        "created_at": now_utc_iso(),
        "phase": "conversation",
        "judge": judge,
        "thresholds": thresholds,
        "settings": settings,
        "deepeval_timeout": timeout_config,
        "summary": {
            "cases": len(test_cases),
            "difficulties": _count_by_field(goldens, "difficulty"),
            "domains": _count_by_field(goldens, "domain"),
        },
        "scorecard": scorecard,
        "deepeval_result": deepeval_result,
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


def _resolve_target_metadata(golden: dict) -> dict:
    explicit = golden.get("target_metadata")
    if isinstance(explicit, dict) and explicit:
        return explicit
    return _extract_target_metadata_from_context(golden.get("context"))


def _extract_target_metadata_from_context(context: object) -> dict:
    if not context:
        return {}
    if isinstance(context, str):
        context_items = [context]
    elif isinstance(context, list):
        context_items = [str(item) for item in context if item is not None]
    else:
        context_items = [str(context)]

    metadata: dict[str, int] = {}
    for text in context_items:
        for key in ("postId", "propertyId", "planningDocumentId", "planYear"):
            if key in metadata:
                continue
            match = re.search(rf"\b{re.escape(key)}\s*=\s*(-?\d+)\b", text)
            if match:
                metadata[key] = int(match.group(1))
        if metadata.get("postId") is not None or metadata.get("planningDocumentId") is not None:
            # One exact target is enough to avoid broad conversational retrieval.
            break
    return metadata


def _conversation_trace_from_record(trace: dict) -> ConversationTrace:
    return ConversationTrace(
        conversation_id=str(trace.get("conversation_id", "conv_case")),
        turns=[TurnTrace(**turn) for turn in trace.get("turns", [])],
    )


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
