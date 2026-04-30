from __future__ import annotations

import argparse
import asyncio
import json
import uuid
import os
import sys
import time
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience dependency
    load_dotenv = None

from evaluation.adapters.langchain_adapter import LangChainEvalAdapter, TurnTrace, turn_trace_to_dict
from evaluation.adapters.trace_to_testcase import build_single_turn_test_case
from evaluation.metrics.metric_helpers import (
    build_single_turn_scorecard,
    resolve_judge_model_and_apply_env,
)
from evaluation.metrics.rag_metrics import build_rag_metrics
from evaluation.runners.common import (
    apply_deepeval_timeout_env,
    build_markdown_summary,
    load_json,
    load_yaml,
    now_utc_iso,
    to_jsonable,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline single-turn RAG evaluation.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to evaluation folder",
    )
    parser.add_argument(
        "--dataset",
        default="single_turn_goldens_tuned.json",
        help="Dataset filename under evaluation/datasets",
    )
    parser.add_argument(
        "--output-name",
        default="baseline_eval_single_turn.json",
        help="Output report filename under evaluation/reports",
    )
    parser.add_argument(
        "--trace-input",
        default="",
        help="Optional trace checkpoint JSON to score without regenerating RAG traces. Relative paths are resolved under evaluation/reports.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, evaluate only the first N cases for smoke/demo runs",
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
        help="Maximum concurrent RAG trace generation calls before DeepEval scoring",
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
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="If > 0, score test cases in smaller batches and merge the DeepEval results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Override evaluation runtime.top_k for this run.",
    )
    parser.add_argument(
        "--experiment-config",
        default="",
        help="Optional JSON object or JSON file path to include in the report.",
    )
    parser.add_argument(
        "--skip-baseline-summary",
        action="store_true",
        help="Do not write reports/baseline_summary.md. Intended for experiment matrix runs.",
    )
    parser.add_argument(
        "--manual-metric-loop",
        action="store_true",
        help="Score cases by calling each DeepEval metric directly. Useful when deepeval.evaluate hangs in smoke/debug runs.",
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
    goldens = load_json(_resolve_dataset_path(eval_root, args.dataset))
    top_k_override = _resolve_top_k_override(args.top_k)
    if top_k_override is not None:
        settings.setdefault("runtime", {})["top_k"] = top_k_override

    experiment_config = _load_experiment_config_arg(args.experiment_config)

    timeout_config = apply_deepeval_timeout_env(
        judge_config=judge,
        per_attempt_timeout_seconds=float(args.per_attempt_timeout_seconds),
        disable_timeouts=bool(args.disable_deepeval_timeouts),
    )

    if args.max_cases and args.max_cases > 0:
        goldens = goldens[: args.max_cases]

    trace_max_concurrent = max(1, int(args.trace_max_concurrent))
    trace_generation_started_at = time.perf_counter()

    if args.trace_input:
        trace_input_path = Path(args.trace_input)
        if not trace_input_path.is_absolute():
            trace_input_path = eval_root / "reports" / trace_input_path
        checkpoint = load_json(trace_input_path)
        trace_records = checkpoint.get("traces") or []
        if args.max_cases and args.max_cases > 0:
            trace_records = trace_records[: args.max_cases]
        trace_results = []
        for index, record in enumerate(trace_records):
            golden = record["golden"]
            trace = TurnTrace(**record["trace"])
            trace_results.append(
                (
                    index,
                    record,
                    build_single_turn_test_case(trace, golden),
                    (checkpoint.get("timing", {}).get("trace_generation_by_case") or [{}] * len(trace_records))[index],
                )
            )
        goldens = [record["golden"] for record in trace_records]
        trace_generation_wall_seconds = float(checkpoint.get("timing", {}).get("trace_generation_seconds") or 0.0)
        print(f"[Trace] Loaded {len(trace_results)} traces from {trace_input_path}", flush=True)
    else:
        adapter = LangChainEvalAdapter(settings=settings)
        await adapter.initialize()

        trace_semaphore = asyncio.Semaphore(trace_max_concurrent)

        async def build_trace_case(index: int, golden: dict):
            async with trace_semaphore:
                started_at = time.perf_counter()
                trace = await adapter.run_single_turn(
                    golden["input"],
                    target_metadata=golden.get("target_metadata"),
                )
                generation_seconds = round(time.perf_counter() - started_at, 3)
                context_chars = sum(len(item or "") for item in trace.retrieval_context)
                profile = {
                    "id": golden.get("id"),
                    "input_preview": str(golden.get("input", ""))[:120],
                    "generation_seconds": generation_seconds,
                    "retrieval_context_docs": len(trace.retrieval_context),
                    "retrieval_context_chars": context_chars,
                    "retrieval_seconds": trace.metadata.get("retrieval_seconds"),
                    "answer_generation_seconds": trace.metadata.get("answer_generation_seconds"),
                    "retrieval_strategy": trace.metadata.get("retrieval_strategy"),
                    "filter_extraction_skipped": trace.metadata.get("filter_extraction_skipped"),
                }
                print(
                    "[Trace] "
                    f"{index + 1}/{len(goldens)} "
                    f"{golden.get('id') or 'case'} "
                    f"done in {generation_seconds}s "
                    f"retrieval={profile['retrieval_seconds']}s "
                    f"answer={profile['answer_generation_seconds']}s "
                    f"docs={len(trace.retrieval_context)}",
                    flush=True,
                )
                return (
                    index,
                    {"golden": golden, "trace": turn_trace_to_dict(trace)},
                    build_single_turn_test_case(trace, golden),
                    profile,
                )

        if trace_max_concurrent == 1:
            trace_results = []
            for index, golden in enumerate(goldens):
                trace_results.append(await build_trace_case(index, golden))
        else:
            trace_results = await asyncio.gather(
                *(build_trace_case(index, golden) for index, golden in enumerate(goldens))
            )
        trace_generation_wall_seconds = round(time.perf_counter() - trace_generation_started_at, 3)

    trace_results.sort(key=lambda item: item[0])
    traces = [item[1] for item in trace_results]
    test_cases = [item[2] for item in trace_results]
    trace_generation_profile = [item[3] for item in trace_results]
    output_report_relpath = Path(args.output_name)
    trace_checkpoint_path = (eval_root / "reports" / output_report_relpath).with_suffix(".traces.json")
    if not args.trace_input:
        write_json(
            trace_checkpoint_path,
            {
                "created_at": now_utc_iso(),
                "phase": "single_turn_trace_generation",
                "dataset": args.dataset,
                "cases": len(test_cases),
                "trace_max_concurrent": trace_max_concurrent,
                "timing": {
                    "trace_generation_seconds": trace_generation_wall_seconds,
                    "trace_generation_by_case": trace_generation_profile,
                },
                "traces": traces,
            },
        )

    # Apply judge provider environment only for scoring to avoid impacting runtime QA model.
    judge_model = resolve_judge_model_and_apply_env(judge)

    metrics = [
        *build_rag_metrics(thresholds, judge_model=judge_model),
    ]
    scoring_started_at = time.perf_counter()
    eval_batch_size = max(0, int(args.eval_batch_size))
    if args.manual_metric_loop:
        deepeval_result = _evaluate_metrics_manually(test_cases, metrics)
    elif eval_batch_size and len(test_cases) > eval_batch_size:
        deepeval_batches: list[dict] = []
        for batch_start in range(0, len(test_cases), eval_batch_size):
            batch = test_cases[batch_start : batch_start + eval_batch_size]
            deepeval_batches.extend(
                _evaluate_batch_with_fallback(
                    batch,
                    metrics,
                    start_index=batch_start,
                    total_cases=len(test_cases),
                    sync=bool(args.sync),
                    throttle_seconds=max(0.0, float(args.throttle_seconds)),
                    max_concurrent=max(1, int(args.max_concurrent)),
                )
            )
        deepeval_result = _merge_deepeval_results(deepeval_batches)
    else:
        eval_result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(
                run_async=not bool(args.sync),
                throttle_value=max(0.0, float(args.throttle_seconds)),
                max_concurrent=max(1, int(args.max_concurrent)),
            ),
        )
        deepeval_result = to_jsonable(eval_result)
    judge_scoring_seconds = round(time.perf_counter() - scoring_started_at, 3)
    scorecard = build_single_turn_scorecard(
        deepeval_result=deepeval_result,
        thresholds=thresholds,
    )

    trace_generation_worker_seconds_total = round(
        sum(float(item.get("generation_seconds", 0.0)) for item in trace_generation_profile),
        3,
    )

    run_id = str(uuid.uuid4())
    report_payload = {
        "run_id": run_id,
        "created_at": now_utc_iso(),
        "phase": "single_turn",
        "judge": judge,
        "thresholds": thresholds,
        "settings": settings,
        "deepeval_timeout": timeout_config,
        "runner": {
            "trace_max_concurrent": trace_max_concurrent,
            "judge_max_concurrent": max(1, int(args.max_concurrent)),
            "throttle_seconds": max(0.0, float(args.throttle_seconds)),
            "sync": bool(args.sync),
        },
        "experiment_config": experiment_config,
        "summary": {
            "cases": len(test_cases),
            "difficulties": _count_by_field(goldens, "difficulty"),
            "domains": _count_by_field(goldens, "domain"),
        },
        "timing": {
            "trace_generation_seconds": trace_generation_wall_seconds,
            "trace_generation_worker_seconds_total": trace_generation_worker_seconds_total,
            "judge_scoring_seconds": judge_scoring_seconds,
            "total_pipeline_seconds": round(trace_generation_wall_seconds + judge_scoring_seconds, 3),
            "trace_generation_by_case": trace_generation_profile,
        },
        "judge_cost_profile": _build_judge_cost_profile(deepeval_result, goldens),
        "scorecard": scorecard,
        "deepeval_result": deepeval_result,
        "traces": traces,
    }

    report_path = eval_root / "reports" / args.output_name
    write_json(report_path, report_payload)

    if not args.skip_baseline_summary:
        summary_md = build_markdown_summary(
            "Single-turn Baseline Summary",
            [
                f"- Run ID: {run_id}",
                f"- Cases: {len(test_cases)}",
                f"- Dataset: {args.dataset}",
                f"- Difficulties: {report_payload['summary']['difficulties']}",
                f"- Domains: {report_payload['summary']['domains']}",
                "- Formula: Q_single = (w_CRec*S_CRec + w_F*S_F + w_AR*S_AR) / (w_CRec + w_F + w_AR)",
                # "- Formula: Q_single = (w_F*S_F + w_AR*S_AR) / (w_F + w_AR)",
                f"- Q_single: {scorecard.get('overall_quality_score')}",
                f"- Detailed JSON: reports/{args.output_name}",
            ],
        )
        (eval_root / "reports" / "baseline_summary.md").write_text(summary_md, encoding="utf-8")


def _resolve_top_k_override(raw_cli_value: int) -> int | None:
    if raw_cli_value and raw_cli_value > 0:
        return int(raw_cli_value)
    raw_env_value = (os.getenv("EVAL_TOP_K") or "").strip()
    if not raw_env_value:
        return None
    try:
        value = int(raw_env_value)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_dataset_path(eval_root: Path, dataset: str) -> Path:
    candidate = Path(dataset)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    direct_eval_path = eval_root / dataset
    if direct_eval_path.exists():
        return direct_eval_path
    return eval_root / "datasets" / dataset


def _load_experiment_config_arg(raw: str) -> dict:
    value = (raw or "").strip()
    if not value:
        env_value = (os.getenv("EVAL_EXPERIMENT_CONFIG") or "").strip()
        value = env_value
    if not value:
        return {}

    if not value.startswith(("{", "[")):
        candidate_path = Path(value)
        if candidate_path.exists():
            return load_json(candidate_path)

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _evaluate_metrics_manually(test_cases: list, metrics: list) -> dict:
    """Minimal DeepEval-compatible result for stable sequential smoke scoring."""
    test_results: list[dict] = []
    for case_index, test_case in enumerate(test_cases):
        metrics_data: list[dict] = []
        print(
            f"[JudgeManual] {case_index + 1}/{len(test_cases)} {getattr(test_case, 'name', '')} start",
            flush=True,
        )
        for metric in metrics:
            metric_name = _metric_display_name(metric)
            started_at = time.perf_counter()
            try:
                metric.measure(test_case)
                score = getattr(metric, "score", None)
                success = getattr(metric, "success", None)
                reason = getattr(metric, "reason", None)
                error = None
            except Exception as exc:
                score = None
                success = False
                reason = None
                error = f"{type(exc).__name__}: {exc}"

            duration = round(time.perf_counter() - started_at, 3)
            print(
                f"[JudgeManual] {case_index + 1}/{len(test_cases)} "
                f"{metric_name} score={score} success={success} time={duration}s",
                flush=True,
            )
            metric_payload = {
                "name": metric_name,
                "threshold": getattr(metric, "threshold", None),
                "score": score,
                "success": success,
                "reason": reason,
                "evaluation_cost": getattr(metric, "evaluation_cost", None),
                "verbose_logs": getattr(metric, "verbose_logs", None),
                "evaluation_model": str(getattr(metric, "evaluation_model", None) or ""),
                "duration_seconds": duration,
            }
            if error:
                metric_payload["error"] = error
            metrics_data.append(metric_payload)

        test_results.append(
            {
                "name": getattr(test_case, "name", ""),
                "success": all(item.get("success") for item in metrics_data),
                "metrics_data": metrics_data,
            }
        )

    return {
        "test_results": test_results,
        "confident_link": None,
        "run_duration": None,
    }


def _metric_display_name(metric) -> str:
    raw_name = getattr(metric, "name", None)
    if raw_name:
        return str(raw_name)

    class_name = metric.__class__.__name__
    if class_name == "ContextualRecallMetric":
        return "Contextual Recall"
    if class_name == "FaithfulnessMetric":
        return "Faithfulness"
    if class_name == "AnswerRelevancyMetric":
        return "Answer Relevancy"
    return class_name


def _count_by_field(items: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field, "unknown"))
        out[key] = out.get(key, 0) + 1
    return out


def _merge_deepeval_results(results: list[dict]) -> dict:
    merged = {
        "test_results": [],
        "confident_link": None,
        "test_run_id": None,
    }
    for result in results:
        merged["test_results"].extend(result.get("test_results") or [])
        if merged["confident_link"] is None and result.get("confident_link"):
            merged["confident_link"] = result.get("confident_link")
        if merged["test_run_id"] is None and result.get("test_run_id"):
            merged["test_run_id"] = result.get("test_run_id")
    return merged


def _evaluate_batch_with_fallback(
    test_cases: list,
    metrics: list,
    *,
    start_index: int,
    total_cases: int,
    sync: bool,
    throttle_seconds: float,
    max_concurrent: int,
) -> list[dict]:
    batch_end = start_index + len(test_cases)
    print(
        f"[EvalBatch] scoring cases {start_index + 1}-{batch_end}/{total_cases}",
        flush=True,
    )
    try:
        batch_result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(
                run_async=not sync,
                throttle_value=throttle_seconds,
                max_concurrent=max_concurrent,
            ),
        )
        return [to_jsonable(batch_result)]
    except BaseException as exc:
        print(
            f"[EvalBatch] cases {start_index + 1}-{batch_end} failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if len(test_cases) <= 1:
            raise

        midpoint = len(test_cases) // 2
        return [
            *_evaluate_batch_with_fallback(
                test_cases[:midpoint],
                metrics,
                start_index=start_index,
                total_cases=total_cases,
                sync=True,
                throttle_seconds=max(throttle_seconds, 1.0),
                max_concurrent=1,
            ),
            *_evaluate_batch_with_fallback(
                test_cases[midpoint:],
                metrics,
                start_index=start_index + midpoint,
                total_cases=total_cases,
                sync=True,
                throttle_seconds=max(throttle_seconds, 1.0),
                max_concurrent=1,
            ),
        ]


def _build_judge_cost_profile(
    deepeval_result: dict,
    goldens: list[dict],
) -> list[dict[str, float | str | None]]:
    results = deepeval_result.get("test_results", []) if isinstance(deepeval_result, dict) else []
    profile: list[dict[str, float | str | None]] = []

    for index, item in enumerate(results):
        if not isinstance(item, dict):
            continue

        metrics_data = item.get("metrics_data", [])
        total_cost = 0.0
        if isinstance(metrics_data, list):
            for metric in metrics_data:
                if not isinstance(metric, dict):
                    continue
                cost = metric.get("evaluation_cost")
                if isinstance(cost, (int, float)):
                    total_cost += float(cost)

        golden = goldens[index] if index < len(goldens) else {}
        profile.append(
            {
                "id": str(golden.get("id") or item.get("name") or f"case_{index}"),
                "input_preview": str(golden.get("input") or item.get("input") or "")[:120],
                "total_evaluation_cost": round(total_cost, 8),
            }
        )

    return profile


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
