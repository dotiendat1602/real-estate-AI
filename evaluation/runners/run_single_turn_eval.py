from __future__ import annotations

import argparse
import asyncio
import uuid
import sys
import time
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig

from evaluation.adapters.langchain_adapter import LangChainEvalAdapter, turn_trace_to_dict
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


async def main() -> None:
    args = parse_args()
    eval_root = Path(args.root)

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

    adapter = LangChainEvalAdapter(settings=settings)
    traces = []
    test_cases = []
    trace_generation_profile = []

    for golden in goldens:
        started_at = time.perf_counter()
        trace = await adapter.run_single_turn(
            golden["input"],
            target_metadata=golden.get("target_metadata"),
        )
        generation_seconds = round(time.perf_counter() - started_at, 3)
        context_chars = sum(len(item or "") for item in trace.retrieval_context)
        trace_generation_profile.append(
            {
                "id": golden.get("id"),
                "input_preview": str(golden.get("input", ""))[:120],
                "generation_seconds": generation_seconds,
                "retrieval_context_docs": len(trace.retrieval_context),
                "retrieval_context_chars": context_chars,
            }
        )
        traces.append({"golden": golden, "trace": turn_trace_to_dict(trace)})
        test_cases.append(build_single_turn_test_case(trace, golden))

    # Apply judge provider environment only for scoring to avoid impacting runtime QA model.
    judge_model = resolve_judge_model_and_apply_env(judge)

    metrics = [
        *build_rag_metrics(thresholds, judge_model=judge_model),
    ]
    scoring_started_at = time.perf_counter()
    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=not bool(args.sync),
            throttle_value=max(0.0, float(args.throttle_seconds)),
            max_concurrent=max(1, int(args.max_concurrent)),
        ),
    )
    judge_scoring_seconds = round(time.perf_counter() - scoring_started_at, 3)
    deepeval_result = to_jsonable(eval_result)
    scorecard = build_single_turn_scorecard(
        deepeval_result=deepeval_result,
        thresholds=thresholds,
    )

    trace_generation_seconds = round(
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
        "summary": {
            "cases": len(test_cases),
            "difficulties": _count_by_field(goldens, "difficulty"),
            "domains": _count_by_field(goldens, "domain"),
        },
        "timing": {
            "trace_generation_seconds": trace_generation_seconds,
            "judge_scoring_seconds": judge_scoring_seconds,
            "total_pipeline_seconds": round(trace_generation_seconds + judge_scoring_seconds, 3),
            "trace_generation_by_case": trace_generation_profile,
        },
        "judge_cost_profile": _build_judge_cost_profile(deepeval_result, goldens),
        "scorecard": scorecard,
        "deepeval_result": deepeval_result,
        "traces": traces,
    }

    report_path = eval_root / "reports" / args.output_name
    write_json(report_path, report_payload)

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


def _count_by_field(items: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field, "unknown"))
        out[key] = out.get(key, 0) + 1
    return out


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
