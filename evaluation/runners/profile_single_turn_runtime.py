from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience dependency
    load_dotenv = None

from evaluation.runners.common import load_json, load_yaml, now_utc_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile single-turn RAG runtime and answer token usage without DeepEval scoring.",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to evaluation folder",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset filename under evaluation/datasets, or absolute/relative JSON path.",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Output report filename under evaluation/reports.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, profile only the first N cases.",
    )
    parser.add_argument(
        "--trace-max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent RAG calls.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Override evaluation runtime.top_k for this profile run.",
    )
    parser.add_argument(
        "--use-target-metadata",
        action="store_true",
        help="Pass golden target_metadata into retrieval. Default is disabled to measure natural retrieval.",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include full traces in the report. Default stores compact per-case profile only.",
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

    from evaluation.adapters.langchain_adapter import LangChainEvalAdapter, turn_trace_to_dict

    settings = load_yaml(eval_root / "config" / "eval_settings.yaml")
    top_k_override = _resolve_top_k_override(args.top_k)
    if top_k_override is not None:
        settings.setdefault("runtime", {})["top_k"] = top_k_override

    goldens = load_json(_resolve_dataset_path(eval_root, args.dataset))
    if args.max_cases and args.max_cases > 0:
        goldens = goldens[: args.max_cases]

    adapter = LangChainEvalAdapter(settings=settings)
    await adapter.initialize()

    trace_max_concurrent = max(1, int(args.trace_max_concurrent))
    semaphore = asyncio.Semaphore(trace_max_concurrent)
    started_at = time.perf_counter()

    async def profile_case(index: int, golden: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            case_started_at = time.perf_counter()
            original_target_metadata = golden.get("target_metadata") or {}
            target_metadata = original_target_metadata if args.use_target_metadata else {}
            trace = await adapter.run_single_turn(
                golden["input"],
                target_metadata=target_metadata,
            )
            generation_seconds = round(time.perf_counter() - case_started_at, 3)
            metadata = trace.metadata or {}
            filter_usage = metadata.get("filter_extraction_token_usage") or {}
            answer_usage = metadata.get("answer_token_usage") or {}
            total_usage = metadata.get("total_token_usage") or {}
            runtime_seconds = metadata.get("runtime_seconds")
            if not isinstance(runtime_seconds, (int, float)):
                runtime_seconds = _sum_numbers(
                    metadata.get("filter_extraction_seconds"),
                    metadata.get("retrieval_seconds"),
                    metadata.get("answer_generation_seconds"),
                )
            record = {
                "id": golden.get("id"),
                "difficulty": golden.get("difficulty"),
                "domain": golden.get("domain"),
                "input_preview": str(golden.get("input", ""))[:160],
                "generation_seconds": generation_seconds,
                "filter_extraction_seconds": metadata.get("filter_extraction_seconds"),
                "retrieval_seconds": metadata.get("retrieval_seconds"),
                "answer_generation_seconds": metadata.get("answer_generation_seconds"),
                "runtime_seconds": runtime_seconds,
                "retrieval_strategy": metadata.get("retrieval_strategy"),
                "filter_extraction_skipped": metadata.get("filter_extraction_skipped"),
                "filter_extraction_skip_reason": metadata.get("filter_extraction_skip_reason"),
                "raw_retrieved_docs_count": metadata.get("raw_retrieved_docs_count"),
                "context_docs_count": metadata.get("context_docs_count"),
                "context_chars": metadata.get("context_chars"),
                "filter_extraction_token_usage": filter_usage,
                "answer_token_usage": answer_usage,
                "total_token_usage": total_usage,
                "target_metadata_present": bool(original_target_metadata),
                "target_metadata_eval_only": _target_metadata_eval_only(original_target_metadata),
                "target_metadata_used": bool(target_metadata),
            }
            if args.include_traces:
                record["trace"] = turn_trace_to_dict(trace)

            print(
                "[Profile] "
                f"{index + 1}/{len(goldens)} "
                f"{golden.get('id') or 'case'} "
                f"runtime={record['runtime_seconds']}s "
                f"filter={record['filter_extraction_seconds']}s "
                f"retrieval={record['retrieval_seconds']}s "
                f"answer={record['answer_generation_seconds']}s "
                f"tokens={total_usage.get('total_tokens')}",
                flush=True,
            )
            return record

    if trace_max_concurrent == 1:
        records = []
        for index, golden in enumerate(goldens):
            records.append(await profile_case(index, golden))
    else:
        records = await asyncio.gather(
            *(profile_case(index, golden) for index, golden in enumerate(goldens))
        )

    wall_seconds = round(time.perf_counter() - started_at, 3)
    report = {
        "created_at": now_utc_iso(),
        "phase": "single_turn_runtime_profile",
        "dataset": args.dataset,
        "cases": len(records),
        "profile_mode": {
            "target_metadata_enabled": bool(args.use_target_metadata),
            "top_k": settings.get("runtime", {}).get("top_k"),
            "trace_max_concurrent": trace_max_concurrent,
            "llm_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "embedding_model": os.getenv("EMBED_MODEL"),
            "planning_collection": os.getenv("PGVECTOR_COLLECTION_PLANNING"),
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "worker_seconds_total": round(sum(_number(item.get("generation_seconds")) for item in records), 3),
        },
        "summary": _summarize_records(records),
        "records": records,
    }

    output_path = eval_root / "reports" / args.output_name
    write_json(output_path, report)
    print(f"[Profile] Wrote {output_path}", flush=True)


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


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "difficulties": _count_by_field(records, "difficulty"),
        "domains": _count_by_field(records, "domain"),
        "target_metadata_present_count": sum(1 for item in records if item.get("target_metadata_present")),
        "target_metadata_eval_only_count": sum(1 for item in records if item.get("target_metadata_eval_only")),
        "target_metadata_used_count": sum(1 for item in records if item.get("target_metadata_used")),
        "filter_extraction_skipped_count": sum(1 for item in records if item.get("filter_extraction_skipped")),
        "averages": {
            "generation_seconds": _avg(records, "generation_seconds"),
            "filter_extraction_seconds": _avg(records, "filter_extraction_seconds"),
            "retrieval_seconds": _avg(records, "retrieval_seconds"),
            "answer_generation_seconds": _avg(records, "answer_generation_seconds"),
            "runtime_seconds": _avg(records, "runtime_seconds"),
            "context_docs_count": _avg(records, "context_docs_count"),
            "context_chars": _avg(records, "context_chars"),
            "filter_input_tokens": _avg_token(records, "filter_extraction_token_usage", "input_tokens"),
            "filter_output_tokens": _avg_token(records, "filter_extraction_token_usage", "output_tokens"),
            "filter_total_tokens": _avg_token(records, "filter_extraction_token_usage", "total_tokens"),
            "answer_input_tokens": _avg_token(records, "input_tokens"),
            "answer_output_tokens": _avg_token(records, "output_tokens"),
            "answer_total_tokens": _avg_token(records, "total_tokens"),
            "total_input_tokens": _avg_token(records, "total_token_usage", "input_tokens"),
            "total_output_tokens": _avg_token(records, "total_token_usage", "output_tokens"),
            "total_tokens": _avg_token(records, "total_token_usage", "total_tokens"),
        },
        "token_totals": {
            "filter_input_tokens": _sum_token(records, "filter_extraction_token_usage", "input_tokens"),
            "filter_output_tokens": _sum_token(records, "filter_extraction_token_usage", "output_tokens"),
            "filter_total_tokens": _sum_token(records, "filter_extraction_token_usage", "total_tokens"),
            "answer_input_tokens": _sum_token(records, "input_tokens"),
            "answer_output_tokens": _sum_token(records, "output_tokens"),
            "answer_total_tokens": _sum_token(records, "total_tokens"),
            "total_input_tokens": _sum_token(records, "total_token_usage", "input_tokens"),
            "total_output_tokens": _sum_token(records, "total_token_usage", "output_tokens"),
            "total_tokens": _sum_token(records, "total_token_usage", "total_tokens"),
        },
        "by_domain": _summarize_by_field(records, "domain"),
    }


def _summarize_by_field(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        key = str(item.get(field) or "unknown")
        grouped.setdefault(key, []).append(item)
    return {
        key: {
            "cases": len(items),
            "runtime_seconds_avg": _avg(items, "runtime_seconds"),
            "retrieval_seconds_avg": _avg(items, "retrieval_seconds"),
            "answer_total_tokens_avg": _avg_token(items, "answer_token_usage", "total_tokens"),
            "total_tokens_avg": _avg_token(items, "total_token_usage", "total_tokens"),
        }
        for key, items in sorted(grouped.items())
    }


def _count_by_field(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field) or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _avg(records: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(item.get(key)) for item in records if item.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _avg_token(
    records: list[dict[str, Any]],
    usage_key: str,
    token_key: str | None = None,
) -> float | None:
    if token_key is None:
        token_key = usage_key
        usage_key = "answer_token_usage"
    values = [
        _number((item.get(usage_key) or {}).get(token_key))
        for item in records
        if isinstance(item.get(usage_key), dict)
        and (item.get(usage_key) or {}).get(token_key) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def _sum_token(
    records: list[dict[str, Any]],
    usage_key: str,
    token_key: str | None = None,
) -> int:
    if token_key is None:
        token_key = usage_key
        usage_key = "answer_token_usage"
    return int(
        sum(
            _number((item.get(usage_key) or {}).get(token_key))
            for item in records
            if isinstance(item.get(usage_key), dict)
        )
    )


def _sum_numbers(*values: Any) -> float:
    return round(sum(_number(value) for value in values), 3)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _target_metadata_eval_only(target_metadata: dict[str, Any]) -> bool:
    value = (target_metadata or {}).get("evalOnly")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
