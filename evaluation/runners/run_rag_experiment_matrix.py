from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from functools import cmp_to_key
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from langchain_core.documents import Document
from sqlalchemy import text

from evaluation.runners.common import load_json, load_yaml, now_utc_iso, write_json


METRIC_KEYS = {
    "CR": "contextual_recall",
    "Faithfulness": "faithfulness",
    "AR": "answer_relevancy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG embedding/top-k/planning chunking experiment matrix.")
    parser.add_argument("--config", default="evaluation/config/rag_experiments.yaml")
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--smoke-summary", default="")
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--smoke-max-cases", type=int, default=0)
    parser.add_argument(
        "--skip-vector-schema-upgrade",
        action="store_true",
        help="Do not alter langchain_pg_embedding.embedding from vector(N) to dimensionless vector.",
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    lowered = (value or "").lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "value"


def _load_config(path: str, eval_root: Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = eval_root.parent / candidate
    return load_yaml(candidate)


def _model_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for raw in config.get("embedding_models") or []:
        if isinstance(raw, str):
            items.append({"name": raw, "slug": _slug(raw)})
        elif isinstance(raw, dict) and raw.get("name"):
            item = dict(raw)
            item.setdefault("slug", _slug(str(item["name"])))
            items.append(item)
    return items


def _matrix_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = _model_items(config)
    top_ks = [int(item) for item in config.get("top_k") or [16]]
    chunk_modes = [str(item) for item in config.get("planning_chunking_modes") or ["planning_baseline_fixed"]]
    out = []
    for model, chunk_mode, top_k in itertools.product(models, chunk_modes, top_ks):
        model_slug = str(model["slug"])
        config_id = f"{model_slug}__{chunk_mode}__k{top_k}"
        out.append(
            {
                "config_id": config_id,
                "embedding_model": str(model["name"]),
                "embedding_model_slug": model_slug,
                "planning_chunking_mode": chunk_mode,
                "top_k": top_k,
                "post_collection": f"post_embeddings__{model_slug}",
                "planning_collection": f"planning_documents__{model_slug}__{chunk_mode}",
            }
        )
    return out


def _select_stage_configs(
    all_configs: list[dict[str, Any]],
    config: dict[str, Any],
    stage: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected_configs = list(all_configs)
    raw_filter_ids = (os.getenv("MATRIX_CONFIG_IDS") or "").strip()
    if raw_filter_ids:
        allowed_ids = {
            item.strip()
            for item in raw_filter_ids.split(",")
            if item.strip()
        }
        selected_configs = [
            item for item in selected_configs if item["config_id"] in allowed_ids
        ]

    if stage == "smoke":
        return selected_configs

    full_cfg = config.get("full") or {}
    explicit_ids = set(str(item) for item in full_cfg.get("config_ids") or [])
    if explicit_ids:
        return [item for item in selected_configs if item["config_id"] in explicit_ids]

    summary_path = args.smoke_summary or full_cfg.get("smoke_summary")
    if not summary_path:
        return selected_configs

    candidate = Path(str(summary_path))
    if not candidate.is_absolute():
        candidate = Path(args.root) / "reports" / str(summary_path)
    if not candidate.exists():
        return selected_configs

    rows = _read_summary_csv(candidate)
    top_n = int(full_cfg.get("top_n", 8))
    selected_ids = [row["config_id"] for row in _rank_rows_for_winner(rows, top_n=top_n)]
    selected = [item for item in selected_configs if item["config_id"] in selected_ids]
    return selected or selected_configs


def _read_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _stage_dataset(
    eval_root: Path,
    config: dict[str, Any],
    stage: str,
    experiment_dir: Path,
    *,
    smoke_max_cases: int = 0,
) -> Path:
    dataset_name = str(config.get("dataset", "single_turn_goldens_medium_tuned.json"))
    dataset_path = eval_root / "datasets" / dataset_name
    goldens = load_json(dataset_path)
    if stage == "smoke":
        smoke_cfg = config.get("smoke") or {}
        max_cases = int(smoke_max_cases or smoke_cfg.get("max_cases", 20))
        goldens = _balanced_subset(goldens, max_cases=max_cases, field="domain")

    out = experiment_dir / "datasets" / f"{stage}_{dataset_name}"
    write_json(out, goldens)
    return out


def _balanced_subset(items: list[dict[str, Any]], max_cases: int, field: str) -> list[dict[str, Any]]:
    if max_cases <= 0 or len(items) <= max_cases:
        return items

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get(field, "unknown")), []).append(item)

    if not groups:
        return items[:max_cases]

    selected: list[dict[str, Any]] = []
    group_keys = list(groups.keys())
    cursor_by_key = {key: 0 for key in group_keys}
    while len(selected) < max_cases:
        made_progress = False
        for key in group_keys:
            cursor = cursor_by_key[key]
            group_items = groups[key]
            if cursor >= len(group_items):
                continue
            selected.append(group_items[cursor])
            cursor_by_key[key] = cursor + 1
            made_progress = True
            if len(selected) >= max_cases:
                break
        if not made_progress:
            break

    return selected


async def _collection_count(collection_name: str) -> int:
    from app.db.pgvector import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :collection_name
                """
            ),
            {"collection_name": collection_name},
        )
        return int(result.scalar_one() or 0)


async def _clear_collection(collection_name: str) -> None:
    from app.db.pgvector import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await session.execute(
            text(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
                """
            ),
            {"collection_name": collection_name},
        )
        await session.commit()


async def _ensure_dimensionless_embedding_column() -> None:
    """Allow evaluation collections with different embedding dimensions.

    LangChain PGVector stores all collections in langchain_pg_embedding. A
    fixed vector(N) column blocks comparing models with different dimensions
    (for example e5-small 384 vs e5-base 768), even when collections are
    isolated by name.
    """
    from app.db.pgvector import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                SELECT a.atttypmod, format_type(a.atttypid, a.atttypmod) AS formatted_type
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = 'langchain_pg_embedding'
                  AND a.attname = 'embedding'
                """
            )
        )
        row = result.mappings().first()
        if not row:
            return
        if int(row["atttypmod"] or -1) < 0:
            print(f"[Experiment] embedding column already supports mixed dimensions: {row['formatted_type']}", flush=True)
            return

        print(
            f"[Experiment] upgrading langchain_pg_embedding.embedding "
            f"from {row['formatted_type']} to dimensionless vector",
            flush=True,
        )
        await session.execute(
            text(
                """
                ALTER TABLE langchain_pg_embedding
                ALTER COLUMN embedding TYPE vector
                USING embedding::vector
                """
            )
        )
        await session.commit()


def _jsonl_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


async def _ingest_posts_manifest(path: Path, collection_name: str) -> int:
    from app.rag.embedder import build_embeddings
    from app.rag.retriever import build_pgvector_store
    from app.utils.chunking import build_splitter

    items = _jsonl_items(path)
    if not items:
        return 0

    splitter = build_splitter(
        chunk_size=int(os.getenv("LISTING_CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("LISTING_CHUNK_OVERLAP", "120")),
    )
    docs: list[Document] = []
    for item in items:
        post_id = item.get("postId")
        content = str(item.get("content") or "").strip()
        if post_id is None or not content:
            continue
        for idx, chunk in enumerate(splitter.split_text(content)):
            md = dict(item.get("metadata") or {})
            md.update({"postId": post_id, "chunkIndex": idx})
            docs.append(Document(page_content=chunk, metadata=md))

    if not docs:
        return 0

    vs = build_pgvector_store(build_embeddings(), collection_name=collection_name)
    await vs.__apost_init__()
    total = 0
    for start in range(0, len(docs), 64):
        ids = await vs.aadd_documents(docs[start : start + 64])
        total += len(ids)
    return total


async def _ingest_planning_manifest(path: Path, collection_name: str) -> int:
    from app.planning.ingestion import PlanningIngestPayload, build_planning_documents
    from app.rag.embedder import build_embeddings
    from app.rag.retriever import build_pgvector_store

    items = _jsonl_items(path)
    if not items:
        return 0

    vs = build_pgvector_store(build_embeddings(), collection_name=collection_name)
    await vs.__apost_init__()
    total = 0
    for item in items:
        payload = PlanningIngestPayload(
            planning_document_id=int(item["planningDocumentId"]),
            title=str(item.get("title") or "Tai lieu quy hoach"),
            source_url=str(item["sourceUrl"]),
            format=item.get("format"),
            document_type=item.get("documentType"),
            dossier_code=item.get("dossierCode"),
            city=item.get("city"),
            district=item.get("district"),
            plan_year=item.get("planYear"),
            property_id=item.get("propertyId"),
            raw_meta=item.get("rawMeta") or {},
        )
        docs, _ = await build_planning_documents(payload)
        for start in range(0, len(docs), 32):
            ids = await vs.aadd_documents(docs[start : start + 32])
            total += len(ids)
    return total


async def _ensure_ingested(
    run_config: dict[str, Any],
    corpus_cfg: dict[str, Any],
    *,
    force_posts: bool,
    force_planning: bool,
    skip: bool,
) -> dict[str, int]:
    post_count = await _collection_count(run_config["post_collection"])
    planning_count = await _collection_count(run_config["planning_collection"])
    if skip:
        return {"post_collection_count": post_count, "planning_collection_count": planning_count}

    posts_path = Path(str(corpus_cfg.get("posts_manifest", "evaluation/corpus/posts.full.jsonl")))
    planning_path = Path(str(corpus_cfg.get("planning_manifest", "evaluation/corpus/planning_documents.full.jsonl")))
    if not posts_path.is_absolute():
        posts_path = Path.cwd() / posts_path
    if not planning_path.is_absolute():
        planning_path = Path.cwd() / planning_path

    if force_posts and post_count:
        await _clear_collection(run_config["post_collection"])
        post_count = 0
    if force_planning and planning_count:
        await _clear_collection(run_config["planning_collection"])
        planning_count = 0

    if post_count == 0:
        if not posts_path.exists():
            raise RuntimeError(f"Post collection is empty and manifest is missing: {posts_path}")
        post_count = await _ingest_posts_manifest(posts_path, run_config["post_collection"])

    if planning_count == 0:
        if not planning_path.exists():
            raise RuntimeError(f"Planning collection is empty and manifest is missing: {planning_path}")
        planning_count = await _ingest_planning_manifest(planning_path, run_config["planning_collection"])

    return {"post_collection_count": post_count, "planning_collection_count": planning_count}


def _run_single_turn_eval(
    run_config: dict[str, Any],
    dataset_path: Path,
    report_name: str,
    eval_root: Path,
    config: dict[str, Any],
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "EMBED_MODEL": run_config["embedding_model"],
            "PGVECTOR_COLLECTION": run_config["post_collection"],
            "PGVECTOR_COLLECTION_PLANNING": run_config["planning_collection"],
            "PLANNING_CHUNKING_MODE": run_config["planning_chunking_mode"],
            "EVAL_TOP_K": str(run_config["top_k"]),
        }
    )
    if run_config["embedding_model"].startswith("intfloat/"):
        env.setdefault("EMBED_QUERY_PREFIX", "query: ")
        env.setdefault("EMBED_DOCUMENT_PREFIX", "passage: ")

    runner_cfg = config.get("runner") or {}
    cmd = [
        sys.executable,
        "-m",
        "evaluation.runners.run_single_turn_eval",
        "--root",
        str(eval_root),
        "--dataset",
        str(dataset_path),
        "--output-name",
        report_name,
        "--top-k",
        str(run_config["top_k"]),
        "--trace-max-concurrent",
        str(runner_cfg.get("trace_max_concurrent", 1)),
        "--max-concurrent",
        str(runner_cfg.get("judge_max_concurrent", 2)),
        "--throttle-seconds",
        str(runner_cfg.get("throttle_seconds", 0.6)),
        "--eval-batch-size",
        str(runner_cfg.get("eval_batch_size", 0)),
        "--experiment-config",
        json.dumps(run_config, ensure_ascii=False),
        "--skip-baseline-summary",
    ]
    if runner_cfg.get("sync"):
        cmd.append("--sync")
    if runner_cfg.get("manual_metric_loop"):
        cmd.append("--manual-metric-loop")

    subprocess.run(cmd, cwd=str(eval_root.parent), env=env, check=True)


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _report_row(report_path: Path, run_config: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    report = load_json(report_path)
    scorecard = report.get("scorecard") or {}
    metric_stats = scorecard.get("metric_statistics") or {}
    timing_by_case = (report.get("timing") or {}).get("trace_generation_by_case") or []
    traces = report.get("traces") or []

    def stat(metric_key: str, field: str) -> Any:
        data = metric_stats.get(metric_key) or {}
        return data.get(field)

    retrieval_values = [
        float(item["retrieval_seconds"])
        for item in timing_by_case
        if isinstance(item, dict) and isinstance(item.get("retrieval_seconds"), (int, float))
    ]
    answer_values = [
        float(item["answer_generation_seconds"])
        for item in timing_by_case
        if isinstance(item, dict) and isinstance(item.get("answer_generation_seconds"), (int, float))
    ]
    eval_context_doc_values: list[float] = []
    eval_context_char_values: list[float] = []
    full_context_char_values: list[float] = []
    for record in traces:
        if not isinstance(record, dict):
            continue
        trace = record.get("trace") or {}
        metadata = trace.get("metadata") or {}
        if isinstance(metadata.get("eval_context_docs_count"), (int, float)):
            eval_context_doc_values.append(float(metadata["eval_context_docs_count"]))
        if isinstance(metadata.get("eval_context_chars"), (int, float)):
            eval_context_char_values.append(float(metadata["eval_context_chars"]))
        if isinstance(metadata.get("context_chars"), (int, float)):
            full_context_char_values.append(float(metadata["context_chars"]))

    retrieval_avg = round(sum(retrieval_values) / len(retrieval_values), 6) if retrieval_values else None
    answer_avg = round(sum(answer_values) / len(answer_values), 6) if answer_values else None
    eval_context_docs_avg = (
        round(sum(eval_context_doc_values) / len(eval_context_doc_values), 6)
        if eval_context_doc_values
        else None
    )
    eval_context_chars_avg = (
        round(sum(eval_context_char_values) / len(eval_context_char_values), 6)
        if eval_context_char_values
        else None
    )
    context_chars_avg = (
        round(sum(full_context_char_values) / len(full_context_char_values), 6)
        if full_context_char_values
        else None
    )
    runtime_seconds_avg = (
        round((retrieval_avg or 0.0) + (answer_avg or 0.0), 6)
        if retrieval_avg is not None or answer_avg is not None
        else None
    )
    runtime_cost_proxy = _cost_proxy(
        top_k=int(run_config["top_k"]),
        context_chars_avg=context_chars_avg,
        eval_context_chars_avg=eval_context_chars_avg,
        planning_chunk_count=int(counts.get("planning_collection_count", 0) or 0),
        runtime_seconds_avg=runtime_seconds_avg,
    )

    return {
        "status": "ok",
        "error": "",
        "config_id": run_config["config_id"],
        "embedding_model": run_config["embedding_model"],
        "planning_chunking_mode": run_config["planning_chunking_mode"],
        "top_k": run_config["top_k"],
        "CR_mean": stat("contextual_recall", "mean_score"),
        "CR_pass_rate": stat("contextual_recall", "pass_rate"),
        "Faithfulness_mean": stat("faithfulness", "mean_score"),
        "Faithfulness_pass_rate": stat("faithfulness", "pass_rate"),
        "AR_mean": stat("answer_relevancy", "mean_score"),
        "AR_pass_rate": stat("answer_relevancy", "pass_rate"),
        "retrieval_seconds_avg": retrieval_avg,
        "answer_seconds_avg": answer_avg,
        "runtime_seconds_avg": runtime_seconds_avg,
        "eval_context_docs_avg": eval_context_docs_avg,
        "eval_context_chars_avg": eval_context_chars_avg,
        "context_chars_avg": context_chars_avg,
        "planning_chunk_count": counts.get("planning_collection_count", 0),
        "post_chunk_count": counts.get("post_collection_count", 0),
        "runtime_cost_proxy": runtime_cost_proxy,
        "report_path": str(report_path),
    }


def _cost_proxy(
    *,
    top_k: int,
    context_chars_avg: float | None,
    eval_context_chars_avg: float | None,
    planning_chunk_count: int,
    runtime_seconds_avg: float | None,
) -> float:
    context_chars = context_chars_avg if context_chars_avg is not None else eval_context_chars_avg
    # Lower is better. The units are intentionally rough: retrieval/generation latency,
    # context-token pressure, embedding storage pressure, and top-k context breadth.
    return round(
        (runtime_seconds_avg or 0.0)
        + ((context_chars or 0.0) / 4000.0)
        + (planning_chunk_count / 100000.0)
        + (top_k / 100.0),
        6,
    )


def _rank_rows_for_winner(rows: list[dict[str, Any]], top_n: int | None = None) -> list[dict[str, Any]]:
    rows = [row for row in rows if str(row.get("status") or "ok") != "failed"]
    if not rows:
        return []

    baseline = next(
        (
            row
            for row in rows
            if "multilingual_e5_small" in str(row.get("config_id", ""))
            and str(row.get("planning_chunking_mode")) == "planning_baseline_fixed"
            and str(row.get("top_k")) == "16"
        ),
        None,
    )
    baseline_f = _metric_value(baseline or {}, "Faithfulness_mean")
    baseline_ar = _metric_value(baseline or {}, "AR_mean")

    eligible = []
    for row in rows:
        faith = _metric_value(row, "Faithfulness_mean")
        ar = _metric_value(row, "AR_mean")
        if baseline_f is not None and faith is not None and faith < baseline_f - 0.02:
            continue
        if baseline_ar is not None and ar is not None and ar < baseline_ar - 0.02:
            continue
        eligible.append(row)
    eligible = eligible or rows

    best_cr = max((_metric_value(row, "CR_mean") or -1.0) for row in eligible)
    quality_tier = [
        row
        for row in eligible
        if (_metric_value(row, "CR_mean") or -1.0) >= best_cr - 0.01
    ]
    if not quality_tier:
        quality_tier = eligible

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        left_cr = _metric_value(left, "CR_mean") or -1.0
        right_cr = _metric_value(right, "CR_mean") or -1.0
        if abs(left_cr - right_cr) > 0.005:
            return -1 if left_cr > right_cr else 1

        left_faith = _metric_value(left, "Faithfulness_mean") or -1.0
        right_faith = _metric_value(right, "Faithfulness_mean") or -1.0
        if abs(left_faith - right_faith) > 0.005:
            return -1 if left_faith > right_faith else 1

        left_ar = _metric_value(left, "AR_mean") or -1.0
        right_ar = _metric_value(right, "AR_mean") or -1.0
        if abs(left_ar - right_ar) > 0.005:
            return -1 if left_ar > right_ar else 1

        left_cost = _metric_value(left, "runtime_cost_proxy") or 1e9
        right_cost = _metric_value(right, "runtime_cost_proxy") or 1e9
        if abs(left_cost - right_cost) > 0.000001:
            return -1 if left_cost < right_cost else 1

        left_latency = _metric_value(left, "runtime_seconds_avg") or 1e9
        right_latency = _metric_value(right, "runtime_seconds_avg") or 1e9
        if abs(left_latency - right_latency) > 0.000001:
            return -1 if left_latency < right_latency else 1

        left_top_k = int(float(left.get("top_k") or 9999))
        right_top_k = int(float(right.get("top_k") or 9999))
        if left_top_k != right_top_k:
            return -1 if left_top_k < right_top_k else 1
        return 0

    ranked = sorted(quality_tier, key=cmp_to_key(compare))
    return ranked[:top_n] if top_n else ranked


def _write_summary_outputs(rows: list[dict[str, Any]], experiment_dir: Path) -> None:
    if not rows:
        return

    summary_csv = experiment_dir / "summary.csv"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def table_for(metric: str) -> str:
        mean_key = f"{metric}_mean"
        ordered = sorted(
            rows,
            key=lambda row: _metric_value(row, mean_key) if _metric_value(row, mean_key) is not None else -1.0,
            reverse=True,
        )[:10]
        lines = [
            f"## Top by {metric}",
            "",
            "| Rank | Config | Mean | Pass Rate | Top-K | Cost Proxy | Runtime Avg | Chunking |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for index, row in enumerate(ordered, start=1):
            lines.append(
                "| "
                f"{index} | {row['config_id']} | {row.get(mean_key)} | "
                f"{row.get(metric + '_pass_rate')} | {row.get('top_k')} | "
                f"{row.get('runtime_cost_proxy')} | {row.get('runtime_seconds_avg')} | "
                f"{row.get('planning_chunking_mode')} |"
            )
        return "\n".join(lines)

    summary_md = "\n\n".join(
        [
            "# RAG Experiment Summary",
            table_for("CR"),
            table_for("Faithfulness"),
            table_for("AR"),
        ]
    )
    (experiment_dir / "summary.md").write_text(summary_md + "\n", encoding="utf-8")

    winner_rows = _rank_rows_for_winner(rows, top_n=1)
    write_json(experiment_dir / "winner.json", winner_rows[0] if winner_rows else {})


def _failed_row(run_config: dict[str, Any], counts: dict[str, int], error: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": error[:1200],
        "config_id": run_config["config_id"],
        "embedding_model": run_config["embedding_model"],
        "planning_chunking_mode": run_config["planning_chunking_mode"],
        "top_k": run_config["top_k"],
        "CR_mean": None,
        "CR_pass_rate": None,
        "Faithfulness_mean": None,
        "Faithfulness_pass_rate": None,
        "AR_mean": None,
        "AR_pass_rate": None,
        "retrieval_seconds_avg": None,
        "answer_seconds_avg": None,
        "runtime_seconds_avg": None,
        "eval_context_docs_avg": None,
        "eval_context_chars_avg": None,
        "context_chars_avg": None,
        "planning_chunk_count": counts.get("planning_collection_count", 0),
        "post_chunk_count": counts.get("post_collection_count", 0),
        "runtime_cost_proxy": None,
        "report_path": "",
    }


async def main() -> None:
    args = parse_args()
    eval_root = Path(args.root).resolve()
    service_root = eval_root.parent
    if load_dotenv is not None:
        load_dotenv(service_root / ".env")

    os.chdir(service_root)
    config = _load_config(args.config, eval_root)
    if not args.skip_vector_schema_upgrade:
        await _ensure_dimensionless_embedding_column()

    experiment_id = args.experiment_id or f"{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = eval_root / "reports" / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = _stage_dataset(
        eval_root,
        config,
        args.stage,
        experiment_dir,
        smoke_max_cases=max(0, int(args.smoke_max_cases)),
    )
    all_configs = _matrix_configs(config)
    selected_configs = _select_stage_configs(all_configs, config, args.stage, args)
    if args.limit_configs and args.limit_configs > 0:
        selected_configs = selected_configs[: int(args.limit_configs)]
    summary_path = experiment_dir / "summary.csv"
    rows: list[dict[str, Any]] = _read_summary_csv(summary_path) if summary_path.exists() else []
    completed_ids = {
        str(row.get("config_id") or "").strip()
        for row in rows
        if str(row.get("config_id") or "").strip()
    }
    selected_configs = [
        item for item in selected_configs if item["config_id"] not in completed_ids
    ]
    prepared_post_collections: set[str] = set()
    prepared_planning_collections: set[str] = set()

    write_json(
        experiment_dir / "experiment_manifest.json",
        {
            "created_at": now_utc_iso(),
            "stage": args.stage,
            "dataset_path": str(dataset_path),
            "configs": selected_configs,
        },
    )

    for index, run_config in enumerate(selected_configs, start=1):
        print(f"[Experiment] {index}/{len(selected_configs)} {run_config['config_id']}", flush=True)
        os.environ["EMBED_MODEL"] = run_config["embedding_model"]
        os.environ["PGVECTOR_COLLECTION"] = run_config["post_collection"]
        os.environ["PGVECTOR_COLLECTION_PLANNING"] = run_config["planning_collection"]
        os.environ["PLANNING_CHUNKING_MODE"] = run_config["planning_chunking_mode"]
        os.environ["EVAL_TOP_K"] = str(run_config["top_k"])
        if run_config["embedding_model"].startswith("intfloat/"):
            os.environ["EMBED_QUERY_PREFIX"] = "query: "
            os.environ["EMBED_DOCUMENT_PREFIX"] = "passage: "
        else:
            os.environ.pop("EMBED_QUERY_PREFIX", None)
            os.environ.pop("EMBED_DOCUMENT_PREFIX", None)

        try:
            counts = await _ensure_ingested(
                run_config,
                config.get("corpus") or {},
                force_posts=bool(args.force_ingest) and run_config["post_collection"] not in prepared_post_collections,
                force_planning=bool(args.force_ingest) and run_config["planning_collection"] not in prepared_planning_collections,
                skip=bool(args.skip_ingest),
            )
        except Exception as exc:
            error = f"Ingest failed: {type(exc).__name__}: {exc}"
            print(f"[Experiment] FAILED {run_config['config_id']}: {error}", flush=True)
            rows.append(_failed_row(run_config, {}, error))
            _write_summary_outputs(rows, experiment_dir)
            continue
        prepared_post_collections.add(run_config["post_collection"])
        prepared_planning_collections.add(run_config["planning_collection"])
        report_name = f"experiments/{experiment_id}/runs/{run_config['config_id']}.json"
        try:
            _run_single_turn_eval(run_config, dataset_path, report_name, eval_root, config)
            report_path = eval_root / "reports" / report_name
            rows.append(_report_row(report_path, run_config, counts))
        except subprocess.CalledProcessError as exc:
            error = f"Command failed with exit code {exc.returncode}: {' '.join(str(part) for part in exc.cmd)}"
            print(f"[Experiment] FAILED {run_config['config_id']}: {error}", flush=True)
            rows.append(_failed_row(run_config, counts, error))
        _write_summary_outputs(rows, experiment_dir)

    _write_summary_outputs(rows, experiment_dir)
    print(f"[Experiment] Done. Summary: {experiment_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
