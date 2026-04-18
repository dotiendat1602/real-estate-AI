from __future__ import annotations

import os
from inspect import signature
from statistics import mean
from typing import Any


_DISPLAY_NAME_BY_KEY = {
    "contextual_recall": "Contextual Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "knowledge_retention": "Knowledge Retention",
}

_ACTIVE_SINGLE_TURN_METRICS = (
    "contextual_recall",
    "faithfulness",
    "answer_relevancy",
)

_DEFAULT_SINGLE_TURN_WEIGHTS = {
    "contextual_recall": 1 / 3,
    "faithfulness": 1 / 3,
    "answer_relevancy": 1 / 3,
    # "faithfulness": 1 / 2,
    # "answer_relevancy": 1 / 2,
}


def resolve_judge_model_and_apply_env(judge_config: dict[str, Any]) -> Any | None:
    """
    Resolve judge model from config and apply provider-specific environment setup.

    DeepEval metrics currently consume OpenAI-compatible environment variables.
    For DeepSeek scoring, we must force OpenAI-compatible variables to DeepSeek
    credentials/endpoints, even when runtime QA uses OpenAI.
    """
    judge = judge_config.get("judge", {}) if isinstance(judge_config, dict) else {}
    provider = str(judge.get("provider", "openai")).strip().lower()
    model = judge.get("model")
    if not model:
        return None

    model_name = str(model)
    os.environ["DEEPEVAL_MODEL"] = model_name

    if provider == "deepseek":
        api_base = str(judge.get("api_base", "https://api.deepseek.com/v1")).strip()
        deepseek_api_key = str(os.getenv("DEEPSEEK_API_KEY", "")).strip()
        if not deepseek_api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is missing while judge provider is set to deepseek."
            )

        # DeepEval judge uses OpenAI-compatible env vars; point them to DeepSeek.
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_KEY"] = deepseek_api_key

        # Prefer native DeepSeekModel for scoring so schema calls use JSON mode
        # consistently for DeepSeek models (avoids brittle plain-text JSON parsing).
        try:
            from deepeval.models import DeepSeekModel

            return DeepSeekModel(model=model_name, api_key=deepseek_api_key)
        except Exception:
            return model_name

    return model_name


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


def canonical_metric_key(metric_name: str) -> str | None:
    normalized = str(metric_name or "").strip().lower()
    if "contextual recall" in normalized:
        return "contextual_recall"
    if "contextual relevancy" in normalized:
        return "contextual_recall"
    if "faithfulness" in normalized:
        return "faithfulness"
    if "answer relevancy" in normalized:
        return "answer_relevancy"
    if "knowledge retention" in normalized:
        return "knowledge_retention"
    return None


def _mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _threshold_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _single_turn_thresholds(thresholds: dict[str, Any]) -> dict[str, float]:
    st = thresholds.get("single_turn", {}) if isinstance(thresholds, dict) else {}
    return {
        "contextual_recall": _threshold_float(
            st.get("contextual_recall", st.get("contextual_relevancy")),
            0.75,
        ),
        "faithfulness": _threshold_float(st.get("faithfulness"), 0.80),
        "answer_relevancy": _threshold_float(st.get("answer_relevancy"), 0.80),
    }


def _conversation_thresholds(thresholds: dict[str, Any]) -> dict[str, float]:
    conv = thresholds.get("conversation", {}) if isinstance(thresholds, dict) else {}
    return {
        "knowledge_retention": _threshold_float(conv.get("knowledge_retention"), 0.75),
    }


def _single_turn_composite_threshold(thresholds: dict[str, Any]) -> float:
    st = thresholds.get("single_turn", {}) if isinstance(thresholds, dict) else {}
    return _threshold_float(st.get("composite_pass_threshold"), 0.75)


def _collect_metric_scores(deepeval_result: dict[str, Any]) -> dict[str, list[float]]:
    by_key: dict[str, list[float]] = {}
    test_results = deepeval_result.get("test_results", []) if isinstance(deepeval_result, dict) else []

    for test_result in test_results:
        if not isinstance(test_result, dict):
            continue
        metrics_data = test_result.get("metrics_data", [])
        if not isinstance(metrics_data, list):
            continue

        for metric in metrics_data:
            if not isinstance(metric, dict):
                continue
            metric_key = canonical_metric_key(str(metric.get("name", "")))
            if not metric_key:
                continue

            score = metric.get("score")
            if isinstance(score, (int, float)):
                by_key.setdefault(metric_key, []).append(float(score))

    return by_key


def _collect_case_metric_scores(
    deepeval_result: dict[str, Any],
    metric_keys: tuple[str, ...],
) -> list[dict[str, float]]:
    case_scores: list[dict[str, float]] = []
    required_keys = set(metric_keys)
    test_results = deepeval_result.get("test_results", []) if isinstance(deepeval_result, dict) else []

    for test_result in test_results:
        if not isinstance(test_result, dict):
            continue
        metrics_data = test_result.get("metrics_data", [])
        if not isinstance(metrics_data, list):
            continue

        score_by_key: dict[str, float] = {}
        for metric in metrics_data:
            if not isinstance(metric, dict):
                continue
            metric_key = canonical_metric_key(str(metric.get("name", "")))
            if metric_key not in required_keys:
                continue

            score = metric.get("score")
            if isinstance(score, (int, float)):
                score_by_key[metric_key] = float(score)

        if required_keys.issubset(score_by_key.keys()):
            case_scores.append({key: score_by_key[key] for key in metric_keys})

    return case_scores


def _build_case_composite_statistics(
    case_scores: list[dict[str, float]],
    metric_keys: tuple[str, ...],
    threshold_by_key: dict[str, float],
    composite_threshold: float,
) -> dict[str, float | int | None]:
    if not case_scores:
        return {
            "sample_size": 0,
            "composite_threshold": composite_threshold,
            "mean_score": None,
            "pass_count": 0,
            "pass_rate": None,
            "strict_all_metric_threshold_pass_count": 0,
            "strict_all_metric_threshold_pass_rate": None,
        }

    composite_scores = [mean([case[key] for key in metric_keys]) for case in case_scores]
    pass_count = sum(1 for value in composite_scores if value >= composite_threshold)
    strict_pass_count = 0
    for case in case_scores:
        if all(case.get(metric_key, -1.0) >= threshold_by_key.get(metric_key, 1.0) for metric_key in metric_keys):
            strict_pass_count += 1

    sample_size = len(case_scores)
    return {
        "sample_size": sample_size,
        "composite_threshold": _round_or_none(composite_threshold),
        "mean_score": _round_or_none(float(mean(composite_scores))),
        "pass_count": pass_count,
        "pass_rate": _round_or_none(pass_count / sample_size),
        "strict_all_metric_threshold_pass_count": strict_pass_count,
        "strict_all_metric_threshold_pass_rate": _round_or_none(strict_pass_count / sample_size),
    }


def _build_metric_statistics(
    metric_keys: tuple[str, ...],
    scores_by_key: dict[str, list[float]],
    threshold_by_key: dict[str, float],
) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    for metric_key in metric_keys:
        values = scores_by_key.get(metric_key, [])
        threshold = threshold_by_key.get(metric_key)
        avg_score = _mean_or_none(values)

        pass_count = 0
        pass_rate: float | None = None
        if threshold is not None and values:
            pass_count = sum(1 for value in values if value >= threshold)
            pass_rate = pass_count / len(values)

        stats[metric_key] = {
            "display_name": _DISPLAY_NAME_BY_KEY.get(metric_key, metric_key),
            "sample_size": len(values),
            "threshold": threshold,
            "mean_score": _round_or_none(avg_score),
            "pass_count": pass_count,
            "pass_rate": _round_or_none(pass_rate),
        }

    return stats


def _weighted_quality_score(
    metric_statistics: dict[str, Any],
    weights: dict[str, float],
) -> float | None:
    numerator = 0.0
    denominator = 0.0

    for metric_key, weight in weights.items():
        stat = metric_statistics.get(metric_key, {})
        avg = stat.get("mean_score") if isinstance(stat, dict) else None
        if avg is None:
            continue
        numerator += float(weight) * float(avg)
        denominator += float(weight)

    if denominator == 0:
        return None
    return numerator / denominator


def build_single_turn_scorecard(
    deepeval_result: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Build transparent formula-driven scorecard for single-turn evaluation."""
    thresholds_by_key = _single_turn_thresholds(thresholds)
    composite_threshold = _single_turn_composite_threshold(thresholds)
    scores_by_key = _collect_metric_scores(deepeval_result)
    case_scores = _collect_case_metric_scores(deepeval_result, _ACTIVE_SINGLE_TURN_METRICS)
    metric_stats = _build_metric_statistics(
        metric_keys=_ACTIVE_SINGLE_TURN_METRICS,
        scores_by_key=scores_by_key,
        threshold_by_key=thresholds_by_key,
    )
    case_composite_stats = _build_case_composite_statistics(
        case_scores=case_scores,
        metric_keys=_ACTIVE_SINGLE_TURN_METRICS,
        threshold_by_key=thresholds_by_key,
        composite_threshold=composite_threshold,
    )

    overall_quality = _weighted_quality_score(metric_stats, _DEFAULT_SINGLE_TURN_WEIGHTS)

    return {
        "included_metrics": list(_ACTIVE_SINGLE_TURN_METRICS),
        "excluded_metrics": [],
        "weights": _DEFAULT_SINGLE_TURN_WEIGHTS,
        "formulas": {
            "metric_mean": "S_m = (1/N) * sum_{i=1..N}(s_{m,i})",
            "metric_pass_rate": "PassRate_m = (1/N) * sum_{i=1..N}(1[s_{m,i} >= T_m])",
            "overall_quality": "Q_single = (w_CRec*S_CRec + w_F*S_F + w_AR*S_AR) / (w_CRec + w_F + w_AR)",
            "case_composite_score": "Q_case = (s_CRec + s_F + s_AR) / 3",
            "case_composite_pass_rate": "PassRate_case = (1/N) * sum_{i=1..N}(1[Q_case,i >= T_case])",
        },
        "symbols": {
            "S_CRec": "mean Contextual Recall score",
            "S_F": "mean Faithfulness score",
            "S_AR": "mean Answer Relevancy score",
            "T_m": "metric threshold",
            "T_case": "composite case threshold",
        },
        "metric_statistics": metric_stats,
        "case_composite_statistics": case_composite_stats,
        "overall_quality_score": _round_or_none(overall_quality),
        "overall_quality_percent": _round_or_none(
            None if overall_quality is None else overall_quality * 100,
            digits=2,
        ),
        "score_range": "[0, 1]",
    }


def build_conversation_scorecard(
    deepeval_result: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Build transparent formula-driven scorecard for conversation evaluation."""
    metric_key = "knowledge_retention"
    threshold_by_key = _conversation_thresholds(thresholds)
    scores_by_key = _collect_metric_scores(deepeval_result)
    metric_stats = _build_metric_statistics(
        metric_keys=(metric_key,),
        scores_by_key=scores_by_key,
        threshold_by_key=threshold_by_key,
    )
    avg_score = metric_stats.get(metric_key, {}).get("mean_score")
    overall_quality = float(avg_score) if isinstance(avg_score, (int, float)) else None

    return {
        "included_metrics": [metric_key],
        "excluded_metrics": [],
        "weights": {metric_key: 1.0},
        "formulas": {
            "metric_mean": "S_KR = (1/N) * sum_{i=1..N}(s_{KR,i})",
            "metric_pass_rate": "PassRate_KR = (1/N) * sum_{i=1..N}(1[s_{KR,i} >= T_KR])",
            "overall_quality": "Q_conversation = S_KR",
        },
        "symbols": {
            "S_KR": "mean Knowledge Retention score",
            "T_KR": "Knowledge Retention threshold",
        },
        "metric_statistics": metric_stats,
        "overall_quality_score": _round_or_none(overall_quality),
        "overall_quality_percent": _round_or_none(
            None if overall_quality is None else overall_quality * 100,
            digits=2,
        ),
        "score_range": "[0, 1]",
    }
