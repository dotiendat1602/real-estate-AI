from __future__ import annotations

from typing import Any

from .metric_helpers import safe_metric_init


def build_rag_metrics(thresholds: dict[str, Any], judge_model: Any | None = None) -> list[Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    st = thresholds.get("single_turn", {})
    contextual_recall_threshold = float(
        st.get("contextual_recall", st.get("contextual_relevancy", 0.75))
    )

    metrics = [
        safe_metric_init(
            ContextualRecallMetric,
            threshold=contextual_recall_threshold,
            model=judge_model,
        ),
        safe_metric_init(
            FaithfulnessMetric,
            threshold=float(st.get("faithfulness", 0.80)),
            model=judge_model,
        ),
        safe_metric_init(
            AnswerRelevancyMetric,
            threshold=float(st.get("answer_relevancy", 0.80)),
            model=judge_model,
        ),
    ]
    return metrics
