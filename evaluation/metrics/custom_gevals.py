from __future__ import annotations

from typing import Any

from .metric_helpers import safe_metric_init


class RobustKnowledgeRetentionMetric:
    """Wrap DeepEval KnowledgeRetention to avoid false zero-score when no verdicts are generated."""

    def __new__(
        cls,
        threshold: float = 0.75,
        model: Any | None = None,
    ):
        from deepeval.metrics import KnowledgeRetentionMetric

        class _Metric(KnowledgeRetentionMetric):
            def _calculate_score(self) -> float:  # type: ignore[override]
                # DeepEval returns 0 when verdict list is empty. Treat this as neutral-pass
                # because there is no extractable prior user fact to evaluate retention on.
                if len(getattr(self, "verdicts", []) or []) == 0:
                    return 1.0
                return super()._calculate_score()

            def _generate_reason(self) -> str | None:  # type: ignore[override]
                if len(getattr(self, "verdicts", []) or []) == 0:
                    return "No prior user facts were extractable for retention checks; treated as neutral pass."
                return super()._generate_reason()

        return _Metric(
            threshold=threshold,
            model=model,
        )


def build_knowledge_retention_metric(
    thresholds: dict[str, Any],
    judge_model: Any | None = None,
):
    conv = thresholds.get("conversation", {})
    return safe_metric_init(
        RobustKnowledgeRetentionMetric,
        threshold=float(conv.get("knowledge_retention", 0.75)),
        model=judge_model,
    )
