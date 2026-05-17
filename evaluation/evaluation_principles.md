# Evaluation Principles

## Definition of quality
- Answers are grounded in retrieved context and avoid unsupported claims.
- Responses are useful, concise, and aligned with user intent.
- Conversational constraints must persist across turns.
- The assistant should ask clarification questions when context is insufficient.

## Phase-1 required metrics
- Contextual Recall
- Faithfulness
- Answer Relevancy
- Knowledge Retention

## Transparent scoring formulas
- Per-metric mean: `S_m = (1/N) * sum_{i=1..N}(s_{m,i})`
- Per-metric pass rate: `PassRate_m = (1/N) * sum_{i=1..N}(1[s_{m,i} >= T_m])`
- Single-turn overall quality: `Q_single = (w_CR*S_CR + w_F*S_F + w_AR*S_AR) / (w_CR + w_F + w_AR)`
- Single-turn case composite: `Q_case = (s_CR + s_F + s_AR) / 3`
- Conversation overall quality: `Q_conversation = S_KR`

## Reproducibility rules
- Keep judge model, judge prompt criteria, and thresholds versioned.
- Default judge provider for offline evaluation is DeepSeek (`deepseek-chat`).
- Run with fixed judge temperature (0.0 where supported).
- Store run id, timestamp, and config snapshot in each report.
- Keep `deepeval_result.test_results` and `traces` aligned; never edit aggregate percentages without rebuilding them from case-level results.
