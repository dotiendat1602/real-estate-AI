# Evaluation Principles

## Definition of quality
- Answers are grounded in retrieved context and avoid unsupported claims.
- Responses are useful, concise, and aligned with user intent.
- Conversational constraints must persist across turns.
- The assistant should ask clarification questions when context is insufficient.

## Phase-1 required metrics
- Contextual Relevancy
- Faithfulness
- Answer Relevancy
- GEval Intent Coverage & Helpfulness
- Conversational GEval Consistency

## Reproducibility rules
- Keep judge model, judge prompt criteria, and thresholds versioned.
- Run with fixed judge temperature (0.0 where supported).
- Store run id, timestamp, and config snapshot in each report.
