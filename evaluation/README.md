# Offline Evaluation for RAG Conversational Chatbot

This package implements a complete offline evaluation workflow for the Estatein AI chatbot with:

- Single-turn RAG evaluation
- Multi-turn conversational evaluation
- RAG grounding and relevance metrics
- Knowledge Retention metric for conversational memory quality
- JSON trace logging and baseline report generation

## Folder layout

- config: judge model, thresholds, runtime settings
- datasets: golden datasets for single-turn and conversations
- adapters: runtime adapter that calls existing LangChain pipeline
- metrics: metric builders for RAG + conversational memory metric
- runners: executable scripts for evaluation
- reports: generated baselines and analysis notes

## Prerequisites

1. Configure environment variables for the existing service (especially PGVECTOR_URL).
2. Configure judge credentials:
	- Preferred (DeepSeek): set `DEEPSEEK_API_KEY`.
	- Fallback compatibility: you can also provide `OPENAI_API_KEY` if your endpoint is OpenAI-compatible.
3. Ensure vector data is already indexed.
4. Install dependencies:

```bash
pip install -e .
```

## Run evaluations

From ai-service root:

```bash
python -m evaluation.runners.run_single_turn_eval
python -m evaluation.runners.run_conversation_eval
```

## Outputs

- reports/baseline_eval_single_turn.json
- reports/baseline_eval_conversation.json
- reports/baseline_summary.md

Single-turn and conversation reports now include a `scorecard` section with explicit formulas and computed values.

Main formulas:

- `S_m = (1/N) * sum_{i=1..N}(s_{m,i})`
- `PassRate_m = (1/N) * sum_{i=1..N}(1[s_{m,i} >= T_m])`
- `Q_single = (w_CR*S_CR + w_F*S_F + w_AR*S_AR) / (w_CR + w_F + w_AR)`
- `Q_conversation = S_KR`

## Notes

- Adapter logs retrieval context, doc IDs, relevance scores (when available), and generation metadata.
- `rewritten_query` is set to null by default because current pipeline does not run explicit query rewriting.
- Add more goldens in datasets for stronger coverage before production gating.
