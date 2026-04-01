# Offline Evaluation for RAG Conversational Chatbot

This package implements a complete offline evaluation workflow for the Estatein AI chatbot with:

- Single-turn RAG evaluation
- Multi-turn conversational evaluation
- RAG grounding and relevance metrics
- Custom GEval metrics for intent, helpfulness, domain correctness, and conversation consistency
- JSON trace logging and baseline report generation

## Folder layout

- config: judge model, thresholds, runtime settings
- datasets: golden datasets for single-turn and conversations
- adapters: runtime adapter that calls existing LangChain pipeline
- metrics: metric builders for RAG + GEval
- runners: executable scripts for evaluation
- reports: generated baselines and analysis notes

## Prerequisites

1. Configure environment variables for the existing service (especially OPENAI_API_KEY and PGVECTOR_URL).
2. Ensure vector data is already indexed.
3. Install dependencies:

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

## Notes

- Adapter logs retrieval context, doc IDs, relevance scores (when available), and generation metadata.
- `rewritten_query` is set to null by default because current pipeline does not run explicit query rewriting.
- Add more goldens in datasets for stronger coverage before production gating.
