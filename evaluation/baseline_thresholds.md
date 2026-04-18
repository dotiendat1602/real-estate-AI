# Baseline Thresholds

| Metric | Threshold |
|---|---:|
| Contextual Relevancy | 0.75 |
| Faithfulness | 0.80 |
| Answer Relevancy | 0.80 |
| Knowledge Retention | 0.75 |

## Scoring formulas

- Per-metric mean score: `S_m = (1/N) * sum_{i=1..N}(s_{m,i})`
- Per-metric pass rate: `PassRate_m = (1/N) * sum_{i=1..N}(1[s_{m,i} >= T_m])`
- Single-turn overall quality:
	`Q_single = (w_CR*S_CR + w_F*S_F + w_AR*S_AR) / (w_CR + w_F + w_AR)`
- Default weights: `w_CR = w_F = w_AR = 1/3`
- Conversation overall quality: `Q_conversation = S_KR`

Where:
- `S_CR`: mean Contextual Relevancy
- `S_F`: mean Faithfulness
- `S_AR`: mean Answer Relevancy
- `S_KR`: mean Knowledge Retention
- `T_m`: metric threshold

## Gate recommendation
- Do not fail any critical conversational scenario.
- Keep pass rate >= 70% for each difficulty band in phase-1.
- Review all low-score hard cases manually before sign-off.
