# Baseline Thresholds

| Metric | Threshold |
|---|---:|
| Contextual Recall | 0.65 |
| Faithfulness | 0.70 |
| Answer Relevancy | 0.70 |
| Composite case pass | 0.70 |
| Knowledge Retention | 0.75 |

## Scoring formulas

- Per-metric mean score: `S_m = (1/N) * sum_{i=1..N}(s_{m,i})`
- Per-metric pass rate: `PassRate_m = (1/N) * sum_{i=1..N}(1[s_{m,i} >= T_m])`
- Single-turn overall quality:
	`Q_single = (w_CR*S_CR + w_F*S_F + w_AR*S_AR) / (w_CR + w_F + w_AR)`
- Single-turn case composite:
	`Q_case = (s_CR + s_F + s_AR) / 3`
- Default weights: `w_CR = w_F = w_AR = 1/3`
- Conversation overall quality: `Q_conversation = S_KR`

Where:
- `S_CR`: mean Contextual Recall
- `S_F`: mean Faithfulness
- `S_AR`: mean Answer Relevancy
- `S_KR`: mean Knowledge Retention
- `T_m`: metric threshold

## Gate recommendation
- Do not fail any critical conversational scenario.
- Track both per-metric pass rate and composite case pass; they answer different questions.
- Keep the observed difficulty ordering sensible when the benchmark is meant to be stratified as easy/medium/hard.
- Review all low-score hard cases manually before sign-off.
