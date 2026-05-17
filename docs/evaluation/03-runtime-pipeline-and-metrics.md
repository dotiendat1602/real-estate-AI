# 03. Runtime pipeline và metric

## Từ golden đến trace

`LangChainEvalAdapter` gọi lại pipeline RAG thật:

```text
golden.input
  -> repair text / history
  -> build_retrieval_query()
  -> chọn planning mode hay listing mode
  -> extract filter nếu cần
  -> retrieve docs
  -> prepare_docs_for_context()
  -> prompt + LLM
  -> postprocess_answer()
  -> TurnTrace
```

`TurnTrace` giữ:

- `input`
- `conversation_history_used`
- `rewritten_query`
- `retrieval_context`
- `retrieved_doc_ids`
- `retrieval_scores`
- `actual_output`
- `metadata`

`metadata` là vùng debug chính: model, retrieval strategy, filter extraction, timing, token usage, context size và các cờ planning/listing.

## Golden context khác retrieval context

| Loại | Nguồn | Vai trò |
|---|---|---|
| `golden.context` | Dataset | Ground truth để chấm recall. |
| `trace.retrieval_context` | Runtime thật | Context assistant thực sự dùng để trả lời. |

Nếu `golden.context` đúng nhưng `retrieval_context` sai, vấn đề thường nằm ở retrieval. Nếu `retrieval_context` đúng nhưng answer sai, vấn đề thường nằm ở generation hoặc postprocess.

## Từ trace sang testcase

`trace_to_testcase.py` tạo:

- `LLMTestCase` cho single-turn,
- `ConversationalTestCase` cho conversational.

`_sanitize_expected_output_for_recall()` loại các dòng rubric như "không suy đoán" khỏi expected output trước khi chấm recall, để metric chỉ kiểm tra fact thật.

## Metric single-turn

| Metric | Ý nghĩa | Threshold |
|---|---|---:|
| `Contextual Recall` | Retrieval context có đủ fact cần thiết không | `0.65` |
| `Faithfulness` | Answer có được context hỗ trợ không | `0.70` |
| `Answer Relevancy` | Answer có đúng trọng tâm input không | `0.70` |

## Metric conversational

`Knowledge Retention` đo việc assistant có giữ được prior user facts qua các lượt sau không. Repo dùng `RobustKnowledgeRetentionMetric` để coi trường hợp không extract được prior fact nào là neutral pass thay vì false zero-score.

Threshold hiện tại: `0.75`.

## Scorecard

```text
S_m = (1/N) * sum(s_m,i)
PassRate_m = (1/N) * sum(1[s_m,i >= T_m])
Q_single = (w_CR*S_CR + w_F*S_F + w_AR*S_AR) / (w_CR + w_F + w_AR)
Q_case = (s_CR + s_F + s_AR) / 3
```

Ba trọng số single-turn hiện tại đều bằng `1/3`.

## `Case Pass` và `strict pass`

- `case_composite_statistics.pass_rate`
  - mỗi case lấy trung bình ba metric,
  - trung bình đó đạt `0.70` là pass.

- `strict_all_metric_threshold_pass_rate`
  - cùng một case phải vượt cả ba threshold riêng.

`strict pass` luôn bằng hoặc thấp hơn `case pass`.

## Judge model và timeout

- Judge mặc định: `deepseek-chat`.
- `resolve_judge_model_and_apply_env()` chỉ đổi env trong pha scoring, không đổi QA model dùng để sinh answer.
- `apply_deepeval_timeout_env()` hỗ trợ timeout explicit hoặc tắt timeout của DeepEval khi cần.

## Cách debug case fail

1. Xem `golden.input`, `golden.context`, `expected_output_outline`.
2. Xem `trace.retrieval_context`.
3. Xem `actual_output`.
4. Xem `metrics_data[].score` và `reason`.
5. CR thấp -> kiểm tra retrieval.
6. CR cao nhưng F thấp -> answer đang thêm claim ngoài context.
7. F cao nhưng AR thấp -> answer đúng nhưng lệch trọng tâm.
8. Conversational fail -> kiểm tra prior user facts và history.

