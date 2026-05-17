# 06. Deep dive evaluation code

File này giải thích các cụm logic dài và dễ đọc sai nhất trong:

- `evaluation/adapters/langchain_adapter.py`
- `evaluation/adapters/trace_to_testcase.py`
- `evaluation/runners/run_single_turn_eval.py`
- `evaluation/runners/run_conversation_eval.py`
- `evaluation/metrics/metric_helpers.py`

## 1. `LangChainEvalAdapter`

### `__init__(settings)`

Adapter đọc runtime setting một lần rồi giữ lại:

- `top_k`
- `max_history_turns`
- `eval_context_max_docs`
- `planning_eval_context_max_docs`
- `eval_context_max_chars_per_doc`
- `eval_explanatory_max_docs`
- `eval_explanatory_max_chars_per_doc`

Nó chưa khởi tạo embeddings/vector store/LLM ngay. Các resource nặng này được để `None` cho tới khi `initialize()` chạy.

### `initialize()`

Mục đích:

1. build embeddings,
2. build listing vector store,
3. build planning vector store,
4. build LLM.

Planning collection được lấy từ `PGVECTOR_COLLECTION_PLANNING`, nên khi experiment matrix đổi env thì adapter tự trỏ sang collection tương ứng với config đó.

### `run_single_turn(...)`

Đây là hàm trung tâm của evaluation runtime.

#### Pha 1: chuẩn hóa input và history

- sửa text lỗi encoding bằng `_repair_text()`,
- cắt history về `max_history_turns`,
- chuyển history sang LangChain messages.

#### Pha 2: xử lý `target_metadata`

- nếu metadata có `evalOnly=true`, nó chỉ còn vai trò mô tả golden,
- nếu không, metadata có thể tham gia retrieval,
- nếu có `planningDocumentId`, adapter có thể force target planning document.

#### Pha 3: rewrite query và chọn mode

- `build_retrieval_query()` dựng query cho follow-up turn,
- nếu query rewrite khác input gốc thì ghi vào `rewritten_query`,
- `_has_planning_intent()` và metadata quyết định planning/listing mode.

#### Pha 4: filter extraction

- planning mode: bỏ qua filter extraction listing,
- exact listing target: cũng bỏ qua vì đã có `postId/propertyId`,
- còn lại: gọi LLM filter extractor.

Metadata sẽ ghi rõ:

- `filter_extraction_skipped`,
- `filter_extraction_skip_reason`,
- `filter_extraction_seconds`,
- `filter_extraction_token_usage`.

#### Pha 5: retrieve docs

Planning:

- `planning_nl` nếu hỏi tự nhiên,
- target-specific planning retrieval nếu ép `planningDocumentId`.

Listing:

- exact retrieval nếu có target exact,
- retriever thường,
- fallback retrieval nếu cần.

#### Pha 6: chuẩn bị context và sinh answer

- `_prepare_eval_context_docs()` áp quota docs/char dành riêng cho eval,
- `_generate_answer()` gọi prompt + LLM,
- `postprocess_answer()` chạy cùng logic runtime thật.

#### Pha 7: đóng gói trace

Trace không chỉ có answer, mà còn có:

- retrieval context thực tế,
- doc ids,
- scores,
- strategy,
- timing,
- token usage,
- target metadata flags.

Nhờ đó report có thể dùng để debug mà không cần chạy lại RAG.

### `run_conversation(...)`

Hàm này không có pipeline riêng. Nó:

1. duyệt từng turn,
2. khi gặp user turn thì gọi lại `run_single_turn()`,
3. append user message và assistant answer mới sinh vào history,
4. giữ explicit scripted assistant turn nếu golden có sẵn.

Điều này bảo đảm conversational eval đo đúng behavior mà runtime single-turn đang tạo ra qua nhiều lượt liên tiếp.

### `_retrieve(...)`

Logic retrieval listing:

1. nếu filter là exact target, thử `_retrieve_exact_listing_target()` trước,
2. nếu không hoặc exact fail, build retriever thường,
3. nếu retriever hỗ trợ score thì lấy score trực tiếp,
4. nếu không, cố truy vấn lại vector store để map relevance score theo stable doc key,
5. nếu backend không hỗ trợ thì giữ `None`.

### `_retrieve_planning_target_docs(...)`

Mục đích: khi golden nhắm vào một tài liệu planning cụ thể, lấy docs trong cùng tài liệu đó nhưng vẫn giữ khả năng chọn evidence phù hợp.

Hàm này:

- load docs của document target,
- score theo metadata target,
- chọn docs theo rank,
- bổ sung neighbor/compact tương tự runtime planning,
- tránh việc chỉ vì biết `planningDocumentId` mà đưa cả tài liệu thô vào prompt.

### `_prepare_eval_context_docs(...)`

Evaluation cần quota context ổn định để các run so sánh được. Hàm này chọn giới hạn khác nhau:

- listing thường,
- planning,
- explanatory query.

Nếu không tách bước này, một run hard có thể vô tình nhận nhiều docs hơn hẳn run easy, làm benchmark khó diễn giải.

## 2. `trace_to_testcase.py`

### `_sanitize_expected_output_for_recall()`

Mục tiêu: tách fact khỏi rubric.

Ví dụ:

```text
- Nêu đúng giá bán: 14 tỷ
- Không suy đoán nếu thiếu dữ liệu
```

Khi chấm recall, dòng thứ hai không phải fact cần retrieval. Hàm này:

- bỏ guidance line,
- giữ dòng có số hoặc fact hint,
- rephrase directive thành fact gọn như `Giá: 14 tỷ`.

### `build_single_turn_test_case()`

Tạo `LLMTestCase` với:

- `input` từ trace,
- `actual_output` từ runtime,
- `retrieval_context` từ trace,
- `context` từ golden,
- `expected_output` đã sanitize,
- metadata bổ sung để report còn biết domain/difficulty/question type.

### `build_conversation_test_case()`

Tạo `ConversationalTestCase` bằng cách chuyển từng `TurnTrace` thành turn DeepEval. Điều này giúp `Knowledge Retention` đọc được chuỗi user/assistant turns đầy đủ.

## 3. `run_single_turn_eval.py`

### Pha 1: load config và dataset

Runner đọc:

- threshold,
- eval settings,
- judge config,
- golden dataset.

Nếu CLI có `--top-k`, runner override setting runtime cho run hiện tại.

### Pha 2: sinh trace hoặc nạp trace cũ

Hai nhánh:

1. có `--trace-input`
   - nạp trace checkpoint,
   - không gọi lại RAG.

2. không có `--trace-input`
   - tạo adapter,
   - chạy từng golden,
   - ghi checkpoint `.traces.json`.

### Pha 3: scoring

Runner build metric list rồi chọn một trong ba đường:

- `--manual-metric-loop`,
- batch scoring qua `--eval-batch-size`,
- hoặc gọi `deepeval.evaluate()` trực tiếp.

Sau đó dựng `scorecard` từ chính `deepeval_result`.

### Pha 4: ghi report

Report single-turn chứa cả:

- raw DeepEval result,
- scorecard dễ đọc,
- traces,
- timing,
- judge cost profile,
- experiment config.

Đây là lý do report single-turn đủ dữ liệu để merge, audit hoặc upload lại.

## 4. `run_conversation_eval.py`

Khác single-turn ở ba điểm:

1. trace có thể được resume từng phần bằng `--resume-trace-input`,
2. mỗi case có nhiều user turns,
3. metric chỉ là `Knowledge Retention`.

Runner ghi checkpoint partial trong lúc sinh trace để nếu một run dài bị dừng thì không phải làm lại toàn bộ conversation đã xong.

## 5. `metric_helpers.py`

### `canonical_metric_key()`

Chuẩn hóa tên metric từ DeepEval về key nội bộ:

- `contextual_recall`
- `faithfulness`
- `answer_relevancy`
- `knowledge_retention`

Điều này làm scorecard không phụ thuộc vào cách DeepEval viết display name.

### `_collect_metric_scores()` và `_collect_case_metric_scores()`

- Hàm đầu gom score theo từng metric.
- Hàm sau chỉ giữ case có đủ mọi metric cần thiết.

Đây là lý do sample size của `case_composite_statistics` có thể nhỏ hơn tổng số case nếu một metric nào đó bị thiếu ở một testcase.

### `_build_metric_statistics()`

Tính:

- `sample_size`
- `threshold`
- `mean_score`
- `pass_count`
- `pass_rate`

### `_build_case_composite_statistics()`

Tính:

- composite mean của từng case,
- `case pass`,
- `strict all metric threshold pass`.

### `build_single_turn_scorecard()`

Ghép toàn bộ metric stats + case stats + formula + weighted quality thành object `scorecard` hoàn chỉnh.

### `build_conversation_scorecard()`

Đơn giản hơn:

- chỉ dùng `knowledge_retention`,
- `Q_conversation = S_KR`.

## 6. Những lỗi diễn giải thường gặp

1. Thấy `mean_score` cao rồi kết luận pass rate cũng cao.
2. Dùng `golden.context` để nói model đã thấy gì, trong khi model thật thấy `retrieval_context`.
3. Sửa `scorecard` mà không rebuild từ `test_results`.
4. So sánh hai run có quota context khác nhau mà coi như cùng điều kiện.
5. Nhìn `difficulty` label mà không kiểm tra profile thực nghiệm của từng bucket.

