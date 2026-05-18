# 01. Tổng quan hệ thống evaluation

## Mục tiêu

`evaluation/` là lớp kiểm thử offline cho chatbot RAG. Nó không thay thế unit test hay integration test của API, mà trả lời các câu hỏi chất lượng ở mức hệ thống:

- Retrieval có lấy được đúng ngữ cảnh không?
- Câu trả lời có bám vào context hay tự suy diễn?
- Câu trả lời có đúng trọng tâm câu hỏi không?
- Trong hội thoại nhiều lượt, hệ thống có giữ được các ràng buộc đã xuất hiện trước đó không?
- Khi thay embedding model, chunking strategy hoặc `top-k`, cấu hình nào cân bằng nhất giữa chất lượng và chi phí?

## Hai nhánh đánh giá

### 1. Single-turn

Mỗi case là một câu hỏi độc lập. Nhánh này đo ba metric:

- `Contextual Recall`: retrieval context có bao phủ đủ fact trong expected output hay không.
- `Faithfulness`: actual answer có bám vào retrieval context hay không.
- `Answer Relevancy`: actual answer có trả lời đúng trọng tâm input hay không.

### 2. Conversational

Mỗi case gồm nhiều lượt hội thoại. Nhánh này dùng `Knowledge Retention` để kiểm tra việc assistant có giữ đúng các fact/ràng buộc người dùng đã nêu ở lượt trước hay không.

## Cấu trúc thư mục

| Thư mục/file | Vai trò |
|---|---|
| `config/` | Threshold, judge model, runtime settings, cấu hình ma trận thực nghiệm RAG. |
| `datasets/` | Golden datasets single-turn và conversational. |
| `adapters/` | Cầu nối từ runtime RAG thật sang trace/testcase đánh giá. |
| `metrics/` | Builder cho metric DeepEval và scorecard tự tính. |
| `runners/` | Entry point để chạy eval, profile runtime, generate candidate data, chạy ma trận RAG. |
| `reports/` | JSON report, trace checkpoint, markdown summary, artifact thực nghiệm. |
| `corpus/` | Manifest corpus dùng khi ingest cho experiment matrix. |

## Luồng tổng quát

```text
golden dataset
  -> LangChainEvalAdapter
  -> runtime RAG thật của app
  -> TurnTrace / ConversationTrace
  -> DeepEval testcase
  -> metric judge
  -> scorecard
  -> JSON report + trace checkpoint
```

Evaluation gọi lại chính runtime đang dùng trong app: embeddings, vector store, filter extractor, planning retrieval, prompt và postprocess answer. Vì vậy nếu runtime thay đổi thì hành vi evaluation cũng thay đổi theo.

## Các file code trung tâm

| File | Vai trò |
|---|---|
| `adapters/langchain_adapter.py` | Chạy một lượt hỏi hoặc một conversation qua pipeline RAG thật và ghi trace. |
| `adapters/trace_to_testcase.py` | Chuyển trace + golden thành `LLMTestCase` hoặc `ConversationalTestCase`. |
| `metrics/rag_metrics.py` | Tạo `ContextualRecallMetric`, `FaithfulnessMetric`, `AnswerRelevancyMetric`. |
| `metrics/custom_gevals.py` | Bọc `KnowledgeRetentionMetric` để xử lý trường hợp không extract được prior fact. |
| `metrics/metric_helpers.py` | Chuẩn hóa metric key, tính scorecard, weighted quality, timeout/judge env. |
| `runners/run_single_turn_eval.py` | Tạo trace, chấm metric, ghi report single-turn. |
| `runners/run_conversation_eval.py` | Tạo/resume trace và ghi report conversational. |
| `runners/profile_single_turn_runtime.py` | Đo runtime/token không cần gọi DeepEval judge. |
| `runners/run_rag_experiment_matrix.py` | Chạy ma trận embedding x chunking x top-k để chọn cấu hình RAG. |

## Cấu hình hiện tại

Các threshold thực sự đang được dùng nằm trong `evaluation/config/thresholds.yaml`:

| Metric | Threshold |
|---|---:|
| Contextual Recall | `0.65` |
| Faithfulness | `0.70` |
| Answer Relevancy | `0.70` |
| Composite case pass | `0.70` |
| Knowledge Retention | `0.75` |

Judge mặc định trong `evaluation/config/judge_model.yaml` là `deepseek-chat` với temperature `0.0`.

## Phân biệt với docs RAG runtime

- `docs/rag-system/*` giải thích hệ thống hoạt động như thế nào khi user chat thật.
- `docs/evaluation/*` giải thích cách hệ thống được đo, benchmark được dựng, và kết quả được tổng hợp.

