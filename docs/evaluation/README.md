# Tài liệu hệ thống evaluation

Nhánh tài liệu này mô tả đầy đủ pipeline đánh giá offline trong `evaluation/` của `ai-service`. Mục tiêu là trả lời bốn câu hỏi:

1. Dataset đánh giá được cấu trúc như thế nào?
2. Một golden đi qua runtime RAG, trace và DeepEval ra sao?
3. Các metric, threshold và scorecard thực sự đo điều gì?
4. Khi chạy runner hoặc đọc report, cần nhìn field nào để không diễn giải sai kết quả?

## Các file tài liệu

- [01-overview.md](01-overview.md): kiến trúc tổng quan, vai trò từng thư mục, hai nhánh single-turn/conversational.
- [02-datasets-and-goldens.md](02-datasets-and-goldens.md): schema dataset, `target_metadata`, review input, benchmark 700 case hiện tại.
- [03-runtime-pipeline-and-metrics.md](03-runtime-pipeline-and-metrics.md): luồng golden -> adapter -> trace -> testcase -> metric -> scorecard.
- [04-running-evaluations-and-reading-reports.md](04-running-evaluations-and-reading-reports.md): lệnh chạy, trace checkpoint, report schema, cách đọc và cách kiểm tra upload format.
- [05-rag-experiments-and-current-artifacts.md](05-rag-experiments-and-current-artifacts.md): ma trận thực nghiệm chọn cấu hình RAG, artifact quan trọng, bộ report final hiện tại.
- [06-deep-dive-evaluation-code.md](06-deep-dive-evaluation-code.md): giải thích theo từng hàm quan trọng trong adapter, runner, testcase builder và scorecard.

## Map nhanh từ code sang docs

| Cần hiểu | File code chính | Tài liệu |
|---|---|---|
| Cấu trúc package evaluation | `evaluation/*` | `01-overview.md` |
| Dataset/golden single-turn và conversational | `evaluation/datasets/*.json` | `02-datasets-and-goldens.md` |
| Cách runtime RAG được gọi trong lúc đánh giá | `evaluation/adapters/langchain_adapter.py` | `03-runtime-pipeline-and-metrics.md` |
| Cách trace biến thành DeepEval testcase | `evaluation/adapters/trace_to_testcase.py` | `03-runtime-pipeline-and-metrics.md` |
| Metric và scorecard | `evaluation/metrics/*.py` | `03-runtime-pipeline-and-metrics.md` |
| Runner single-turn/conversation | `evaluation/runners/run_single_turn_eval.py`, `run_conversation_eval.py` | `04-running-evaluations-and-reading-reports.md` |
| Runtime profiling không chấm metric | `evaluation/runners/profile_single_turn_runtime.py` | `04-running-evaluations-and-reading-reports.md` |
| Ma trận chọn cấu hình RAG | `evaluation/runners/run_rag_experiment_matrix.py` | `05-rag-experiments-and-current-artifacts.md` |
| Đọc sâu từng hàm evaluation | `evaluation/adapters/*`, `evaluation/runners/*`, `evaluation/metrics/*` | `06-deep-dive-evaluation-code.md` |

## Nguyên tắc khi đọc kết quả

- `mean_score` và `pass_rate` là hai đại lượng khác nhau. Một tập có điểm trung bình cao vẫn có thể có nhiều case rơi dưới threshold.
- `case_composite_statistics.pass_rate` không giống `strict_all_metric_threshold_pass_rate`.
  - `case pass`: trung bình ba metric của từng case đạt `0.70`.
  - `strict all metric pass`: cả ba metric của cùng case đều vượt threshold riêng.
- `Contextual Recall`, `Faithfulness`, `Answer Relevancy` là metric single-turn.
- `Knowledge Retention` là metric conversational.
- Report merged cuối cùng phải vẫn giữ `deepeval_result.test_results` và `traces` khớp nhau; không được chỉ sửa phần tổng hợp phía trên.
