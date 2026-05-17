# Tài liệu `ai-service`

Thư mục này gom tài liệu kỹ thuật của `ai-service` theo hai tuyến chính:

- [rag-system/README.md](rag-system/README.md): kiến trúc RAG runtime, ingest, retrieval, prompt, citations và các hàm quan trọng trong `app/`.
- [evaluation/README.md](evaluation/README.md): kiến trúc đánh giá offline, dataset/golden, trace, metric, runner, report và thực nghiệm chọn cấu hình RAG.

Nếu cần hiểu hệ thống theo thứ tự triển khai:

1. Đọc `rag-system/01-overview.md` để nắm runtime.
2. Đọc `evaluation/01-overview.md` để hiểu vì sao hệ thống được đánh giá theo hai nhánh single-turn và conversational.
3. Đọc `evaluation/03-runtime-pipeline-and-metrics.md` khi cần lần theo một test case từ golden -> trace -> DeepEval testcase -> scorecard.
4. Đọc `evaluation/04-running-evaluations-and-reading-reports.md` khi cần chạy smoke/full, tái sử dụng trace, hoặc kiểm tra một report có thể upload lên DeepEval web hay không.
5. Đọc `evaluation/06-deep-dive-evaluation-code.md` khi cần bám trực tiếp vào các hàm trong adapter/runner để debug.
