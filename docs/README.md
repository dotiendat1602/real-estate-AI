# Tài liệu `ai-service`

Thư mục này gom tài liệu kỹ thuật của `ai-service` theo ba tuyến chính:

- [code-reading/README.md](code-reading/README.md): bản đồ đọc hiểu code theo file, hàm, dữ liệu vào/ra và playbook debug. Nên bắt đầu ở đây nếu mục tiêu là bảo trì hoặc giải thích code.
- [rag-system/README.md](rag-system/README.md): kiến trúc RAG runtime, ingest, retrieval, prompt, citations và các hàm quan trọng trong `app/`.
- [evaluation/README.md](evaluation/README.md): kiến trúc đánh giá offline, dataset/golden, trace, metric, runner, report và thực nghiệm chọn cấu hình RAG.

Nếu cần hiểu hệ thống theo thứ tự triển khai:

1. Đọc `code-reading/01-service-map.md` để nắm cấu trúc thư mục, entrypoint, API và biến môi trường.
2. Đọc `code-reading/02-chat-runtime-code-walkthrough.md` để lần theo `/api/chat` từ request đến answer/citation.
3. Đọc `code-reading/03-retrieval-and-context-code.md` khi cần debug retrieval, fallback, context hoặc citation.
4. Đọc `code-reading/04-ingestion-code-walkthrough.md` để hiểu dữ liệu listing/planning được ingest và lưu metadata như thế nào.
5. Đọc `code-reading/05-planning-code-walkthrough.md` nếu cần đọc nhánh quy hoạch.
6. Đọc `code-reading/06-evaluation-code-walkthrough.md` để hiểu benchmark và cách đọc trace/report.
7. Đọc `code-reading/07-debugging-playbook.md` khi gặp lỗi thực tế trong chatbot.

Các tài liệu cũ vẫn hữu ích:

1. Đọc `rag-system/01-overview.md` để nắm runtime ở mức kiến trúc.
2. Đọc `rag-system/06-deep-dive-ingestion-chunking.md` khi cần trình bày dữ liệu được ingest và chunk như thế nào.
3. Đọc `evaluation/01-overview.md` để hiểu vì sao hệ thống được đánh giá theo hai nhánh single-turn và conversational.
4. Đọc `evaluation/03-runtime-pipeline-and-metrics.md` khi cần lần theo một test case từ golden -> trace -> DeepEval testcase -> scorecard.
5. Đọc `evaluation/04-running-evaluations-and-reading-reports.md` khi cần chạy smoke/full, tái sử dụng trace, hoặc kiểm tra report.
6. Đọc `evaluation/06-deep-dive-evaluation-code.md` khi cần bám trực tiếp vào các hàm trong adapter/runner để debug.
