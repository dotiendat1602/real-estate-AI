# Tài liệu hệ thống RAG trong `ai-service`

Tài liệu này mô tả cách hệ thống RAG của `real-estate-AI/ai-service` hoạt động dựa trên code hiện tại. Hệ thống dùng FastAPI, LangChain, pgvector/PostgreSQL, HuggingFace embeddings và ChatOpenAI.

Các file tài liệu:

- [01-overview.md](01-overview.md): kiến trúc tổng quan, dữ liệu, cấu hình chính.
- [02-chat-runtime-flow.md](02-chat-runtime-flow.md): luồng `/api/chat`, listing RAG, planning RAG, citations, history, token usage.
- [03-ingestion-storage-flow.md](03-ingestion-storage-flow.md): luồng ingest bài đăng và tài liệu quy hoạch vào pgvector.
- [04-function-catalog.md](04-function-catalog.md): catalog các hàm/class chính và mục đích của từng hàm theo module.
- [05-deep-dive-chat-chain.md](05-deep-dive-chat-chain.md): giải thích sâu các hàm dài trong `app/api/chat.py` và `app/rag/chain.py`.
- [06-deep-dive-ingestion-chunking.md](06-deep-dive-ingestion-chunking.md): giải thích sâu listing/planning ingestion, OCR, chunking modes, metadata và ảnh hưởng đến retrieval.

Tài liệu đánh giá offline được tách riêng tại [../evaluation/README.md](../evaluation/README.md). Phần đó giải thích dataset, golden, trace, metric, scorecard, runner và các report benchmark thay vì runtime phục vụ request thật.

## Bức tranh tổng thể

RAG trong service có 2 miền dữ liệu:

1. **Listing RAG**: hỏi đáp/tìm kiếm bất động sản từ bài đăng đã ingest hoặc từ fallback SQL trực tiếp sang bảng backend.
2. **Planning RAG**: hoi dap tai lieu quy hoach/ke hoach su dung dat. Nhanh nay nhan dien intent, uu tien dung ho so qua `dossierCode` khi suy ra duoc quan+nam, xep hang bang vector retrieval + RRF tren query candidates toi gian, roi compact context truoc khi dua vao LLM.

Luồng runtime chung:

```text
User
  -> FastAPI /api/chat
  -> lấy/tạo chat session + history
  -> build_retrieval_query()
  -> build_llm()
  -> extract_filters_from_query_with_usage()
  -> chọn listing mode hoặc planning mode
  -> retriever lấy Document từ pgvector/SQL
  -> RagChain.prepare_docs_for_context()
  -> prompt + ChatOpenAI
  -> postprocess_answer()
  -> citations + token usage + lưu history
  -> ChatResponse
```

## Các thành phần code chính

| Thành phần | File | Vai trò |
|---|---|---|
| FastAPI app/startup | `app/main.py` | Load `.env`, đăng ký router, preload vector stores. |
| Chat API | `app/api/chat.py` | Entry point `/api/chat`, chọn mode listing/planning, điều phối retrieval và generation. |
| Listing ingest API | `app/api/ingest.py` | Chunk bài đăng, add/update/delete embedding cho listing. |
| Planning ingest API | `app/api/planning.py` | Ingest tài liệu quy hoạch và xem chunks đã ingest. Câu hỏi AI về quy hoạch đi qua `/api/chat`. |
| RAG chain | `app/rag/chain.py` + helper modules | `chain.py` chi orchestration; context/query/postprocess da tach sang `context_preparation.py`, `query_rewrite.py`, `answer_processing.py`. |
| Retriever | `app/rag/retriever.py` | Tạo PGVector store, metadata filter, hybrid/vector retriever, lexical SQL search. |
| Vector resources | `app/rag/resources.py` | Cache embeddings và vector store cho listing/planning collection. |
| Embeddings | `app/rag/embedder.py` | HuggingFace embeddings, E5 query/passage prefix, local cache snapshot. |
| LLM | `app/rag/llm.py` | Tạo ChatOpenAI theo env. |
| Prompt | `app/rag/prompt.py` | System prompt và template bắt buộc chỉ dùng context. |
| Planning ingestion | `app/planning/ingestion.py` | Download PDF/TXT, extract/OCR, chunk theo cấu trúc, metadata, enrich text. |
| Planning retrieval/selection | `app/planning/ranker.py`, `selector.py`, `query_builders.py`, `features.py` | Query expansion, term extraction, metadata/quality gates và selection text/table. |
| DB | `app/db/pgvector.py`, `app/db/models.py` | Async SQLAlchemy session và ORM cho chat/session/embedding model cũ. |

## Các collection pgvector

Code sử dụng bảng LangChain Postgres mặc định:

- `langchain_pg_collection`
- `langchain_pg_embedding`

Collection listing mặc định:

```text
post_embeddings__multilingual_e5_base_ghr1
```

Collection planning mặc định:

```text
planning_documents__multilingual_e5_base_ghr1__planning_hierarchical_parent_context
```

Tên collection listing lấy từ `PGVECTOR_COLLECTION`; planning lấy từ `PGVECTOR_COLLECTION_PLANNING`.
