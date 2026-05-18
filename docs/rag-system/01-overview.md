# 01. Tổng quan kiến trúc RAG

## Stack kỹ thuật

- **FastAPI**: API runtime trong `app/main.py`.
- **LangChain**: `Document`, retriever interface, prompt chain, ChatOpenAI wrapper.
- **pgvector/PostgreSQL**: lưu vectors qua `langchain-postgres` (`PGVector`) trong bảng `langchain_pg_embedding`.
- **HuggingFaceEmbeddings**: model mặc định `intfloat/multilingual-e5-base`, normalize vectors để similarity ổn định.
- **ChatOpenAI**: model mặc định `gpt-4o-mini`, temperature mặc định `0.2`.

## Hai nhánh RAG

### 1. Listing RAG

Mục đích: trả lời/tìm kiếm bài đăng bất động sản.

Nguồn context:

- Chunks embedding đã ingest từ `/api/ingest/posts`.
- Fallback SQL trực tiếp từ bảng backend (`posts`, `properties`, `districts`, `wards`, `categories`, `amenities`) khi query có intent listing rõ hoặc kết quả vector quá ít.

Luồng chính:

```text
/api/chat
  -> extract listing filters bằng LLM
  -> build_retriever(vs, filters)
  -> ListingFallbackRetriever
  -> RagChain.run()
```

### 2. Planning RAG

Mục đích: hỏi đáp tài liệu quy hoạch/kế hoạch sử dụng đất.

Nguồn context:

- Tài liệu PDF/TXT được ingest qua `/api/planning/ingest-documents`.
- Metadata quan trọng: `documentScope=planning`, `planningDocumentId`, `dossierCode`, `district`, `districtCanonical`, `districtRaw`, `planYear`, `chunkType`, `sectionHeading`, `hierarchyPath`, `globalChunkIndex`, `pageNumber`, `sourceLocator`.

Luồng chính:

```text
/api/chat
  -> _has_planning_intent() hoặc request có planningContexts
  -> initialize_planning_vector_store()
  -> _retrieve_planning_docs_for_nl_query()
  -> vector RRF/select/rebalance/compact planning docs
  -> RagChain với _StaticDocumentsRetriever(planning_docs)
```

Planning RAG khác listing RAG ở chỗ retrieval không chỉ vector similarity. Nó còn:

- Sinh nhiều query ứng viên theo intent.
- Khi suy ra được quận/huyện và năm, ưu tiên filter chính xác theo `dossierCode`; sau đó mới thử strict district+year, district-only, year-only, broad.
- Xếp hạng pool bằng Reciprocal Rank Fusion từ vector search; các rule còn lại chủ yếu là filter metadata, bỏ mục lục/heading yếu và cân bằng text/table trước khi đưa vào prompt.
- Compact context để giảm noise trước khi đưa vào prompt.

## Startup và cache resource

`app/main.py` gọi `chat.initialize_vector_store()` ở startup nếu `AI_PRELOAD_VECTOR_STORES=true`.

`app/rag/resources.py` cache:

- `_embeddings`: một instance embedding dùng chung.
- `_vector_stores`: dict cache vector store theo collection.
- lock async để tránh nhiều request cùng khởi tạo trùng.

Điểm quan trọng: listing và planning dùng cùng embedding model nhưng khác collection.

## Prompt contract

`app/rag/prompt.py` định nghĩa system prompt với các ràng buộc chính:

- Chỉ dùng `CONTEXT`.
- Nếu có `EVIDENCE CONTRACT`, planning/numeric claims chỉ dùng facts được label.
- Trả lời theo ngôn ngữ người dùng.
- Không bịa thuộc tính thiếu.
- Planning fact phải bám theo tài liệu được truy xuất.
- Numeric planning chỉ trả số khi số đó xuất hiện đúng trong context.

## Cấu hình env quan trọng

| Env | Ý nghĩa |
|---|---|
| `DATABASE_URL` | URL DB async SQLAlchemy. Bắt buộc. |
| `PGVECTOR_URL` | URL PGVector cho LangChain. Bắt buộc khi dùng vector store. |
| `PGVECTOR_COLLECTION` | Collection listing. |
| `PGVECTOR_COLLECTION_PLANNING` | Collection planning. |
| `EMBED_MODEL` | Model embedding, mặc định `intfloat/multilingual-e5-base`. |
| `EMBED_DEVICE` | `cpu` hoặc `cuda`. |
| `EMBED_QUERY_PREFIX`, `EMBED_DOCUMENT_PREFIX` | Prefix cho query/document embedding. |
| `OPENAI_MODEL` | Chat model, mặc định `gpt-4o-mini`. |
| `OPENAI_TEMPERATURE` | Temperature, mặc định `0.2`. |
| `RAG_RETRIEVER_MODE` | `hybrid` hoặc vector similarity thường. |
| `RAG_LEXICAL_DISABLE` | Tắt lexical retrieval SQL. |
| `RAG_HYBRID_LEXICAL_SPARSE_ONLY` | Chỉ gọi lexical trong hybrid retriever khi semantic trả thiếu `k` docs; mặc định bật. |
| `RAG_CONTEXT_MAX_DOCS` | Số docs tối đa đưa vào context trong `RagChain`. |
| `RAG_CONTEXT_MAX_CHARS_PER_DOC` | Số ký tự tối đa mỗi doc context. |
| `CHAT_PLANNING_RETRIEVAL_TIMEOUT_SECONDS` | Timeout retrieval planning trong `/api/chat`. |
| `PLANNING_CHUNKING_MODE` | Chế độ chunk planning, mặc định hierarchical parent context. |
| `PLANNING_PDF_OCR_FALLBACK_ENABLED` | Bật OCR fallback cho PDF scan/low quality. |
| `PLANNING_INGEST_SOFT_TIMEOUT_SECONDS` | Soft timeout khi ingest planning. |
