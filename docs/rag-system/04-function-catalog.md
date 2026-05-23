# 04. Catalog hàm và class chính

File này liệt kê các hàm/class phục vụ RAG và mục đích của chúng theo module. Một số module có rất nhiều helper regex/postprocess; phần này gom theo vai trò để dễ đọc nhưng vẫn nêu rõ hàm chính.

## `app/main.py`

| Hàm/class | Mục đích |
|---|---|
| `_env_bool(name, default)` | Parse biến môi trường boolean. |
| `lifespan(app)` | Startup/shutdown FastAPI; preload vector stores nếu `AI_PRELOAD_VECTOR_STORES=true`. |
| `create_app()` | Tạo FastAPI app và include health/chat/ingest/planning routers. |

## `app/rag/resources.py`

| Hàm/class | Mục đích |
|---|---|
| `planning_collection_name()` | Resolve tên collection planning từ `PGVECTOR_COLLECTION_PLANNING` hoặc default. |
| `_store_key(collection_name)` | Chuẩn hóa key cache vector store. |
| `get_embeddings()` | Lazy-load và cache embeddings dùng chung. |
| `initialize_vector_store(collection_name)` | Khởi tạo PGVector async cho collection, dùng lock để tránh race. |
| `initialize_listing_vector_store()` | Khởi tạo collection listing mặc định. |
| `initialize_planning_vector_store()` | Khởi tạo collection planning. |

## `app/rag/embedder.py`

| Hàm/class | Mục đích |
|---|---|
| `_resolve_local_hf_snapshot(model_name)` | Tìm snapshot HuggingFace local cache để tránh gọi network metadata. |
| `PrefixingEmbeddings` | Wrapper thêm prefix `query:`/`passage:` cho E5 embeddings. |
| `PrefixingEmbeddings._prefix()` | Tránh thêm prefix trùng. |
| `embed_documents()` | Embed documents sau khi thêm document prefix. |
| `embed_query()` | Embed query sau khi thêm query prefix. |
| `aembed_documents()` | Async embed documents nếu inner hỗ trợ; fallback sync. |
| `aembed_query()` | Async embed query nếu inner hỗ trợ; fallback sync. |
| `__getattr__()` | Delegate thuộc tính còn lại sang embedding inner. |
| `build_embeddings()` | Tạo `HuggingFaceEmbeddings`, normalize vectors, tự thêm E5 prefixes. |

## `app/rag/llm.py`

| Hàm/class | Mục đích |
|---|---|
| `build_llm()` | Tạo `ChatOpenAI` theo `OPENAI_MODEL`, `OPENAI_TEMPERATURE`, timeout và retry. |

## `app/rag/prompt.py`

| Thành phần | Mục đích |
|---|---|
| `SYSTEM` | System prompt quy định chỉ dùng context, cách trả lời listing/planning, numeric rules, không bịa thông tin. |
| `prompt` | `ChatPromptTemplate` gồm system, history và human message chứa question/context. |

## `app/rag/retriever.py`

| Hàm/class | Mục đích |
|---|---|
| `_env_float(name, default)` | Parse env float an toàn. |
| `_env_bool(name, default)` | Parse env boolean an toàn. |
| `_rrf_score(rank, k)` | Tính Reciprocal Rank Fusion score. |
| `_doc_identity(doc)` | Tạo key dedupe/fusion cho listing/planning docs. |
| `build_pgvector_store(embeddings, collection_name)` | Tạo LangChain `PGVector` async store. |
| `build_metadata_filter(filters)` | Chuyển filters domain sang PGVector JSONB filter (`$and`, `$gte`, `$lte`, `$in`). |
| `_merge_pgvector_filters(*filters)` | Gộp nhiều PGVector filter thành một filter. |
| `HybridRetriever` | Retriever kết hợp semantic vector search và lexical SQL search bằng RRF. |
| `HybridRetriever._semantic_search(query, k)` | Gọi PGVector similarity search, ưu tiên score nếu backend hỗ trợ. |
| `HybridRetriever.ainvoke_with_scores(query)` | Chạy semantic + lexical song song, fuse score, trả docs + scores. |
| `HybridRetriever.ainvoke(query)` | Trả docs top k, interface tương thích LangChain retriever. |
| `build_retriever(vs, k, filters, base_filter, mode_override)` | Chọn hybrid hoặc vector retriever, apply metadata filters. |
| `_normalize_text(value)` | Normalize text cho lexical matching. |
| `_extract_lexical_terms(query, max_terms)` | Tách token/phrase/numeric terms cho SQL lexical search. |
| `_sql_normalized_text_expr(expr, compact)` | Sinh SQL expression normalize tiếng Việt/compact text. |
| `_coerce_number(value)` | Ép số cho numeric metadata filters. |
| `_merge_raw_filters(base_filter, filters)` | Gộp raw filters cho lexical SQL, đổi `chunkTypes` thành `chunkType $in`. |
| `_append_in_clause(...)` | Thêm SQL `IN` clause cho metadata JSONB. |
| `_build_metadata_where_clauses(filters, params)` | Sinh SQL where clauses cho metadata filters. |
| `lexical_search_documents(...)` | SQL lexical retrieval trên `document`, `title`, `dossierCode`, `sourceLocator` với metadata-aware filters, CTE normalize một lần và timeout. |

## `app/rag/filter_extractor.py`

| Hàm/class | Mục đích |
|---|---|
| `PropertyFilters` | Schema filter listing mà LLM cần trích xuất. |
| `_normalize_text(value)` | Normalize text để kiểm tra address/district. |
| `_sanitize_text_value(value)` | Chuẩn hóa string optional. |
| `_looks_like_address_fragment(value)` | Chặn city/district nếu LLM trả nhầm địa chỉ cụ thể. |
| `_to_positive_int(value)` | Parse số nguyên dương. |
| `_sanitize_extracted_filters(filters)` | Làm sạch filter LLM, validate post type, bedrooms, min/max. |
| `extract_filters_from_query_with_usage(question, llm)` | Prompt LLM parse filters JSON, trả filters và token usage. |

## `app/rag/listing_fallback.py`

| Hàm/class | Mục đích |
|---|---|
| `_contains_any(value, needles)` | Helper kiểm tra keyword. |
| `_clean(value)` | Convert/strip value sang string. |
| `_format_price(value)` | Format giá VNĐ thành tỷ/triệu/VNĐ. |
| `_append_text_filter(...)` | Thêm SQL LIKE clause cho city/district/text fields. |
| `_has_structured_listing_intent(query, filters)` | Xác định có nên dùng fallback SQL listing. |
| `search_listing_documents(query, filters, k, _disable_city_hint)` | Query trực tiếp bảng backend để tạo `Document` listing giàu metadata. |
| `ListingFallbackRetriever` | Wrapper retriever: lấy vector docs trước, fallback SQL nếu cần, merge dedupe. |
| `ListingFallbackRetriever.ainvoke(query)` | Thực thi primary retrieval + fallback SQL, ưu tiên fallback khi query có structured intent. |

## `app/rag/message_history.py`

| Hàm/class | Mục đích |
|---|---|
| `MessageHistoryManager` | Quản lý session và messages trong DB. |
| `get_or_create_session(user_id, session_id)` | Validate session cũ thuộc user hoặc tạo session mới. |
| `get_messages(session_id, limit)` | Lấy messages gần nhất, convert sang LangChain `HumanMessage`/`AIMessage`. |
| `add_message(session_id, role, content)` | Lưu message mới. |

## `app/rag/llm_usage.py`

| Hàm/class | Mục đích |
|---|---|
| `message_content_to_text(content)` | Convert content string/list/dict thành text. |
| `extract_token_usage(message)` | Trích token usage từ `usage_metadata` hoặc `response_metadata`. |
| `sum_token_usage(usages)` | Cộng input/output/total tokens. |
| `_dict_get(value, key)` | Safe get cho dict. |
| `_coerce_int(value)` | Parse int optional. |

## RAG runtime modules

Sau refactor, `app/rag/chain.py` chi giu orchestration va public import surface. Logic dai duoc tach sang cac module sau:

- `app/rag/query_rewrite.py`: rewrite retrieval query cho follow-up.
- `app/rag/query_intents.py`: nhan dien intent listing/planning dung chung.
- `app/rag/context_preparation.py`: dedupe, giu thu tu retriever/planning rank, compact context va build citation.
- `app/rag/answer_processing.py`: postprocess answer va detect language.
- `app/rag/listing_processing.py`: helper suitability/listing dung chung cho context va answer.
- `app/utils/text.py`: normalize/sanitize text dung chung, gom ca sua mojibake va normalize tieng Viet cho matching; `app/rag/text_utils.py` chi re-export de giu import cu.

### Data model và retrieval query

| Hàm/class | Mục đích |
|---|---|
| `ChatResult` | Dataclass kết quả RAG gồm answer, citations, token usage, timings. |
| `_normalize_text(text)` | Normalize text cho matching/postprocess. |
| `sanitize_llm_text(text, max_len)` | Làm sạch control chars/surrogates trước khi gửi LLM. |
| `_history_message_role(message)` | Resolve role từ dict/LangChain message. |
| `_history_message_content(message)` | Resolve text content từ history item. |
| `_recent_user_messages(history, max_messages)` | Lấy các user messages gần nhất. |
| `_looks_follow_up_question(question)` | Nhận diện follow-up thiếu anchor. |
| `_is_anchor_rich_message(message)` | Kiểm tra message trước có đủ anchor listing/location/số liệu. |
| `build_retrieval_query(question, history)` | Rewrite query follow-up bằng cách ghép anchor trước đó. |

### Chuan bi context va citations

| Ham/class | Muc dich |
|---|---|
| `detect_lang(text)` | Xac dinh ngon ngu tra loi. |
| `_dedupe_repeated_blocks(text)` | Loai text block lap. |
| `_split_long_line(line)` | Chia dong qua dai. |
| `_compact_doc_content(question, content, max_chars, max_lines)` | Rut gon document tho theo thu tu tu nhien sau khi dedupe. |
| `_doc_identity(doc)` | Key dedupe context. |
| `_is_planning_context_doc(doc)` | Kiem tra doc thuoc planning. |
| `_listing_raw_evidence(question, doc)` | Tao raw excerpt ngan cho structured listing context. |
| `prepare_docs_for_context(question, docs, max_docs, max_chars_per_doc)` | Dedupe, giu thu tu retriever/planning rank, compact, chuan bi docs cho prompt. |
| `build_citations(docs)` | Build citations tu docs context. |
| `format_docs(docs)` | Join page_content da sanitize thanh context string. |

## `app/rag/listing_context.py`

| Hàm/class | Mục đích |
|---|---|
| `build_structured_listing_context(question, doc, raw_evidence_builder)` | Tạo context listing có cấu trúc từ metadata + snippet. |
| `merge_context_snippets(primary, secondary, max_chars)` | Gộp structured context với raw excerpt. |

### LLM chain

| Hàm/class | Mục đích |
|---|---|
| `RagChain` | Orchestrator: retrieve docs, prepare context, prompt, call LLM, postprocess, return citations/timings. |
| `RagChain.__init__(llm, retriever)` | Tạo prompt chain dùng question/context/history/language. |
| `RagChain.run(question, history, extra_context)` | Chạy full RAG generation cho một câu hỏi. |

### Postprocess answer

`app/rag/answer_processing.py` chi giu hau xu ly chung, khong con rewrite hep theo tung mau cau hoi planning/listing.

| Nhom ham | Muc dich |
|---|---|
| `_strip_unrequested_price_lines` | Bo dong gia khi user hoi suitability nhung khong hoi gia. |
| `_strip_unrequested_contact_lines` | Bo thong tin lien he neu user khong hoi lien he. |
| `_strip_spaciousness_extra_lines` | Voi cau hoi khong gian rong/thoang, bo vai dong tien ich xung quanh khong truc tiep lien quan. |
| `postprocess_answer(question, answer)` | Loai generic closing lines va goi cac bo loc chung o tren. |

## `app/rag/static_retriever.py`

| Ham/class | Muc dich |
|---|---|
| `StaticDocumentsRetriever` | Retriever nho tra lai danh sach docs co san; dung khi planning retrieval da chon docs rieng truoc khi goi `RagChain`. |

## `app/rag/citation_utils.py`

| Ham/class | Muc dich |
|---|---|
| `rerank_citations(citations)` | Sap xep citation theo planning document va score co san tu retriever, khong cham diem keyword overlap. |
| `build_planning_citations(docs)` | Build citation giau metadata cho planning docs. |

## `app/api/chat.py`

### API/session/runtime

| Hàm/class | Mục đích |
|---|---|
| `_with_timeout(coro, seconds, label, fallback)` | Bọc coroutine với timeout, trả fallback nếu quá thời gian. |
| `initialize_vector_store()` | Startup init listing + planning vector store. |
| `ChatRequest` | Schema request `/api/chat`: `userId`, `sessionId`, `message`. |
| `ChatResponse` | Schema response `/api/chat`. |
| `chat(req, db)` | Entry point full runtime RAG. |

## `app/rag/planning_retrieval.py`

### Intent và extraction planning

| Hàm/class | Mục đích |
|---|---|
| `_strip_accents`, `_normalize_nl` | Alias import từ `app/utils/text.py` để bỏ dấu/sửa mojibake/normalize text cho matching. |
| `_has_planning_intent(message)` | Nhận diện câu hỏi quy hoạch. |
| `_planning_query_profile(message)` | Lấy profile intent planning. |
| `_extract_district_from_message(message)` | Extract quận/huyện từ message bằng alias chuẩn dùng chung với planning metadata. |
| `_extract_plan_year_from_message(message)` | Extract năm 20xx từ message. |
| `_history_role(item)`, `_history_content(item)` | Đọc role/content từ history. |
| `_extract_district_from_history(history_messages)` | Lấy district từ history nếu current query thiếu. |
| `_extract_plan_year_from_history(history_messages)` | Lấy plan year từ history nếu current query thiếu. |
| `_is_planning_fact_query(message)` | Nhận diện câu hỏi fact/numeric planning. |

### Matching/filtering planning

| Hàm/class | Mục đích |
|---|---|
| `_district_tokens(district)` | Tách token district. |
| `_district_aliases_for_matching(district)` | Sinh alias district để match content/metadata. |
| `_doc_matches_district(doc, district)` | Kiểm tra doc có thuộc district không. |
| `_doc_matches_plan_year(doc, plan_year)` | Kiểm tra doc có đúng năm kế hoạch không. |
| `_planning_query_candidates(...)` | Wrapper sinh query candidates. |
| `_select_ranked_planning_docs(...)` | Wrapper selector cho docs đã rank. |
| `_rrf_score(rank, k)` | RRF score local cho planning retrieval. |
| `_planning_scope_min_docs(scope, final_k, fact_query)` | Tính số docs tối thiểu để accept strict/district scope. |

### Planning compact/cache

| Hàm/class | Mục đích |
|---|---|
| `_compact_planning_doc(...)` | Compact một planning doc. |
| `_compact_planning_docs(...)` | Compact danh sách planning docs. |
| `_planning_query_cache_key(...)` | Tạo cache key cho query planning. |
| `_planning_query_cache_get(key)` | Lấy cache vector result. |
| `_planning_query_cache_put(key, value)` | Lưu cache result. |
| `_resolve_planning_retrieval_context(...)` | Gom district/year/fact_query/final_k/probe_k/query candidates cho một request. |
| `_planning_base_filter_candidates(...)` | Sinh base filters từ hẹp tới rộng, có dedupe. |
| `_rank_planning_docs_for_filter(...)` | Retrieve theo query candidates và xếp hạng bằng RRF từ vector rank. |
| `_planning_scope_pools(...)` | Chia ranked docs thành strict/district/broad pools. |
| `_retrieve_planning_docs_for_nl_query(...)` | Pipeline retrieval planning NL: query expansion, exact dossier filter khi biết district+năm, filter scopes, vector retrieval, RRF rank, compact, fallback. |

## `app/api/ingest.py`

| Hàm/class | Mục đích |
|---|---|
| `_int_env(name, default)` | Parse positive int env. |
| `_get_splitter()` | Lazy init listing splitter theo env chunk size/overlap. |
| `initialize_vector_store()` | Init listing vector store. |
| `IngestPost`, `IngestRequest`, `UpdatePostRequest` | Pydantic schemas cho ingest/update listing; update lay `post_id` tu path. |
| `list_ingested_posts(limit, offset)` | List post đã ingest và số chunks. |
| `get_ingested_post(post_id, limit, offset)` | Xem chunks/metadata của một post. |
| `ingest_posts(req)` | Split content bài đăng, add documents vào listing vector store. |
| `update_post_embeddings(post_id, req)` | Delete chunks cũ rồi ingest chunks mới. |
| `delete_post_embeddings(post_id)` | Delete toàn bộ chunks của một post. |

## `app/api/planning.py`

| Hàm/class | Mục đích |
|---|---|
| `initialize_vector_store()` | Init planning vector store. |
| `PlanningIngestDocument`, `PlanningIngestRequest` | Schemas ingest tài liệu quy hoạch. |
| `PlanningExplainSummary`, `PlanningExplainDocument`, `PlanningExplainRequest`, `PlanningExplainResponse` | Schemas endpoint explain planning. |
| `_planning_collection_name()` | Lấy tên collection planning. |
| `list_ingested_planning_documents(limit, offset)` | List tài liệu planning đã ingest và thống kê chunks. |
| `get_ingested_planning_document_chunks(planning_document_id, limit, offset)` | Xem chunks của một tài liệu planning. |
| `ingest_planning_documents(req)` | Orchestrate replace/skip/build docs/add embeddings cho planning documents. |
| `explain_planning(req)` | Dùng summary property + planning retriever + RagChain để giải thích quy hoạch. |

## `app/planning/ingestion_config.py`

| Ham/class | Muc dich |
|---|---|
| `resolve_planning_chunking_mode`, `is_hierarchical_chunking_mode` | Doc config chunking mode cho planning ingest. |
| `resolve_http_verify`, `resolve_ssl_allow_insecure_fallback` | Config TLS khi download tai lieu planning. |
| `resolve_pdf_ocr_fallback_enabled`, `resolve_pdf_ocr_max_pages`, `resolve_pdf_ocr_render_scale` | Config OCR fallback cho PDF. |
| `resolve_pdf_text_quality_min_score`, `resolve_pdf_force_ocr_on_low_quality` | Config danh gia chat luong text layer PDF. |
| `resolve_ingest_soft_timeout_seconds`, `resolve_ingest_require_full`, `resolve_ocr_progress_every_pages` | Config timeout/full ingest/OCR progress. |

## `app/planning/ingestion.py`

### Config/extract/OCR

| Hàm/class | Mục đích |
|---|---|
| `_get_splitter()` | Splitter fallback cho planning fixed chunks. |
| `_resolve_*` aliases | Import tu `app/planning/ingestion_config.py` de giu code ingest chinh tap trung vao extract/OCR/chunking. |
| `_extract_rapidocr_lines(ocr_output)` | Chuẩn hóa output RapidOCR thành lines. |
| `_ocr_lines_with_stripes(ocr_engine, arr)` | OCR fallback chia ảnh thành stripes nếu OCR trang đầy đủ fail. |
| `PlanningIngestPayload` | Dataclass payload chuẩn hóa từ API request. |
| `_extract_pdf_text_and_pages(binary)` | Extract text từng page bằng pypdf. |
| `_clean_text(text)` | Làm sạch whitespace/null. |
| `_normalize_for_match(text)` | Normalize để match line/page. |
| `_estimate_pdf_text_quality(page_texts, cleaned_text)` | Tính quality score text layer PDF. |
| `_build_page_line_maps(page_texts)` | Tạo map page -> lines/normLines/cursor. |

### Chunk construction

| Hàm/class | Mục đích |
|---|---|
| `_is_table_chunk(content)` | Heuristic xác định chunk dạng bảng. |
| `_locate_chunk_line_span(chunk_text, page_maps, preferred_page)` | Tìm page/line_start/line_end cho chunk. |
| `_dedupe_structural_chunks(items)` | Loại structural chunks trùng. |
| `_looks_like_continuation_lead(line)`, `_ends_with_continuation_signal(text)` | Nhận diện đoạn tiếp nối. |
| `_is_structurally_weak_chunk(item)` | Loại chunk quá ngắn/yếu. |
| `_merge_continuation_chunks(items)` | Gộp chunks liên tiếp cùng section khi đoạn trước/sau nối nhau. |
| `_ocr_structural_chunks(...)` | Render PDF pages, OCR, tạo structural chunks. |
| `_hierarchy_line_level(line)`, `_is_new_item_marker(line)`, `_is_projectish_line(line)`, `_is_project_table_header_line(line)` | Heuristics nhận diện cấu trúc dòng/tables/project rows. |
| `_hierarchy_contextual_chunks(page_maps)` | Tạo chunks theo hierarchy từ page lines. |
| `_hierarchy_signature(path, page_number)` | Tạo signature cho hierarchy chunk. |
| `_hierarchical_chunk_content(...)` | Build content chunk có parent/heading context. |
| `_append_hierarchical_parent_chunks(...)` | Thêm parent chunks vào danh sách hierarchical. |
| `build_planning_hierarchical_chunks(text, page_maps, mode)` | Build chunks theo mode hierarchical. |
| `_flatten_page_entries(page_maps)` | Flatten page lines để xử lý fallback/archetype. |
| `_detect_document_archetype(text, page_maps)` | Nhận diện kiểu tài liệu để chọn chunk strategy. |
| `_decision_structural_chunks(page_maps, text)` | Chunk riêng cho quyết định/phụ lục. |
| `_section_table_fallback_chunks(text, page_maps)` | Chunk fallback theo section/table blocks. |
| `_fallback_structural_chunks(text, page_maps)` | Chunk fallback tổng quát. |
| `_build_source_locator(page_number, line_start, line_end)` | Tạo locator dạng page/line. |
| `_looks_like_table_line(line)`, `_is_section_heading_line(line)` | Heuristic table/heading. |
| `_split_text_sections(block)` | Split text block theo heading. |
| `_chunk_table_block_rows(block, rows_per_chunk)` | Cắt bảng theo số rows. |
| `_split_text_and_table_blocks(text)` | Tách text blocks và table blocks. |
| `_enrich_for_embedding(content, metadata)` | Thêm metadata descriptor trước content để embedding dễ retrieve. |
| `fetch_document_bytes(source_url, timeout)` | Download tài liệu bằng httpx, có xử lý SSL fallback. |
| `build_planning_documents(payload)` | Full pipeline download/extract/OCR/chunk/metadata/enrich -> `Document[]`. |

## `app/planning/metadata.py`

| Hàm/class | Mục đích |
|---|---|
| `PLANNING_DISTRICT_ALIASES` | Nguồn alias địa bàn chuẩn dùng chung cho canonicalize và extract district runtime. |
| `_strip_accents(text)` | Bỏ dấu để match district. |
| `_normalize_text(text)` | Normalize district/title/dossier. |
| `_compact_token(text)` | Compact token không dấu, không ký tự đặc biệt. |
| `_match_from_text(text)` | Match district canonical từ alias hoặc compact token. |
| `canonicalize_planning_district(value, title, dossier_code)` | Chuẩn hóa district từ value/title/dossier code. |

## `app/planning/docs.py`

| Hàm/class | Mục đích |
|---|---|
| `_normalize_text(text)` | Normalize identity text. |
| `dedupe_planning_docs(docs)` | Dedupe planning docs theo document id/chunk/page/content prefix. |
| `planning_doc_identity(doc)` | Tạo identity đầy đủ cho planning doc. |
| `planning_chunk_type(doc)` | Lấy `chunkType` lowercase. |

## `app/planning/query_builders.py`

| Hàm/class | Mục đích |
|---|---|
| `_normalize_text(text)` | Normalize query. |
| `district_code_fragment(district)` | Tạo fragment mã quận, ví dụ phục vụ query dossier `HN-...-KH2025`. |
| `planning_query_candidates(message, district, plan_year, max_query_candidates)` | Sinh candidates t?i gi?n: c?u h?i g?c, bi?n th? c? district/year v? dossier code khi ?? th?ng tin. |
| `planning_query_candidates(message, district, plan_year, max_query_candidates)` | Sinh candidates toi gian: cau hoi goc, bien the co district/year va dossier code khi du thong tin. |
## `app/planning/ranker.py`

| Hàm/class | Mục đích |
|---|---|
| `planning_query_terms(message, max_terms)` | Tách terms quan trọng, bỏ stopwords. |

## `app/planning/selector.py`

| Hàm/class | Mục đích |
|---|---|
| `select_ranked_planning_docs(...)` | Ch?n docs t? pool ?? rank b?ng vector RRF, b? TOC/heading y?u v? ??y l?ch qu?n/n?m xu?ng sau. |
| `select_ranked_planning_docs(...)` | Chon docs tu pool da rank bang vector RRF, bo TOC/heading yeu va day lech quan/nam xuong sau. |
## `app/planning/features.py`

| Nhóm hàm | Mục đích |
|---|---|
| Normalize/metadata: `is_planning_metadata_line`, `strip_planning_metadata_lines`, `planning_doc_haystack` | Chuẩn hóa content/metadata để matching/filtering không bị noise descriptor. |
| Chunk type/quality: `planning_is_heading_or_incomplete_chunk`, `planning_is_toc_like_chunk` | Nhận diện chunk heading/TOC. |
| Numeric/entity evidence: `planning_named_entity_hits`, `planning_has_explicit_project_row` | Nhận diện dòng dự án và entity rõ ràng. |
| Admin overview: `planning_admin_unit_header_hits`, `planning_has_direct_admin_unit_count_phrase`, `planning_has_direct_natural_area_phrase` | Tìm evidence diện tích tự nhiên và đơn vị hành chính. |
| Planning intent evidence: `planning_explanatory_evidence_hits`, `planning_registered_plan_evidence_hits` | Tìm evidence giải thích/danh mục đăng ký. |
| Land change: `planning_land_change_label_hits` | Tìm evidence biến động đất/chỉ tiêu đất chưa sử dụng. |

## `app/planning/profiles.py`

| Hàm/class | Mục đích |
|---|---|
| `normalize_planning_text(text)` | Normalize text planning. |
| `strip_accents(text)` | Bỏ dấu dùng chung cho planning modules. |
| `planning_focus_phrases(message)` | Extract focus phrases từ message. |
| `PlanningQueryProfile` | Dataclass/structure chứa các flags intent planning. |
| `build_planning_query_profile(message, planning_intent)` | Build profile intent tổng hợp cho planning ranker/selector/query builder. |

## `app/rag/planning_pipeline.py`

| Hàm/class | Mục đích |
|---|---|
| `choose_better_planning_fallback(current_docs, candidate_docs)` | Chọn fallback giàu context hơn giữa các scope/base filters, giữ thứ tự vector-ranked đã có. |

## `app/db/pgvector.py`

| Hàm/class | Mục đích |
|---|---|
| `engine` | Async SQLAlchemy engine với search path `ai,public`. |
| `AsyncSessionLocal` | Factory session async. |
| `get_db()` | FastAPI dependency yield `AsyncSession`. |

## `app/db/models.py`

| Hàm/class | Mục đích |
|---|---|
| `Base` | Declarative base SQLAlchemy. |
| `PostEmbedding` | ORM model cũ cho bảng `ai.post_embeddings`; runtime hiện chủ yếu dùng LangChain tables. |
| `ChatSession` | ORM model lưu session chat. |
| `ChatMessage` | ORM model lưu từng message user/assistant. |
