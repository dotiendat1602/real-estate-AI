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
| `get_initialized_vector_store(collection_name)` | Lấy vector store đã init, lỗi nếu chưa init. |
| `get_initialized_listing_vector_store()` | Lấy listing vector store đã init. |
| `get_initialized_planning_vector_store()` | Lấy planning vector store đã init. |

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
| `lexical_search_documents(...)` | SQL lexical retrieval trên `document`, `title`, `dossierCode`, `sourceLocator` với metadata-aware filters và timeout. |

## `app/rag/filter_extractor.py`

| Hàm/class | Mục đích |
|---|---|
| `PropertyFilters` | Schema filter listing mà LLM cần trích xuất. |
| `_normalize_text(value)` | Normalize text để kiểm tra address/district. |
| `_sanitize_text_value(value)` | Chuẩn hóa string optional. |
| `_looks_like_address_fragment(value)` | Chặn city/district nếu LLM trả nhầm địa chỉ cụ thể. |
| `_to_positive_int(value)` | Parse số nguyên dương. |
| `_sanitize_extracted_filters(filters)` | Làm sạch filter LLM, validate post type, bedrooms, min/max. |
| `extract_filters_from_query(question, llm)` | API đơn giản trả filters. |
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
| `clear_session(session_id)` | Intended clear session; hiện code chỉ execute select rồi commit, chưa delete thực sự. |

## `app/rag/llm_usage.py`

| Hàm/class | Mục đích |
|---|---|
| `message_content_to_text(content)` | Convert content string/list/dict thành text. |
| `extract_token_usage(message)` | Trích token usage từ `usage_metadata` hoặc `response_metadata`. |
| `sum_token_usage(usages)` | Cộng input/output/total tokens. |
| `_dict_get(value, key)` | Safe get cho dict. |
| `_coerce_int(value)` | Parse int optional. |

## `app/rag/chain.py`

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

### Chuẩn bị context và citations

| Hàm/class | Mục đích |
|---|---|
| `detect_lang(text)` | Xác định ngôn ngữ trả lời. |
| `_extract_query_terms(question, max_terms)` | Tách terms từ câu hỏi cho scoring listing. |
| `_build_query_intents(question)` | Nhận diện intent listing như price/area/amenity/suitability. |
| `_dedupe_repeated_blocks(text)` | Loại text block lặp. |
| `_split_long_line(line)` | Chia dòng quá dài. |
| `_line_relevance_score(...)` | Chấm điểm dòng content theo câu hỏi. |
| `_compact_doc_content(question, content, max_chars, max_lines)` | Rút gọn document thô theo dòng liên quan nhất. |
| `_doc_identity(doc)` | Key dedupe context. |
| `_is_planning_context_doc(doc)` | Kiểm tra doc thuộc planning. |
| `_doc_relevance_score(question, doc)` | Chấm relevance listing/non-planning doc. |
| `_build_structured_listing_context(question, doc)` | Tạo context listing có cấu trúc từ metadata + snippet. |
| `_merge_context_snippets(primary, secondary, max_chars)` | Gộp structured context với raw excerpt. |
| `_planning_context_lines(context_text)` | Tách dòng planning context. |
| `_planning_pick_summary_lines(...)` | Chọn dòng planning có marker liên quan. |
| `_planning_pick_focus_phrase_lines(...)` | Chọn dòng có focus phrase trong planning. |
| `_planning_contract_markers(question)` | Xác định marker cần đưa vào evidence contract. |
| `_planning_evidence_source_label(doc)` | Tạo label nguồn cho planning fact. |
| `_build_planning_evidence_contract(question, docs, max_facts)` | Tạo EVIDENCE CONTRACT để khóa numeric/planning claims. |
| `_build_planning_context_summary(question, docs)` | Tạo summary ngắn theo intent planning. |
| `_augment_planning_context_summary(question, compacted, doc)` | Gắn summary vào doc nếu hữu ích. |
| `_apply_global_planning_context_summaries(question, docs)` | Gắn summary tổng vào doc đầu. |
| `prepare_docs_for_context(question, docs, max_docs, max_chars_per_doc)` | Dedupe, select, compact, build evidence contract, chuẩn bị docs cho prompt. |
| `_build_citations(docs)` | Build citations từ docs context. |
| `_format_docs(docs)` | Join page_content đã sanitize thành context string. |

### LLM chain

| Hàm/class | Mục đích |
|---|---|
| `RagChain` | Orchestrator: retrieve docs, prepare context, prompt, call LLM, postprocess, return citations/timings. |
| `RagChain.__init__(llm, retriever)` | Tạo prompt chain dùng question/context/history/language. |
| `RagChain.run(question, history, extra_context)` | Chạy full RAG generation cho một câu hỏi. |

### Postprocess answer

`app/rag/chain.py` có nhiều hàm hậu xử lý để sửa wording, loại thông tin không được hỏi, và chuẩn hóa câu trả lời planning/listing. Các nhóm chính:

| Nhóm hàm | Mục đích |
|---|---|
| `_looks_uncertain_or_no_data`, `_strip_unrequested_price_lines`, `_strip_unrequested_area_suffix` | Loại câu trả lời thiếu dữ liệu hoặc thông tin giá/diện tích không được hỏi. |
| `_extract_named_project_line`, `_extract_flexible_project_line` | Tìm tên dự án trong answer. |
| `_normalize_single_new_project_answer`, `_normalize_project_composition_wording`, `_normalize_gpmb_listing`, `_normalize_hdnd_grouping_answer` | Chuẩn hóa câu trả lời theo intent dự án/quy hoạch. |
| `_normalize_planning_reporting_chain`, `_normalize_post_approval_execution_answer`, `_normalize_article67_listing` | Chỉnh wording cho các câu hỏi quy trình/phê duyệt/Điều 67. |
| `_extract_context_natural_area_ha`, `_extract_context_admin_unit_breakdown`, `_extract_context_admin_unit_count` | Trích fact diện tích tự nhiên/đơn vị hành chính từ context. |
| `_normalize_admin_overview_answer`, `_normalize_city_level_listing_from_context`, `_normalize_article67_answer_from_context` | Dựng answer trực tiếp từ context cho một số fact planning. |
| `_extract_land_change_pair`, `_normalize_land_change_answer`, `_normalize_land_use_variation_recaps` | Xử lý câu hỏi biến động chỉ tiêu đất. |
| `_extract_registered_plan_total_fact`, `_extract_registered_plan_added_fact`, `_extract_registered_plan_resolution_count`, `_extract_registered_plan_resolution_area`, `_normalize_registered_plan_composition_from_context` | Xử lý số liệu danh mục đăng ký/kế hoạch. |
| `_normalize_focus_management_answer`, `_normalize_focus_management_from_context`, `_normalize_public_purpose_composition_ratios`, `_normalize_auction_scope_focus` | Chuẩn hóa các câu hỏi phân tích chuyên biệt. |
| `_extract_price_from_text`, `_extract_area_from_text`, `_extract_bedrooms_from_text`, `_extract_bathrooms_from_text`, `_extract_min_rental_period_from_text`, `_extract_monthly_cashflow_from_text`, `_extract_district_from_text`, `_extract_direction_from_text` | Extract field listing từ text. |
| `_structured_highlights`, `_condense_suitability_answer`, `_normalize_indoor_amenities_answer`, `_normalize_indoor_amenities_relevancy` | Làm câu trả lời listing gọn và đúng intent. |
| `postprocess_answer(question, answer, context_docs)` | Hàm tổng gọi các normalizer phù hợp sau khi LLM trả lời. |

## `app/api/chat.py`

### API/session/runtime

| Hàm/class | Mục đích |
|---|---|
| `_with_timeout(coro, seconds, label, fallback)` | Bọc coroutine với timeout, trả fallback nếu quá thời gian. |
| `_StaticDocumentsRetriever` | Retriever giả trả danh sách docs có sẵn, dùng cho planning mode sau khi retrieval đã làm riêng. |
| `initialize_vector_store()` | Startup init listing + planning vector store. |
| `ChatRequest`, `ChatRequest.PlanningContext` | Schema request `/api/chat`. |
| `ChatResponse` | Schema response `/api/chat`. |
| `chat(req, db)` | Entry point full runtime RAG. |

### Intent và extraction planning

| Hàm/class | Mục đích |
|---|---|
| `_canonical_district_name(seed)` | Map district text về canonical name nội bộ. |
| `_repair_mojibake(text)` | Cố sửa text lỗi encoding. |
| `_strip_accents(text)` | Bỏ dấu tiếng Việt cho matching. |
| `_normalize_nl(text)` | Normalize natural language string. |
| `_has_planning_intent(message)` | Nhận diện câu hỏi quy hoạch. |
| `_planning_query_profile(message)` | Lấy profile intent planning. |
| `_extract_district_from_message(message)` | Extract quận/huyện từ message. |
| `_extract_plan_year_from_message(message)` | Extract năm 20xx từ message. |
| `_history_role(item)`, `_history_content(item)` | Đọc role/content từ history. |
| `_extract_district_from_history(history_messages)` | Lấy district từ history nếu current query thiếu. |
| `_extract_plan_year_from_history(history_messages)` | Lấy plan year từ history nếu current query thiếu. |
| `_is_planning_fact_query(message)` | Nhận diện câu hỏi fact/numeric planning. |
| `_is_planning_project_listing_query(message)` | Nhận diện câu hỏi list dự án/công trình. |
| `_is_planning_land_change_query(message)` | Nhận diện câu hỏi biến động chỉ tiêu đất. |

### Matching/scoring/filtering planning

| Hàm/class | Mục đích |
|---|---|
| `_district_tokens(district)` | Tách token district. |
| `_district_aliases_for_matching(district)` | Sinh alias district để match content/metadata. |
| `_doc_matches_district(doc, district)` | Kiểm tra doc có thuộc district không. |
| `_doc_matches_plan_year(doc, plan_year)` | Kiểm tra doc có đúng năm kế hoạch không. |
| `_planning_doc_score(doc, message, district, plan_year)` | Wrapper scoring planning document. |
| `_planning_specialized_evidence_score(...)` | Wrapper score evidence chuyên biệt. |
| `_planning_intent_rescue_queries(...)` | Wrapper sinh rescue queries theo intent. |
| `_planning_query_candidates(...)` | Wrapper sinh query candidates. |
| `_select_ranked_planning_docs(...)` | Wrapper selector cho docs đã rank. |
| `_rrf_score(rank, k)` | RRF score local cho planning retrieval. |
| `_planning_scope_min_docs(scope, final_k, fact_query)` | Tính số docs tối thiểu để accept strict/district scope. |

### Planning augment/compact/cache/debug

| Hàm/class | Mục đích |
|---|---|
| `_planning_lexical_augments_enabled()` | Đọc env bật lexical augment planning. |
| `_augment_planning_text_neighbors(...)` | Bổ sung text chunks lân cận. |
| `_augment_planning_continuation_neighbors(...)` | Bổ sung chunk tiếp nối khi đoạn bị cắt. |
| `_augment_planning_land_change_fact_docs(...)` | Bổ sung docs chứa fact biến động đất. |
| `_augment_planning_operational_fact_docs(...)` | Bổ sung docs fact vận hành/quy trình. |
| `_augment_planning_table_neighbors(...)` | Bổ sung table chunks gần chunk đã chọn. |
| `_augment_recovery_grouping_with_neighbors(...)` | Bổ sung neighbor cho câu hỏi nhóm thu hồi/không thu hồi. |
| `_augment_planning_intent_evidence(...)` | Bổ sung evidence theo intent. |
| `_augment_planning_land_recovery_evidence(...)` | Bổ sung evidence thu hồi đất. |
| `_force_planning_specialized_evidence(...)` | Ép thêm evidence chuyên biệt nếu retrieval thường thiếu. |
| `_rebalance_planning_chunk_mix(selected, limit, fact_query)` | Cân bằng text/table và limit docs. |
| `_select_relevant_content_lines(...)` | Chọn dòng liên quan trong chunk planning. |
| `_compact_planning_doc(...)` | Compact một planning doc. |
| `_compact_planning_docs(...)` | Compact danh sách planning docs. |
| `_planning_query_cache_key(...)` | Tạo cache key cho query planning. |
| `_planning_query_cache_get(key)` | Lấy cache vector/lexical result. |
| `_planning_query_cache_put(key, value)` | Lưu cache result. |
| `_planning_sync_database_url()` | Resolve DB URL sync cho SQL rescue. |
| `_load_planning_document_docs_sync(...)` | Load toàn bộ chunks một planning document bằng SQL sync. |
| `_load_admin_overview_sql_rescue_docs_sync(...)` | SQL rescue docs cho câu hỏi tổng quan hành chính. |
| `_load_admin_overview_sql_rescue_docs(...)` | Async wrapper cho SQL rescue. |
| `_planning_debug_enabled()` | Đọc env debug planning. |
| `_planning_doc_debug_fields(doc)` | Chuẩn hóa fields debug. |
| `_planning_debug_log(event, payload)` | Print debug event planning. |
| `_planning_debug_doc_list(docs, limit)` | Tạo danh sách doc debug ngắn. |
| `_retrieve_planning_docs_for_nl_query(...)` | Pipeline retrieval planning NL đầy đủ: query expansion, filter scopes, vector/lexical, rerank, augment, compact, fallback. |

### Citation

| Hàm/class | Mục đích |
|---|---|
| `_tokenize(text)` | Tách token cho rerank citation. |
| `_rerank_citations(message, citations, planning_contexts)` | Rerank citation theo overlap và planning property. |
| `_build_planning_citations(docs)` | Build citation giàu metadata cho planning docs. |

## `app/api/ingest.py`

| Hàm/class | Mục đích |
|---|---|
| `_int_env(name, default)` | Parse positive int env. |
| `_get_splitter()` | Lazy init listing splitter theo env chunk size/overlap. |
| `initialize_vector_store()` | Init listing vector store. |
| `IngestPost`, `IngestRequest`, `UpdatePostRequest` | Pydantic schemas cho ingest/update listing. |
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

## `app/planning/ingestion.py`

### Config/extract/OCR

| Hàm/class | Mục đích |
|---|---|
| `_get_splitter()` | Splitter fallback cho planning fixed chunks. |
| `_resolve_bool_env`, `_resolve_planning_chunking_mode`, `_is_hierarchical_chunking_mode` | Đọc config chunking. |
| `_resolve_http_verify`, `_resolve_ssl_allow_insecure_fallback` | Config TLS download tài liệu. |
| `_resolve_pdf_ocr_fallback_enabled`, `_resolve_pdf_ocr_max_pages`, `_resolve_pdf_ocr_render_scale` | Config OCR fallback. |
| `_resolve_pdf_text_quality_min_score`, `_resolve_pdf_force_ocr_on_low_quality` | Config đánh giá quality text layer PDF. |
| `_resolve_ingest_soft_timeout_seconds`, `_resolve_ingest_require_full`, `_resolve_ocr_progress_every_pages` | Config timeout/full ingest/OCR progress. |
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
| `planning_doc_pid_idx(doc)` | Lấy `(planningDocumentId, chunkIndex)` dạng int optional. |
| `planning_chunk_type(doc)` | Lấy `chunkType` lowercase. |

## `app/planning/query_builders.py`

| Hàm/class | Mục đích |
|---|---|
| `_normalize_text(text)` | Normalize query. |
| `district_code_fragment(district)` | Tạo fragment mã quận, ví dụ phục vụ query dossier `HN-...-KH2025`. |
| `planning_specialized_limit(message)` | Tính số evidence chuyên biệt nên lấy theo intent. |
| `planning_fact_subqueries(message)` | Sinh query phụ từ terms/focus/năm. |
| `planning_intent_rescue_queries(message, district, plan_year)` | Sinh rescue queries theo intent planning cụ thể. |
| `planning_query_candidates(message, district, plan_year, max_query_candidates, max_fact_subqueries)` | Kết hợp original query, rescue query, fact subquery, district/year variants. |

## `app/planning/ranker.py`

| Hàm/class | Mục đích |
|---|---|
| `_normalize_ranker_text(text)` | Normalize text cho scoring. |
| `_planning_query_profile(message)` | Cache profile intent planning. |
| `planning_query_terms(message, max_terms)` | Tách terms quan trọng, bỏ stopwords. |
| `planning_neighbor_offsets(doc, message)` | Quyết định cần lấy chunk lân cận nào theo intent/doc signal. |
| `planning_rescue_query_score(doc, query_text)` | Score doc theo rescue query terms, numeric, entity, marker. |
| `planning_specialized_evidence_score(...)` | Score/quality gate evidence chuyên biệt theo từng intent. |
| `planning_intent_markers(message)` | Sinh set marker cần có trong doc theo intent. |
| `planning_intent_alignment_score(doc, intent_markers, query_years)` | Score mức align doc với intent/năm. |
| `planning_doc_score(doc, message, district, plan_year, ...)` | Score tổng quát planning doc trước selection. |

## `app/planning/selector.py`

| Hàm/class | Mục đích |
|---|---|
| `_normalize_selector_text(text)` | Normalize text cho selector. |
| `select_ranked_planning_docs(...)` | Chọn docs tốt nhất từ ranked pool, áp priority theo intent, cân bằng text/table, ưu tiên evidence bắt buộc. |

## `app/planning/features.py`

| Nhóm hàm | Mục đích |
|---|---|
| Normalize/metadata: `_strip_accents`, `_normalize_text`, `is_planning_metadata_line`, `strip_planning_metadata_lines`, `planning_doc_haystack`, `planning_doc_content_norm` | Chuẩn hóa content/metadata để scoring không bị noise descriptor. |
| Chunk type/quality: `planning_chunk_type_hint`, `planning_is_heading_or_incomplete_chunk`, `planning_is_tabular_header_fragment`, `planning_is_toc_like_chunk`, `planning_continuation_signal` | Nhận diện chunk bảng/heading/TOC/đoạn thiếu. |
| Numeric/entity evidence: `planning_count_pattern_score`, `planning_named_entity_hits`, `planning_has_explicit_project_row` | Chấm tín hiệu số liệu và dòng dự án. |
| Admin overview: `planning_admin_unit_header_hits`, `planning_has_admin_unit_evidence`, `planning_has_direct_admin_unit_count_phrase`, `planning_has_direct_natural_area_phrase`, `planning_has_natural_area_admin_evidence` | Tìm evidence diện tích tự nhiên và đơn vị hành chính. |
| Planning intent evidence: `planning_explanatory_evidence_hits`, `planning_registered_plan_evidence_hits`, `planning_has_registered_resolution_count_evidence` | Tìm evidence giải thích/danh mục đăng ký/nghị quyết. |
| Land change: `planning_land_change_label_hits`, `planning_has_land_pair_evidence`, `planning_has_unused_zero_evidence`, `has_land_split_markers` | Tìm evidence biến động đất/chỉ tiêu đất chưa sử dụng. |

## `app/planning/profiles.py`

| Hàm/class | Mục đích |
|---|---|
| `normalize_planning_text(text)` | Normalize text planning. |
| `strip_accents(text)` | Bỏ dấu dùng chung cho planning modules. |
| `planning_marker_hits(haystack, markers)` | Đếm marker hits. |
| `planning_asks_reason(normalized)` | Detect câu hỏi “vì sao/lý do”. |
| `planning_has_structure_composition_request(normalized)` | Detect câu hỏi cấu thành/phân loại. |
| `planning_has_result_grouping_request(normalized)` | Detect câu hỏi nhóm kết quả. |
| `planning_has_registered_plan_request(normalized)` | Detect câu hỏi danh mục đăng ký kế hoạch. |
| `planning_focus_phrases(message)` | Extract focus phrases từ message. |
| `PlanningQueryProfile` | Dataclass/structure chứa các flags intent planning. |
| `build_planning_query_profile(message, planning_intent)` | Build profile intent tổng hợp cho planning ranker/selector/query builder. |

## `app/rag/planning_pipeline.py`

| Hàm/class | Mục đích |
|---|---|
| `_has_grouping_project_context(normalized_blob)` | Kiểm tra context có liên quan nhóm dự án thu hồi/không thu hồi. |
| `_has_large_grouping_project_count(normalized_blob)` | Kiểm tra có count dự án lớn đáng tin trong grouping. |
| `is_recovery_grouping_query(normalized_message)` | Detect query phân nhóm thu hồi/không thu hồi. |
| `recovery_grouping_signal_score(normalized_blob)` | Score chất lượng evidence grouping. |
| `has_min_recovery_grouping_evidence(normalized_blob)` | Quality gate bắt buộc cho grouping evidence. |
| `score_planning_fallback_candidate(docs, doc_score_fn)` | Score fallback docs theo top docs. |
| `choose_better_planning_fallback(current_docs, current_score, candidate_docs, doc_score_fn)` | Chọn fallback tốt hơn giữa các scope/base filters. |

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

