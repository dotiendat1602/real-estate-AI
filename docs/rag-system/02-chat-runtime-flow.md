# 02. Luồng runtime `/api/chat`

Entry point chính nằm ở `app/api/chat.py::chat`.

## Request/response model

`ChatRequest` gom:

- `userId`: bat buoc de lay/luu history.
- `sessionId`: optional, neu khong co se tao session moi.
- `message`: cau hoi nguoi dung.

`topK` khong con la public request field; runtime lay tu env `TOP_K_DEFAULT`.

`ChatResponse` gom:

- `sessionId`
- `answer`
- `citations`
- `extractedFilters`
- `tokenUsage`
- `timings`
## Luồng chi tiết trong `chat()`

Phần dưới mô tả luồng ở mức pipeline. Nếu cần đọc chi tiết từng khối logic trong `chat.py` và `chain.py`, xem thêm [05-deep-dive-chat-chain.md](05-deep-dive-chat-chain.md).

```text
1. Validate userId.
2. MessageHistoryManager.get_or_create_session().
3. MessageHistoryManager.get_messages(limit=6).
4. build_retrieval_query(req.message, history).
5. Lay top_k tu TOP_K_DEFAULT.
6. build_llm().
7. Xac dinh use_planning_mode bang _has_planning_intent(req.message).
8. Neu listing mode: extract_filters_from_query_with_usage().
9. Neu planning mode:
   - initialize_planning_vector_store()
   - retrieve planning docs theo natural-language query
10. Neu listing mode:
   - initialize_listing_vector_store()
   - build_retriever() + ListingFallbackRetriever()
11. Tao RagChain.
12. Planning mode doi retriever thanh StaticDocumentsRetriever(planning_docs).
13. RagChain.run().
14. Merge citations + planning citations.
15. Luu user/assistant message.
16. Tra ChatResponse.
```

## Nhan dien planning mode

`use_planning_mode = _has_planning_intent(req.message)`

Planning mode duoc bat khi message co cac marker nhu quy hoach, ke hoach su dung dat, thu hoi dat, chi tieu, phe duyet, HDND, du an quy hoach...

Rieng `du an`/`cong trinh` la marker de trung voi listing; chung chi bat planning mode khi di cung nam ke hoach va hint dia ban/xa-phuong.
## Listing RAG flow

Khi không phải planning:

```text
build_llm()
  -> extract_filters_from_query_with_usage()
  -> initialize_listing_vector_store()
  -> build_retriever(vs, k=top_k, filters=filters)
  -> ListingFallbackRetriever(primary, query, filters, k)
  -> RagChain.run()
```

### Filter extraction

`extract_filters_from_query_with_usage()` dùng LLM parse các filter:

- `city`
- `district`
- `postType`
- `priceMin`, `priceMax`
- `areaMin`, `areaMax`
- `bedrooms`

Sau đó `_sanitize_extracted_filters()` loại filter rủi ro:

- Không dùng địa chỉ cụ thể làm city/district.
- Chỉ nhận `postType` thuộc `SALE`, `RENT`, `OTHER`.
- Đảo min/max nếu LLM trả ngược.

### Retriever listing

`build_retriever()` mặc định dùng `HybridRetriever` nếu `RAG_RETRIEVER_MODE=hybrid`.

Hybrid retrieval:

```text
semantic search PGVector
  + lexical_search_documents() bằng SQL LIKE
  -> fuse bằng Reciprocal Rank Fusion
  -> trả top k Document
```

Nếu query có intent listing rõ hoặc kết quả vector ít, `ListingFallbackRetriever` gọi `search_listing_documents()` để query SQL trực tiếp bảng domain. Fallback này thường cho context giàu cấu trúc hơn vì lấy cả property fields, amenities, location, price, area.

## Planning RAG flow từ natural language

Khi user hỏi quy hoạch nhưng không có `planning docs`, code gọi:

`_retrieve_planning_docs_for_nl_query(planning_vs, retrieval_message, top_k, history_messages=history)`

Luồng trong hàm này:

```text
1. Extract district và planYear từ message/history.
2. Phân loại fact query.
3. Tính final_k, probe_k.
4. Sinh query_candidates.
5. Sinh base_filter candidates:
   - planning + dossierCode chính xác nếu suy ra được district canonical và năm
   - planning + district + year
   - planning + district
   - planning + force planningDocumentId + year
   - planning + force planningDocumentId
   - planning + year
   - planning broad
6. Với từng base_filter:
   - vector retrieval cho từng query candidate
   - cache kết quả query
   - dedupe docs
   - cộng RRF theo thứ hạng vector của từng query candidate
   - chia strict/district/broad pool
   - select_ranked_planning_docs()
   - rebalance text/table
   - compact docs
   - nếu đủ min docs thì return
7. Nếu không đủ strict/district: chọn best broad fallback.
```

## Query expansion planning

`_planning_query_candidates()` trong `app/rag/planning_retrieval.py` wrap `app/planning/query_builders.py::planning_query_candidates()`.

Nó tạo query phụ theo intent:

- biến động chỉ tiêu đất 2024/2025
- danh mục dự án/công trình
- nhóm công trình/dự án trong `biểu 1A`
- thu hồi đất/chuyển mục đích
- GPMB
- đất công cộng
- Điều 67
- công trình cấp thành phố
- tổng số theo quyết định phê duyệt
- diện tích tự nhiên, đơn vị hành chính

Việc query nhiều biến thể giúp vector search không phụ thuộc hoàn toàn vào câu hỏi tự nhiên ban đầu.

## Ranking và selection planning

Planning retrieval hiện không dùng lớp score thủ công cho từng document. Thứ tự chính đến từ vector retrieval và Reciprocal Rank Fusion khi cùng một document xuất hiện qua nhiều query candidates.

Sau đó `select_ranked_planning_docs()` chỉ làm các bước dễ giải thích:

- giữ thứ tự RRF đã có;
- bỏ chunk giống mục lục;
- đẩy doc lệch quận/năm hoặc chunk heading yếu xuống sau;
- Khong can bang text/table bang rule rieng.

Sau selection, pipeline compact planning docs theo thu tu da chon truoc khi dua vao context.

## Context preparation trong `RagChain`

`RagChain.run()` luôn gọi:

```text
docs = await retriever.ainvoke(retrieval_query)
docs_for_context = prepare_docs_for_context(...)
context = _format_docs(docs_for_context)
context += extra_context nếu có
prompt -> llm -> postprocess_answer()
```

Với planning mode, retriever là `StaticDocumentsRetriever(planning_docs)` trong `app/rag/static_retriever.py`, nên `RagChain` chỉ chuẩn hóa/compact và prompt, không đi search lại.

`prepare_docs_for_context()`:

- Dedupe document.
- Voi listing: giu thu tu retriever, compact theo thu tu tu nhien va build structured listing context neu co metadata.
- Voi planning: giu thu tu da rank va cat ky tu, khong them contract phu hay score dong thu cong.
- Sanitize text trước khi gửi LLM.

## Citations

Citation được build ở 2 nơi:

- `RagChain._build_citations()` cho docs đưa vào context.
- `app/rag/citation_utils.py::build_planning_citations()` cho planning docs lấy ngoài chain.

Sau đó `chat()` merge và `app/rag/citation_utils.py::rerank_citations()`:

- Ưu tiên score có sẵn từ retriever nếu citation có `score`.
- Uu tien citation planning document neu co `planningDocumentId`.
- Dedupe theo key metadata planning/listing.

## Token usage và timing

`extract_filters_from_query_with_usage()` trả token usage cho bước filter extraction.

`RagChain.run()` trả:

- token usage của answer generation.
- `retrieval_seconds`
- `answer_generation_seconds`
- `runtime_seconds`

`chat()` cộng token bằng `sum_token_usage()`.
