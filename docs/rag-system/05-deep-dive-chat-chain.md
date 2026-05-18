# 05. Deep dive `chat.py` và `chain.py`

File này giải thích kỹ hơn các hàm dài và các cụm logic quan trọng trong:

- `app/api/chat.py`
- `app/rag/chain.py`

Mục tiêu là đọc code theo hướng "hàm này nhận gì, làm gì, vì sao cần, và ảnh hưởng tới RAG thế nào".

## 1. `app/api/chat.py`

`chat.py` là tầng orchestration của API `/api/chat`. Nó không chỉ gọi retriever rồi gọi LLM, mà còn quyết định request thuộc miền listing hay planning, sửa query dựa trên history, trích filter, lấy planning context theo nhiều chiến lược, compact context và rerank citations.

### 1.1. Nhóm normalize và nhận diện planning intent

Các helper normalize được dùng trong `chat.py` qua alias `_normalize_nl` và `_strip_accents`, implementation nằm ở `app/utils/text.py`.

#### `repair_mojibake(text)`

Mục đích: cố sửa chuỗi bị lỗi encoding trước khi normalize tiếng Việt.

Input:

- `text`: chuỗi bất kỳ từ user/context/metadata.

Logic:

1. Nếu text không chứa marker mojibake như `Ã`, `Â`, `Ä`, `�` thì trả nguyên chuỗi.
2. Tạo danh sách candidates gồm text gốc và các bản thử decode lại theo `latin1` hoặc `cp1252` rồi decode `utf-8`.
3. Đếm số marker lỗi còn lại của từng candidate.
4. Trả candidate có ít marker lỗi nhất.

Vai trò trong RAG:

- Giúp `_normalize_nl()` và các hàm matching không bị fail vì text tiếng Việt lưu/truyền sai encoding.
- Đặc biệt hữu ích với metadata hoặc nội dung tài liệu quy hoạch OCR/PDF bị lỗi.

#### `strip_vietnamese_accents(text)` và `normalize_vietnamese_search_text(text)`

Mục đích: chuẩn hóa text để so khớp keyword, quận/huyện, năm, marker quy hoạch.

`strip_vietnamese_accents()`:

1. Gọi `repair_mojibake()`.
2. Đổi `đ/Đ` thành `d/D`.
3. Dùng Unicode NFD để tách dấu.
4. Bỏ các ký tự dấu.

`normalize_vietnamese_search_text()`:

1. Lowercase.
2. Bỏ dấu.
3. Thay ký tự không phải chữ/số/space thành space.
4. Gom nhiều khoảng trắng.

Vai trò trong RAG:

- Tất cả intent detection, district matching, marker matching đều dựa trên dạng normalized này.
- Cho phép user hỏi không dấu: "quy hoach quan cau giay nam 2025".

#### `_has_planning_intent(message)`

Mục đích: quyết định request có đi vào planning RAG hay listing RAG.

Input:

- `message`: câu hỏi user.

Output:

- `True`: dùng planning retrieval.
- `False`: dùng listing retrieval.

Logic theo thứ tự:

1. Normalize message.
2. Nếu chứa keyword planning trực tiếp trong `_PLANNING_KEYWORDS`, trả `True`.
3. Extract tín hiệu phụ:
   - `has_district_hint`: có quận/huyện không.
   - `has_plan_year`: có năm dạng `20xx` không.
   - `has_admin_unit_hint`: có `phuong`, `xa`, `thi tran` không.
4. Nếu có thuật ngữ đất/hành chính mạnh (`thu hoi`, `dat nong nghiep`, `luat dat dai`...) và có district/year/admin hint, trả `True`.
5. Với marker dễ trùng listing như `cong trinh`/`du an`, chỉ trả `True` khi đồng thời có năm kế hoạch và district/admin hint.
6. Nếu có thuật ngữ lý do/bối cảnh (`quan ly dat dai`, `ha tang`, `thoat nuoc`, `giao thong`, `gpmb`...) và có district/year/admin hint, trả `True`.
7. Nếu không có structural term thì trả `False`.
8. Nếu có structural term và có district/year/admin hint thì trả `True`.

Ý nghĩa:

- Tránh đưa mọi câu có chữ "dự án" vào planning RAG.
- Nhưng vẫn bắt được câu planning không nói thẳng "quy hoạch", ví dụ hỏi "năm 2025 quận Hoàng Mai có bao nhiêu dự án thu hồi đất?".

Rủi ro/điểm cần biết:

- Đây là heuristic. Nếu user hỏi listing nhưng có nhiều từ như "dự án", "quận", "2025", có thể bị route sang planning.
- Nếu user hỏi planning quá mơ hồ, không có district/year/admin hint, có thể route sang listing.

### 1.2. Nhóm extract district/year và match tài liệu

#### `_extract_district_from_message(message)`

Mục đích: tìm quận/huyện từ câu hỏi.

Logic:

1. Normalize message.
2. Duyệt `_DISTRICT_MATCH_ALIASES`, hiện dùng chung nguồn alias chuẩn `PLANNING_DISTRICT_ALIASES` với module metadata.
3. Nếu alias xuất hiện như một cụm từ riêng trong message, trả canonical district.
4. Nếu không match district trực tiếp, thử `_WARD_DISTRICT_HINTS`, ví dụ phường `Mai Dịch` suy ra `Cầu Giấy`.

Vai trò:

- District được dùng để tạo base filter cho planning collection.
- District cũng được dùng trong scoring và compact context.

#### `_extract_plan_year_from_message(message)`

Mục đích: tìm năm kế hoạch từ câu hỏi.

Logic:

1. Regex `\b(20\d{2})\b`.
2. Ép int.
3. Chỉ nhận năm trong khoảng 2000-2100.
4. Trả năm đầu tiên hợp lệ.

Vai trò:

- Làm filter `planYear`.
- Dùng trong `_doc_matches_plan_year()` và query expansion.

#### `_extract_district_from_history()` và `_extract_plan_year_from_history()`

Mục đích: hỗ trợ câu hỏi follow-up thiếu thông tin.

Logic:

1. Duyệt history từ mới nhất về cũ nhất.
2. Ưu tiên message role `user`.
3. Nếu không tìm thấy trong user messages, duyệt cả history.

Ví dụ:

```text
User: Kế hoạch sử dụng đất quận Hoàng Mai năm 2025 có gì đáng chú ý?
User: Còn các dự án thu hồi đất thì sao?
```

Query thứ hai không có quận/năm, nhưng hàm lấy lại `Hoàng Mai` và `2025` từ history.

#### `_doc_matches_district(doc, district)`

Mục đích: kiểm tra một planning `Document` có đúng địa bàn không.

Logic:

1. Nếu không có district yêu cầu, trả `True`.
2. Canonicalize district từ input.
3. Canonicalize district của doc từ metadata `district`, `districtRaw`, `title`, `dossierCode`.
4. Nếu canonical match, trả `True`.
5. Nếu metadata thiếu/sai, kiểm tra alias trong haystack của doc.
6. Fallback cuối: tách token district và yêu cầu tất cả token xuất hiện trong haystack.

Vai trò:

- Chia pool strict/district/broad trong planning retrieval.
- Tránh lấy nhầm tài liệu quận khác khi vector similarity gần nhau.

#### `_doc_matches_plan_year(doc, plan_year)`

Mục đích: kiểm tra doc có đúng năm kế hoạch không.

Logic:

1. Nếu user không hỏi năm, trả `True`.
2. Ưu tiên metadata `planYear`.
3. Nếu metadata không dùng được, tìm chuỗi năm trong haystack.

Vai trò:

- Tương tự district matching nhưng cho năm.
- Cho phép fallback khi metadata ingest thiếu năm nhưng nội dung có năm.

### 1.3. Nhóm wrapper sang planning modules

Các hàm sau trong `chat.py` chủ yếu wrap module `app/planning/*` để truyền đúng dependency:

- `_planning_query_candidates()`
- `_select_ranked_planning_docs()`

Vì sao cần wrapper:

- `planning.ranker` chỉ giữ helper tách query terms; `planning.selector` áp rule chọn tài liệu sau khi vector RRF đã xếp hạng.
- `planning.selector` cần hàm match district/year do API layer biết cách match metadata hiện tại.
- Giữ logic selection tách khỏi FastAPI nhưng vẫn tái sử dụng được matcher local.

### 1.4. `_rebalance_planning_chunk_mix(docs, limit, fact_query)`

Mục đích: cân bằng `text` và `table` chunks trước khi đưa vào context.

Input:

- `docs`: docs đã rank.
- `limit`: số docs tối đa.
- `fact_query`: câu hỏi fact/numeric hay không.

Logic:

1. Dedupe theo `_planning_doc_identity()`.
2. Chia docs thành `text_docs`, `table_docs`, `other_docs`.
3. Nếu chỉ có một loại thì trả theo thứ tự hiện có.
4. Nếu là fact query:
   - table target tối đa khoảng nửa `limit`, cap 3.
5. Nếu không phải fact query:
   - table target nhỏ hơn, khoảng 1/3 `limit`, cap 2.
6. Merge xen kẽ, thường bắt đầu bằng text nếu text nhiều hơn.
7. Fill phần còn lại bằng docs chưa dùng.

Vì sao cần:

- Planning docs có cả đoạn giải thích và bảng số liệu.
- Nếu chỉ lấy table, LLM thiếu bối cảnh.
- Nếu chỉ lấy text, LLM thiếu số liệu/dòng dự án cụ thể.

### 1.5. `_select_relevant_content_lines()`

Mục đích: chọn các dòng quan trọng trong một planning chunk dài.

Input:

- `text`: body chunk.
- `message`: câu hỏi.
- `district`, `plan_year`: hint để ưu tiên dòng.
- `max_lines`: số dòng tối đa.

Khóa sắp xếp từng dòng:

- Dòng đầu chunk để giữ bối cảnh.
- Số term từ câu hỏi xuất hiện trong dòng.
- Số alias district khớp với dòng.
- Năm kế hoạch có xuất hiện hay không.
- Dòng có số liệu hay không.
- Dòng có dạng dự án/công trình rõ hay không.
- Số marker evidence quy hoạch như danh mục đăng ký, nghị quyết hoặc biến động đất.

Sau khi tạo khóa:

1. Sort theo khóa relevance giảm dần.
2. Chọn top `max_lines`.
3. Sort lại theo thứ tự xuất hiện ban đầu để context vẫn đọc được.

Vai trò:

- Giảm token noise trong planning context.
- Giữ các dòng có khả năng chứa answer.

### 1.6. `_compact_planning_doc()` và `_compact_planning_docs()`

#### `_compact_planning_doc()`

Mục đích: rút gọn một planning doc nhưng giữ metadata header và các dòng liên quan.

Logic:

1. Lấy `raw_content`.
2. Bỏ metadata lines bằng `_strip_planning_metadata_lines()`.
3. Nếu body rỗng thì fallback về content gốc.
4. Build header:
   - `[city=...]`
   - `[district=...]`
   - `[plan_year=...]`
   - `[dossier_code=...]`
   - `[title=...]`
   - dòng descriptor `document_type | district | plan_year | chunk_type | title`
5. Nếu body ngắn:
   - fact query: giữ body nếu <= 2200 chars.
   - non-fact: giữ body nếu <= 1200 chars.
6. Nếu body dài:
   - gọi `_select_relevant_content_lines()`.
   - fact query chọn nhiều dòng hơn (`28`) so với non-fact (`12`).
   - nếu không chọn được dòng nào thì cắt prefix.
7. Trả `Document` mới với content đã compact và metadata cũ.

#### `_compact_planning_docs()`

Mục đích: compact cả list và dedupe sau compact.

Logic:

1. Gọi `_compact_planning_doc()` từng doc.
2. Loại doc compact xong không còn body meaningful.
3. Tạo key bằng `planningDocumentId`, `chunkType`, chunk index và prefix normalized content.
4. Dedupe rồi trả list.

### 1.7. Cache planning query

Các hàm:

- `_planning_query_cache_key()`
- `_planning_query_cache_get()`
- `_planning_query_cache_put()`

Mục đích:

- Cache kết quả vector retrieval theo query normalized, base filter và `probe_k`.

Implementation:

- Dùng `OrderedDict`.
- Sau mỗi get/put, move item về cuối.
- Nếu vượt `_PLANNING_QUERY_CACHE_SIZE`, pop item cũ nhất.

Vai trò:

- Planning retrieval chạy nhiều query candidates và nhiều scopes.
- Cache giảm gọi vector store lặp lại trong cùng process.

### 1.8. `_retrieve_planning_docs_for_nl_query()`

Đây là hàm dài và quan trọng nhất trong planning runtime.

Mục đích:

- Từ câu hỏi tự nhiên, lấy ra danh sách planning docs tốt nhất để đưa vào LLM.
- Không phụ thuộc hoàn toàn vào vector similarity.
- Có cơ chế strict -> relaxed -> broad fallback.

Input:

- `planning_vs`: PGVector store planning.
- `message`: query đã có thể được rewrite từ history.
- `top_k`: số lượng user/API yêu cầu.
- `history_messages`: dùng để infer district/year nếu message thiếu.
- `force_planning_document_id`: optional, ép search trong một tài liệu cụ thể.

Output:

- `list[Document]` đã selected, compacted và rebalanced.

Các pha xử lý:

#### Pha 1: derive query context

Hàm tính:

- `district`: từ message hoặc history.
- `plan_year`: từ message hoặc history.
- `fact_query`: profile intent cho câu hỏi fact/numeric.
- `final_k`: giới hạn docs cuối, fact query tối đa 16, non-fact tối đa 12.
- `strict_min_docs`, `district_min_docs`: số docs tối thiểu để accept scope.
- `probe_k`: số docs lấy từ vector mỗi query, lớn hơn final để có pool cho RRF rank và filter scope.

#### Pha 2: sinh query candidates

`query_candidates = _planning_query_candidates(message, district, plan_year)`

Query candidates gồm:

- câu hỏi gốc
- câu hỏi gốc + district/year
- focus terms
- rescue queries theo intent
- fact subqueries
- dossier-like query như `HN-...-KH2025`

Mục tiêu:

- Một câu hỏi tự nhiên có thể không giống wording trong PDF.
- Query expansion tăng xác suất lấy đúng chunk.

#### Pha 3: sinh base filter candidates

Hàm thử nhiều filter từ hẹp tới rộng:

1. `documentScope=planning + dossierCode` nếu suy ra được district canonical và năm.
2. `documentScope=planning + district + planYear`
3. `documentScope=planning + district`
4. nếu `force_planning_document_id`: document id + year
5. nếu `force_planning_document_id`: document id
6. `documentScope=planning + planYear`
7. `documentScope=planning`

Sau đó dedupe filters bằng JSON string.

Ý nghĩa:

- Strict trước để tránh nhiễu.
- Nếu strict quá ít docs thì nới dần.
- Broad fallback vẫn đảm bảo có answer nếu metadata quận/năm thiếu hoặc sai.

#### Pha 4: retrieve cho từng base filter và query candidate

Với mỗi `base_filter`:

1. Build retriever ở mode vector:
   - `filters={"chunkTypes": ["text", "table"]}`
   - `base_filter=base_filter`
   - `mode_override="vector"`
2. Với từng `query_text`:
   - Tạo cache key.
   - Nếu cache hit, dùng lại `vector_docs`.
   - Nếu cache miss:
     - gọi `planning_retriever.ainvoke(query_text)`
     - lưu cache.
3. Mỗi doc được dedupe theo `_planning_doc_identity()`.
4. Lưu vector RRF boost theo rank để giữ tín hiệu semantic rank trong pool.

#### Pha 5: xếp hạng pool

Sau khi thu docs:

1. Dedupe bằng `_dedupe_planning_docs()`.
2. Lấy RRF score đã cộng từ thứ hạng vector của các query candidates.
3. Sort giảm dần theo RRF score.

Điểm quan trọng: runtime không còn cộng điểm document bằng marker/keyword thủ công. Metadata như `dossierCode`, `district`, `planYear` được dùng ở bước filter và chia scope; heading/TOC yếu được xử lý như quality gate trong selector.

#### Pha 6: chia scope pool

Từ ranked docs, tạo:

- `strict`: match district và plan year.
- `relaxed`: match district.
- `broad`: tất cả ranked docs.

Sau đó duyệt theo thứ tự strict -> district -> broad.

#### Pha 7: select, rebalance, compact

Với mỗi scope pool:

1. `_select_ranked_planning_docs()` chọn docs tốt nhất theo intent.
2. `_rebalance_planning_chunk_mix()`.
3. `_compact_planning_docs()`.
4. Rebalance lần nữa sau compact.

Vì sao rebalance trước và sau compact:

- Trước compact: kiểm soát loại chunk được giữ.
- Sau compact: dedupe/compact có thể loại bớt docs, cần cân bằng lại.

#### Pha 8: quyết định return hay fallback

Nếu scope là `strict`:

- Return nếu selected docs >= `strict_min_docs`.
- Nếu không đủ, tiếp tục scope rộng hơn.

Nếu scope là `district`:

- Lưu candidate vào `best_fallback`.
- Return nếu selected docs >= `district_min_docs`.

Nếu scope là `broad`:

- Không return ngay theo min docs.
- Dùng `choose_better_planning_fallback()` để giữ fallback tốt nhất.

Cuối cùng:

- Nếu có `best_fallback`, chỉ rebalance rồi return vì candidate này đã được compact ở pha chọn trước đó.
- Nếu không có docs nào, return `[]`.

### 1.11. Planning citations

Hai helper citation đã được tách khỏi `chat.py` sang `app/rag/citation_utils.py`; `chat.py` chỉ import alias để merge vào response.

#### `build_planning_citations(docs)`

Mục đích: tạo citation metadata từ planning docs.

Dedupe key:

```text
planningDocumentId:chunkType:globalChunkIndex:chunkIndex:pageNumber
```

Fields citation:

- `planningDocumentId`
- `documentScope`
- `documentType`
- `dossierCode`
- `city`, `district`, `planYear`
- `title`, `sourceUrl`, `format`
- `chunkType`, `chunkIndex`, `globalChunkIndex`
- `pageNumber`, `lineStart`, `lineEnd`, `sourceLocator`
- `snippet`

#### `rerank_citations(message, citations, planning_contexts)`

Mục đích: sắp lại citations để nguồn liên quan hơn đứng trước.

Ordering:

- ưu tiên citation thuộc planning context cụ thể nếu request có `planningContexts`;
- sau đó dùng `score` có sẵn từ retriever nếu citation có field này;
- nếu không có score, giữ thứ tự xuất hiện ban đầu.

### 1.12. `chat(req, db)`

Đây là hàm API chính.

Input:

- `ChatRequest`
- `AsyncSession` từ dependency `get_db()`

Output:

- `ChatResponse`

Luồng chi tiết:

1. Validate `userId`.
   - Nếu thiếu, raise `ValueError`.

2. Khởi tạo history manager.
   - `get_or_create_session(req.userId, req.sessionId)`.
   - `get_messages(session_id, limit=6)`.

3. Rewrite retrieval query.
   - `retrieval_message = build_retrieval_query(req.message, history)`.
   - Nếu câu hiện tại là follow-up, query retrieval sẽ được ghép với anchor từ history.

4. Tạo LLM.
   - `llm = build_llm()`.

5. Extract listing filters.
   - Dùng `filter_query`.
   - Nếu `retrieval_message` khác message gốc thì dùng `retrieval_message`, vì follow-up cần filter từ context đã ghép.
   - `extract_filters_from_query_with_usage()` trả filters và token usage.

6. Chọn mode.
   - `use_planning_mode = bool(req.planningContexts) or _has_planning_intent(req.message)`.

7. Nếu planning mode:
   - Init planning vector store.
   - Build planning retriever tạm với `documentScope=planning`, chunk types text/table.
   - Lúc này `retriever` chỉ là placeholder ban đầu; sau khi lấy `planning_docs`, code thay bằng `StaticDocumentsRetriever` từ `app/rag/static_retriever.py`.

8. Nếu listing mode:
   - Init listing vector store.
   - Build primary retriever bằng `build_retriever(vs, k=req.topK, filters=filters)`.
   - Wrap bằng `ListingFallbackRetriever`.

9. Nếu request có `planningContexts`:
   - Build `extra_context` từ summary backend:
     - propertyId
     - planningStatus
     - riskLevel
     - landUseCurrent/Planned
     - dossier
     - checkedAt
     - report summaries
   - Build planning retriever với base filter:
     - `documentScope=planning`
     - `propertyId IN planning_property_ids`
   - Retrieve planning docs theo `retrieval_message`.
   - Append docs text vào `extra_context` dưới header `PLANNING VECTOR CONTEXT`.
   - Build planning citations.

10. Nếu planning mode nhưng không có `planningContexts`:
    - Extract district/plan year từ message để ghi header.
    - Gọi `_retrieve_planning_docs_for_nl_query()`.
    - Nếu có docs:
      - Join docs thành planning text.
      - Thêm header `PLANNING VECTOR CONTEXT (AUTO FROM NATURAL LANGUAGE QUERY)`.
      - Build planning citations.

11. Nếu planning mode:
    - Rebuild chain bằng `RagChain(llm, StaticDocumentsRetriever(planning_docs))`.
    - Lý do: planning docs đã được retrieve/compact riêng, không muốn `RagChain` gọi retriever thường thêm lần nữa.

12. Chạy generation.
    - `result = await chain.run(req.message, history=history, extra_context=extra_context)`.

13. Citations.
    - `merged_citations = result.citations + planning_citations`.
    - `rerank_citations(...)` từ `app/rag/citation_utils.py`.

14. Lưu history.
    - Lưu message user.
    - Lưu answer assistant.

15. Trả response.
    - answer
    - citations
    - extractedFilters
    - token usage filter + answer + total
    - timings từ `RagChain`

Điểm cần chú ý:

- Filter extraction luôn chạy cả khi planning mode. Nó chủ yếu phục vụ listing, nhưng response vẫn trả `extractedFilters`.
- Planning mode dùng `extra_context` cộng thêm vào context của `RagChain`. Với `planningContexts`, context có cả summary backend và vector docs.
- Nếu planning retrieval timeout, `_with_timeout()` trả fallback `[]`; LLM vẫn có thể trả dựa trên `extra_context` nếu có.

## 2. `app/rag/chain.py`

`chain.py` hien la thin orchestrator cho RAG runtime. Neu `chat.py` quyet dinh lay docs nao, thi cac module RAG runtime quyet dinh rewrite query, compact context, goi LLM va postprocess answer.

Module split hien tai:

- `app/rag/chain.py`: orchestration, prompt call, timings, citations.
- `app/rag/query_rewrite.py`: `build_retrieval_query()`.
- `app/rag/query_intents.py`: intent flags dung chung cho context va postprocess.
- `app/rag/context_preparation.py`: `prepare_docs_for_context()`, evidence contract cho planning fact query, structured listing context.
- `app/rag/answer_processing.py`: `postprocess_answer()` va `detect_lang()`.
- `app/rag/listing_processing.py`: suitability/listing helper dung chung.
- `app/rag/citation_utils.py`: planning citation builder va citation reranker dung o API response.
- `app/rag/static_retriever.py`: retriever tra docs co san cho planning mode.
- `app/utils/text.py`: normalize/sanitize text dung chung, gom ca sua mojibake; `app/rag/text_utils.py` chi re-export de giu import cu.

### 2.1. `build_retrieval_query(question, history)`

Mục đích: rewrite query cho câu hỏi follow-up để retriever không mất ngữ cảnh.

Logic:

1. Nếu question rỗng, trả rỗng.
2. Nếu không có history hoặc câu hỏi không giống follow-up, trả question gốc.
3. Lấy các user messages gần nhất bằng `_recent_user_messages()`.
4. Tìm anchor message gần nhất đủ giàu thông tin bằng `_is_anchor_rich_message()`:
   - có số
   - có location
   - có entity listing như nhà/căn/bất động sản/đất/quận/đường
5. Nếu không có anchor rich, dùng user message gần nhất.
6. Nếu anchor đã nằm trong current hoặc current nằm trong anchor, trả current.
7. Nếu cần rewrite, trả dạng:

```text
Bat dong san dang duoc nhac toi: <anchor>
Cau hoi hien tai: <current>
```

Vai trò:

- Query "còn giá thì sao?" sẽ retrieve đúng listing đã hỏi trước đó.
- Query planning follow-up cũng lấy lại district/year qua context.

### 2.2. `postprocess_answer(question, answer)`

Mục đích: làm sạch và chuẩn hóa answer sau LLM.

Các bước chính:

1. Nếu answer rỗng, trả nguyên.
2. Strip answer.
3. Loại generic trailing invites, ví dụ "Nếu bạn cần thêm...".
4. Build query intents bằng `_build_query_intents()`.
5. Nhận diện câu hỏi:
   - hỏi số lượng dự án/công trình.
   - hỏi danh sách tên dự án.
6. Nếu user không hỏi diện tích, gọi `_strip_unrequested_area_suffix()`.
7. Chạy `_PRE_CONTEXT_ANSWER_TRANSFORMS`.
   - Các transform này chỉ cần question + answer.
   - Ví dụ chuẩn hóa câu trả lời dự án đăng ký mới, thu hồi đất, GPMB.
8. Chạy `_POST_CONTEXT_ANSWER_TRANSFORMS`.
   - Chỉnh wording cuối cùng cho planning/listing.
9. Nếu là planning fact hoặc câu hỏi đất nông nghiệp/phi nông nghiệp/chưa sử dụng:
    - `_strip_unrequested_planning_extra_lines()` loại dòng ngoài phạm vi hỏi.
10. Nếu là suitability query mà không hỏi giá:
    - `_strip_unrequested_price_lines()`.
11. Nếu hỏi tiện ích trong nhà:
    - `_strip_outdoor_detail_lines()`.
    - `_normalize_indoor_amenities_answer()`.
12. Nếu suitability query:
    - `_condense_suitability_answer()` rút gọn.
13. Nếu không phải planning fact, trả cleaned.
15. Nếu planning fact nhưng answer là "không biết/không đủ dữ liệu", trả cleaned.
16. Trả cleaned.

Vai trò:

- Giữ answer đúng intent, đặc biệt khi prompt/LLM có xu hướng thêm thông tin phụ.
- Giảm hallucination dạng tính toán thêm hoặc thêm giá/diện tích khi user không hỏi.

### 2.3. `_build_query_intents(question)`

Mục đích: phân tích nhu cầu của câu hỏi listing để compact context và postprocess answer.

Các flags chính:

- `needs_price`
- `needs_area`
- `needs_direction`
- `needs_main_door_direction`
- `needs_bedrooms`
- `needs_bathrooms`
- `needs_furnishing`
- `needs_min_rental_period`
- `needs_cashflow`
- `needs_indoor_amenities`
- `needs_location`
- `suitability_query`
- `business_query`
- `investment_query`
- `study_work_query`
- `explanatory_query`

Vai trò:

- `_line_relevance_key()` dùng flags để chọn dòng context.
- `postprocess_answer()` dùng flags để bỏ thông tin không được hỏi.
- `_build_structured_listing_context()` dùng flags để chọn highlights phù hợp.

### 2.4. `_line_relevance_key(line, query_terms, number_terms, intents)`

Mục đích: tạo khóa sắp xếp một dòng listing context trước khi compact, không dùng trọng số thực nghiệm kiểu cộng/trừ điểm.

Khóa gồm các tín hiệu:

- Ít noise hơn: hạn chế dòng liên hệ, boilerplate, giá/hướng/vị trí khi user không hỏi.
- Có header context hay không.
- Số term từ câu hỏi xuất hiện trong dòng.
- Số trong câu hỏi xuất hiện trong dòng.
- Số marker khớp intent như suitability, study/work, explanatory.
- Số hint thông tin phổ biến như phòng ngủ, diện tích, nội thất, tiện ích.
- Business marker nếu user hỏi kinh doanh/đầu tư/phù hợp.

Vai trò:

- Context listing đưa vào LLM không phải toàn bộ bài đăng, mà là các dòng có khả năng trả lời đúng intent.

### 2.5. `_compact_doc_content(question, content, max_chars, max_lines)`

Mục đích: rút gọn content listing/raw doc.

Logic:

1. Thay `<br/>`, `<br>` thành newline.
2. `_dedupe_repeated_blocks()` để bỏ block trùng.
3. Nếu content đã ngắn và ít dòng, trả nguyên.
4. Split thành lines.
5. Dòng quá dài được chia bằng `_split_long_line()`.
6. Dedupe line theo normalized text.
7. Extract query terms và intents.
8. Giữ dòng header đầu tiên nếu có để LLM biết listing nào.
9. Tạo khóa sắp xếp từng dòng bằng `_line_relevance_key()`.
10. Sort theo khóa relevance.
11. Chọn tối đa `max_lines`, không vượt `max_chars`.
12. Nếu không chọn được gì, fallback prefix content.

Vai trò:

- Giảm context dài/loạn từ bài đăng bất động sản.
- Giữ lại facts đúng câu hỏi: giá, diện tích, phòng ngủ, nội thất, tiện ích, vị trí.

### 2.6. `_build_structured_listing_context(question, doc)`

Mục đích: tạo context listing có cấu trúc từ metadata, thay vì chỉ đưa raw chunk.

Điều kiện:

- Chỉ dùng cho listing docs có `postId`.
- Không dùng cho planning docs.

Logic:

1. Nếu doc không có `postId` và không có `planningDocumentId`, trả `None`.
2. Nếu là planning doc, trả `None`.
3. Tạo `text_blob` từ metadata + page_content để fallback extract.
4. Bắt đầu bằng header `=== BAT DONG SAN <postId> ===`.
5. Thêm các field nếu có hoặc extract được:
   - title
   - post type
   - district
   - address
   - price
   - area
   - bedrooms
   - bathrooms
   - furnishing
   - direction
   - min rental period
   - monthly cashflow
   - highlights
6. Compact raw evidence bằng `_compact_doc_content()`.
7. Thêm section `--- CHI TIET ---`.
8. Nếu số dòng quá ít, trả `None` để fallback sang raw compact.

Vai trò:

- LLM nhận context dạng ổn định, dễ đọc, ít phụ thuộc vào format bài đăng gốc.
- Tránh bỏ sót metadata quan trọng như price/area/bedrooms nếu raw content không có.

### 2.7. `_build_planning_evidence_contract(question, docs, max_facts)`

Mục đích: tạo một "hợp đồng evidence" cho planning fact/numeric questions.

Điều kiện:

- Chỉ chạy nếu docs không rỗng và `_is_planning_fact_question(question)` là `True`.

Logic:

1. Sinh markers bằng `_planning_contract_markers(question)`.
2. Duyệt từng doc theo rank.
3. Tách dòng planning bằng `_planning_context_lines()`.
4. Với mỗi dòng:
   - bỏ dòng quá ngắn.
   - đếm marker hits.
   - ghi nhận dòng có số liệu hay không.
   - ghi nhận dòng có core fact như `tong so`, `dien tich`, `du an`, `cong trinh`, `ha` hay không.
   - ghi nhận dòng đủ ngắn để đưa vào fact contract hay không.
5. Sort candidates theo:
   - marker hits giảm dần
   - ưu tiên dòng có số liệu/core fact/ngắn gọn
   - doc rank tăng dần
   - line index tăng dần
6. Dedupe dòng.
7. Chọn tối đa `max_facts`, nhưng ít nhất target là 4.
8. Nếu có dưới 2 facts, không tạo contract.
9. Trả block:

```text
EVIDENCE CONTRACT
- Use only facts [F#] below for planning and numeric statements.
...
[F1] ... (src: pid=...,chunk=...,idx=...)
```

Vai trò:

- System prompt có rule: nếu context có `EVIDENCE CONTRACT`, planning/numeric claims chỉ dùng facts `[F#]`.
- Đây là guardrail để LLM không lấy số phụ hoặc suy diễn từ đoạn dài.

### 2.8. `prepare_docs_for_context(question, docs, max_docs, max_chars_per_doc)`

`prepare_docs_for_context()` hien nam trong `app/rag/context_preparation.py`; day la ham quan trong nhat truoc khi goi LLM.

Mục đích:

- Chọn docs nào được đưa vào prompt.
- Compact từng doc.
- Dedupe.
- Với planning fact, thêm evidence contract.

Input:

- `question`: thường là retrieval query hoặc question đã rewrite.
- `docs`: docs retriever trả về.
- `max_docs`: default từ env `RAG_CONTEXT_MAX_DOCS`.
- `max_chars_per_doc`: default từ env `RAG_CONTEXT_MAX_CHARS_PER_DOC`.

Pha 1: validate và dedupe

1. Nếu docs rỗng, trả `[]`.
2. `max_docs >= 1`, `max_chars_per_doc >= 400`.
3. Dedupe theo `_doc_identity()`.

Pha 2: xác định mode

- `planning_only`: tất cả docs là planning docs.
- `planning_fact_query`: planning_only và question là planning fact.

Pha 3: select docs

Nếu planning_only:

- Không rerank lại sâu.
- Lấy `deduped[:max_docs]` vì planning retrieval ở `chat.py` đã rank/compact.

Nếu không planning_only:

1. Giữ nguyên thứ tự retriever sau dedupe.
2. Lấy `deduped[:max_docs]`.
3. Không rerank lại ở tầng chain bằng keyword score; nguồn rank chính là retriever/fallback trước đó.

Pha 4: compact từng doc

Với planning doc:

1. Dùng content đã có.
2. Planning char limit ít nhất 2600.

Với listing doc:

1. Thử `_build_structured_listing_context()`.
2. Nếu structured context có và post chưa structured trước đó:
   - dùng structured context.
   - nếu cần rich listing context, merge thêm raw excerpt từ `_compact_doc_content()`.
3. Nếu không structured, dùng `_compact_doc_content()`.

Sau đó:

- `sanitize_llm_text()`.
- Dedupe compacted content.
- Append vào `prepared`.

Pha 5: evidence contract cho planning fact query

Nếu `prepared` có docs và planning_only:

- Nếu là planning fact:
  - `_build_planning_evidence_contract()`.
  - Nếu contract có, insert contract thành doc đầu tiên với metadata `isPlanningEvidenceContract=True`.

Pha 6: fallback cuối

- Nếu mọi compact bị loại, giữ prefix của selected doc đầu tiên.

Vai trò:

- Đây là lớp cuối cùng kiểm soát prompt context.
- Listing context được làm giàu bằng metadata.
- Planning context được giữ theo rank đã tính ở `chat.py` nhưng thêm guardrail evidence contract.

### 2.9. `_build_citations(docs)`

Mục đích: tạo citations từ docs thật sự đưa vào prompt.

Logic:

1. Bỏ doc có metadata `isPlanningEvidenceContract`.
2. Dedupe listing theo `postId`.
3. Trả metadata citation cho cả listing và planning:
   - post/property fields
   - planning document fields
   - location/price/area/bedrooms
   - snippet 300 ký tự.

Vai trò:

- Citation phản ánh docs context sau compact/selection, không phải toàn bộ raw retrieval.

### 2.10. `RagChain.__init__()`

Mục đích: khởi tạo chain prompt.

Input:

- `llm`: ChatOpenAI hoặc LangChain chat model.
- `retriever`: object có `ainvoke(query)`.

Nó tạo `prompt_chain` mapping:

- `question`
- `context`
- `history`
- `answer_language`

rồi pipe vào `prompt` từ `app/rag/prompt.py`.

### 2.11. `RagChain.run(question, history, extra_context)`

Mục đích: chạy full RAG generation sau khi API đã chọn retriever.

Luồng chi tiết:

1. Nếu history `None`, đổi thành `[]`.
2. `retrieval_query = build_retrieval_query(question, history)`.
3. Bắt đầu timer retrieval.
4. `docs = await self.retriever.ainvoke(retrieval_query or question)`.
5. Tính `retrieval_seconds`.
6. Đọc env:
   - `RAG_CONTEXT_MAX_DOCS`
   - `RAG_CONTEXT_MAX_CHARS_PER_DOC`
7. `docs_for_context = prepare_docs_for_context(...)`.
8. `context = _format_docs(docs_for_context)`.
9. Nếu có `extra_context`, append vào context.
10. Detect language bằng `detect_lang(question)`.
11. Build payload:
    - question gốc
    - context
    - history
    - answer language
12. Tạo prompt value bằng `prompt_chain.ainvoke(payload)`.
13. Gọi LLM: `self.llm.ainvoke(prompt_value)`.
14. Lấy text answer bằng `message_content_to_text()`.
15. Extract token usage.
16. `postprocess_answer(question, answer)`.
17. Return `ChatResult`:
    - answer
    - citations từ `_build_citations(docs_for_context)`
    - token usage
    - timings

Điểm quan trọng:

- `question` đưa cho LLM vẫn là câu gốc, không phải retrieval query rewrite. Điều này giúp answer không bị lộ câu rewrite.
- `retrieval_query` chỉ dùng để lấy context.
- `extra_context` là cơ chế để `chat.py` đưa backend planning context và planning vector context vào prompt ngoài docs retriever.

## 3. Mapping trách nhiệm giữa `chat.py` và `chain.py`

| Trách nhiệm | `chat.py` | `chain.py` |
|---|---:|---:|
| Validate request/session | Có | Không |
| Lấy history từ DB | Có | Không |
| Quyết định listing/planning mode | Có | Không |
| Extract listing filters | Có | Không |
| Planning NL retrieval phức tạp | Có | Không |
| Build planning extra context từ backend | Có | Không |
| Gọi retriever tổng quát | Một phần | Có |
| Rewrite follow-up query | Gọi qua chain import | Có |
| Compact docs cho prompt | Không | Có |
| Evidence contract | Không | Có |
| Build prompt và gọi LLM | Không | Có |
| Postprocess answer | Không | Có |
| Lưu chat history | Có | Không |
| Merge/rerank citations | Có | Build citations cơ bản |

## 4. Cách đọc/debug một request thực tế

Khi cần debug vì câu trả lời sai, đi theo thứ tự sau:

1. Xác định mode:
   - `_has_planning_intent(req.message)` trả gì?
   - request có `planningContexts` không?

2. Nếu listing:
   - `extract_filters_from_query_with_usage()` ra filters gì?
   - `ListingFallbackRetriever` có dùng fallback SQL không?
   - `prepare_docs_for_context()` có bỏ mất dòng cần thiết không?

3. Nếu planning:
   - district/year extract đúng không?
   - query candidates có chứa phrasing phù hợp không?
   - base filter strict có quá hẹp không?
   - selected docs có đủ text/table không?
   - compact có cắt mất dòng answer không?

4. Nếu retrieval đúng nhưng answer sai:
   - kiểm tra `EVIDENCE CONTRACT` có được tạo không.
   - kiểm tra prompt context có số/dòng đúng không.
   - kiểm tra `postprocess_answer()` có rewrite/strip sai không.

5. Nếu citations sai:
   - kiểm tra docs trong `docs_for_context`.
   - kiểm tra planning citations merge từ `chat.py`.
   - kiểm tra `rerank_citations()` có nhận score/citation metadata đúng không.

