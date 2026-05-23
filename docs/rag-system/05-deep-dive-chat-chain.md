# 05. Deep Dive Chat Runtime

File này giải thích kỹ hơn các hàm dài và các cụm logic quan trọng trong:

- `app/api/chat.py`
- `app/rag/planning_retrieval.py`
- `app/rag/chain.py`

Mục tiêu là đọc code theo hướng "hàm này nhận gì, làm gì, vì sao cần, và ảnh hưởng tới RAG thế nào".

## 1. `app/api/chat.py`

`chat.py` là tầng orchestration của API `/api/chat`: đọc history, quyết định listing/planning mode, gọi retriever phù hợp, gọi `RagChain`, merge citations và lưu lịch sử. Logic planning retrieval dài đã được tách sang `app/rag/planning_retrieval.py`.

### 1.1. Nhóm normalize và nhận diện planning intent

Các helper normalize được dùng trong runtime qua alias `_normalize_nl` và `_strip_accents`, implementation nằm ở `app/utils/text.py`.

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

Đây là hàm orchestration quan trọng nhất trong planning runtime. Phần xử lý đã được tách thành helper nhỏ để đọc theo từng bước:

- `_resolve_planning_retrieval_context(...)`: gom district/year/fact query/final_k/probe_k/query candidates.
- `_planning_base_filter_candidates(...)`: sinh filter từ hẹp tới rộng.
- `_rank_planning_docs_for_filter(...)`: retrieve vector theo nhiều query candidates và rank bằng RRF.
- `_planning_scope_pools(...)`: chia pool strict/district/broad trước khi selector và compact.

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

#### Pha 7: select v? compact

V?i m?i scope pool:

1. `_select_ranked_planning_docs()` ch?n docs t?t nh?t theo th? t? vector/RRF v? scope.
2. `_compact_planning_docs()` c?t g?n n?i dung tr??c khi ??a v?o prompt.
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

#### `rerank_citations(citations)`

Muc dich: sap lai citations de nguon planning va nguon co score dung truoc.

Ordering:

- uu tien citation planning document neu co `planningDocumentId`;
- sau do dung `score` co san tu retriever neu citation co field nay;
- neu khong co score, giu thu tu xuat hien ban dau.
### 1.12. `chat(req, db)`

Muc dich: entry point runtime `/api/chat`.

Lu?ng ch?nh:

1. Validate `userId`.
2. Tao hoac lay `sessionId`.
3. Lay 6 messages gan nhat.
4. Rewrite retrieval query cho follow-up.
5. Lay `top_k` tu env `TOP_K_DEFAULT`.
6. Tao LLM.
7. Chon mode bang `_has_planning_intent(req.message)`.
8. Neu listing mode:
   - extract listing filters;
   - init listing vector store;
   - build primary retriever bang `build_retriever(vs, k=top_k, filters=filters)`;
   - wrap bang `ListingFallbackRetriever`.
9. Neu planning mode:
   - init planning vector store;
   - goi `_retrieve_planning_docs_for_nl_query(planning_vs, retrieval_message, top_k, history_messages=history)`;
   - neu co docs, build planning context va citations;
   - doi retriever thanh `StaticDocumentsRetriever(planning_docs)`.
10. Chay `RagChain.run()`.
11. Merge/rerank citations.
12. Luu user/assistant message.
13. Tra `ChatResponse` gom answer, citations, filters, token usage va timings.

Diem can chu y:

- `topK` va `planningContexts` khong con la public request fields.
- Filter extraction chi chay o listing mode.
- Planning retrieval timeout se fallback `[]`; khi khong co context, prompt se yeu cau noi ro thong tin khong co trong retrieved context.

## 2. `app/rag/chain.py`

`chain.py` hien la thin orchestrator cho RAG runtime. Neu `chat.py` quyet dinh lay docs nao, thi cac module RAG runtime quyet dinh rewrite query, compact context, goi LLM va postprocess answer.

Module split hien tai:

- `app/rag/chain.py`: orchestration, prompt call, timings, citations.
- `app/rag/query_rewrite.py`: `build_retrieval_query()`.
- `app/rag/query_intents.py`: intent flags dung chung cho context va postprocess.
- `app/rag/listing_context.py`: structured listing context từ metadata + snippet.
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

M?c ??ch: l?m s?ch answer sau LLM b?ng c?c rule chung, kh?ng rewrite theo t?ng m?u c?u h?i eval.

C?c b??c ch?nh:

1. N?u answer r?ng, tr? nguy?n.
2. Strip answer.
3. Lo?i generic trailing invites, v? d? "N?u b?n c?n th?m...".
4. Build query intents.
5. N?u l? suitability query m? kh?ng h?i gi?, b? d?ng gi? kh?ng c?n thi?t.
6. B? th?ng tin li?n h? n?u user kh?ng h?i li?n h?.
7. V?i c?u h?i kh?ng gian r?ng/tho?ng, b? v?i d?ng ti?n ?ch xung quanh kh?ng tr?c ti?p li?n quan.
8. Tr? answer ?? l?m s?ch.

Vai tr?:

- Gi? h?u x? l? ? m?c d? gi?i th?ch.
- Kh?ng c?n c?c transform h?p cho t?ng wording planning/listing.
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

- `postprocess_answer()` dùng flags để bỏ thông tin không được hỏi.
- `build_structured_listing_context()` dùng flags để chọn highlights phù hợp.

### 2.4. `_compact_doc_content(question, content, max_chars, max_lines)`

M?c ??ch: r?t g?n content listing/raw doc m? kh?ng rerank d?ng b?ng score th? c?ng.

Logic:

1. Thay `<br/>`, `<br>` th?nh newline.
2. `_dedupe_repeated_blocks()` ?? b? block tr?ng.
3. N?u content ?? ng?n v? ?t d?ng, tr? nguy?n.
4. Split th?nh lines.
5. D?ng qu? d?i ???c chia b?ng `_split_long_line()`.
6. Dedupe line theo normalized text.
7. Gi? header ??u ti?n n?u c?.
8. L?y c?c d?ng c?n l?i theo th? t? t? nhi?n, kh?ng v??t `max_lines` v? `max_chars`.
9. N?u kh?ng ch?n ???c g?, fallback prefix content.

Vai tr?:

- Gi?m context d?i/l?p t? b?i ??ng b?t ??ng s?n.
- Tr?nh th?m m?t l?p keyword score kh? gi?i th?ch ? t?ng context preparation.
### 2.6. `build_structured_listing_context(question, doc)`

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

### 2.8. `prepare_docs_for_context(question, docs, max_docs, max_chars_per_doc)`

`prepare_docs_for_context()` hien nam trong `app/rag/context_preparation.py`; day la ham quan trong nhat truoc khi goi LLM.

Mục đích:

- Chọn docs nào được đưa vào prompt.
- Compact từng doc.
- Dedupe.

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

1. Thử `build_structured_listing_context()` từ `app/rag/listing_context.py`.
2. Nếu structured context có và post chưa structured trước đó:
   - dùng structured context.
   - nếu cần rich listing context, merge thêm raw excerpt từ `_compact_doc_content()`.
3. Nếu không structured, dùng `_compact_doc_content()`.

Sau đó:

- `sanitize_llm_text()`.
- Dedupe compacted content.
- Append vào `prepared`.

Pha 5: fallback cu?i

- N?u m?i compact b? lo?i, gi? prefix c?a selected doc ??u ti?n.
