# 03. Luồng ingest và lưu trữ vector

## Listing ingest: `/api/ingest/posts`

File chính: `app/api/ingest.py`.

### Data model

`IngestPost`:

- `postId`
- `content`
- `metadata`

`IngestRequest`:

- `posts: list[IngestPost]`

### Luồng POST `/api/ingest/posts`

```text
1. Nếu request không có posts -> return ingestedChunks=0.
2. initialize_listing_vector_store().
3. Với từng post:
   - strip content
   - split bằng RecursiveCharacterTextSplitter
   - mỗi chunk tạo Document(page_content=chunk, metadata={...metadata, postId, chunkIndex})
4. vs.aadd_documents(docs).
5. Trả số chunks đã ingest.
```

Splitter listing lấy từ `_get_splitter()`:

- `LISTING_CHUNK_SIZE`, mặc định `900`
- `LISTING_CHUNK_OVERLAP`, mặc định `120`

### Update listing embeddings

PUT `/api/ingest/posts/{post_id}`:

```text
1. Check body postId khớp URL.
2. DELETE old chunks trong langchain_pg_embedding theo cmetadata->>'postId'.
3. Nếu content rỗng -> chỉ trả deletedChunks.
4. Split content mới.
5. aadd_documents().
```

### Delete listing embeddings

DELETE `/api/ingest/posts/{post_id}`:

```text
DELETE FROM langchain_pg_embedding
WHERE cmetadata->>'postId' = :post_id
```

## Planning ingest: `/api/planning/ingest-documents`

Phần dưới mô tả luồng ở mức pipeline. Nếu cần giải thích sâu từng chiến lược OCR/chunking/metadata để đưa vào báo cáo, xem thêm [06-deep-dive-ingestion-chunking.md](06-deep-dive-ingestion-chunking.md).

File chính:

- API: `app/api/planning.py`
- Build documents: `app/planning/ingestion.py`
- Config/env resolver: `app/planning/ingestion_config.py`

### Data model

`PlanningIngestDocument`:

- `planningDocumentId`
- `title`
- `sourceUrl`
- `format`
- `documentType`
- `dossierCode`
- `city`
- `district`
- `planYear`
- `propertyId`
- `rawMeta`

`PlanningIngestRequest`:

- `replaceExisting`: mặc định `True`
- `skipIfExists`: mặc định `True`
- `documents`

### Luồng API planning ingest

```text
1. Nếu documents rỗng -> return.
2. initialize_planning_vector_store().
3. Với từng document:
   - nếu replaceExisting: xóa chunks cũ theo planningDocumentId.
   - nếu không replaceExisting và skipIfExists: check đã tồn tại thì skip.
   - tạo PlanningIngestPayload.
   - build_planning_documents(payload).
   - vs.aadd_documents(docs).
   - ghi result item gồm deleted/ingested/text/table/skipped/reason.
4. Trả tổng ingestedChunks và failedDocuments.
```

## `build_planning_documents()`

Đây là lõi ingest planning.

Luồng chi tiết:

```text
1. Đọc chunking mode từ PLANNING_CHUNKING_MODE.
2. Download file bằng fetch_document_bytes().
3. Xác định format: pdf hoặc txt.
4. PDF:
   - extract text/page_texts bằng pypdf.
   - estimate quality score.
   - nếu không có text layer hoặc quality thấp -> force OCR.
5. Tạo page line maps để truy ngược page/line.
6. Nếu cần OCR:
   - _ocr_structural_chunks().
7. Nếu không có OCR chunks:
   - hierarchical chunks nếu mode hierarchical.
   - fallback structural chunks nếu hierarchical không ra kết quả.
8. Merge continuation chunks nếu không dùng hierarchical chuẩn.
9. Loại chunk quá yếu.
10. Build metadata chuẩn.
11. Với từng structural chunk:
    - xác định chunkType text/table.
    - locate page/line span.
    - set chunkIndex, globalChunkIndex.
    - build sourceLocator.
    - enrich content cho embedding bằng metadata descriptor.
    - tạo LangChain Document.
12. Trả docs và counts.
```

## Planning chunking modes

Các mode trong `app/planning/ingestion.py`:

- `planning_baseline_fixed`
- `planning_hierarchical_leaf`
- `planning_hierarchical_parent_context`
- `planning_hierarchical_parent_child`

Default: `planning_hierarchical_parent_context`.

Hierarchical chunking giữ cấu trúc mục/tiểu mục tốt hơn fixed splitting. Điều này rất quan trọng với tài liệu quy hoạch vì số liệu thường nằm trong bảng hoặc đoạn phụ thuộc heading phía trước.

## Metadata planning

Mỗi `Document` planning được lưu với metadata:

| Field | Ý nghĩa |
|---|---|
| `documentScope` | Luôn là `planning`. |
| `planningDocumentId` | ID tài liệu quy hoạch. |
| `title` | Tên tài liệu. |
| `sourceUrl` | URL nguồn. |
| `format` | PDF/TXT... |
| `documentType` | Loại tài liệu nếu backend gửi. |
| `dossierCode` | Mã hồ sơ. |
| `city`, `district`, `districtCanonical`, `districtRaw` | Địa bàn đã canonicalize. |
| `planYear` | Năm kế hoạch. |
| `propertyId` | Property liên quan nếu có. |
| `chunkType` | `text` hoặc `table`. |
| `chunkIndex` | Index riêng theo loại chunk. |
| `globalChunkIndex` | Index toàn tài liệu. |
| `pageNumber`, `lineStart`, `lineEnd` | Vị trí nguồn. |
| `sourceLocator` | Chuỗi định vị page/line. |
| `chunker` | Cách chunk được tạo. |
| `sectionHeading`, `hierarchyPath`, `hierarchyLevel` | Thông tin cấu trúc. |
| `parentChunkId`, `siblingIndex`, `isParentChunk` | Quan hệ parent/child. |
| `chunkingMode`, `chunkingFallback` | Mode chunking và fallback flag. |
| `chunkPreview` | Preview ngắn. |

## Enrich content trước khi embedding

`_enrich_for_embedding(content, metadata)` thêm descriptor đầu chunk:

```text
document_type:... | district:... | plan_year:... | chunk_type:... | title:...
<chunk content>
```

Mục đích:

- Tăng khả năng vector search match theo quận/năm/loại tài liệu/title.
- Giúp retrieval không chỉ dựa vào nội dung OCR thô.

## Lưu trữ trong pgvector

`PGVector.aadd_documents(docs)` ghi vào LangChain tables:

- `langchain_pg_collection`: collection name.
- `langchain_pg_embedding`: document text, embedding vector, `cmetadata`.

Do `use_jsonb=True`, metadata nằm trong `cmetadata` dạng JSONB, giúp filter SQL bằng `cmetadata->>'field'`.

## API kiểm tra ingest

Listing:

- GET `/api/ingest/posts`
- GET `/api/ingest/posts/{post_id}`

Planning:

- GET `/api/planning/ingested-documents`
- GET `/api/planning/ingested-documents/{planning_document_id}/chunks`

Các endpoint này query trực tiếp `langchain_pg_embedding` để xem số chunk, metadata mẫu và text preview.
