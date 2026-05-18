# 02. Dataset và golden

## Golden là gì

Một golden là đơn vị chuẩn để kiểm tra hệ thống. Nó mô tả:

- user hỏi gì,
- context tham chiếu là gì,
- output kỳ vọng cần chứa fact nào,
- case thuộc miền nào và khó ở mức nào,
- nếu cần, case đang nhắm tới entity cụ thể nào trong DB.

## Schema single-turn

```json
{
  "id": "ST_EASY_101",
  "domain": "real_estate",
  "difficulty": "easy",
  "question_type": "what",
  "input": "Mình đang xem tin căn hộ ở ... diện tích đang ghi bao nhiêu m2?",
  "context": ["postId=..., propertyId=..., title=..., price=..., area=..."],
  "expected_output_outline": ["Giá/diện tích trong dữ liệu: ..."],
  "target_metadata": {
    "postId": 231,
    "propertyId": 230,
    "evalOnly": true
  }
}
```

| Field | Ý nghĩa |
|---|---|
| `id` | Map giữa dataset, trace, test result và report. |
| `domain` | `real_estate` hoặc `land_use_planning`. |
| `difficulty` | Nhãn tầng độ khó dùng khi tổng hợp. |
| `question_type` | Thường là `what`, `how`, `why`. |
| `input` | Câu hỏi thật đưa vào runtime RAG. |
| `context` | Ground-truth context tham chiếu cho recall. |
| `expected_output_outline` | Các fact kỳ vọng. |
| `target_metadata` | Metadata hỗ trợ đánh giá hoặc force target. |

## `target_metadata` và `evalOnly`

`LangChainEvalAdapter.run_single_turn()` tách hai trường hợp:

1. `target_metadata.evalOnly = true`
   - metadata chỉ mô tả golden,
   - không ép vào retrieval,
   - phù hợp khi đo natural retrieval.

2. `target_metadata` không có `evalOnly`
   - có thể merge vào filter retrieval,
   - phù hợp khi cần kiểm tra một target cụ thể.

## Schema conversational

Golden conversational thêm `turns`:

```json
{
  "id": "CONV_HARD_001",
  "domain": "land_use_planning",
  "difficulty": "hard",
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "context": ["..."]
}
```

`run_conversation_eval.py` sẽ chạy từng user turn bằng cùng adapter single-turn rồi tích lũy history.

## Quy trình dựng dataset tốt

Repo có các runner lấy candidate từ dữ liệu DB thật, nhưng dataset final vẫn cần review thủ công:

1. Lấy candidate từ DB/pgvector.
2. Viết hoặc chỉnh `input` cho tự nhiên như người dùng thật.
3. Loại input trùng nguyên văn.
4. Giữ phân bố `real_estate` và `land_use_planning` gần cân bằng.
5. Review lại wording lặp, câu máy móc, nhãn difficulty không hợp lý.
6. Chỉ sau đó mới chạy DeepEval.

## Benchmark single-turn hiện tại

Tính đến `2026-05-17`, benchmark final single-turn gồm `700` case:

| Tập | Số case | Bất động sản | Quy hoạch |
|---|---:|---:|---:|
| Easy | 233 | 116 | 117 |
| Medium | 233 | 116 | 117 |
| Hard | 234 | 117 | 117 |

Dataset final:

- `single_turn_goldens_easy_233_v2.json`
- `single_turn_goldens_medium_233_v2.json`
- `single_turn_goldens_hard_234_v3.json`

## Vì sao có rebucket độ khó

Sau lần merge đầu, pass rate quan sát được bị đảo chiều so với nhãn difficulty. Để benchmark phản ánh thực nghiệm tốt hơn, 400 case mới đã được rebucket theo profile kết quả quan sát được, nhưng không sửa:

- score metric,
- answer,
- trace,
- số lượng case,
- phân bố domain.

Audit artifact:

- `single_turn_700_rebucket_allocation_20260517.json`
- `difficulty_rebucket_manifest_20260517.json`

Sau rebucket:

| Tập | CR Pass | Faithfulness Pass | AR Pass | Case Pass |
|---|---:|---:|---:|---:|
| Easy | 80.26% | 88.41% | 82.40% | 79.83% |
| Medium | 75.54% | 84.12% | 78.11% | 74.25% |
| Hard | 71.37% | 76.82% | 73.08% | 66.52% |

## Checklist trước khi thêm case

1. `input` có giống user thật hỏi không?
2. `input` có trùng nguyên văn case cũ không?
3. `expected_output_outline` có chứa fact thật không?
4. `context` có đủ bằng chứng không?
5. `target_metadata` có nên là `evalOnly` không?
6. Domain balance có bị lệch không?
7. Nhãn difficulty có còn hợp lý sau khi xem cả wording lẫn độ khó retrieval không?

