# 05. Thực nghiệm RAG và artifact hiện tại

## Mục đích

`run_rag_experiment_matrix.py` dùng để chọn cấu hình RAG trước khi đánh giá chatbot cuối cùng:

- embedding model,
- planning chunking mode,
- `top-k`.

## Không gian cấu hình mặc định

| Trục | Giá trị |
|---|---|
| Embedding | `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base`, `BAAI/bge-m3` |
| Chunking | `planning_baseline_fixed`, `planning_hierarchical_parent_context`, `planning_hierarchical_parent_child` |
| Top-k | `8`, `16`, `24` |

Tổng cộng có `27` cấu hình.

## Smoke và full

- `smoke`: chạy subset nhỏ cân bằng domain để loại cấu hình yếu hoặc lỗi sớm.
- `full`: chạy tập lớn hơn trên các cấu hình đã qua sàng lọc.

## Cách runner chạy một config

1. Set env cho embedding model, post collection, planning collection, chunking mode và top-k.
2. Đảm bảo collection đã ingest.
3. Gọi `run_single_turn_eval.py` dưới dạng subprocess.
4. Bóc scorecard/runtime ra `summary.csv`.

## Cách chọn winner

Runner không chỉ sort theo CR:

1. Loại config fail.
2. Dùng baseline `multilingual_e5_small + baseline_fixed + k16`.
3. Loại config có Faithfulness hoặc AR tụt quá `0.02` so với baseline.
4. Lấy quality tier gần best CR.
5. Tie-break theo CR, Faithfulness, AR, rồi cost/runtime.

## Cấu hình đang được chọn

Kết quả hiện tại dùng trong luận văn chọn:

```text
intfloat/multilingual-e5-base
+ planning_hierarchical_parent_context
+ top-k = 16
```

Lý do:

- chất lượng gần nhóm dẫn đầu,
- AR tốt,
- phù hợp hơn với môi trường tài nguyên giới hạn,
- `top-k = 16` cân bằng hơn giữa ngữ cảnh và chi phí.

## Artifact benchmark single-turn 700

Thư mục:

```text
evaluation/reports/experiments/single_turn_700_v2_20260515/
```

Dataset final:

- `evaluation/datasets/single_turn_goldens_easy_233_v2.json`
- `evaluation/datasets/single_turn_goldens_medium_233_v2.json`
- `evaluation/datasets/single_turn_goldens_hard_234_v3.json`

Report final:

- `easy_single_turn_233_merged_report.json`
- `medium_single_turn_233_merged_report.json`
- `hard_single_turn_234_merged_report_v3.json`

Audit artifact:

- `dataset_generation_and_input_review_manifest.json`
- `single_turn_700_merge_summary.json`
- `single_turn_700_rebucket_allocation_20260517.json`
- `difficulty_rebucket_manifest_20260517.json`

## Kết quả final sau rebucket ngày 2026-05-17

| Tập | Cases | CR mean | CR pass | F mean | F pass | AR mean | AR pass | Case pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 233 | 0.823820 | 80.26% | 0.895238 | 88.41% | 0.866960 | 82.40% | 79.83% |
| Medium | 233 | 0.767883 | 75.54% | 0.878348 | 84.12% | 0.803377 | 78.11% | 74.25% |
| Hard | 234 | 0.737179 | 71.37% | 0.835687 | 76.82% | 0.774452 | 73.08% | 66.52% |

## Cách hiểu rebucket đúng

Rebucket không phải là sửa report cho đẹp số. Nó chỉ đổi lại difficulty bucket của 400 case mới dựa trên profile thực nghiệm đã quan sát được:

- score metric giữ nguyên,
- answer giữ nguyên,
- trace giữ nguyên,
- final report được dựng lại từ case-level result thật.

Nếu runtime, prompt, expected output hoặc metric definition đổi, phải chạy lại evaluation thay vì tái dùng artifact cũ.

