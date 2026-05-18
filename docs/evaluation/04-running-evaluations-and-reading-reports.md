# 04. Chạy evaluation và đọc report

## Điều kiện trước khi chạy

- `.env` có DB/vector env phù hợp.
- Có `DEEPSEEK_API_KEY` cho judge mặc định.
- Collection vector đã ingest.
- Đã cài dependency:

```bash
pip install -e .
```

## Chạy single-turn

```bash
python -m evaluation.runners.run_single_turn_eval
```

Ví dụ chạy dataset cụ thể:

```bash
python -m evaluation.runners.run_single_turn_eval \
  --dataset single_turn_goldens_easy_233_v2.json \
  --output-name experiments/single_turn_700_v2_20260515/easy_run.json
```

Option quan trọng:

| Option | Khi dùng |
|---|---|
| `--max-cases` | Smoke run. |
| `--trace-max-concurrent` | Số request RAG chạy song song lúc sinh trace. |
| `--max-concurrent` | Số metric judge chạy song song. |
| `--sync` | Chạy judge tuần tự để ổn định hơn. |
| `--trace-input` | Chấm lại từ trace cũ. |
| `--eval-batch-size` | Chia batch khi scoring. |
| `--manual-metric-loop` | Debug khi `deepeval.evaluate()` bị treo. |
| `--top-k` | Override runtime top-k. |

## Chạy conversational

```bash
python -m evaluation.runners.run_conversation_eval \
  --dataset conversational_goldens_tuned.json \
  --output-name full_conversation.json
```

Các option riêng:

- `--resume-trace-input`
- `--trace-attempts`

## Trace checkpoint

Single-turn runner luôn ghi:

```text
<output-name>.traces.json
```

Trace checkpoint hữu ích khi cần chấm lại mà không gọi lại RAG runtime.

## Profile runtime không chấm metric

```bash
python -m evaluation.runners.profile_single_turn_runtime \
  --dataset single_turn_goldens_easy_233_v2.json \
  --output-name runtime_easy_profile.json
```

Mặc định đo natural retrieval; thêm `--use-target-metadata` nếu muốn dùng metadata target khi retrieve.

## Schema report single-turn

```text
run_id
created_at
phase
judge
thresholds
settings
deepeval_timeout
runner
experiment_config
summary
timing
judge_cost_profile
scorecard
deepeval_result
traces
```

Các vùng cần đọc:

| Vùng | Ý nghĩa |
|---|---|
| `summary` | Số case, difficulty, domain. |
| `timing` | Thời gian sinh trace và chấm judge. |
| `scorecard.metric_statistics` | Mean/pass từng metric. |
| `scorecard.case_composite_statistics` | Case pass và strict pass. |
| `deepeval_result.test_results` | Chi tiết từng case. |
| `traces` | Golden + runtime trace dùng để sinh testcase. |

## Điều kiện report final hợp lệ

1. Schema top-level giữ đúng format cũ.
2. `deepeval_result` có:
   - `test_results`
   - `confident_link`
   - `run_duration`
3. `len(test_results) == len(traces)`.
4. `summary.cases` khớp số testcase.
5. `scorecard` được tính từ `test_results`, không sửa tay.
6. Nếu có merge/rebucket, nguồn và audit artifact phải được giữ lại.

## Đọc report đúng

Ví dụ easy final hiện tại:

- CR pass: `80.26%`
- Faithfulness pass: `88.41%`
- AR pass: `82.40%`
- Case pass: `79.83%`
- Strict all metric pass: `63.52%`

Không nên kết luận chỉ từ một chỉ số tổng. Nếu kết quả lạ, cần mở xuống `test_results` và `traces`.

## Report final và report trung gian

Trong `single_turn_700_v2_20260515/`:

- `*_new_*_full.json`: report 400 case mới đã chạy thật.
- `*_merged_report*.json`: report final sau merge.
- `single_turn_700_merge_summary.json`: tóm tắt tổng hợp.
- `difficulty_rebucket_manifest_20260517.json`: audit rebucket.

Khi upload artifact final hoặc viết luận văn, dùng merged report cuối cùng.

