# RAG Experiment Summary

## Top by CR

| Rank | Config | Mean | Pass Rate | Top-K | Cost Proxy | Runtime Avg | Chunking |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | bge_m3_ghr1__planning_hierarchical_parent_context__k16 | 0.792667 | 0.7 | 16 | 7.65258 | 5.76626 | planning_hierarchical_parent_context |
| 2 | bge_m3_ghr1__planning_hierarchical_parent_context__k24 | 0.791 | 0.7 | 24 | 8.568205 | 6.60072 | planning_hierarchical_parent_context |
| 3 | bge_m3_ghr1__planning_hierarchical_parent_context__k8 | 0.787667 | 0.7 | 8 | 7.64533 | 6.27952 | planning_hierarchical_parent_context |
| 4 | multilingual_e5_base_ghr1__planning_hierarchical_parent_context__k16 | 0.784 | 0.7 | 16 | 7.846885 | 5.95765 | planning_hierarchical_parent_context |
| 5 | multilingual_e5_base_ghr1__planning_hierarchical_parent_context__k24 | 0.782333 | 0.71 | 24 | 8.60769 | 6.6382 | planning_hierarchical_parent_context |
| 6 | multilingual_e5_base_ghr1__planning_hierarchical_parent_context__k8 | 0.782333 | 0.7 | 8 | 6.847285 | 5.48531 | planning_hierarchical_parent_context |
| 7 | multilingual_e5_base_ghr1__planning_baseline_fixed__k8 | 0.715667 | 0.64 | 8 | 6.797283 | 5.72585 | planning_baseline_fixed |
| 8 | multilingual_e5_small_ghr1__planning_baseline_fixed__k8 | 0.712 | 0.63 | 8 | 7.129605 | 6.06034 | planning_baseline_fixed |
| 9 | bge_m3_ghr1__planning_baseline_fixed__k8 | 0.706 | 0.63 | 8 | 7.839387 | 6.76412 | planning_baseline_fixed |
| 10 | bge_m3_ghr1__planning_baseline_fixed__k16 | 0.706 | 0.62 | 16 | 8.938852 | 7.52201 | planning_baseline_fixed |

## Top by Faithfulness

| Rank | Config | Mean | Pass Rate | Top-K | Cost Proxy | Runtime Avg | Chunking |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | multilingual_e5_small_ghr1__planning_baseline_fixed__k16 | 0.97892 | 1.0 | 16 | 7.215138 | 5.80445 | planning_baseline_fixed |
| 2 | bge_m3_ghr1__planning_baseline_fixed__k8 | 0.973753 | 0.989899 | 8 | 7.839387 | 6.76412 | planning_baseline_fixed |
| 3 | multilingual_e5_small_ghr1__planning_hierarchical_parent_context__k8 | 0.973256 | 0.979798 | 8 | 5.720218 | 4.79159 | planning_hierarchical_parent_context |
| 4 | multilingual_e5_small_ghr1__planning_hierarchical_parent_context__k24 | 0.971754 | 0.97 | 24 | 6.332473 | 4.96107 | planning_hierarchical_parent_context |
| 5 | multilingual_e5_small_ghr1__planning_hierarchical_parent_child__k24 | 0.971603 | 0.98 | 24 | 7.79546 | 6.25568 | planning_hierarchical_parent_child |
| 6 | multilingual_e5_base_ghr1__planning_baseline_fixed__k16 | 0.970528 | 0.98 | 16 | 7.846777 | 6.42702 | planning_baseline_fixed |
| 7 | bge_m3_ghr1__planning_baseline_fixed__k16 | 0.970032 | 0.97 | 16 | 8.938852 | 7.52201 | planning_baseline_fixed |
| 8 | multilingual_e5_small_ghr1__planning_hierarchical_parent_child__k8 | 0.969702 | 0.95 | 8 | 7.25937 | 6.19493 | planning_hierarchical_parent_child |
| 9 | bge_m3_ghr1__planning_hierarchical_parent_context__k8 | 0.969115 | 0.96 | 8 | 7.64533 | 6.27952 | planning_hierarchical_parent_context |
| 10 | bge_m3_ghr1__planning_baseline_fixed__k24 | 0.967751 | 0.97 | 24 | 8.032887 | 6.53488 | planning_baseline_fixed |

## Top by AR

| Rank | Config | Mean | Pass Rate | Top-K | Cost Proxy | Runtime Avg | Chunking |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | multilingual_e5_base_ghr1__planning_hierarchical_parent_context__k16 | 0.890056 | 0.9 | 16 | 7.846885 | 5.95765 | planning_hierarchical_parent_context |
| 2 | bge_m3_ghr1__planning_hierarchical_parent_context__k8 | 0.889683 | 0.9 | 8 | 7.64533 | 6.27952 | planning_hierarchical_parent_context |
| 3 | bge_m3_ghr1__planning_hierarchical_parent_context__k24 | 0.887698 | 0.88 | 24 | 8.568205 | 6.60072 | planning_hierarchical_parent_context |
| 4 | bge_m3_ghr1__planning_hierarchical_parent_context__k16 | 0.88481 | 0.9 | 16 | 7.65258 | 5.76626 | planning_hierarchical_parent_context |
| 5 | multilingual_e5_base_ghr1__planning_hierarchical_parent_context__k24 | 0.87794 | 0.88 | 24 | 8.60769 | 6.6382 | planning_hierarchical_parent_context |
| 6 | multilingual_e5_base_ghr1__planning_hierarchical_parent_context__k8 | 0.87031 | 0.84 | 8 | 6.847285 | 5.48531 | planning_hierarchical_parent_context |
| 7 | bge_m3_ghr1__planning_baseline_fixed__k8 | 0.810337 | 0.82 | 8 | 7.839387 | 6.76412 | planning_baseline_fixed |
| 8 | bge_m3_ghr1__planning_baseline_fixed__k16 | 0.807484 | 0.83 | 16 | 8.938852 | 7.52201 | planning_baseline_fixed |
| 9 | multilingual_e5_base_ghr1__planning_baseline_fixed__k16 | 0.803291 | 0.8 | 16 | 7.846777 | 6.42702 | planning_baseline_fixed |
| 10 | multilingual_e5_small_ghr1__planning_baseline_fixed__k16 | 0.803262 | 0.81 | 16 | 7.215138 | 5.80445 | planning_baseline_fixed |
