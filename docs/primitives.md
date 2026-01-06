# HWX primitives reference

This page lists the public primitives exported by the HWX crate (primarily via `hwx::dispatch`, re-exported at the crate root).

_Generated from `src/dispatch.rs` on 2026-01-06._

Notes:
- Most APIs are designed to be allocation-light and use in-place mutation where practical (e.g. filtering/sorting/dedup).
- Many functions dispatch between scalar / CPU SIMD / CUDA depending on platform capabilities and input sizes.

## Hardware detection

| Function | What it does |
|---|---|
| `get_hw_capabilities` | Detect available CPU/GPU capabilities at runtime. |
| `has_hw_support` | Check whether a specific instruction set / backend is supported. |

## Distances / similarity

| Function | What it does |
|---|---|
| `distance_cosine_f32` | Cosine distance between two f32 slices. |
| `distance_dot_f32` | Dot product between two f32 slices. |
| `distance_hamming_u16` | Hamming distance between two u16 slices (interpreted as bitstrings). |
| `distance_hamming_u32` | Hamming distance between two u32 slices (interpreted as bitstrings). |
| `distance_hellinger_f32` | Hellinger distance between two f32 slices. |
| `distance_jaccard_u16` | Jaccard distance between two u16 slices. |
| `distance_jeffreys_f32` | Jeffreys divergence between two f32 slices. |
| `distance_jensen_shannon_f32` | Jensen–Shannon divergence between two f32 slices. |
| `distance_l1_f32` | Manhattan (L1) distance between two f32 slices. |
| `distance_l2_f32` | Euclidean (L2) distance between two f32 slices. |
| `distance_levenshtein_u16` | Levenshtein edit distance between two u16 slices. |

## Set operations (sorted arrays)

| Function | What it does |
|---|---|
| `intersect_sorted_u32` | In-place intersection of sorted arrays (optionally dedup and/or descending depending on args). |
| `intersect_sorted_u64` | In-place intersection of sorted arrays (optionally dedup and/or descending depending on args). |
| `set_difference_sorted_u32` | In-place set difference (A \ B) for sorted arrays. |
| `union_sorted_u32` | Union of sorted arrays. |

## Deduplication (sorted arrays)

| Function | What it does |
|---|---|
| `dedup_sorted_f64` | In-place deduplication of a sorted array. |
| `dedup_sorted_i64` | In-place deduplication of a sorted array. |
| `dedup_sorted_u32` | In-place deduplication of a sorted array. |
| `dedup_sorted_u64` | In-place deduplication of a sorted array. |

## Filtering

| Function | What it does |
|---|---|
| `filter_counts_ge_threshold_u64` | In-place filter based on per-item counts/threshold. |
| `filter_range_f64` | In-place filter: keep values within an inclusive range. |
| `filter_range_u32` | In-place filter: keep values within an inclusive range. |
| `filter_range_u64` | In-place filter: keep values within an inclusive range. |
| `filter_regex_terms` | In-place filter of terms by a compiled regex (bytes regex). |
| `filter_u32` | In-place filter primitive. |
| `filter_u32_by_u64_range` | Filter doc IDs using a parallel u64 range (typically timestamps). |
| `filter_wildcard_terms` | In-place filter of terms by a wildcard pattern. |

## Sorting

| Function | What it does |
|---|---|
| `sort_f64` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |
| `sort_i64` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |
| `sort_strings` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |
| `sort_u32` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |
| `sort_u32_by_f64` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |
| `sort_u32_by_i64` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |
| `sort_u32_by_u64` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |
| `sort_u64` | Sorting primitive (in-place; may use SIMD/CUDA depending on size/backend). |

## Search (sorted arrays)

| Function | What it does |
|---|---|
| `binary_search_ge_time` | Search primitive on ascending-sorted arrays. |
| `binary_search_ge_u32` | Search primitive on ascending-sorted arrays. |
| `binary_search_ge_u64` | Search primitive on ascending-sorted arrays. |
| `binary_search_le_time` | Search primitive on ascending-sorted arrays. |
| `binary_search_le_u32` | Search primitive on ascending-sorted arrays. |
| `binary_search_le_u64` | Search primitive on ascending-sorted arrays. |
| `exponential_search_ge_u32` | Search primitive on ascending-sorted arrays (good when target is near the start). |
| `exponential_search_ge_u64` | Search primitive on ascending-sorted arrays (good when target is near the start). |
| `exponential_search_le_u32` | Search primitive on ascending-sorted arrays (good when target is near the start). |
| `exponential_search_le_u64` | Search primitive on ascending-sorted arrays (good when target is near the start). |
| `is_sorted_u32` | Check whether an array is sorted ascending. |

## Reductions / stats

| Function | What it does |
|---|---|
| `avg_f64` | Time-series/statistics primitive. |
| `changes_f64` | Time-series/statistics primitive. |
| `deriv_f64` | Time-series/statistics primitive. |
| `find_min_max_f64` | Compute min and max in one pass. |
| `find_min_max_i64` | Compute min and max in one pass. |
| `find_min_max_u32` | Compute min and max in one pass. |
| `increase_f64` | Time-series/statistics primitive. |
| `mad_f64` | Time-series/statistics primitive. |
| `predict_linear_f64` | Time-series/statistics primitive. |
| `present_f64` | Time-series/statistics primitive. |
| `reduce_max_f64` | Reduction: maximum. |
| `reduce_max_u32` | Reduction: maximum. |
| `reduce_min_f64` | Reduction: minimum. |
| `reduce_min_u32` | Reduction: minimum. |
| `reduce_sum_f64` | Reduction: sum. |
| `reduce_sum_u32` | Reduction: sum. |
| `reduce_sum_u64` | Reduction: sum. |
| `reduce_weighted_sum_f64` | Reduction: sum. |
| `resets_f64` | Time-series/statistics primitive. |
| `stdvar_f64` | Time-series/statistics primitive. |

## Percentiles / interpolation / quantization

| Function | What it does |
|---|---|
| `calculate_multi_percentiles_f64` | Percentile calculation primitive. |
| `calculate_percentiles_batch_f64` | Percentile calculation primitive. |
| `fma_f64` | Numeric helper primitive. |
| `linear_interpolate_f64` | Numeric helper primitive. |
| `vectorized_quantize_f64` | Numeric helper primitive. |
| `vectorized_subtract_u32` | Numeric helper primitive. |

## Element-wise arithmetic (f64)

| Function | What it does |
|---|---|
| `add_f64` | Element-wise arithmetic over f64 slices. |
| `divide_f64` | Element-wise arithmetic over f64 slices. |
| `modulo_f64` | Element-wise arithmetic over f64 slices. |
| `multiply_f64` | Element-wise arithmetic over f64 slices. |
| `power_f64` | Element-wise arithmetic over f64 slices. |
| `subtract_f64` | Element-wise arithmetic over f64 slices. |

## Element-wise comparisons (f64)

| Function | What it does |
|---|---|
| `equal_f64` | Element-wise comparison over f64 slices (writes a mask/output slice). |
| `greater_equal_f64` | Element-wise comparison over f64 slices (writes a mask/output slice). |
| `greater_than_f64` | Element-wise comparison over f64 slices (writes a mask/output slice). |
| `less_equal_f64` | Element-wise comparison over f64 slices (writes a mask/output slice). |
| `less_than_f64` | Element-wise comparison over f64 slices (writes a mask/output slice). |
| `not_equal_f64` | Element-wise comparison over f64 slices (writes a mask/output slice). |

## Element-wise math transforms (f64)

| Function | What it does |
|---|---|
| `abs_f64` | In-place math transform over f64 slices. |
| `acos_f64` | In-place math transform over f64 slices. |
| `acosh_f64` | In-place math transform over f64 slices. |
| `asin_f64` | In-place math transform over f64 slices. |
| `asinh_f64` | In-place math transform over f64 slices. |
| `atan_f64` | In-place math transform over f64 slices. |
| `atanh_f64` | In-place math transform over f64 slices. |
| `ceil_f64` | In-place math transform over f64 slices. |
| `clamp_f64` | In-place math transform over f64 slices. |
| `clamp_max_f64` | In-place math transform over f64 slices. |
| `clamp_min_f64` | In-place math transform over f64 slices. |
| `cos_f64` | In-place math transform over f64 slices. |
| `cosh_f64` | In-place math transform over f64 slices. |
| `deg_f64` | In-place math transform over f64 slices. |
| `exp_f64` | In-place math transform over f64 slices. |
| `floor_f64` | In-place math transform over f64 slices. |
| `log10_f64` | In-place math transform over f64 slices. |
| `log2_f64` | In-place math transform over f64 slices. |
| `log_f64` | In-place math transform over f64 slices. |
| `neg_f64` | In-place math transform over f64 slices. |
| `rad_f64` | In-place math transform over f64 slices. |
| `round_f64` | In-place math transform over f64 slices. |
| `sgn_f64` | In-place math transform over f64 slices. |
| `sin_f64` | In-place math transform over f64 slices. |
| `sinh_f64` | In-place math transform over f64 slices. |
| `sqrt_f64` | In-place math transform over f64 slices. |
| `tan_f64` | In-place math transform over f64 slices. |
| `tanh_f64` | In-place math transform over f64 slices. |

## Strings / tokenization / classification

| Function | What it does |
|---|---|
| `classify_string` | Classify a UTF-8 string into an `HwxType` with optional numeric payload. |
| `match_exact_phrases` | String matching primitive (in-place filtering of inputs). |
| `match_field_phrases` | String matching primitive (in-place filtering of inputs). |
| `match_field_prefixes` | String matching primitive (in-place filtering of inputs). |
| `match_prefix_strings` | String matching primitive (in-place filtering of inputs). |
| `to_lowercase` | Lowercase a UTF-8 string (fast path where possible). |

## Date/time extraction

| Function | What it does |
|---|---|
| `day_of_month_u64` | Timestamp/date-time extraction primitive. |
| `day_of_week_u64` | Timestamp/date-time extraction primitive. |
| `day_of_year_u64` | Timestamp/date-time extraction primitive. |
| `days_in_month_u64` | Timestamp/date-time extraction primitive. |
| `hour_u64` | Timestamp/date-time extraction primitive. |
| `minute_u64` | Timestamp/date-time extraction primitive. |
| `month_u64` | Timestamp/date-time extraction primitive. |
| `timestamp_u64` | Timestamp/date-time extraction primitive. |
| `year_u64` | Timestamp/date-time extraction primitive. |

## Other dispatch exports

| Function | What it does |
|---|---|
| `check_range_overlaps_f64` | Dispatch-layer primitive. |
| `get_chunk_size_arrays` | Dispatch-layer primitive. |
| `get_chunk_size_datasets` | Dispatch-layer primitive. |
| `get_chunk_size_strings` | Dispatch-layer primitive. |
| `get_quicksort_stack_size` | Dispatch-layer primitive. |
| `get_unroll_factor` | Dispatch-layer primitive. |
| `is_nan_f64` | Dispatch-layer primitive. |
| `usize_to_u32` | Dispatch-layer primitive. |
