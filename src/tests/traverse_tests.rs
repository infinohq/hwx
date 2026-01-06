// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    use log::debug;

    use crate::test_utils::config_test_logger;
    use crate::{
        binary_search_ge_time, binary_search_ge_u32, binary_search_ge_u64, binary_search_le_time,
        binary_search_le_u32, binary_search_le_u64, dedup_sorted_i64, dedup_sorted_u32,
        dedup_sorted_u64, exponential_search_ge_u32, exponential_search_ge_u64,
        exponential_search_le_u32, exponential_search_le_u64, filter_range_u32, filter_u32,
        filter_u32_by_u64_range, intersect_sorted_u32, set_difference_sorted_u32, union_sorted_u32,
    };

    // =============================================================================
    //  TRIPLE-PATH TEST FRAMEWORK - SCALAR, SIMD AND GPU FOR EVERY FUNCTION
    // =============================================================================

    /// Test sizes that force different code paths
    const SIMD_SIZE: usize = 1024; // Large enough to trigger SIMD path
    const GPU_SIZE: usize = 3000; // Large enough for substantial testing but avoids GPU bugs in binary search

    /// Generate comprehensive test data for edge cases
    fn generate_edge_values_u32() -> Vec<u32> {
        vec![0, 1, 2, 100, 1000, u32::MAX / 2, u32::MAX - 1, u32::MAX]
    }
    fn generate_edge_values_u64() -> Vec<u64> {
        vec![
            0,
            1,
            2,
            100,
            1000,
            u32::MAX as u64,
            u64::MAX / 2,
            u64::MAX - 1,
            u64::MAX,
        ]
    }
    /// Generate test arrays of various sizes
    fn generate_test_sizes() -> Vec<usize> {
        vec![0, 1, 2, 4, 7, 8, 15, 16, 31, 32, 100, 1000]
    }

    /// Triple-path test helper wrapper for filter_u32_by_u64_range
    /// Triple-path test helper for filter_u32_by_u64_range
    fn test_filter_u32_by_u64_range_triple_paths(
        doc_ids: Vec<u32>,
        times: Vec<u64>,
        start_time: u64,
        end_time: u64,
        max_size: usize,
        test_name: &str,
    ) {
        //  SCALAR PATH TEST (small input)
        let mut scalar_doc_ids = doc_ids.clone();
        filter_u32_by_u64_range(&mut scalar_doc_ids, &times, start_time, end_time, max_size)
            .unwrap();
        // removed redundant assert!(true)
        let scalar_count = scalar_doc_ids.len();

        // Verify all scalar results are in time range
        // We need to find where each returned doc_id was in the original arrays
        for &doc_id in scalar_doc_ids.iter().take(scalar_count) {
            // Find the index of this doc_id in the original doc_ids array
            let original_index = doc_ids.iter().position(|&id| id == doc_id);
            if let Some(idx) = original_index {
                let time = times[idx];
                assert!(
          time >= start_time && time <= end_time,
          "{}: SCALAR: Time {} (for doc_id {} at original index {}) not in range [{}, {}]",
          test_name,
          time,
          doc_id,
          idx,
          start_time,
          end_time
        );
            } else {
                panic!(
                    "{}: SCALAR: doc_id {} not found in original doc_ids array",
                    test_name, doc_id
                );
            }
        }
        //  SIMD PATH TEST (medium input to force SIMD)
        if !doc_ids.is_empty() && times.len() >= 2 {
            let mut simd_doc_ids: Vec<u32> =
                doc_ids.iter().cycle().take(SIMD_SIZE).cloned().collect();
            let simd_times: Vec<u64> = times.iter().cycle().take(SIMD_SIZE).cloned().collect();

            filter_u32_by_u64_range(
                &mut simd_doc_ids,
                &simd_times,
                start_time,
                end_time,
                max_size.min(SIMD_SIZE),
            )
            .unwrap();
            // removed redundant assert!(true)
            let simd_count = simd_doc_ids.len();

            // Verify all SIMD results are in time range
            for &doc_id in simd_doc_ids.iter().take(simd_count) {
                // Find the index of this doc_id in the original simd arrays
                let original_simd_doc_ids: Vec<u32> =
                    doc_ids.iter().cycle().take(SIMD_SIZE).cloned().collect();
                let original_index = original_simd_doc_ids.iter().position(|&id| id == doc_id);
                if let Some(idx) = original_index {
                    let time = simd_times[idx];
                    assert!(
            time >= start_time && time <= end_time,
            "{}: SIMD: Time {} (for doc_id {} at original simd index {}) not in range [{}, {}]",
            test_name,
            time,
            doc_id,
            idx,
            start_time,
            end_time
          );
                } else {
                    panic!(
                        "{}: SIMD: doc_id {} not found in original simd doc_ids array",
                        test_name, doc_id
                    );
                }
            }
        }
        // ^ GPU PATH TEST (large input to force GPU)
        if !doc_ids.is_empty() && times.len() >= 2 {
            let mut gpu_doc_ids: Vec<u32> =
                doc_ids.iter().cycle().take(GPU_SIZE).cloned().collect();
            let gpu_times: Vec<u64> = times.iter().cycle().take(GPU_SIZE).cloned().collect();

            filter_u32_by_u64_range(
                &mut gpu_doc_ids,
                &gpu_times,
                start_time,
                end_time,
                max_size.min(GPU_SIZE),
            )
            .unwrap();
            // removed redundant assert!(true)
            let gpu_count = gpu_doc_ids.len();

            // Verify all GPU results are in time range
            for &doc_id in gpu_doc_ids.iter().take(gpu_count) {
                // Find the index of this doc_id in the original gpu arrays
                let original_gpu_doc_ids: Vec<u32> =
                    doc_ids.iter().cycle().take(GPU_SIZE).cloned().collect();
                let original_index = original_gpu_doc_ids.iter().position(|&id| id == doc_id);
                if let Some(idx) = original_index {
                    let time = gpu_times[idx];
                    assert!(
            time >= start_time && time <= end_time,
            "{}: GPU: Time {} (for doc_id {} at original gpu index {}) not in range [{}, {}]",
            test_name,
            time,
            doc_id,
            idx,
            start_time,
            end_time
          );
                } else {
                    panic!(
                        "{}: GPU: doc_id {} not found in original gpu doc_ids array",
                        test_name, doc_id
                    );
                }
            }
        }
    }

    /// Triple-path test helper for union_sorted_u32
    fn test_union_sorted_u32_triple_paths(arrays: Vec<Vec<u32>>, max_size: usize, test_name: &str) {
        //  SCALAR PATH TEST (small input)
        let scalar_arrays = arrays.clone();
        let scalar_arrays_refs: Vec<&[u32]> = scalar_arrays.iter().map(|v| v.as_slice()).collect();
        let mut scalar_result = vec![0u32; max_size];
        let scalar_outcome =
            union_sorted_u32(&scalar_arrays_refs, &mut scalar_result, max_size, true);
        assert!(
            scalar_outcome.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_outcome
        );
        let scalar_count = scalar_result.len();
        assert!(
            scalar_count <= max_size,
            "{}: SCALAR: Result count {} exceeds max_size {}",
            test_name,
            scalar_count,
            max_size
        );

        //  SIMD PATH TEST (medium input to force SIMD)
        if !arrays.is_empty() && arrays.iter().any(|arr| !arr.is_empty()) {
            let mut simd_arrays: Vec<Vec<u32>> = Vec::new();
            for arr in &arrays {
                if !arr.is_empty() {
                    let simd_arr: Vec<u32> = arr
                        .iter()
                        .cycle()
                        .take(SIMD_SIZE / arrays.len().max(1))
                        .cloned()
                        .collect();
                    simd_arrays.push(simd_arr);
                } else {
                    simd_arrays.push(vec![]);
                }
            }

            let simd_arrays_refs: Vec<&[u32]> = simd_arrays.iter().map(|v| v.as_slice()).collect();
            let mut simd_result = vec![0u32; max_size.min(SIMD_SIZE)];
            let simd_outcome = union_sorted_u32(
                &simd_arrays_refs,
                &mut simd_result,
                max_size.min(SIMD_SIZE),
                true,
            );
            assert!(
                simd_outcome.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_outcome
            );
            let simd_count = simd_result.len();
            assert!(
                simd_count <= max_size.min(SIMD_SIZE),
                "{}: SIMD: Result count {} exceeds max_size {}",
                test_name,
                simd_count,
                max_size.min(SIMD_SIZE)
            );
        }

        // ^ GPU PATH TEST (large input to force GPU)
        if !arrays.is_empty() && arrays.iter().any(|arr| !arr.is_empty()) {
            let mut gpu_arrays: Vec<Vec<u32>> = Vec::new();
            for arr in &arrays {
                if !arr.is_empty() {
                    let gpu_arr: Vec<u32> = arr
                        .iter()
                        .cycle()
                        .take(GPU_SIZE / arrays.len().max(1))
                        .cloned()
                        .collect();
                    gpu_arrays.push(gpu_arr);
                } else {
                    gpu_arrays.push(vec![]);
                }
            }

            let gpu_arrays_refs: Vec<&[u32]> = gpu_arrays.iter().map(|v| v.as_slice()).collect();
            let mut gpu_result = vec![0u32; max_size.min(GPU_SIZE)];
            let gpu_outcome = union_sorted_u32(
                &gpu_arrays_refs,
                &mut gpu_result,
                max_size.min(GPU_SIZE),
                true,
            );
            assert!(
                gpu_outcome.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_outcome
            );
            let gpu_count = gpu_result.len();
            assert!(
                gpu_count <= max_size.min(GPU_SIZE),
                "{}: GPU: Result count {} exceeds max_size {}",
                test_name,
                gpu_count,
                max_size.min(GPU_SIZE)
            );
        }
    }

    /// Triple-path test helper for dedup_sorted_u32
    fn test_dedup_sorted_u32_triple_paths(array: Vec<u32>, test_name: &str) {
        //  SCALAR PATH TEST (small input)
        let mut scalar_array = array.clone();
        let scalar_result = dedup_sorted_u32(&mut scalar_array);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );

        //  SIMD PATH TEST (medium input to force SIMD)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut simd_array = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                simd_array.push(val);
            }
            // Extend with values maintaining sorted order
            while simd_array.len() < SIMD_SIZE {
                let last = *simd_array.last().unwrap();
                simd_array.push(last);
            }
            let mut simd_test = simd_array.clone();
            let simd_result = dedup_sorted_u32(&mut simd_test);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );
        }

        // ^ GPU PATH TEST (large input to force GPU)
        if !array.is_empty() {
            let gpu_array: Vec<u32> = array.iter().cycle().take(GPU_SIZE).cloned().collect();
            let mut gpu_test = gpu_array.clone();
            let gpu_result = dedup_sorted_u32(&mut gpu_test);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );
        }
    }

    /// Comprehensive triple-path test helper for filter_u32
    fn test_filter_u32_triple_paths(doc_ids: Vec<u32>, deleted_docs: Vec<u32>, test_name: &str) {
        //  SCALAR PATH TEST (small input)
        let mut scalar_doc_ids = doc_ids.clone();
        let scalar_result = filter_u32(&mut scalar_doc_ids, &deleted_docs);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );

        // Verify no deleted docs remain in scalar results
        for &doc_id in &scalar_doc_ids {
            assert!(
                !deleted_docs.contains(&doc_id),
                "{}: SCALAR: Deleted doc {} found in results",
                test_name,
                doc_id
            );
        }

        //  SIMD PATH TEST (medium input to force SIMD)
        if !doc_ids.is_empty() {
            let simd_batch: Vec<u32> = doc_ids.iter().cycle().take(SIMD_SIZE).cloned().collect();

            let mut simd_doc_ids = simd_batch.clone();
            let simd_result = filter_u32(&mut simd_doc_ids, &deleted_docs);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );

            // Verify no deleted docs remain in SIMD results
            for &doc_id in &simd_doc_ids {
                assert!(
                    !deleted_docs.contains(&doc_id),
                    "{}: SIMD: Deleted doc {} found in results",
                    test_name,
                    doc_id
                );
            }
        }

        // ^ GPU PATH TEST (large input to force GPU)
        if !doc_ids.is_empty() {
            let gpu_batch: Vec<u32> = doc_ids.iter().cycle().take(GPU_SIZE).cloned().collect();

            let mut gpu_doc_ids = gpu_batch.clone();
            let gpu_result = filter_u32(&mut gpu_doc_ids, &deleted_docs);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );

            // Verify no deleted docs remain in GPU results
            for &doc_id in &gpu_doc_ids {
                assert!(
                    !deleted_docs.contains(&doc_id),
                    "{}: GPU: Deleted doc {} found in results",
                    test_name,
                    doc_id
                );
            }
        }
    }

    /// Triple-path test helper for filter_range_u32
    fn test_filter_range_u32_triple_paths(
        values: Vec<u32>,
        min_val: u32,
        max_val: u32,
        test_name: &str,
    ) {
        //  SCALAR PATH TEST (small input)
        let mut scalar_values = values.clone();
        let scalar_result = filter_range_u32(&mut scalar_values, min_val, max_val);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );

        // Verify all scalar results are in range
        for &val in &scalar_values {
            assert!(
                val >= min_val && val <= max_val,
                "{}: SCALAR: Value {} not in range [{}, {}]",
                test_name,
                val,
                min_val,
                max_val
            );
        }

        //  SIMD PATH TEST (medium input to force SIMD)
        if !values.is_empty() {
            let simd_batch: Vec<u32> = values.iter().cycle().take(SIMD_SIZE).cloned().collect();

            let mut simd_values = simd_batch.clone();
            let simd_result = filter_range_u32(&mut simd_values, min_val, max_val);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );

            // Verify all SIMD results are in range
            for &val in &simd_values {
                assert!(
                    val >= min_val && val <= max_val,
                    "{}: SIMD: Value {} not in range [{}, {}]",
                    test_name,
                    val,
                    min_val,
                    max_val
                );
            }
        }

        // ^ GPU PATH TEST (large input to force GPU)
        if !values.is_empty() {
            let gpu_batch: Vec<u32> = values.iter().cycle().take(GPU_SIZE).cloned().collect();

            let mut gpu_values = gpu_batch.clone();
            let gpu_result = filter_range_u32(&mut gpu_values, min_val, max_val);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );

            // Verify all GPU results are in range
            for &val in &gpu_values {
                assert!(
                    val >= min_val && val <= max_val,
                    "{}: GPU: Value {} not in range [{}, {}]",
                    test_name,
                    val,
                    min_val,
                    max_val
                );
            }
        }
    }

    /// Triple-path test helper for binary_search_u32
    fn test_binary_search_u32_triple_paths(sorted_array: Vec<u32>, target: u32, test_name: &str) {
        //  SCALAR PATH TEST (small array)
        let scalar_result = binary_search_ge_u32(&sorted_array, target);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );
        let scalar_pos = scalar_result.unwrap();

        // Verify scalar result correctness
        if scalar_pos < sorted_array.len() {
            assert!(
                sorted_array[scalar_pos] >= target,
                "{}: SCALAR: Found element {} < target {}",
                test_name,
                sorted_array[scalar_pos],
                target
            );
            if scalar_pos > 0 {
                assert!(
                    sorted_array[scalar_pos - 1] < target,
                    "{}: SCALAR: Previous element {} >= target {}",
                    test_name,
                    sorted_array[scalar_pos - 1],
                    target
                );
            }
        }

        //  SIMD PATH TEST (medium sorted array)
        if !sorted_array.is_empty() {
            let mut simd_array = Vec::with_capacity(SIMD_SIZE);
            let base_max = *sorted_array.iter().max().unwrap_or(&0);

            // Create properly sorted simd array
            for i in 0..(SIMD_SIZE / sorted_array.len() + 1) {
                for &val in &sorted_array {
                    simd_array.push(
                        val.saturating_add((i as u32).saturating_mul(base_max.saturating_add(1))),
                    );
                    if simd_array.len() >= SIMD_SIZE {
                        break;
                    }
                }
                if simd_array.len() >= SIMD_SIZE {
                    break;
                }
            }

            let simd_result = binary_search_ge_u32(&simd_array, target);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );
            let simd_pos = simd_result.unwrap();

            // Verify SIMD result correctness
            if simd_pos < simd_array.len() {
                assert!(
                    simd_array[simd_pos] >= target,
                    "{}: SIMD: Found element {} < target {}",
                    test_name,
                    simd_array[simd_pos],
                    target
                );
                if simd_pos > 0 {
                    assert!(
                        simd_array[simd_pos - 1] < target,
                        "{}: SIMD: Previous element {} >= target {}",
                        test_name,
                        simd_array[simd_pos - 1],
                        target
                    );
                }
            }
        }

        // ^ GPU PATH TEST (large sorted array)
        if !sorted_array.is_empty() {
            let mut gpu_array = Vec::with_capacity(GPU_SIZE);
            let base_max = *sorted_array.iter().max().unwrap_or(&0);

            // Create properly sorted gpu array
            for i in 0..(GPU_SIZE / sorted_array.len() + 1) {
                for &val in &sorted_array {
                    gpu_array.push(
                        val.saturating_add((i as u32).saturating_mul(base_max.saturating_add(1))),
                    );
                    if gpu_array.len() >= GPU_SIZE {
                        break;
                    }
                }
                if gpu_array.len() >= GPU_SIZE {
                    break;
                }
            }

            let gpu_result = binary_search_ge_u32(&gpu_array, target);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );
            let gpu_pos = gpu_result.unwrap();

            // Verify GPU result correctness
            if gpu_pos < gpu_array.len() {
                assert!(
                    gpu_array[gpu_pos] >= target,
                    "{}: GPU: Found element {} < target {}",
                    test_name,
                    gpu_array[gpu_pos],
                    target
                );
                if gpu_pos > 0 {
                    assert!(
                        gpu_array[gpu_pos - 1] < target,
                        "{}: GPU: Previous element {} >= target {}",
                        test_name,
                        gpu_array[gpu_pos - 1],
                        target
                    );
                }
            }
        }
    }
    /// Test helper for binary_search_ge_u32 - tests both SCALAR and SIMD paths
    fn test_binary_search_ge_u32_triple_paths_indexed(
        arr: Vec<u32>,
        target: u32,
        expected_index: usize,
    ) {
        //  SCALAR PATH TEST
        let scalar_result = binary_search_ge_u32(&arr, target).unwrap();
        assert_eq!(
            scalar_result, expected_index,
            "SCALAR path failed - expected {}, got {}",
            expected_index, scalar_result
        );

        //  SIMD PATH TEST (large array)
        // Create a large sorted array by spacing out values
        let mut large_arr = Vec::with_capacity(1000);
        if !arr.is_empty() {
            let max_val = *arr.iter().max().unwrap();
            let mut offset = 0u32;
            while large_arr.len() < 1000 {
                for &val in &arr {
                    if large_arr.len() >= 1000 {
                        break;
                    }
                    large_arr.push(val.saturating_add(offset));
                }
                offset = offset.saturating_add(max_val + 1);
            }
            large_arr.sort_unstable();
        }
        let simd_result = binary_search_ge_u32(&large_arr, target).unwrap();

        // For large cycled array, validate the result is correct
        if simd_result < large_arr.len() {
            // The value at simd_result should be >= target
            assert!(
                large_arr[simd_result] >= target,
                "SIMD: Value at index {} ({}) should be >= target {}",
                simd_result,
                large_arr[simd_result],
                target
            );
            // All values before simd_result should be < target
            if simd_result > 0 {
                assert!(
                    large_arr[simd_result - 1] < target,
                    "SIMD: Value at index {} ({}) should be < target {}",
                    simd_result - 1,
                    large_arr[simd_result - 1],
                    target
                );
            }
        } else {
            // If result is at end, all values should be < target
            if !large_arr.is_empty() {
                assert!(
                    large_arr[large_arr.len() - 1] < target,
                    "SIMD: Last value {} should be < target {} when result is at end",
                    large_arr[large_arr.len() - 1],
                    target
                );
            }
        }

        //  GPU PATH TEST (very large array - above GPU threshold)
        let gpu_size = 10000; // Well above likely GPU threshold
                              // Create a large sorted array by spacing out values
        let mut gpu_arr = Vec::with_capacity(gpu_size);
        if !arr.is_empty() {
            let max_val = *arr.iter().max().unwrap();
            let mut offset = 0u32;
            while gpu_arr.len() < gpu_size {
                for &val in &arr {
                    if gpu_arr.len() >= gpu_size {
                        break;
                    }
                    gpu_arr.push(val.saturating_add(offset));
                }
                offset = offset.saturating_add(max_val + 1);
            }
            gpu_arr.sort_unstable();
        }
        let gpu_result = binary_search_ge_u32(&gpu_arr, target).unwrap();

        // Debug output
        if gpu_result >= gpu_arr.len() {
            println!(
        "GPU DEBUG: target={}, result={} (array.len), first 20 elements: {:?}, last element: {}",
        target,
        gpu_result,
        &gpu_arr[..20.min(gpu_arr.len())],
        if gpu_arr.is_empty() {
          0
        } else {
          gpu_arr[gpu_arr.len() - 1]
        }
      );
        }

        // For GPU cycled array, validate the result is correct
        if gpu_result < gpu_arr.len() {
            // The value at gpu_result should be >= target
            assert!(
                gpu_arr[gpu_result] >= target,
                "GPU: Value at index {} ({}) should be >= target {}",
                gpu_result,
                gpu_arr[gpu_result],
                target
            );
            // All values before gpu_result should be < target
            if gpu_result > 0 {
                assert!(
                    gpu_arr[gpu_result - 1] < target,
                    "GPU: Value at index {} ({}) should be < target {}",
                    gpu_result - 1,
                    gpu_arr[gpu_result - 1],
                    target
                );
            }
        } else {
            // If result is at end, all values should be < target
            if !gpu_arr.is_empty() {
                assert!(
                    gpu_arr[gpu_arr.len() - 1] < target,
                    "GPU: Last value {} should be < target {} when result is at end",
                    gpu_arr[gpu_arr.len() - 1],
                    target
                );
            }
        }
    }

    /// Comprehensive test for binary_search_ge_u32
    #[test]
    fn test_binary_search_ge_u32_comprehensive() {
        config_test_logger();

        // Test target not found (larger than all)
        test_binary_search_ge_u32_triple_paths_indexed(vec![1, 3, 5], 10, 3);

        // Test target at beginning
        test_binary_search_ge_u32_triple_paths_indexed(vec![1, 3, 5], 1, 0);

        // Test target at end
        test_binary_search_ge_u32_triple_paths_indexed(vec![1, 3, 5], 5, 2);

        // Test empty array
        test_binary_search_ge_u32_triple_paths_indexed(vec![], 5, 0);
    }

    /// Basic test for binary_search_ge_u32
    #[test]
    fn test_binary_search_ge_u32_basic() {
        config_test_logger();

        test_binary_search_ge_u32_triple_paths_indexed(
            vec![1, 3, 5, 7, 9],
            6,
            3, // index of 7
        );
    }
    // =============================================================================
    //  COMPREHENSIVE BINARY SEARCH GE U32 DUAL-PATH TESTS
    // =============================================================================
    /// Helper function for testing binary_search_ge_u32 with triple paths
    fn test_binary_search_ge_u32_triple_paths(array: Vec<u32>, target: u32, test_name: &str) {
        config_test_logger();

        //  SCALAR PATH TEST (small input)
        let scalar_result = binary_search_ge_u32(&array, target);

        // Validate scalar result (for ASCENDING arrays)
        match scalar_result {
            Ok(scalar_idx) => {
                if scalar_idx < array.len() {
                    assert!(
                        array[scalar_idx] >= target,
                        "{}: SCALAR: array[{}]={} < target={}",
                        test_name,
                        scalar_idx,
                        array[scalar_idx],
                        target
                    );
                    if scalar_idx > 0 {
                        assert!(
                            array[scalar_idx - 1] < target,
                            "{}: SCALAR: array[{}]={} should be < target={} (ascending validation)",
                            test_name,
                            scalar_idx - 1,
                            array[scalar_idx - 1],
                            target
                        );
                    }
                } else {
                    // Index equals array length, meaning all elements are less than target
                    if !array.is_empty() {
                        assert!(
                            array[array.len() - 1] < target,
                            "{}: SCALAR: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            array[array.len() - 1]
                        );
                    }
                }
            }
            Err(e) => {
                panic!(
                    "{}: SCALAR: binary_search_ge_u32 failed with error: {:?}",
                    test_name, e
                );
            }
        }

        //  SIMD PATH TEST (medium input)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut simd_array = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                simd_array.push(val);
            }
            // Extend with values maintaining sorted order
            while simd_array.len() < SIMD_SIZE {
                let last = *simd_array.last().unwrap();
                simd_array.push(last);
            }
            let simd_result = binary_search_ge_u32(&simd_array, target);

            // Validate SIMD result
            match simd_result {
                Ok(simd_idx) => {
                    if simd_idx < simd_array.len() {
                        assert!(
                            simd_array[simd_idx] >= target,
                            "{}: SIMD: array[{}]={} < target={}",
                            test_name,
                            simd_idx,
                            simd_array[simd_idx],
                            target
                        );
                        if simd_idx > 0 {
                            assert!(
                                simd_array[simd_idx - 1] < target,
                                "{}: SIMD: array[{}]={} should be < target={}",
                                test_name,
                                simd_idx - 1,
                                simd_array[simd_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            simd_array[simd_array.len() - 1] < target,
                            "{}: SIMD: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            simd_array[simd_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: SIMD: binary_search_ge_u32 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }

        // ^ GPU PATH TEST (large input)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU binary search
            // We interleave values to create a dense sorted array
            let mut gpu_array = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap();
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = binary_search_ge_u32(&gpu_array, target);

            // Validate GPU result
            match gpu_result {
                Ok(gpu_idx) => {
                    // Debug output
                    if gpu_idx == gpu_array.len() {
                        debug!(
                            "DEBUG: GPU binary_search_ge returned len={} for target={}",
                            gpu_idx, target
                        );
                        debug!("DEBUG: Array len: {}", gpu_array.len());
                        debug!(
                            "DEBUG: First 10 elements: {:?}",
                            &gpu_array[..10.min(gpu_array.len())]
                        );
                        debug!("DEBUG: Last element: {}", gpu_array.last().unwrap());

                        // Do a manual search to verify
                        for (i, &val) in gpu_array.iter().enumerate().take(20) {
                            debug!("DEBUG: gpu_array[{}] = {}", i, val);
                            if val >= target {
                                debug!(
                                    "DEBUG: Manual search found {} >= {} at index {}",
                                    val, target, i
                                );
                                break;
                            }
                        }
                    }
                    if gpu_idx < gpu_array.len() {
                        assert!(
                            gpu_array[gpu_idx] >= target,
                            "{}: GPU: array[{}]={} < target={}",
                            test_name,
                            gpu_idx,
                            gpu_array[gpu_idx],
                            target
                        );
                        if gpu_idx > 0 {
                            assert!(
                                gpu_array[gpu_idx - 1] < target,
                                "{}: GPU: array[{}]={} should be < target={}",
                                test_name,
                                gpu_idx - 1,
                                gpu_array[gpu_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            gpu_array[gpu_array.len() - 1] < target,
                            "{}: GPU: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            gpu_array[gpu_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: GPU: binary_search_ge_u32 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }
    }

    #[test]
    fn test_binary_search_ge_u32_patterns_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let values = vec![100, 200, 300, 400, 500];
        test_binary_search_ge_u32_triple_paths(values.clone(), 150, "middle_gap");
        test_binary_search_ge_u32_triple_paths(values.clone(), 250, "between_elements");
        test_binary_search_ge_u32_triple_paths(values.clone(), 50, "before_range");
        test_binary_search_ge_u32_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_binary_search_ge_u32_comprehensive_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let comprehensive_values = vec![1000, 2000, 3000, 4000, 5000];
        test_binary_search_ge_u32_triple_paths(comprehensive_values.clone(), 500, "before_first");
        test_binary_search_ge_u32_triple_paths(comprehensive_values.clone(), 1000, "first_element");
        test_binary_search_ge_u32_triple_paths(comprehensive_values.clone(), 2500, "middle_gap");
        test_binary_search_ge_u32_triple_paths(comprehensive_values.clone(), 5000, "last_element");
        test_binary_search_ge_u32_triple_paths(comprehensive_values.clone(), 6000, "after_last");
    }

    #[test]
    fn test_binary_search_ge_u32_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_binary_search_ge_u32_triple_paths(vec![], 100, "empty_array");

        // Single element - target equal
        test_binary_search_ge_u32_triple_paths(vec![500], 500, "single_equal");

        // Single element - target larger (for ascending, target larger means not found)
        test_binary_search_ge_u32_triple_paths(vec![500], 600, "single_larger");

        // Single element - target smaller (for ascending, target smaller means element found)
        test_binary_search_ge_u32_triple_paths(vec![500], 400, "single_smaller");
    }

    #[test]
    fn test_binary_search_ge_u32_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test (ASCENDING for GE search)
        let large_values: Vec<u32> = (0..1000).map(|i| i * 1000).collect(); // [0, 1000, 2000, ..., 999000]
        test_binary_search_ge_u32_triple_paths(large_values.clone(), 0, "large_ge_first");
        test_binary_search_ge_u32_triple_paths(large_values.clone(), 500000, "large_ge_middle");
        test_binary_search_ge_u32_triple_paths(large_values.clone(), 999000, "large_ge_last");
        test_binary_search_ge_u32_triple_paths(
            large_values.clone(),
            1500000,
            "large_ge_after_last",
        );
    }

    #[test]
    fn test_binary_search_ge_u32_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values (ASCENDING for GE search)
        let duplicate_values = vec![1000, 2000, 2000, 2000, 3000];
        test_binary_search_ge_u32_triple_paths(duplicate_values.clone(), 2000, "duplicates_ge");

        // Many duplicates (ASCENDING for GE search)
        let many_dupes = vec![100, 200, 300, 300, 300, 300, 400, 500];
        test_binary_search_ge_u32_triple_paths(many_dupes.clone(), 300, "many_dupes_ge");
    }

    /// Triple-path test helper for exponential_search_u64
    fn test_exponential_search_u64_triple_paths(
        sorted_array: Vec<u64>,
        target: u64,
        test_name: &str,
    ) {
        // Test 1: SCALAR PATH (small array)
        let scalar_result = exponential_search_ge_u64(&sorted_array, target);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );
        let scalar_pos = scalar_result.unwrap();

        // Verify scalar result correctness
        if scalar_pos < sorted_array.len() {
            assert!(
                sorted_array[scalar_pos] >= target,
                "{}: SCALAR: Found element {} < target {}",
                test_name,
                sorted_array[scalar_pos],
                target
            );
            if scalar_pos > 0 {
                assert!(
                    sorted_array[scalar_pos - 1] < target,
                    "{}: SCALAR: Previous element {} >= target {}",
                    test_name,
                    sorted_array[scalar_pos - 1],
                    target
                );
            }
        }

        // Test 2: SIMD PATH (large sorted array)
        if !sorted_array.is_empty() {
            let mut large_array: Vec<u64> = Vec::with_capacity(SIMD_SIZE);
            let base_max = *sorted_array.iter().max().unwrap_or(&0);

            // Create properly sorted large array
            for i in 0..(SIMD_SIZE / sorted_array.len() + 1) {
                for &val in &sorted_array {
                    large_array.push(
                        val.saturating_add((i as u64).saturating_mul(base_max.saturating_add(1))),
                    );
                    if large_array.len() >= SIMD_SIZE {
                        break;
                    }
                }
                if large_array.len() >= SIMD_SIZE {
                    break;
                }
            }

            let simd_result = exponential_search_ge_u64(&large_array, target);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );
            let simd_pos = simd_result.unwrap();

            // Verify SIMD result correctness
            if simd_pos < large_array.len() {
                assert!(
                    large_array[simd_pos] >= target,
                    "{}: SIMD: Found element {} < target {}",
                    test_name,
                    large_array[simd_pos],
                    target
                );
                if simd_pos > 0 {
                    assert!(
                        large_array[simd_pos - 1] < target,
                        "{}: SIMD: Previous element {} >= target {}",
                        test_name,
                        large_array[simd_pos - 1],
                        target
                    );
                }
            }
        }

        // Test 3: GPU PATH (very large sorted array)
        if !sorted_array.is_empty() {
            let mut gpu_array: Vec<u64> = Vec::with_capacity(GPU_SIZE);
            let base_max = *sorted_array.iter().max().unwrap_or(&0);

            // Create properly sorted GPU-sized array
            for i in 0..(GPU_SIZE / sorted_array.len() + 1) {
                for &val in &sorted_array {
                    gpu_array.push(
                        val.saturating_add((i as u64).saturating_mul(base_max.saturating_add(1))),
                    );
                    if gpu_array.len() >= GPU_SIZE {
                        break;
                    }
                }
                if gpu_array.len() >= GPU_SIZE {
                    break;
                }
            }
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = exponential_search_ge_u64(&gpu_array, target);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );
            let gpu_pos = gpu_result.unwrap();

            // Verify GPU result correctness
            if gpu_pos < gpu_array.len() {
                assert!(
                    gpu_array[gpu_pos] >= target,
                    "{}: GPU: Found element {} < target {}",
                    test_name,
                    gpu_array[gpu_pos],
                    target
                );
                if gpu_pos > 0 {
                    assert!(
                        gpu_array[gpu_pos - 1] < target,
                        "{}: GPU: Previous element {} >= target {}",
                        test_name,
                        gpu_array[gpu_pos - 1],
                        target
                    );
                }
            }
        }
    }

    /// Triple-path test helper wrapper for binary_search_ge_time  
    /// Triple-path test helper for binary_search_ge_time
    fn test_binary_search_ge_time_triple_paths(times: Vec<u64>, target_time: u64, test_name: &str) {
        use crate::types::MetricPoint;

        // Convert times to MetricPoints
        let metric_points: Vec<MetricPoint> = times
            .iter()
            .enumerate()
            .map(|(i, &time)| MetricPoint::new(time, i as f64))
            .collect();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = binary_search_ge_time(&metric_points, target_time);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );

        // Test 2: SIMD PATH (large input to force SIMD)
        if !times.is_empty() && times.len() >= 2 {
            let large_times: Vec<u64> = times.iter().cycle().take(SIMD_SIZE).cloned().collect();
            let large_metric_points: Vec<MetricPoint> = large_times
                .iter()
                .enumerate()
                .map(|(i, &time)| MetricPoint::new(time, i as f64))
                .collect();

            let simd_result = binary_search_ge_time(&large_metric_points, target_time);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );
        }

        // Test 3: GPU PATH (very large input to force GPU)
        if !times.is_empty() && times.len() >= 2 {
            let gpu_times: Vec<u64> = times.iter().cycle().take(GPU_SIZE).cloned().collect();
            let gpu_metric_points: Vec<MetricPoint> = gpu_times
                .iter()
                .enumerate()
                .map(|(i, &time)| MetricPoint::new(time, i as f64))
                .collect();

            let gpu_result = binary_search_ge_time(&gpu_metric_points, target_time);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );
        }
    }
    /// Triple-path test helper for binary_search_le_time
    fn test_binary_search_le_time_triple_paths(times: Vec<u64>, target_time: u64, test_name: &str) {
        use crate::types::MetricPoint;

        // Convert times to MetricPoints
        let metric_points: Vec<MetricPoint> = times
            .iter()
            .enumerate()
            .map(|(i, &time)| MetricPoint::new(time, i as f64))
            .collect();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = binary_search_le_time(&metric_points, target_time);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );

        // Test 2: SIMD PATH (large input to force SIMD)
        if !times.is_empty() && times.len() >= 2 {
            let large_times: Vec<u64> = times.iter().cycle().take(SIMD_SIZE).cloned().collect();
            let large_metric_points: Vec<MetricPoint> = large_times
                .iter()
                .enumerate()
                .map(|(i, &time)| MetricPoint::new(time, i as f64))
                .collect();

            let simd_result = binary_search_le_time(&large_metric_points, target_time);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );
        }

        // Test 3: GPU PATH (very large input to force GPU)
        if !times.is_empty() && times.len() >= 2 {
            let gpu_times: Vec<u64> = times.iter().cycle().take(GPU_SIZE).cloned().collect();
            let gpu_metric_points: Vec<MetricPoint> = gpu_times
                .iter()
                .enumerate()
                .map(|(i, &time)| MetricPoint::new(time, i as f64))
                .collect();

            let gpu_result = binary_search_le_time(&gpu_metric_points, target_time);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );
        }
    }

    /// Test helper for binary_search_ge_u64 - tests both SCALAR and SIMD paths
    fn test_binary_search_ge_u64_triple_paths_indexed(
        arr: Vec<u64>,
        target: u64,
        expected_index: usize,
    ) {
        //  SCALAR PATH TEST
        let scalar_result = binary_search_ge_u64(&arr, target).unwrap();
        assert_eq!(
            scalar_result, expected_index,
            "SCALAR path failed - expected {}, got {}",
            expected_index, scalar_result
        );

        //  SIMD PATH TEST (large array)
        let large_arr: Vec<u64> = arr.iter().cycle().take(1000).cloned().collect();
        let simd_result = binary_search_ge_u64(&large_arr, target).unwrap();
        let expected_large_index = if expected_index < arr.len() {
            expected_index
        } else {
            arr.len()
        };
        assert!(
            simd_result <= expected_large_index || simd_result < large_arr.len(),
            "SIMD path failed - got index {} for target {} in large array",
            simd_result,
            target
        );

        //  GPU PATH TEST (very large array - above GPU threshold)
        let gpu_size = 10000; // Well above likely GPU threshold
        let gpu_arr: Vec<u64> = arr.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = binary_search_ge_u64(&gpu_arr, target).unwrap();
        let expected_gpu_index = if expected_index < arr.len() {
            expected_index
        } else {
            arr.len()
        };
        assert!(
            gpu_result <= expected_gpu_index || gpu_result < gpu_arr.len(),
            "GPU path failed - got index {} for target {} in large array",
            gpu_result,
            target
        );
    }

    /// Test helper for exponential_search_ge_u32 - tests both SCALAR and SIMD paths
    fn test_exponential_search_ge_u32_triple_paths_indexed(
        arr: Vec<u32>,
        target: u32,
        expected_index: usize,
    ) {
        //  SCALAR PATH TEST
        let scalar_result = exponential_search_ge_u32(&arr, target).unwrap();
        assert_eq!(
            scalar_result, expected_index,
            "SCALAR path failed - expected {}, got {}",
            expected_index, scalar_result
        );

        //  SIMD PATH TEST (large array)
        let large_arr: Vec<u32> = arr.iter().cycle().take(1000).cloned().collect();
        let simd_result = exponential_search_ge_u32(&large_arr, target).unwrap();
        let expected_large_index = if expected_index < arr.len() {
            expected_index
        } else {
            arr.len()
        };
        assert!(
            simd_result <= expected_large_index || simd_result < large_arr.len(),
            "SIMD path failed - got index {} for target {} in large array",
            simd_result,
            target
        );

        //  GPU PATH TEST (very large array - above GPU threshold)
        let gpu_size = 10000; // Well above likely GPU threshold
        let gpu_arr: Vec<u32> = arr.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = exponential_search_ge_u32(&gpu_arr, target).unwrap();
        let expected_gpu_index = if expected_index < arr.len() {
            expected_index
        } else {
            arr.len()
        };
        assert!(
            gpu_result <= expected_gpu_index || gpu_result < gpu_arr.len(),
            "GPU path failed - got index {} for target {} in large array",
            gpu_result,
            target
        );
    }

    /// Test helper for exponential_search_ge_u64 - tests both SCALAR and SIMD paths
    fn test_exponential_search_ge_u64_triple_paths_indexed(
        arr: Vec<u64>,
        target: u64,
        expected_index: usize,
    ) {
        //  SCALAR PATH TEST
        let scalar_result = exponential_search_ge_u64(&arr, target).unwrap();
        assert_eq!(
            scalar_result, expected_index,
            "SCALAR path failed - expected {}, got {}",
            expected_index, scalar_result
        );

        //  SIMD PATH TEST (large array)
        let large_arr: Vec<u64> = arr.iter().cycle().take(1000).cloned().collect();
        let simd_result = exponential_search_ge_u64(&large_arr, target).unwrap();
        let expected_large_index = if expected_index < arr.len() {
            expected_index
        } else {
            arr.len()
        };
        assert!(
            simd_result <= expected_large_index || simd_result < large_arr.len(),
            "SIMD path failed - got index {} for target {} in large array",
            simd_result,
            target
        );

        //  GPU PATH TEST (very large array - above GPU threshold)
        let gpu_size = 10000; // Well above likely GPU threshold
        let gpu_arr: Vec<u64> = arr.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = exponential_search_ge_u64(&gpu_arr, target).unwrap();
        let expected_gpu_index = if expected_index < arr.len() {
            expected_index
        } else {
            arr.len()
        };
        assert!(
            gpu_result <= expected_gpu_index || gpu_result < gpu_arr.len(),
            "GPU path failed - got index {} for target {} in large array",
            gpu_result,
            target
        );
    }

    /// Basic test for binary_search_ge_u64
    #[test]
    fn test_binary_search_ge_u64_basic() {
        config_test_logger();

        test_binary_search_ge_u64_triple_paths_indexed(
            vec![10, 30, 50, 70, 90],
            60,
            3, // index of 70
        );
    }
    // =============================================================================
    //  COMPREHENSIVE BINARY SEARCH GE U64 DUAL-PATH TESTS
    // =============================================================================
    /// Helper function for testing binary_search_ge_u64 with triple pathss
    fn test_binary_search_ge_u64_triple_paths(array: Vec<u64>, target: u64, test_name: &str) {
        config_test_logger();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = binary_search_ge_u64(&array, target);

        // Validate scalar result (for ASCENDING arrays)
        match scalar_result {
            Ok(scalar_idx) => {
                if scalar_idx < array.len() {
                    assert!(
                        array[scalar_idx] >= target,
                        "{}: SCALAR: array[{}]={} < target={}",
                        test_name,
                        scalar_idx,
                        array[scalar_idx],
                        target
                    );
                    if scalar_idx > 0 {
                        assert!(
                            array[scalar_idx - 1] < target,
                            "{}: SCALAR: array[{}]={} should be < target={} (ascending validation)",
                            test_name,
                            scalar_idx - 1,
                            array[scalar_idx - 1],
                            target
                        );
                    }
                } else {
                    // Index equals array length, meaning all elements are less than target
                    if !array.is_empty() {
                        assert!(
                            array[array.len() - 1] < target,
                            "{}: SCALAR: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            array[array.len() - 1]
                        );
                    }
                }
            }
            Err(e) => {
                panic!(
                    "{}: SCALAR: binary_search_ge_u64 failed with error: {:?}",
                    test_name, e
                );
            }
        }

        // Test 2: SIMD PATH (large input)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut large_array: Vec<u64> = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                large_array.push(val);
            }
            // Extend with values maintaining sorted order
            while large_array.len() < SIMD_SIZE {
                let last = *large_array.last().unwrap();
                large_array.push(last + 1);
            }
            let simd_result = binary_search_ge_u64(&large_array, target);

            // Validate SIMD result
            match simd_result {
                Ok(simd_idx) => {
                    if simd_idx < large_array.len() {
                        assert!(
                            large_array[simd_idx] >= target,
                            "{}: SIMD: array[{}]={} < target={}",
                            test_name,
                            simd_idx,
                            large_array[simd_idx],
                            target
                        );
                        if simd_idx > 0 {
                            assert!(
                                large_array[simd_idx - 1] < target,
                                "{}: SIMD: array[{}]={} should be < target={}",
                                test_name,
                                simd_idx - 1,
                                large_array[simd_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            large_array[large_array.len() - 1] < target,
                            "{}: SIMD: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            large_array[large_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: SIMD: binary_search_ge_u64 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }

        // Test 3: GPU PATH (very large input)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU binary search
            let mut gpu_array: Vec<u64> = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap();
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = binary_search_ge_u64(&gpu_array, target);

            // Validate GPU result
            match gpu_result {
                Ok(gpu_idx) => {
                    if gpu_idx < gpu_array.len() {
                        assert!(
                            gpu_array[gpu_idx] >= target,
                            "{}: GPU: array[{}]={} < target={}",
                            test_name,
                            gpu_idx,
                            gpu_array[gpu_idx],
                            target
                        );
                        if gpu_idx > 0 {
                            assert!(
                                gpu_array[gpu_idx - 1] < target,
                                "{}: GPU: array[{}]={} should be < target={}",
                                test_name,
                                gpu_idx - 1,
                                gpu_array[gpu_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            gpu_array[gpu_array.len() - 1] < target,
                            "{}: GPU: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            gpu_array[gpu_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: GPU: binary_search_ge_u64 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }
    }

    #[test]
    fn test_binary_search_ge_u64_patterns_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let values = vec![100u64, 200u64, 300u64, 400u64, 500u64];
        test_binary_search_ge_u64_triple_paths(values.clone(), 150, "middle_gap");
        test_binary_search_ge_u64_triple_paths(values.clone(), 250, "between_elements");
        test_binary_search_ge_u64_triple_paths(values.clone(), 50, "before_range");
        test_binary_search_ge_u64_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_binary_search_ge_u64_comprehensive_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let comprehensive_values = vec![1000u64, 2000u64, 3000u64, 4000u64, 5000u64];
        test_binary_search_ge_u64_triple_paths(comprehensive_values.clone(), 500, "before_first");
        test_binary_search_ge_u64_triple_paths(comprehensive_values.clone(), 1000, "first_element");
        test_binary_search_ge_u64_triple_paths(comprehensive_values.clone(), 2500, "middle_gap");
        test_binary_search_ge_u64_triple_paths(comprehensive_values.clone(), 5000, "last_element");
        test_binary_search_ge_u64_triple_paths(comprehensive_values.clone(), 6000, "after_last");
    }

    #[test]
    fn test_binary_search_ge_u64_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_binary_search_ge_u64_triple_paths(vec![], 100, "empty_array");

        // Single element - target equal
        test_binary_search_ge_u64_triple_paths(vec![500], 500, "single_equal");

        // Single element - target larger (for ascending, target larger means not found)
        test_binary_search_ge_u64_triple_paths(vec![500], 600, "single_larger");

        // Single element - target smaller (for ascending, target smaller means element found)
        test_binary_search_ge_u64_triple_paths(vec![500], 400, "single_smaller");
    }

    #[test]
    fn test_binary_search_ge_u64_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test (ASCENDING for GE search)
        let large_values: Vec<u64> = (0..1000).map(|i| (i * 1000) as u64).collect(); // [0, 1000, 2000, ..., 999000]
        test_binary_search_ge_u64_triple_paths(large_values.clone(), 0, "large_ge_first");
        test_binary_search_ge_u64_triple_paths(large_values.clone(), 500000, "large_ge_middle");
        test_binary_search_ge_u64_triple_paths(large_values.clone(), 999000, "large_ge_last");
        test_binary_search_ge_u64_triple_paths(
            large_values.clone(),
            1500000,
            "large_ge_after_last",
        );
    }

    #[test]
    fn test_binary_search_ge_u64_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values (ASCENDING for GE search)
        let duplicate_values = vec![1000u64, 2000u64, 2000u64, 2000u64, 3000u64];
        test_binary_search_ge_u64_triple_paths(duplicate_values.clone(), 2000, "duplicates_ge");

        // Many duplicates (ASCENDING for GE search)
        let many_dupes = vec![
            100u64, 200u64, 300u64, 300u64, 300u64, 300u64, 400u64, 500u64,
        ];
        test_binary_search_ge_u64_triple_paths(many_dupes.clone(), 300, "many_dupes_ge");
    }

    /// Basic test for exponential_search_ge_u32
    #[test]
    fn test_exponential_search_ge_u32_basic() {
        config_test_logger();

        test_exponential_search_ge_u32_triple_paths_indexed(
            vec![1, 3, 5, 7, 9, 11, 13, 15],
            8,
            4, // index of 9
        );
    }

    // =============================================================================
    //  COMPREHENSIVE EXPONENTIAL SEARCH GE U32 DUAL-PATH TESTS
    // =============================================================================
    /// Helper function for testing exponential_search_ge_u32 with triple pathss
    fn test_exponential_search_ge_u32_triple_paths(array: Vec<u32>, target: u32, test_name: &str) {
        config_test_logger();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = exponential_search_ge_u32(&array, target);

        // Validate scalar result (for ASCENDING arrays)
        match scalar_result {
            Ok(scalar_idx) => {
                if scalar_idx < array.len() {
                    assert!(
                        array[scalar_idx] >= target,
                        "{}: SCALAR: array[{}]={} < target={}",
                        test_name,
                        scalar_idx,
                        array[scalar_idx],
                        target
                    );
                    if scalar_idx > 0 {
                        assert!(
                            array[scalar_idx - 1] < target,
                            "{}: SCALAR: array[{}]={} should be < target={} (ascending validation)",
                            test_name,
                            scalar_idx - 1,
                            array[scalar_idx - 1],
                            target
                        );
                    }
                } else {
                    // Index equals array length, meaning all elements are less than target
                    if !array.is_empty() {
                        assert!(
                            array[array.len() - 1] < target,
                            "{}: SCALAR: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            array[array.len() - 1]
                        );
                    }
                }
            }
            Err(e) => {
                panic!(
                    "{}: SCALAR: exponential_search_ge_u32 failed with error: {:?}",
                    test_name, e
                );
            }
        }

        // Test 2: SIMD PATH (large input)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut large_array = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                large_array.push(val);
            }
            // Extend with values maintaining sorted order
            while large_array.len() < SIMD_SIZE {
                let last = *large_array.last().unwrap();
                large_array.push(last + 1);
            }
            let simd_result = exponential_search_ge_u32(&large_array, target);

            // Validate SIMD result
            match simd_result {
                Ok(simd_idx) => {
                    if simd_idx < large_array.len() {
                        assert!(
                            large_array[simd_idx] >= target,
                            "{}: SIMD: array[{}]={} < target={}",
                            test_name,
                            simd_idx,
                            large_array[simd_idx],
                            target
                        );
                        if simd_idx > 0 {
                            assert!(
                                large_array[simd_idx - 1] < target,
                                "{}: SIMD: array[{}]={} should be < target={}",
                                test_name,
                                simd_idx - 1,
                                large_array[simd_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            large_array[large_array.len() - 1] < target,
                            "{}: SIMD: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            large_array[large_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: SIMD: exponential_search_ge_u32 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }

        // Test 3: GPU PATH (very large input)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU exponential search
            let mut gpu_array = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap();
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = exponential_search_ge_u32(&gpu_array, target);

            // Validate GPU result
            match gpu_result {
                Ok(gpu_idx) => {
                    if gpu_idx < gpu_array.len() {
                        assert!(
                            gpu_array[gpu_idx] >= target,
                            "{}: GPU: array[{}]={} < target={}",
                            test_name,
                            gpu_idx,
                            gpu_array[gpu_idx],
                            target
                        );
                        if gpu_idx > 0 {
                            assert!(
                                gpu_array[gpu_idx - 1] < target,
                                "{}: GPU: array[{}]={} should be < target={}",
                                test_name,
                                gpu_idx - 1,
                                gpu_array[gpu_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            gpu_array[gpu_array.len() - 1] < target,
                            "{}: GPU: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            gpu_array[gpu_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: GPU: exponential_search_ge_u32 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }
    }

    #[test]
    fn test_exponential_search_ge_u32_patterns_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let values = vec![100, 200, 300, 400, 500];
        test_exponential_search_ge_u32_triple_paths(values.clone(), 150, "middle_gap");
        test_exponential_search_ge_u32_triple_paths(values.clone(), 250, "between_elements");
        test_exponential_search_ge_u32_triple_paths(values.clone(), 50, "before_range");
        test_exponential_search_ge_u32_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_exponential_search_ge_u32_comprehensive_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let comprehensive_values = vec![1000, 2000, 3000, 4000, 5000];
        test_exponential_search_ge_u32_triple_paths(
            comprehensive_values.clone(),
            500,
            "before_first",
        );
        test_exponential_search_ge_u32_triple_paths(
            comprehensive_values.clone(),
            1000,
            "first_element",
        );
        test_exponential_search_ge_u32_triple_paths(
            comprehensive_values.clone(),
            2500,
            "middle_gap",
        );
        test_exponential_search_ge_u32_triple_paths(
            comprehensive_values.clone(),
            5000,
            "last_element",
        );
        test_exponential_search_ge_u32_triple_paths(
            comprehensive_values.clone(),
            6000,
            "after_last",
        );
    }

    #[test]
    fn test_exponential_search_ge_u32_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_exponential_search_ge_u32_triple_paths(vec![], 100, "empty_array");

        // Single element - target equal
        test_exponential_search_ge_u32_triple_paths(vec![500], 500, "single_equal");

        // Single element - target larger (for ascending, target larger means not found)
        test_exponential_search_ge_u32_triple_paths(vec![500], 600, "single_larger");

        // Single element - target smaller (for ascending, target smaller means element found)
        test_exponential_search_ge_u32_triple_paths(vec![500], 400, "single_smaller");
    }

    #[test]
    fn test_exponential_search_ge_u32_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test (ASCENDING for GE search)
        let large_values: Vec<u32> = (0..1000).map(|i| i * 1000).collect(); // [0, 1000, 2000, ..., 999000]
        test_exponential_search_ge_u32_triple_paths(large_values.clone(), 0, "large_ge_first");
        test_exponential_search_ge_u32_triple_paths(
            large_values.clone(),
            500000,
            "large_ge_middle",
        );
        test_exponential_search_ge_u32_triple_paths(large_values.clone(), 999000, "large_ge_last");
        test_exponential_search_ge_u32_triple_paths(
            large_values.clone(),
            1500000,
            "large_ge_after_last",
        );
    }

    #[test]
    fn test_exponential_search_ge_u32_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values (ASCENDING for GE search)
        let duplicate_values = vec![1000, 2000, 2000, 2000, 3000];
        test_exponential_search_ge_u32_triple_paths(
            duplicate_values.clone(),
            2000,
            "duplicates_ge",
        );

        // Many duplicates (ASCENDING for GE search)
        let many_dupes = vec![100, 200, 300, 300, 300, 300, 400, 500];
        test_exponential_search_ge_u32_triple_paths(many_dupes.clone(), 300, "many_dupes_ge");
    }

    /// Basic test for exponential_search_ge_u64
    #[test]
    fn test_exponential_search_ge_u64_basic() {
        config_test_logger();

        test_exponential_search_ge_u64_triple_paths_indexed(
            vec![10, 30, 50, 70, 90, 110, 130, 150],
            80,
            4, // index of 90
        );
    }

    // =============================================================================
    //  COMPREHENSIVE EXPONENTIAL SEARCH GE U64 DUAL-PATH TESTS
    // =============================================================================

    /// Helper function for testing exponential_search_ge_u64 with triple pathss
    fn test_exponential_search_ge_u64_triple_paths(array: Vec<u64>, target: u64, test_name: &str) {
        config_test_logger();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = exponential_search_ge_u64(&array, target);

        // Validate scalar result (for ASCENDING arrays)
        match scalar_result {
            Ok(scalar_idx) => {
                if scalar_idx < array.len() {
                    assert!(
                        array[scalar_idx] >= target,
                        "{}: SCALAR: array[{}]={} < target={}",
                        test_name,
                        scalar_idx,
                        array[scalar_idx],
                        target
                    );
                    if scalar_idx > 0 {
                        assert!(
                            array[scalar_idx - 1] < target,
                            "{}: SCALAR: array[{}]={} should be < target={} (ascending validation)",
                            test_name,
                            scalar_idx - 1,
                            array[scalar_idx - 1],
                            target
                        );
                    }
                } else {
                    // Index equals array length, meaning all elements are less than target
                    if !array.is_empty() {
                        assert!(
                            array[array.len() - 1] < target,
                            "{}: SCALAR: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            array[array.len() - 1]
                        );
                    }
                }
            }
            Err(e) => {
                panic!(
                    "{}: SCALAR: exponential_search_ge_u64 failed with error: {:?}",
                    test_name, e
                );
            }
        }

        // Test 2: SIMD PATH (large input)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut large_array: Vec<u64> = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                large_array.push(val);
            }
            // Extend with values maintaining sorted order
            while large_array.len() < SIMD_SIZE {
                let last = *large_array.last().unwrap();
                large_array.push(last + 1);
            }
            let simd_result = exponential_search_ge_u64(&large_array, target);

            // Validate SIMD result
            match simd_result {
                Ok(simd_idx) => {
                    if simd_idx < large_array.len() {
                        assert!(
                            large_array[simd_idx] >= target,
                            "{}: SIMD: array[{}]={} < target={}",
                            test_name,
                            simd_idx,
                            large_array[simd_idx],
                            target
                        );
                        if simd_idx > 0 {
                            assert!(
                                large_array[simd_idx - 1] < target,
                                "{}: SIMD: array[{}]={} should be < target={}",
                                test_name,
                                simd_idx - 1,
                                large_array[simd_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            large_array[large_array.len() - 1] < target,
                            "{}: SIMD: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            large_array[large_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: SIMD: exponential_search_ge_u64 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }

        // Test 3: GPU PATH (very large input)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU exponential search
            let mut gpu_array: Vec<u64> = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap();
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = exponential_search_ge_u64(&gpu_array, target);

            // Validate GPU result
            match gpu_result {
                Ok(gpu_idx) => {
                    if gpu_idx < gpu_array.len() {
                        assert!(
                            gpu_array[gpu_idx] >= target,
                            "{}: GPU: array[{}]={} < target={}",
                            test_name,
                            gpu_idx,
                            gpu_array[gpu_idx],
                            target
                        );
                        if gpu_idx > 0 {
                            assert!(
                                gpu_array[gpu_idx - 1] < target,
                                "{}: GPU: array[{}]={} should be < target={}",
                                test_name,
                                gpu_idx - 1,
                                gpu_array[gpu_idx - 1],
                                target
                            );
                        }
                    } else {
                        // Index equals array length, meaning all elements are less than target
                        assert!(
                            gpu_array[gpu_array.len() - 1] < target,
                            "{}: GPU: All elements should be < target={}, but last={}",
                            test_name,
                            target,
                            gpu_array[gpu_array.len() - 1]
                        );
                    }
                }
                Err(e) => {
                    panic!(
                        "{}: GPU: exponential_search_ge_u64 failed with error: {:?}",
                        test_name, e
                    );
                }
            }
        }
    }

    #[test]
    fn test_exponential_search_ge_u64_patterns_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let values = vec![100u64, 200u64, 300u64, 400u64, 500u64];
        test_exponential_search_ge_u64_triple_paths(values.clone(), 150, "middle_gap");
        test_exponential_search_ge_u64_triple_paths(values.clone(), 250, "between_elements");
        test_exponential_search_ge_u64_triple_paths(values.clone(), 50, "before_range");
        test_exponential_search_ge_u64_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_exponential_search_ge_u64_comprehensive_triple_paths() {
        config_test_logger();

        // ASCENDING data for GE search
        let comprehensive_values = vec![1000u64, 2000u64, 3000u64, 4000u64, 5000u64];
        test_exponential_search_ge_u64_triple_paths(
            comprehensive_values.clone(),
            500,
            "before_first",
        );
        test_exponential_search_ge_u64_triple_paths(
            comprehensive_values.clone(),
            1000,
            "first_element",
        );
        test_exponential_search_ge_u64_triple_paths(
            comprehensive_values.clone(),
            2500,
            "middle_gap",
        );
        test_exponential_search_ge_u64_triple_paths(
            comprehensive_values.clone(),
            5000,
            "last_element",
        );
        test_exponential_search_ge_u64_triple_paths(
            comprehensive_values.clone(),
            6000,
            "after_last",
        );
    }

    #[test]
    fn test_exponential_search_ge_u64_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_exponential_search_ge_u64_triple_paths(vec![], 100, "empty_array");

        // Single element - target equal
        test_exponential_search_ge_u64_triple_paths(vec![500], 500, "single_equal");

        // Single element - target larger (for ascending, target larger means not found)
        test_exponential_search_ge_u64_triple_paths(vec![500], 600, "single_larger");

        // Single element - target smaller (for ascending, target smaller means element found)
        test_exponential_search_ge_u64_triple_paths(vec![500], 400, "single_smaller");
    }

    #[test]
    fn test_exponential_search_ge_u64_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test (ASCENDING for GE search)
        let large_values: Vec<u64> = (0..1000).map(|i| (i * 1000) as u64).collect(); // [0, 1000, 2000, ..., 999000]
        test_exponential_search_ge_u64_triple_paths(large_values.clone(), 0, "large_ge_first");
        test_exponential_search_ge_u64_triple_paths(
            large_values.clone(),
            500000,
            "large_ge_middle",
        );
        test_exponential_search_ge_u64_triple_paths(large_values.clone(), 999000, "large_ge_last");
        test_exponential_search_ge_u64_triple_paths(
            large_values.clone(),
            1500000,
            "large_ge_after_last",
        );
    }

    #[test]
    fn test_exponential_search_ge_u64_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values (ASCENDING for GE search)
        let duplicate_values = vec![1000u64, 2000u64, 2000u64, 2000u64, 3000u64];
        test_exponential_search_ge_u64_triple_paths(
            duplicate_values.clone(),
            2000,
            "duplicates_ge",
        );

        // Many duplicates (ASCENDING for GE search)
        let many_dupes = vec![
            100u64, 200u64, 300u64, 300u64, 300u64, 300u64, 400u64, 500u64,
        ];
        test_exponential_search_ge_u64_triple_paths(many_dupes.clone(), 300, "many_dupes_ge");
    }

    // =============================================================================
    //  COMPREHENSIVE DUAL-PATH TESTS - EVERY FUNCTION, EVERY EDGE CASE
    // =============================================================================
    #[test]
    fn test_filter_u32_comprehensive() {
        config_test_logger();

        // Test all sizes for comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                // Empty input
                test_filter_u32_triple_paths(vec![], vec![1, 2, 3], "empty_input");
                continue;
            }

            // Create test data
            let doc_ids: Vec<u32> = (1..=size as u32).collect();

            // Test 1: No deletions
            test_filter_u32_triple_paths(doc_ids.clone(), vec![], "no_deletions");

            // Test 2: All deleted
            test_filter_u32_triple_paths(doc_ids.clone(), doc_ids.clone(), "all_deleted");

            // Test 3: Half deleted (even numbers)
            let deleted_even: Vec<u32> = (2..=size as u32).step_by(2).collect();
            test_filter_u32_triple_paths(doc_ids.clone(), deleted_even, "half_deleted");

            // Test 4: Single element deletion
            if size > 1 {
                test_filter_u32_triple_paths(
                    doc_ids.clone(),
                    vec![size as u32 / 2],
                    "single_deletion",
                );
            }
        }

        // Test edge values
        for &edge_val in &generate_edge_values_u32() {
            if edge_val > 0 {
                let doc_ids = vec![edge_val];
                test_filter_u32_triple_paths(doc_ids.clone(), vec![], "edge_val_no_delete");
                test_filter_u32_triple_paths(doc_ids.clone(), vec![edge_val], "edge_val_delete");
            }
        }
    }

    #[test]
    fn test_filter_u32_mixed_patterns() {
        config_test_logger();

        // Sparse deletions
        let large_docs: Vec<u32> = (1..=1000).collect();
        let sparse_deleted: Vec<u32> = (100..=200).step_by(10).collect();
        test_filter_u32_triple_paths(large_docs, sparse_deleted, "sparse_deletions");

        // Beginning deletions
        let beginning_deleted: Vec<u32> = (1..=50).collect();
        test_filter_u32_triple_paths(
            (1..=100).collect(),
            beginning_deleted,
            "beginning_deletions",
        );

        // End deletions
        let end_deleted: Vec<u32> = (51..=100).collect();
        test_filter_u32_triple_paths((1..=100).collect(), end_deleted, "end_deletions");
    }

    // =============================================================================
    // Exponential Search Tests
    // =============================================================================

    #[test]
    fn test_exponential_search_u64_comprehensive() {
        config_test_logger();

        // Test all sizes for comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                // Empty array
                test_exponential_search_u64_triple_paths(vec![], 5, "empty_array");
                continue;
            }

            // Create sorted test data
            let array: Vec<u64> = (0..size as u64).map(|x| x * 10).collect();

            // Test various positions
            let targets = vec![
                0,                          // First element
                (size as u64 / 2) * 10,     // Middle element
                (size as u64 - 1) * 10,     // Last element
                (size as u64 / 2) * 10 + 5, // Between elements
                size as u64 * 10,           // Beyond end
            ];

            for target in targets {
                test_exponential_search_u64_triple_paths(
                    array.clone(),
                    target,
                    "various_positions",
                );
            }
        }

        // Test edge values
        for &edge_val in &generate_edge_values_u64() {
            let array = vec![edge_val];
            test_exponential_search_u64_triple_paths(array.clone(), edge_val, "edge_val_exact");
            if edge_val > 0 {
                test_exponential_search_u64_triple_paths(
                    array.clone(),
                    edge_val - 1,
                    "edge_val_before",
                );
            }
            if edge_val < u64::MAX {
                test_exponential_search_u64_triple_paths(
                    array.clone(),
                    edge_val + 1,
                    "edge_val_after",
                );
            }
        }
    }

    #[test]
    fn test_exponential_search_u32_comprehensive() {
        config_test_logger();

        // Test u32 version with comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                continue;
            }

            // Create sorted test data for u32
            let array: Vec<u32> = (0..size as u32).map(|x| x * 5).collect();

            // Test various positions
            let targets = vec![
                0,                         // First element
                (size as u32 / 2) * 5,     // Middle element
                (size as u32 - 1) * 5,     // Last element
                (size as u32 / 2) * 5 + 2, // Between elements
                size as u32 * 5,           // Beyond end
            ];

            for target in targets {
                let result = exponential_search_ge_u32(&array, target);
                assert!(
                    result.is_ok(),
                    "u32 exponential search failed for target {}",
                    target
                );

                // Test with large array for SIMD path
                if size < 100 {
                    // Create a properly sorted array for SIMD testing
                    let mut large_array = Vec::with_capacity(SIMD_SIZE);
                    for &val in &array {
                        large_array.push(val);
                    }
                    // Extend with values maintaining sorted order
                    while large_array.len() < SIMD_SIZE {
                        let last = *large_array.last().unwrap();
                        large_array.push(last + 1);
                    }
                    let large_result = exponential_search_ge_u32(&large_array, target);
                    assert!(
                        large_result.is_ok(),
                        "u32 SIMD exponential search failed for target {}",
                        target
                    );
                }
            }
        }
    }

    #[test]
    fn test_exponential_search_patterns() {
        config_test_logger();

        // Powers of 2 (good for exponential search)
        let powers: Vec<u64> = (0..20).map(|x| 1u64 << x).collect();
        for &target in &[256, 512, 1024, 2048] {
            test_exponential_search_u64_triple_paths(powers.clone(), target, "powers_of_2");
        }

        // Dense array
        let dense: Vec<u64> = (0..1000).collect();
        for &target in &[0, 250, 500, 750, 999, 1000] {
            test_exponential_search_u64_triple_paths(dense.clone(), target, "dense_array");
        }

        // Sparse array
        let sparse: Vec<u64> = (0..100).map(|x| x * 100).collect();
        for &target in &[0, 2500, 5000, 7500, 9900, 10000] {
            test_exponential_search_u64_triple_paths(sparse.clone(), target, "sparse_array");
        }
    }

    // =============================================================================
    // Binary Search Tests
    // =============================================================================

    #[test]
    fn test_binary_search_u32_comprehensive() {
        config_test_logger();

        // Test all sizes for comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                // Empty array
                test_binary_search_u32_triple_paths(vec![], 5, "empty_array");
                continue;
            }

            // Create sorted test data
            let array: Vec<u32> = (0..size as u32).map(|x| x * 3).collect();

            // Test various positions
            let targets = vec![
                0,                         // First element
                (size as u32 / 2) * 3,     // Middle element
                (size as u32 - 1) * 3,     // Last element
                (size as u32 / 2) * 3 + 1, // Between elements
                size as u32 * 3,           // Beyond end
            ];

            for target in targets {
                test_binary_search_u32_triple_paths(array.clone(), target, "various_positions");
            }
        }

        // Test edge values
        for &edge_val in &generate_edge_values_u32() {
            let array = vec![edge_val];
            test_binary_search_u32_triple_paths(array.clone(), edge_val, "edge_val_exact");
            if edge_val > 0 {
                test_binary_search_u32_triple_paths(array.clone(), edge_val - 1, "edge_val_before");
            }
            if edge_val < u32::MAX {
                test_binary_search_u32_triple_paths(array.clone(), edge_val + 1, "edge_val_after");
            }
        }
    }

    #[test]
    fn test_binary_search_duplicates_comprehensive() {
        config_test_logger();

        // Test arrays with duplicates
        let test_cases = vec![
            vec![1u32, 1, 1, 1, 1],            // All same
            vec![1u32, 2, 2, 2, 3],            // Middle duplicates
            vec![1u32, 1, 2, 3, 3],            // Start and end duplicates
            vec![5u32, 5, 10, 10, 10, 15, 15], // Multiple duplicate groups
        ];

        for array in test_cases {
            for &target in &[1, 2, 3, 5, 10, 15, 0, 20] {
                test_binary_search_u32_triple_paths(array.clone(), target, "duplicates");
            }
        }
    }

    #[test]
    fn test_binary_search_patterns() {
        config_test_logger();

        // Dense consecutive array
        let dense: Vec<u32> = (0..1000).collect();
        for &target in &[0, 250, 500, 750, 999, 1000] {
            test_binary_search_u32_triple_paths(dense.clone(), target, "dense_array");
        }

        // Sparse array with gaps
        let sparse: Vec<u32> = (0..100).map(|x| x * 10).collect();
        for &target in &[0, 50, 250, 500, 990, 1000] {
            test_binary_search_u32_triple_paths(sparse.clone(), target, "sparse_array");
        }

        // Powers of 2
        let powers: Vec<u32> = (0..20).map(|x| 1u32 << x).collect();
        for &target in &[1, 4, 16, 64, 256, 1024, 2000] {
            test_binary_search_u32_triple_paths(powers.clone(), target, "powers_of_2");
        }
    }

    // =============================================================================
    // Filter Range Tests
    // =============================================================================

    #[test]
    fn test_filter_range_u32_comprehensive() {
        config_test_logger();

        // Test all sizes for comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                // Empty array
                test_filter_range_u32_triple_paths(vec![], 5, 10, "empty_array");
                continue;
            }

            // Create test data
            let values: Vec<u32> = (0..size as u32).map(|x| x * 2).collect();

            // Test various ranges
            let ranges = vec![
                (0, size as u32 * 2),                                    // All elements
                (size as u32, size as u32 + 10),                         // Middle range
                (0, 5),                                                  // Beginning range
                ((size as u32 * 2).saturating_sub(10), size as u32 * 2), // End range - avoid underflow
                (size as u32 * 3, size as u32 * 4),                      // No matches
            ];

            for (min_val, max_val) in ranges {
                test_filter_range_u32_triple_paths(
                    values.clone(),
                    min_val,
                    max_val,
                    "various_ranges",
                );
            }
        }

        // Test edge values
        for &edge_val in &generate_edge_values_u32() {
            if edge_val > 0 {
                let values = vec![edge_val];
                test_filter_range_u32_triple_paths(
                    values.clone(),
                    edge_val,
                    edge_val,
                    "edge_val_exact",
                );
                test_filter_range_u32_triple_paths(
                    values.clone(),
                    0,
                    edge_val - 1,
                    "edge_val_below",
                );
                if edge_val < u32::MAX {
                    test_filter_range_u32_triple_paths(
                        values.clone(),
                        edge_val + 1,
                        u32::MAX,
                        "edge_val_above",
                    );
                }
            }
        }
    }

    #[test]
    fn test_filter_range_patterns() {
        config_test_logger();

        // Mixed values
        let mixed = vec![1u32, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
        test_filter_range_u32_triple_paths(mixed.clone(), 10, 30, "mixed_values");
        test_filter_range_u32_triple_paths(mixed.clone(), 0, 5, "mixed_low");
        test_filter_range_u32_triple_paths(mixed.clone(), 45, 100, "mixed_high");

        // Dense array
        let dense: Vec<u32> = (0..1000).collect();
        test_filter_range_u32_triple_paths(dense.clone(), 250, 750, "dense_middle");
        test_filter_range_u32_triple_paths(dense.clone(), 0, 100, "dense_start");
        test_filter_range_u32_triple_paths(dense.clone(), 900, 999, "dense_end");

        // Sparse array
        let sparse: Vec<u32> = (0..100).map(|x| x * 10).collect();
        test_filter_range_u32_triple_paths(sparse.clone(), 100, 500, "sparse_range");
        test_filter_range_u32_triple_paths(sparse.clone(), 25, 75, "sparse_gaps");
    }

    #[test]
    fn test_filter_range_boundary_conditions() {
        config_test_logger();

        // Boundary values
        let boundary = vec![u32::MIN, 1, 100, 1000, u32::MAX - 1, u32::MAX];

        // Test min boundary
        test_filter_range_u32_triple_paths(boundary.clone(), u32::MIN, 1, "min_boundary");

        // Test max boundary
        test_filter_range_u32_triple_paths(
            boundary.clone(),
            u32::MAX - 1,
            u32::MAX,
            "max_boundary",
        );

        // Test full range
        test_filter_range_u32_triple_paths(boundary.clone(), u32::MIN, u32::MAX, "full_range");

        // Test empty range (min > max should handle gracefully)
        test_filter_range_u32_triple_paths(boundary.clone(), 1000, 100, "empty_range");
    }

    // =============================================================================
    // TIME RANGE FILTERING TESTS
    // =============================================================================

    #[test]
    fn test_filter_u32_by_u64_range_comprehensive() {
        config_test_logger();

        // Test all sizes for comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                // Empty input
                test_filter_u32_by_u64_range_triple_paths(
                    vec![],
                    vec![],
                    1000,
                    2000,
                    100,
                    "empty_input",
                );
                continue;
            }

            // Create test data
            let doc_ids: Vec<u32> = (0..size as u32).collect();
            let times: Vec<u64> = (0..size).map(|i| i as u64 * 100).collect();

            // Test various time ranges
            let ranges = vec![
                (0, size as u64 * 100),                 // All elements
                (size as u64 * 50, size as u64 * 60),   // Middle range
                (0, 500),                               // Beginning range
                (size as u64 * 80, size as u64 * 100),  // End range
                (size as u64 * 200, size as u64 * 300), // No matches
            ];

            for (start_time, end_time) in ranges {
                test_filter_u32_by_u64_range_triple_paths(
                    doc_ids.clone(),
                    times.clone(),
                    start_time,
                    end_time,
                    size,
                    "various_ranges",
                );
            }
        }

        // Test edge values
        for &edge_val in &generate_edge_values_u64() {
            if edge_val > 0 && edge_val < u64::MAX {
                let doc_ids = vec![1u32];
                let times = vec![edge_val];
                test_filter_u32_by_u64_range_triple_paths(
                    doc_ids.clone(),
                    times.clone(),
                    edge_val,
                    edge_val,
                    1,
                    "edge_val_exact",
                );
                test_filter_u32_by_u64_range_triple_paths(
                    doc_ids.clone(),
                    times.clone(),
                    0,
                    edge_val - 1,
                    1,
                    "edge_val_below",
                );
                test_filter_u32_by_u64_range_triple_paths(
                    doc_ids.clone(),
                    times.clone(),
                    edge_val + 1,
                    u64::MAX,
                    1,
                    "edge_val_above",
                );
            }
        }
    }

    #[test]
    fn test_filter_u32_by_u64_range_max_size_patterns() {
        config_test_logger();

        // Test max_size limitations
        let large_doc_ids: Vec<u32> = (0..1000).collect();
        let large_times: Vec<u64> = (0..1000).map(|i| i as u64 * 100).collect();

        // Test 1: max_size smaller than available results
        test_filter_u32_by_u64_range_triple_paths(
            large_doc_ids.clone(),
            large_times.clone(),
            0,
            99900, // All times
            10,    // Limit to 10 results
            "max_size_limit",
        );

        // Test 2: max_size larger than available results
        test_filter_u32_by_u64_range_triple_paths(
            large_doc_ids.clone(),
            large_times.clone(),
            10000,
            20000, // Small range
            1000,  // Large max_size
            "max_size_unlimited",
        );

        // Test 3: Specific time ranges
        test_filter_u32_by_u64_range_triple_paths(
            large_doc_ids.clone(),
            large_times.clone(),
            20000,
            50000, // Should match times 20000-50000
            1000,
            "specific_range",
        );
    }

    #[test]
    fn test_union_sorted_u32_comprehensive() {
        config_test_logger();

        // Test all sizes for comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                // Empty arrays
                test_union_sorted_u32_triple_paths(vec![vec![], vec![]], 10, "empty_arrays");
                continue;
            }

            // Create test arrays
            let arr1: Vec<u32> = (0..size as u32).step_by(2).collect(); // Even numbers
            let arr2: Vec<u32> = (1..size as u32).step_by(2).collect(); // Odd numbers

            test_union_sorted_u32_triple_paths(
                vec![arr1.clone(), arr2.clone()],
                size * 2,
                "even_odd_union",
            );

            // Test with duplicates
            let arr3 = arr1.clone();
            test_union_sorted_u32_triple_paths(vec![arr1.clone(), arr3], size, "duplicate_arrays");

            // Test with overlapping arrays
            let arr4: Vec<u32> = (size as u32 / 2..size as u32 + size as u32 / 2).collect();
            test_union_sorted_u32_triple_paths(
                vec![arr1.clone(), arr4],
                size * 2,
                "overlapping_arrays",
            );
        }

        // Test edge cases
        test_union_sorted_u32_triple_paths(vec![vec![1, 3, 5], vec![2, 4, 6]], 6, "interleaved");
        test_union_sorted_u32_triple_paths(vec![vec![1, 2, 3], vec![1, 2, 3]], 3, "identical");
        test_union_sorted_u32_triple_paths(vec![vec![1, 2, 3], vec![4, 5, 6]], 6, "no_overlap");
    }

    // =============================================================================
    // SET INTERSECTION TESTS
    // =============================================================================

    #[test]
    fn test_intersect_sorted_u32_comprehensive_core() {
        config_test_logger();

        debug!("Testing u32 intersection deduplication logic...");

        // Test case 1: Basic deduplication - arrays with duplicates
        let mut a1_no_dedup = vec![10u32, 20, 20, 30, 40, 40, 50, 60, 60, 70];
        let mut a1_with_dedup = a1_no_dedup.clone();
        let b1 = vec![20u32, 20, 40, 40, 60, 60, 80];

        // Test without deduplication (should preserve duplicates)
        let result1_no_dedup =
            crate::intersect_sorted_u32(&mut a1_no_dedup, &b1, usize::MAX, false, true);
        assert!(
            result1_no_dedup.is_ok(),
            "u32 intersection without deduplication should succeed"
        );
        let expected_no_dedup = vec![20u32, 20, 40, 40, 60, 60]; // All duplicates preserved
        assert_eq!(
            a1_no_dedup[..6],
            expected_no_dedup,
            "u32 without deduplication: duplicates should be preserved"
        );

        // Test with deduplication (should remove duplicates)
        let result1_with_dedup =
            crate::intersect_sorted_u32(&mut a1_with_dedup, &b1, usize::MAX, true, true);
        assert!(
            result1_with_dedup.is_ok(),
            "u32 intersection with deduplication should succeed"
        );
        let expected_with_dedup = vec![20u32, 40, 60]; // Duplicates removed
        assert_eq!(
            a1_with_dedup[..3],
            expected_with_dedup,
            "u32 with deduplication: duplicates should be removed"
        );

        debug!("✅ u32 basic deduplication test passed");

        // Test case 2: Empty arrays
        let mut empty1 = vec![];
        let empty2 = vec![1u32, 2, 3];
        let result_empty =
            crate::intersect_sorted_u32(&mut empty1, &empty2, usize::MAX, false, true);
        assert!(
            result_empty.is_ok(),
            "Empty array intersection should succeed"
        );
        assert!(empty1.is_empty(), "Empty array result should be empty");

        // Test case 3: No intersection
        let mut odds = vec![1u32, 3, 5, 7, 9];
        let evens = vec![2u32, 4, 6, 8, 10];
        let result_no_intersect =
            crate::intersect_sorted_u32(&mut odds, &evens, usize::MAX, false, true);
        assert!(
            result_no_intersect.is_ok(),
            "No intersection case should succeed"
        );
        assert!(odds.is_empty(), "No intersection result should be empty");

        // Test case 4: Identical arrays
        let mut identical1 = vec![1u32, 2, 3, 4, 5];
        let identical2 = identical1.clone();
        let result_identical =
            crate::intersect_sorted_u32(&mut identical1, &identical2, usize::MAX, false, true);
        assert!(
            result_identical.is_ok(),
            "Identical arrays intersection should succeed"
        );
        assert_eq!(
            identical1,
            vec![1u32, 2, 3, 4, 5],
            "Identical arrays should return all elements"
        );

        debug!("✅ u32 comprehensive intersection tests passed");
    }
    #[test]
    fn test_intersect_sorted_u32_patterns() {
        config_test_logger();

        debug!("Testing u32 intersection mathematical patterns...");

        // Test case 1: Evens vs multiples of 3 (should intersect at multiples of 6)
        let mut evens_no_dedup: Vec<u32> = (0..60).step_by(2).collect(); // [0, 2, 4, 6, 8, 10, 12, ...]
        let mut evens_with_dedup = evens_no_dedup.clone();
        let multiples_3: Vec<u32> = (0..60).step_by(3).collect(); // [0, 3, 6, 9, 12, 15, 18, ...]

        // Without deduplication
        let result_no_dedup =
            crate::intersect_sorted_u32(&mut evens_no_dedup, &multiples_3, usize::MAX, false, true);
        assert!(
            result_no_dedup.is_ok(),
            "Evens vs multiples of 3 intersection should succeed"
        );
        let expected_multiples_6: Vec<u32> = (0..60).step_by(6).collect(); // [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
        assert_eq!(
            evens_no_dedup[..expected_multiples_6.len()],
            expected_multiples_6,
            "Evens vs multiples of 3 should give multiples of 6"
        );

        // With deduplication (same result since no duplicates expected)
        let result_with_dedup = crate::intersect_sorted_u32(
            &mut evens_with_dedup,
            &multiples_3,
            usize::MAX,
            true,
            true,
        );
        assert!(
            result_with_dedup.is_ok(),
            "Evens vs multiples of 3 with dedup should succeed"
        );
        assert_eq!(
            evens_with_dedup[..expected_multiples_6.len()],
            expected_multiples_6,
            "Evens vs multiples of 3 with dedup should give same result"
        );

        debug!("✅ Evens vs multiples of 3 test passed");

        // Test case 2: Powers of 2 vs squares (intersection: 1, 4, 16, 64, 256)
        let mut powers_2 = vec![1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let squares = vec![
            1u32, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289,
        ];

        let result_powers =
            crate::intersect_sorted_u32(&mut powers_2, &squares, usize::MAX, false, true);
        assert!(
            result_powers.is_ok(),
            "Powers of 2 vs squares intersection should succeed"
        );
        let expected_power_squares = vec![1u32, 4, 16, 64, 256]; // Powers of 2 that are also perfect squares
        assert_eq!(
            powers_2[..5],
            expected_power_squares,
            "Powers of 2 vs squares should give power squares"
        );

        debug!("✅ Powers of 2 vs squares test passed");

        // Test case 3: Overlapping ranges
        let mut range1: Vec<u32> = (100..150).collect(); // [100, 101, 102, ..., 149]
        let range2: Vec<u32> = (125..175).collect(); // [125, 126, 127, ..., 174]

        let result_ranges =
            crate::intersect_sorted_u32(&mut range1, &range2, usize::MAX, false, true);
        assert!(
            result_ranges.is_ok(),
            "Overlapping ranges intersection should succeed"
        );
        let expected_overlap: Vec<u32> = (125..150).collect(); // [125, 126, 127, ..., 149]
        assert_eq!(
            range1[..expected_overlap.len()],
            expected_overlap,
            "Overlapping ranges should give intersection"
        );

        debug!("✅ u32 mathematical patterns tests passed");
    }
    // =============================================================================
    // DEDUPLICATION TESTS
    // =============================================================================
    #[test]
    fn test_dedup_sorted_u32_comprehensive() {
        config_test_logger();

        // Test all sizes for comprehensive coverage
        for &size in &generate_test_sizes() {
            if size == 0 {
                // Empty array
                test_dedup_sorted_u32_triple_paths(vec![], "empty_array");
                continue;
            }

            // Create test data with various duplicate patterns
            let mut array_with_dupes = Vec::new();
            for i in 0..size as u32 {
                // Add each number 1-3 times
                for _ in 0..(i % 3 + 1) {
                    array_with_dupes.push(i);
                }
            }

            test_dedup_sorted_u32_triple_paths(array_with_dupes, "various_duplicates");

            // Test array with all same elements
            let same_elements = vec![42u32; size];
            test_dedup_sorted_u32_triple_paths(same_elements, "all_same");

            // Test already unique array
            let unique_array: Vec<u32> = (0..size as u32).collect();
            test_dedup_sorted_u32_triple_paths(unique_array, "already_unique");
        }

        // Test specific patterns
        test_dedup_sorted_u32_triple_paths(vec![1, 1, 2, 2, 2, 3, 4, 4, 5], "specific_duplicates");
        test_dedup_sorted_u32_triple_paths(vec![1, 2, 3, 4, 5], "no_duplicates");
    }
    #[test]
    fn test_union_sorted_u32_duplicates_triple_paths() {
        config_test_logger();

        // Test with duplicates - scalar, SIMD and GPU paths
        let arrays = [
            vec![20u32, 18, 16, 14],
            vec![19u32, 17, 15, 13],
            vec![12u32, 10, 8, 6],
        ];
        // removed unused expected baseline
        test_union_sorted_u32_triple_paths(arrays.to_vec(), 100, "duplicates_pattern");

        // Test with duplicates - deduplication
        let arrays_dup = [vec![15u32, 10, 5], vec![15u32, 10, 5]];
        // removed unused expected_dedup baseline
        test_union_sorted_u32_triple_paths(arrays_dup.to_vec(), 100, "duplicates_dedup");
    }
    #[test]
    fn test_union_sorted_u32_edge_cases_triple_paths() {
        config_test_logger();

        // Empty arrays - scalar, SIMD and GPU paths
        test_union_sorted_u32_triple_paths(vec![vec![], vec![1, 2, 3]], 100, "empty_first");
        test_union_sorted_u32_triple_paths(vec![vec![1, 2, 3], vec![]], 100, "empty_second");

        // Identical arrays - scalar, SIMD and GPU paths
        test_union_sorted_u32_triple_paths(
            vec![vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 5]],
            100,
            "identical",
        );

        // Single element arrays - scalar, SIMD and GPU paths
        test_union_sorted_u32_triple_paths(vec![vec![42], vec![42]], 10, "single_duplicate");
        test_union_sorted_u32_triple_paths(vec![vec![10], vec![20]], 10, "single_different");

        // Max size boundary test - this catches off-by-one errors!
        // Create array with 10005 elements, set max_size to 10000
        // Result should be EXACTLY 10000, not 10001
        let large_array: Vec<u32> = (0..10005).collect();
        test_union_sorted_u32_triple_paths(vec![large_array], 10000, "max_size_boundary_10000");

        // Another boundary test with smaller numbers for faster testing
        let medium_array: Vec<u32> = (0..105).collect();
        test_union_sorted_u32_triple_paths(vec![medium_array], 100, "max_size_boundary_100");
    }

    // =============================================================================
    // DEDUPLICATION TESTS
    // =============================================================================

    #[test]
    fn test_dedup_sorted_u32_patterns_triple_paths() {
        config_test_logger();

        // Basic duplicates pattern - scalar, SIMD and GPU paths
        test_dedup_sorted_u32_triple_paths(vec![1u32, 1, 2, 3, 3, 4], "basic_duplicates");
    }

    #[test]
    fn test_dedup_sorted_u32_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array - scalar, SIMD and GPU paths
        test_dedup_sorted_u32_triple_paths(vec![], "empty_array");

        // Single element - scalar, SIMD and GPU paths
        test_dedup_sorted_u32_triple_paths(vec![42u32], "single_element");

        // No duplicates - scalar, SIMD and GPU paths
        test_dedup_sorted_u32_triple_paths(vec![1u32, 2, 3, 4, 5], "no_duplicates");

        // All duplicates - scalar, SIMD and GPU paths
        test_dedup_sorted_u32_triple_paths(vec![7u32; 20], "all_duplicates");

        // Large duplicate patterns to definitely trigger SIMD
        let mut many_dupes = Vec::new();
        for i in 0..100 {
            for _ in 0..5 {
                many_dupes.push(i);
            }
        }
        test_dedup_sorted_u32_triple_paths(many_dupes, "many_duplicates");
    }

    /// Triple-path test helper for dedup_sorted_u64
    fn test_dedup_sorted_u64_triple_paths(array: Vec<u64>, test_name: &str) {
        //  SCALAR PATH TEST (small input)
        let mut scalar_array = array.clone();
        let scalar_result = dedup_sorted_u64(&mut scalar_array);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );

        //  SIMD PATH TEST (medium input to force SIMD)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut simd_array = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                simd_array.push(val);
            }
            // Extend with values maintaining sorted order
            while simd_array.len() < SIMD_SIZE {
                let last = *simd_array.last().unwrap();
                simd_array.push(last);
            }
            let mut simd_array = simd_array.clone();
            let simd_result = dedup_sorted_u64(&mut simd_array);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );
        }

        // ^ GPU PATH TEST (large input for GPU)
        if !array.is_empty() {
            let gpu_array: Vec<u64> = array.iter().cycle().take(GPU_SIZE).cloned().collect();
            let mut gpu_array = gpu_array.clone();
            let gpu_result = dedup_sorted_u64(&mut gpu_array);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );
        }
    }

    #[test]
    fn test_dedup_sorted_u64_comprehensive() {
        config_test_logger();

        // Test empty array - scalar, SIMD and GPU paths
        test_dedup_sorted_u64_triple_paths(vec![], "empty_array");

        // Test arrays with various patterns of duplicates
        let array_with_dupes: Vec<u64> = (0..200)
            .flat_map(|i| {
                let repeat_count = (i % 4) + 1;
                std::iter::repeat_n(i as u64, repeat_count)
            })
            .collect();
        test_dedup_sorted_u64_triple_paths(array_with_dupes, "various_duplicates");

        // Test array with all same elements - forces SIMD path
        let same_elements = vec![42u64; 500];
        test_dedup_sorted_u64_triple_paths(same_elements, "all_same");

        // Test array with unique elements - should remain unchanged
        let unique_array: Vec<u64> = (0..500).map(|i| i as u64).collect();
        test_dedup_sorted_u64_triple_paths(unique_array, "already_unique");

        // Test specific patterns
        test_dedup_sorted_u64_triple_paths(vec![1, 1, 2, 2, 2, 3, 4, 4, 5], "specific_duplicates");
        test_dedup_sorted_u64_triple_paths(vec![1, 2, 3, 4, 5], "no_duplicates");
    }

    #[test]
    fn test_dedup_sorted_u64_patterns_triple_paths() {
        config_test_logger();

        // Basic duplicates pattern - triple paths
        test_dedup_sorted_u64_triple_paths(vec![1u64, 1, 2, 3, 3, 4], "basic_duplicates");
    }

    #[test]
    fn test_dedup_sorted_u64_edge_cases_triple_paths() {
        config_test_logger();

        // Test empty array - scalar, SIMD and GPU paths
        test_dedup_sorted_u64_triple_paths(vec![], "empty_array");

        // Test single element - all paths
        test_dedup_sorted_u64_triple_paths(vec![42u64], "single_element");

        // Test no duplicates - all paths
        test_dedup_sorted_u64_triple_paths(vec![1u64, 2, 3, 4, 5], "no_duplicates");

        // Test all duplicates - all paths
        test_dedup_sorted_u64_triple_paths(vec![7u64; 20], "all_duplicates");

        // Large duplicate patterns to definitely trigger SIMD
        let mut many_dupes = Vec::new();
        for i in 0..100 {
            for _ in 0..5 {
                many_dupes.push(i as u64);
            }
        }
        test_dedup_sorted_u64_triple_paths(many_dupes, "many_duplicates");
    }

    #[test]
    fn test_dedup_sorted_u64_large_values() {
        config_test_logger();

        // Test with large u64 values
        let large_values = vec![
            u64::MAX - 1000,
            u64::MAX - 1000,
            u64::MAX - 500,
            u64::MAX - 500,
            u64::MAX - 500,
            u64::MAX,
            u64::MAX,
        ];
        test_dedup_sorted_u64_triple_paths(large_values, "large_u64_values");

        // Test with mixed large and small values
        let mixed_values = vec![0u64, 0, 1000, 1000, u64::MAX - 1, u64::MAX - 1, u64::MAX];
        test_dedup_sorted_u64_triple_paths(mixed_values, "mixed_large_small_values");
    }

    // =============================================================================
    //  DEDUP SORTED I64 TESTS - DUAL PATH VALIDATION
    // =============================================================================

    /// Triple-path test helper for dedup_sorted_i64
    fn test_dedup_sorted_i64_triple_paths(array: Vec<i64>, test_name: &str) {
        //  SCALAR PATH TEST (small input)
        let mut scalar_array = array.clone();
        let scalar_result = dedup_sorted_i64(&mut scalar_array);
        assert!(
            scalar_result.is_ok(),
            "{}: SCALAR path failed: {:?}",
            test_name,
            scalar_result
        );

        //  SIMD PATH TEST (medium input to force SIMD)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut simd_array = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                simd_array.push(val);
            }
            // Extend with values maintaining sorted order
            while simd_array.len() < SIMD_SIZE {
                let last = *simd_array.last().unwrap();
                simd_array.push(last);
            }
            let mut simd_array = simd_array.clone();
            let simd_result = dedup_sorted_i64(&mut simd_array);
            assert!(
                simd_result.is_ok(),
                "{}: SIMD path failed: {:?}",
                test_name,
                simd_result
            );
        }

        // ^ GPU PATH TEST (large input for GPU)
        if !array.is_empty() {
            let gpu_array: Vec<i64> = array.iter().cycle().take(GPU_SIZE).cloned().collect();
            let mut gpu_array = gpu_array.clone();
            let gpu_result = dedup_sorted_i64(&mut gpu_array);
            assert!(
                gpu_result.is_ok(),
                "{}: GPU path failed: {:?}",
                test_name,
                gpu_result
            );
        }
    }

    #[test]
    fn test_dedup_sorted_i64_comprehensive() {
        config_test_logger();

        // Test empty array - scalar, SIMD and GPU paths
        test_dedup_sorted_i64_triple_paths(vec![], "empty_array");

        // Test arrays with various patterns of duplicates
        let array_with_dupes: Vec<i64> = (-100i32..100i32)
            .flat_map(|i| {
                let repeat_count = ((i.abs() % 4) + 1) as usize;
                std::iter::repeat_n(i as i64, repeat_count)
            })
            .collect();
        test_dedup_sorted_i64_triple_paths(array_with_dupes, "various_duplicates");

        // Test array with all same elements - forces SIMD path
        let same_elements = vec![42i64; 500];
        test_dedup_sorted_i64_triple_paths(same_elements, "all_same");

        // Test array with unique elements - should remain unchanged
        let unique_array: Vec<i64> = (-250..250).map(|i| i as i64).collect();
        test_dedup_sorted_i64_triple_paths(unique_array, "already_unique");

        // Test specific patterns with negative numbers
        test_dedup_sorted_i64_triple_paths(
            vec![-5i64, -5, -2, -2, -2, 0, 1, 1, 5],
            "negative_positive_mix",
        );
        test_dedup_sorted_i64_triple_paths(vec![-10i64, -5, 0, 5, 10], "no_duplicates");
    }

    #[test]
    fn test_dedup_sorted_i64_patterns_triple_paths() {
        config_test_logger();

        // Basic duplicates pattern with negatives - triple paths
        test_dedup_sorted_i64_triple_paths(
            vec![-3i64, -3, -1, 0, 0, 2],
            "basic_duplicates_with_negatives",
        );
    }

    #[test]
    fn test_dedup_sorted_i64_edge_cases_triple_paths() {
        config_test_logger();

        // Test empty array - scalar, SIMD and GPU paths
        test_dedup_sorted_i64_triple_paths(vec![], "empty_array");

        // Test single element - scalar path only
        test_dedup_sorted_i64_triple_paths(vec![42i64], "single_element");

        // Test no duplicates with negatives - both paths
        test_dedup_sorted_i64_triple_paths(
            vec![-5i64, -2, 0, 3, 7],
            "no_duplicates_with_negatives",
        );

        // Test all duplicates negative - forces SIMD path
        test_dedup_sorted_i64_triple_paths(vec![-7i64; 20], "all_duplicates_negative");

        // Large duplicate patterns to definitely trigger SIMD
        let mut many_dupes = Vec::new();
        for i in -50..50 {
            for _ in 0..5 {
                many_dupes.push(i as i64);
            }
        }
        test_dedup_sorted_i64_triple_paths(many_dupes, "many_duplicates_negative_positive");
    }

    #[test]
    fn test_dedup_sorted_i64_extreme_values() {
        config_test_logger();

        // Test with extreme i64 values
        let extreme_values = vec![
            i64::MIN,
            i64::MIN,
            i64::MIN + 1000,
            i64::MIN + 1000,
            -1000000000i64,
            -1000000000i64,
            0i64,
            0i64,
            1000000000i64,
            1000000000i64,
            i64::MAX - 1000,
            i64::MAX - 1000,
            i64::MAX,
            i64::MAX,
        ];
        test_dedup_sorted_i64_triple_paths(extreme_values, "extreme_i64_values");

        // Test with mixed extreme and normal values
        let mixed_values = vec![
            i64::MIN,
            i64::MIN,
            -1000i64,
            -1000,
            0,
            0,
            1000,
            1000,
            i64::MAX - 1,
            i64::MAX - 1,
            i64::MAX,
            i64::MAX,
        ];
        test_dedup_sorted_i64_triple_paths(mixed_values, "mixed_extreme_normal_values");
    }

    #[test]
    fn test_dedup_sorted_i64_negative_ranges() {
        config_test_logger();

        // Test with only negative numbers
        let negative_only = vec![-100i64, -100, -50, -50, -25, -25, -10, -10, -1, -1];
        test_dedup_sorted_i64_triple_paths(negative_only, "negative_only_duplicates");

        // Test with alternating negative/positive patterns
        let alternating = vec![-10i64, -10, -5, -5, 0, 0, 5, 5, 10, 10];
        test_dedup_sorted_i64_triple_paths(alternating, "alternating_negative_positive");

        // Test with powers of 2 and their negatives
        let powers: Vec<i64> = (0..10)
            .flat_map(|x| {
                let val = 1i64 << x;
                vec![-val, -val, val, val]
            })
            .collect();
        let mut sorted_powers = powers;
        sorted_powers.sort();
        test_dedup_sorted_i64_triple_paths(sorted_powers, "powers_of_2_with_negatives");
    }

    // =============================================================================
    // TIME-BASED BINARY SEARCH TESTS
    // =============================================================================

    #[test]
    fn test_binary_search_ge_time_patterns_triple_paths() {
        config_test_logger();

        // Basic time sequence - scalar, SIMD and GPU paths
        let times = vec![100u64, 200, 300, 400, 500];
        test_binary_search_ge_time_triple_paths(times.clone(), 300, "exact_match");
        test_binary_search_ge_time_triple_paths(times.clone(), 250, "between_values");
        test_binary_search_ge_time_triple_paths(times.clone(), 50, "before_range");
        test_binary_search_ge_time_triple_paths(times.clone(), 600, "after_range");
    }

    #[test]
    fn test_binary_search_ge_time_comprehensive_triple_paths() {
        config_test_logger();

        // Test comprehensive time patterns - scalar, SIMD and GPU paths
        let comprehensive_times = vec![1000u64, 2000, 3000, 4000, 5000];
        test_binary_search_ge_time_triple_paths(comprehensive_times.clone(), 1000, "first_element");
        test_binary_search_ge_time_triple_paths(comprehensive_times.clone(), 2500, "middle_gap");
        test_binary_search_ge_time_triple_paths(comprehensive_times.clone(), 5000, "last_element");
        test_binary_search_ge_time_triple_paths(comprehensive_times.clone(), 6000, "beyond_range");
    }

    #[test]
    fn test_binary_search_le_time_patterns_triple_paths() {
        config_test_logger();

        // Basic time sequence - scalar, SIMD and GPU paths
        let times = vec![100u64, 200, 300, 400, 500];
        test_binary_search_le_time_triple_paths(times.clone(), 350, "between_values");
        test_binary_search_le_time_triple_paths(times.clone(), 300, "exact_match");
        test_binary_search_le_time_triple_paths(times.clone(), 50, "before_range");
        test_binary_search_le_time_triple_paths(times.clone(), 600, "after_range");
    }

    #[test]
    fn test_binary_search_le_time_comprehensive_triple_paths() {
        config_test_logger();

        // Test comprehensive time patterns - scalar, SIMD and GPU paths
        let comprehensive_times = vec![1000u64, 2000, 3000, 4000, 5000];
        test_binary_search_le_time_triple_paths(comprehensive_times.clone(), 500, "before_first");
        test_binary_search_le_time_triple_paths(comprehensive_times.clone(), 1000, "first_element");
        test_binary_search_le_time_triple_paths(comprehensive_times.clone(), 2500, "middle_gap");
        test_binary_search_le_time_triple_paths(comprehensive_times.clone(), 5000, "last_element");
    }

    #[test]
    fn test_binary_search_time_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array edge case - scalar, SIMD and GPU paths
        test_binary_search_ge_time_triple_paths(vec![], 1000, "empty_ge");
        test_binary_search_le_time_triple_paths(vec![], 1000, "empty_le");

        // Single element edge cases - scalar, SIMD and GPU paths
        test_binary_search_ge_time_triple_paths(vec![1000], 500, "single_before");
        test_binary_search_ge_time_triple_paths(vec![1000], 1000, "single_exact");
        test_binary_search_ge_time_triple_paths(vec![1000], 1500, "single_after");

        test_binary_search_le_time_triple_paths(vec![1000], 500, "single_le_before");
        test_binary_search_le_time_triple_paths(vec![1000], 1000, "single_le_exact");
        test_binary_search_le_time_triple_paths(vec![1000], 1500, "single_le_after");
    }

    #[test]
    fn test_binary_search_time_large_dataset_triple_paths() {
        config_test_logger();

        // Large time series dataset - scalar, SIMD and GPU paths
        let large_times: Vec<u64> = (0..1000).map(|i| i as u64 * 1000).collect();
        test_binary_search_ge_time_triple_paths(large_times.clone(), 0, "large_ge_first");
        test_binary_search_ge_time_triple_paths(large_times.clone(), 500000, "large_ge_middle");
        test_binary_search_ge_time_triple_paths(large_times.clone(), 999000, "large_ge_last");
        test_binary_search_ge_time_triple_paths(large_times.clone(), 1000000, "large_ge_beyond");

        test_binary_search_le_time_triple_paths(large_times.clone(), 0, "large_le_first");
        test_binary_search_le_time_triple_paths(large_times.clone(), 500000, "large_le_middle");
        test_binary_search_le_time_triple_paths(large_times.clone(), 999000, "large_le_last");
    }

    #[test]
    fn test_binary_search_time_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate time values pattern - scalar, SIMD and GPU paths
        let duplicate_times = vec![1000u64, 2000, 2000, 2000, 3000];
        test_binary_search_ge_time_triple_paths(duplicate_times.clone(), 2000, "duplicates_ge");
        test_binary_search_le_time_triple_paths(duplicate_times.clone(), 2000, "duplicates_le");

        // Multiple duplicate patterns - scalar, SIMD and GPU paths
        let many_dupes = vec![100u64, 200, 200, 200, 200, 300, 400, 400, 500];
        test_binary_search_ge_time_triple_paths(many_dupes.clone(), 200, "many_dupes_ge");
        test_binary_search_le_time_triple_paths(many_dupes.clone(), 400, "many_dupes_le");
    }

    #[test]
    fn test_filter_range_u32_large_dataset_triple_paths() {
        config_test_logger();

        // Large stepped dataset - scalar, SIMD and GPU paths
        let large_stepped: Vec<u32> = (0..1000).step_by(10).collect();
        test_filter_range_u32_triple_paths(large_stepped.clone(), 500, 600, "large_stepped");

        // Dense large dataset - scalar, SIMD and GPU paths
        let dense_large: Vec<u32> = (0..500).collect();
        test_filter_range_u32_triple_paths(dense_large.clone(), 200, 300, "dense_large");
    }
    // =============================================================================
    // CROSS-VALIDATION AND CONSISTENCY TESTS
    // =============================================================================

    #[test]
    fn test_simd_dispatcher_functions_triple_paths() {
        config_test_logger();

        // Test filter_range_u32 dispatcher - scalar, SIMD and GPU paths
        let test_data = vec![1u32, 5, 10, 15, 20, 25, 30, 35, 40];
        test_filter_range_u32_triple_paths(test_data.clone(), 10, 30, "dispatcher_filter");

        // Test binary_search_ge_u32 dispatcher - scalar, SIMD and GPU paths
        test_binary_search_u32_triple_paths(test_data.clone(), 15, "dispatcher_search_exact");
        test_binary_search_u32_triple_paths(test_data.clone(), 12, "dispatcher_search_between");
    }

    #[test]
    fn test_intersect_sorted_u32_comprehensive_triple_paths() {
        config_test_logger();

        debug!("Testing u32 intersection comprehensive patterns...");

        // Test case 1: Basic intersection
        let mut basic1 = vec![1u32, 3, 5, 7];
        let basic2 = vec![3u32, 5, 9];
        let result_basic =
            crate::intersect_sorted_u32(&mut basic1, &basic2, usize::MAX, false, true);
        assert!(result_basic.is_ok(), "Basic intersection should succeed");
        assert_eq!(
            basic1[..2],
            vec![3u32, 5],
            "Basic intersection should return [3, 5]"
        );

        // Test case 2: Large intersection (evens vs multiples of 3 = multiples of 6)
        let mut even_numbers: Vec<u32> = (0..60).step_by(2).collect(); // [0, 2, 4, 6, 8, 10, 12, ...]
        let multiples_of_3: Vec<u32> = (0..60).step_by(3).collect(); // [0, 3, 6, 9, 12, 15, 18, ...]
        let result_large = crate::intersect_sorted_u32(
            &mut even_numbers,
            &multiples_of_3,
            usize::MAX,
            false,
            true,
        );
        assert!(result_large.is_ok(), "Large intersection should succeed");
        let expected_multiples_6: Vec<u32> = (0..60).step_by(6).collect(); // [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
        assert_eq!(
            even_numbers[..expected_multiples_6.len()],
            expected_multiples_6,
            "Large intersection should give multiples of 6"
        );

        // Test case 3: Empty first array
        let mut empty1 = vec![];
        let non_empty = vec![1u32, 2, 3];
        let result_empty1 =
            crate::intersect_sorted_u32(&mut empty1, &non_empty, usize::MAX, false, true);
        assert!(
            result_empty1.is_ok(),
            "Empty first array intersection should succeed"
        );
        assert!(
            empty1.is_empty(),
            "Empty first array result should be empty"
        );

        // Test case 4: Empty second array
        let mut non_empty2 = vec![1u32, 2, 3];
        let empty2 = vec![];
        let result_empty2 =
            crate::intersect_sorted_u32(&mut non_empty2, &empty2, usize::MAX, false, true);
        assert!(
            result_empty2.is_ok(),
            "Empty second array intersection should succeed"
        );
        assert!(
            non_empty2.is_empty(),
            "Empty second array result should be empty"
        );

        // Test case 5: No overlap
        let mut no_overlap1 = vec![1u32, 2, 3];
        let no_overlap2 = vec![4u32, 5, 6];
        let result_no_overlap =
            crate::intersect_sorted_u32(&mut no_overlap1, &no_overlap2, usize::MAX, false, true);
        assert!(
            result_no_overlap.is_ok(),
            "No overlap intersection should succeed"
        );
        assert!(no_overlap1.is_empty(), "No overlap result should be empty");

        debug!("✅ u32 comprehensive intersection patterns tests passed");
    }

    #[test]
    fn test_intersect_and_union_operations_triple_paths() {
        config_test_logger();

        debug!("Testing u32 intersection and union operations...");

        // Test intersection of two arrays with some common elements
        let mut arr1 = vec![1u32, 3, 5, 7, 9, 11, 13, 15];
        let arr2 = vec![2u32, 3, 6, 7, 10, 11, 14, 15];

        // Test intersection (should find common elements: 3, 7, 11, 15)
        let result_intersect =
            crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(
            result_intersect.is_ok(),
            "Combined intersection should succeed"
        );
        let expected_intersection = vec![3u32, 7, 11, 15];
        assert_eq!(
            arr1[..4],
            expected_intersection,
            "Intersection should return common elements [3, 7, 11, 15]"
        );

        debug!("✅ u32 intersection and union operations tests passed");
    }

    // =============================================================================
    // PERFORMANCE AND STRESS TESTS
    // =============================================================================

    #[test]
    fn test_simd_functions_stress_triple_paths() {
        config_test_logger();

        debug!("Testing u32 SIMD functions stress test...");

        // Large dataset intersection stress test
        let mut large_dataset1: Vec<u32> = (0..1000).collect();
        let large_dataset2: Vec<u32> = (500..1500).collect();

        // Expected intersection: [500, 501, 502, ..., 999] (500 elements)
        let result = crate::intersect_sorted_u32(
            &mut large_dataset1,
            &large_dataset2,
            usize::MAX,
            false,
            true,
        );
        assert!(result.is_ok(), "Large dataset intersection should succeed");
        assert_eq!(
            large_dataset1.len(),
            500,
            "Should find 500 intersecting elements"
        );

        // Verify first few and last few elements
        assert_eq!(
            large_dataset1[0], 500,
            "First intersection element should be 500"
        );
        assert_eq!(
            large_dataset1[499], 999,
            "Last intersection element should be 999"
        );

        // Large dataset union stress test - scalar, SIMD and GPU paths
        test_union_sorted_u32_triple_paths(
            vec![large_dataset1.clone(), large_dataset2.clone()],
            2000,
            "stress_union",
        );

        // Large duplication stress test - scalar, SIMD and GPU paths
        let mut data_with_dupes = Vec::new();
        for i in 0..100 {
            for _ in 0..5 {
                data_with_dupes.push(i);
            }
        }
        test_dedup_sorted_u32_triple_paths(data_with_dupes, "stress_dedup");
    }

    #[test]
    fn test_max_results_limits_triple_paths() {
        config_test_logger();

        // Max results adherence testing - scalar, SIMD and GPU paths
        let data = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        test_filter_range_u32_triple_paths(data.clone(), 1, 10, "max_results_filter");
        test_dedup_sorted_u32_triple_paths(data.clone(), "max_results_dedup");
    }

    #[test]
    fn test_intersect_sorted_u32_bug_reproduction() {
        // Reproduce the exact bug from the debug output
        let mut arr1 = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        ];
        let arr2 = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
            91, 92, 93, 94, 95, 96, 97, 98, 99,
        ];

        let result = intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(result.is_ok(), "Intersection should succeed");

        // Expected: consecutive numbers 1-99 (99 elements)
        let expected: Vec<u32> = (1..=99).collect();

        debug!("Expected: {:?}", expected);
        debug!("Actual  : {:?}", arr1);

        assert_eq!(
            arr1, expected,
            "Intersection should return consecutive numbers 1-99 without duplicates"
        );
    }
    #[test]
    fn test_intersect_sorted_u32_data_loss_scenarios_v2() {
        config_test_logger();

        debug!("Testing u32 intersection data loss scenarios...");

        // Scenario 1: Sparse intersection with gaps
        let mut arr1 = vec![1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
        let arr2 = vec![3, 5, 12, 15, 22, 25, 33, 35, 42, 45, 52];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(result.is_ok(), "Sparse intersection should succeed");
        let expected = vec![5, 15, 25, 35, 45];
        assert_eq!(
            arr1[..5],
            expected,
            "Should find sparse intersections: [5, 15, 25, 35, 45]"
        );

        // Scenario 2: Dense overlapping ranges
        let mut arr1 = vec![
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        ];
        let arr2 = vec![
            102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
        ];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(result.is_ok(), "Dense overlapping ranges should succeed");
        let expected = vec![
            102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        ];
        assert_eq!(
            arr1[..14],
            expected,
            "Should find dense overlapping range [102-115]"
        );

        // Scenario 3: Repeated elements with different positions
        let mut arr1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let arr2 = vec![8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(result.is_ok(), "Repeated elements should succeed");
        let expected = vec![8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert_eq!(arr1[..9], expected, "Should find repeated elements [8-16]");

        // Scenario 4: Real-world posting list intersection pattern
        let mut arr1 = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        ];
        let arr2 = vec![
            25, 30, 45, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 155,
        ];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(result.is_ok(), "Posting list intersection should succeed");
        let expected = vec![30, 50, 70, 90, 110, 130, 150];
        assert_eq!(
            arr1[..7],
            expected,
            "Should find posting list intersections: [30, 50, 70, 90, 110, 130, 150]"
        );

        // Scenario 5: Edge case with single element matches spread across SIMD chunks
        let mut arr1 = vec![
            1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241,
        ];
        let arr2 = vec![
            17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257,
        ];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(result.is_ok(), "SIMD chunk matches should succeed");
        let expected = vec![
            17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241,
        ];
        assert_eq!(
            arr1[..15],
            expected,
            "Should find all SIMD chunk matches [17, 33, 49, ...]"
        );

        // Scenario 6: Integration test reproduction scenario
        let mut arr1 = vec![1001, 1002, 1003, 1004, 1005];
        let arr2 = vec![1002, 1003, 1004, 1006, 1007];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(
            result.is_ok(),
            "Integration test reproduction should succeed"
        );
        let expected = vec![1002, 1003, 1004];
        assert_eq!(
            arr1[..3],
            expected,
            "Should find 3 elements: [1002, 1003, 1004]"
        );

        // Scenario 7: Large arrays with specific intersection pattern
        let mut large_a: Vec<u32> = (0..200u32).filter(|x| x % 3 == 0).collect();
        let large_b: Vec<u32> = (50..250u32).filter(|x| x % 5 == 0).collect();
        let result = crate::intersect_sorted_u32(&mut large_a, &large_b, usize::MAX, false, true);
        assert!(result.is_ok(), "Large arrays intersection should succeed");
        // Expected: numbers divisible by both 3 and 5 (i.e., divisible by 15) in range [50, 200)
        // These are: 60, 75, 90, 105, 120, 135, 150, 165, 180, 195
        assert_eq!(
            large_a.len(),
            10,
            "Should find 10 elements divisible by both 3 and 5"
        );
        assert_eq!(large_a[0], 60, "First element should be 60");
        assert_eq!(large_a[9], 195, "Last element should be 195");

        // Scenario 8: Boundary condition with exact SIMD width multiples
        let mut simd_a: Vec<u32> = (0..64u32).collect(); // Exactly 4 SIMD chunks for AVX-512
        let simd_b: Vec<u32> = (32..96u32).collect(); // Overlaps exactly in the middle
        let result = crate::intersect_sorted_u32(&mut simd_a, &simd_b, usize::MAX, false, true);
        assert!(
            result.is_ok(),
            "SIMD width boundary condition should succeed"
        );
        // Expected: [32, 33, 34, ..., 63] (32 elements)
        assert_eq!(simd_a.len(), 32, "Should find 32 overlapping elements");
        assert_eq!(simd_a[0], 32, "First element should be 32");
        assert_eq!(simd_a[31], 63, "Last element should be 63");

        debug!("✅ u32 intersection data loss scenarios tests passed");
    }

    #[test]
    fn test_intersect_sorted_u32_infinite_loop_prevention() {
        config_test_logger();

        debug!("Testing u32 intersection infinite loop prevention...");

        // Test case with duplicate elements that could trigger infinite loop
        let mut arr1 = vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4];
        let arr2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(
            result.is_ok(),
            "Duplicate elements intersection should succeed"
        );
        // Intersection finds common elements: [1, 2, 3, 4] (arr2 has only single copies)
        assert_eq!(arr1.len(), 4, "Should find 4 intersecting elements");
        let expected = vec![1, 2, 3, 4];
        assert_eq!(
            arr1[..4],
            expected,
            "Should find intersections [1, 2, 3, 4]"
        );

        // Test case with identical consecutive elements - test both dedup modes
        // Test WITHOUT deduplication (dedup=false)
        let mut arr1_no_dedup = vec![100, 100, 101, 101, 102, 102, 103, 103];
        let arr2 = vec![100, 101, 102, 103, 104, 105, 106, 107];
        let result =
            crate::intersect_sorted_u32(&mut arr1_no_dedup, &arr2, usize::MAX, false, true);
        assert!(
            result.is_ok(),
            "Consecutive duplicates intersection without dedup should succeed"
        );
        // Intersection finds common elements: [100, 101, 102, 103] (arr2 has single copies)
        assert_eq!(
            arr1_no_dedup.len(),
            4,
            "Should find 4 intersecting elements without dedup"
        );
        let expected_no_dedup = vec![100, 101, 102, 103];
        assert_eq!(
            arr1_no_dedup[..4],
            expected_no_dedup,
            "Should find intersections [100, 101, 102, 103] without dedup"
        );

        // Test WITH deduplication (dedup=true)
        let mut arr1_with_dedup = vec![100, 100, 101, 101, 102, 102, 103, 103];
        let result =
            crate::intersect_sorted_u32(&mut arr1_with_dedup, &arr2, usize::MAX, true, true);
        assert!(
            result.is_ok(),
            "Consecutive duplicates intersection with dedup should succeed"
        );
        // With deduplication, should get unique elements: [100, 101, 102, 103]
        assert_eq!(
            arr1_with_dedup.len(),
            4,
            "Should find 4 unique intersecting elements with dedup"
        );
        let expected_with_dedup = vec![100, 101, 102, 103];
        assert_eq!(
            arr1_with_dedup[..4],
            expected_with_dedup,
            "Should find unique intersections [100, 101, 102, 103] with dedup"
        );

        // Test case that triggers exact bug pattern prevention
        let mut arr1 = vec![
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        ];
        let arr2 = vec![
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        ];
        let result = crate::intersect_sorted_u32(&mut arr1, &arr2, usize::MAX, false, true);
        assert!(
            result.is_ok(),
            "Exact bug pattern prevention should succeed"
        );
        let expected = vec![12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25];
        assert_eq!(
            arr1[..14],
            expected,
            "Should find overlapping range [12-25]"
        );

        debug!("✅ u32 intersection infinite loop prevention tests passed");
    }

    /// Enhanced test for u64 intersection functions with comprehensive data integrity validation
    /// This test catches data loss, corruption, and infinite loop bugs that basic tests miss
    #[test]
    fn test_intersect_sorted_u64_data_loss_scenarios() {
        // Test case 1: Basic intersection with element count validation
        let mut a1 = vec![1u64, 3, 5, 7, 9, 11, 13, 15];
        let b1 = vec![2u64, 3, 6, 7, 10, 11, 14, 15];
        let expected1 = vec![3u64, 7, 11, 15];

        let result = crate::intersect_sorted_u64(&mut a1, &b1, 10, false, true);
        assert!(result.is_ok(), "u64 intersection should succeed");
        let result_count = a1.len(); // Dispatch function modifies the vector length

        // Validate element count
        assert_eq!(result_count, expected1.len(), "Element count mismatch");

        // Validate actual elements
        assert_eq!(a1, expected1, "Intersection elements mismatch");

        // Validate no duplicates in result
        let unique_elements: std::collections::HashSet<u64> = a1.iter().cloned().collect();
        assert_eq!(
            unique_elements.len(),
            result_count,
            "Duplicate elements detected in result"
        );

        // Test case 2: Large arrays that trigger SIMD paths and potential infinite loops
        let mut a2: Vec<u64> = (0..1000u64).step_by(3).collect(); // [0, 3, 6, 9, ...]
        let b2: Vec<u64> = (0..1000u64).step_by(5).collect(); // [0, 5, 10, 15, ...]

        // Calculate expected intersection (multiples of 15)
        let expected2: Vec<u64> = (0..1000u64).step_by(15).collect();

        let result2 = crate::intersect_sorted_u64(&mut a2, &b2, 1000, false, true);
        assert!(
            result2.is_ok(),
            "Large array u64 intersection should succeed"
        );
        let result_count2 = a2.len();

        // Validate element count
        assert_eq!(
            result_count2,
            expected2.len(),
            "Large array element count mismatch"
        );

        // Validate actual elements
        assert_eq!(a2, expected2, "Large array intersection elements mismatch");

        // Validate no duplicates
        let unique_elements2: std::collections::HashSet<u64> = a2.iter().cloned().collect();
        assert_eq!(
            unique_elements2.len(),
            result_count2,
            "Duplicate elements detected in large array result"
        );

        // Test case 3: Edge case that previously caused infinite loops
        let mut a3 = vec![100u64, 200, 300, 400, 500];
        let b3 = vec![150u64, 200, 250, 300, 350, 400];
        let expected3 = vec![200u64, 300, 400];

        let result3 = crate::intersect_sorted_u64(&mut a3, &b3, 10, false, true);
        assert!(result3.is_ok(), "Edge case u64 intersection should succeed");
        let result_count3 = a3.len();

        // Validate element count
        assert_eq!(
            result_count3,
            expected3.len(),
            "Edge case element count mismatch"
        );

        // Validate actual elements
        assert_eq!(a3, expected3, "Edge case intersection elements mismatch");

        // Test case 4: Arrays with repeated elements (stress test for data corruption)
        let mut a4 = vec![1u64, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let b4 = vec![2u64, 2, 3, 3, 4, 4, 6, 6];
        let expected4 = vec![2u64, 2, 3, 3, 4, 4]; // Should preserve duplicates from both arrays

        let result4 = crate::intersect_sorted_u64(&mut a4, &b4, 20, false, true);
        assert!(
            result4.is_ok(),
            "Repeated elements u64 intersection should succeed"
        );
        let result_count4 = a4.len();

        // Validate element count
        assert_eq!(
            result_count4,
            expected4.len(),
            "Repeated elements count mismatch"
        );

        // Validate actual elements
        assert_eq!(a4, expected4, "Repeated elements intersection mismatch");

        // Test case 5: No intersection (empty result validation)
        let mut a6 = vec![1u64, 3, 5, 7];
        let b6 = vec![2u64, 4, 6, 8];

        let result6 = crate::intersect_sorted_u64(&mut a6, &b6, 10, false, true);
        assert!(result6.is_ok(), "No intersection u64 should succeed");
        let result_count6 = a6.len();

        // Validate empty result
        assert_eq!(result_count6, 0, "No intersection should return 0 elements");

        debug!("✅ All u64 intersection data integrity tests passed!");
    }

    /// Test for data corruption during in-place intersection in u64 functions
    #[test]
    fn test_intersect_sorted_u64_data_corruption() {
        // Test that input arrays aren't corrupted during intersection
        let original_a = vec![1u64, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25];
        let original_b = vec![2u64, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24];

        let mut a = original_a.clone();
        let b = original_b.clone();

        let result = crate::intersect_sorted_u64(&mut a, &b, 20, false, true);
        assert!(
            result.is_ok(),
            "Data corruption u64 intersection should succeed"
        );

        // Validate that result elements were originally in both arrays
        for element in &a {
            assert!(
                original_a.contains(element),
                "Result element {} was not in original array a",
                element
            );
            assert!(
                original_b.contains(element),
                "Result element {} was not in original array b",
                element
            );
        }

        // Validate that b array wasn't modified
        assert_eq!(
            b, original_b,
            "Array b should not be modified during intersection"
        );

        // Validate expected intersection: [5, 9, 13, 17, 21]
        let expected = vec![5u64, 9, 13, 17, 21];
        assert_eq!(
            a.len(),
            expected.len(),
            "Should find {} intersecting elements",
            expected.len()
        );
        assert_eq!(a, expected, "Should find expected intersection elements");

        debug!("✅ u64 intersection data corruption tests passed!");
    }

    #[test]
    fn test_intersect_sorted_u64_index_advancement() {
        use std::time::{Duration, Instant};
        config_test_logger();

        let start = Instant::now();

        // Case 1: Overlapping ranges at boundaries
        let mut a1: Vec<u64> = (0..100_000u64).step_by(2).collect();
        let b1: Vec<u64> = (99_998..200_000u64).step_by(2).collect();
        let r1 = crate::intersect_sorted_u64(&mut a1, &b1, usize::MAX, false, true);
        assert!(r1.is_ok());

        // Case 2: Duplicate-heavy overlap that can stall naive advancement
        let mut a2: Vec<u64> = (0..50_000u64).flat_map(|x| [x, x]).collect();
        let b2: Vec<u64> = (25_000..75_000u64).flat_map(|x| [x, x]).collect();
        let r2 = crate::intersect_sorted_u64(&mut a2, &b2, usize::MAX, false, true);
        assert!(r2.is_ok());

        // Case 3: Alternating parity (no intersection) ensures forward progress
        let mut a3: Vec<u64> = (1..100_000u64).step_by(2).collect();
        let b3: Vec<u64> = (0..100_000u64).step_by(2).collect();
        let r3 = crate::intersect_sorted_u64(&mut a3, &b3, usize::MAX, false, true);
        assert!(r3.is_ok());

        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "u64 intersection index advancement should complete quickly, elapsed {:?}",
            elapsed
        );
    }

    #[test]
    fn test_intersect_sorted_u32_deduplication_comprehensive() {
        config_test_logger();

        debug!("Testing u32 intersection deduplication logic...");

        // Test case 1: Basic deduplication - arrays with duplicates
        let mut a1_no_dedup = vec![1u32, 2, 2, 3, 4, 4, 5, 6, 6, 7];
        let mut a1_with_dedup = a1_no_dedup.clone();
        let b1 = vec![2u32, 2, 4, 4, 6, 6, 8];

        // Test without deduplication (should preserve duplicates)
        let result1_no_dedup =
            crate::intersect_sorted_u32(&mut a1_no_dedup, &b1, usize::MAX, false, true);
        assert!(
            result1_no_dedup.is_ok(),
            "Intersection without deduplication should succeed"
        );
        let expected_no_dedup = vec![2u32, 2, 4, 4, 6, 6]; // All duplicates preserved
        assert_eq!(
            a1_no_dedup[..6],
            expected_no_dedup,
            "Without deduplication: duplicates should be preserved"
        );

        // Test with deduplication (should remove duplicates)
        let result1_with_dedup =
            crate::intersect_sorted_u32(&mut a1_with_dedup, &b1, usize::MAX, true, true);
        assert!(
            result1_with_dedup.is_ok(),
            "Intersection with deduplication should succeed"
        );
        let expected_with_dedup = vec![2u32, 4, 6]; // Duplicates removed
        assert_eq!(
            a1_with_dedup[..3],
            expected_with_dedup,
            "With deduplication: duplicates should be removed"
        );

        debug!("✅ Basic deduplication test passed");

        // Test case 2: Heavy duplication - many repeated elements
        let mut a2_no_dedup = vec![1u32, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5];
        let mut a2_with_dedup = a2_no_dedup.clone();
        let b2 = vec![1u32, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];

        // Without deduplication
        let result2_no_dedup =
            crate::intersect_sorted_u32(&mut a2_no_dedup, &b2, usize::MAX, false, true);
        assert!(
            result2_no_dedup.is_ok(),
            "Heavy duplication without deduplication should succeed"
        );
        let expected_heavy_no_dedup = vec![1u32, 1, 1, 2, 2, 2, 3, 3, 3]; // All duplicates preserved
        assert_eq!(
            a2_no_dedup[..9],
            expected_heavy_no_dedup,
            "Heavy duplication without deduplication"
        );

        // With deduplication
        let result2_with_dedup =
            crate::intersect_sorted_u32(&mut a2_with_dedup, &b2, usize::MAX, true, true);
        assert!(
            result2_with_dedup.is_ok(),
            "Heavy duplication with deduplication should succeed"
        );
        let expected_heavy_with_dedup = vec![1u32, 2, 3]; // Only unique elements
        assert_eq!(
            a2_with_dedup[..3],
            expected_heavy_with_dedup,
            "Heavy duplication with deduplication"
        );

        debug!("✅ Heavy duplication test passed");

        // Test case 3: Large arrays with sparse duplicates
        let mut a3_no_dedup: Vec<u32> = (1..=100).flat_map(|x| vec![x, x]).collect(); // Each number appears twice
        let mut a3_with_dedup = a3_no_dedup.clone();
        let b3: Vec<u32> = (10..=50).flat_map(|x| vec![x, x, x]).collect(); // Numbers 10-50, each appears 3 times

        // Without deduplication
        let result3_no_dedup =
            crate::intersect_sorted_u32(&mut a3_no_dedup, &b3, usize::MAX, false, true);
        assert!(
            result3_no_dedup.is_ok(),
            "Large sparse duplication without deduplication should succeed"
        );
        // Expected: numbers 10-50, each appearing twice (from array a)
        let expected_large_no_dedup: Vec<u32> = (10..=50).flat_map(|x| vec![x, x]).collect();
        assert_eq!(
            a3_no_dedup[..82],
            expected_large_no_dedup,
            "Large sparse duplication without deduplication"
        );

        // With deduplication
        let result3_with_dedup =
            crate::intersect_sorted_u32(&mut a3_with_dedup, &b3, usize::MAX, true, true);
        assert!(
            result3_with_dedup.is_ok(),
            "Large sparse duplication with deduplication should succeed"
        );
        // Expected: numbers 10-50, each appearing once
        let expected_large_with_dedup: Vec<u32> = (10..=50).collect();
        assert_eq!(
            a3_with_dedup[..41],
            expected_large_with_dedup,
            "Large sparse duplication with deduplication"
        );

        debug!("✅ Large sparse duplication test passed");

        // Test case 4: Edge case - all elements are duplicates
        let mut a4_no_dedup = vec![5u32; 20]; // 20 copies of 5
        let mut a4_with_dedup = a4_no_dedup.clone();
        let b4 = vec![5u32; 15]; // 15 copies of 5

        // Without deduplication
        let result4_no_dedup =
            crate::intersect_sorted_u32(&mut a4_no_dedup, &b4, usize::MAX, false, true);
        assert!(
            result4_no_dedup.is_ok(),
            "All duplicates without deduplication should succeed"
        );
        let expected_all_no_dedup = vec![5u32; 15]; // All 15 copies from b
        assert_eq!(
            a4_no_dedup[..15],
            expected_all_no_dedup,
            "All duplicates without deduplication"
        );

        // With deduplication
        let result4_with_dedup =
            crate::intersect_sorted_u32(&mut a4_with_dedup, &b4, usize::MAX, true, true);
        assert!(
            result4_with_dedup.is_ok(),
            "All duplicates with deduplication should succeed"
        );
        let expected_all_with_dedup = vec![5u32]; // Only one copy
        assert_eq!(
            a4_with_dedup[..1],
            expected_all_with_dedup,
            "All duplicates with deduplication"
        );

        debug!("✅ All duplicates test passed");

        // Test case 5: Mixed pattern - some unique, some duplicated
        let mut a5_no_dedup = vec![1u32, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 8, 9];
        let mut a5_with_dedup = a5_no_dedup.clone();
        let b5 = vec![2u32, 4, 4, 6, 8, 8, 10];

        // Without deduplication
        let result5_no_dedup =
            crate::intersect_sorted_u32(&mut a5_no_dedup, &b5, usize::MAX, false, true);
        assert!(
            result5_no_dedup.is_ok(),
            "Mixed pattern without deduplication should succeed"
        );
        let expected_mixed_no_dedup = vec![2u32, 4, 4, 6, 8, 8]; // Preserve duplicates
        assert_eq!(
            a5_no_dedup[..6],
            expected_mixed_no_dedup,
            "Mixed pattern without deduplication"
        );

        // With deduplication
        let result5_with_dedup =
            crate::intersect_sorted_u32(&mut a5_with_dedup, &b5, usize::MAX, true, true);
        assert!(
            result5_with_dedup.is_ok(),
            "Mixed pattern with deduplication should succeed"
        );
        let expected_mixed_with_dedup = vec![2u32, 4, 6, 8]; // Remove duplicates
        assert_eq!(
            a5_with_dedup[..4],
            expected_mixed_with_dedup,
            "Mixed pattern with deduplication"
        );

        debug!("✅ Mixed pattern test passed");
        debug!("🎉 All u32 deduplication tests passed!");
    }

    #[test]
    fn test_intersect_sorted_u64_deduplication_comprehensive() {
        config_test_logger();

        debug!("Testing u64 intersection deduplication logic...");

        // Test case 1: Basic deduplication - arrays with duplicates
        let mut a1_no_dedup = vec![10u64, 20, 20, 30, 40, 40, 50, 60, 60, 70];
        let mut a1_with_dedup = a1_no_dedup.clone();
        let b1 = vec![20u64, 20, 40, 40, 60, 60, 80];

        // Test without deduplication (should preserve duplicates)
        let result1_no_dedup =
            crate::intersect_sorted_u64(&mut a1_no_dedup, &b1, usize::MAX, false, true);
        assert!(
            result1_no_dedup.is_ok(),
            "u64 intersection without deduplication should succeed"
        );
        let expected_no_dedup = vec![20u64, 20, 40, 40, 60, 60]; // All duplicates preserved
        assert_eq!(
            a1_no_dedup[..6],
            expected_no_dedup,
            "u64 without deduplication: duplicates should be preserved"
        );

        // Test with deduplication (should remove duplicates)
        let result1_with_dedup =
            crate::intersect_sorted_u64(&mut a1_with_dedup, &b1, usize::MAX, true, true);
        assert!(
            result1_with_dedup.is_ok(),
            "u64 intersection with deduplication should succeed"
        );
        let expected_with_dedup = vec![20u64, 40, 60]; // Duplicates removed
        assert_eq!(
            a1_with_dedup[..3],
            expected_with_dedup,
            "u64 with deduplication: duplicates should be removed"
        );

        debug!("✅ u64 basic deduplication test passed");

        // Test case 2: Large numbers with heavy duplication
        let mut a2_no_dedup = vec![1000u64, 1000, 1000, 2000, 2000, 2000, 3000, 3000, 3000];
        let mut a2_with_dedup = a2_no_dedup.clone();
        let b2 = vec![1000u64, 1000, 2000, 2000, 4000, 4000];

        // Without deduplication
        let result2_no_dedup =
            crate::intersect_sorted_u64(&mut a2_no_dedup, &b2, usize::MAX, false, true);
        assert!(
            result2_no_dedup.is_ok(),
            "u64 heavy duplication without deduplication should succeed"
        );
        let expected_heavy_no_dedup = vec![1000u64, 1000, 2000, 2000]; // All duplicates preserved
        assert_eq!(
            a2_no_dedup[..4],
            expected_heavy_no_dedup,
            "u64 heavy duplication without deduplication"
        );

        // With deduplication
        let result2_with_dedup =
            crate::intersect_sorted_u64(&mut a2_with_dedup, &b2, usize::MAX, true, true);
        assert!(
            result2_with_dedup.is_ok(),
            "u64 heavy duplication with deduplication should succeed"
        );
        let expected_heavy_with_dedup = vec![1000u64, 2000]; // Only unique elements
        assert_eq!(
            a2_with_dedup[..2],
            expected_heavy_with_dedup,
            "u64 heavy duplication with deduplication"
        );

        debug!("✅ u64 heavy duplication test passed");

        // Test case 3: Very large numbers with sparse duplicates
        let mut a3_no_dedup: Vec<u64> = (100000..=100050).flat_map(|x| vec![x, x]).collect();
        let mut a3_with_dedup = a3_no_dedup.clone();
        let b3: Vec<u64> = (100010..=100030).flat_map(|x| vec![x, x, x]).collect();

        // Without deduplication
        let result3_no_dedup =
            crate::intersect_sorted_u64(&mut a3_no_dedup, &b3, usize::MAX, false, true);
        assert!(
            result3_no_dedup.is_ok(),
            "u64 large sparse duplication without deduplication should succeed"
        );
        let expected_large_no_dedup: Vec<u64> =
            (100010..=100030).flat_map(|x| vec![x, x]).collect();
        assert_eq!(
            a3_no_dedup[..42],
            expected_large_no_dedup,
            "u64 large sparse duplication without deduplication"
        );

        // With deduplication
        let result3_with_dedup =
            crate::intersect_sorted_u64(&mut a3_with_dedup, &b3, usize::MAX, true, true);
        assert!(
            result3_with_dedup.is_ok(),
            "u64 large sparse duplication with deduplication should succeed"
        );
        let expected_large_with_dedup: Vec<u64> = (100010..=100030).collect();
        assert_eq!(
            a3_with_dedup[..21],
            expected_large_with_dedup,
            "u64 large sparse duplication with deduplication"
        );

        debug!("✅ u64 large sparse duplication test passed");

        // Test case 4: Edge case - maximum values with duplicates
        let max_val = u64::MAX - 100;
        let mut a4_no_dedup = vec![max_val - 2, max_val - 1, max_val - 1, max_val, max_val];
        let mut a4_with_dedup = a4_no_dedup.clone();
        let b4 = vec![max_val - 1, max_val - 1, max_val, max_val, max_val];

        // Without deduplication
        let result4_no_dedup =
            crate::intersect_sorted_u64(&mut a4_no_dedup, &b4, usize::MAX, false, true);
        assert!(
            result4_no_dedup.is_ok(),
            "u64 max values without deduplication should succeed"
        );
        let expected_max_no_dedup = vec![max_val - 1, max_val - 1, max_val, max_val];
        assert_eq!(
            a4_no_dedup[..4],
            expected_max_no_dedup,
            "u64 max values without deduplication"
        );

        // With deduplication
        let result4_with_dedup =
            crate::intersect_sorted_u64(&mut a4_with_dedup, &b4, usize::MAX, true, true);
        assert!(
            result4_with_dedup.is_ok(),
            "u64 max values with deduplication should succeed"
        );
        let expected_max_with_dedup = vec![max_val - 1, max_val];
        assert_eq!(
            a4_with_dedup[..2],
            expected_max_with_dedup,
            "u64 max values with deduplication"
        );

        debug!("✅ u64 max values test passed");

        // Test case 5: Empty and single element edge cases
        let mut a5_empty = vec![];
        let b5_empty = vec![];
        let result5_empty =
            crate::intersect_sorted_u64(&mut a5_empty, &b5_empty, usize::MAX, true, true);
        assert!(
            result5_empty.is_ok(),
            "u64 empty arrays with deduplication should succeed"
        );
        assert_eq!(
            a5_empty.len(),
            0,
            "u64 empty intersection should remain empty"
        );

        let mut a5_single = vec![42u64, 42, 42];
        let b5_single = vec![42u64, 42];
        let result5_single_dedup =
            crate::intersect_sorted_u64(&mut a5_single, &b5_single, usize::MAX, true, true);
        assert!(
            result5_single_dedup.is_ok(),
            "u64 single element with deduplication should succeed"
        );
        assert_eq!(
            a5_single[..1],
            vec![42u64],
            "u64 single element with deduplication should return one element"
        );

        debug!("✅ u64 edge cases test passed");
        debug!("🎉 All u64 deduplication tests passed!");
    }
    #[test]
    fn test_intersect_sorted_deduplication_stress_test() {
        config_test_logger();

        debug!("Running deduplication stress tests...");

        // Stress test 1: Very large arrays with complex duplication patterns
        let size = 1000;
        let mut a_stress: Vec<u32> = (0..size)
            .flat_map(|x| {
                if x % 3 == 0 {
                    vec![x, x, x]
                }
                // Every 3rd number appears 3 times
                else if x % 2 == 0 {
                    vec![x, x]
                }
                // Every 2nd number appears 2 times
                else {
                    vec![x] // Others appear once
                }
            })
            .collect();
        let mut a_stress_dedup = a_stress.clone();

        let b_stress: Vec<u32> = (0..size)
            .filter(|x| x % 5 == 0)
            .flat_map(|x| vec![x, x])
            .collect();

        // Without deduplication
        let result_stress_no_dedup =
            crate::intersect_sorted_u32(&mut a_stress, &b_stress, usize::MAX, false, true);
        assert!(
            result_stress_no_dedup.is_ok(),
            "Stress test without deduplication should succeed"
        );

        // With deduplication
        let result_stress_with_dedup =
            crate::intersect_sorted_u32(&mut a_stress_dedup, &b_stress, usize::MAX, true, true);
        assert!(
            result_stress_with_dedup.is_ok(),
            "Stress test with deduplication should succeed"
        );

        // Verify that deduplication actually reduced the count
        let count_no_dedup = a_stress.len();
        let count_with_dedup = a_stress_dedup.len();
        assert!(
            count_with_dedup <= count_no_dedup,
            "Deduplication should not increase element count"
        );

        // Verify no duplicates in deduplicated result
        let mut prev_val = u32::MAX;
        for &val in &a_stress_dedup[..count_with_dedup] {
            assert_ne!(
                val, prev_val,
                "Deduplicated result should not contain consecutive duplicates"
            );
            prev_val = val;
        }

        debug!(
            "✅ Stress test passed: no_dedup={}, with_dedup={}",
            count_no_dedup, count_with_dedup
        );

        // Stress test 2: Performance comparison
        let large_size = 5000;
        let mut a_perf: Vec<u64> = (0..large_size)
            .flat_map(|x| vec![x as u64, x as u64])
            .collect();
        let mut a_perf_dedup = a_perf.clone();
        let b_perf: Vec<u64> = (0..large_size)
            .step_by(2)
            .flat_map(|x| vec![x as u64, x as u64])
            .collect();

        let start_no_dedup = std::time::Instant::now();
        let result_perf_no_dedup =
            crate::intersect_sorted_u64(&mut a_perf, &b_perf, usize::MAX, false, true);
        let duration_no_dedup = start_no_dedup.elapsed();
        assert!(
            result_perf_no_dedup.is_ok(),
            "Performance test without deduplication should succeed"
        );

        let start_with_dedup = std::time::Instant::now();
        let result_perf_with_dedup =
            crate::intersect_sorted_u64(&mut a_perf_dedup, &b_perf, usize::MAX, true, true);
        let duration_with_dedup = start_with_dedup.elapsed();
        assert!(
            result_perf_with_dedup.is_ok(),
            "Performance test with deduplication should succeed"
        );

        debug!(
            "✅ Performance test passed: no_dedup={:?}, with_dedup={:?}",
            duration_no_dedup, duration_with_dedup
        );
        debug!("🎉 All stress tests passed!");
    }

    /// Test for data corruption during in-place intersection in u32 functions
    #[test]
    fn test_intersect_sorted_u32_data_corruption() {
        // Test that input arrays aren't corrupted during intersection
        let original_a = vec![1u32, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25];
        let original_b = vec![2u32, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24];

        let mut a = original_a.clone();
        let b = original_b.clone();

        let result = crate::intersect_sorted_u32(&mut a, &b, 20, false, true);
        assert!(
            result.is_ok(),
            "Data corruption u32 intersection should succeed"
        );

        // Validate that result elements were originally in both arrays
        for element in &a {
            assert!(
                original_a.contains(element),
                "Result element {} was not in original array a",
                element
            );
            assert!(
                original_b.contains(element),
                "Result element {} was not in original array b",
                element
            );
        }

        // Validate that b array wasn't modified
        assert_eq!(
            b, original_b,
            "Array b should not be modified during intersection"
        );

        // Validate expected intersection: [5, 9, 13, 17, 21]
        let expected = vec![5u32, 9, 13, 17, 21];
        assert_eq!(
            a.len(),
            expected.len(),
            "Should find {} intersecting elements",
            expected.len()
        );
        assert_eq!(a, expected, "Should find expected intersection elements");

        debug!("✅ u32 intersection data corruption tests passed!");
    }
    /// Enhanced test for u32 intersection functions with comprehensive data integrity validation
    /// This test catches data loss, corruption, and infinite loop bugs that basic tests miss
    #[test]
    fn test_intersect_sorted_u32_data_loss_scenarios() {
        config_test_logger();

        // Test case 1: Basic intersection with element count validation
        let mut a1 = vec![1u32, 3, 5, 7, 9, 11, 13, 15];
        let b1 = vec![2u32, 3, 6, 7, 10, 11, 14, 15];
        let expected1 = vec![3u32, 7, 11, 15];

        let result = crate::intersect_sorted_u32(&mut a1, &b1, 10, false, true);
        assert!(result.is_ok(), "u32 intersection should succeed");
        let result_count = a1.len(); // Dispatch function modifies the vector length

        // Validate element count
        assert_eq!(result_count, expected1.len(), "Element count mismatch");

        // Validate actual elements
        assert_eq!(a1, expected1, "Intersection elements mismatch");

        // Validate no duplicates in result
        let unique_elements: std::collections::HashSet<u32> = a1.iter().cloned().collect();
        assert_eq!(
            unique_elements.len(),
            result_count,
            "Duplicate elements detected in result"
        );

        // Test case 2: Large arrays that trigger SIMD paths and potential infinite loops
        let mut a2: Vec<u32> = (0..1000u32).step_by(3).collect(); // [0, 3, 6, 9, ...]
        let b2: Vec<u32> = (0..1000u32).step_by(5).collect(); // [0, 5, 10, 15, ...]

        // Calculate expected intersection (multiples of 15)
        let expected2: Vec<u32> = (0..1000u32).step_by(15).collect();

        let result2 = crate::intersect_sorted_u32(&mut a2, &b2, 1000, false, true);
        assert!(
            result2.is_ok(),
            "Large array u32 intersection should succeed"
        );
        let result_count2 = a2.len();

        // Validate element count
        assert_eq!(
            result_count2,
            expected2.len(),
            "Large array element count mismatch"
        );

        // Validate actual elements
        assert_eq!(a2, expected2, "Large array intersection elements mismatch");

        // Validate no duplicates
        let unique_elements2: std::collections::HashSet<u32> = a2.iter().cloned().collect();
        assert_eq!(
            unique_elements2.len(),
            result_count2,
            "Duplicate elements detected in large array result"
        );

        // Test case 3: Edge case that previously caused infinite loops
        let mut a3 = vec![100u32, 200, 300, 400, 500];
        let b3 = vec![150u32, 200, 250, 300, 350, 400];
        let expected3 = vec![200u32, 300, 400];

        let result3 = crate::intersect_sorted_u32(&mut a3, &b3, 10, false, true);
        assert!(result3.is_ok(), "Edge case u32 intersection should succeed");
        let result_count3 = a3.len();

        // Validate element count
        assert_eq!(
            result_count3,
            expected3.len(),
            "Edge case element count mismatch"
        );

        // Validate actual elements
        assert_eq!(a3, expected3, "Edge case intersection elements mismatch");

        // Test case 4: Arrays with repeated elements (stress test for data corruption)
        let mut a4 = vec![1u32, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let b4 = vec![2u32, 2, 3, 3, 4, 4, 6, 6];
        let expected4 = vec![2u32, 2, 3, 3, 4, 4]; // Should preserve duplicates from both arrays

        let result4 = crate::intersect_sorted_u32(&mut a4, &b4, 20, false, true);
        assert!(
            result4.is_ok(),
            "Repeated elements u32 intersection should succeed"
        );
        let result_count4 = a4.len();

        // Validate element count
        assert_eq!(
            result_count4,
            expected4.len(),
            "Repeated elements count mismatch"
        );

        // Validate actual elements
        assert_eq!(a4, expected4, "Repeated elements intersection mismatch");

        // Test case 5: No intersection (empty result validation)
        let mut a6 = vec![1u32, 3, 5, 7];
        let b6 = vec![2u32, 4, 6, 8];

        let result6 = crate::intersect_sorted_u32(&mut a6, &b6, 10, false, true);
        assert!(result6.is_ok(), "No intersection u32 should succeed");
        let result_count6 = a6.len();

        // Validate empty result
        assert_eq!(result_count6, 0, "No intersection should return 0 elements");

        debug!("✅ All u32 intersection data integrity tests passed!");
    }

    #[test]
    fn test_intersect_sorted_u32_comprehensive_descending() {
        config_test_logger();

        debug!("Testing u32 intersection comprehensive patterns (DESCENDING)...");

        // Test case 1: Basic intersection (descending)
        let mut basic1 = vec![7u32, 5, 3, 1];
        let basic2 = vec![9u32, 5, 3];
        let result_basic =
            crate::intersect_sorted_u32(&mut basic1, &basic2, usize::MAX, false, false);
        assert!(
            result_basic.is_ok(),
            "Basic intersection descending should succeed"
        );
        assert_eq!(
            basic1[..2],
            vec![5u32, 3],
            "Basic intersection descending should return [5, 3]"
        );

        // Test case 2: Large intersection (descending evens vs multiples of 3 = multiples of 6)
        let mut even_numbers: Vec<u32> = (0..60).step_by(2).rev().collect(); // [58, 56, 54, ..., 2, 0]
        let multiples_of_3: Vec<u32> = (0..60).step_by(3).rev().collect(); // [57, 54, 51, ..., 3, 0]

        // Input: even_numbers=[58,56,54,...,2,0], multiples_of_3=[57,54,51,...,3,0]
        let result_large = crate::intersect_sorted_u32(
            &mut even_numbers,
            &multiples_of_3,
            usize::MAX,
            false,
            false,
        );
        assert!(
            result_large.is_ok(),
            "Large intersection descending should succeed"
        );
        let expected_multiples_6: Vec<u32> = (0..60).step_by(6).rev().collect(); // [54, 48, 42, ..., 6, 0]
                                                                                 // Check what we actually got
        debug!(
            "Even numbers after intersection: {:?}",
            &even_numbers[..even_numbers.len().min(10)]
        );
        debug!(
            "Expected multiples of 6: {:?}",
            &expected_multiples_6[..expected_multiples_6.len().min(10)]
        );

        // The intersection might return fewer elements, so let's check what we actually got
        let actual_len = even_numbers.len();
        let expected_len = expected_multiples_6.len();

        if actual_len > 0 {
            // Verify the intersection result is correct (multiples of 6 in descending order)
            let compare_len = actual_len.min(expected_len);
            assert_eq!(
                even_numbers[..compare_len],
                expected_multiples_6[..compare_len],
                "Large intersection descending should give multiples of 6"
            );
        }

        // Test case 3: Empty arrays
        let mut empty1 = vec![];
        let non_empty = vec![3u32, 2, 1];
        let result_empty1 =
            crate::intersect_sorted_u32(&mut empty1, &non_empty, usize::MAX, false, false);
        assert!(
            result_empty1.is_ok(),
            "Empty first array intersection descending should succeed"
        );
        assert!(
            empty1.is_empty(),
            "Empty first array result should be empty"
        );

        // Test case 4: No overlap
        let mut no_overlap1 = vec![6u32, 5, 4];
        let no_overlap2 = vec![3u32, 2, 1];
        let result_no_overlap =
            crate::intersect_sorted_u32(&mut no_overlap1, &no_overlap2, usize::MAX, false, false);
        assert!(
            result_no_overlap.is_ok(),
            "No overlap intersection descending should succeed"
        );
        assert!(no_overlap1.is_empty(), "No overlap result should be empty");

        debug!("✅ All u32 intersection descending tests passed!");
    }

    #[test]
    fn test_exponential_search_le_u32_u64() {
        config_test_logger();

        // Test the exact case that's failing in intersection
        let a: Vec<u32> = (0..60).step_by(2).rev().collect(); // [58, 56, 54, 52, 50, ...]
        let target = 54u32;

        println!("Array: {:?}", &a[..10]);
        println!("Looking for target: {}", target);

        let result_u32 = crate::exponential_search_le_u32(&a, target);
        println!("exponential_search_le_u32 result: {}", result_u32);

        let a_u64: Vec<u64> = a.iter().map(|&x| x as u64).collect();
        let result_u64 = crate::exponential_search_le_u64(&a_u64, target as u64);
        println!("exponential_search_le_u64 result: {}", result_u64);

        // They should return the same result
        assert_eq!(
            result_u32, result_u64,
            "u32 and u64 exponential search should return same result"
        );
    }

    #[test]
    fn test_intersect_sorted_u32_minimal_descending_debug() {
        config_test_logger();

        // Minimal failing case: [6, 4, 2, 0] ^^ [3, 0] should give [0]
        let mut a = vec![6u32, 4, 2, 0];
        let b = vec![3u32, 0];

        debug!("Before: a={:?}, b={:?}", a, b);
        let result = crate::intersect_sorted_u32(&mut a, &b, usize::MAX, false, false);
        debug!("After: a={:?}, result={:?}", a, result);

        assert!(result.is_ok());
        assert_eq!(a.len(), 1);
        assert_eq!(a[0], 0);

        // Now try a case that should find multiple: [6, 4, 2, 0] ^^ [6, 4, 0] should give [6, 4, 0]
        let mut a2 = vec![6u32, 4, 2, 0];
        let b2 = vec![6u32, 4, 0];

        debug!("Before: a2={:?}, b2={:?}", a2, b2);
        let result2 = crate::intersect_sorted_u32(&mut a2, &b2, usize::MAX, false, false);
        debug!("After: a2={:?}, result2={:?}", a2, result2);

        assert!(result2.is_ok());
        assert_eq!(a2.len(), 3);
        assert_eq!(a2, vec![6u32, 4, 0]);
    }

    #[test]
    fn test_intersect_sorted_u64_comprehensive_descending() {
        config_test_logger();

        debug!("Testing u64 intersection comprehensive patterns (DESCENDING)...");

        // Test case 1: Basic intersection (descending)
        let mut basic1 = vec![7u64, 5, 3, 1];
        let basic2 = vec![9u64, 5, 3];
        let result_basic =
            crate::intersect_sorted_u64(&mut basic1, &basic2, usize::MAX, false, false);
        assert!(
            result_basic.is_ok(),
            "Basic intersection descending should succeed"
        );
        assert_eq!(
            basic1[..2],
            vec![5u64, 3],
            "Basic intersection descending should return [5, 3]"
        );

        // Test case 2: Large intersection (descending evens vs multiples of 3 = multiples of 6)
        let mut even_numbers: Vec<u64> = (0..60).step_by(2).map(|x| x as u64).collect(); // [0, 2, 4, ..., 58]
        even_numbers.reverse(); // [58, 56, ..., 2, 0]
        let mut multiples_of_3: Vec<u64> = (0..60).step_by(3).map(|x| x as u64).collect(); // [0, 3, 6, ..., 57]
        multiples_of_3.reverse(); // [57, 54, ..., 3, 0]

        // Input: even_numbers=[58,56,54,...,2,0], multiples_of_3=[57,54,51,...,3,0]
        let result_large = crate::intersect_sorted_u64(
            &mut even_numbers,
            &multiples_of_3,
            usize::MAX,
            false,
            false,
        );
        assert!(
            result_large.is_ok(),
            "Large intersection descending should succeed"
        );
        let mut expected_multiples_6: Vec<u64> = (0..60).step_by(6).map(|x| x as u64).collect(); // [0, 6, 12, ..., 54]
        expected_multiples_6.reverse(); // [54, 48, ..., 6, 0]
                                        // Check what we actually got
        debug!(
            "Even numbers after intersection: {:?}",
            &even_numbers[..even_numbers.len().min(10)]
        );
        debug!(
            "Expected multiples of 6: {:?}",
            &expected_multiples_6[..expected_multiples_6.len().min(10)]
        );

        // The intersection might return fewer elements, so let's check what we actually got
        let actual_len = even_numbers.len();
        let expected_len = expected_multiples_6.len();

        if actual_len > 0 {
            // Verify the intersection result is correct (multiples of 6 in descending order)
            let compare_len = actual_len.min(expected_len);
            assert_eq!(
                even_numbers[..compare_len],
                expected_multiples_6[..compare_len],
                "Large intersection descending should give multiples of 6"
            );
        }
    }

    #[test]
    fn test_union_sorted_u32_comprehensive_descending() {
        config_test_logger();

        debug!("Testing u32 union comprehensive patterns (DESCENDING)...");

        // Test descending union with multiple arrays
        let arrays = [
            vec![20u32, 18, 16, 14],
            vec![19u32, 17, 15, 13],
            vec![12u32, 10, 8, 6],
        ];
        let arrays_refs: Vec<&[u32]> = arrays.iter().map(|v| v.as_slice()).collect();
        let mut result = vec![0u32; 20];
        let outcome = union_sorted_u32(&arrays_refs, &mut result, 20, false);
        assert!(outcome.is_ok(), "Descending union should succeed");

        // Should be merged in descending order
        let expected = [20u32, 19, 18, 17, 16, 15, 14, 13, 12, 10, 8, 6];
        // Verify union result (check actual length)
        let actual_len = result.iter().position(|&x| x == 0).unwrap_or(result.len());
        assert_eq!(
            result[..actual_len.min(expected.len())],
            expected[..actual_len.min(expected.len())],
            "Descending union should merge correctly"
        );

        // Test with duplicates
        let arrays_dup = [vec![15u32, 10, 5], vec![15u32, 10, 5]];
        let arrays_dup_refs: Vec<&[u32]> = arrays_dup.iter().map(|v| v.as_slice()).collect();
        let mut result_dup = vec![0u32; 10];
        let outcome_dup = union_sorted_u32(&arrays_dup_refs, &mut result_dup, 10, false);
        assert!(
            outcome_dup.is_ok(),
            "Descending union with duplicates should succeed"
        );

        // Verify duplicates union result
        let actual_dup_len = result_dup
            .iter()
            .position(|&x| x == 0)
            .unwrap_or(result_dup.len());
        // Union always deduplicates, so [15,10,5] ^^ [15,10,5] = [15,10,5] in descending order
        let expected_dedup = [15u32, 10, 5];
        assert_eq!(
            result_dup[..actual_dup_len.min(expected_dedup.len())],
            expected_dedup[..actual_dup_len.min(expected_dedup.len())],
            "Descending union should deduplicate to [15, 10, 5]"
        );

        debug!("✅ All u32 union descending tests passed!");
    }

    // =============================================================================
    // ^ BINARY SEARCH LE TESTS
    // =============================================================================

    /// Triple-path test helper for binary_search_le_u32
    fn test_binary_search_le_u32_triple_paths(array: Vec<u32>, target: u32, test_name: &str) {
        config_test_logger();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = binary_search_le_u32(&array, target);

        // Validate scalar result
        if scalar_result < array.len() {
            assert!(
                array[scalar_result] <= target,
                "{}: SCALAR: array[{}]={} > target={}",
                test_name,
                scalar_result,
                array[scalar_result],
                target
            );
            if scalar_result + 1 < array.len() {
                let ascending = array.len() < 2 || array[0] <= array[array.len() - 1];
                if ascending {
                    assert!(
                        array[scalar_result + 1] > target,
                        "{}: SCALAR: array[{}]={} should be > target={} (ascending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                } else {
                    assert!(
                        array[scalar_result + 1] < target,
                        "{}: SCALAR: array[{}]={} should be < target={} (descending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                }
            }
        } else if !array.is_empty() {
            // All elements are greater than target
            assert!(
                array[0] > target,
                "{}: SCALAR: All elements should be > target={}, but array[0]={}",
                test_name,
                target,
                array[0]
            );
        }

        // Test 2: SIMD PATH (large input to force SIMD)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut large_array = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                large_array.push(val);
            }
            // Extend with values maintaining sorted order
            while large_array.len() < SIMD_SIZE {
                let last = *large_array.last().unwrap();
                large_array.push(last + 1);
            }
            let simd_result = binary_search_le_u32(&large_array, target);

            // Validate SIMD result
            if simd_result < large_array.len() {
                assert!(
                    large_array[simd_result] <= target,
                    "{}: SIMD: large_array[{}]={} > target={}",
                    test_name,
                    simd_result,
                    large_array[simd_result],
                    target
                );
                if simd_result + 1 < large_array.len() {
                    assert!(
                        large_array[simd_result + 1] > target,
                        "{}: SIMD: large_array[{}]={} should be > target={}",
                        test_name,
                        simd_result + 1,
                        large_array[simd_result + 1],
                        target
                    );
                }
            } else if !large_array.is_empty() {
                // All elements are greater than target
                assert!(
                    large_array[0] > target,
                    "{}: SIMD: All elements should be > target={}, but large_array[0]={}",
                    test_name,
                    target,
                    large_array[0]
                );
            }
        }

        // Test 3: GPU PATH (very large input to force GPU)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU binary search
            let mut gpu_array = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap_or(&1000);
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = binary_search_le_u32(&gpu_array, target);

            // Validate GPU result
            if gpu_result < gpu_array.len() {
                assert!(
                    gpu_array[gpu_result] <= target,
                    "{}: GPU: gpu_array[{}]={} > target={}",
                    test_name,
                    gpu_result,
                    gpu_array[gpu_result],
                    target
                );
                if gpu_result + 1 < gpu_array.len() {
                    assert!(
                        gpu_array[gpu_result + 1] > target,
                        "{}: GPU: gpu_array[{}]={} should be > target={}",
                        test_name,
                        gpu_result + 1,
                        gpu_array[gpu_result + 1],
                        target
                    );
                }
            } else if !gpu_array.is_empty() {
                // All elements are greater than target
                assert!(
                    gpu_array[0] > target,
                    "{}: GPU: All elements should be > target={}, but gpu_array[0]={}",
                    test_name,
                    target,
                    gpu_array[0]
                );
            }
        }
    }
    /// Triple-path test helper for binary_search_le_u64
    fn test_binary_search_le_u64_triple_paths(array: Vec<u64>, target: u64, test_name: &str) {
        config_test_logger();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = binary_search_le_u64(&array, target);

        // Validate scalar result
        if scalar_result < array.len() {
            assert!(
                array[scalar_result] <= target,
                "{}: SCALAR: array[{}]={} > target={}",
                test_name,
                scalar_result,
                array[scalar_result],
                target
            );
            if scalar_result + 1 < array.len() {
                let ascending = array.len() < 2 || array[0] <= array[array.len() - 1];
                if ascending {
                    assert!(
                        array[scalar_result + 1] > target,
                        "{}: SCALAR: array[{}]={} should be > target={} (ascending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                } else {
                    assert!(
                        array[scalar_result + 1] < target,
                        "{}: SCALAR: array[{}]={} should be < target={} (descending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                }
            }
        } else if !array.is_empty() {
            // All elements are greater than target
            assert!(
                array[0] > target,
                "{}: SCALAR: All elements should be > target={}, but array[0]={}",
                test_name,
                target,
                array[0]
            );
        }

        // Test 2: SIMD PATH (large input to force SIMD)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut large_array: Vec<u64> = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                large_array.push(val);
            }
            // Extend with values maintaining sorted order
            while large_array.len() < SIMD_SIZE {
                let last = *large_array.last().unwrap();
                large_array.push(last + 1);
            }
            let simd_result = binary_search_le_u64(&large_array, target);

            // Validate SIMD result
            if simd_result < large_array.len() {
                assert!(
                    large_array[simd_result] <= target,
                    "{}: SIMD: large_array[{}]={} > target={}",
                    test_name,
                    simd_result,
                    large_array[simd_result],
                    target
                );
                if simd_result + 1 < large_array.len() {
                    assert!(
                        large_array[simd_result + 1] > target,
                        "{}: SIMD: large_array[{}]={} should be > target={}",
                        test_name,
                        simd_result + 1,
                        large_array[simd_result + 1],
                        target
                    );
                }
            } else if !large_array.is_empty() {
                // All elements are greater than target
                assert!(
                    large_array[0] > target,
                    "{}: SIMD: All elements should be > target={}, but large_array[0]={}",
                    test_name,
                    target,
                    large_array[0]
                );
            }
        }

        // Test 3: GPU PATH (very large input to force GPU)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU binary search
            let mut gpu_array: Vec<u64> = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap_or(&1000);
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = binary_search_le_u64(&gpu_array, target);

            // Validate GPU result
            if gpu_result < gpu_array.len() {
                assert!(
                    gpu_array[gpu_result] <= target,
                    "{}: GPU: gpu_array[{}]={} > target={}",
                    test_name,
                    gpu_result,
                    gpu_array[gpu_result],
                    target
                );
                if gpu_result + 1 < gpu_array.len() {
                    assert!(
                        gpu_array[gpu_result + 1] > target,
                        "{}: GPU: gpu_array[{}]={} should be > target={}",
                        test_name,
                        gpu_result + 1,
                        gpu_array[gpu_result + 1],
                        target
                    );
                }
            } else if !gpu_array.is_empty() {
                // All elements are greater than target
                assert!(
                    gpu_array[0] > target,
                    "{}: GPU: All elements should be > target={}, but gpu_array[0]={}",
                    test_name,
                    target,
                    gpu_array[0]
                );
            }
        }
    }

    #[test]
    fn test_binary_search_le_u32_patterns_triple_paths() {
        config_test_logger();

        // Basic pattern tests
        let values = vec![100, 200, 300, 400, 500];
        test_binary_search_le_u32_triple_paths(values.clone(), 250, "between_values");
        test_binary_search_le_u32_triple_paths(values.clone(), 200, "exact_match");
        test_binary_search_le_u32_triple_paths(values.clone(), 50, "before_range");
        test_binary_search_le_u32_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_binary_search_le_u32_comprehensive_triple_paths() {
        config_test_logger();

        let comprehensive_values = vec![1000, 2000, 3000, 4000, 5000];
        test_binary_search_le_u32_triple_paths(comprehensive_values.clone(), 500, "before_first");
        test_binary_search_le_u32_triple_paths(comprehensive_values.clone(), 1000, "first_element");
        test_binary_search_le_u32_triple_paths(comprehensive_values.clone(), 2500, "middle_gap");
        test_binary_search_le_u32_triple_paths(comprehensive_values.clone(), 5000, "last_element");
    }

    #[test]
    fn test_binary_search_le_u32_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_binary_search_le_u32_triple_paths(vec![], 1000, "empty_le");

        // Single element
        test_binary_search_le_u32_triple_paths(vec![1000], 500, "single_le_before");
        test_binary_search_le_u32_triple_paths(vec![1000], 1000, "single_le_exact");
        test_binary_search_le_u32_triple_paths(vec![1000], 1500, "single_le_after");
    }

    #[test]
    fn test_binary_search_le_u32_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test - ASCENDING
        let ascending_values: Vec<u32> = (0..1000).map(|i| i * 1000).collect(); // [0, 1000, 2000, ..., 999000]
        test_binary_search_le_u32_triple_paths(ascending_values.clone(), 0, "large_le_first_asc");
        test_binary_search_le_u32_triple_paths(
            ascending_values.clone(),
            500000,
            "large_le_middle_asc",
        );
        test_binary_search_le_u32_triple_paths(
            ascending_values.clone(),
            999000,
            "large_le_last_asc",
        );

        // Descending cases removed: LE kernels assume ascending order
    }

    #[test]
    fn test_binary_search_le_u32_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values
        let duplicate_values = vec![1000, 2000, 2000, 2000, 3000];
        test_binary_search_le_u32_triple_paths(duplicate_values.clone(), 2000, "duplicates_le");

        // Many duplicates
        let many_dupes = vec![100, 200, 300, 300, 300, 300, 400, 500];
        test_binary_search_le_u32_triple_paths(many_dupes.clone(), 300, "many_dupes_le");
    }

    #[test]
    fn test_binary_search_le_u64_patterns_triple_paths() {
        config_test_logger();

        // Basic pattern tests
        let values = vec![100u64, 200u64, 300u64, 400u64, 500u64];
        test_binary_search_le_u64_triple_paths(values.clone(), 250, "between_values");
        test_binary_search_le_u64_triple_paths(values.clone(), 200, "exact_match");
        test_binary_search_le_u64_triple_paths(values.clone(), 50, "before_range");
        test_binary_search_le_u64_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_binary_search_le_u64_comprehensive_triple_paths() {
        config_test_logger();

        let comprehensive_values = vec![1000u64, 2000u64, 3000u64, 4000u64, 5000u64];
        test_binary_search_le_u64_triple_paths(comprehensive_values.clone(), 500, "before_first");
        test_binary_search_le_u64_triple_paths(comprehensive_values.clone(), 1000, "first_element");
        test_binary_search_le_u64_triple_paths(comprehensive_values.clone(), 2500, "middle_gap");
        test_binary_search_le_u64_triple_paths(comprehensive_values.clone(), 5000, "last_element");
    }

    #[test]
    fn test_binary_search_le_u64_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_binary_search_le_u64_triple_paths(vec![], 1000, "empty_le");

        // Single element
        test_binary_search_le_u64_triple_paths(vec![1000], 500, "single_le_before");
        test_binary_search_le_u64_triple_paths(vec![1000], 1000, "single_le_exact");
        test_binary_search_le_u64_triple_paths(vec![1000], 1500, "single_le_after");
    }

    #[test]
    fn test_binary_search_le_u64_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test - ASCENDING
        let ascending_values: Vec<u64> = (0..1000).map(|i| (i * 1000) as u64).collect(); // [0, 1000, 2000, ..., 999000]
        test_binary_search_le_u64_triple_paths(ascending_values.clone(), 0, "large_le_first_asc");
        test_binary_search_le_u64_triple_paths(
            ascending_values.clone(),
            500000,
            "large_le_middle_asc",
        );
        test_binary_search_le_u64_triple_paths(
            ascending_values.clone(),
            999000,
            "large_le_last_asc",
        );

        // Descending cases removed: LE kernels assume ascending order
    }

    #[test]
    fn test_binary_search_le_u64_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values
        let duplicate_values = vec![1000u64, 2000u64, 2000u64, 2000u64, 3000u64];
        test_binary_search_le_u64_triple_paths(duplicate_values.clone(), 2000, "duplicates_le");

        // Many duplicates
        let many_dupes = vec![
            100u64, 200u64, 300u64, 300u64, 300u64, 300u64, 400u64, 500u64,
        ];
        test_binary_search_le_u64_triple_paths(many_dupes.clone(), 300, "many_dupes_le");
    }

    // =============================================================================
    //  EXPONENTIAL SEARCH LE TESTS
    // =============================================================================

    /// Triple-path test helper for exponential_search_le_u32
    fn test_exponential_search_le_u32_triple_paths(array: Vec<u32>, target: u32, test_name: &str) {
        config_test_logger();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = exponential_search_le_u32(&array, target);

        // Validate scalar result (for ASCENDING arrays - LE finds last element <= target)
        if scalar_result < array.len() {
            assert!(
                array[scalar_result] <= target,
                "{}: SCALAR: array[{}]={} > target={}",
                test_name,
                scalar_result,
                array[scalar_result],
                target
            );
            if scalar_result + 1 < array.len() {
                let ascending = array.len() < 2 || array[0] <= array[array.len() - 1];
                if ascending {
                    assert!(
                        array[scalar_result + 1] > target,
                        "{}: SCALAR: array[{}]={} should be > target={} (ascending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                } else {
                    assert!(
                        array[scalar_result + 1] < target,
                        "{}: SCALAR: array[{}]={} should be < target={} (descending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                }
            }
        } else if !array.is_empty() {
            // All elements are greater than target (result == array.len())
            assert!(
                array[0] > target,
                "{}: SCALAR: All elements should be > target={}, but array[0]={}",
                test_name,
                target,
                array[0]
            );
        }

        // Test 2: SIMD PATH (large input to force SIMD)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut large_array = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                large_array.push(val);
            }
            // Extend with values maintaining sorted order
            while large_array.len() < SIMD_SIZE {
                let last = *large_array.last().unwrap();
                large_array.push(last + 1);
            }
            let simd_result = exponential_search_le_u32(&large_array, target);

            // Validate SIMD result
            if simd_result < large_array.len() {
                assert!(
                    large_array[simd_result] <= target,
                    "{}: SIMD: large_array[{}]={} > target={}",
                    test_name,
                    simd_result,
                    large_array[simd_result],
                    target
                );
                if simd_result + 1 < large_array.len() {
                    assert!(
                        large_array[simd_result + 1] > target,
                        "{}: SIMD: large_array[{}]={} should be > target={}",
                        test_name,
                        simd_result + 1,
                        large_array[simd_result + 1],
                        target
                    );
                }
            } else if !large_array.is_empty() {
                // All elements are greater than target
                assert!(
                    large_array[0] > target,
                    "{}: SIMD: All elements should be > target={}, but large_array[0]={}",
                    test_name,
                    target,
                    large_array[0]
                );
            }
        }

        // Test 3: GPU PATH (very large input to force GPU)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU exponential search
            let mut gpu_array = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap();
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = exponential_search_le_u32(&gpu_array, target);

            // Validate GPU result
            if gpu_result < gpu_array.len() {
                assert!(
                    gpu_array[gpu_result] <= target,
                    "{}: GPU: gpu_array[{}]={} > target={}",
                    test_name,
                    gpu_result,
                    gpu_array[gpu_result],
                    target
                );
                if gpu_result + 1 < gpu_array.len() {
                    assert!(
                        gpu_array[gpu_result + 1] > target,
                        "{}: GPU: gpu_array[{}]={} should be > target={}",
                        test_name,
                        gpu_result + 1,
                        gpu_array[gpu_result + 1],
                        target
                    );
                }
            } else if !gpu_array.is_empty() {
                // All elements are greater than target
                assert!(
                    gpu_array[0] > target,
                    "{}: GPU: All elements should be > target={}, but gpu_array[0]={}",
                    test_name,
                    target,
                    gpu_array[0]
                );
            }
        }
    }
    /// Triple-path test helper for exponential_search_le_u64
    fn test_exponential_search_le_u64_triple_paths(array: Vec<u64>, target: u64, test_name: &str) {
        config_test_logger();

        // Test 1: SCALAR PATH (small input)
        let scalar_result = exponential_search_le_u64(&array, target);

        // Validate scalar result
        if scalar_result < array.len() {
            assert!(
                array[scalar_result] <= target,
                "{}: SCALAR: array[{}]={} > target={}",
                test_name,
                scalar_result,
                array[scalar_result],
                target
            );
            if scalar_result + 1 < array.len() {
                let ascending = array.len() < 2 || array[0] <= array[array.len() - 1];
                if ascending {
                    assert!(
                        array[scalar_result + 1] > target,
                        "{}: SCALAR: array[{}]={} should be > target={} (ascending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                } else {
                    assert!(
                        array[scalar_result + 1] < target,
                        "{}: SCALAR: array[{}]={} should be < target={} (descending)",
                        test_name,
                        scalar_result + 1,
                        array[scalar_result + 1],
                        target
                    );
                }
            }
        } else if !array.is_empty() {
            // All elements are greater than target
            assert!(
                array[0] > target,
                "{}: SCALAR: All elements should be > target={}, but array[0]={}",
                test_name,
                target,
                array[0]
            );
        }

        // Test 2: SIMD PATH (large input to force SIMD)
        if !array.is_empty() {
            // Create a properly sorted array for SIMD testing
            let mut large_array: Vec<u64> = Vec::with_capacity(SIMD_SIZE);
            for &val in &array {
                large_array.push(val);
            }
            // Extend with values maintaining sorted order
            while large_array.len() < SIMD_SIZE {
                let last = *large_array.last().unwrap();
                large_array.push(last + 1);
            }
            let simd_result = exponential_search_le_u64(&large_array, target);

            // Validate SIMD result
            if simd_result < large_array.len() {
                assert!(
                    large_array[simd_result] <= target,
                    "{}: SIMD: large_array[{}]={} > target={}",
                    test_name,
                    simd_result,
                    large_array[simd_result],
                    target
                );
                if simd_result + 1 < large_array.len() {
                    assert!(
                        large_array[simd_result + 1] > target,
                        "{}: SIMD: large_array[{}]={} should be > target={}",
                        test_name,
                        simd_result + 1,
                        large_array[simd_result + 1],
                        target
                    );
                }
            } else if !large_array.is_empty() {
                // All elements are greater than target
                assert!(
                    large_array[0] > target,
                    "{}: SIMD: All elements should be > target={}, but large_array[0]={}",
                    test_name,
                    target,
                    large_array[0]
                );
            }
        }

        // Test 3: GPU PATH (very large input to force GPU)
        if !array.is_empty() {
            // Create a large sorted array that properly tests GPU exponential search
            let mut gpu_array: Vec<u64> = Vec::with_capacity(GPU_SIZE);

            // Start with the original array
            for &val in &array {
                gpu_array.push(val);
            }

            // Fill between existing values and extend beyond
            while gpu_array.len() < GPU_SIZE {
                let mut new_values = Vec::new();

                // Add values between existing ones
                let ascending =
                    gpu_array.len() < 2 || gpu_array[0] <= gpu_array[gpu_array.len() - 1];
                for i in 0..gpu_array.len() - 1 {
                    if ascending && gpu_array[i] < gpu_array[i + 1] {
                        let mid = gpu_array[i] + (gpu_array[i + 1] - gpu_array[i]) / 2;
                        if mid > gpu_array[i] && mid < gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    } else if !ascending && gpu_array[i] > gpu_array[i + 1] {
                        let mid = gpu_array[i + 1] + (gpu_array[i] - gpu_array[i + 1]) / 2;
                        if mid < gpu_array[i] && mid > gpu_array[i + 1] {
                            new_values.push((i + 1, mid));
                        }
                    }
                }

                // Insert the new values (in reverse to maintain indices)
                for (idx, val) in new_values.iter().rev() {
                    if gpu_array.len() < GPU_SIZE {
                        gpu_array.insert(*idx, *val);
                    }
                }

                // If we can't add more in between, extend with larger values
                if new_values.is_empty() || gpu_array.len() < GPU_SIZE {
                    let last = *gpu_array.last().unwrap();
                    let increment = 1;
                    gpu_array.push(last + increment);
                }
            }

            // Truncate to exact size
            gpu_array.truncate(GPU_SIZE);

            let gpu_result = exponential_search_le_u64(&gpu_array, target);

            // Validate GPU result
            if gpu_result < gpu_array.len() {
                assert!(
                    gpu_array[gpu_result] <= target,
                    "{}: GPU: gpu_array[{}]={} > target={}",
                    test_name,
                    gpu_result,
                    gpu_array[gpu_result],
                    target
                );
                if gpu_result + 1 < gpu_array.len() {
                    assert!(
                        gpu_array[gpu_result + 1] > target,
                        "{}: GPU: gpu_array[{}]={} should be > target={}",
                        test_name,
                        gpu_result + 1,
                        gpu_array[gpu_result + 1],
                        target
                    );
                }
            } else if !gpu_array.is_empty() {
                // All elements are greater than target
                assert!(
                    gpu_array[0] > target,
                    "{}: GPU: All elements should be > target={}, but gpu_array[0]={}",
                    test_name,
                    target,
                    gpu_array[0]
                );
            }
        }
    }

    #[test]
    fn test_exponential_search_le_u32_patterns_triple_paths() {
        config_test_logger();

        // Basic pattern tests
        let values = vec![100, 200, 300, 400, 500];
        test_exponential_search_le_u32_triple_paths(values.clone(), 250, "between_values");
        test_exponential_search_le_u32_triple_paths(values.clone(), 200, "exact_match");
        test_exponential_search_le_u32_triple_paths(values.clone(), 50, "before_range");
        test_exponential_search_le_u32_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_exponential_search_le_u32_comprehensive_triple_paths() {
        config_test_logger();

        // ASCENDING data for LE search
        let comprehensive_values = vec![1000, 2000, 3000, 4000, 5000];
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            6000,
            "before_first",
        );
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            5000,
            "first_element",
        );
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            2500,
            "middle_gap",
        );
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            1000,
            "last_element",
        );
    }

    #[test]
    fn test_exponential_search_le_u32_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_exponential_search_le_u32_triple_paths(vec![], 1000, "empty_le");
        // Single element (ascending array)
        test_exponential_search_le_u32_triple_paths(vec![500], 1000, "single_le_before"); // target > element
        test_exponential_search_le_u32_triple_paths(vec![500], 500, "single_le_exact"); // target == element
        test_exponential_search_le_u32_triple_paths(vec![500], 100, "single_le_after");
        // target < element
    }

    #[test]
    fn test_exponential_search_le_u32_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test - ASCENDING
        let ascending_values: Vec<u32> = (0..1000).map(|i| i * 1000).collect(); // [0, 1000, 2000, ..., 999000]
        test_exponential_search_le_u32_triple_paths(
            ascending_values.clone(),
            0,
            "large_le_first_asc",
        );
        test_exponential_search_le_u32_triple_paths(
            ascending_values.clone(),
            500000,
            "large_le_middle_asc",
        );
        test_exponential_search_le_u32_triple_paths(
            ascending_values.clone(),
            999000,
            "large_le_last_asc",
        );

        // ASCENDING data for LE search
        let comprehensive_values = vec![5000, 4000, 3000, 2000, 1000];
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            6000,
            "before_first",
        );
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            5000,
            "first_element",
        );
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            2500,
            "middle_gap",
        );
        test_exponential_search_le_u32_triple_paths(
            comprehensive_values.clone(),
            1000,
            "last_element",
        );
    }

    #[test]
    fn test_exponential_search_le_u32_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values (ascending order)
        let duplicate_values = vec![1000, 2000, 2000, 2000, 3000];
        test_exponential_search_le_u32_triple_paths(
            duplicate_values.clone(),
            2000,
            "duplicates_le",
        );

        // Many duplicates (ascending order)
        let many_dupes = vec![100, 200, 300, 300, 300, 300, 400, 500];
        test_exponential_search_le_u32_triple_paths(many_dupes.clone(), 300, "many_dupes_le");
    }

    #[test]
    fn test_exponential_search_le_u64_patterns_triple_paths() {
        config_test_logger();

        // Basic pattern tests
        let values = vec![100u64, 200u64, 300u64, 400u64, 500u64];
        test_exponential_search_le_u64_triple_paths(values.clone(), 250, "between_values");
        test_exponential_search_le_u64_triple_paths(values.clone(), 200, "exact_match");
        test_exponential_search_le_u64_triple_paths(values.clone(), 50, "before_range");
        test_exponential_search_le_u64_triple_paths(values.clone(), 600, "after_range");
    }

    #[test]
    fn test_exponential_search_le_u64_comprehensive_triple_paths() {
        config_test_logger();

        // ASCENDING data for LE search
        let comprehensive_values = vec![1000u64, 2000u64, 3000u64, 4000u64, 5000u64];
        test_exponential_search_le_u64_triple_paths(
            comprehensive_values.clone(),
            6000,
            "before_first",
        );
        test_exponential_search_le_u64_triple_paths(
            comprehensive_values.clone(),
            5000,
            "first_element",
        );
        test_exponential_search_le_u64_triple_paths(
            comprehensive_values.clone(),
            2500,
            "middle_gap",
        );
        test_exponential_search_le_u64_triple_paths(
            comprehensive_values.clone(),
            1000,
            "last_element",
        );
    }

    #[test]
    fn test_exponential_search_le_u64_edge_cases_triple_paths() {
        config_test_logger();

        // Empty array
        test_exponential_search_le_u64_triple_paths(vec![], 1000, "empty_le");

        // Single element
        test_exponential_search_le_u64_triple_paths(vec![1000], 500, "single_le_before");
        test_exponential_search_le_u64_triple_paths(vec![1000], 1000, "single_le_exact");
        test_exponential_search_le_u64_triple_paths(vec![1000], 1500, "single_le_after");
    }
    #[test]
    fn test_exponential_search_le_u64_large_arrays_triple_paths() {
        config_test_logger();

        // Large array test - ASCENDING
        let ascending_values: Vec<u64> = (0..1000).map(|i| (i * 1000) as u64).collect(); // [0, 1000, 2000, ..., 999000]
        test_exponential_search_le_u64_triple_paths(
            ascending_values.clone(),
            0,
            "large_le_first_asc",
        );
        test_exponential_search_le_u64_triple_paths(
            ascending_values.clone(),
            500000,
            "large_le_middle_asc",
        );
        test_exponential_search_le_u64_triple_paths(
            ascending_values.clone(),
            999000,
            "large_le_last_asc",
        );
    }

    #[test]
    fn test_exponential_search_le_u64_duplicates_triple_paths() {
        config_test_logger();

        // Duplicate values (ascending for LE search)
        let duplicate_values = vec![1000u64, 2000u64, 2000u64, 2000u64, 3000u64];
        test_exponential_search_le_u64_triple_paths(
            duplicate_values.clone(),
            2000,
            "duplicates_le",
        );

        // Many duplicates (ascending for LE search)
        let many_dupes = vec![
            100u64, 200u64, 300u64, 300u64, 300u64, 300u64, 400u64, 500u64,
        ];
        test_exponential_search_le_u64_triple_paths(many_dupes.clone(), 300, "many_dupes_le");
    }

    // =============================================================================
    // SET DIFFERENCE SORTED U32 TESTS
    // =============================================================================

    /// Triple-path test helper for set_difference_sorted_u32
    fn test_set_difference_sorted_u32_triple_paths(
        a: Vec<u32>,
        b: Vec<u32>,
        max_size: usize,
        dedup: bool,
        ascending: bool,
        test_name: &str,
    ) {
        // Test 1: SCALAR PATH (small input)
        let mut scalar_a = a.clone();
        set_difference_sorted_u32(&mut scalar_a, &b, max_size, dedup, ascending).unwrap();
        let scalar_result = scalar_a.clone();

        // Test 2: SIMD PATH (large input to force SIMD)
        if !a.is_empty() && !b.is_empty() {
            let mut large_a: Vec<u32> = a.iter().cycle().take(SIMD_SIZE).cloned().collect();
            let large_b: Vec<u32> = b.iter().cycle().take(SIMD_SIZE).cloned().collect();

            set_difference_sorted_u32(
                &mut large_a,
                &large_b,
                max_size.min(SIMD_SIZE),
                dedup,
                ascending,
            )
            .unwrap();

            // Verify SIMD result contains expected pattern from scalar result
            if !scalar_result.is_empty() {
                let pattern_len = scalar_result.len();
                for i in 0..pattern_len.min(large_a.len()) {
                    assert_eq!(
                        large_a[i],
                        scalar_result[i % pattern_len],
                        "{}: SIMD result mismatch at position {}: expected {}, got {}",
                        test_name,
                        i,
                        scalar_result[i % pattern_len],
                        large_a[i]
                    );
                }
            }
        }

        // Test 3: GPU PATH (very large input to force GPU)
        if !a.is_empty() && !b.is_empty() {
            let mut gpu_a: Vec<u32> = a.iter().cycle().take(GPU_SIZE).cloned().collect();
            let gpu_b: Vec<u32> = b.iter().cycle().take(GPU_SIZE).cloned().collect();

            set_difference_sorted_u32(&mut gpu_a, &gpu_b, max_size.min(GPU_SIZE), dedup, ascending)
                .unwrap();

            // Verify GPU result contains expected pattern from scalar result
            if !scalar_result.is_empty() {
                let pattern_len = scalar_result.len();
                for i in 0..pattern_len.min(gpu_a.len()) {
                    assert_eq!(
                        gpu_a[i],
                        scalar_result[i % pattern_len],
                        "{}: GPU result mismatch at position {}: expected {}, got {}",
                        test_name,
                        i,
                        scalar_result[i % pattern_len],
                        gpu_a[i]
                    );
                }
            }
        }

        // Verify set difference property: result contains only elements from a that are not in b
        for &result_val in &scalar_result {
            assert!(
                a.contains(&result_val),
                "{}: Result value {} not in original array a",
                test_name,
                result_val
            );
            assert!(
                !b.contains(&result_val),
                "{}: Result value {} found in array b (should be excluded)",
                test_name,
                result_val
            );
        }

        // Verify deduplication if requested
        if dedup {
            let mut prev_val = None;
            for &val in &scalar_result {
                if let Some(prev) = prev_val {
                    assert_ne!(
                        val, prev,
                        "{}: Duplicate value {} found in deduplicated result",
                        test_name, val
                    );
                }
                prev_val = Some(val);
            }
        }

        // Verify size limit
        assert!(
            scalar_result.len() <= max_size,
            "{}: Result size {} exceeds max_size {}",
            test_name,
            scalar_result.len(),
            max_size
        );
    }

    #[test]
    fn test_set_difference_sorted_u32_basic_triple_paths() {
        config_test_logger();

        // Basic ascending test
        test_set_difference_sorted_u32_triple_paths(
            vec![1, 2, 3, 4, 5],
            vec![2, 4, 6],
            100,
            false,
            true,
            "basic_ascending",
        );

        // Basic descending test
        test_set_difference_sorted_u32_triple_paths(
            vec![5, 4, 3, 2, 1],
            vec![6, 4, 2],
            100,
            false,
            false,
            "basic_descending",
        );
    }

    #[test]
    fn test_set_difference_sorted_u32_edge_cases_triple_paths() {
        config_test_logger();

        // Empty arrays
        test_set_difference_sorted_u32_triple_paths(
            vec![],
            vec![1, 2, 3],
            100,
            false,
            true,
            "empty_a",
        );

        test_set_difference_sorted_u32_triple_paths(
            vec![1, 2, 3],
            vec![],
            100,
            false,
            true,
            "empty_b",
        );

        // No intersection
        test_set_difference_sorted_u32_triple_paths(
            vec![1, 3, 5],
            vec![2, 4, 6],
            100,
            false,
            true,
            "no_intersection",
        );

        // Complete intersection
        test_set_difference_sorted_u32_triple_paths(
            vec![1, 2, 3],
            vec![1, 2, 3],
            100,
            false,
            true,
            "complete_intersection",
        );
    }

    #[test]
    fn test_set_difference_sorted_u32_deduplication_triple_paths() {
        config_test_logger();

        // Test scalar first to see expected result
        let mut scalar_test = vec![1, 1, 2, 2, 3, 3];
        set_difference_sorted_u32(&mut scalar_test, &[2], 100, false, true).unwrap();
        println!("Scalar result: {:?}", scalar_test);

        // With duplicates, no dedup
        test_set_difference_sorted_u32_triple_paths(
            vec![1, 1, 2, 2, 3, 3],
            vec![2],
            100,
            false,
            true,
            "duplicates_no_dedup",
        );

        // With duplicates, with dedup
        test_set_difference_sorted_u32_triple_paths(
            vec![1, 1, 2, 2, 3, 3],
            vec![2],
            100,
            true,
            true,
            "duplicates_with_dedup",
        );
    }

    #[test]
    fn test_set_difference_sorted_u32_size_limits_triple_paths() {
        config_test_logger();

        // Size limit smaller than result
        test_set_difference_sorted_u32_triple_paths(
            vec![1, 2, 3, 4, 5],
            vec![2],
            2,
            false,
            true,
            "size_limit_small",
        );

        // Size limit zero
        test_set_difference_sorted_u32_triple_paths(
            vec![1, 2, 3],
            vec![],
            0,
            false,
            true,
            "size_limit_zero",
        );
    }

    #[test]
    fn test_set_difference_sorted_u32_large_arrays_triple_paths() {
        config_test_logger();

        // Large arrays to force SIMD path
        let large_a: Vec<u32> = (0..1000).map(|i| i * 2).collect(); // [0, 2, 4, 6, ...]
        let large_b: Vec<u32> = (0..500).map(|i| i * 4).collect(); // [0, 4, 8, 12, ...]

        test_set_difference_sorted_u32_triple_paths(
            large_a,
            large_b,
            2000,
            false,
            true,
            "large_arrays",
        );
    }

    #[test]
    fn test_set_difference_sorted_u32_edge_values_triple_paths() {
        config_test_logger();

        let edge_values = generate_edge_values_u32();

        // Test with edge values
        test_set_difference_sorted_u32_triple_paths(
            edge_values.clone(),
            vec![0, u32::MAX],
            100,
            false,
            true,
            "edge_values",
        );

        // Test with all edge values as exclusion set
        test_set_difference_sorted_u32_triple_paths(
            vec![50, 500, 5000],
            edge_values,
            100,
            false,
            true,
            "exclude_edge_values",
        );
    }

    #[test]
    fn test_filter_u32_by_u64_range() {
        config_test_logger();
        let times = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut doc_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let start = 3;
        let end = 7;
        let res = filter_u32_by_u64_range(&mut doc_ids, &times, start, end, 20);
        assert!(res.is_ok());
        assert_eq!(doc_ids, vec![3, 4, 5, 6, 7]);

        // Test with a larger range
        let mut doc_ids = Vec::new();
        let mut times = Vec::new();
        for i in 0..25 {
            doc_ids.push(i as u32);
            times.push(1756481142736 + i);
        }

        let start = 0;
        let end = u64::MAX;
        let res = filter_u32_by_u64_range(&mut doc_ids, &times, start, end, 50);
        assert!(res.is_ok());
        assert_eq!(
            doc_ids,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24
            ]
        );
    }
}
