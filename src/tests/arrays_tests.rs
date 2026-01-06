// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
  use crate::dispatch::{
    avg_f64, calculate_multi_percentiles_f64, calculate_percentiles_batch_f64, clamp_f64,
    clamp_max_f64, clamp_min_f64, dedup_sorted_u32, deg_f64, filter_counts_ge_threshold_u64,
    filter_range_f64, filter_range_u32, filter_range_u64, find_min_max_f64, find_min_max_i64,
    find_min_max_u32, fma_f64, is_sorted_u32, linear_interpolate_f64, log2_f64, log10_f64, mad_f64,
    present_f64, rad_f64, reduce_max_f64, reduce_max_u32, reduce_min_f64, reduce_min_u32,
    reduce_sum_f64, reduce_sum_u32, reduce_sum_u64, reduce_weighted_sum_f64, sgn_f64, sort_f64,
    sort_i64, sort_u32, sort_u32_by_f64, sort_u32_by_i64, sort_u32_by_u64, stdvar_f64,
    union_sorted_u32, usize_to_u32, vectorized_quantize_f64, vectorized_subtract_u32,
  };

  // =============================================================================
  //   TRIPLE-PATH TEST HELPERS - ARRAYS EDITION 
  // =============================================================================

  /// Test helper for reduce_sum_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_sum_f64_triple_paths(values: Vec<f64>, expected_sum: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let values_copy = values.clone();
    let scalar_result = reduce_sum_f64(&values_copy).unwrap();

    // Handle special cases: infinity and NaN
    if expected_sum.is_infinite() {
      assert_eq!(
        scalar_result, expected_sum,
        "SCALAR path failed - expected {}, got {}",
        expected_sum, scalar_result
      );
    } else {
      assert!(
        (scalar_result - expected_sum).abs() < 1e-10,
        "SCALAR path failed - expected {}, got {}",
        expected_sum,
        scalar_result
      );
    }

    //  SIMD PATH TEST (large array - above threshold)
    let large_batch: Vec<f64> = values
      .iter()
      .cycle()
      .take(1000) // Ensure we're above SIMD threshold
      .cloned()
      .collect();

    // Calculate expected result correctly by counting cycles and remainders
    let cycles = 1000 / values.len().max(1);
    let remainder = 1000 % values.len().max(1);
    let expected_large_sum = expected_sum * cycles as f64
      + if remainder > 0 {
        // Add partial cycle contribution
        let partial_sum: f64 = values[..remainder].iter().sum();
        partial_sum
      } else {
        0.0
      };

    let simd_result = reduce_sum_f64(&large_batch).unwrap();

    // Handle special cases: infinity
    if expected_large_sum.is_infinite() {
      assert_eq!(
        simd_result, expected_large_sum,
        "SIMD path failed - expected {}, got {}",
        expected_large_sum, simd_result
      );
    } else {
      assert!(
        (simd_result - expected_large_sum).abs() < 1e-6, // Allow for floating point error accumulation
        "SIMD path failed - expected {}, got {}",
        expected_large_sum,
        simd_result
      );
    }

    // ^ GPU PATH TEST (massive array - if GPU available)

    {
      let gpu_batch: Vec<f64> = values
        .iter()
        .cycle()
        .take(100000) // Ensure we're above GPU threshold
        .cloned()
        .collect();

      // Calculate expected result for GPU path
      let gpu_cycles = 100000 / values.len().max(1);
      let gpu_remainder = 100000 % values.len().max(1);
      let expected_gpu_sum = expected_sum * gpu_cycles as f64
        + if gpu_remainder > 0 {
          let partial_sum: f64 = values[..gpu_remainder].iter().sum();
          partial_sum
        } else {
          0.0
        };

      let gpu_result = reduce_sum_f64(&gpu_batch).unwrap();

      // Handle special cases: infinity
      if expected_gpu_sum.is_infinite() {
        assert_eq!(
          gpu_result, expected_gpu_sum,
          "GPU path failed - expected {}, got {}",
          expected_gpu_sum, gpu_result
        );
      } else {
        assert!(
          (gpu_result - expected_gpu_sum).abs() < 1e-4, // Allow more error for GPU
          "GPU path failed - expected {}, got {}",
          expected_gpu_sum,
          gpu_result
        );
      }
    }
  }

  // =============================================================================
  // ARRAY REDUCTION TESTS - REDUCE_SUM  // =============================================================================

  #[test]
  fn test_reduce_sum_f64_basic() {
    test_reduce_sum_f64_triple_paths(vec![1.0, 2.0, 3.0, 4.0, 5.0], 15.0);
  }

  #[test]
  fn test_reduce_sum_f64_empty() {
    test_reduce_sum_f64_triple_paths(vec![], 0.0);
  }

  #[test]
  fn test_reduce_sum_f64_single() {
    test_reduce_sum_f64_triple_paths(vec![42.0], 42.0);
  }

  #[test]
  fn test_reduce_sum_f64_negative() {
    test_reduce_sum_f64_triple_paths(vec![-1.0, -2.0, -3.0, -4.0, -5.0], -15.0);
  }

  #[test]
  fn test_reduce_sum_f64_mixed() {
    test_reduce_sum_f64_triple_paths(vec![-5.0, -2.5, 0.0, 2.5, 5.0], 0.0);
  }

  #[test]
  fn test_reduce_sum_f64_precision() {
    test_reduce_sum_f64_triple_paths(vec![1e-10, 2e-10, 3e-10], 6e-10);
  }

  #[test]
  fn test_reduce_sum_f64_large_values() {
    test_reduce_sum_f64_triple_paths(vec![1e10, 2e10, 3e10], 6e10);
  }

  #[test]
  fn test_reduce_sum_f64_special_values() {
    // Test with infinity - all paths should handle infinity
    test_reduce_sum_f64_triple_paths(vec![1.0, 2.0, f64::INFINITY, 3.0], f64::INFINITY);

    // Test with NaN - can't use triple paths as NaN != NaN
    let with_nan = vec![1.0, 2.0, f64::NAN, 3.0];
    assert!(reduce_sum_f64(&with_nan).unwrap().is_nan());

    // Test with negative infinity - all paths should handle neg infinity
    test_reduce_sum_f64_triple_paths(vec![1.0, 2.0, f64::NEG_INFINITY, 3.0], f64::NEG_INFINITY);
  }

  // =============================================================================
  //  TEST HELPER FOR REDUCE_SUM_U64 - TRIPLE PATH TESTING 
  // =============================================================================

  /// Test helper for reduce_sum_u64 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_sum_u64_triple_paths(values: Vec<u64>, expected_sum: u64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = reduce_sum_u64(&values).unwrap();
    assert_eq!(
      scalar_result, expected_sum,
      "SCALAR path failed - expected {}, got {}",
      expected_sum, scalar_result
    );

    //  SIMD PATH TEST (large array - above threshold)
    if values.is_empty() {
      return;
    }
    let large_batch: Vec<u64> = values
      .iter()
      .cycle()
      .take(1000) // Ensure we're above SIMD threshold
      .cloned()
      .collect();

    // Calculate expected result with wrapping arithmetic (like SIMD does)
    let cycles = 1000 / values.len().max(1);
    let remainder = 1000 % values.len().max(1);
    let expected_large_sum =
      expected_sum
        .wrapping_mul(cycles as u64)
        .wrapping_add(if remainder > 0 {
          values[..remainder]
            .iter()
            .fold(0u64, |acc, &x| acc.wrapping_add(x))
        } else {
          0
        });

    let simd_result = reduce_sum_u64(&large_batch).unwrap();
    assert_eq!(
      simd_result, expected_large_sum,
      "SIMD path failed - expected {}, got {}",
      expected_large_sum, simd_result
    );

    // ^ GPU PATH TEST (massive array - if GPU available)

    {
      let gpu_batch: Vec<u64> = values
        .iter()
        .cycle()
        .take(100000) // Ensure we're above GPU threshold
        .cloned()
        .collect();

      // Calculate expected result for GPU path with wrapping arithmetic
      let gpu_cycles = 100000 / values.len().max(1);
      let gpu_remainder = 100000 % values.len().max(1);
      let expected_gpu_sum =
        expected_sum
          .wrapping_mul(gpu_cycles as u64)
          .wrapping_add(if gpu_remainder > 0 {
            values[..gpu_remainder]
              .iter()
              .fold(0u64, |acc, &x| acc.wrapping_add(x))
          } else {
            0
          });

      let gpu_result = reduce_sum_u64(&gpu_batch).unwrap();
      assert_eq!(
        gpu_result, expected_gpu_sum,
        "GPU path failed - expected {}, got {}",
        expected_gpu_sum, gpu_result
      );
    }
  }

  // =============================================================================
  // ARRAY REDUCTION TESTS - REDUCE_SUM_U64  // =============================================================================

  #[test]
  fn test_reduce_sum_u64_basic() {
    test_reduce_sum_u64_triple_paths(vec![1, 2, 3, 4, 5], 15);
  }

  #[test]
  fn test_reduce_sum_u64_empty() {
    test_reduce_sum_u64_triple_paths(vec![], 0);
  }

  #[test]
  fn test_reduce_sum_u64_single() {
    test_reduce_sum_u64_triple_paths(vec![42], 42);
  }

  #[test]
  fn test_reduce_sum_u64_large_values() {
    test_reduce_sum_u64_triple_paths(vec![1000000, 2000000, 3000000], 6000000);
  }

  #[test]
  fn test_reduce_sum_u64_max_values() {
    let values = vec![u64::MAX / 4, u64::MAX / 4, u64::MAX / 4];
    let expected = (u64::MAX / 4).saturating_mul(3);
    test_reduce_sum_u64_triple_paths(values, expected);
  }

  #[test]
  fn test_reduce_sum_u64_overflow_protection() {
    // Test that all paths handle overflow by wrapping (u64::MAX + 1 = 0)
    test_reduce_sum_u64_triple_paths(vec![u64::MAX, 1], 0);
  }

  // =============================================================================
  //  TEST HELPER FOR REDUCE_SUM_U32 - TRIPLE PATH TESTING 
  // =============================================================================

  /// Test helper for reduce_sum_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_sum_u32_triple_paths(values: Vec<u32>, expected_sum: u64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = reduce_sum_u32(&values).unwrap();
    assert_eq!(
      scalar_result, expected_sum,
      "SCALAR path failed - expected {}, got {}",
      expected_sum, scalar_result
    );

    //  SIMD PATH TEST (large array - above threshold)
    let large_batch: Vec<u32> = values
      .iter()
      .cycle()
      .take(1000) // Ensure we're above SIMD threshold
      .cloned()
      .collect();

    // Calculate expected result correctly by counting cycles and remainders
    let cycles = 1000 / values.len().max(1);
    let remainder = 1000 % values.len().max(1);
    let values_sum: u64 = values.iter().map(|&x| x as u64).sum();
    let expected_large_sum = values_sum * cycles as u64
      + if remainder > 0 {
        values[..remainder].iter().map(|&x| x as u64).sum()
      } else {
        0
      };

    let simd_result = reduce_sum_u32(&large_batch).unwrap();
    assert_eq!(
      simd_result, expected_large_sum,
      "SIMD path failed - expected {}, got {}",
      expected_large_sum, simd_result
    );

    // ^ GPU PATH TEST (massive array - if GPU available)

    {
      let gpu_batch: Vec<u32> = values
        .iter()
        .cycle()
        .take(100000) // Ensure we're above GPU threshold
        .cloned()
        .collect();

      // Calculate expected result for GPU path
      let gpu_cycles = 100000 / values.len().max(1);
      let gpu_remainder = 100000 % values.len().max(1);
      let expected_gpu_sum = values_sum * gpu_cycles as u64
        + if gpu_remainder > 0 {
          values[..gpu_remainder].iter().map(|&x| x as u64).sum()
        } else {
          0
        };

      let gpu_result = reduce_sum_u32(&gpu_batch).unwrap();
      assert_eq!(
        gpu_result, expected_gpu_sum,
        "GPU path failed - expected {}, got {}",
        expected_gpu_sum, gpu_result
      );
    }
  }

  // =============================================================================
  // ARRAY REDUCTION TESTS - REDUCE_SUM_U32  // =============================================================================

  #[test]
  fn test_reduce_sum_u32_basic() {
    test_reduce_sum_u32_triple_paths(vec![1, 2, 3, 4, 5], 15);
  }

  #[test]
  fn test_reduce_sum_u32_empty() {
    test_reduce_sum_u32_triple_paths(vec![], 0);
  }

  #[test]
  fn test_reduce_sum_u32_single() {
    test_reduce_sum_u32_triple_paths(vec![42], 42);
  }

  #[test]
  fn test_reduce_sum_u32_large_values() {
    test_reduce_sum_u32_triple_paths(vec![1000000, 2000000, 3000000], 6000000);
  }

  #[test]
  fn test_reduce_sum_u32_max_values() {
    // Test that we can sum beyond u32::MAX
    let values = vec![u32::MAX, u32::MAX];
    let expected = (u32::MAX as u64) * 2;
    test_reduce_sum_u32_triple_paths(values, expected);
  }

  // =============================================================================
  //  TEST HELPER FOR REDUCE_MIN/MAX - TRIPLE PATH TESTING 
  // =============================================================================

  /// Test helper for reduce_min_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_min_f64_triple_paths(values: Vec<f64>, expected_min: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = reduce_min_f64(&values).unwrap();
    assert!(
      (scalar_result - expected_min).abs() < 1e-10
        || (scalar_result.is_nan() && expected_min.is_nan())
        || (scalar_result.is_infinite()
          && expected_min.is_infinite()
          && scalar_result.signum() == expected_min.signum()),
      "SCALAR path failed - expected {}, got {}",
      expected_min,
      scalar_result
    );

    //  SIMD PATH TEST (large array - above threshold)
    if !values.is_empty() {
      let large_batch: Vec<f64> = values
        .iter()
        .cycle()
        .take(1000) // Ensure we're above SIMD threshold
        .cloned()
        .collect();

      let simd_result = reduce_min_f64(&large_batch).unwrap();
      assert!(
        (simd_result - expected_min).abs() < 1e-10
          || (simd_result.is_nan() && expected_min.is_nan())
          || (simd_result.is_infinite()
            && expected_min.is_infinite()
            && simd_result.signum() == expected_min.signum()),
        "SIMD path failed - expected {}, got {}",
        expected_min,
        simd_result
      );
    }

    // ^ GPU PATH TEST (massive array - if GPU available)
    if !values.is_empty() {
      let gpu_batch: Vec<f64> = values
        .iter()
        .cycle()
        .take(100000) // Ensure we're above GPU threshold
        .cloned()
        .collect();

      let gpu_result = reduce_min_f64(&gpu_batch).unwrap();
      assert!(
        (gpu_result - expected_min).abs() < 1e-10
          || (gpu_result.is_nan() && expected_min.is_nan())
          || (gpu_result.is_infinite()
            && expected_min.is_infinite()
            && gpu_result.signum() == expected_min.signum()),
        "GPU path failed - expected {}, got {}",
        expected_min,
        gpu_result
      );
    }
  }

  /// Test helper for reduce_max_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_max_f64_triple_paths(values: Vec<f64>, expected_max: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = reduce_max_f64(&values).unwrap();
    assert!(
      (scalar_result - expected_max).abs() < 1e-10
        || (scalar_result.is_nan() && expected_max.is_nan())
        || (scalar_result.is_infinite()
          && expected_max.is_infinite()
          && scalar_result.signum() == expected_max.signum()),
      "SCALAR path failed - expected {}, got {}",
      expected_max,
      scalar_result
    );

    //  SIMD PATH TEST (large array - above threshold)
    if !values.is_empty() {
      let large_batch: Vec<f64> = values
        .iter()
        .cycle()
        .take(1000) // Ensure we're above SIMD threshold
        .cloned()
        .collect();

      let simd_result = reduce_max_f64(&large_batch).unwrap();
      assert!(
        (simd_result - expected_max).abs() < 1e-10
          || (simd_result.is_nan() && expected_max.is_nan())
          || (simd_result.is_infinite()
            && expected_max.is_infinite()
            && simd_result.signum() == expected_max.signum()),
        "SIMD path failed - expected {}, got {}",
        expected_max,
        simd_result
      );
    }

    // ^ GPU PATH TEST (massive array - if GPU available)
    if !values.is_empty() {
      let gpu_batch: Vec<f64> = values
        .iter()
        .cycle()
        .take(100000) // Ensure we're above GPU threshold
        .cloned()
        .collect();

      let gpu_result = reduce_max_f64(&gpu_batch).unwrap();
      assert!(
        (gpu_result - expected_max).abs() < 1e-10
          || (gpu_result.is_nan() && expected_max.is_nan())
          || (gpu_result.is_infinite()
            && expected_max.is_infinite()
            && gpu_result.signum() == expected_max.signum()),
        "GPU path failed - expected {}, got {}",
        expected_max,
        gpu_result
      );
    }
  }

  // =============================================================================
  // ARRAY REDUCTION TESTS - REDUCE_MIN/MAX_F64  // =============================================================================

  #[test]
  fn test_reduce_min_f64_basic() {
    test_reduce_min_f64_triple_paths(vec![5.0, 2.0, 8.0, 1.0, 9.0], 1.0);
  }

  #[test]
  fn test_reduce_min_f64_empty() {
    test_reduce_min_f64_triple_paths(vec![], f64::INFINITY);
  }

  #[test]
  fn test_reduce_min_f64_single() {
    test_reduce_min_f64_triple_paths(vec![42.0], 42.0);
  }

  #[test]
  fn test_reduce_min_f64_negative() {
    test_reduce_min_f64_triple_paths(vec![-5.0, -2.0, -8.0, -1.0, -9.0], -9.0);
  }

  #[test]
  fn test_reduce_min_f64_with_nan() {
    test_reduce_min_f64_triple_paths(vec![5.0, f64::NAN, 2.0, 8.0], f64::NAN);
  }

  #[test]
  fn test_reduce_min_f64_with_infinity() {
    test_reduce_min_f64_triple_paths(
      vec![5.0, f64::INFINITY, 2.0, f64::NEG_INFINITY],
      f64::NEG_INFINITY,
    );
  }

  #[test]
  fn test_reduce_max_f64_basic() {
    test_reduce_max_f64_triple_paths(vec![5.0, 2.0, 8.0, 1.0, 9.0], 9.0);
  }

  #[test]
  fn test_reduce_max_f64_empty() {
    test_reduce_max_f64_triple_paths(vec![], f64::NEG_INFINITY);
  }

  #[test]
  fn test_reduce_max_f64_single() {
    test_reduce_max_f64_triple_paths(vec![42.0], 42.0);
  }

  #[test]
  fn test_reduce_max_f64_negative() {
    test_reduce_max_f64_triple_paths(vec![-5.0, -2.0, -8.0, -1.0, -9.0], -1.0);
  }

  #[test]
  fn test_reduce_max_f64_with_nan() {
    test_reduce_max_f64_triple_paths(vec![5.0, f64::NAN, 2.0, 8.0], f64::NAN);
  }

  #[test]
  fn test_reduce_max_f64_with_infinity() {
    test_reduce_max_f64_triple_paths(
      vec![5.0, f64::INFINITY, 2.0, f64::NEG_INFINITY],
      f64::INFINITY,
    );
  }

  // =============================================================================
  //  TEST HELPER FOR REDUCE_MIN/MAX_U32 - TRIPLE PATH TESTING 
  // =============================================================================

  /// Test helper for reduce_min_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_min_u32_triple_paths(values: Vec<u32>, expected_min: u32) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = reduce_min_u32(&values).unwrap();
    assert_eq!(
      scalar_result, expected_min,
      "SCALAR path failed - expected {}, got {}",
      expected_min, scalar_result
    );

    //  SIMD PATH TEST (large array - above threshold)
    let large_batch: Vec<u32> = values
      .iter()
      .cycle()
      .take(1000) // Ensure we're above SIMD threshold
      .cloned()
      .collect();

    let simd_result = reduce_min_u32(&large_batch).unwrap();
    assert_eq!(
      simd_result, expected_min,
      "SIMD path failed - expected {}, got {}",
      expected_min, simd_result
    );

    // ^ GPU PATH TEST (massive array - if GPU available)

    {
      let gpu_batch: Vec<u32> = values
        .iter()
        .cycle()
        .take(100000) // Ensure we're above GPU threshold
        .cloned()
        .collect();

      let gpu_result = reduce_min_u32(&gpu_batch).unwrap();
      assert_eq!(
        gpu_result, expected_min,
        "GPU path failed - expected {}, got {}",
        expected_min, gpu_result
      );
    }
  }

  /// Test helper for reduce_max_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_max_u32_triple_paths(values: Vec<u32>, expected_max: u32) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = reduce_max_u32(&values).unwrap();
    assert_eq!(
      scalar_result, expected_max,
      "SCALAR path failed - expected {}, got {}",
      expected_max, scalar_result
    );

    //  SIMD PATH TEST (large array - above threshold)
    let large_batch: Vec<u32> = values
      .iter()
      .cycle()
      .take(1000) // Ensure we're above SIMD threshold
      .cloned()
      .collect();

    let simd_result = reduce_max_u32(&large_batch).unwrap();
    assert_eq!(
      simd_result, expected_max,
      "SIMD path failed - expected {}, got {}",
      expected_max, simd_result
    );

    // ^ GPU PATH TEST (massive array - if GPU available)

    {
      let gpu_batch: Vec<u32> = values
        .iter()
        .cycle()
        .take(100000) // Ensure we're above GPU threshold
        .cloned()
        .collect();

      let gpu_result = reduce_max_u32(&gpu_batch).unwrap();
      assert_eq!(
        gpu_result, expected_max,
        "GPU path failed - expected {}, got {}",
        expected_max, gpu_result
      );
    }
  }

  // =============================================================================
  // ARRAY REDUCTION TESTS - REDUCE_MIN/MAX_U32  // =============================================================================

  #[test]
  fn test_reduce_min_u32_basic() {
    test_reduce_min_u32_triple_paths(vec![5, 2, 8, 1, 9], 1);
  }

  #[test]
  fn test_reduce_min_u32_empty() {
    test_reduce_min_u32_triple_paths(vec![], u32::MAX);
  }

  #[test]
  fn test_reduce_min_u32_single() {
    test_reduce_min_u32_triple_paths(vec![42], 42);
  }

  #[test]
  fn test_reduce_min_u32_all_same() {
    test_reduce_min_u32_triple_paths(vec![7, 7, 7, 7], 7);
  }

  #[test]
  fn test_reduce_min_u32_with_zero() {
    test_reduce_min_u32_triple_paths(vec![5, 2, 0, 8, 1], 0);
  }

  #[test]
  fn test_reduce_min_u32_with_max() {
    test_reduce_min_u32_triple_paths(vec![5, u32::MAX, 2, 8], 2);
  }

  #[test]
  fn test_reduce_max_u32_basic() {
    test_reduce_max_u32_triple_paths(vec![5, 2, 8, 1, 9], 9);
  }

  #[test]
  fn test_reduce_max_u32_empty() {
    test_reduce_max_u32_triple_paths(vec![], 0);
  }

  #[test]
  fn test_reduce_max_u32_single() {
    test_reduce_max_u32_triple_paths(vec![42], 42);
  }

  #[test]
  fn test_reduce_max_u32_all_same() {
    test_reduce_max_u32_triple_paths(vec![7, 7, 7, 7], 7);
  }

  #[test]
  fn test_reduce_max_u32_with_zero() {
    test_reduce_max_u32_triple_paths(vec![5, 2, 0, 8, 1], 8);
  }

  #[test]
  fn test_reduce_max_u32_with_max() {
    test_reduce_max_u32_triple_paths(vec![5, u32::MAX, 2, 8], u32::MAX);
  }

  // =============================================================================
  //  FIND_MIN_MAX TESTS - COMBINED MIN/MAX IN SINGLE PASS 
  // =============================================================================

  /// Test helper for find_min_max_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_find_min_max_f64_triple_paths(values: Vec<f64>, expected_min: f64, expected_max: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = find_min_max_f64(&values).unwrap();
    assert_eq!(
      scalar_result,
      (expected_min, expected_max),
      "SCALAR path failed - expected ({}, {}), got ({}, {})",
      expected_min,
      expected_max,
      scalar_result.0,
      scalar_result.1
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold but below GPU threshold)
    let simd_batch: Vec<f64> = values
      .iter()
      .cycle()
      .take(100) // Ensure we're above SIMD threshold (32) but below GPU threshold (1024)
      .cloned()
      .collect();

    let simd_result = find_min_max_f64(&simd_batch).unwrap();
    assert_eq!(
      simd_result,
      (expected_min, expected_max),
      "SIMD path failed - expected ({}, {}), got ({}, {})",
      expected_min,
      expected_max,
      simd_result.0,
      simd_result.1
    );

    // ^ GPU PATH TEST (massive array)
    {
      let gpu_batch: Vec<f64> = values
        .iter()
        .cycle()
        .take(10000) // Well above GPU_THRESHOLD_MATH (1024)
        .cloned()
        .collect();

      let gpu_result = find_min_max_f64(&gpu_batch).unwrap();
      assert_eq!(
        gpu_result,
        (expected_min, expected_max),
        "GPU path failed - expected ({}, {}), got ({}, {})",
        expected_min,
        expected_max,
        gpu_result.0,
        gpu_result.1
      );
    }
  }

  #[test]
  fn test_find_min_max_f64_basic() {
    test_find_min_max_f64_triple_paths(vec![5.0, 2.0, 8.0, 1.0, 9.0], 1.0, 9.0);
  }

  #[test]
  fn test_find_min_max_f64_empty() {
    test_find_min_max_f64_triple_paths(vec![], f64::INFINITY, f64::NEG_INFINITY);
  }

  #[test]
  fn test_find_min_max_f64_single() {
    test_find_min_max_f64_triple_paths(vec![42.0], 42.0, 42.0);
  }

  /// Test helper for find_min_max_f64 with NaN - tests SCALAR, SIMD, and GPU paths
  fn test_find_min_max_f64_with_nan_triple_paths(values: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let (min, max) = find_min_max_f64(&values).unwrap();
    assert!(min.is_nan(), "SCALAR path min should be NaN");
    assert!(max.is_nan(), "SCALAR path max should be NaN");

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let (simd_min, simd_max) = find_min_max_f64(&simd_values).unwrap();
    assert!(simd_min.is_nan(), "SIMD path min should be NaN");
    assert!(simd_max.is_nan(), "SIMD path max should be NaN");

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    let (gpu_min, gpu_max) = find_min_max_f64(&gpu_values).unwrap();
    assert!(gpu_min.is_nan(), "GPU path min should be NaN");
    assert!(gpu_max.is_nan(), "GPU path max should be NaN");
  }

  #[test]
  fn test_find_min_max_f64_with_nan() {
    test_find_min_max_f64_with_nan_triple_paths(vec![5.0, f64::NAN, 2.0, 8.0]);
  }

  /// Test helper for find_min_max_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_find_min_max_u32_triple_paths(values: Vec<u32>, expected_min: u32, expected_max: u32) {
    //  SCALAR PATH TEST (small array - below threshold)
    let scalar_result = find_min_max_u32(&values).unwrap();
    assert_eq!(
      scalar_result,
      (expected_min, expected_max),
      "SCALAR path failed - expected ({}, {}), got ({}, {})",
      expected_min,
      expected_max,
      scalar_result.0,
      scalar_result.1
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold but below GPU threshold)
    let simd_batch: Vec<u32> = values
      .iter()
      .cycle()
      .take(100) // Ensure we're above SIMD threshold (32) but below GPU threshold (1024)
      .cloned()
      .collect();

    let simd_result = find_min_max_u32(&simd_batch).unwrap();
    assert_eq!(
      simd_result,
      (expected_min, expected_max),
      "SIMD path failed - expected ({}, {}), got ({}, {})",
      expected_min,
      expected_max,
      simd_result.0,
      simd_result.1
    );

    // ^ GPU PATH TEST (massive array)
    {
      let gpu_batch: Vec<u32> = values
        .iter()
        .cycle()
        .take(10000) // Well above GPU_THRESHOLD_MATH (1024)
        .cloned()
        .collect();

      let gpu_result = find_min_max_u32(&gpu_batch).unwrap();
      assert_eq!(
        gpu_result,
        (expected_min, expected_max),
        "GPU path failed - expected ({}, {}), got ({}, {})",
        expected_min,
        expected_max,
        gpu_result.0,
        gpu_result.1
      );
    }
  }

  #[test]
  fn test_find_min_max_u32_basic() {
    test_find_min_max_u32_triple_paths(vec![5, 2, 8, 1, 9], 1, 9);
  }

  #[test]
  fn test_find_min_max_u32_empty() {
    test_find_min_max_u32_triple_paths(vec![], u32::MAX, 0);
  }

  #[test]
  fn test_find_min_max_u32_single() {
    test_find_min_max_u32_triple_paths(vec![42], 42, 42);
  }

  #[test]
  fn test_find_min_max_u32_all_same() {
    test_find_min_max_u32_triple_paths(vec![7, 7, 7, 7], 7, 7);
  }

  #[test]
  fn test_find_min_max_u32_with_zero() {
    test_find_min_max_u32_triple_paths(vec![5, 2, 0, 8, 1], 0, 8);
  }

  #[test]
  fn test_find_min_max_u32_with_max() {
    test_find_min_max_u32_triple_paths(vec![5, u32::MAX, 2, 8], 2, u32::MAX);
  }

  /// Test helper for find_min_max_i64 - tests SCALAR, SIMD, and GPU paths
  fn test_find_min_max_i64_triple_paths(values: Vec<i64>, expected_min: i64, expected_max: i64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut values_copy = values.clone();
    let scalar_result = find_min_max_i64(&mut values_copy).unwrap();
    assert_eq!(
      scalar_result,
      (expected_min, expected_max),
      "SCALAR path failed - expected ({}, {}), got ({}, {})",
      expected_min,
      expected_max,
      scalar_result.0,
      scalar_result.1
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold but below GPU threshold)
    let simd_batch: Vec<i64> = values
      .iter()
      .cycle()
      .take(100) // Ensure we're above SIMD threshold (32) but below GPU threshold (1024)
      .cloned()
      .collect();

    let mut simd_batch_mut = simd_batch;
    let simd_result = find_min_max_i64(&mut simd_batch_mut).unwrap();
    assert_eq!(
      simd_result,
      (expected_min, expected_max),
      "SIMD path failed - expected ({}, {}), got ({}, {})",
      expected_min,
      expected_max,
      simd_result.0,
      simd_result.1
    );

    // ^ GPU PATH TEST (massive array)
    {
      let gpu_batch: Vec<i64> = values
        .iter()
        .cycle()
        .take(10000) // Well above GPU_THRESHOLD_MATH (1024)
        .cloned()
        .collect();

      let mut gpu_batch_mut = gpu_batch;
      let gpu_result = find_min_max_i64(&mut gpu_batch_mut).unwrap();
      assert_eq!(
        gpu_result,
        (expected_min, expected_max),
        "GPU path failed - expected ({}, {}), got ({}, {})",
        expected_min,
        expected_max,
        gpu_result.0,
        gpu_result.1
      );
    }
  }

  #[test]
  fn test_find_min_max_i64_basic() {
    test_find_min_max_i64_triple_paths(vec![5i64, -2, 8, -10, 9], -10, 9);
  }

  #[test]
  fn test_find_min_max_i64_empty() {
    test_find_min_max_i64_triple_paths(vec![], i64::MAX, i64::MIN);
  }

  // =============================================================================
  //  REDUCE_WEIGHTED_SUM TESTS 
  // =============================================================================

  /// Test helper for reduce_weighted_sum_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_weighted_sum_f64_triple_paths(values: Vec<f64>, weights: Vec<u64>, expected: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let result = reduce_weighted_sum_f64(&values, &weights).unwrap();
    assert!(
      (result - expected).abs() < 1e-10,
      "SCALAR path failed - expected {}, got {}",
      expected,
      result
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let simd_weights: Vec<u64> = weights.iter().cycle().take(100).cloned().collect();
    let simd_expected: f64 = if values.is_empty() {
      0.0
    } else {
      (100 / values.len()) as f64 * expected
        + if 100 % values.len() > 0 {
          values
            .iter()
            .zip(weights.iter())
            .take(100 % values.len())
            .map(|(v, w)| v * (*w as f64))
            .sum::<f64>()
        } else {
          0.0
        }
    };
    let simd_result = reduce_weighted_sum_f64(&simd_values, &simd_weights).unwrap();
    assert!(
      (simd_result - simd_expected).abs() < 1e-8,
      "SIMD path failed - expected {}, got {}",
      simd_expected,
      simd_result
    );

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    let gpu_weights: Vec<u64> = weights.iter().cycle().take(10000).cloned().collect();
    let gpu_expected: f64 = if values.is_empty() {
      0.0
    } else {
      (10000 / values.len()) as f64 * expected
        + if 10000 % values.len() > 0 {
          values
            .iter()
            .zip(weights.iter())
            .take(10000 % values.len())
            .map(|(v, w)| v * (*w as f64))
            .sum::<f64>()
        } else {
          0.0
        }
    };
    let gpu_result = reduce_weighted_sum_f64(&gpu_values, &gpu_weights).unwrap();
    assert!(
      (gpu_result - gpu_expected).abs() < 1e-6,
      "GPU path failed - expected {}, got {}",
      gpu_expected,
      gpu_result
    );
  }

  #[test]
  fn test_reduce_weighted_sum_f64_basic() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let weights = vec![1u64, 2, 3, 4];
    let expected = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0; // 1 + 4 + 9 + 16 = 30
    test_reduce_weighted_sum_f64_triple_paths(values, weights, expected);
  }

  #[test]
  fn test_reduce_weighted_sum_f64_empty() {
    test_reduce_weighted_sum_f64_triple_paths(vec![], vec![], 0.0);
  }

  /// Test helper for reduce_weighted_sum_f64 error cases - tests SCALAR, SIMD, and GPU paths
  fn test_reduce_weighted_sum_f64_error_triple_paths(values: Vec<f64>, weights: Vec<u64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let result = reduce_weighted_sum_f64(&values, &weights);
    assert!(result.is_err(), "SCALAR path should return error");

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let simd_weights: Vec<u64> = weights.iter().cycle().take(50).cloned().collect(); // Mismatched length
    let simd_result = reduce_weighted_sum_f64(&simd_values, &simd_weights);
    assert!(simd_result.is_err(), "SIMD path should return error");

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    let gpu_weights: Vec<u64> = weights.iter().cycle().take(5000).cloned().collect(); // Mismatched length
    let gpu_result = reduce_weighted_sum_f64(&gpu_values, &gpu_weights);
    assert!(gpu_result.is_err(), "GPU path should return error");
  }

  #[test]
  fn test_reduce_weighted_sum_f64_mismatched_lengths() {
    test_reduce_weighted_sum_f64_error_triple_paths(
      vec![1.0, 2.0, 3.0],
      vec![1u64, 2], // Shorter than values
    );
  }

  // =============================================================================
  //  DEDUPLICATION TESTS 
  // =============================================================================

  /// Test helper for dedup_sorted_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_dedup_sorted_u32_triple_paths(values: Vec<u32>, expected: Vec<u32>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    dedup_sorted_u32(&mut scalar_values).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values = Vec::new();
    for val in &values {
      // Repeat each value to ensure we get above SIMD threshold
      for _ in 0..10 {
        simd_values.push(*val);
      }
    }
    dedup_sorted_u32(&mut simd_values).unwrap();
    assert_eq!(
      simd_values, expected,
      "SIMD path failed - expected {:?}, got {:?}",
      expected, simd_values
    );

    // ^ GPU PATH TEST (massive array)
    {
      let mut gpu_values = Vec::new();
      for val in &values {
        // Repeat each value many times to get above GPU threshold
        for _ in 0..1000 {
          gpu_values.push(*val);
        }
      }
      dedup_sorted_u32(&mut gpu_values).unwrap();
      assert_eq!(
        gpu_values, expected,
        "GPU path failed - expected {:?}, got {:?}",
        expected, gpu_values
      );
    }
  }

  #[test]
  fn test_dedup_sorted_u32_basic() {
    test_dedup_sorted_u32_triple_paths(vec![1, 1, 2, 2, 2, 3, 4, 4, 5], vec![1, 2, 3, 4, 5]);
  }

  #[test]
  fn test_dedup_sorted_u32_empty() {
    let mut values: Vec<u32> = vec![];
    dedup_sorted_u32(&mut values).unwrap();
    assert_eq!(values, Vec::<u32>::new());
  }

  #[test]
  fn test_dedup_sorted_u32_no_duplicates() {
    let mut values = vec![1, 2, 3, 4, 5];
    dedup_sorted_u32(&mut values).unwrap();
    assert_eq!(values, vec![1, 2, 3, 4, 5]);
  }

  #[test]
  fn test_dedup_sorted_u32_all_same() {
    let mut values = vec![7, 7, 7, 7, 7];
    dedup_sorted_u32(&mut values).unwrap();
    assert_eq!(values, vec![7]);
  }

  // =============================================================================
  //  SORTING TESTS 
  // =============================================================================

  /// Test helper for sort_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_sort_u32_triple_paths(
    values: Vec<u32>,
    dedup: bool,
    ascending: bool,
    expected: Vec<u32>,
  ) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    sort_u32(&mut scalar_values, dedup, ascending).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<u32> = values.iter().cycle().take(100).cloned().collect();
    sort_u32(&mut simd_values, dedup, ascending).unwrap();
    // Verify first few elements match expected pattern
    if dedup {
      assert_eq!(
        &simd_values[..expected.len()],
        &expected[..],
        "SIMD path failed"
      );
    } else {
      // Check sorting order is maintained
      for i in 1..simd_values.len() {
        if ascending {
          assert!(
            simd_values[i - 1] <= simd_values[i],
            "SIMD path not sorted correctly"
          );
        } else {
          assert!(
            simd_values[i - 1] >= simd_values[i],
            "SIMD path not sorted correctly"
          );
        }
      }
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<u32> = values.iter().cycle().take(10000).cloned().collect();
    sort_u32(&mut gpu_values, dedup, ascending).unwrap();
    // Verify sorting order
    for i in 1..gpu_values.len().min(1000) {
      if ascending {
        assert!(
          gpu_values[i - 1] <= gpu_values[i],
          "GPU path not sorted correctly at {}, values[{}]={}, values[{}]={}",
          i,
          i - 1,
          gpu_values[i - 1],
          i,
          gpu_values[i]
        );
      } else {
        assert!(
          gpu_values[i - 1] >= gpu_values[i],
          "GPU path not sorted correctly at {}, values[{}]={}, values[{}]={}",
          i,
          i - 1,
          gpu_values[i - 1],
          i,
          gpu_values[i]
        );
      }
    }
  }

  #[test]
  fn test_sort_u32_ascending() {
    test_sort_u32_triple_paths(vec![5, 2, 8, 1, 9, 3], false, true, vec![1, 2, 3, 5, 8, 9]);
  }

  #[test]
  fn test_sort_u32_descending() {
    test_sort_u32_triple_paths(vec![5, 2, 8, 1, 9, 3], false, false, vec![9, 8, 5, 3, 2, 1]);
  }

  #[test]
  fn test_sort_u32_with_dedup() {
    test_sort_u32_triple_paths(
      vec![5, 2, 8, 2, 1, 9, 3, 5],
      true,
      true,
      vec![1, 2, 3, 5, 8, 9],
    );
  }

  /// Test helper for sort_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_sort_f64_triple_paths(values: Vec<f64>, dedup: bool, ascending: bool) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    sort_f64(&mut scalar_values, dedup, ascending).unwrap();
    // Verify NaN handling and sort order
    let non_nan_count = scalar_values.iter().filter(|v| !v.is_nan()).count();
    for i in 1..non_nan_count {
      if ascending {
        assert!(
          scalar_values[i - 1] <= scalar_values[i],
          "SCALAR path not sorted"
        );
      } else {
        assert!(
          scalar_values[i - 1] >= scalar_values[i],
          "SCALAR path not sorted"
        );
      }
    }
    // NaN should be at the end
    for val in scalar_values.iter().skip(non_nan_count) {
      assert!(val.is_nan(), "NaN not at end in SCALAR path");
    }

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    sort_f64(&mut simd_values, dedup, ascending).unwrap();
    let simd_non_nan = simd_values.iter().filter(|v| !v.is_nan()).count();
    for i in 1..simd_non_nan.min(50) {
      if ascending {
        assert!(simd_values[i - 1] <= simd_values[i], "SIMD path not sorted");
      } else {
        assert!(simd_values[i - 1] >= simd_values[i], "SIMD path not sorted");
      }
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    sort_f64(&mut gpu_values, dedup, ascending).unwrap();
    let gpu_non_nan = gpu_values.iter().filter(|v| !v.is_nan()).count();
    for i in 1..gpu_non_nan.min(100) {
      if ascending {
        assert!(
          gpu_values[i - 1] <= gpu_values[i],
          "GPU path not sorted at {}",
          i
        );
      } else {
        assert!(
          gpu_values[i - 1] >= gpu_values[i],
          "GPU path not sorted at {}",
          i
        );
      }
    }
  }

  #[test]
  fn test_sort_f64_with_nan() {
    test_sort_f64_triple_paths(vec![5.0, f64::NAN, 2.0, 8.0, f64::NAN, 1.0], false, true);
  }

  /// Test helper for sort_i64 - tests SCALAR, SIMD, and GPU paths
  fn test_sort_i64_triple_paths(
    values: Vec<i64>,
    dedup: bool,
    ascending: bool,
    expected: Vec<i64>,
  ) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    sort_i64(&mut scalar_values, dedup, ascending).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<i64> = values.iter().cycle().take(100).cloned().collect();
    sort_i64(&mut simd_values, dedup, ascending).unwrap();
    // Verify sorting order
    for i in 1..simd_values.len() {
      if ascending {
        assert!(
          simd_values[i - 1] <= simd_values[i],
          "SIMD path not sorted correctly"
        );
      } else {
        assert!(
          simd_values[i - 1] >= simd_values[i],
          "SIMD path not sorted correctly"
        );
      }
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<i64> = values.iter().cycle().take(10000).cloned().collect();
    sort_i64(&mut gpu_values, dedup, ascending).unwrap();
    // Verify sorting order
    for i in 1..gpu_values.len().min(1000) {
      if ascending {
        assert!(
          gpu_values[i - 1] <= gpu_values[i],
          "GPU path not sorted correctly at {}",
          i
        );
      } else {
        assert!(
          gpu_values[i - 1] >= gpu_values[i],
          "GPU path not sorted correctly at {}",
          i
        );
      }
    }
  }

  #[test]
  fn test_sort_i64_negative() {
    test_sort_i64_triple_paths(
      vec![5, -2, 8, -10, 0, 3],
      false,
      true,
      vec![-10, -2, 0, 3, 5, 8],
    );
  }

  // =============================================================================
  //  FILTER RANGE TESTS 
  // =============================================================================

  /// Test helper for filter_range_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_filter_range_u32_triple_paths(values: Vec<u32>, min: u32, max: u32, expected: Vec<u32>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    filter_range_u32(&mut scalar_values, min, max).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<u32> = values.iter().cycle().take(100).cloned().collect();
    filter_range_u32(&mut simd_values, min, max).unwrap();
    let simd_expected: Vec<u32> = expected
      .iter()
      .cycle()
      .take(simd_values.len())
      .cloned()
      .collect();
    assert_eq!(simd_values, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array)
    {
      let mut gpu_values: Vec<u32> = values.iter().cycle().take(10000).cloned().collect();
      filter_range_u32(&mut gpu_values, min, max).unwrap();
      let gpu_expected: Vec<u32> = expected
        .iter()
        .cycle()
        .take(gpu_values.len())
        .cloned()
        .collect();
      assert_eq!(gpu_values, gpu_expected, "GPU path failed");
    }
  }

  #[test]
  fn test_filter_range_u32_basic() {
    test_filter_range_u32_triple_paths(vec![1, 5, 10, 15, 20, 25], 10, 20, vec![10, 15, 20]);
  }

  /// Test helper for filter_range_u64 - tests SCALAR, SIMD, and GPU paths
  fn test_filter_range_u64_triple_paths(values: Vec<u64>, min: u64, max: u64, expected: Vec<u64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    filter_range_u64(&mut scalar_values, min, max).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<u64> = values.iter().cycle().take(100).cloned().collect();
    filter_range_u64(&mut simd_values, min, max).unwrap();
    let simd_expected: Vec<u64> = expected
      .iter()
      .cycle()
      .take(simd_values.len())
      .cloned()
      .collect();
    assert_eq!(simd_values, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array)
    {
      let mut gpu_values: Vec<u64> = values.iter().cycle().take(10000).cloned().collect();
      filter_range_u64(&mut gpu_values, min, max).unwrap();
      let gpu_expected: Vec<u64> = expected
        .iter()
        .cycle()
        .take(gpu_values.len())
        .cloned()
        .collect();
      assert_eq!(gpu_values, gpu_expected, "GPU path failed");
    }
  }

  #[test]
  fn test_filter_range_u64_basic() {
    test_filter_range_u64_triple_paths(vec![1, 5, 10, 15, 20, 25], 10, 20, vec![10, 15, 20]);
  }

  /// Test helper for filter_range_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_filter_range_f64_triple_paths(values: Vec<f64>, min: f64, max: f64, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    filter_range_f64(&mut scalar_values, min, max).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    filter_range_f64(&mut simd_values, min, max).unwrap();
    let simd_expected: Vec<f64> = expected
      .iter()
      .cycle()
      .take(simd_values.len())
      .cloned()
      .collect();
    assert_eq!(simd_values, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array)
    {
      let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
      filter_range_f64(&mut gpu_values, min, max).unwrap();
      let gpu_expected: Vec<f64> = expected
        .iter()
        .cycle()
        .take(gpu_values.len())
        .cloned()
        .collect();
      assert_eq!(gpu_values, gpu_expected, "GPU path failed");
    }
  }

  #[test]
  fn test_filter_range_f64_basic() {
    test_filter_range_f64_triple_paths(
      vec![1.0, 5.5, 10.0, 15.5, 20.0, 25.5],
      10.0,
      20.0,
      vec![10.0, 15.5, 20.0],
    );
  }

  #[test]
  fn test_filter_range_u32_empty_result() {
    let mut values = vec![1, 2, 3, 4, 5];
    filter_range_u32(&mut values, 10, 20).unwrap();
    assert_eq!(values, Vec::<u32>::new());
  }

  // =============================================================================
  //  UNION TESTS 
  // =============================================================================

  /// Test helper for union_sorted_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_union_sorted_u32_triple_paths(a: Vec<u32>, b: Vec<u32>, expected: Vec<u32>) {
    //  SCALAR PATH TEST (small arrays - below threshold)
    let arrays = vec![a.as_slice(), b.as_slice()];
    let mut result = vec![0u32; expected.len() + 2];
    let result_len = result.len();
    union_sorted_u32(&arrays, &mut result, result_len, true).unwrap();
    assert_eq!(
      &result[..expected.len()],
      &expected[..],
      "SCALAR path failed - expected {:?}, got {:?}",
      &expected[..],
      &result[..expected.len()]
    );

    //  SIMD PATH TEST (medium arrays - above SIMD threshold)
    let simd_a: Vec<u32> = a.iter().flat_map(|&x| vec![x; 10]).collect();
    let simd_b: Vec<u32> = b.iter().flat_map(|&x| vec![x; 10]).collect();
    let simd_arrays = vec![simd_a.as_slice(), simd_b.as_slice()];
    let mut simd_result = vec![0u32; expected.len() + 2];
    let simd_result_len = simd_result.len();
    union_sorted_u32(&simd_arrays, &mut simd_result, simd_result_len, true).unwrap();
    assert_eq!(
      &simd_result[..expected.len()],
      &expected[..],
      "SIMD path failed"
    );

    // ^ GPU PATH TEST (massive arrays)
    {
      let gpu_a: Vec<u32> = a.iter().flat_map(|&x| vec![x; 1000]).collect();
      let gpu_b: Vec<u32> = b.iter().flat_map(|&x| vec![x; 1000]).collect();
      let gpu_arrays = vec![gpu_a.as_slice(), gpu_b.as_slice()];
      let mut gpu_result = vec![0u32; expected.len() + 2];
      let gpu_result_len = gpu_result.len();
      union_sorted_u32(&gpu_arrays, &mut gpu_result, gpu_result_len, true).unwrap();
      assert_eq!(
        &gpu_result[..expected.len()],
        &expected[..],
        "GPU path failed"
      );
    }
  }

  #[test]
  fn test_union_sorted_u32_basic() {
    test_union_sorted_u32_triple_paths(
      vec![1, 3, 5, 7],
      vec![2, 4, 6, 8],
      vec![1, 2, 3, 4, 5, 6, 7, 8],
    );
  }

  #[test]
  fn test_union_sorted_u32_with_duplicates() {
    let a = vec![1, 3, 5];
    let b = vec![3, 5, 7];
    let arrays = vec![a.as_slice(), b.as_slice()];
    let mut result = vec![0u32; 10];
    union_sorted_u32(&arrays, &mut result, 10, true).unwrap();
    // Should contain all unique elements
    assert_eq!(&result[..4], &[1, 3, 5, 7]);
  }

  #[test]
  fn test_union_sorted_u32_empty_arrays() {
    let a: Vec<u32> = vec![];
    let b = vec![1, 2, 3];
    let arrays = vec![a.as_slice(), b.as_slice()];
    let mut result = vec![0u32; 10];
    union_sorted_u32(&arrays, &mut result, 10, true).unwrap();
    assert_eq!(&result[..3], &[1, 2, 3]);
  }

  // =============================================================================
  //  PERCENTILE TESTS 
  // =============================================================================

  /// Test helper for calculate_multi_percentiles_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_calculate_multi_percentiles_f64_triple_paths(
    base_values: Vec<f64>,
    percentiles: Vec<f64>,
    tolerance: f64,
  ) {
    // First sort the base values as the function expects sorted input
    let mut sorted_base = base_values.clone();
    sorted_base.sort_by(|a, b| a.partial_cmp(b).unwrap());

    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_results = vec![0.0; percentiles.len()];
    calculate_multi_percentiles_f64(&sorted_base, &percentiles, &mut scalar_results).unwrap();

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    // To test SIMD path correctly, we need more unique values to maintain percentile accuracy
    let mut simd_values: Vec<f64> = Vec::with_capacity(100);
    // Create 100 unique values that maintain the same percentile distribution
    for i in 0..100 {
      // Map 0-99 to the range of base_values maintaining distribution
      let min_val = *sorted_base.first().unwrap();
      let max_val = *sorted_base.last().unwrap();
      let val = min_val + (max_val - min_val) * (i as f64 / 99.0);
      simd_values.push(val);
    }
    let mut simd_results = vec![0.0; percentiles.len()];
    calculate_multi_percentiles_f64(&simd_values, &percentiles, &mut simd_results).unwrap();

    // SIMD results should be close to scalar results with the interpolated data
    for i in 0..percentiles.len() {
      assert!(
        (simd_results[i] - scalar_results[i]).abs() < tolerance,
        "SIMD path failed for percentile {}: expected {}, got {}",
        percentiles[i],
        scalar_results[i],
        simd_results[i]
      );
    }

    // ^ GPU PATH TEST (massive array)
    {
      // Create 10000 unique values that maintain the same percentile distribution
      let mut gpu_values: Vec<f64> = Vec::with_capacity(10000);
      let min_val = *sorted_base.first().unwrap();
      let max_val = *sorted_base.last().unwrap();
      for i in 0..10000 {
        let val = min_val + (max_val - min_val) * (i as f64 / 9999.0);
        gpu_values.push(val);
      }
      let mut gpu_results = vec![0.0; percentiles.len()];
      calculate_multi_percentiles_f64(&gpu_values, &percentiles, &mut gpu_results).unwrap();
      for i in 0..percentiles.len() {
        assert!(
          (gpu_results[i] - scalar_results[i]).abs() < tolerance,
          "GPU path failed for percentile {}: expected {}, got {}",
          percentiles[i],
          scalar_results[i],
          gpu_results[i]
        );
      }
    }
  }

  #[test]
  fn test_calculate_multi_percentiles_f64_basic() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let percentiles = vec![25.0, 50.0, 75.0];
    test_calculate_multi_percentiles_f64_triple_paths(values, percentiles, 1.0);
  }

  /// Test helper for calculate_percentiles_batch_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_calculate_percentiles_batch_f64_triple_paths(
    dataset1: Vec<f64>,
    dataset2: Vec<f64>,
    percentile: f64,
    tolerance: f64,
  ) {
    //  SCALAR PATH TEST (small datasets - below threshold)
    let datasets = vec![dataset1.as_slice(), dataset2.as_slice()];
    let mut results = vec![0.0; 2];
    calculate_percentiles_batch_f64(&datasets, percentile, &mut results).unwrap();
    let scalar_results = results.clone();

    //  SIMD PATH TEST (medium datasets - above SIMD threshold)
    let simd_dataset1: Vec<f64> = dataset1.iter().cycle().take(100).cloned().collect();
    let simd_dataset2: Vec<f64> = dataset2.iter().cycle().take(100).cloned().collect();
    let simd_datasets = vec![simd_dataset1.as_slice(), simd_dataset2.as_slice()];
    let mut simd_results = vec![0.0; 2];
    calculate_percentiles_batch_f64(&simd_datasets, percentile, &mut simd_results).unwrap();
    for i in 0..2 {
      assert!(
        (simd_results[i] - scalar_results[i]).abs() < tolerance,
        "SIMD path failed for dataset {}: expected {}, got {}",
        i,
        scalar_results[i],
        simd_results[i]
      );
    }

    // ^ GPU PATH TEST (massive datasets)
    {
      let gpu_dataset1: Vec<f64> = dataset1.iter().cycle().take(10000).cloned().collect();
      let gpu_dataset2: Vec<f64> = dataset2.iter().cycle().take(10000).cloned().collect();
      let gpu_datasets = vec![gpu_dataset1.as_slice(), gpu_dataset2.as_slice()];
      let mut gpu_results = vec![0.0; 2];
      calculate_percentiles_batch_f64(&gpu_datasets, percentile, &mut gpu_results).unwrap();
      for i in 0..2 {
        assert!(
          (gpu_results[i] - scalar_results[i]).abs() < tolerance,
          "GPU path failed for dataset {}: expected {}, got {}",
          i,
          scalar_results[i],
          gpu_results[i]
        );
      }
    }
  }

  #[test]
  fn test_calculate_percentiles_batch_f64_basic() {
    test_calculate_percentiles_batch_f64_triple_paths(
      vec![1.0, 2.0, 3.0, 4.0, 5.0],
      vec![10.0, 20.0, 30.0, 40.0, 50.0],
      50.0,
      1.0,
    );
  }

  // =============================================================================
  //  FMA (FUSED MULTIPLY-ADD) TESTS 
  // =============================================================================

  /// Test helper for fma_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_fma_f64_triple_paths(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_results = vec![0.0; a.len()];
    fma_f64(&a, &b, &c, &mut scalar_results).unwrap();
    assert_eq!(
      scalar_results, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_results
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_a: Vec<f64> = a.iter().cycle().take(100).cloned().collect();
    let simd_b: Vec<f64> = b.iter().cycle().take(100).cloned().collect();
    let simd_c: Vec<f64> = c.iter().cycle().take(100).cloned().collect();
    let mut simd_results = vec![0.0; 100];
    fma_f64(&simd_a, &simd_b, &simd_c, &mut simd_results).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    for i in 0..100 {
      assert!(
        (simd_results[i] - simd_expected[i]).abs() < 1e-10,
        "SIMD path failed at index {} - expected {}, got {}",
        i,
        simd_expected[i],
        simd_results[i]
      );
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_a: Vec<f64> = a.iter().cycle().take(10000).cloned().collect();
    let gpu_b: Vec<f64> = b.iter().cycle().take(10000).cloned().collect();
    let gpu_c: Vec<f64> = c.iter().cycle().take(10000).cloned().collect();
    let mut gpu_results = vec![0.0; 10000];
    fma_f64(&gpu_a, &gpu_b, &gpu_c, &mut gpu_results).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_results[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_results[i]
      );
    }
  }

  #[test]
  fn test_fma_f64_basic() {
    test_fma_f64_triple_paths(
      vec![1.0, 2.0, 3.0, 4.0],
      vec![2.0, 3.0, 4.0, 5.0],
      vec![1.0, 1.0, 1.0, 1.0],
      vec![3.0, 7.0, 13.0, 21.0], // a[i] * b[i] + c[i]
    );
  }

  #[test]
  fn test_fma_f64_zeros() {
    test_fma_f64_triple_paths(
      vec![0.0, 0.0, 0.0],
      vec![5.0, 6.0, 7.0],
      vec![1.0, 2.0, 3.0],
      vec![1.0, 2.0, 3.0], // 0 * b + c = c
    );
  }

  // =============================================================================
  //  LINEAR INTERPOLATION TESTS 
  // =============================================================================

  /// Test helper for linear_interpolate_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_linear_interpolate_f64_triple_paths(
    lower: Vec<f64>,
    upper: Vec<f64>,
    weights: Vec<f64>,
    expected: Vec<f64>,
  ) {
    //  SCALAR PATH TEST (small arrays - below threshold)
    let mut results = vec![0.0; lower.len()];
    linear_interpolate_f64(&lower, &upper, &weights, &mut results).unwrap();
    assert_eq!(
      results, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, results
    );

    //  SIMD PATH TEST (medium arrays - above SIMD threshold)
    let simd_lower: Vec<f64> = lower.iter().cycle().take(100).cloned().collect();
    let simd_upper: Vec<f64> = upper.iter().cycle().take(100).cloned().collect();
    let simd_weights: Vec<f64> = weights.iter().cycle().take(100).cloned().collect();
    let mut simd_results = vec![0.0; 100];
    linear_interpolate_f64(&simd_lower, &simd_upper, &simd_weights, &mut simd_results).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_results, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive arrays)
    {
      let gpu_lower: Vec<f64> = lower.iter().cycle().take(10000).cloned().collect();
      let gpu_upper: Vec<f64> = upper.iter().cycle().take(10000).cloned().collect();
      let gpu_weights: Vec<f64> = weights.iter().cycle().take(10000).cloned().collect();
      let mut gpu_results = vec![0.0; 10000];
      linear_interpolate_f64(&gpu_lower, &gpu_upper, &gpu_weights, &mut gpu_results).unwrap();
      let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
      for i in (0..10000).step_by(1000) {
        assert!(
          (gpu_results[i] - gpu_expected[i]).abs() < 0.01,
          "GPU path failed at index {}",
          i
        );
      }
    }
  }

  #[test]
  fn test_linear_interpolate_f64_basic() {
    test_linear_interpolate_f64_triple_paths(
      vec![0.0, 10.0, 20.0],
      vec![10.0, 20.0, 30.0],
      vec![0.5, 0.25, 0.75],
      vec![5.0, 12.5, 27.5], // lower[i] + (upper[i] - lower[i]) * weights[i]
    );
  }

  #[test]
  fn test_linear_interpolate_f64_edge_weights() {
    let lower = vec![0.0, 10.0];
    let upper = vec![100.0, 20.0];
    let weights = vec![0.0, 1.0];
    let mut results = vec![0.0; 2];
    linear_interpolate_f64(&lower, &upper, &weights, &mut results).unwrap();

    // weight 0.0 gives lower, weight 1.0 gives upper
    assert_eq!(results, vec![0.0, 20.0]);
  }

  // =============================================================================
  //  QUANTIZATION TESTS 
  // =============================================================================

  /// Test helper for vectorized_quantize_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_vectorized_quantize_f64_triple_paths(
    values: Vec<f64>,
    min_val: f64,
    scale: f64,
    max_val: f64,
    expected: Vec<u32>,
  ) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut output = vec![0u32; values.len()];
    vectorized_quantize_f64(&values, min_val, scale, max_val, &mut output).unwrap();
    assert_eq!(
      output, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, output
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let mut simd_output = vec![0u32; 100];
    vectorized_quantize_f64(&simd_values, min_val, scale, max_val, &mut simd_output).unwrap();
    let simd_expected: Vec<u32> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_output, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array)
    {
      let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
      let mut gpu_output = vec![0u32; 10000];
      vectorized_quantize_f64(&gpu_values, min_val, scale, max_val, &mut gpu_output).unwrap();
      let gpu_expected: Vec<u32> = expected.iter().cycle().take(10000).cloned().collect();
      for i in (0..10000).step_by(1000) {
        assert_eq!(
          gpu_output[i], gpu_expected[i],
          "GPU path failed at index {}",
          i
        );
      }
    }
  }

  #[test]
  fn test_vectorized_quantize_f64_basic() {
    test_vectorized_quantize_f64_triple_paths(
      vec![0.0, 5.0, 10.0, 15.0, 20.0],
      0.0,
      20.0,  // scale
      100.0, // max_val
      vec![0, 25, 50, 75, 100],
    );
  }

  #[test]
  fn test_vectorized_quantize_f64_clipping() {
    let values = vec![-5.0, 0.0, 10.0, 20.0, 25.0];
    let mut output = vec![0u32; 5];
    vectorized_quantize_f64(&values, 0.0, 20.0, 100.0, &mut output).unwrap();

    // Values outside [0-20] should be clipped
    assert_eq!(output, vec![0, 0, 50, 100, 100]);
  }

  // =============================================================================
  //  DELTA ENCODING TESTS 
  // =============================================================================

  /// Test helper for vectorized_subtract_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_vectorized_subtract_u32_triple_paths(values: Vec<u32>, base: u32, expected: Vec<u32>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut output = vec![0u32; values.len()];
    vectorized_subtract_u32(&values, base, &mut output).unwrap();
    assert_eq!(
      output, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, output
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<u32> = values.iter().cycle().take(100).cloned().collect();
    let mut simd_output = vec![0u32; 100];
    vectorized_subtract_u32(&simd_values, base, &mut simd_output).unwrap();
    let simd_expected: Vec<u32> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_output, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array)
    {
      let gpu_values: Vec<u32> = values.iter().cycle().take(10000).cloned().collect();
      let mut gpu_output = vec![0u32; 10000];
      vectorized_subtract_u32(&gpu_values, base, &mut gpu_output).unwrap();
      let gpu_expected: Vec<u32> = expected.iter().cycle().take(10000).cloned().collect();
      for i in (0..10000).step_by(1000) {
        assert_eq!(
          gpu_output[i], gpu_expected[i],
          "GPU path failed at index {}",
          i
        );
      }
    }
  }

  #[test]
  fn test_vectorized_subtract_u32_basic() {
    test_vectorized_subtract_u32_triple_paths(
      vec![100, 105, 110, 115, 120],
      100,
      vec![0, 5, 10, 15, 20],
    );
  }

  #[test]
  fn test_vectorized_subtract_u32_underflow() {
    let values = vec![50, 100, 150];
    let base = 100;
    let mut output = vec![0u32; 3];
    vectorized_subtract_u32(&values, base, &mut output).unwrap();

    // Underflow should saturate to 0
    assert_eq!(output, vec![0, 0, 50]);
  }

  // =============================================================================
  //  IS_SORTED TESTS 
  // =============================================================================

  /// Test helper for is_sorted_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_is_sorted_u32_triple_paths(values: Vec<u32>, expected: bool) {
    //  SCALAR PATH TEST (small array - below threshold)
    let result = is_sorted_u32(&values).unwrap();
    assert_eq!(
      result, expected,
      "SCALAR path failed - expected {}, got {}",
      expected, result
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    if !values.is_empty() {
      // Create sorted prefix then append values
      let mut simd_values = Vec::new();
      for i in 0..50 {
        simd_values.push(i as u32);
      }
      simd_values.extend_from_slice(&values);
      let simd_result = is_sorted_u32(&simd_values).unwrap();
      // Only sorted if original was sorted and starts after sorted prefix
      let simd_expected = expected && (values.is_empty() || values[0] >= 49);
      assert_eq!(
        simd_result, simd_expected,
        "SIMD path failed - expected {}, got {}",
        simd_expected, simd_result
      );
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    if !values.is_empty() {
      let mut gpu_values = Vec::new();
      for i in 0..5000 {
        gpu_values.push(i as u32);
      }
      gpu_values.extend_from_slice(&values);
      let gpu_result = is_sorted_u32(&gpu_values).unwrap();
      let gpu_expected = expected && (values.is_empty() || values[0] >= 4999);
      assert_eq!(
        gpu_result, gpu_expected,
        "GPU path failed - expected {}, got {}",
        gpu_expected, gpu_result
      );
    }
  }

  #[test]
  fn test_is_sorted_u32_ascending() {
    test_is_sorted_u32_triple_paths(vec![1, 2, 3, 4, 5], true);
    test_is_sorted_u32_triple_paths(vec![1, 2, 2, 3, 4], true); // Duplicates are fine
    test_is_sorted_u32_triple_paths(vec![1, 3, 2, 4, 5], false);
  }

  #[test]
  fn test_is_sorted_u32_empty() {
    let empty: &[u32] = &[];
    assert!(is_sorted_u32(empty).unwrap());
  }

  #[test]
  fn test_is_sorted_u32_single() {
    assert!(is_sorted_u32(&[42]).unwrap());
  }

  // =============================================================================
  //  FILTER COUNTS TESTS 
  // =============================================================================

  /// Test helper for filter_counts_ge_threshold_u64 - tests SCALAR, SIMD, and GPU paths
  fn test_filter_counts_ge_threshold_u64_triple_paths(
    values: Vec<u32>,
    counts: Vec<u64>,
    threshold: u64,
    expected: Vec<u32>,
  ) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    filter_counts_ge_threshold_u64(&mut scalar_values, &counts, threshold, 100).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<u32> = values.iter().cycle().take(100).cloned().collect();
    let simd_counts: Vec<u64> = counts.iter().cycle().take(100).cloned().collect();
    filter_counts_ge_threshold_u64(&mut simd_values, &simd_counts, threshold, 1000).unwrap();
    // Count how many times expected pattern should repeat
    let expected_len = (100 / values.len()) * expected.len()
      + if 100 % values.len() > expected.len() {
        expected.len()
      } else {
        (100 % values.len()).min(expected.len())
      };
    assert_eq!(
      simd_values.len(),
      expected_len,
      "SIMD path failed - expected {} elements, got {}",
      expected_len,
      simd_values.len()
    );

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<u32> = values.iter().cycle().take(10000).cloned().collect();
    let gpu_counts: Vec<u64> = counts.iter().cycle().take(10000).cloned().collect();
    filter_counts_ge_threshold_u64(&mut gpu_values, &gpu_counts, threshold, 100000).unwrap();
    let gpu_expected_len = (10000 / values.len()) * expected.len()
      + if 10000 % values.len() > expected.len() {
        expected.len()
      } else {
        (10000 % values.len()).min(expected.len())
      };
    assert_eq!(
      gpu_values.len(),
      gpu_expected_len,
      "GPU path failed - expected {} elements, got {}",
      gpu_expected_len,
      gpu_values.len()
    );
  }

  #[test]
  fn test_filter_counts_ge_threshold_u64_basic() {
    test_filter_counts_ge_threshold_u64_triple_paths(
      vec![1, 2, 3, 4, 5],
      vec![10, 5, 20, 3, 15],
      10,
      vec![1, 3, 5], // Keep elements where count >= 10
    );
  }

  #[test]
  fn test_filter_counts_ge_threshold_u64_none_pass() {
    let mut input = vec![1u32, 2, 3];
    let counts = vec![5u64, 7, 9];
    let threshold = 10u64;
    filter_counts_ge_threshold_u64(&mut input, &counts, threshold, 10).unwrap();

    assert_eq!(input, Vec::<u32>::new());
  }

  // =============================================================================
  //  USIZE TO U32 CONVERSION TESTS 
  // =============================================================================

  /// Test helper for usize_to_u32 - tests SCALAR, SIMD, and GPU paths
  fn test_usize_to_u32_triple_paths(values: Vec<usize>, expected: Vec<u32>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_results = vec![0u32; values.len()];
    usize_to_u32(&values, &mut scalar_results).unwrap();
    assert_eq!(
      scalar_results, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_results
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<usize> = values.iter().cycle().take(100).cloned().collect();
    let mut simd_results = vec![0u32; 100];
    usize_to_u32(&simd_values, &mut simd_results).unwrap();
    let simd_expected: Vec<u32> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_results, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<usize> = values.iter().cycle().take(10000).cloned().collect();
    let mut gpu_results = vec![0u32; 10000];
    usize_to_u32(&gpu_values, &mut gpu_results).unwrap();
    let gpu_expected: Vec<u32> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert_eq!(
        gpu_results[i], gpu_expected[i],
        "GPU path failed at index {}",
        i
      );
    }
  }

  #[test]
  fn test_usize_to_u32_basic() {
    test_usize_to_u32_triple_paths(vec![100usize, 200, 300, 400], vec![100u32, 200, 300, 400]);
  }

  #[test]
  fn test_usize_to_u32_overflow() {
    let lengths = vec![u32::MAX as usize + 1];
    let mut results = vec![0u32; 1];
    let _ = usize_to_u32(&lengths, &mut results);
  }

  // =============================================================================
  //  SORT_U32_BY_U64 TESTS - TRIPLE PATH TESTING 
  // =============================================================================

  fn test_sort_u32_by_u64_triple_paths(
    indices: Vec<u32>,
    values: Vec<u64>,
    expected_indices: Vec<u32>,
    ascending: bool,
    test_name: &str,
  ) {
    // config_test_logger();

    // Test 1: SCALAR PATH (small input)
    let mut scalar_indices = indices.clone();
    if scalar_indices.len() <= 4 {
      sort_u32_by_u64(&mut scalar_indices, &values, ascending).unwrap();
      assert_eq!(
        scalar_indices, expected_indices,
        "{}: Scalar sort_u32_by_u64 failed",
        test_name
      );
    }

    // Test 2: SIMD PATH (large input to force SIMD)
    if !indices.is_empty() {
      // Create larger arrays to force SIMD path
      let repeat_factor = if indices.len() < 20 {
        20 / indices.len() + 1
      } else {
        1
      };

      let mut simd_indices: Vec<u32> = Vec::new();
      let mut simd_values: Vec<u64> = Vec::new();

      for _ in 0..repeat_factor {
        let base = simd_indices.len() as u32;
        simd_indices.extend(indices.iter().map(|&i| i + base));
        simd_values.extend(&values);
      }

      //  Test with SIMD-sized array (dispatch will choose SIMD path)
      // No need to directly test internal AVX2 function
      sort_u32_by_u64(&mut simd_indices, &simd_values, ascending).unwrap();

      // Verify the sort is correct (may not be stable, so we check value ordering)
      for i in 1..simd_indices.len() {
        let prev_val = simd_values[simd_indices[i - 1] as usize];
        let curr_val = simd_values[simd_indices[i] as usize];
        if ascending {
          assert!(
            prev_val <= curr_val,
            "{}: SIMD sort_u32_by_u64 not in ascending order at index {}: {} > {}",
            test_name,
            i,
            prev_val,
            curr_val
          );
        } else {
          assert!(
            prev_val >= curr_val,
            "{}: SIMD sort_u32_by_u64 not in descending order at index {}: {} < {}",
            test_name,
            i,
            prev_val,
            curr_val
          );
        }
      }
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    if !indices.is_empty() {
      let mut gpu_indices: Vec<u32> = Vec::new();
      let mut gpu_values: Vec<u64> = Vec::new();

      // Create 10000+ elements to force GPU path
      for i in 0..10000 {
        let base_idx = i % indices.len();
        gpu_indices.push(indices[base_idx] + (i / indices.len()) as u32 * indices.len() as u32);
        gpu_values.push(values[base_idx]);
      }

      sort_u32_by_u64(&mut gpu_indices, &gpu_values, ascending).unwrap();

      // Verify GPU sort correctness
      for i in 1..gpu_indices.len().min(1000) {
        let prev_val = gpu_values[gpu_indices[i - 1] as usize % gpu_values.len()];
        let curr_val = gpu_values[gpu_indices[i] as usize % gpu_values.len()];
        if ascending {
          assert!(
            prev_val <= curr_val,
            "{}: GPU sort_u32_by_u64 not in ascending order at index {}: {} > {}",
            test_name,
            i,
            prev_val,
            curr_val
          );
        } else {
          assert!(
            prev_val >= curr_val,
            "{}: GPU sort_u32_by_u64 not in descending order at index {}: {} < {}",
            test_name,
            i,
            prev_val,
            curr_val
          );
        }
      }
    }

    // Test 3: MAIN API PATH (uses automatic dispatch)
    let mut auto_indices = indices.clone();
    sort_u32_by_u64(&mut auto_indices, &values, ascending).unwrap();
    assert_eq!(
      auto_indices, expected_indices,
      "{}: Auto-dispatch sort_u32_by_u64 failed",
      test_name
    );
  }

  #[test]
  fn test_sort_u32_by_u64_ascending() {
    test_sort_u32_by_u64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40, 10, 30, 20],
      vec![1, 3, 2, 0], // Indices sorted by values: 10, 20, 30, 40
      true,
      "ascending",
    );
  }

  #[test]
  fn test_sort_u32_by_u64_descending() {
    test_sort_u32_by_u64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40, 10, 30, 20],
      vec![0, 2, 3, 1], // Indices sorted by values: 40, 30, 20, 10
      false,
      "descending",
    );
  }

  #[test]
  fn test_sort_u32_by_u64_equal_values() {
    test_sort_u32_by_u64_triple_paths(
      vec![0, 1, 2, 3],
      vec![10, 10, 10, 10],
      vec![0, 1, 2, 3], // Stable sort keeps original order for equal values
      true,
      "equal_values",
    );
  }

  #[test]
  fn test_sort_u32_by_u64_single() {
    test_sort_u32_by_u64_triple_paths(vec![0], vec![100], vec![0], true, "single");
  }

  #[test]
  fn test_sort_u32_by_u64_two_elements() {
    test_sort_u32_by_u64_triple_paths(vec![1, 0], vec![20, 10], vec![1, 0], true, "two_ascending");

    test_sort_u32_by_u64_triple_paths(
      vec![0, 1],
      vec![10, 20],
      vec![1, 0],
      false,
      "two_descending",
    );
  }

  #[test]
  fn test_sort_u32_by_u64_already_sorted() {
    test_sort_u32_by_u64_triple_paths(
      vec![0, 1, 2, 3],
      vec![10, 20, 30, 40],
      vec![0, 1, 2, 3],
      true,
      "already_sorted",
    );
  }

  #[test]
  fn test_sort_u32_by_u64_reverse_sorted() {
    test_sort_u32_by_u64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40, 30, 20, 10],
      vec![3, 2, 1, 0],
      true,
      "reverse_sorted",
    );
  }

  #[test]
  fn test_sort_u32_by_u64_max_values() {
    test_sort_u32_by_u64_triple_paths(
      vec![0, 1, 2],
      vec![u64::MAX, 0, u64::MAX / 2],
      vec![1, 2, 0], // 0, MAX/2, MAX
      true,
      "max_values",
    );
  }

  // =============================================================================
  //  SORT_U32_BY_F64 TESTS - TRIPLE PATH TESTING WITH NAN HANDLING 
  // =============================================================================

  fn test_sort_u32_by_f64_triple_paths(
    indices: Vec<u32>,
    values: Vec<f64>,
    expected_indices: Vec<u32>,
    ascending: bool,
    test_name: &str,
  ) {
    // config_test_logger();

    // Test 1: SCALAR PATH (small input)
    let mut scalar_indices = indices.clone();
    let scalar_len = scalar_indices.len();
    if scalar_len <= 4 {
      // Use public API instead of internal function
      sort_u32_by_f64(&mut scalar_indices, &values, ascending).unwrap();
      assert_eq!(
        scalar_indices, expected_indices,
        "{}: Scalar sort_u32_by_f64 failed",
        test_name
      );
    }

    // Test 2: SIMD PATH (large input to force SIMD)
    if !indices.is_empty() {
      // Create larger arrays to force SIMD path
      let repeat_factor = if indices.len() < 20 {
        20 / indices.len() + 1
      } else {
        1
      };

      let mut simd_indices: Vec<u32> = Vec::new();
      let mut large_values: Vec<f64> = Vec::new();

      for _ in 0..repeat_factor {
        let base = simd_indices.len() as u32;
        simd_indices.extend(indices.iter().map(|&i| i + base));
        large_values.extend(&values);
      }

      //  Test with SIMD-sized array (dispatch will choose SIMD path)
      // No need to directly test internal AVX2/AVX512 functions
      sort_u32_by_f64(&mut simd_indices, &large_values, ascending).unwrap();

      // Verify the sort is correct (may not be stable, so we check value ordering)
      for i in 1..simd_indices.len() {
        let prev_val = large_values[simd_indices[i - 1] as usize];
        let curr_val = large_values[simd_indices[i] as usize];
        if ascending {
          // For ascending: NaN should be at the end, skip NaN comparisons
          if !prev_val.is_nan() && !curr_val.is_nan() {
            assert!(
              prev_val <= curr_val,
              "{}: SIMD sort_u32_by_f64 not in ascending order at index {}: {} > {}",
              test_name,
              i,
              prev_val,
              curr_val
            );
          }
        } else {
          // For descending: NaN should be at the end, skip NaN comparisons
          if !prev_val.is_nan() && !curr_val.is_nan() {
            assert!(
              prev_val >= curr_val,
              "{}: SIMD sort_u32_by_f64 not in descending order at index {}: {} < {}",
              test_name,
              i,
              prev_val,
              curr_val
            );
          }
        }
      }
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    if !indices.is_empty() {
      let mut gpu_indices: Vec<u32> = Vec::new();
      let mut gpu_values: Vec<f64> = Vec::new();

      // Create 10000+ elements to force GPU path
      for i in 0..10000 {
        let base_idx = i % indices.len();
        gpu_indices.push(indices[base_idx] + (i / indices.len()) as u32 * indices.len() as u32);
        gpu_values.push(values[base_idx]);
      }

      sort_u32_by_f64(&mut gpu_indices, &gpu_values, ascending).unwrap();

      // Verify GPU sort correctness
      for i in 1..gpu_indices.len().min(1000) {
        let prev_val = gpu_values[gpu_indices[i - 1] as usize % gpu_values.len()];
        let curr_val = gpu_values[gpu_indices[i] as usize % gpu_values.len()];
        if ascending {
          // Handle NaN correctly - NaN should sort to end
          if !prev_val.is_nan() && !curr_val.is_nan() {
            assert!(
              prev_val <= curr_val,
              "{}: GPU sort_u32_by_f64 not in ascending order at index {}: {} > {}",
              test_name,
              i,
              prev_val,
              curr_val
            );
          }
        } else {
          // For descending: NaN should be at the end, skip NaN comparisons
          if !prev_val.is_nan() && !curr_val.is_nan() {
            assert!(
              prev_val >= curr_val,
              "{}: GPU sort_u32_by_f64 not in descending order at index {}: {} < {}",
              test_name,
              i,
              prev_val,
              curr_val
            );
          }
        }
      }
    }

    // Test 3: MAIN API PATH (uses automatic dispatch)
    let mut auto_indices = indices.clone();
    sort_u32_by_f64(&mut auto_indices, &values, ascending).unwrap();
    assert_eq!(
      auto_indices, expected_indices,
      "{}: Auto-dispatch sort_u32_by_f64 failed",
      test_name
    );
  }

  #[test]
  fn test_sort_u32_by_f64_ascending() {
    test_sort_u32_by_f64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40.5, 10.2, 30.7, 20.3],
      vec![1, 3, 2, 0], // Indices sorted by values
      true,
      "f64_ascending",
    );
  }

  #[test]
  fn test_sort_u32_by_f64_descending() {
    test_sort_u32_by_f64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40.5, 10.2, 30.7, 20.3],
      vec![0, 2, 3, 1], // Indices sorted by values descending
      false,
      "f64_descending",
    );
  }

  #[test]
  fn test_sort_u32_by_f64_with_nan() {
    // NaN should sort to the end regardless of order
    test_sort_u32_by_f64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40.5, f64::NAN, 10.2, 20.3],
      vec![2, 3, 0, 1], // NaN at the end
      true,
      "f64_with_nan_asc",
    );

    test_sort_u32_by_f64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40.5, f64::NAN, 10.2, 20.3],
      vec![0, 3, 2, 1], // NaN at the end even in desc
      false,
      "f64_with_nan_desc",
    );
  }

  #[test]
  fn test_sort_u32_by_f64_all_nan() {
    test_sort_u32_by_f64_triple_paths(
      vec![0, 1, 2],
      vec![f64::NAN, f64::NAN, f64::NAN],
      vec![0, 1, 2], // Stable sort preserves order
      true,
      "f64_all_nan",
    );
  }

  #[test]
  fn test_sort_u32_by_f64_infinity() {
    test_sort_u32_by_f64_triple_paths(
      vec![0, 1, 2, 3, 4],
      vec![0.0, f64::INFINITY, -10.0, f64::NEG_INFINITY, 10.0],
      vec![3, 2, 0, 4, 1], // -inf, -10, 0, 10, inf
      true,
      "f64_infinity",
    );
  }

  #[test]
  fn test_sort_u32_by_f64_negative_zero() {
    // -0.0 and 0.0 should compare equal
    test_sort_u32_by_f64_triple_paths(
      vec![0, 1, 2],
      vec![0.0, -0.0, 1.0],
      vec![0, 1, 2], // -0 and 0 are equal, stable sort
      true,
      "f64_negative_zero",
    );
  }

  // =============================================================================
  //  SORT_U32_BY_I64 TESTS - TRIPLE PATH TESTING WITH SIGNED VALUES 
  // =============================================================================

  fn test_sort_u32_by_i64_triple_paths(
    indices: Vec<u32>,
    values: Vec<i64>,
    expected_indices: Vec<u32>,
    ascending: bool,
    test_name: &str,
  ) {
    // config_test_logger();

    // Test 1: SCALAR PATH (small input)
    let mut scalar_indices = indices.clone();
    let scalar_len = scalar_indices.len();
    if scalar_len <= 4 {
      sort_u32_by_i64(&mut scalar_indices, &values, ascending).unwrap();
      assert_eq!(
        scalar_indices, expected_indices,
        "{}: Scalar sort_u32_by_i64 failed",
        test_name
      );
    }

    // Test 2: SIMD PATH (large input to force SIMD)
    if !indices.is_empty() {
      // Create larger arrays to force SIMD path
      let repeat_factor = if indices.len() < 20 {
        20 / indices.len() + 1
      } else {
        1
      };

      let mut simd_indices: Vec<u32> = Vec::new();
      let mut simd_values: Vec<i64> = Vec::new();

      for _ in 0..repeat_factor {
        let base = simd_indices.len() as u32;
        simd_indices.extend(indices.iter().map(|&i| i + base));
        simd_values.extend(&values);
      }

      //  Test with SIMD-sized array (dispatch will choose SIMD path)
      // No need to directly test internal AVX2/AVX512 functions
      sort_u32_by_i64(&mut simd_indices, &simd_values, ascending).unwrap();

      // Verify the sort is correct (may not be stable, so we check value ordering)
      for i in 1..simd_indices.len() {
        let prev_val = simd_values[simd_indices[i - 1] as usize];
        let curr_val = simd_values[simd_indices[i] as usize];
        if ascending {
          assert!(
            prev_val <= curr_val,
            "{}: SIMD sort_u32_by_i64 not in ascending order at index {}: {} > {}",
            test_name,
            i,
            prev_val,
            curr_val
          );
        } else {
          assert!(
            prev_val >= curr_val,
            "{}: SIMD sort_u32_by_i64 not in descending order at index {}: {} < {}",
            test_name,
            i,
            prev_val,
            curr_val
          );
        }
      }
    }

    // Test 3: MAIN API PATH (uses automatic dispatch)
    let mut auto_indices = indices.clone();
    sort_u32_by_i64(&mut auto_indices, &values, ascending).unwrap();
    assert_eq!(
      auto_indices, expected_indices,
      "{}: Auto-dispatch sort_u32_by_i64 failed",
      test_name
    );
  }

  #[test]
  fn test_sort_u32_by_i64_ascending() {
    test_sort_u32_by_i64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40, -10, 30, -20],
      vec![3, 1, 2, 0], // -20, -10, 30, 40
      true,
      "i64_ascending",
    );
  }

  #[test]
  fn test_sort_u32_by_i64_descending() {
    test_sort_u32_by_i64_triple_paths(
      vec![0, 1, 2, 3],
      vec![40, -10, 30, -20],
      vec![0, 2, 1, 3], // 40, 30, -10, -20
      false,
      "i64_descending",
    );
  }

  #[test]
  fn test_sort_u32_by_i64_all_negative() {
    test_sort_u32_by_i64_triple_paths(
      vec![0, 1, 2, 3],
      vec![-100, -50, -200, -25],
      vec![2, 0, 1, 3], // -200, -100, -50, -25
      true,
      "i64_all_negative",
    );
  }

  #[test]
  fn test_sort_u32_by_i64_mixed_sign() {
    test_sort_u32_by_i64_triple_paths(
      vec![0, 1, 2, 3, 4],
      vec![100, -50, 0, -100, 50],
      vec![3, 1, 2, 4, 0], // -100, -50, 0, 50, 100
      true,
      "i64_mixed_sign",
    );
  }

  #[test]
  fn test_sort_u32_by_i64_extremes() {
    test_sort_u32_by_i64_triple_paths(
      vec![0, 1, 2],
      vec![i64::MAX, i64::MIN, 0],
      vec![1, 2, 0], // MIN, 0, MAX
      true,
      "i64_extremes",
    );
  }

  #[test]
  fn test_sort_u32_by_i64_single() {
    test_sort_u32_by_i64_triple_paths(vec![0], vec![-100], vec![0], true, "i64_single");
  }

  #[test]
  fn test_sort_u32_by_i64_two_elements() {
    test_sort_u32_by_i64_triple_paths(vec![0, 1], vec![20, -10], vec![1, 0], true, "i64_two_asc");

    test_sort_u32_by_i64_triple_paths(vec![0, 1], vec![-10, 20], vec![1, 0], false, "i64_two_desc");
  }

  // =============================================================================
  // ELEMENT-WISE OPERATION TESTS - Using functions from dispatch
  // =============================================================================

  /// Test helper for clamp_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_clamp_f64_triple_paths(values: Vec<f64>, min: f64, max: f64, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    clamp_f64(&mut scalar_values, min, max).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    clamp_f64(&mut simd_values, min, max).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_values, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    clamp_f64(&mut gpu_values, min, max).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_values[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_values[i]
      );
    }
  }

  #[test]
  fn test_clamp_f64() {
    // Basic clamping test
    test_clamp_f64_triple_paths(
      vec![0.5, 1.5, 2.5, 3.5, 4.5],
      1.0,
      3.0,
      vec![1.0, 1.5, 2.5, 3.0, 3.0],
    );

    // Test with all values outside range
    let mut all_outside = vec![-10.0, -5.0, 100.0, 200.0];
    clamp_f64(&mut all_outside, 0.0, 10.0).unwrap();
    assert_eq!(all_outside, vec![0.0, 0.0, 10.0, 10.0]);

    // Test with NaN values (NaN should remain NaN)
    let mut with_nan = vec![1.0, f64::NAN, 5.0, f64::NAN];
    clamp_f64(&mut with_nan, 2.0, 4.0).unwrap();
    assert_eq!(with_nan[0], 2.0);
    assert!(with_nan[1].is_nan());
    assert_eq!(with_nan[2], 4.0);
    assert!(with_nan[3].is_nan());

    // Test with infinity
    let mut with_inf = vec![f64::NEG_INFINITY, 2.5, f64::INFINITY];
    clamp_f64(&mut with_inf, 0.0, 10.0).unwrap();
    assert_eq!(with_inf, vec![0.0, 2.5, 10.0]);

    // Test empty array
    let mut empty: Vec<f64> = vec![];
    clamp_f64(&mut empty, 0.0, 1.0).unwrap();
    assert_eq!(empty, Vec::<f64>::new());
  }

  /// Test helper for clamp_min_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_clamp_min_f64_triple_paths(values: Vec<f64>, min: f64, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    clamp_min_f64(&mut scalar_values, min).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    clamp_min_f64(&mut simd_values, min).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_values, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    clamp_min_f64(&mut gpu_values, min).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_values[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_values[i]
      );
    }
  }

  #[test]
  fn test_clamp_min_f64() {
    // Basic minimum clamping
    test_clamp_min_f64_triple_paths(
      vec![-2.0, -1.0, 0.0, 1.0, 2.0],
      0.0,
      vec![0.0, 0.0, 0.0, 1.0, 2.0],
    );

    // Test with all values below minimum
    let mut all_below = vec![-10.0, -5.0, -1.0];
    clamp_min_f64(&mut all_below, 0.0).unwrap();
    assert_eq!(all_below, vec![0.0, 0.0, 0.0]);

    // Test with NaN (should remain NaN)
    let mut with_nan = vec![-5.0, f64::NAN, 5.0];
    clamp_min_f64(&mut with_nan, 0.0).unwrap();
    assert_eq!(with_nan[0], 0.0);
    assert!(with_nan[1].is_nan());
    assert_eq!(with_nan[2], 5.0);

    // Test with negative infinity
    let mut with_neg_inf = vec![f64::NEG_INFINITY, 0.0, 10.0];
    clamp_min_f64(&mut with_neg_inf, 5.0).unwrap();
    assert_eq!(with_neg_inf, vec![5.0, 5.0, 10.0]);
  }

  /// Test helper for clamp_max_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_clamp_max_f64_triple_paths(values: Vec<f64>, max: f64, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    clamp_max_f64(&mut scalar_values, max).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    clamp_max_f64(&mut simd_values, max).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_values, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    clamp_max_f64(&mut gpu_values, max).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_values[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_values[i]
      );
    }
  }

  #[test]
  fn test_clamp_max_f64() {
    // Basic maximum clamping
    test_clamp_max_f64_triple_paths(
      vec![-2.0, -1.0, 0.0, 1.0, 2.0],
      0.0,
      vec![-2.0, -1.0, 0.0, 0.0, 0.0],
    );

    // Test with all values above maximum
    let mut all_above = vec![10.0, 5.0, 1.0];
    clamp_max_f64(&mut all_above, 0.0).unwrap();
    assert_eq!(all_above, vec![0.0, 0.0, 0.0]);

    // Test with NaN (should remain NaN)
    let mut with_nan = vec![-5.0, f64::NAN, 5.0];
    clamp_max_f64(&mut with_nan, 0.0).unwrap();
    assert_eq!(with_nan[0], -5.0);
    assert!(with_nan[1].is_nan());
    assert_eq!(with_nan[2], 0.0);

    // Test with positive infinity
    let mut with_inf = vec![f64::INFINITY, 0.0, -10.0];
    clamp_max_f64(&mut with_inf, 5.0).unwrap();
    assert_eq!(with_inf, vec![5.0, 0.0, -10.0]);
  }

  /// Test helper for deg_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_deg_f64_triple_paths(values: Vec<f64>, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    deg_f64(&mut scalar_values).unwrap();
    for (i, (result, exp)) in scalar_values.iter().zip(expected.iter()).enumerate() {
      assert!(
        (result - exp).abs() < 1e-10,
        "SCALAR path failed at index {} - expected {}, got {}",
        i,
        exp,
        result
      );
    }

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    deg_f64(&mut simd_values).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    for (i, (result, exp)) in simd_values.iter().zip(simd_expected.iter()).enumerate() {
      assert!(
        (result - exp).abs() < 1e-10,
        "SIMD path failed at index {} - expected {}, got {}",
        i,
        exp,
        result
      );
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    deg_f64(&mut gpu_values).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_values[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_values[i]
      );
    }
  }

  #[test]
  fn test_deg_f64() {
    use std::f64::consts::PI;
    // Test common radian to degree conversions
    test_deg_f64_triple_paths(
      vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI],
      vec![0.0, 90.0, 180.0, 270.0, 360.0],
    );

    // Test negative angles
    let mut negative = vec![-PI, -PI / 2.0, -PI / 4.0];
    deg_f64(&mut negative).unwrap();
    assert!((negative[0] - (-180.0)).abs() < 1e-10);
    assert!((negative[1] - (-90.0)).abs() < 1e-10);
    assert!((negative[2] - (-45.0)).abs() < 1e-10);

    // Test with NaN and infinity
    let mut special = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    deg_f64(&mut special).unwrap();
    assert!(special[0].is_nan());
    assert!(special[1].is_infinite() && special[1].is_sign_positive());
    assert!(special[2].is_infinite() && special[2].is_sign_negative());

    // Test empty array
    let mut empty: Vec<f64> = vec![];
    deg_f64(&mut empty).unwrap();
    assert_eq!(empty, Vec::<f64>::new());
  }

  /// Test helper for rad_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_rad_f64_triple_paths(values: Vec<f64>, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    rad_f64(&mut scalar_values).unwrap();
    for (i, (result, exp)) in scalar_values.iter().zip(expected.iter()).enumerate() {
      assert!(
        (result - exp).abs() < 1e-10,
        "SCALAR path failed at index {} - expected {}, got {}",
        i,
        exp,
        result
      );
    }

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    rad_f64(&mut simd_values).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    for (i, (result, exp)) in simd_values.iter().zip(simd_expected.iter()).enumerate() {
      assert!(
        (result - exp).abs() < 1e-10,
        "SIMD path failed at index {} - expected {}, got {}",
        i,
        exp,
        result
      );
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    rad_f64(&mut gpu_values).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_values[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_values[i]
      );
    }
  }

  #[test]
  fn test_rad_f64() {
    use std::f64::consts::PI;
    // Test common degree to radian conversions
    test_rad_f64_triple_paths(
      vec![0.0, 90.0, 180.0, 270.0, 360.0],
      vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI],
    );

    // Test negative angles
    let mut negative = vec![-180.0, -90.0, -45.0];
    rad_f64(&mut negative).unwrap();
    assert!((negative[0] - (-PI)).abs() < 1e-10);
    assert!((negative[1] - (-PI / 2.0)).abs() < 1e-10);
    assert!((negative[2] - (-PI / 4.0)).abs() < 1e-10);

    // Test with NaN and infinity
    let mut special = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    rad_f64(&mut special).unwrap();
    assert!(special[0].is_nan());
    assert!(special[1].is_infinite() && special[1].is_sign_positive());
    assert!(special[2].is_infinite() && special[2].is_sign_negative());
  }

  /// Test helper for log2_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_log2_f64_triple_paths(values: Vec<f64>, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    log2_f64(&mut scalar_values).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    log2_f64(&mut simd_values).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    for (i, (result, exp)) in simd_values.iter().zip(simd_expected.iter()).enumerate() {
      assert!(
        (result - exp).abs() < 1e-10,
        "SIMD path failed at index {} - expected {}, got {}",
        i,
        exp,
        result
      );
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    log2_f64(&mut gpu_values).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_values[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_values[i]
      );
    }
  }

  #[test]
  fn test_log2_f64() {
    // Test powers of 2
    test_log2_f64_triple_paths(
      vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
      vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    );

    // Test non-power of 2 values
    let mut values = vec![3.0, 5.0, 10.0];
    log2_f64(&mut values).unwrap();
    assert!((values[0] - 3.0_f64.log2()).abs() < 1e-10);
    assert!((values[1] - 5.0_f64.log2()).abs() < 1e-10);
    assert!((values[2] - 10.0_f64.log2()).abs() < 1e-10);

    // Test special values
    let mut special = vec![0.0, -1.0, f64::INFINITY];
    log2_f64(&mut special).unwrap();
    assert!(special[0].is_infinite() && special[0].is_sign_negative()); // log2(0) = -inf
    assert!(special[1].is_nan()); // log2(-1) = NaN
    assert!(special[2].is_infinite() && special[2].is_sign_positive()); // log2(inf) = inf

    // Test fractional values
    let mut fractions = vec![0.5, 0.25, 0.125];
    log2_f64(&mut fractions).unwrap();
    assert_eq!(fractions, vec![-1.0, -2.0, -3.0]);
  }

  /// Test helper for log10_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_log10_f64_triple_paths(values: Vec<f64>, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    log10_f64(&mut scalar_values).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    log10_f64(&mut simd_values).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    for (i, (result, exp)) in simd_values.iter().zip(simd_expected.iter()).enumerate() {
      assert!(
        (result - exp).abs() < 1e-10,
        "SIMD path failed at index {} - expected {}, got {}",
        i,
        exp,
        result
      );
    }

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    log10_f64(&mut gpu_values).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert!(
        (gpu_values[i] - gpu_expected[i]).abs() < 1e-10,
        "GPU path failed at index {} - expected {}, got {}",
        i,
        gpu_expected[i],
        gpu_values[i]
      );
    }
  }

  #[test]
  fn test_log10_f64() {
    // Test powers of 10
    test_log10_f64_triple_paths(
      vec![1.0, 10.0, 100.0, 1000.0, 10000.0],
      vec![0.0, 1.0, 2.0, 3.0, 4.0],
    );

    // Test non-power of 10 values
    let mut values = vec![2.0, 5.0, 50.0];
    log10_f64(&mut values).unwrap();
    assert!((values[0] - 2.0_f64.log10()).abs() < 1e-10);
    assert!((values[1] - 5.0_f64.log10()).abs() < 1e-10);
    assert!((values[2] - 50.0_f64.log10()).abs() < 1e-10);

    // Test special values
    let mut special = vec![0.0, -1.0, f64::INFINITY];
    log10_f64(&mut special).unwrap();
    assert!(special[0].is_infinite() && special[0].is_sign_negative()); // log10(0) = -inf
    assert!(special[1].is_nan()); // log10(-1) = NaN
    assert!(special[2].is_infinite() && special[2].is_sign_positive()); // log10(inf) = inf

    // Test fractional values
    let mut fractions = vec![0.1, 0.01, 0.001];
    log10_f64(&mut fractions).unwrap();
    assert_eq!(fractions, vec![-1.0, -2.0, -3.0]);
  }

  /// Test helper for sgn_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_sgn_f64_triple_paths(values: Vec<f64>, expected: Vec<f64>) {
    //  SCALAR PATH TEST (small array - below threshold)
    let mut scalar_values = values.clone();
    sgn_f64(&mut scalar_values).unwrap();
    assert_eq!(
      scalar_values, expected,
      "SCALAR path failed - expected {:?}, got {:?}",
      expected, scalar_values
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let mut simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    sgn_f64(&mut simd_values).unwrap();
    let simd_expected: Vec<f64> = expected.iter().cycle().take(100).cloned().collect();
    assert_eq!(simd_values, simd_expected, "SIMD path failed");

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let mut gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    sgn_f64(&mut gpu_values).unwrap();
    let gpu_expected: Vec<f64> = expected.iter().cycle().take(10000).cloned().collect();
    // Spot check instead of full comparison
    for i in (0..10000).step_by(1000) {
      assert_eq!(
        gpu_values[i], gpu_expected[i],
        "GPU path failed at index {} - expected {}, got {}",
        i, gpu_expected[i], gpu_values[i]
      );
    }
  }

  #[test]
  fn test_sgn_f64() {
    // Test positive values
    test_sgn_f64_triple_paths(
      vec![1.0, 5.5, 100.0, f64::INFINITY],
      vec![1.0, 1.0, 1.0, 1.0],
    );

    // Test negative values
    let mut negative = vec![-1.0, -5.5, -100.0, f64::NEG_INFINITY];
    sgn_f64(&mut negative).unwrap();
    assert_eq!(negative, vec![-1.0, -1.0, -1.0, -1.0]);

    // Test zero (both positive and negative)
    let mut zeros = vec![0.0, -0.0];
    sgn_f64(&mut zeros).unwrap();
    assert_eq!(zeros, vec![0.0, 0.0]);

    // Test NaN (should remain NaN)
    let mut with_nan = vec![f64::NAN, 5.0, -5.0, 0.0];
    sgn_f64(&mut with_nan).unwrap();
    assert!(with_nan[0].is_nan());
    assert_eq!(with_nan[1], 1.0);
    assert_eq!(with_nan[2], -1.0);
    assert_eq!(with_nan[3], 0.0);

    // Test mixed values
    let mut mixed = vec![-10.0, -0.5, 0.0, 0.5, 10.0];
    sgn_f64(&mut mixed).unwrap();
    assert_eq!(mixed, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);

    // Test empty array
    let mut empty: Vec<f64> = vec![];
    sgn_f64(&mut empty).unwrap();
    assert_eq!(empty, Vec::<f64>::new());

    // Test large array for SIMD path
    let mut large_values: Vec<f64> = (0..1000)
      .map(|i| {
        if i % 3 == 0 {
          -(i as f64)
        } else if i % 3 == 1 {
          i as f64
        } else {
          0.0
        }
      })
      .collect();

    let expected_large: Vec<f64> = (0..1000)
      .map(|i| {
        if i % 3 == 0 {
          if i == 0 { 0.0 } else { -1.0 }
        } else if i % 3 == 1 {
          1.0
        } else {
          0.0
        }
      })
      .collect();
    sgn_f64(&mut large_values).unwrap();

    for i in 0..large_values.len() {
      assert!(
        (large_values[i] - expected_large[i]).abs() < 1e-10,
        "sgn large array failed at index {}: expected {}, got {}",
        i,
        expected_large[i],
        large_values[i]
      );
    }
  }

  /// Test helper for stdvar_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_stdvar_f64_triple_paths(values: Vec<f64>, expected: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let result = stdvar_f64(&values).unwrap();
    assert!(
      (result - expected).abs() < 1e-10,
      "SCALAR path failed - expected {}, got {}",
      expected,
      result
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let simd_result = stdvar_f64(&simd_values).unwrap();
    // The variance should be the same for cycled data
    assert!(
      (simd_result - expected).abs() < 1e-8,
      "SIMD path failed - expected {}, got {}",
      expected,
      simd_result
    );

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    let gpu_result = stdvar_f64(&gpu_values).unwrap();
    assert!(
      (gpu_result - expected).abs() < 1e-6,
      "GPU path failed - expected {}, got {}",
      expected,
      gpu_result
    );
  }

  #[test]
  fn test_stdvar_f64() {
    // Basic test - Triple path testing
    // For values [1, 2, 3, 4, 5]: mean = 3, variance = (1+1+0+1+4)/5 = 1.4
    let expected_var = 2.0; // ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = (4+1+0+1+4)/5 = 2.0
    test_stdvar_f64_triple_paths(vec![1.0, 2.0, 3.0, 4.0, 5.0], expected_var);

    // Empty array should return NaN
    let empty: Vec<f64> = vec![];
    let result_empty = stdvar_f64(&empty).unwrap();
    assert!(result_empty.is_nan(), "stdvar of empty array should be NaN");

    // Single value should have variance 0
    let single = vec![42.0];
    let result_single = stdvar_f64(&single).unwrap();
    assert!(
      result_single.abs() < 1e-10,
      "stdvar of single value should be 0, got {}",
      result_single
    );

    // Test with NaN values (should skip NaN)
    let with_nan = vec![1.0, f64::NAN, 3.0, 5.0, f64::NAN];
    let result_nan = stdvar_f64(&with_nan).unwrap();
    // Mean of [1, 3, 5] = 3, variance = ((1-3)^2 + (3-3)^2 + (5-3)^2)/3 = (4+0+4)/3 = 8/3 (~2.6667)
    let expected_nan = 8.0 / 3.0;
    assert!(
      (result_nan - expected_nan).abs() < 1e-10,
      "Expected {}, got {}",
      expected_nan,
      result_nan
    );

    // Test large array for SIMD/GPU paths
    let large_values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let large_result = stdvar_f64(&large_values).unwrap();
    // For 1..=100, mean = 50.5, variance = sum((i - 50.5)^2)/100 for i in 1..=100
    // This is (100^2 - 1) / 12 = 833.25 for uniform distribution
    let expected_large = 833.25;
    assert!(
      (large_result - expected_large).abs() < 0.1,
      "Large array stdvar expected {}, got {}",
      expected_large,
      large_result
    );

    // Test all NaN should return NaN
    let all_nan = vec![f64::NAN, f64::NAN, f64::NAN];
    let result_all_nan = stdvar_f64(&all_nan).unwrap();
    assert!(result_all_nan.is_nan(), "stdvar of all NaN should be NaN");

    // Test constant values (should have variance 0)
    let constant = vec![7.0, 7.0, 7.0, 7.0];
    let result_const = stdvar_f64(&constant).unwrap();
    assert!(
      result_const.abs() < 1e-10,
      "stdvar of constant values should be 0, got {}",
      result_const
    );
  }

  // DELETED stddev test - stddev is now just sqrt(stdvar) computed outside HWX
  // Users can calculate: let stddev = stdvar_f64(&values)?.sqrt();

  /// Test helper for avg_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_avg_f64_triple_paths(values: Vec<f64>, expected: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let result = avg_f64(&values).unwrap();
    assert!(
      (result - expected).abs() < 1e-10,
      "SCALAR path failed - expected {}, got {}",
      expected,
      result
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let simd_result = avg_f64(&simd_values).unwrap();
    // The average should be the same for cycled data
    assert!(
      (simd_result - expected).abs() < 1e-10,
      "SIMD path failed - expected {}, got {}",
      expected,
      simd_result
    );

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    let gpu_result = avg_f64(&gpu_values).unwrap();
    assert!(
      (gpu_result - expected).abs() < 1e-8,
      "GPU path failed - expected {}, got {}",
      expected,
      gpu_result
    );
  }

  // Test avg implementation - Triple path testing
  #[test]
  fn test_avg_f64() {
    // Basic test
    test_avg_f64_triple_paths(
      vec![1.0, 2.0, 3.0, 4.0, 5.0],
      3.0, // Average of [1,2,3,4,5] should be 3
    );

    // Empty array should return NaN
    let empty: Vec<f64> = vec![];
    let result_empty = avg_f64(&empty).unwrap();
    assert!(
      result_empty.is_nan(),
      "Average of empty array should be NaN"
    );

    // Single value
    let single = vec![42.0];
    let result_single = avg_f64(&single).unwrap();
    assert_eq!(
      result_single, 42.0,
      "Average of single value should be the value"
    );

    // Test with NaN values (should return NaN)
    let with_nan = vec![1.0, f64::NAN, 3.0];
    let result_nan = avg_f64(&with_nan).unwrap();
    assert!(result_nan.is_nan(), "Average with NaN should be NaN");

    // Test large array for SIMD/GPU paths
    let large_values: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
    let large_result = avg_f64(&large_values).unwrap();
    assert_eq!(large_result, 500.5, "Average of 1..=1000 should be 500.5");

    // Test with negative values
    let negative = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
    let result_neg = avg_f64(&negative).unwrap();
    assert_eq!(result_neg, 0.0, "Average should handle negative values");

    // Test with infinity
    let with_inf = vec![1.0, 2.0, f64::INFINITY];
    let result_inf = avg_f64(&with_inf).unwrap();
    assert!(
      result_inf.is_infinite() && result_inf.is_sign_positive(),
      "Average with infinity should be infinity"
    );

    // Test with negative infinity
    let with_neg_inf = vec![1.0, 2.0, f64::NEG_INFINITY];
    let result_neg_inf = avg_f64(&with_neg_inf).unwrap();
    assert!(
      result_neg_inf.is_infinite() && result_neg_inf.is_sign_negative(),
      "Average with negative infinity should be negative infinity"
    );

    // Test precision with known values
    let precision = vec![0.1, 0.2, 0.3];
    let result_prec = avg_f64(&precision).unwrap();
    let expected = 0.2;
    assert!(
      (result_prec - expected).abs() < 1e-10,
      "Average of [0.1,0.2,0.3] expected {}, got {}",
      expected,
      result_prec
    );
  }

  /// Test helper for present_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_present_f64_triple_paths(values: Vec<f64>, expected: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let result = present_f64(&values).unwrap();
    assert_eq!(
      result, expected,
      "SCALAR path failed - expected {}, got {}",
      expected, result
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let simd_result = present_f64(&simd_values).unwrap();
    assert_eq!(
      simd_result, expected,
      "SIMD path failed - expected {}, got {}",
      expected, simd_result
    );

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    let gpu_result = present_f64(&gpu_values).unwrap();
    assert_eq!(
      gpu_result, expected,
      "GPU path failed - expected {}, got {}",
      expected, gpu_result
    );
  }

  // Test present_over_time implementation - Triple path testing
  #[test]
  fn test_present_f64() {
    // Basic test - has data
    test_present_f64_triple_paths(
      vec![1.0, 2.0, 3.0, 4.0, 5.0],
      1.0, // present_over_time with data should return 1.0
    );

    // Empty array should return 0.0
    let empty: Vec<f64> = vec![];
    let result_empty = present_f64(&empty).unwrap();
    assert_eq!(
      result_empty, 0.0,
      "present_over_time of empty array should return 0.0"
    );

    // All NaN should return 0.0
    let all_nan = vec![f64::NAN, f64::NAN, f64::NAN];
    let result_nan = present_f64(&all_nan).unwrap();
    assert_eq!(
      result_nan, 0.0,
      "present_over_time of all NaN should return 0.0"
    );

    // Mix of NaN and values should return 1.0
    let mixed = vec![f64::NAN, f64::NAN, 42.0, f64::NAN];
    let result_mixed = present_f64(&mixed).unwrap();
    assert_eq!(
      result_mixed, 1.0,
      "present_over_time with any non-NaN should return 1.0"
    );

    // Single non-NaN value should return 1.0
    let single = vec![42.0];
    let result_single = present_f64(&single).unwrap();
    assert_eq!(
      result_single, 1.0,
      "present_over_time with single value should return 1.0"
    );

    // Single NaN should return 0.0
    let single_nan = vec![f64::NAN];
    let result_single_nan = present_f64(&single_nan).unwrap();
    assert_eq!(
      result_single_nan, 0.0,
      "present_over_time with single NaN should return 0.0"
    );

    // Test large array to trigger SIMD paths
    let large_values: Vec<f64> = (0..5000)
      .map(|i| if i == 2500 { 42.0 } else { f64::NAN })
      .collect();
    let result_large = present_f64(&large_values).unwrap();
    assert_eq!(
      result_large, 1.0,
      "Large array present_over_time with one value should return 1.0"
    );

    // Test large array all NaN to trigger SIMD paths
    let large_nan: Vec<f64> = (0..5000).map(|_| f64::NAN).collect();
    let result_large_nan = present_f64(&large_nan).unwrap();
    assert_eq!(
      result_large_nan, 0.0,
      "Large array present_over_time all NaN should return 0.0"
    );

    // Test with infinity and -infinity (should count as present)
    let with_inf = vec![f64::NAN, f64::INFINITY, f64::NAN];
    let result_inf = present_f64(&with_inf).unwrap();
    assert_eq!(
      result_inf, 1.0,
      "present_over_time with infinity should return 1.0"
    );

    let with_neg_inf = vec![f64::NAN, f64::NEG_INFINITY, f64::NAN];
    let result_neg_inf = present_f64(&with_neg_inf).unwrap();
    assert_eq!(
      result_neg_inf, 1.0,
      "present_over_time with negative infinity should return 1.0"
    );
  }

  /// Test helper for mad_f64 - tests SCALAR, SIMD, and GPU paths
  fn test_mad_f64_triple_paths(values: Vec<f64>, expected: f64) {
    //  SCALAR PATH TEST (small array - below threshold)
    let result = mad_f64(&values).unwrap();
    assert!(
      (result - expected).abs() < 1e-10,
      "SCALAR path failed - expected {}, got {}",
      expected,
      result
    );

    //  SIMD PATH TEST (medium array - above SIMD threshold)
    let simd_values: Vec<f64> = values.iter().cycle().take(100).cloned().collect();
    let simd_result = mad_f64(&simd_values).unwrap();
    assert!(
      (simd_result - expected).abs() < 1e-8,
      "SIMD path failed - expected {}, got {}",
      expected,
      simd_result
    );

    // ^ GPU PATH TEST (massive array - GPU threshold)
    let gpu_values: Vec<f64> = values.iter().cycle().take(10000).cloned().collect();
    let gpu_result = mad_f64(&gpu_values).unwrap();
    assert!(
      (gpu_result - expected).abs() < 1e-6,
      "GPU path failed - expected {}, got {}",
      expected,
      gpu_result
    );
  }

  // Test mad_f64 implementation - Triple path testing
  #[test]
  fn test_mad_f64() {
    // Basic test - known MAD calculation
    // Values: [1, 2, 3, 4, 5], mean = 3
    // Absolute deviations: [2, 1, 0, 1, 2]
    // MAD = (2 + 1 + 0 + 1 + 2) / 5 = 6/5 = 1.2
    test_mad_f64_triple_paths(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1.2);

    let result = mad_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert!(
      (result - 1.2).abs() < 1e-10,
      "Expected MAD of 1.2, got {}",
      result
    );

    // Empty array should return NaN
    let empty: Vec<f64> = vec![];
    let result_empty = mad_f64(&empty).unwrap();
    assert!(result_empty.is_nan(), "MAD of empty array should be NaN");

    // Single value should return 0.0 (no deviation from itself)
    let single = vec![42.0];
    let result_single = mad_f64(&single).unwrap();
    assert_eq!(result_single, 0.0, "MAD of single value should be 0.0");

    // All same values should return 0.0
    let same = vec![7.0, 7.0, 7.0, 7.0];
    let result_same = mad_f64(&same).unwrap();
    assert_eq!(result_same, 0.0, "MAD of identical values should be 0.0");

    // Test with NaN values (should skip NaN)
    // Values: [1, NaN, 3, 5, NaN] -> [1, 3, 5], mean = 3
    // Absolute deviations: [2, 0, 2]
    // MAD = (2 + 0 + 2) / 3 = 4/3 (~1.3333)
    let with_nan = vec![1.0, f64::NAN, 3.0, 5.0, f64::NAN];
    let result_nan = mad_f64(&with_nan).unwrap();
    let expected_mad = 4.0 / 3.0;
    assert!(
      (result_nan - expected_mad).abs() < 1e-10,
      "MAD should skip NaN values, expected {}, got {}",
      expected_mad,
      result_nan
    );

    // Test all NaN should return NaN
    let all_nan = vec![f64::NAN, f64::NAN, f64::NAN];
    let result_all_nan = mad_f64(&all_nan).unwrap();
    assert!(result_all_nan.is_nan(), "MAD of all NaN should be NaN");

    // Test with infinity - should include in calculation
    let with_inf = vec![1.0, 2.0, 3.0, f64::INFINITY];
    let result_inf = mad_f64(&with_inf).unwrap();
    assert!(
      result_inf.is_infinite(),
      "MAD with infinity should be infinity, got {}",
      result_inf
    );

    // Test large array for SIMD/GPU paths
    let large_values: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
    let result_large = mad_f64(&large_values).unwrap();
    // For 1..=1000, mean = 500.5, MAD = average(|i - 500.5|) for i in 1..=1000
    // This should be around 250 for uniform distribution
    assert!(
      (result_large - 250.0).abs() < 1.0,
      "Large array MAD expected ~250, got {}",
      result_large
    );

    // Test precision with small deviations
    let precision = vec![1.001, 1.002, 1.003, 1.004, 1.005];
    let result_prec = mad_f64(&precision).unwrap();
    // Mean = 1.003, MAD = (0.002 + 0.001 + 0 + 0.001 + 0.002) / 5 = 0.0012
    assert!(
      (result_prec - 0.0012).abs() < 1e-10,
      "Precision MAD expected 0.0012, got {}",
      result_prec
    );
  }
} // End of mod tests
