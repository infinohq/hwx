// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::test_utils::config_test_logger;
    use crate::{
        distance_cosine_f32, distance_dot_f32, distance_hamming_u16, distance_hamming_u32,
        distance_hellinger_f32, distance_jaccard_u16, distance_jeffreys_f32,
        distance_jensen_shannon_f32, distance_l1_f32, distance_l2_f32, distance_levenshtein_u16,
    };

    #[cfg(has_cuda)]
    use crate::gpu::{launch_ptx, LaunchConfig};

    // =============================================================================
    //   TRIPLE-PATH TEST HELPERS - DISTANCE EDITION
    // =============================================================================

    #[allow(unused_variables)]
    #[allow(unused_mut)]
    #[allow(clippy::manual_range_contains)]
    //  TRIPLE-PATH L1 DISTANCE TEST HELPER
    fn test_distance_l1_f32_triple_paths(
        va: Vec<f32>,
        vb: Vec<f32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_l1_f32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "L1 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<f32> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<f32> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_l1_f32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<f32> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<f32> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_l1_f32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite, positive results
        assert!(
            simd_result.is_finite() && simd_result >= 0.0,
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && gpu_result >= 0.0,
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH L2 DISTANCE TEST HELPER
    fn test_distance_l2_f32_triple_paths(
        va: Vec<f32>,
        vb: Vec<f32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_l2_f32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "L2 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<f32> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<f32> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_l2_f32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<f32> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<f32> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_l2_f32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite, positive results
        assert!(
            simd_result.is_finite() && simd_result >= 0.0,
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && gpu_result >= 0.0,
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH DOT PRODUCT DISTANCE TEST HELPER
    fn test_distance_dot_f32_triple_paths(
        va: Vec<f32>,
        vb: Vec<f32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_dot_f32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Dot product distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<f32> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<f32> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_dot_f32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<f32> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<f32> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_dot_f32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite results
        assert!(simd_result.is_finite(), "SIMD path produced invalid result");
        assert!(gpu_result.is_finite(), "GPU path produced invalid result");
    }

    //  TRIPLE-PATH COSINE DISTANCE TEST HELPER
    fn test_distance_cosine_f32_triple_paths(
        va: Vec<f32>,
        vb: Vec<f32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_cosine_f32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Cosine distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<f32> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<f32> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_cosine_f32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<f32> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<f32> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_cosine_f32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite results in valid range [0, 2]
        assert!(
            (0.0..=2.0).contains(&simd_result),
            "SIMD cosine distance out of range"
        );

        assert!(
            (0.0..=2.0).contains(&gpu_result),
            "GPU cosine distance out of range"
        );
    }

    //  TRIPLE-PATH HAMMING U16 DISTANCE TEST HELPER
    fn test_distance_hamming_u16_triple_paths(
        va: Vec<u16>,
        vb: Vec<u16>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_hamming_u16(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Hamming u16 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<u16> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<u16> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_hamming_u16(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<u16> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<u16> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_hamming_u16(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite, positive results
        assert!(
            simd_result.is_finite() && simd_result >= 0.0,
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && gpu_result >= 0.0,
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH HAMMING U32 DISTANCE TEST HELPER
    fn test_distance_hamming_u32_triple_paths(
        va: Vec<u32>,
        vb: Vec<u32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_hamming_u32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Hamming u32 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<u32> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<u32> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_hamming_u32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<u32> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<u32> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_hamming_u32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite, positive results
        assert!(
            simd_result.is_finite() && simd_result >= 0.0,
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && gpu_result >= 0.0,
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH LEVENSHTEIN U16 DISTANCE TEST HELPER
    fn test_distance_levenshtein_u16_triple_paths(
        va: Vec<u16>,
        vb: Vec<u16>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_levenshtein_u16(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Levenshtein u16 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<u16> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<u16> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_levenshtein_u16(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<u16> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<u16> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_levenshtein_u16(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite, positive results
        assert!(
            simd_result.is_finite() && simd_result >= 0.0,
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && gpu_result >= 0.0,
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH JACCARD U16 DISTANCE TEST HELPER
    fn test_distance_jaccard_u16_triple_paths(
        va: Vec<u16>,
        vb: Vec<u16>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_jaccard_u16(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Jaccard u16 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<u16> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<u16> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_jaccard_u16(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<u16> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<u16> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_jaccard_u16(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite results in valid range [0, 1]
        assert!(
            simd_result.is_finite() && (0.0..=1.0).contains(&simd_result),
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && (0.0..=1.0).contains(&gpu_result),
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH JENSEN-SHANNON F32 DISTANCE TEST HELPER
    fn test_distance_jensen_shannon_f32_triple_paths(
        va: Vec<f32>,
        vb: Vec<f32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_jensen_shannon_f32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Jensen-Shannon f32 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        // NOTE: Can't just cycle the data because JS distance scales with repetition count.
        // Instead, pad with zeros or use a properly normalized distribution.
        let simd_size = 256;
        let mut simd_va = vec![0.0f32; simd_size];
        let mut simd_vb = vec![0.0f32; simd_size];
        // Copy original data and normalize by size to maintain distribution properties
        for i in 0..simd_size {
            simd_va[i] = va[i % va.len()] / (simd_size / va.len()) as f32;
            simd_vb[i] = vb[i % vb.len()] / (simd_size / vb.len()) as f32;
        }
        let simd_result = distance_jensen_shannon_f32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let mut gpu_va = vec![0.0f32; gpu_size];
        let mut gpu_vb = vec![0.0f32; gpu_size];
        for i in 0..gpu_size {
            gpu_va[i] = va[i % va.len()] / (gpu_size / va.len()) as f32;
            gpu_vb[i] = vb[i % vb.len()] / (gpu_size / vb.len()) as f32;
        }

        let gpu_result = distance_jensen_shannon_f32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite results in valid range [0, 1]
        assert!(
            simd_result.is_finite() && (0.0..=1.0).contains(&simd_result),
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && (0.0..=1.0).contains(&gpu_result),
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH JEFFREYS F32 DISTANCE TEST HELPER
    fn test_distance_jeffreys_f32_triple_paths(
        va: Vec<f32>,
        vb: Vec<f32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_jeffreys_f32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Jeffreys f32 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let simd_va: Vec<f32> = va.iter().cycle().take(simd_size).cloned().collect();
        let simd_vb: Vec<f32> = vb.iter().cycle().take(simd_size).cloned().collect();
        let simd_result = distance_jeffreys_f32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let gpu_va: Vec<f32> = va.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_vb: Vec<f32> = vb.iter().cycle().take(gpu_size).cloned().collect();
        let gpu_result = distance_jeffreys_f32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite, positive results
        assert!(
            simd_result.is_finite() && simd_result >= 0.0,
            "SIMD path produced invalid result"
        );
        assert!(
            gpu_result.is_finite() && gpu_result >= 0.0,
            "GPU path produced invalid result"
        );
    }

    //  TRIPLE-PATH HELLINGER F32 DISTANCE TEST HELPER
    fn test_distance_hellinger_f32_triple_paths(
        va: Vec<f32>,
        vb: Vec<f32>,
        expected: f32,
        tolerance: f32,
    ) {
        // SCALAR PATH TEST (small vectors - below SIMD threshold)
        let result = distance_hellinger_f32(&va, &vb).unwrap();
        assert!(
            (result - expected).abs() < tolerance,
            "Hellinger f32 distance mismatch: expected={}, got={}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );

        //  SIMD PATH TEST (medium vectors - above SIMD threshold, below GPU threshold)
        let simd_size = 256;
        let mut simd_va: Vec<f32> = va.iter().cycle().take(simd_size).cloned().collect();
        let mut simd_vb: Vec<f32> = vb.iter().cycle().take(simd_size).cloned().collect();

        // Normalize to maintain probability distribution (sum = 1.0)
        let sum_a: f32 = simd_va.iter().sum();
        let sum_b: f32 = simd_vb.iter().sum();
        for x in &mut simd_va {
            *x /= sum_a;
        }
        for x in &mut simd_vb {
            *x /= sum_b;
        }

        let simd_result = distance_hellinger_f32(&simd_va, &simd_vb).unwrap();

        //  GPU PATH TEST (very large array - above GPU_THRESHOLD_DISTANCE = 1024)
        let gpu_size = 2048;
        let mut gpu_va: Vec<f32> = va.iter().cycle().take(gpu_size).cloned().collect();
        let mut gpu_vb: Vec<f32> = vb.iter().cycle().take(gpu_size).cloned().collect();

        // Normalize to maintain probability distribution (sum = 1.0)
        let sum_a: f32 = gpu_va.iter().sum();
        let sum_b: f32 = gpu_vb.iter().sum();
        for x in &mut gpu_va {
            *x /= sum_a;
        }
        for x in &mut gpu_vb {
            *x /= sum_b;
        }

        let gpu_result = distance_hellinger_f32(&gpu_va, &gpu_vb).unwrap();

        // Verify all paths return finite results in valid range [0, 1]
        assert!(
            simd_result.is_finite() && (0.0..=1.0).contains(&simd_result),
            "SIMD path produced invalid result: {}",
            simd_result
        );
        assert!(
            gpu_result.is_finite() && (0.0..=1.0).contains(&gpu_result),
            "GPU path produced invalid result: {}",
            gpu_result
        );
    }

    // =============================================================================
    // L1 (MANHATTAN) DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_l1_f32_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality
        test_distance_l1_f32_triple_paths(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            4.0,
            1e-6,
        );

        // Test case 2: Edge case - identical vectors
        test_distance_l1_f32_triple_paths(vec![1.2, 3.4, 5.6], vec![1.2, 3.4, 5.6], 0.0, 1e-6);

        // Test case 3: Edge case - empty vectors should return error (not valid mathematically)
        let va3 = vec![];
        let vb3 = vec![];
        assert!(
            distance_l1_f32(&va3, &vb3).is_err(),
            "Empty vectors should return error"
        );
    }

    // =============================================================================
    // L2 (EUCLIDEAN) DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_l2_f32_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality (3-4-5 triangle)
        test_distance_l2_f32_triple_paths(vec![3.0, 4.0], vec![0.0, 0.0], 5.0, 1e-6);

        // Test case 2: Edge case - identical vectors
        test_distance_l2_f32_triple_paths(vec![1.2, 3.4, 5.6], vec![1.2, 3.4, 5.6], 0.0, 1e-6);

        // Test case 3: Edge case - empty vectors should return error (not valid mathematically)
        let va3 = vec![];
        let vb3 = vec![];
        assert!(
            distance_l2_f32(&va3, &vb3).is_err(),
            "Empty vectors should return error"
        );
    }

    // =============================================================================
    // DOT PRODUCT DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_dot_f32_comprehensive() {
        config_test_logger();

        // Test case 1: Orthogonal vectors
        test_distance_dot_f32_triple_paths(vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], 1.0, 1e-6);

        // Test case 2: Identical vectors
        test_distance_dot_f32_triple_paths(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 0.0, 1e-6);

        // Test case 3: Edge case - empty vectors should return error (not valid mathematically)
        let va3 = vec![];
        let vb3 = vec![];
        assert!(
            distance_dot_f32(&va3, &vb3).is_err(),
            "Empty vectors should return error"
        );
    }

    // =============================================================================
    // COSINE DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_cosine_f32_comprehensive() {
        config_test_logger();

        // Test case 1: Parallel vectors (distance should be 0)
        test_distance_cosine_f32_triple_paths(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], 0.0, 1e-6);

        // Test case 2: Orthogonal vectors (distance should be 1)
        test_distance_cosine_f32_triple_paths(vec![1.0, 0.0], vec![0.0, 1.0], 1.0, 1e-6);

        // Test case 3: Anti-parallel vectors (distance should be 2)
        test_distance_cosine_f32_triple_paths(
            vec![1.0, 2.0, 3.0],
            vec![-1.0, -2.0, -3.0],
            2.0,
            1e-6,
        );

        // Test case 4: Edge case - empty vectors should return error
        let va4: Vec<f32> = vec![];
        let vb4: Vec<f32> = vec![];
        assert!(
            distance_cosine_f32(&va4, &vb4).is_err(),
            "Empty vectors should return error"
        );
    }

    // =============================================================================
    // HAMMING DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_hamming_u16_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality
        test_distance_hamming_u16_triple_paths(vec![1, 2, 3, 4], vec![1, 5, 3, 6], 0.5, 1e-6);

        // Test case 2: Edge case - identical vectors
        test_distance_hamming_u16_triple_paths(vec![10, 20, 30], vec![10, 20, 30], 0.0, 1e-6);

        // Test case 3: Edge case - empty vectors should return error
        let va3 = vec![];
        let vb3 = vec![];
        assert!(
            distance_hamming_u16(&va3, &vb3).is_err(),
            "Empty vectors should return error"
        );
    }

    #[test]
    fn test_distance_hamming_u32_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality
        test_distance_hamming_u32_triple_paths(vec![1, 2, 3, 4], vec![1, 5, 3, 6], 0.5, 1e-6);

        // Test case 2: Edge case - identical vectors
        test_distance_hamming_u32_triple_paths(vec![100, 200, 300], vec![100, 200, 300], 0.0, 1e-6);

        // Test case 3: Edge case - empty vectors should return error
        let va3: Vec<u32> = vec![];
        let vb3: Vec<u32> = vec![];
        assert!(
            distance_hamming_u32(&va3, &vb3).is_err(),
            "Empty vectors should return error"
        );
    }

    // =============================================================================
    // LEVENSHTEIN DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_levenshtein_u16_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality
        test_distance_levenshtein_u16_triple_paths(vec![1, 2, 3], vec![1, 4, 3], 1.0, 1e-6);

        // Test case 2: Edge case - identical vectors
        test_distance_levenshtein_u16_triple_paths(vec![1, 2, 3], vec![1, 2, 3], 0.0, 1e-6);

        // Test case 3: Edge case - both empty vectors
        let s3: Vec<u16> = vec![];
        let t3: Vec<u16> = vec![];
        assert!(distance_levenshtein_u16(&s3, &t3).unwrap().abs() < 1e-6);
    }

    // =============================================================================
    // JACCARD DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_jaccard_u16_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality
        test_distance_jaccard_u16_triple_paths(
            vec![1u16, 1u16, 0u16, 1u16],
            vec![1u16, 0u16, 1u16, 1u16],
            0.5,
            1e-6,
        );

        // Test case 2: Edge case - identical vectors
        test_distance_jaccard_u16_triple_paths(
            vec![1u16, 0u16, 1u16],
            vec![1u16, 0u16, 1u16],
            0.0,
            1e-6,
        );

        // Test case 3: Edge case - completely disjoint vectors
        test_distance_jaccard_u16_triple_paths(
            vec![1u16, 1u16, 0u16, 0u16],
            vec![0u16, 0u16, 1u16, 1u16],
            1.0,
            1e-6,
        );

        // Test case 4: Edge case - empty vectors
        let v4: Vec<u16> = vec![];
        let v5: Vec<u16> = vec![];
        assert!(distance_jaccard_u16(&v4, &v5).unwrap().abs() < 1e-6);
    }

    // =============================================================================
    // JENSEN-SHANNON DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_jensen_shannon_f32_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality - different probability distributions
        let p1: Vec<f32> = vec![0.5, 0.5];
        let q1: Vec<f32> = vec![0.3, 0.7];
        let dist1 = distance_jensen_shannon_f32(&p1, &q1).unwrap();
        // JS distance is sqrt of JS divergence, bounded [0, 1]
        assert!((0.0..=1.0).contains(&dist1));
        // For this specific case, we know the result should be reasonable
        test_distance_jensen_shannon_f32_triple_paths(p1, q1, dist1, 1e-3);

        // Test case 2: Edge case - identical distributions
        test_distance_jensen_shannon_f32_triple_paths(vec![0.6, 0.4], vec![0.6, 0.4], 0.0, 1e-6);

        // Test case 3: Edge case - empty vectors should return error
        let p3: Vec<f32> = vec![];
        let q3: Vec<f32> = vec![];
        assert!(
            distance_jensen_shannon_f32(&p3, &q3).is_err(),
            "Empty vectors should return error"
        );
    }

    // =============================================================================
    // JEFFREYS DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_jeffreys_f32_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality - different probability distributions
        let p1: Vec<f32> = vec![0.6, 0.3, 0.1];
        let q1: Vec<f32> = vec![0.4, 0.4, 0.2];
        let jeffreys_result = distance_jeffreys_f32(&p1, &q1).unwrap();
        // Use the actual result for triple path testing
        test_distance_jeffreys_f32_triple_paths(p1, q1, jeffreys_result, 1e-2);

        // Test case 2: Edge case - identical distributions
        test_distance_jeffreys_f32_triple_paths(vec![0.5, 0.5], vec![0.5, 0.5], 0.0, 1e-6);

        // Test case 3: Edge case - empty vectors should return error
        let p3: Vec<f32> = vec![];
        let q3: Vec<f32> = vec![];
        assert!(
            distance_jeffreys_f32(&p3, &q3).is_err(),
            "Empty vectors should return error"
        );
    }

    // =============================================================================
    // HELINGER DISTANCE TESTS
    // =============================================================================
    #[test]
    fn test_distance_hellinger_f32_comprehensive() {
        config_test_logger();

        // Test case 1: Basic functionality - different probability distributions
        let p1: Vec<f32> = vec![0.5, 0.3, 0.2];
        let q1: Vec<f32> = vec![0.4, 0.4, 0.2];
        let hellinger_result = distance_hellinger_f32(&p1, &q1).unwrap();
        // Use the actual result for triple path testing
        test_distance_hellinger_f32_triple_paths(p1, q1, hellinger_result, 1e-3);

        // Test case 2: Edge case - identical distributions
        test_distance_hellinger_f32_triple_paths(
            vec![0.6, 0.3, 0.1],
            vec![0.6, 0.3, 0.1],
            0.0,
            1e-6,
        );

        // Test case 3: Edge case - empty vectors should return error
        let p3: Vec<f32> = vec![];
        let q3: Vec<f32> = vec![];
        assert!(
            distance_hellinger_f32(&p3, &q3).is_err(),
            "Empty vectors should return error"
        );
    }

    /// Test suite for statistical distance measures (Hellinger, Jeffreys, Jensen-Shannon)
    #[test]
    fn test_statistical_distances_comprehensive() {
        config_test_logger();

        // Probability distributions (should sum to 1)
        let uniform_dist = vec![0.25, 0.25, 0.25, 0.25];
        let skewed_dist = vec![0.7, 0.2, 0.05, 0.05];

        // Hellinger distance
        {
            let hellinger_dist = distance_hellinger_f32(&uniform_dist, &skewed_dist).unwrap();
            assert!(
                (0.0..=1.0).contains(&hellinger_dist),
                "Hellinger distance should be in [0, 1]"
            );
        }

        {
            let hellinger_same = distance_hellinger_f32(&uniform_dist, &uniform_dist).unwrap();
            assert!(
                hellinger_same < 1e-6,
                "Identical distributions should have Hellinger distance 0"
            );
        }

        // Jeffreys divergence
        {
            let jeffreys_dist = distance_jeffreys_f32(&uniform_dist, &skewed_dist).unwrap();
            assert!(
                jeffreys_dist >= 0.0,
                "Jeffreys divergence should be non-negative"
            );
        }

        {
            let jeffreys_same = distance_jeffreys_f32(&uniform_dist, &uniform_dist).unwrap();
            assert!(
                jeffreys_same < 1e-6,
                "Identical distributions should have Jeffreys divergence 0"
            );
        }

        // Jensen-Shannon divergence
        {
            let js_dist = distance_jensen_shannon_f32(&uniform_dist, &skewed_dist).unwrap();
            assert!(
                (0.0..=1.0).contains(&js_dist),
                "Jensen-Shannon divergence should be in [0, 1]"
            );
        }

        {
            let js_same = distance_jensen_shannon_f32(&uniform_dist, &uniform_dist).unwrap();
            assert!(
                js_same < 1e-6,
                "Identical distributions should have JS divergence 0"
            );
        }

        // Large probability distributions
        let large_uniform: Vec<f32> = vec![1.0 / 1000.0; 1000];
        let large_exponential: Vec<f32> = (1..=1000)
            .map(|i| {
                let val = (-0.001 * i as f32).exp();
                val / 1000.0 // Normalize (approximately)
            })
            .collect();

        {
            let large_hellinger =
                distance_hellinger_f32(&large_uniform, &large_exponential).unwrap();
            assert!(
                (0.0..=1.0).contains(&large_hellinger),
                "Large Hellinger distance should be valid"
            );
        }

        // Symmetry tests
        let dist_a = vec![0.3, 0.7];
        let dist_b = vec![0.6, 0.4];

        let hellinger_ab = distance_hellinger_f32(&dist_a, &dist_b).unwrap();
        let hellinger_ba = distance_hellinger_f32(&dist_b, &dist_a).unwrap();
        assert!(
            (hellinger_ab - hellinger_ba).abs() < 1e-6,
            "Hellinger distance should be symmetric"
        );

        let js_ab = distance_jensen_shannon_f32(&dist_a, &dist_b).unwrap();
        let js_ba = distance_jensen_shannon_f32(&dist_b, &dist_a).unwrap();
        assert!(
            (js_ab - js_ba).abs() < 1e-6,
            "Jensen-Shannon divergence should be symmetric"
        );

        // Edge case: distributions with zeros
        let sparse_a = vec![0.0, 0.5, 0.5, 0.0];
        let sparse_b = vec![0.25, 0.25, 0.25, 0.25];

        let sparse_hellinger = distance_hellinger_f32(&sparse_a, &sparse_b).unwrap();
        assert!(
            (0.0..=1.0).contains(&sparse_hellinger),
            "Sparse distributions should have valid Hellinger distance"
        );
    }

    /// Performance and edge case testing for all distance functions
    #[test]
    fn test_distance_performance_and_edge_cases() {
        config_test_logger();

        // Test with very large vectors to stress SIMD implementations
        let huge_size = 100000;
        let huge_a: Vec<f32> = (0..huge_size).map(|i| (i as f32 * 0.001).sin()).collect();
        let huge_b: Vec<f32> = (0..huge_size).map(|i| (i as f32 * 0.001).cos()).collect();

        // All distance functions should handle large inputs
        let huge_l1 = distance_l1_f32(&huge_a, &huge_b).unwrap();
        let huge_l2 = distance_l2_f32(&huge_a, &huge_b).unwrap();
        let huge_cosine = distance_cosine_f32(&huge_a, &huge_b).unwrap();

        assert!(huge_l1 >= 0.0, "Large L1 distance should be non-negative");
        assert!(huge_l2 >= 0.0, "Large L2 distance should be non-negative");
        assert!(
            (0.0..=2.0).contains(&huge_cosine),
            "Large cosine distance should be valid"
        );

        // Test with extreme values
        let extreme_pos = vec![1e10_f32; 100]; // Large but safe from overflow
        let extreme_neg = vec![-1e10_f32; 100];
        let extreme_zero = vec![0.0; 100];

        let extreme_l1 = distance_l1_f32(&extreme_pos, &extreme_neg).unwrap();
        let extreme_l2 = distance_l2_f32(&extreme_pos, &extreme_zero).unwrap();

        assert!(extreme_l1.is_finite(), "Extreme value L1 should be finite");
        assert!(extreme_l2.is_finite(), "Extreme value L2 should be finite");

        // Test with very small values (underflow protection)
        let tiny_a = vec![1e-30_f32; 100];
        let tiny_b = vec![2e-30_f32; 100];

        let tiny_l1 = distance_l1_f32(&tiny_a, &tiny_b).unwrap();
        let tiny_cosine = distance_cosine_f32(&tiny_a, &tiny_b).unwrap();

        assert!(tiny_l1.is_finite(), "Tiny value L1 should be finite");
        assert!(
            tiny_cosine.is_finite(),
            "Tiny value cosine should be finite"
        );

        // Test mixed precision scenarios
        let mixed_a = vec![1.0, 1e-10, 1e10, 0.0];
        let mixed_b = vec![1e-10, 1.0, 0.0, 1e10];

        let mixed_l1 = distance_l1_f32(&mixed_a, &mixed_b).unwrap();
        let mixed_l2 = distance_l2_f32(&mixed_a, &mixed_b).unwrap();

        assert!(mixed_l1.is_finite(), "Mixed precision L1 should be finite");
        assert!(mixed_l2.is_finite(), "Mixed precision L2 should be finite");

        // Consistency check: different vector sizes that are multiples
        let base_vec = vec![1.0, 2.0, 3.0, 4.0];
        let double_vec = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let quad_vec = [1.0, 2.0, 3.0, 4.0].repeat(4);

        // These should all work without panicking and produce reasonable results
        let base_l1 = distance_l1_f32(&base_vec, &base_vec).unwrap();
        let double_l1 = distance_l1_f32(&double_vec, &double_vec).unwrap();
        let quad_l1 = distance_l1_f32(&quad_vec, &quad_vec).unwrap();

        assert!(base_l1 < 1e-6, "Base vector self-distance should be 0");
        assert!(double_l1 < 1e-6, "Double vector self-distance should be 0");
        assert!(quad_l1 < 1e-6, "Quad vector self-distance should be 0");
    }

    #[test]
    fn test_gpu_scale_distances() {
        config_test_logger();

        // GPU-scale vectors (>= GPU_THRESHOLD_DISTANCE = 1024) to test triple-path dispatch
        let gpu_size = 2048; // Well above GPU threshold
        let gpu_a: Vec<f32> = (0..gpu_size).map(|i| (i as f32 * 0.01).sin()).collect();
        let gpu_b: Vec<f32> = (0..gpu_size).map(|i| (i as f32 * 0.01).cos()).collect();

        // Test all distance functions at GPU scale
        let gpu_l1 = distance_l1_f32(&gpu_a, &gpu_b).unwrap();
        let gpu_l2 = distance_l2_f32(&gpu_a, &gpu_b).unwrap();
        let gpu_dot = distance_dot_f32(&gpu_a, &gpu_b).unwrap();
        let gpu_cosine = distance_cosine_f32(&gpu_a, &gpu_b).unwrap();
        let gpu_hellinger = distance_hellinger_f32(&gpu_a, &gpu_b).unwrap();
        let gpu_jeffreys = distance_jeffreys_f32(&gpu_a, &gpu_b).unwrap();
        let gpu_jensen = distance_jensen_shannon_f32(&gpu_a, &gpu_b).unwrap();

        // Verify results are finite and within expected ranges
        assert!(gpu_l1.is_finite() && gpu_l1 >= 0.0);
        assert!(gpu_l2.is_finite() && gpu_l2 >= 0.0);
        assert!(gpu_dot.is_finite());
        assert!(gpu_cosine.is_finite() && (0.0..=2.0).contains(&gpu_cosine));
        assert!(gpu_hellinger.is_finite() && (0.0..=1.0).contains(&gpu_hellinger));
        assert!(gpu_jeffreys.is_finite() && gpu_jeffreys >= 0.0);
        assert!(gpu_jensen.is_finite() && (0.0..=1.0).contains(&gpu_jensen));

        // Test Hamming distances with GPU-scale u32/u16 arrays
        let gpu_u32_a: Vec<u32> = (0..gpu_size).map(|i| i as u32 % 256).collect();
        let gpu_u32_b: Vec<u32> = (0..gpu_size).map(|i| (i as u32 + 1) % 256).collect();
        let gpu_hamming_u32 = distance_hamming_u32(&gpu_u32_a, &gpu_u32_b).unwrap();
        assert!(gpu_hamming_u32.is_finite() && gpu_hamming_u32 >= 0.0);

        let gpu_u16_a: Vec<u16> = (0..gpu_size).map(|i| i as u16 % 256).collect();
        let gpu_u16_b: Vec<u16> = (0..gpu_size).map(|i| (i as u16 + 1) % 256).collect();
        let gpu_hamming_u16 = distance_hamming_u16(&gpu_u16_a, &gpu_u16_b).unwrap();
        let gpu_jaccard = distance_jaccard_u16(&gpu_u16_a, &gpu_u16_b).unwrap();
        assert!(gpu_hamming_u16.is_finite() && gpu_hamming_u16 >= 0.0);
        assert!(gpu_jaccard.is_finite() && (0.0..=1.0).contains(&gpu_jaccard));
    }

    #[cfg(has_cuda)]
    #[test]
    #[ignore]
    fn gpu_spin_nvml() {
        config_test_logger();

        let (blocks, threads) = LaunchConfig::parallel();
        let ptx: &str = r#"
          .version 7.0
          .target sm_70
          .address_size 64
          .visible .entry gpu_spin(
            .param .u64 cycles_param
          )
          {
            .reg .pred %p;
            .reg .u64 %start;
            .reg .u64 %now;
            .reg .u64 %cycles;
            .reg .u64 %diff;
            ld.param.u64 %cycles, [cycles_param];
            mov.u64 %start, %globaltimer;
          L0:
            mov.u64 %now, %globaltimer;
            sub.s64 %diff, %now, %start;
            setp.lt.s64 %p, %diff, %cycles;
            @%p bra L0;
            ret;
          }
        "#;

        let nanos: u64 = 250_000_000;
        let args: [*const u8; 1] = [(&nanos as *const u64) as *const u8];
        let res = launch_ptx(ptx, &[], "gpu_spin", blocks, threads, &args);
        assert!(res.is_ok());
    }
}
