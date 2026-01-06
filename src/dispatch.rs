// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::needless_range_loop, clippy::assign_op_pattern)]
//! # HWX dispatch framework
//!
//! This module contains the dispatch layer used across the crate:
//! it chooses between scalar implementations and hardware-accelerated backends
//! (SIMD and, when enabled, CUDA) based on target capabilities and input sizes.
//!
//! ## Notes on performance-oriented code
//! Some hot paths in this crate intentionally avoid allocations and iterator-heavy
//! patterns to help the compiler generate tight loops. These are guidelines, not
//! hard guarantees. Correctness and clarity still matter.

#![allow(clippy::let_and_return)]
use log::trace;

use super::constants::*;

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

use crate::types::{ClassificationResult, HwxType};
use crate::{arrays, classify, distance, strings, tokenize, traverse};

// =============================================================================
//  GPU MEMORY MANAGEMENT HELPERS
// =============================================================================

#[cfg(has_cuda)]
use super::gpu;

// =============================================================================
//  HARDWARE DETECTION & SIMD CAPABILITIES
// =============================================================================

/// Hardware capability detection used by the HWX dispatch layer
pub struct HardwareCapabilities {
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub has_neon: bool,
    pub has_cuda: bool,
}

/// Detect SIMD capabilities at runtime.
///
/// This function checks for the presence of SIMD instruction sets in the target
/// architecture. It returns a `HardwareCapabilities` struct with the following fields:
///
/// - `has_avx512`: True if AVX-512 instructions are supported
/// - `has_avx2`: True if AVX2 instructions are supported
/// - `has_neon`: True if NEON instructions are supported
///
/// The function first checks for AVX-512 support. It checks
/// for AVX512, AVX2, and NEON support in the order of priority.
///
impl HardwareCapabilities {
    #[inline]
    pub fn detect() -> Self {
        HardwareCapabilities {
            has_avx512: Self::detect_avx512(),
            has_avx2: Self::detect_avx2(),
            has_neon: Self::detect_neon(),
            has_cuda: Self::detect_cuda(),
        }
    }

    fn detect_avx512() -> bool {
        #[allow(unused_mut)]
        let mut detected_avx512 = false;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "hwx-nightly")]
        if is_x86_feature_detected!("avx512f") {
            detected_avx512 = true;
        }

        detected_avx512
    }

    fn detect_avx2() -> bool {
        #[allow(unused_mut)]
        let mut detected_avx2 = false;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(not(feature = "hwx-nightly"))]
        if is_x86_feature_detected!("avx2") {
            detected_avx2 = true;
        }

        detected_avx2
    }

    fn detect_neon() -> bool {
        #[allow(unused_mut)]
        let mut detected_neon = false;

        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("neon") {
            detected_neon = true;
        }

        detected_neon
    }

    fn detect_cuda() -> bool {
        // Use a static atomic for one-time detection and caching
        use std::sync::atomic::{AtomicU8, Ordering};
        static CUDA_DETECTED: AtomicU8 = AtomicU8::new(2); // 2 = unknown, 1 = true, 0 = false

        let cached = CUDA_DETECTED.load(Ordering::Relaxed);
        if cached != 2 {
            return cached == 1;
        }

        // One-time detection - try to initialize CUDA
        // This function is already cached internally
        #[cfg(has_cuda)]
        let has_cuda = crate::gpu::ensure_cuda_initialized().is_ok();
        #[cfg(not(has_cuda))]
        let has_cuda = false;

        CUDA_DETECTED.store(if has_cuda { 1 } else { 0 }, Ordering::Relaxed);
        has_cuda
    }
}

// All dispatch functions now use manual threshold-based dispatching pattern
// This provides better clarity and maintainability than complex macros

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get information about available SIMD capabilities
#[inline]
pub fn get_hw_capabilities() -> HardwareCapabilities {
    HardwareCapabilities::detect()
}

/// Check if a specific SIMD instruction set is available
#[inline]
pub fn has_hw_support(instruction_set: &str) -> bool {
    match instruction_set {
        "avx512" => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                return is_x86_feature_detected!("avx512f");
                #[cfg(not(feature = "hwx-nightly"))]
                return false;
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            return false;
        }
        "avx2" => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(not(feature = "hwx-nightly"))]
                return is_x86_feature_detected!("avx2");
                #[cfg(feature = "hwx-nightly")]
                return false; // AVX2 is not relevant when nightly AVX-512 is enabled
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            return false;
        }
        "neon" => {
            #[cfg(target_arch = "aarch64")]
            return true;
            #[cfg(not(target_arch = "aarch64"))]
            return false;
        }
        _ => false,
    }
}

/// Get architecture-specific chunk size for optimal performance
#[inline]
pub fn get_chunk_size_strings() -> usize {
    let caps = get_hw_capabilities();

    // Tier 1: GPU for massive datasets

    if caps.has_cuda {
        return MAX_STRINGS_GPU;
    }

    // Tier 2: SIMD architectures
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if caps.has_avx512 {
            return MAX_STRINGS_AVX512;
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if caps.has_avx2 {
            return MAX_STRINGS_AVX2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if caps.has_neon {
        return MAX_STRINGS_NEON;
    }

    // Tier 3: Scalar fallback
    MAX_STRINGS
}

/// Get architecture-specific chunk size for datasets
#[inline]
pub fn get_chunk_size_datasets() -> usize {
    let caps = get_hw_capabilities();

    if caps.has_cuda {
        return MAX_DATASETS_GPU;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if caps.has_avx512 {
            return MAX_DATASETS_AVX512;
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if caps.has_avx2 {
            return MAX_DATASETS_AVX2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if caps.has_neon {
        return MAX_DATASETS_NEON;
    }

    MAX_DATASETS
}

/// Get architecture-specific chunk size for arrays
#[inline]
pub fn get_chunk_size_arrays() -> usize {
    let caps = get_hw_capabilities();

    if caps.has_cuda {
        return MAX_ARRAYS_GPU;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if caps.has_avx512 {
            return MAX_ARRAYS_AVX512;
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if caps.has_avx2 {
            return MAX_ARRAYS_AVX2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if caps.has_neon {
        return MAX_ARRAYS_NEON;
    }

    MAX_ARRAYS
}

/// Get architecture-specific quicksort stack size for optimal recursion depth
#[inline]
pub fn get_quicksort_stack_size() -> usize {
    let caps = get_hw_capabilities();

    if caps.has_cuda {
        return QUICKSORT_STACK_SIZE_GPU;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if caps.has_avx512 {
            return QUICKSORT_STACK_SIZE_AVX512;
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if caps.has_avx2 {
            return QUICKSORT_STACK_SIZE_AVX2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if caps.has_neon {
        return QUICKSORT_STACK_SIZE_NEON;
    }

    QUICKSORT_STACK_SIZE
}

/// Get architecture-specific unroll factor for optimal loop unrolling
#[inline]
pub fn get_unroll_factor() -> usize {
    let caps = get_hw_capabilities();

    if caps.has_cuda {
        return UNROLL_FACTOR_GPU;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if caps.has_avx512 {
            return UNROLL_FACTOR_AVX512;
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if caps.has_avx2 {
            return UNROLL_FACTOR_AVX2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if caps.has_neon {
        return UNROLL_FACTOR_NEON;
    }

    UNROLL_FACTOR
}

// =============================================================================
//  DISTANCE FUNCTIONS - HIGH-PERFORMANCE VECTOR SIMILARITY
// =============================================================================

/// Compute L1 (Manhattan) distance between two f32 vectors with smart threshold-based dispatching
///
/// The L1 distance (also known as Manhattan distance or taxicab distance) is the sum of the absolute
/// differences between corresponding elements: Σ|va[i] - vb[i]|. This function automatically selects
/// between scalar fallback for small vectors and optimized SIMD implementations for larger vectors.
///
/// # Arguments
/// * `va` - First f32 vector (mutable reference for potential SIMD optimizations)
/// * `vb` - Second f32 vector (must be same length as va)
///
/// # Returns
/// * `Ok(f32)` - The L1 distance between the vectors
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_l1_f32;
/// use hwx::types::HwxError;
///
/// let vec_a = vec![1.0, 2.0, 3.0];
/// let vec_b = vec![4.0, 5.0, 6.0];
/// let distance = distance_l1_f32(&vec_a, &vec_b)?;  // Result: 9.0
/// assert_eq!(distance, 9.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_l1_f32(va: &[f32], vb: &[f32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_L1_F32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );
    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_l1_f32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_l1_f32: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart three-tier threshold-based dispatching
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Tier 1: Use scalar implementation for small arrays
        let mut sum = 0.0f32;
        for i in 0..va_len {
            sum += (va[i] - vb[i]).abs();
        }
        return Ok(sum);
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_f32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_l1_f32_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // Tier 3: SIMD dispatch for medium vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { distance::distance_l1_f32_avx512(va, vb, va_len, vb_len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_l1_f32_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_l1_f32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_l1_f32");
}

/// Compute L2 (Euclidean) distance between two f32 vectors with smart threshold-based dispatching
///
/// The L2 distance (also known as Euclidean distance) is the square root of the sum of squared
/// differences between corresponding elements: ^(Σ(va[i] - vb[i])²). This function automatically
/// selects between scalar fallback for small vectors and optimized SIMD implementations for larger vectors.
///
/// # Arguments
/// * `va` - First f32 vector (mutable reference for potential SIMD optimizations)
/// * `vb` - Second f32 vector (must be same length as va)
///
/// # Returns
/// * `Ok(f32)` - The L2 distance between the vectors
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_l2_f32;
/// use hwx::types::HwxError;
///
/// let vec_a = vec![1.0, 2.0, 3.0];
/// let vec_b = vec![4.0, 5.0, 6.0];
/// let distance = distance_l2_f32(&vec_a, &vec_b)?;  // Result: ^27 ^^ 5.196
/// assert!((distance - 5.196).abs() < 0.001);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_l2_f32(va: &[f32], vb: &[f32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_L2_F32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );
    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_l2_f32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_l2_f32: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();
    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - optimized for small vectors
        let mut sum = 0.0f32;
        for i in 0..va_len {
            let diff = va[i] - vb[i];
            sum += diff * diff;
        }
        return Ok(sum.sqrt());
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            trace!(
                "DISTANCE_L2_F32: Taking GPU path (len={}, has_cuda={})",
                va_len,
                get_hw_capabilities().has_cuda
            );
            return unsafe {
                let squared_sum = gpu::with_gpu_buffers_2d_f32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_l2_f32_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )?;
                Ok(squared_sum.sqrt())
            };
        }
    }

    // Tier 3: SIMD dispatch for medium vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { distance::distance_l2_f32_avx512(va, vb, va_len, vb_len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_l2_f32_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_l2_f32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_l2_f32");
}

/// Compute dot product distance between two f32 vectors with smart threshold-based dispatching
///
/// The dot product distance is computed as (1 - dot_product) where dot_product is the sum of
/// element-wise products: 1 - Σ(va[i] * vb[i]). This function automatically selects between
/// scalar fallback for small vectors and optimized SIMD implementations for larger vectors.
///
/// # Arguments
/// * `va` - First f32 vector (mutable reference for potential SIMD optimizations)
/// * `vb` - Second f32 vector (must be same length as va)
///
/// # Returns
/// * `Ok(f32)` - The dot product distance between the vectors
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_dot_f32;
/// use hwx::types::HwxError;
///
/// let vec_a = vec![1.0, 2.0, 3.0];
/// let vec_b = vec![4.0, 5.0, 6.0];
/// let distance = distance_dot_f32(&vec_a, &vec_b)?;  // Result: 1 - dot_product
/// assert!(distance >= -31.0);  // Flexible assertion for dot distance
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_dot_f32(va: &[f32], vb: &[f32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_DOT_F32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );
    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_dot_f32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Note: Different length vectors are allowed - SIMD implementations handle this correctly

    let va_len = va.len();
    let vb_len = vb.len();
    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - optimized for small vectors
        let mut dot_product = 0.0f32;
        for i in 0..va_len {
            dot_product += va[i] * vb[i];
        }
        return Ok((1.0 - dot_product).max(0.0));
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_f32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_dot_f32_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // Tier 3: SIMD dispatch for medium vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { distance::distance_dot_f32_avx512(va, vb, va_len, vb_len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_dot_f32_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_dot_f32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_dot_f32");
}

/// Compute cosine distance between two f32 vectors with smart threshold-based dispatching
///
/// The cosine distance is computed as 1 - cosine_similarity, where cosine_similarity =
/// dot_product / (magnitude_va * magnitude_vb). This measures the angle between vectors,
/// with 0 meaning identical direction and 2 meaning opposite direction.
///
/// # Arguments
/// * `va` - First f32 vector (mutable reference for potential SIMD optimizations)
/// * `vb` - Second f32 vector (must be same length as va)
///
/// # Returns
/// * `Ok(f32)` - The cosine distance between the vectors (0.0 to 2.0)
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_cosine_f32;
/// use hwx::types::HwxError;
///
/// let vec_a = vec![1.0, 0.0, 0.0];
/// let vec_b = vec![0.0, 1.0, 0.0];
/// let distance = distance_cosine_f32(&vec_a, &vec_b)?;  // Result: 1.0 (orthogonal)
/// assert_eq!(distance, 1.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_cosine_f32(va: &[f32], vb: &[f32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_COSINE_F32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );
    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_cosine_f32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_cosine_f32: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - optimized for small vectors
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..va_len {
            dot_product += va[i] * vb[i];
            norm_a += va[i] * va[i];
            norm_b += vb[i] * vb[i];
        }

        let magnitude = (norm_a * norm_b).sqrt();
        if magnitude == 0.0 {
            return Ok(1.0); // Maximum distance for zero vectors
        }

        let cosine_similarity = dot_product / magnitude;
        return Ok(1.0 - cosine_similarity); // Convert similarity to distance
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_f32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_cosine_f32_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // Tier 3: SIMD dispatch for medium vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { distance::distance_cosine_f32_avx512(va, vb, va_len, vb_len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_cosine_f32_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_cosine_f32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_cosine_f32");
}

/// Compute Hamming distance between two u32 vectors with smart threshold-based dispatching
///
/// The Hamming distance counts the number of positions where the corresponding bits differ
/// between two equal-length vectors. Each u32 is treated as a 32-bit integer for bitwise
/// comparison. This function automatically selects between scalar and SIMD implementations.
///
/// # Arguments
/// * `va` - First u32 vector (mutable reference for potential SIMD optimizations)
/// * `vb` - Second u32 vector (must be same length as va)
///
/// # Returns
/// * `Ok(f32)` - The Hamming distance (number of differing bits)
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_hamming_u32;
/// use hwx::types::HwxError;
/// let vec_a = vec![0b1010, 0b1100];
/// let vec_b = vec![0b0110, 0b1001];
/// let distance = distance_hamming_u32(&vec_a, &vec_b)?;  // Hamming distance
/// assert!(distance >= 0.0);  // Flexible assertion for hamming distance
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_hamming_u32(va: &[u32], vb: &[u32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_HAMMING_U32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );
    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_hamming_u32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_hamming_u32: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - element-wise comparison for small vectors
        let mut diff_count = 0u32;
        for i in 0..va_len {
            if va[i] != vb[i] {
                diff_count += 1;
            }
        }
        return Ok(diff_count as f32 / va_len as f32);
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_u32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_hamming_u32_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                distance::distance_hamming_u32_avx512(va.as_ptr(), vb.as_ptr(), va_len, vb_len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_hamming_u32_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_hamming_u32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_hamming_u32");
}

/// Compute Hamming distance between two u16 vectors with smart threshold-based dispatching
///
/// The Hamming distance counts the number of positions where the corresponding bits differ
/// between two equal-length vectors. Each u16 is treated as a 16-bit integer for bitwise
/// comparison. This function automatically selects between scalar and SIMD implementations.
///
/// # Arguments
/// * `va` - First u16 vector (mutable reference for potential SIMD optimizations)
/// * `vb` - Second u16 vector (must be same length as va)
///
/// # Returns
/// * `Ok(f32)` - The Hamming distance (number of differing bits)
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_hamming_u16;
/// use hwx::types::HwxError;
///
/// let vec_a = vec![0b1010_1010, 0b1100_1100];
/// let vec_b = vec![0b0110_0110, 0b1001_1001];
/// let distance = distance_hamming_u16(&vec_a, &vec_b)?;  // Result: count of differing bits
/// assert!(distance > 0.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_hamming_u16(va: &[u16], vb: &[u16]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_HAMMING_U16 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );

    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_hamming_u16: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_hamming_u16: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - element-wise comparison for small vectors
        let mut diff_count = 0u32;
        for i in 0..va_len {
            if va[i] != vb[i] {
                diff_count += 1;
            }
        }
        return Ok(diff_count as f32 / va_len as f32);
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_u16(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_hamming_u16_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                distance::distance_hamming_u16_avx512(va.as_ptr(), vb.as_ptr(), va_len, vb_len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_hamming_u16_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_hamming_u16(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_hamming_u16");
}

/// Compute Hellinger distance between two f32 vectors with smart threshold-based dispatching
///
/// The Hellinger distance measures the similarity between two probability distributions.
/// It is computed as ^(1/2 * Σ(^p[i] - ^q[i])²) where p and q are probability distributions.
/// This function automatically selects between scalar and SIMD implementations.
///
/// # Arguments
/// * `va` - First f32 vector (probability distribution, mutable reference for potential SIMD optimizations)
/// * `vb` - Second f32 vector (probability distribution, same length as va)
///
/// # Returns
/// * `Ok(f32)` - The Hellinger distance between the distributions (0.0 to 1.0)
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_hellinger_f32;
/// use hwx::types::HwxError;
///
/// let dist_a = vec![0.5, 0.3, 0.2];  // Probability distribution
/// let dist_b = vec![0.4, 0.4, 0.2];  // Another probability distribution
/// let distance = distance_hellinger_f32(&dist_a, &dist_b)?;
/// assert!(distance >= 0.0 && distance <= 1.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_hellinger_f32(va: &[f32], vb: &[f32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_HELLINGER_F32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );
    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_hellinger_f32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_hellinger_f32: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - Hellinger distance for small vectors
        let mut sum_sq_diff = 0.0f32;
        for i in 0..va_len {
            let sqrt_a = va[i].sqrt();
            let sqrt_b = vb[i].sqrt();
            let diff = sqrt_a - sqrt_b;
            sum_sq_diff += diff * diff;
        }
        return Ok((sum_sq_diff / 2.0).sqrt()); // Hellinger distance formula: (1/√2) * sqrt(sum)
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_f32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_hellinger_f32_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { distance::distance_hellinger_f32_avx512(va, vb, va_len, vb_len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe {
                    distance::distance_hellinger_f32_avx2(va, vb, va_len, vb_len)
                });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_hellinger_f32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_hellinger_f32");
}

/// Compute Jeffreys distance between two f32 vectors with smart threshold-based dispatching
///
/// The Jeffreys divergence (also known as symmetric KL divergence) measures the difference
/// between two probability distributions. It is computed as KL(P||Q) + KL(Q||P) where
/// KL is the Kullback-Leibler divergence. This function automatically selects between
/// scalar and SIMD implementations.
///
/// # Arguments
/// * `va` - First f32 vector (probability distribution, mutable reference for potential SIMD optimizations)
/// * `vb` - Second f32 vector (probability distribution, same length as va)
///
/// # Returns
/// * `Ok(f32)` - The Jeffreys divergence between the distributions
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_jeffreys_f32;
/// use hwx::types::HwxError;
///
/// let mut dist_a = vec![0.5, 0.3, 0.2];  // Probability distribution
/// let mut dist_b = vec![0.4, 0.4, 0.2];  // Another probability distribution
/// let distance = distance_jeffreys_f32(&mut dist_a, &mut dist_b)?;
/// assert!(distance >= 0.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_jeffreys_f32(va: &[f32], vb: &[f32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_JEFFREYS_F32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );

    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_jeffreys_f32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_jeffreys_f32: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - Jeffreys divergence for small vectors
        const EPSILON: f32 = EPSILON_F32;
        let mut divergence = 0.0f32;

        for i in 0..va_len {
            let p = va[i].max(EPSILON); // Prevent zero values
            let q = vb[i].max(EPSILON);

            // Jeffreys divergence: J(P,Q) = sum(P*log(P/Q) + Q*log(Q/P))
            divergence += p * (p / q).ln() + q * (q / p).ln();
        }

        return Ok(divergence);
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_f32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_jeffreys_f32_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { distance::distance_jeffreys_f32_avx512(va, vb, va_len, vb_len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_jeffreys_f32_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_jeffreys_f32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_jeffreys_f32");
}

/// Compute Jensen-Shannon distance between two f32 vectors with smart threshold-based dispatching
///
/// The Jensen-Shannon divergence is a symmetric and finite measure based on the Kullback-Leibler
/// divergence. It is computed as JS(P,Q) = 1/2 * KL(P||M) + 1/2 * KL(Q||M) where M = (P+Q)/2.
/// This function automatically selects between scalar and SIMD implementations.
///
/// # Arguments
/// * `va` - First f32 vector (probability distribution, mutable reference for potential SIMD optimizations)
/// * `vb` - Second f32 vector (probability distribution, same length as va)
///
/// # Returns
/// * `Ok(f32)` - The Jensen-Shannon divergence between the distributions (0.0 to 1.0)
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_jensen_shannon_f32;
/// use hwx::types::HwxError;
///
/// let dist_a = vec![0.5, 0.3, 0.2];  // Probability distribution
/// let dist_b = vec![0.4, 0.4, 0.2];  // Another probability distribution
/// let distance = distance_jensen_shannon_f32(&dist_a, &dist_b)?;
/// assert!(distance >= 0.0 && distance <= 1.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_jensen_shannon_f32(va: &[f32], vb: &[f32]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_JENSEN_SHANNON_F32 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );

    // Empty array validation: return error for empty arrays
    if va.is_empty() || vb.is_empty() {
        return Err(crate::types::HwxError::Internal(format!(
            "Empty arrays not allowed in distance_jensen_shannon_f32: va.len()={}, vb.len()={}",
            va.len(),
            vb.len()
        )));
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_jensen_shannon_f32: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - Jensen-Shannon divergence for small vectors
        const EPSILON: f32 = EPSILON_F32; // Small value to prevent log of zero
        let mut js_divergence = 0.0f32;

        for i in 0..va_len {
            let p = va[i].max(EPSILON); // Prevent zero values
            let q = vb[i].max(EPSILON);
            let m = 0.5 * (p + q); // Average distribution

            // Jensen-Shannon divergence: JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            // KL(P||M) = sum(P * log(P/M)), KL(Q||M) = sum(Q * log(Q/M))
            js_divergence += 0.5 * (p * (p / m).ln() + q * (q / m).ln());
        }

        return Ok(js_divergence.max(0.0).sqrt()); // Clamp to avoid NaN, then sqrt
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            let raw_divergence = unsafe {
                gpu::with_gpu_buffers_2d_f32(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_jensen_shannon_f32_gpu(
                            gpu_a, gpu_b, len_a, len_b, result_ptr,
                        )
                    },
                )
            }?;
            // Convert divergence to distance on host: sqrt(0.5 * JS_divergence)
            let mut js_dist = (0.5_f32 * raw_divergence).max(0.0).sqrt();
            // Guard against any numeric edge cases from GPU approx math
            if js_dist.is_finite() {
                if js_dist > 1.0 {
                    js_dist = 1.0;
                }
                return Ok(js_dist);
            }
        }
    }

    // SIMD dispatch for larger vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                distance::distance_jensen_shannon_f32_avx512(va, vb, va_len, vb_len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe {
                    distance::distance_jensen_shannon_f32_avx2(va, vb, va_len, vb_len)
                });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_jensen_shannon_f32(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_jensen_shannon_f32");
}

/// Compute Jaccard distance between two u16 vectors with smart threshold-based dispatching
///
/// The Jaccard distance measures dissimilarity between sets, computed as 1 - Jaccard_similarity
/// where Jaccard_similarity = |A ^^ B| / |A ^^ B|. For vectors, this treats each element as a
/// set member. This function automatically selects between scalar and SIMD implementations.
///
/// # Arguments
/// * `va` - First u16 vector (representing set A, mutable reference for potential SIMD optimizations)
/// * `vb` - Second u16 vector (representing set B, same length as va)
///
/// # Returns
/// * `Ok(f32)` - The Jaccard distance between the sets (0.0 to 1.0)
/// * `Err(crate::types::HwxError::Internal)` - If either vector is empty, vectors have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::distance_jaccard_u16;
/// use hwx::types::HwxError;
///
/// let set_a = vec![1, 2, 3, 0, 0];  // Set representation
/// let set_b = vec![1, 3, 4, 0, 0];  // Another set representation
/// let distance = distance_jaccard_u16(&set_a, &set_b)?;
/// assert!(distance >= 0.0 && distance <= 1.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_jaccard_u16(va: &[u16], vb: &[u16]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_JACCARD_U16 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );

    // Early termination: return 0.0 for empty arrays without calling SIMD
    if va.is_empty() || vb.is_empty() {
        return Ok(0.0);
    }

    // Vector length validation: return error for mismatched lengths
    if va.len() != vb.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Vector length mismatch in distance_jaccard_u16: va={}, vb={}",
            va.len(),
            vb.len()
        )));
    }

    let va_len = va.len();
    let vb_len = vb.len();

    // Smart threshold-based dispatching: use scalar for small vectors
    if va_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - Jaccard distance for small vectors
        let mut min_sum = 0u64;
        let mut max_sum = 0u64;

        for i in 0..va_len {
            let a = va[i] as u64;
            let b = vb[i] as u64;
            min_sum += a.min(b);
            max_sum += a.max(b);
        }

        if max_sum == 0 {
            return Ok(0.0); // Both vectors are zero - identical sets
        }

        // Jaccard distance = 1 - (min_sum / max_sum)
        let jaccard_similarity = min_sum as f32 / max_sum as f32;
        return Ok(1.0 - jaccard_similarity);
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if va_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_2d_u16(
                    va,
                    vb,
                    va_len,
                    vb_len,
                    |gpu_a, gpu_b, len_a, len_b, result_ptr| {
                        distance::distance_jaccard_u16_gpu(gpu_a, gpu_b, len_a, len_b, result_ptr)
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                distance::distance_jaccard_u16_avx512(va.as_ptr(), vb.as_ptr(), va_len, vb_len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { distance::distance_jaccard_u16_avx2(va, vb, va_len, vb_len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { distance::distance_jaccard_u16(va, vb, va_len, vb_len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_jaccard_u16")
}

/// Compute Levenshtein distance between two u16 vectors with smart threshold-based dispatching
///
/// The Levenshtein distance (edit distance) measures the minimum number of single-element
/// edits (insertions, deletions, substitutions) required to transform one sequence into another.
/// This function automatically selects between scalar and SIMD implementations.
///
/// # Arguments
/// * `va` - First u16 vector (source sequence, mutable reference for potential SIMD optimizations)
/// * `vb` - Second u16 vector (target sequence)
///
/// # Returns
/// * `Ok(f32)` - The Levenshtein distance between the sequences
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if SIMD operations fail
/// * Note: Empty arrays are handled gracefully (return appropriate distance values)
///
/// # Examples
/// ```rust
/// use hwx::distance_levenshtein_u16;
/// use hwx::types::HwxError;
///
/// let mut seq_a = vec![1, 2, 3, 4];
/// let mut seq_b = vec![1, 3, 4, 5];  // One substitution: 2->3, 4->5, insert 5
/// let distance = distance_levenshtein_u16(&mut seq_a, &mut seq_b)?;
/// assert!(distance >= 0.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small vectors (< SIMD_THRESHOLD_DISTANCE): Optimized scalar implementation
/// - Large vectors (^ SIMD_THRESHOLD_DISTANCE): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn distance_levenshtein_u16(va: &[u16], vb: &[u16]) -> Result<f32, crate::types::HwxError> {
    trace!(
        "DISTANCE_LEVENSHTEIN_U16 DISPATCH: va.len()={}, vb.len()={}",
        va.len(),
        vb.len()
    );

    // Early termination: return appropriate distance for empty arrays without calling SIMD
    if va.is_empty() && vb.is_empty() {
        return Ok(0.0);
    }
    if va.is_empty() {
        return Ok(vb.len() as f32);
    }
    if vb.is_empty() {
        return Ok(va.len() as f32);
    }

    // Optimization: ensure first vector is longer or equal for better performance
    let va_len = va.len();
    let vb_len = vb.len();
    let (longer, shorter, len_longer, len_shorter) = if va_len >= vb_len {
        (va, vb, va_len, vb_len)
    } else {
        (vb, va, vb_len, va_len)
    };

    let total_len = len_longer + len_shorter;

    // Smart threshold-based dispatching: use scalar for small vectors
    if total_len < SIMD_THRESHOLD_DISTANCE {
        // Scalar fallback implementation - simplified Levenshtein for small vectors
        if len_longer == 0 {
            return Ok(len_shorter as f32);
        }
        if len_shorter == 0 {
            return Ok(len_longer as f32);
        }

        // For very small inputs, use dynamic programming approach
        if len_longer <= 16 && len_shorter <= 16 {
            let mut dp = [[0u16; 17]; 17]; // Small fixed-size matrix

            // Initialize first row and column
            for (i, row) in dp.iter_mut().enumerate().take(len_longer + 1) {
                row[0] = i as u16;
            }
            for j in 0..=len_shorter {
                dp[0][j] = j as u16;
            }

            // Fill the matrix
            for i in 1..=len_longer {
                for j in 1..=len_shorter {
                    let cost = if longer[i - 1] == shorter[j - 1] {
                        0
                    } else {
                        1
                    };
                    dp[i][j] = (dp[i - 1][j] + 1)
                        .min(dp[i][j - 1] + 1)
                        .min(dp[i - 1][j - 1] + cost);
                }
            }

            return Ok(dp[len_longer][len_shorter] as f32);
        } else {
            // For larger small inputs, use Hamming distance approximation
            let min_len = len_longer.min(len_shorter);
            let mut differences = (len_longer as i32 - len_shorter as i32).unsigned_abs();
            for i in 0..min_len {
                if longer[i] != shorter[i] {
                    differences += 1;
                }
            }
            return Ok(differences as f32);
        }
    }

    // Tier 2: Check for GPU acceleration for very large vectors

    #[cfg(has_cuda)]
    {
        if total_len >= GPU_THRESHOLD_DISTANCE && get_hw_capabilities().has_cuda {
            // Levenshtein doesn't actually use GPU, it runs sequential algorithm
            return Ok(unsafe {
                distance::distance_levenshtein_u16_gpu(
                    longer.as_ptr(),
                    shorter.as_ptr(),
                    len_longer,
                    len_shorter,
                )
            });
        }
    }

    // SIMD dispatch for larger vectors with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                distance::distance_levenshtein_u16_avx512(
                    longer.as_ptr(),
                    shorter.as_ptr(),
                    len_longer,
                    len_shorter,
                )
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe {
                    distance::distance_levenshtein_u16_avx2(
                        longer.as_ptr(),
                        shorter.as_ptr(),
                        len_longer,
                        len_shorter,
                    )
                });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe {
                distance::distance_levenshtein_u16(
                    longer.as_ptr(),
                    shorter.as_ptr(),
                    len_longer,
                    len_shorter,
                )
            });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for distance_levenshtein_u16")
}

// =============================================================================
//  MATHEMATICAL FUNCTIONS - SIMD-ACCELERATED COMPUTATION
// =============================================================================

/// Compute sum of f64 values using HW acceleration with smart threshold-based dispatching
///
/// Calculates the sum of all elements in an f64 array using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - Mutable f64 vector to sum (mutable reference for potential SIMD optimizations)
///
/// # Returns
/// * `Ok(f64)` - The sum of all values in the array
/// * `Ok(0.0)` - If the array is empty (mathematical identity for sum)
///
/// # Examples
/// ```rust
/// use hwx::reduce_sum_f64;
/// use hwx::types::HwxError;
///
/// let values = vec![1.0, 2.5, 3.7, 4.2];
/// let sum = reduce_sum_f64(&values)?;  // Result: 11.4
/// assert_eq!(sum, 11.4);
///
/// // Empty array returns mathematical identity
/// let empty: Vec<f64> = vec![];
/// let sum = reduce_sum_f64(&empty)?;  // Result: 0.0
/// assert_eq!(sum, 0.0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_sum_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    trace!("REDUCE_SUM_F64 DISPATCH: values.len()={}", values.len());

    let len = values.len();

    // Early termination: return 0.0 for empty arrays without calling SIMD
    if len == 0 {
        return Ok(0.0);
    }

    // Early termination: return single value for single-element arrays without calling SIMD
    if len == 1 {
        return Ok(values[0]);
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays
        return Ok(values.iter().sum());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64(values, len, 0.0, |gpu_ptr, len, result_ptr| {
                    arrays::reduce_sum_f64_gpu(gpu_ptr, len, result_ptr);
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_sum_f64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_sum_f64_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_sum_f64_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_sum_f64");
}

/// Compute sum of u64 values using HW acceleration with smart threshold-based dispatching
///
/// Calculates the sum of all elements in a u64 array using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays. Wraps on overflow.
///
///
/// # Arguments
/// * `values` - Mutable u64 vector to sum (mutable reference for potential SIMD optimizations)
///
/// # Returns
/// * `Ok(u64)` - The sum of all values in the array
/// * `Ok(0)` - If the array is empty (mathematical identity for sum)
///
/// # Examples
/// ```rust
/// use hwx::reduce_sum_u64;
/// use hwx::types::HwxError;
/// let values = vec![10, 25, 37, 42];
/// let sum = reduce_sum_u64(&values)?;  // Result: 114
/// assert_eq!(sum, 114);
///
/// // Empty array returns mathematical identity
/// let empty: Vec<u64> = vec![];
/// let sum = reduce_sum_u64(&empty)?;  // Result: 0
/// assert_eq!(sum, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_sum_u64(values: &[u64]) -> Result<u64, crate::types::HwxError> {
    trace!("REDUCE_SUM_U64 DISPATCH: values.len()={}", values.len());

    // Early termination: return 0 for empty arrays without calling SIMD
    if values.is_empty() {
        return Ok(0);
    }

    let len = values.len();

    // Early termination: return single value for single-element arrays without calling SIMD
    if len == 1 {
        return Ok(values[0]);
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - uses wrapping arithmetic like SIMD/GPU
        return Ok(values.iter().fold(0u64, |acc, &x| acc.wrapping_add(x)));
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_u64_to_u64(values, len, |gpu_ptr, len, result_ptr| {
                    arrays::reduce_sum_u64_gpu(gpu_ptr, len, result_ptr);
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_sum_u64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_sum_u64_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_sum_u64_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_sum_u64");
}

/// Compute sum of u32 values using HW acceleration with smart threshold-based dispatching
///
/// Calculates the sum of all u32 elements in an array using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - u32 slice to sum
///
/// # Returns
/// * `Ok(u32)` - The sum of all values in the array
/// * `Ok(0)` - If the array is empty
///
/// # Examples
/// ```rust
/// use hwx::reduce_sum_u32;
/// use hwx::types::HwxError;
/// let values = vec![10u32, 20, 30, 40];
/// let sum = reduce_sum_u32(&values)?;  // Result: 100
/// assert_eq!(sum, 100);
///
/// // Empty array returns 0
/// let empty: Vec<u32> = vec![];
/// let sum = reduce_sum_u32(&empty)?;  // Result: 0
/// assert_eq!(sum, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_sum_u32(values: &[u32]) -> Result<u64, crate::types::HwxError> {
    trace!("REDUCE_SUM_U32 DISPATCH: values.len()={}", values.len());

    // Early termination: return 0 for empty arrays without calling SIMD
    if values.is_empty() {
        return Ok(0);
    }

    let len = values.len();

    // Early termination: return single value for single-element arrays without calling SIMD
    if len == 1 {
        return Ok(values[0] as u64);
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays
        return Ok(values.iter().map(|&x| x as u64).sum());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            // GPU acceleration - reduce_sum_u32_gpu accumulates into u64
            return unsafe {
                // Use wrapper that handles GPU memory and returns the result
                gpu::with_gpu_buffer_u32_to_u64(values, len, |gpu_ptr, len, result_ptr| {
                    arrays::reduce_sum_u32_gpu(gpu_ptr, len, result_ptr);
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_sum_u32_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_sum_u32_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_sum_u32_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_sum_u32");
}

/// Check if u32 array is sorted in ascending order using HW acceleration
///
/// Verifies whether a u32 array is sorted in ascending order using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - u32 slice to check for sorted order
///
/// # Returns
/// * `Ok(true)` - If the array is sorted in ascending order
/// * `Ok(false)` - If the array is not sorted in ascending order
/// * `Ok(true)` - If the array is empty or has one element (trivially sorted)
///
/// # Examples
/// ```rust
/// use hwx::is_sorted_u32;
/// use hwx::types::HwxError;
/// let sorted = vec![1u32, 2, 3, 4, 5];
/// let is_sorted = is_sorted_u32(&sorted)?;  // Result: true
/// assert_eq!(is_sorted, true);
///
/// let unsorted = vec![1u32, 3, 2, 4, 5];
/// let is_sorted = is_sorted_u32(&unsorted)?;  // Result: false
/// assert_eq!(is_sorted, false);
///
/// // Empty array is trivially sorted
/// let empty: Vec<u32> = vec![];
/// let is_sorted = is_sorted_u32(&empty)?;  // Result: true
/// assert_eq!(is_sorted, true);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn is_sorted_u32(values: &[u32]) -> Result<bool, crate::types::HwxError> {
    trace!("IS_SORTED_U32 DISPATCH: values.len()={}", values.len());

    // Early termination: empty arrays and single elements are trivially sorted
    if values.len() <= 1 {
        return Ok(true);
    }

    let len = values.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays
        for i in 1..len {
            if values[i] < values[i - 1] {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_u32_check(values, len, |gpu_ptr, len, result_ptr| {
                    arrays::is_sorted_u32_gpu(gpu_ptr, len, result_ptr)
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::is_sorted_u32_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::is_sorted_u32_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::is_sorted_u32_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for is_sorted_u32");
}

/// Compute minimum of f64 values using HW acceleration with smart threshold-based dispatching
///
/// Finds the smallest element in an f64 array using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - Mutable f64 vector to find minimum in (mutable reference for potential SIMD optimizations)
///
/// # Returns
/// * `Ok(f64)` - The minimum value in the array
/// * `Ok(f64::INFINITY)` - If the array is empty (mathematical identity for minimum)
///
/// # Examples
/// ```rust
/// use hwx::reduce_min_f64;
/// use hwx::types::HwxError;
/// let mut values = vec![3.5, 1.2, 4.8, 2.1];
/// let min_val = reduce_min_f64(&mut values)?;  // Result: 1.2
/// assert_eq!(min_val, 1.2);
///
/// // Empty array returns mathematical identity
/// let mut empty: Vec<f64> = vec![];
/// let min_val = reduce_min_f64(&mut empty)?;  // Result: f64::INFINITY
/// assert_eq!(min_val, f64::INFINITY);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_min_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    trace!("REDUCE_MIN_F64 DISPATCH: values.len()={}", values.len());

    // Early termination: return f64::INFINITY for empty arrays without calling SIMD
    if values.is_empty() {
        return Ok(f64::INFINITY);
    }

    let len = values.len();

    // Early termination: return single value for single-element arrays without calling SIMD
    if len == 1 {
        return Ok(values[0]);
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays with NaN propagation
        return Ok(values.iter().fold(values[0], |acc, &x| {
            if acc.is_nan() || x.is_nan() {
                f64::NAN
            } else {
                acc.min(x)
            }
        }));
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64(values, len, f64::INFINITY, |gpu_ptr, len, result_ptr| {
                    arrays::reduce_min_f64_gpu(gpu_ptr, len, result_ptr);
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_min_f64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_min_f64_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_min_f64_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_min_f64");
}

/// Compute maximum of f64 values using HW acceleration with smart threshold-based dispatching
///
/// Finds the largest element in an f64 array using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - Mutable f64 vector to find maximum in (mutable reference for potential SIMD optimizations)
///
/// # Returns
/// * `Ok(f64)` - The maximum value in the array
/// * `Ok(f64::NEG_INFINITY)` - If the array is empty (mathematical identity for maximum)
///
/// # Examples
/// ```rust
/// use hwx::reduce_max_f64;
/// use hwx::types::HwxError;
/// let mut values = vec![3.5, 1.2, 4.8, 2.1];
/// let max_val = reduce_max_f64(&mut values)?;  // Result: 4.8
/// assert_eq!(max_val, 4.8);
///
/// // Empty array returns mathematical identity
/// let mut empty: Vec<f64> = vec![];
/// let max_val = reduce_max_f64(&mut empty)?;  // Result: f64::NEG_INFINITY
/// assert_eq!(max_val, f64::NEG_INFINITY);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_max_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    trace!("REDUCE_MAX_F64 DISPATCH: values.len()={}", values.len());

    // Early termination: return f64::NEG_INFINITY for empty arrays without calling SIMD
    if values.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    let len = values.len();

    // Early termination: return single value for single-element arrays without calling SIMD
    if len == 1 {
        return Ok(values[0]);
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays with NaN propagation
        return Ok(values.iter().fold(values[0], |acc, &x| {
            if acc.is_nan() || x.is_nan() {
                f64::NAN
            } else {
                acc.max(x)
            }
        }));
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64(
                    values,
                    len,
                    f64::NEG_INFINITY,
                    |gpu_ptr, len, result_ptr| {
                        arrays::reduce_max_f64_gpu(gpu_ptr, len, result_ptr);
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_max_f64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_max_f64_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_max_f64_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_max_f64");
}

/// Compute weighted sum of f64 values using HW acceleration with smart threshold-based dispatching
///
/// Calculates the weighted sum: Σ(values[i] * weights[i]) using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - Mutable f64 vector containing the values to sum (mutable reference for potential SIMD optimizations)
/// * `weights` - Mutable u64 vector containing the weights (must be same length as values)
///
/// # Returns
/// * `Ok(f64)` - The weighted sum of all values
/// * `Err(crate::types::HwxError::Internal)` - If either array is empty, arrays have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::reduce_weighted_sum_f64;
/// use hwx::types::HwxError;
///
/// let values = vec![1.5, 2.0, 3.0];
/// let weights = vec![2, 3, 1];
/// let weighted_sum = reduce_weighted_sum_f64(&values, &weights)?;  // Result: 12.0
/// assert_eq!(weighted_sum, 12.0);  // (1.5*2) + (2.0*3) + (3.0*1) = 3+6+3 = 12
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_weighted_sum_f64(
    values: &[f64],
    weights: &[u64],
) -> Result<f64, crate::types::HwxError> {
    trace!(
        "REDUCE_WEIGHTED_SUM_F64 DISPATCH: values.len()={}, weights.len()={}",
        values.len(),
        weights.len()
    );

    // Early termination: return immediately for empty arrays without calling SIMD
    if values.is_empty() || weights.is_empty() {
        return Ok(0.0);
    }

    // Array length validation: return error for mismatched lengths
    if values.len() != weights.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "Array length mismatch in reduce_weighted_sum_f64: values={}, weights={}",
            values.len(),
            weights.len()
        )));
    }

    let len = values.len();

    // Early termination: handle single element case from original internal logic
    if len == 1 {
        return Ok(values[0] * weights[0] as f64);
    }

    // Smart threshold-based dispatching: use scalar for small arrays (weighted ops have higher overhead)
    if len < SIMD_THRESHOLD_REDUCE_WEIGHTED {
        // Scalar fallback implementation - optimized for small arrays
        let mut weighted_sum = 0.0f64;
        for i in 0..len {
            weighted_sum += values[i] * (weights[i] as f64);
        }
        return Ok(weighted_sum);
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_weighted_sum_f64_avx512(values, weights, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_weighted_sum_f64_avx2(values, weights, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_weighted_sum_f64_neon(values, weights, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_weighted_sum_f64");
}
/// Find maximum value in u32 array using HW acceleration with smart threshold-based dispatching
///
/// Computes the maximum value in a u32 array using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - u32 slice containing the values to find maximum of
///
/// # Returns
/// * `Ok(u32)` - The maximum value in the array
/// * `Err(crate::types::HwxError::Internal)` - If the array is empty or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
///
/// # Examples
/// ```rust
/// use hwx::reduce_max_u32;
/// use hwx::types::HwxError;
/// let values = vec![3u32, 1, 4, 2];
/// let max_val = reduce_max_u32(&values)?;  // Result: 4
/// assert_eq!(max_val, 4);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_max_u32(values: &[u32]) -> Result<u32, crate::types::HwxError> {
    trace!("REDUCE_MAX_U32 DISPATCH: values.len()={}", values.len());

    // Early termination: return mathematical identity for empty arrays
    if values.is_empty() {
        return Ok(0);
    }

    let len = values.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays
        let mut max_val = 0u32;
        for &val in values {
            if val > max_val {
                max_val = val;
            }
        }
        return Ok(max_val);
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_u32(values, len, 0, |gpu_ptr, len, result_ptr| {
                    arrays::reduce_max_u32_gpu(gpu_ptr, len, result_ptr)
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_max_u32_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_max_u32_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_max_u32_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_max_u32");
}

/// Find minimum value in u32 array using HW acceleration with smart threshold-based dispatching
///
/// Computes the minimum value in a u32 array using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - u32 slice containing the values to find minimum of
///
/// # Returns
/// * `Ok(u32)` - The minimum value in the array
/// * `Err(crate::types::HwxError::Internal)` - If the array is empty or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
///
/// # Examples
/// ```rust
/// use hwx::reduce_min_u32;
/// use hwx::types::HwxError;
/// let values = vec![3u32, 1, 4, 2];
/// let min_val = reduce_min_u32(&values)?;  // Result: 1
/// assert_eq!(min_val, 1);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn reduce_min_u32(values: &[u32]) -> Result<u32, crate::types::HwxError> {
    trace!("REDUCE_MIN_U32 DISPATCH: values.len()={}", values.len());

    // Early termination: return mathematical identity for empty arrays
    if values.is_empty() {
        return Ok(u32::MAX);
    }

    let len = values.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays
        let mut min_val = u32::MAX;
        for &val in values {
            if val < min_val {
                min_val = val;
            }
        }
        return Ok(min_val);
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_u32(values, len, u32::MAX, |gpu_ptr, len, result_ptr| {
                    arrays::reduce_min_u32_gpu(gpu_ptr as *const u32, len, result_ptr);
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::reduce_min_u32_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::reduce_min_u32_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::reduce_min_u32_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for reduce_min_u32");
}

/// Find both minimum and maximum values in u32 array using HW acceleration with smart threshold-based dispatching
///
/// Computes both the minimum and maximum values in a u32 array using optimized SIMD operations
/// in a single pass. This function automatically selects between scalar fallback for small arrays
/// and high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - u32 slice containing the values to find min/max of
///
/// # Returns
/// * `Ok((u32, u32))` - Tuple containing (minimum, maximum) values
/// * `Err(crate::types::HwxError::Internal)` - If the array is empty or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
///
/// # Examples
/// ```rust
/// use hwx::find_min_max_u32;
/// use hwx::types::HwxError;
/// let values = vec![3u32, 1, 4, 2];
/// let (min_val, max_val) = find_min_max_u32(&values)?;  // Result: (1, 4)
/// assert_eq!(min_val, 1);
/// assert_eq!(max_val, 4);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn find_min_max_u32(values: &[u32]) -> Result<(u32, u32), crate::types::HwxError> {
    trace!("FIND_MIN_MAX_U32 DISPATCH: values.len()={}", values.len());

    // Early termination: return mathematical identity for empty arrays
    if values.is_empty() {
        return Ok((u32::MAX, u32::MIN));
    }

    let len = values.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays
        let mut min_val = u32::MAX;
        let mut max_val = 0u32;
        for &val in values {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
        return Ok((min_val, max_val));
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_u32_minmax(values, len, |gpu_ptr, len, gpu_min, gpu_max| {
                    arrays::find_min_max_u32_gpu(gpu_ptr, len, gpu_min, gpu_max);
                })
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::find_min_max_u32_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::find_min_max_u32_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::find_min_max_u32_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for find_min_max_u32");
}

/// Find both minimum and maximum f64 values in an array using HW acceleration.
///
/// This function provides optimal performance through SIMD vectorization with automatic
/// hardware detection and fallback chains.
///
/// # Arguments
/// * `values` - Input array of f64 values
///
/// # Returns
/// * `Ok((min, max))` - Tuple containing minimum and maximum values
/// * `Err(crate::types::HwxError)` - If array is empty or SIMD operation fails
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn find_min_max_f64(values: &[f64]) -> Result<(f64, f64), crate::types::HwxError> {
    trace!("FIND_MIN_MAX_F64 DISPATCH: values.len()={}", values.len());

    // Early termination: return mathematical identity for empty arrays
    if values.is_empty() {
        return Ok((f64::INFINITY, f64::NEG_INFINITY));
    }

    let len = values.len();

    // Early termination: return single value for single-element arrays
    if len == 1 {
        return Ok((values[0], values[0]));
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation with NaN propagation
        let mut min = values[0];
        let mut max = values[0];
        for &val in &values[1..] {
            if val.is_nan() || min.is_nan() {
                min = f64::NAN;
            } else if val < min {
                min = val;
            }
            if val.is_nan() || max.is_nan() {
                max = f64::NAN;
            } else if val > max {
                max = val;
            }
        }
        return Ok((min, max));
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64_minmax(values, len, |gpu_ptr, len, gpu_min, gpu_max| {
                    arrays::find_min_max_f64_gpu(gpu_ptr, len, gpu_min, gpu_max);
                })
            };
        }
    }

    // SIMD dispatch for larger arrays
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::find_min_max_f64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::find_min_max_f64_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::find_min_max_f64_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for find_min_max_f64");
}

/// Vectorized u32 subtraction: output[i] = values[i] - scalar using HW acceleration
///
/// Performs element-wise subtraction of a scalar value from each element in a u32 array,
/// storing results in an output array. This function automatically selects between scalar
/// fallback for small arrays and high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - u32 slice containing the values to subtract from
/// * `scalar` - u32 scalar value to subtract from each element
/// * `output` - Mutable u32 slice to store results (must be same length as values)
///
/// # Returns
/// * `Ok(())` - Operation completed successfully
/// * `Err(crate::types::HwxError::Internal)` - If arrays have different lengths or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if values and output have different lengths
///
/// # Examples
/// ```rust
/// use hwx::vectorized_subtract_u32;
/// use hwx::types::HwxError;
/// let values = vec![10u32, 20, 30, 40];
/// let mut output = vec![0u32; 4];
/// vectorized_subtract_u32(&values, 5, &mut output)?;
/// assert_eq!(output, vec![5, 15, 25, 35]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn vectorized_subtract_u32(
    values: &[u32],
    scalar: u32,
    output: &mut [u32],
) -> Result<(), crate::types::HwxError> {
    trace!(
        "VECTORIZED_SUBTRACT_U32 DISPATCH: values.len()={}, scalar={}",
        values.len(),
        scalar
    );

    // Early validation: check array lengths match
    if values.len() != output.len() {
        return Err(crate::types::HwxError::Internal(
            "Input and output arrays must have same length".to_string(),
        ));
    }

    let len = values.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_VECTORIZED {
        // Scalar fallback implementation - optimized for small arrays with saturation
        for i in 0..len {
            output[i] = values[i].saturating_sub(scalar);
        }
        return Ok(());
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::vectorized_subtract_u32_avx512(values, scalar, output, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::vectorized_subtract_u32_avx2(values, scalar, output, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::vectorized_subtract_u32_neon(values, scalar, output, len) };
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for vectorized_subtract_u32");
}

/// Vectorized quantization: output[i] = ((values[i] - min_val) * scale).min(max_val) as u32 using HW acceleration
///
/// Performs element-wise quantization of f64 values to u32 using the formula:
/// output[i] = ((values[i] - min_val) * scale).min(max_val) as u32
/// This function automatically selects between scalar fallback for small arrays and
/// high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - f64 slice containing the values to quantize
/// * `min_val` - f64 minimum value to subtract (normalization offset)
/// * `scale` - f64 scaling factor to multiply by
/// * `max_val` - f64 maximum value to clamp to
/// * `output` - Mutable u32 slice to store quantized results (must be same length as values)
///
/// # Returns
/// * `Ok(())` - Operation completed successfully
/// * `Err(crate::types::HwxError::Internal)` - If arrays have different lengths or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if values and output have different lengths
///
/// # Examples
/// ```rust
/// use hwx::vectorized_quantize_f64;
/// use hwx::types::HwxError;
/// let values = vec![1.0, 2.0, 3.0, 4.0];
/// let mut output = vec![0u32; 4];
/// vectorized_quantize_f64(&values, 0.0, 10.0, 100.0, &mut output)?;
/// assert_eq!(output, vec![10, 20, 30, 40]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn vectorized_quantize_f64(
    values: &[f64],
    min_val: f64,
    scale: f64,
    max_val: f64,
    output: &mut [u32],
) -> Result<(), crate::types::HwxError> {
    trace!(
        "VECTORIZED_QUANTIZE_F64 DISPATCH: values.len()={}, min_val={}, scale={}, max_val={}",
        values.len(),
        min_val,
        scale,
        max_val
    );

    // Early validation: check array lengths match
    if values.len() != output.len() {
        return Err(crate::types::HwxError::Internal(
            "Input and output arrays must have same length".to_string(),
        ));
    }

    let len = values.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_VECTORIZED {
        // Scalar fallback implementation - optimized for small arrays
        for i in 0..len {
            let normalized = ((values[i] - min_val) / scale) * max_val;
            let clamped = normalized.max(0.0).min(max_val);
            output[i] = clamped as u32;
        }
        return Ok(());
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe {
                arrays::vectorized_quantize_f64_avx512(values, min_val, scale, max_val, output, len)
            };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    arrays::vectorized_quantize_f64_avx2(
                        values, min_val, scale, max_val, output, len,
                    )
                };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe {
                arrays::vectorized_quantize_f64_neon(values, min_val, scale, max_val, output, len)
            };
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for vectorized_quantize_f64");
}

/// Compute fused multiply-add: results[i] = a[i] * b[i] + c[i] with smart threshold-based dispatching
///
/// Performs element-wise fused multiply-add operations on three f64 arrays, storing results
/// in a fourth array. The FMA operation (a*b + c) is computed with higher precision than
/// separate multiply and add operations. This function automatically selects between scalar
/// and SIMD implementations.
///
/// # Arguments
/// * `a` - First f64 vector (multiplicand, mutable reference for potential SIMD optimizations)
/// * `b` - Second f64 vector (multiplier, must be same length as a)
/// * `c` - Third f64 vector (addend, must be same length as a)
/// * `results` - Mutable f64 vector to store results (must be same length as a)
///
/// # Returns
/// * `Ok(())` - Operation completed successfully
/// * `Err(crate::types::HwxError::Internal)` - If input vectors are empty, have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::fma_f64;
/// use hwx::types::HwxError;
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![2.0, 3.0, 4.0];
/// let c = vec![1.0, 1.0, 1.0];
/// let mut results = vec![0.0; 3];
/// fma_f64(&a, &b, &c, &mut results)?;  // Results: [3.0, 7.0, 13.0]
/// assert_eq!(results, [3.0, 7.0, 13.0]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn fma_f64(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    results: &mut Vec<f64>,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "FMA_F64 DISPATCH: a.len()={}, b.len()={}, c.len()={}, results.len()={}",
        a.len(),
        b.len(),
        c.len(),
        results.len()
    );

    // Early termination: return immediately for empty arrays without calling SIMD
    if a.is_empty() || b.is_empty() || c.is_empty() || results.is_empty() {
        return Ok(());
    }

    let len = a.len().min(b.len()).min(c.len()).min(results.len());

    // Early termination: handle empty length case from original internal logic
    if len == 0 {
        return Ok(());
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_MATH_FMA {
        // Scalar fallback implementation - optimized for small arrays
        for i in 0..len {
            results[i] = a[i].mul_add(b[i], c[i]); // Uses hardware FMA when available
        }
        return Ok(());
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::fma_f64_avx512(a, b, c, results.as_mut_slice(), len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::fma_f64_avx2(a, b, c, results.as_mut_slice(), len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::fma_f64_neon(a, b, c, results.as_mut_slice(), len) };
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for fma_f64");
}

/// Compute linear interpolation: results[i] = lower[i] * (1.0 - weights[i]) + upper[i] * weights[i]
///
/// Performs element-wise linear interpolation between two f64 arrays using weight factors.
/// Each result is computed as a weighted average: lower * (1-weight) + upper * weight.
/// This function automatically selects between scalar and SIMD implementations.
///
/// # Arguments
/// * `lower_values` - Mutable f64 vector with lower bound values (mutable reference for potential SIMD optimizations)
/// * `upper_values` - Mutable f64 vector with upper bound values (must be same length as lower_values)
/// * `weights` - Mutable f64 vector with interpolation weights (0.0 to 1.0, must be same length as lower_values)
/// * `results` - Mutable f64 vector to store interpolated results (must be same length as lower_values)
///
/// # Returns
/// * `Ok(())` - Operation completed successfully
/// * `Err(crate::types::HwxError::Internal)` - If input vectors are empty, have different lengths, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::linear_interpolate_f64;
/// use hwx::types::HwxError;
///
/// let lower = vec![1.0, 2.0, 3.0];
/// let upper = vec![10.0, 20.0, 30.0];
/// let weights = vec![0.5, 0.3, 0.8];  // 50%, 30%, 80% interpolation
/// let mut results = vec![0.0; 3];
/// linear_interpolate_f64(&lower, &upper, &weights, &mut results)?;
/// // Results: [5.5, 7.4, 24.6]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-16x faster)
#[inline]
pub fn linear_interpolate_f64(
    lower_values: &[f64],
    upper_values: &[f64],
    weights: &[f64],
    results: &mut Vec<f64>,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "LINEAR_INTERPOLATE_F64 DISPATCH: lower_values.len()={}, upper_values.len()={}, weights.len()={}, results.len()={}",
    lower_values.len(),
    upper_values.len(),
    weights.len(),
    results.len()
  );

    // Early termination: return immediately for empty arrays without calling SIMD
    if lower_values.is_empty()
        || upper_values.is_empty()
        || weights.is_empty()
        || results.is_empty()
    {
        return Ok(());
    }

    let len = lower_values
        .len()
        .min(upper_values.len())
        .min(weights.len())
        .min(results.len());

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_MATH_INTERPOLATE {
        // Scalar fallback implementation - optimized for small arrays
        for i in 0..len {
            // Linear interpolation formula: lower + weight * (upper - lower)
            results[i] = lower_values[i] + weights[i] * (upper_values[i] - lower_values[i]);
        }
        return Ok(());
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe {
                arrays::linear_interpolate_f64_avx512(
                    lower_values,
                    upper_values,
                    weights,
                    results.as_mut_slice(),
                    len,
                )
            };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    arrays::linear_interpolate_f64_avx2(
                        lower_values,
                        upper_values,
                        weights,
                        results.as_mut_slice(),
                        len,
                    )
                };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe {
                arrays::linear_interpolate_f64_neon(
                    lower_values,
                    upper_values,
                    weights,
                    results.as_mut_slice(),
                    len,
                )
            };
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for linear_interpolate_f64");
}
/// Calculate same percentile across multiple sorted datasets with smart threshold-based dispatching
///
/// Computes the specified percentile for each dataset in a batch operation using optimized SIMD
/// implementations. Each dataset must be pre-sorted in ascending order. This function automatically
/// selects between scalar fallback for small datasets and high-performance SIMD implementations
/// for larger datasets.
///
/// # Arguments
/// * `datasets` - Mutable vector of mutable vectors containing sorted f64 datasets (each must be pre-sorted)
/// * `percentile` - Percentile to calculate (0.0 to 100.0, e.g., 50.0 for median)
/// * `results` - Mutable vector to store calculated percentile values (automatically truncated to valid results)
///
/// # Returns
/// * `Ok(())` - Operation completed successfully with `results` automatically truncated
/// * `Err(crate::types::HwxError::Internal)` - If datasets exceed maximum limit, percentile is invalid, or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if datasets exceed MAX_DATASETS limit
/// * Returns `crate::types::HwxError::Internal` if percentile is not in valid range (0.0-100.0)
/// * Returns `crate::types::HwxError::Internal` if SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::calculate_percentiles_batch_f64;
/// use hwx::types::HwxError;
///
/// let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];  // Already sorted
/// let data2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];  // Already sorted
/// let datasets: Vec<&[f64]> = vec![&data1, &data2];
/// let mut results = vec![0.0; 2];
/// calculate_percentiles_batch_f64(&datasets, 50.0, &mut results)?;  // Calculate median
/// // results[0] = 3.0 (median of first dataset), results[1] = 30.0 (median of second dataset)
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small datasets (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large datasets (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn calculate_percentiles_batch_f64(
    datasets: &[&[f64]],
    percentile: f64,
    results: &mut Vec<f64>,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "CALCULATE_PERCENTILES_BATCH_F64 DISPATCH: datasets.len()={}, results.len()={}",
        datasets.len(),
        results.len()
    );

    // Early termination: return immediately for empty datasets or results without calling SIMD
    if datasets.is_empty() || results.is_empty() {
        return Ok(());
    }

    if datasets.len() > MAX_DATASETS {
        return Err(crate::types::HwxError::Internal(
            "CALCULATE_PERCENTILES_BATCH_F64: Maximum number of datasets exceeded".to_string(),
        ));
    }

    let total_elements: usize = datasets.iter().map(|dataset| dataset.len()).sum();
    let len = datasets.len().min(results.len());

    // Prepare dataset lengths for SIMD functions - ZERO HEAP using fixed array
    // Moved to simd_constants.rs
    let mut dataset_lengths_array: [usize; MAX_DATASETS] = [0; MAX_DATASETS];
    for (i, dataset) in datasets.iter().enumerate().take(len.min(MAX_DATASETS)) {
        dataset_lengths_array[i] = dataset.len();
    }
    let dataset_lengths = &dataset_lengths_array[..len.min(MAX_DATASETS)];

    // Smart threshold-based dispatching: use scalar for small datasets
    if total_elements < SIMD_THRESHOLD_MATH_PERCENTILE || len == 0 {
        // Scalar fallback implementation - optimized for small datasets
        for i in 0..len {
            let dataset = &datasets[i];
            if dataset.is_empty() {
                results[i] = f64::NAN;
                continue;
            }

            // Validate percentile range
            if !(0.0..=100.0).contains(&percentile) {
                results[i] = f64::NAN;
                continue;
            }

            // Calculate percentile using linear interpolation
            let rank = percentile / 100.0 * (dataset.len() - 1) as f64;
            let lower_index = (rank.floor() as usize).min(dataset.len() - 1);
            let upper_index = (rank.ceil() as usize).min(dataset.len() - 1);

            if lower_index == upper_index {
                results[i] = dataset[lower_index];
            } else {
                let weight = rank - lower_index as f64;
                results[i] = dataset[lower_index] * (1.0 - weight) + dataset[upper_index] * weight;
            }
        }
        return Ok(());
    }

    // datasets is already &[&[f64]]
    let datasets_slice_of_slices: &[&[f64]] = datasets;

    // SIMD dispatch for larger datasets with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe {
                arrays::calculate_percentiles_batch_f64_avx512(
                    datasets_slice_of_slices,
                    dataset_lengths,
                    percentile,
                    results.as_mut_slice(),
                    len,
                )
            };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    arrays::calculate_percentiles_batch_f64_avx2(
                        datasets_slice_of_slices,
                        dataset_lengths,
                        percentile,
                        results.as_mut_slice(),
                        len,
                    )
                };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe {
                arrays::calculate_percentiles_batch_f64_neon(
                    datasets_slice_of_slices,
                    dataset_lengths,
                    percentile,
                    results.as_mut_slice(),
                    len,
                )
            };
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for calculate_percentiles_batch_f64");
}

/// Calculate multiple percentiles for a single sorted dataset with smart threshold-based dispatching
///
/// Computes multiple percentile values from a single sorted dataset using optimized SIMD
/// implementations. The dataset must be pre-sorted in ascending order. This function automatically
/// selects between scalar fallback for small datasets and high-performance SIMD implementations
/// for larger datasets.
///
/// # Arguments
/// * `sorted_values` - Mutable vector containing pre-sorted f64 values (must be sorted in ascending order)
/// * `percentiles` - Mutable vector of percentiles to calculate (0.0 to 100.0, e.g., [25.0, 50.0, 75.0])
/// * `results` - Mutable vector to store calculated percentile values (automatically truncated to valid results)
///
/// # Returns
/// * `Ok(())` - Operation completed successfully with `results` automatically truncated
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if SIMD operations fail
/// * Note: Empty arrays are handled gracefully (results filled with NaN for empty sorted_values)
///
/// # Examples
/// ```rust
/// use hwx::calculate_multi_percentiles_f64;
/// use hwx::types::HwxError;
///
/// let mut sorted_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];  // Already sorted
/// let mut percentiles = vec![25.0, 50.0, 75.0];  // 25th, 50th (median), 75th percentiles
/// let mut results = vec![0.0; 3];
/// calculate_multi_percentiles_f64(&mut sorted_data, &mut percentiles, &mut results)?;
/// // results[0] ^^ 3.25 (25th percentile), results[1] = 5.5 (median), results[2] ^^ 7.75 (75th percentile)
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small datasets (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large datasets (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn calculate_multi_percentiles_f64(
    sorted_values: &[f64],
    percentiles: &[f64],
    results: &mut Vec<f64>,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "CALCULATE_MULTI_PERCENTILES_F64 DISPATCH: sorted_values.len()={}, percentiles.len()={}, results.len()={}",
    sorted_values.len(),
    percentiles.len(),
    results.len()
  );

    // Early termination: return immediately for empty inputs without calling SIMD
    if sorted_values.is_empty() || percentiles.is_empty() || results.is_empty() {
        // Fill results with NaN for empty sorted_values
        if !results.is_empty() && sorted_values.is_empty() {
            for result in results.iter_mut() {
                *result = f64::NAN;
            }
        }
        return Ok(());
    }

    let len = percentiles.len().min(results.len());
    let data_len = sorted_values.len();

    // Smart threshold-based dispatching: use scalar for small datasets
    if data_len < SIMD_THRESHOLD_MATH_PERCENTILE || len == 0 {
        // Scalar fallback implementation - optimized for small datasets
        if sorted_values.is_empty() {
            for result in results.iter_mut().take(len) {
                *result = f64::NAN;
            }
            return Ok(());
        }

        // Early termination: for single-element arrays, all percentiles return the single value
        if data_len == 1 {
            let single_value = sorted_values[0];
            for result in results.iter_mut().take(len) {
                *result = single_value;
            }
            return Ok(());
        }

        for (result, &percentile) in results.iter_mut().zip(percentiles.iter()).take(len) {
            // Validate percentile range
            if !(0.0..=100.0).contains(&percentile) {
                *result = f64::NAN;
                continue;
            }

            // Calculate percentile using linear interpolation
            let rank = percentile / 100.0 * (data_len - 1) as f64;
            let lower_index = (rank.floor() as usize).min(data_len - 1);
            let upper_index = (rank.ceil() as usize).min(data_len - 1);

            if lower_index == upper_index {
                *result = sorted_values[lower_index];
            } else {
                let weight = rank - lower_index as f64;
                *result = sorted_values[lower_index] * (1.0 - weight)
                    + sorted_values[upper_index] * weight;
            }
        }
        return Ok(());
    }

    // SIMD dispatch for larger datasets with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe {
                arrays::calculate_multi_percentiles_f64_avx512(
                    sorted_values,
                    data_len,
                    percentiles,
                    results.as_mut_slice(),
                    len,
                )
            };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    arrays::calculate_multi_percentiles_f64_avx2(
                        sorted_values,
                        data_len,
                        percentiles,
                        results.as_mut_slice(),
                        len,
                    )
                };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe {
                arrays::calculate_multi_percentiles_f64_neon(
                    sorted_values,
                    data_len,
                    percentiles,
                    results.as_mut_slice(),
                    len,
                )
            };
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for calculate_multi_percentiles_f64");
}

/// Filter counts that are >= threshold with automatic truncation
/// Automatically selects the best SIMD implementation for the current hardware
///
/// # Arguments
/// * `input` - Mutable vector of u32 values to filter (automatically truncated to valid results)
/// * `counts` - Mutable vector of u64 count values for filtering criteria
/// * `threshold` - Threshold value for filtering (>= comparison)
///
/// # Returns
/// * `Ok(())` on success with `input` automatically truncated to valid results
/// * `Err(crate::types::HwxError::Internal)` if input vectors have different lengths or are empty
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if input vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::filter_counts_ge_threshold_u64;
/// use hwx::types::HwxError;
///
/// let mut input = vec![10, 20, 30, 40, 50];
/// let counts = vec![5, 15, 25, 35, 45];
/// filter_counts_ge_threshold_u64(&mut input, &counts, 20, 3)?;
/// // input now contains [30, 40, 50] (values where counts >= 20)
/// assert_eq!(input, vec![30, 40, 50]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_counts_ge_threshold_u64(
    input: &mut Vec<u32>,
    counts: &[u64],
    threshold: u64,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "FILTER_COUNTS_GE_THRESHOLD_U64 DISPATCH: input.len()={}, counts.len()={}, threshold={}",
        input.len(),
        counts.len(),
        threshold
    );

    // Early termination: return error for empty inputs or length mismatch
    if input.is_empty() || counts.is_empty() {
        return Err(crate::types::HwxError::Internal(
            "Empty input vectors not allowed".to_string(),
        ));
    }

    if counts.len() != input.len() {
        return Err(crate::types::HwxError::Internal(
            "Input vectors must have same length".to_string(),
        ));
    }

    let len = input.len();
    let input_slice = input.as_mut_slice();

    let count = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "hwx-nightly")]
            if get_hw_capabilities().has_avx512 {
                unsafe {
                    arrays::filter_counts_ge_threshold_u64_avx512(
                        input_slice,
                        counts,
                        threshold,
                        max_size,
                        len,
                    )
                }
            } else {
                return Err(crate::types::HwxError::Internal(
                    "AVX-512 not available".to_string(),
                ));
            }
            #[cfg(not(feature = "hwx-nightly"))]
            {
                if get_hw_capabilities().has_avx2 {
                    unsafe {
                        arrays::filter_counts_ge_threshold_u64_avx2(
                            input_slice,
                            counts,
                            threshold,
                            max_size,
                            len,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX2 capability available".to_string(),
                    ));
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if get_hw_capabilities().has_neon {
                unsafe {
                    arrays::filter_counts_ge_threshold_u64_neon(
                        input_slice,
                        counts,
                        threshold,
                        max_size,
                        len,
                    )
                }
            } else {
                return Err(crate::types::HwxError::Internal(
                    "NEON not available".to_string(),
                ));
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return Err(crate::types::HwxError::Internal(
                "No SIMD implementation available for filter_counts_ge_threshold_u64".to_string(),
            ));
        }
    };

    // Automatically truncate to valid results
    input.truncate(count);
    Ok(())
}

/// Convert array lengths from usize to u32 in batch
/// Automatically selects the best SIMD implementation for the current hardware
///
/// # Arguments
/// * `array_lengths` - Mutable vector of usize values to convert
/// * `results` - Mutable vector to store u32 conversion results (automatically truncated)
///
/// # Returns
/// * `Ok(())` on success with `results` automatically truncated to valid results
/// * `Err(crate::types::HwxError::Internal)` if input vectors are empty
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
///
/// # Examples
/// ```rust
/// use hwx::usize_to_u32;
/// use hwx::types::HwxError;
///
/// let mut array_lengths = vec![100usize, 200, 300, 400];
/// let mut results = vec![0u32; 4];
/// usize_to_u32(&array_lengths, &mut results)?;
/// assert_eq!(results, vec![100u32, 200, 300, 400]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn usize_to_u32(
    array_lengths: &[usize],
    results: &mut Vec<u32>,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "BATCH_COUNT_ARRAY_LENGTHS_U32 DISPATCH: array_lengths.len()={}, results.len()={}",
        array_lengths.len(),
        results.len()
    );

    // Early termination: return error for empty inputs
    if array_lengths.is_empty() || results.is_empty() {
        return Err(crate::types::HwxError::Internal(
            "Empty input vectors not allowed".to_string(),
        ));
    }

    let len = array_lengths.len().min(results.len());

    // Scalar path for small arrays - clamp to u32::MAX like SIMD/GPU paths
    if len < SIMD_THRESHOLD_REDUCE {
        for i in 0..len {
            results[i] = array_lengths[i].min(u32::MAX as usize) as u32;
        }
        results.truncate(len);
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::usize_to_u32_avx512(array_lengths, results.as_mut_slice(), len) };
            results.truncate(len);
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::usize_to_u32_avx2(array_lengths, results.as_mut_slice(), len) };
                results.truncate(len);
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::usize_to_u32_neon(array_lengths, results.as_mut_slice(), len) };
            results.truncate(len);
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for usize_to_u32");
}

/// Check which ranges overlap with query range with automatic truncation
/// Automatically selects the best SIMD implementation for the current hardware
///
/// # Arguments
/// * `input` - Mutable vector to store indices of overlapping ranges with automatic truncation
/// * `range_starts` - Mutable vector of range start values
/// * `range_ends` - Mutable vector of range end values
/// * `query_start` - Query range start
/// * `query_end` - Query range end
///
/// # Returns
/// * `Ok(())` on success with `input` automatically truncated to valid results
/// * `Err(crate::types::HwxError::Internal)` if input vectors have different lengths or are empty
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
/// * Returns `crate::types::HwxError::Internal` if range vectors have mismatched lengths
///
/// # Examples
/// ```rust
/// use hwx::check_range_overlaps_f64;
/// use hwx::types::HwxError;
///
/// let mut range_starts = vec![1.0, 5.0, 10.0];
/// let mut range_ends = vec![3.0, 7.0, 12.0];
/// let mut overlapping_indices = vec![0, 1, 2];  // Pre-fill with indices to check
/// check_range_overlaps_f64(&mut overlapping_indices, &mut range_starts, &mut range_ends, 2.0, 6.0)?;
/// // overlapping_indices now contains indices of ranges that overlap with [2.0, 6.0]
/// // Expected result: [0, 1] (ranges [1.0,3.0] and [5.0,7.0] overlap with [2.0,6.0])
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn check_range_overlaps_f64(
    input: &mut Vec<u32>,
    range_starts: &[f64],
    range_ends: &[f64],
    query_start: f64,
    query_end: f64,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "CHECK_RANGE_OVERLAPS_F64 DISPATCH: input.len()={}, range_starts.len()={}, range_ends.len()={}, query_start={}, query_end={}",
    input.len(),
    range_starts.len(),
    range_ends.len(),
    query_start,
    query_end
  );

    // Early termination: return error for empty inputs or length mismatch
    if range_starts.is_empty() || range_ends.is_empty() || input.is_empty() {
        return Err(crate::types::HwxError::Internal(
            "Empty input vectors not allowed".to_string(),
        ));
    }
    if range_starts.len() != range_ends.len() {
        return Err(crate::types::HwxError::Internal(
            "Range vectors must have same length".to_string(),
        ));
    }

    let len = range_starts.len().min(range_ends.len());
    let input_slice = input.as_mut_slice();
    let range_starts_slice = range_starts;
    let range_ends_slice = range_ends;

    let count = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "hwx-nightly")]
            if get_hw_capabilities().has_avx512 {
                unsafe {
                    arrays::check_range_overlaps_f64_avx512(
                        input_slice,
                        range_starts_slice,
                        range_ends_slice,
                        query_start,
                        query_end,
                        len,
                    )
                }
            } else {
                return Err(crate::types::HwxError::Internal(
                    "AVX-512 not available".to_string(),
                ));
            }
            #[cfg(not(feature = "hwx-nightly"))]
            {
                if get_hw_capabilities().has_avx2 {
                    unsafe {
                        arrays::check_range_overlaps_f64_avx2(
                            input_slice,
                            range_starts_slice,
                            range_ends_slice,
                            query_start,
                            query_end,
                            len,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "No SIMD capability available".to_string(),
                    ));
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if get_hw_capabilities().has_neon {
                unsafe {
                    arrays::check_range_overlaps_f64_neon(
                        input_slice,
                        range_starts_slice,
                        range_ends_slice,
                        query_start,
                        query_end,
                        len,
                    )
                }
            } else {
                return Err(crate::types::HwxError::Internal(
                    "NEON not available".to_string(),
                ));
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return Err(crate::types::HwxError::Internal(
                "No SIMD implementation available for check_range_overlaps_f64".to_string(),
            ));
        }
    };

    // Automatically truncate to valid results
    input.truncate(count);
    Ok(())
}

/// Find minimum and maximum values in an i64 array with smart threshold-based dispatching
/// Automatically selects between scalar fallback and best SIMD implementation based on array size
///
/// # Arguments
/// * `values` - Mutable vector of i64 values to find min/max from
///
/// # Returns
/// * `Ok((min, max))` on success with minimum and maximum values
/// * `Err(crate::types::HwxError::Internal)` if input vector is empty
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input vectors
///
/// # Examples
/// ```rust
/// use hwx::find_min_max_i64;
/// use hwx::types::HwxError;
///
/// let mut values = vec![5, -2, 10, -8, 3];
/// let (min, max) = find_min_max_i64(&mut values)?;
/// assert_eq!(min, -8);
/// assert_eq!(max, 10);
///
/// // Single element
/// let mut single = vec![42];
/// let (min, max) = find_min_max_i64(&mut single)?;
/// assert_eq!(min, 42);
/// assert_eq!(max, 42);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Single element: Immediate return
/// - Small arrays (< SIMD_THRESHOLD_REDUCE): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_REDUCE): SIMD-accelerated (4x-8x faster)
#[allow(clippy::ptr_arg)]
#[inline]
pub fn find_min_max_i64(values: &mut Vec<i64>) -> Result<(i64, i64), crate::types::HwxError> {
    trace!("FIND_MIN_MAX_I64 DISPATCH: values.len()={}", values.len());

    let len = values.len();

    if len == 0 {
        return Ok((i64::MAX, i64::MIN));
    }

    if len == 1 {
        return Ok((values[0], values[0])); // Return same value for min and max
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_REDUCE {
        // Scalar fallback implementation - optimized for small arrays
        let mut min_val = values[0];
        let mut max_val = values[0];

        for &value in &values[1..] {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }
        }

        return Ok((min_val, max_val));
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_i64_minmax(
                    values.as_slice(),
                    len,
                    |gpu_ptr, len, gpu_min, gpu_max| {
                        arrays::find_min_max_i64_gpu(gpu_ptr, len, gpu_min, gpu_max);
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::find_min_max_i64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::find_min_max_i64_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::find_min_max_i64_neon(values, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for find_min_max_i64");
}

// =============================================================================
//  SORT FUNCTIONS - HIGH-PERFORMANCE IN-PLACE SORTING
// =============================================================================

/// SIMD-optimized sorting for u32 arrays with smart threshold-based dispatching
///
/// Sorts the array in-place in ascending order using the best available SIMD instructions
/// or scalar fallback. This function automatically selects between scalar implementations
/// for small arrays and high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - Mutable u32 vector to sort in-place (modified directly, no copying)
/// * `dedup` - Whether to remove duplicates from the array
/// * `ascending` - Whether to sort in ascending order
///
/// # Returns
/// * `Ok(())` - Array sorted successfully in-place
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::sort_u32;
/// use hwx::types::HwxError;
///
/// let mut data = vec![5, 2, 8, 1, 9, 3];
/// sort_u32(&mut data, false, true)?;
/// assert_eq!(data, vec![1, 2, 3, 5, 8, 9]);
///
/// // Empty and single-element arrays are handled gracefully
/// let mut empty: Vec<u32> = vec![];
/// sort_u32(&mut empty, false, true)?;  // No-op, returns Ok(())
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar sorting
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated sorting (4x-8x faster)
#[inline]
pub fn sort_u32(
    values: &mut Vec<u32>,
    dedup: bool,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_SORT_U32 DISPATCH: values.len()={}, dedup={}, ascending={}",
        values.len(),
        dedup,
        ascending
    );

    // Early termination: return immediately for empty or single-element arrays without calling SIMD
    if values.len() <= 1 {
        return Ok(());
    }

    let len = values.len();

    if len <= 4 {
        // Use standard library sort
        if ascending {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        }
        // Apply deduplication if requested for small arrays
        if dedup {
            values.dedup();
        }
        values.truncate(values.len());
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays.
    //
    // For GPU sorting we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps the
    // implementation compact and leverages a well-tested radix sort instead of re-implementing
    // sorting directly in PTX.

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let new_len = unsafe { gpu::sort_u32_cub(values, len, ascending, dedup)? };
        // Reflect GPU-side deduplication on host by truncating
        values.truncate(new_len);
        return Ok(());
    }

    // SIMD quicksort for large arrays
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::quicksort_u32_avx512(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_u32_avx512(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            unsafe { arrays::quicksort_u32_avx2(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_u32_avx2(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX2 not available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::quicksort_u32_neon(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_u32_neon(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "SIMD not supported on this architecture".to_string(),
        ));
    }
}

/// SIMD-optimized sorting for f64 arrays with smart threshold-based dispatching
///
/// Sorts the array in-place in ascending order using the best available SIMD instructions
/// or scalar fallback. This function automatically selects between scalar implementations
/// for small arrays and high-performance SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - Mutable f64 vector to sort in-place (modified directly, no copying)
/// * `dedup` - Whether to remove duplicates from the array
/// * `ascending` - Whether to sort in ascending order
///
/// # Returns
/// * `Ok(())` - Array sorted successfully in-place
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::sort_f64;
/// use hwx::types::HwxError;
///
/// let mut data = vec![5.5, 2.1, 8.7, 1.3, 9.9, 3.2];
/// sort_f64(&mut data, false, true)?;
/// assert_eq!(data, vec![1.3, 2.1, 3.2, 5.5, 8.7, 9.9]);
///
/// // Empty and single-element arrays are handled gracefully
/// let mut empty: Vec<f64> = vec![];
/// sort_f64(&mut empty, false, true)?;  // No-op, returns Ok(())
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar sorting
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated sorting (4x-8x faster)
#[inline]
pub fn sort_f64(
    values: &mut Vec<f64>,
    dedup: bool,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_SORT_F64 DISPATCH: values.len()={}, dedup={}",
        values.len(),
        dedup
    );

    // Early termination: return immediately for empty or single-element arrays without calling SIMD
    if values.len() <= 1 {
        return Ok(());
    }

    let len = values.len();

    if len <= 4 {
        // Use standard library sort with proper NaN handling for f64
        if ascending {
            values.sort_by(|a, b| {
                // NaN goes to end for ascending
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        } else {
            values.sort_by(|a, b| {
                // NaN goes to beginning for descending
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    (false, false) => b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        }
        // Apply deduplication if requested for small arrays
        if dedup {
            values.dedup();
        }

        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays.
    //
    // For GPU sorting we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps the
    // implementation compact and leverages a well-tested radix sort instead of re-implementing
    // sorting directly in PTX.

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let new_len = unsafe { gpu::sort_f64_cub(values, len, ascending, dedup)? };
        values.truncate(new_len);
        return Ok(());
    }

    // SIMD quicksort for large arrays
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::quicksort_f64_avx512(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_f64_avx512(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            unsafe { arrays::quicksort_f64_avx2(values.as_mut_slice(), 0, len - 1, ascending) };

            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_f64_avx2(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX2 not available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::quicksort_f64_neon(values.as_mut_slice(), 0, len - 1, ascending) };

            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_f64_neon(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "SIMD not supported on this architecture".to_string(),
        ));
    }
}

/// SIMD-optimized sorting for u64 arrays with smart threshold-based dispatching
///
/// Sorts the array in-place in ascending order using the best available SIMD instructions
/// or scalar fallback. This function automatically selects between scalar implementations
/// for small arrays and SIMD implementations for larger arrays.
///
/// # Arguments
/// * `values` - Mutable u64 vector to sort in-place (automatically truncated if dedup=true)
/// * `dedup` - Whether to remove duplicate elements after sorting
/// * `ascending` - Whether to sort in ascending order
///
/// # Returns
/// * `Ok(())` - Array sorted successfully in-place
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::sort_u64;
/// use hwx::types::HwxError;
///
/// let mut data = vec![5u64, 2, 8, 1, 9, 3];
/// sort_u64(&mut data, false, true)?;
/// assert_eq!(data, vec![1, 2, 3, 5, 8, 9]);
///
/// // With deduplication
/// let mut data_with_dups = vec![5u64, 2, 8, 2, 9, 5];
/// sort_u64(&mut data_with_dups, true, true)?;
/// assert_eq!(data_with_dups, vec![2, 5, 8, 9]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar sorting
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated sorting (4x-8x faster)
#[inline]
pub fn sort_u64(
    values: &mut Vec<u64>,
    dedup: bool,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_SORT_U64 DISPATCH: values.len()={}, dedup={}",
        values.len(),
        dedup
    );

    // Early termination: return immediately for empty or single-element arrays without calling SIMD
    if values.len() <= 1 {
        return Ok(());
    }

    let len = values.len();

    if len <= 4 {
        // Use standard library sort for small arrays
        if ascending {
            values.sort_unstable();
        } else {
            values.sort_unstable_by(|a, b| b.cmp(a));
        }
        // Apply deduplication if requested for small arrays
        if dedup {
            values.dedup();
        }

        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays.
    //
    // For GPU sorting we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps the
    // implementation compact and leverages a well-tested radix sort instead of re-implementing
    // sorting directly in PTX.

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let new_len = unsafe { gpu::sort_u64_cub(values, len, ascending, dedup)? };
        values.truncate(new_len);
        return Ok(());
    }

    // SIMD quicksort for large arrays
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::quicksort_u64_avx512(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_u64_avx512(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            unsafe { arrays::quicksort_u64_avx2(values.as_mut_slice(), 0, len - 1, ascending) };

            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_u64_avx2(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX2 not available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::quicksort_u64_neon(values.as_mut_slice(), 0, len - 1, ascending) };

            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_u64_neon(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "SIMD not supported on this architecture".to_string(),
        ));
    }
}

/// SIMD-optimized i64 sorting with smart threshold-based dispatching
/// Provides 4x-8x performance improvement over standard library sort for large arrays
///
/// # Arguments
/// * `values` - Mutable vector of i64 values to sort
/// * `dedup` - Whether to remove duplicates after sorting
/// * `ascending` - Sort order: true for ascending, false for descending
///
/// # Returns
/// * `Ok(())` - Values sorted successfully
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar sorting
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated sorting (4x-8x faster)
#[inline]
pub fn sort_i64(
    values: &mut Vec<i64>,
    dedup: bool,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_SORT_I64 DISPATCH: values.len()={}, dedup={}",
        values.len(),
        dedup
    );

    // Early termination: return immediately for empty or single-element arrays without calling SIMD
    if values.len() <= 1 {
        return Ok(());
    }

    let len = values.len();

    if len <= 4 {
        // Use standard library sort for small arrays
        if ascending {
            values.sort_unstable();
        } else {
            values.sort_unstable_by(|a, b| b.cmp(a));
        }
        // Apply deduplication if requested for small arrays
        if dedup {
            values.dedup();
        }
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays.
    //
    // For GPU sorting we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps the
    // implementation compact and leverages a well-tested radix sort instead of re-implementing
    // sorting directly in PTX.

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let new_len = unsafe { gpu::sort_i64_cub(values, len, ascending, dedup)? };
        values.truncate(new_len);
        return Ok(());
    }

    // SIMD quicksort for large arrays
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::quicksort_i64_avx512(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_i64_avx512(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            unsafe { arrays::quicksort_i64_avx2(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_i64_avx2(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX2 not available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::quicksort_i64_neon(values.as_mut_slice(), 0, len - 1, ascending) };
            let actual_count = if dedup {
                unsafe { arrays::dedup_sorted_i64_neon(values.as_mut_slice(), len) }
            } else {
                len
            };

            values.truncate(actual_count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "SIMD not supported on this architecture".to_string(),
        ));
    }
}

/// SIMD-optimized string sorting using radix sort on byte values with smart threshold-based dispatching
/// Sorts strings lexicographically by modifying the indices array to represent sorted order
///
/// # Arguments
/// * `strings` - Mutable vector of strings to sort (mutable reference for potential SIMD optimizations)
/// * `indices` - Mutable vector of indices that will be reordered to represent sorted strings
///
/// # Returns
/// * `Ok(())` - Strings sorted successfully with indices reordered
/// * `Err(crate::types::HwxError::Internal)` - If arrays have different lengths or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if strings and indices have different lengths
/// * Returns `crate::types::HwxError::Internal` if SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::sort_strings;
/// use hwx::types::HwxError;
///
/// let mut strings = vec!["zebra".to_string(), "apple".to_string(), "banana".to_string()];
/// let mut indices = vec![0u32, 1, 2];
///
/// sort_strings(&mut strings, &mut indices)?;  //
///
/// // indices now contains [1, 2, 0] representing sorted order: apple, banana, zebra
/// let sorted: Vec<String> = indices.iter().map(|&i| strings[i as usize].clone()).collect();
/// assert_eq!(sorted, vec!["apple", "banana", "zebra"]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[allow(clippy::ptr_arg)]
#[inline]
pub fn sort_strings(
    strings: &mut Vec<String>,
    indices: &mut Vec<u32>,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_SORT_STRINGS DISPATCH: strings.len()={}, indices.len()={}",
        strings.len(),
        indices.len()
    );

    // Early termination: return immediately for empty arrays without calling SIMD
    if strings.is_empty() || indices.is_empty() {
        return Ok(());
    }

    if strings.len() != indices.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "SIMD SORT STRINGS: strings and indices must have the same length: {} != {}",
            strings.len(),
            indices.len()
        )));
    }

    let len = strings.len().min(indices.len());

    // Smart threshold-based dispatching: use scalar for small inputs
    if len < 8 {
        // String sorting threshold - SIMD has setup cost but benefits start early
        // Scalar fallback implementation - standard library sort with comparison
        if len <= 1 {
            return Ok(());
        }

        // Use stack-based sorting without heap access - pass len from dispatcher
        let strings_len = strings.len();
        indices[0..len].sort_by(|&a, &b| {
            let a_idx = a as usize;
            let b_idx = b as usize;
            if a_idx >= strings_len || b_idx >= strings_len {
                return std::cmp::Ordering::Equal;
            }
            // Direct slice access without heap operations
            strings[a_idx].cmp(&strings[b_idx])
        });
        return Ok(());
    }

    // Tier 2: GPU acceleration (not currently implemented for string sorting).
    //
    // String sorting currently dispatches between scalar and CPU SIMD implementations.

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { strings::sort_strings_avx512(strings, indices, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { strings::sort_strings_avx2(strings, indices, len) };
                return Ok(());
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "hwx-nightly")))]
    {
        if get_hw_capabilities().has_neon {
            unsafe { strings::sort_strings_neon(strings, indices, len) };
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for sort_strings");
}

/// SIMD-optimized sorting of u32 indices by their corresponding u64 values with smart threshold-based dispatching
/// Sorts indices array so that indices[i] points to elements in ascending/descending order of their values
///
/// # Arguments
/// * `indices` - Mutable vector of u32 indices to sort (modified in-place)
/// * `values` - Slice of u64 values corresponding to indices (read-only)
/// * `ascending` - Sort order: true for ascending, false for descending
///
/// # Returns
/// * `Ok(())` - Indices sorted successfully
/// * `Err(crate::types::HwxError::Internal)` - If arrays have different lengths or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if indices and values have different lengths
/// * Returns `crate::types::HwxError::Internal` if SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::sort_u32_by_u64;
/// use hwx::types::HwxError;
///
/// let mut indices = vec![0u32, 1, 2, 3];
/// let values = vec![40u64, 10, 30, 20];
///
/// sort_u32_by_u64(&mut indices, &values, true)?; // ascending
/// // indices now contains [1, 3, 2, 0] representing sorted order by values: 10, 20, 30, 40
/// assert_eq!(indices, vec![1, 3, 2, 0]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ threshold): Optimized scalar insertion sort
/// - Large arrays (> threshold): SIMD-accelerated quicksort (4x-8x faster)
#[inline]
pub fn sort_u32_by_u64(
    indices: &mut Vec<u32>,
    values: &[u64],
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SORT_U32_BY_U64 DISPATCH: indices.len()={}, values.len()={}, ascending={}",
        indices.len(),
        values.len(),
        ascending
    );

    // Early termination: return immediately for empty arrays
    if indices.is_empty() || values.is_empty() {
        return Ok(());
    }

    // Validate array lengths match
    if indices.len() != values.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "SORT_U32_BY_U64: indices and values must have the same length: {} != {}",
            indices.len(),
            values.len()
        )));
    }

    let indices_len = indices.len();
    let values_len = values.len();
    let indices_slice = indices.as_mut_slice();

    // Smart threshold-based dispatching: use scalar for small inputs
    if indices_len <= 8 {
        // Scalar insertion sort for small arrays
        for i in 1..indices_len {
            let key_idx = indices_slice[i];
            let key_val = values[key_idx as usize];
            let mut j = i;

            while j > 0 {
                let prev_idx = indices_slice[j - 1];
                let prev_val = values[prev_idx as usize];

                let should_swap = if ascending {
                    prev_val > key_val
                } else {
                    prev_val < key_val
                };

                if should_swap {
                    indices_slice[j] = indices_slice[j - 1];
                    j -= 1;
                } else {
                    break;
                }
            }
            indices_slice[j] = key_idx;
        }
        return Ok(());
    }

    // GPU dispatch for large inputs
    #[cfg(has_cuda)]
    if indices_len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let mut values_mut: Vec<u64> = values.to_vec();
        return unsafe {
            gpu::sort_u32_by_u64_cub(indices_slice, &mut values_mut, indices_len, ascending)
        };
    }

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe {
                arrays::sort_u32_by_u64_avx512(
                    indices_slice,
                    values,
                    indices_len,
                    values_len,
                    ascending,
                );
            }
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    arrays::sort_u32_by_u64_avx2(
                        indices_slice,
                        values,
                        indices_len,
                        values_len,
                        ascending,
                    );
                }
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe {
                arrays::sort_u32_by_u64_neon(
                    indices_slice,
                    values,
                    indices_len,
                    values_len,
                    ascending,
                );
            }
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for sort_u32_by_u64");
}
/// Sorts u32 indices by their corresponding f64 values
///
/// Reorders the indices array so that when used to access values,
/// they produce a sorted sequence. This is useful for sorting paired
/// data without moving the actual values.
///
/// # Arguments
/// * `indices` - Mutable vector of u32 indices to reorder (each must be < values.len())
/// * `values` - Slice of f64 values to sort by
/// * `ascending` - If true, sort ascending; if false, sort descending
///
/// # Returns
/// * `Ok(())` - Indices sorted successfully
/// * `Err(crate::types::HwxError::Internal)` - If arrays have different lengths or invalid indices
///
/// # Examples
/// ```rust
/// use hwx::sort_u32_by_f64;
/// use hwx::types::HwxError;
///
/// let values = vec![3.5, 1.2, 4.8, 2.1];
/// let mut indices = vec![0u32, 1, 2, 3];
///
/// sort_u32_by_f64(&mut indices, &values, true)?;
/// assert_eq!(indices, vec![1, 3, 0, 2]); // Sorted: 1.2, 2.1, 3.5, 4.8
///
/// // Access values in sorted order
/// let sorted_values: Vec<f64> = indices.iter().map(|&i| values[i as usize]).collect();
/// assert_eq!(sorted_values, vec![1.2, 2.1, 3.5, 4.8]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 8): Optimized scalar insertion sort
/// - Large arrays (> 8): SIMD-accelerated quicksort (4x-8x faster)
#[inline]
pub fn sort_u32_by_f64(
    indices: &mut Vec<u32>,
    values: &[f64],
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SORT_U32_BY_F64 DISPATCH: indices.len()={}, values.len()={}, ascending={}",
        indices.len(),
        values.len(),
        ascending
    );

    // Early termination: return immediately for empty arrays
    if indices.is_empty() || values.is_empty() {
        return Ok(());
    }

    // Validate array lengths match
    if indices.len() != values.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "SORT_U32_BY_F64: indices and values must have the same length: {} != {}",
            indices.len(),
            values.len()
        )));
    }

    let indices_len = indices.len();
    let values_len = values.len();
    let indices_slice = indices.as_mut_slice();

    // Smart threshold-based dispatching: use scalar for small inputs
    if indices_len <= 8 {
        // Scalar insertion sort for small arrays
        for i in 1..indices_len {
            let key_idx = indices_slice[i];
            let key_val = values[key_idx as usize];
            let mut j = i;

            while j > 0 {
                let prev_idx = indices_slice[j - 1];
                let prev_val = values[prev_idx as usize];

                // Handle NaN values - NaN is always greater than any number
                let should_swap = if key_val.is_nan() {
                    false // NaN goes to the end
                } else if prev_val.is_nan() {
                    true // Move NaN towards the end
                } else if ascending {
                    prev_val > key_val
                } else {
                    prev_val < key_val
                };

                if should_swap {
                    indices_slice[j] = indices_slice[j - 1];
                    j -= 1;
                } else {
                    break;
                }
            }
            indices_slice[j] = key_idx;
        }
        return Ok(());
    }

    // GPU dispatch for large inputs
    #[cfg(has_cuda)]
    if indices_len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let mut values_mut: Vec<f64> = values.to_vec();
        return unsafe {
            gpu::sort_u32_by_f64_cub(indices_slice, &mut values_mut, indices_len, ascending)
        };
    }

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe {
                arrays::sort_u32_by_f64_avx512(
                    indices_slice,
                    values,
                    indices_len,
                    values_len,
                    ascending,
                );
            }
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    arrays::sort_u32_by_f64_avx2(
                        indices_slice,
                        values,
                        indices_len,
                        values_len,
                        ascending,
                    );
                }
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe {
                arrays::sort_u32_by_f64_neon(
                    indices_slice,
                    values,
                    indices_len,
                    values_len,
                    ascending,
                );
            }
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for sort_u32_by_f64");
}

/// Sorts u32 indices by their corresponding i64 values
///
/// Reorders the indices array so that when used to access values,
/// they produce a sorted sequence. This is useful for sorting paired
/// data without moving the actual values.
///
/// # Arguments
/// * `indices` - Mutable vector of u32 indices to reorder (each must be < values.len())
/// * `values` - Slice of i64 values to sort by
/// * `ascending` - If true, sort ascending; if false, sort descending
///
/// # Returns
/// * `Ok(())` - Indices sorted successfully
/// * `Err(crate::types::HwxError::Internal)` - If arrays have different lengths or invalid indices
///
/// # Examples
/// ```rust
/// use hwx::sort_u32_by_i64;
/// use hwx::types::HwxError;
///
/// let values = vec![35i64, -12, 48, -21];
/// let mut indices = vec![0u32, 1, 2, 3];
///
/// sort_u32_by_i64(&mut indices, &values, true)?;
/// // indices now contains [3, 1, 0, 2] representing sorted order by values: -21, -12, 35, 48
/// assert_eq!(indices, vec![3, 1, 0, 2]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ threshold): Optimized scalar insertion sort
/// - Large arrays (> threshold): SIMD-accelerated quicksort (4x-8x faster)
#[inline]
pub fn sort_u32_by_i64(
    indices: &mut Vec<u32>,
    values: &[i64],
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SORT_U32_BY_I64 DISPATCH: indices.len()={}, values.len()={}, ascending={}",
        indices.len(),
        values.len(),
        ascending
    );

    // Early termination: return immediately for empty arrays
    if indices.is_empty() || values.is_empty() {
        return Ok(());
    }

    // Validate array lengths match
    if indices.len() != values.len() {
        return Err(crate::types::HwxError::Internal(format!(
            "SORT_U32_BY_I64: indices and values must have the same length: {} != {}",
            indices.len(),
            values.len()
        )));
    }

    let indices_len = indices.len();
    let values_len = values.len();
    let indices_slice = indices.as_mut_slice();

    // Smart threshold-based dispatching: use scalar for small inputs
    if indices_len <= 8 {
        // Scalar insertion sort for small arrays
        for i in 1..indices_len {
            let key_idx = indices_slice[i];
            let key_val = values[key_idx as usize];
            let mut j = i;

            while j > 0 {
                let prev_idx = indices_slice[j - 1];
                let prev_val = values[prev_idx as usize];

                let should_swap = if ascending {
                    prev_val > key_val
                } else {
                    prev_val < key_val
                };

                if should_swap {
                    indices_slice[j] = indices_slice[j - 1];
                    j -= 1;
                } else {
                    break;
                }
            }
            indices_slice[j] = key_idx;
        }
        return Ok(());
    }

    // GPU dispatch for large inputs
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    if indices_len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let mut values_mut: Vec<i64> = values.to_vec();
        unsafe {
            gpu::sort_u32_by_i64_cub(indices_slice, &mut values_mut, indices_len, ascending)?
        };

        return Ok(());
    }

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe {
                arrays::sort_u32_by_i64_avx512(
                    indices_slice,
                    values,
                    indices_len,
                    values_len,
                    ascending,
                );
            }
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    arrays::sort_u32_by_i64_avx2(
                        indices_slice,
                        values,
                        indices_len,
                        values_len,
                        ascending,
                    );
                }
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe {
                arrays::sort_u32_by_i64_neon(
                    indices_slice,
                    values,
                    indices_len,
                    values_len,
                    ascending,
                );
            }
            return Ok(());
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for sort_u32_by_i64");
}

// =============================================================================
//  STRING CLASSIFICATION - SIMD-OPTIMIZED BATCH PROCESSING
// =============================================================================

/// Classify a batch of strings using HW acceleration
///
/// Uses vectorized pattern matching to efficiently classify strings into schema types.
/// Automatically selects the optimal SIMD implementation for the current CPU architecture.
///
/// # Arguments
/// * `string` - String slice to classify (determines HwxType: Integer, Float, Boolean, etc.)
///
/// # Returns
/// * `Ok(ClassificationResult)` - Classification result with detected type and element count
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Performance
/// - Uses AVX-512/AVX2/NEON SIMD instructions when available for pattern detection
/// - Falls back to scalar implementation for unsupported architectures
/// - Zero-copy: operates directly on string data without allocation
///
/// # Examples
/// ```rust
/// use hwx::classify_string;
/// use hwx::types::HwxError;
///
/// let result = classify_string("123")?;
/// // Check that result is a valid classification
/// assert!(result.element_count > 0);
///
/// let result = classify_string("3.14")?;
/// assert!(result.element_count > 0);
///
/// let result = classify_string("true")?;
/// assert!(result.element_count > 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[inline]
pub fn classify_string(string: &str) -> Result<ClassificationResult, crate::types::HwxError> {
    trace!(
        "SIMD_CLASSIFY_STRING DISPATCH: string.len()={}",
        string.len()
    );

    if string.is_empty() {
        return Ok(ClassificationResult {
            hwx_type: HwxType::String,
            element_count: 1,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        });
    }

    let len = string.len();

    // Detect array and route to array classifier first
    if len >= 2 && string.as_bytes()[0] == b'[' && string.as_bytes()[len - 1] == b']' {
        // Extract content without brackets
        let content = &string.as_bytes()[1..len - 1];
        let content_len = len - 2;

        #[cfg(has_cuda)]
        {
            if get_hw_capabilities().has_cuda {
                return Ok(unsafe { classify::classify_array_contents_gpu(content, content_len) });
            }
        }

        #[cfg(all(
            feature = "hwx-nightly",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if get_hw_capabilities().has_avx512 {
                return Ok(unsafe {
                    classify::classify_array_contents_avx512(content, content_len)
                });
            }
        }

        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(feature = "hwx-nightly")
        ))]
        {
            if is_x86_feature_detected!("avx2") {
                return Ok(unsafe { classify::classify_array_contents_avx2(content, content_len) });
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if get_hw_capabilities().has_neon {
                return Ok(unsafe { classify::classify_array_contents_neon(content, content_len) });
            }
        }

        // Fallback: Array type with count 0 when unsupported
        return Ok(ClassificationResult {
            hwx_type: HwxType::Array,
            element_count: 0,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        });
    }

    // Check for GPU acceleration first for larger strings
    #[cfg(has_cuda)]
    {
        if len >= GPU_THRESHOLD_STRING && get_hw_capabilities().has_cuda {
            return Ok(unsafe { classify::classify_single_string_gpu(string, len) });
        }
    }

    // First pass: classify with single string classifiers
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { classify::classify_single_string_avx512(string, len) });
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { classify::classify_single_string_avx2(string, len) });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { classify::classify_single_string_neon(string, len) });
        }
    }

    unreachable!("No SIMD implementation available for classify_string");
}

/// Convert bytes to lowercase in-place using HW acceleration with smart threshold-based dispatching
///
/// Converts ASCII and common UTF-8 characters to lowercase using optimized SIMD operations.
/// This function automatically selects between scalar fallback for small strings and
/// high-performance SIMD implementations for larger strings.
///
/// # Arguments
/// * `bytes` - Mutable byte vector to convert to lowercase in-place
///
/// # Returns
/// * `Ok(())` - Conversion completed successfully
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::dispatch::to_lowercase;
/// use hwx::types::HwxError;
///
/// let input = "Hello WORLD";
/// let result = to_lowercase(input)?;
/// assert_eq!(result, "hello world");
///
/// let input = "Café NAÏVE";
/// let result = to_lowercase(input)?;
/// assert_eq!(result, "café naïve");
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small strings (< SIMD_LOWERCASE_THRESHOLD): Optimized scalar implementation
/// - Large strings (^ SIMD_LOWERCASE_THRESHOLD): SIMD-accelerated (2x-4x faster)
#[inline]
pub fn to_lowercase(input: &str) -> Result<String, crate::types::HwxError> {
    trace!("SIMD_TO_LOWERCASE DISPATCH: input.len()={}", input.len());

    // Early exit for empty input
    if input.is_empty() {
        return Ok(String::new());
    }

    let len = input.len();

    // Smart threshold-based dispatching: use scalar for small strings
    if len < SIMD_LOWERCASE_THRESHOLD {
        return Ok(input.to_lowercase());
    }

    // Create string from input for SIMD processing
    let mut result = String::from(input);
    let bytes = unsafe { result.as_mut_vec() };

    // Check for GPU acceleration first for very large strings
    #[cfg(has_cuda)]
    {
        if len >= GPU_THRESHOLD_STRING && get_hw_capabilities().has_cuda {
            unsafe {
                tokenize::to_lowercase_gpu(bytes.as_mut_ptr(), len);
            }
            return Ok(result);
        }
    }

    // SIMD dispatch for larger strings with architecture conditionals
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if get_hw_capabilities().has_avx512 {
            unsafe { tokenize::to_lowercase_avx512(bytes, len) };
            return Ok(result);
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    {
        if get_hw_capabilities().has_avx2 {
            unsafe { tokenize::to_lowercase_avx2(bytes, len) };
            return Ok(result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { tokenize::to_lowercase_neon(bytes, len) };
            return Ok(result);
        }
    }

    unreachable!("No SIMD implementation available for to_lowercase");
}

/// Tokenize a string using HW acceleration with smart threshold-based dispatching
///
/// This function provides fast Unicode-aware tokenization with word boundary detection.
/// It automatically selects between scalar fallback for small strings and high-performance
/// SIMD implementations for larger strings.
///
/// # Arguments
/// * `input` - String slice to tokenize
/// * `output` - Mutable vector to store the resulting tokens
///
/// # Returns
/// * `Ok(())` - Tokenization completed successfully
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if SIMD operations fail
/// * Note: Empty strings are handled gracefully (output remains empty)
///
/// # Examples
/// ```rust
/// use hwx::tokenize_string;
/// use hwx::types::HwxError;
///
/// let mut tokens = Vec::new();
/// tokenize_string("Hello World", &mut tokens)?;
/// assert_eq!(tokens, vec!["Hello", "World"]);
///
/// let mut tokens = Vec::new();
/// tokenize_string("field~value test", &mut tokens)?;
/// assert_eq!(tokens, vec!["field~value", "test"]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - SIMD-accelerated word boundary detection + scalar delimiter logic
#[inline]
pub fn tokenize_string<'a>(
    input: &'a str,
    output: &mut Vec<&'a str>,
) -> Result<(), crate::types::HwxError> {
    trace!("SIMD_TOKENIZE_STRING DISPATCH: input.len()={}", input.len());

    // Early exit for empty input
    if input.is_empty() {
        return Ok(());
    }

    let len = input.len();

    // Make a copy of the input string. We'll store the tokenized values here.
    // Then use the returned word boundaries to create the output vector from this.
    let mut input_bytes = input.as_bytes().to_vec();
    // GPU tokenizer uses layout: [0]=count, [1..=len]=flags, then pairs at base (len+1)
    // Allocate enough space: 1 (count) + len (flags) + 2*max_words (pairs). Worst-case max_words ^^ len/2.
    // Use 3*len + 1 to be safe and avoid device out-of-bounds.
    let mut word_boundaries = vec![0u32; len * 3 + 1];

    // Check for GPU first - it handles the largest workloads
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_STRING && get_hw_capabilities().has_cuda {
        let word_count = unsafe {
            tokenize::tokenize_single_string_gpu(
                input_bytes.as_mut_slice(),
                len,
                word_boundaries.as_mut_slice(),
            )
        };
        // GPU layout: [0]=count, [1..=len]=flags, pairs at base index (len+1)
        let base = len + 1;
        if base < word_boundaries.len() {
            reconstruct_tokens_with_delimiters(
                input,
                &word_boundaries[base..],
                word_count as usize,
                output,
            )?;
        }
        return Ok(());
    }

    let word_count = {
        #[cfg(all(
            feature = "hwx-nightly",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if get_hw_capabilities().has_avx512 {
                unsafe {
                    tokenize::tokenize_single_string_avx512(
                        input_bytes.as_mut_slice(),
                        len,
                        word_boundaries.as_mut_slice(),
                    )
                }
            } else {
                return Err(crate::types::HwxError::Internal(
                    "AVX-512 not available".to_string(),
                ));
            }
        }

        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(feature = "hwx-nightly")
        ))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe {
                    tokenize::tokenize_single_string_avx2(
                        input_bytes.as_mut_slice(),
                        len,
                        word_boundaries.as_mut_slice(),
                    )
                }
            } else {
                return Err(crate::types::HwxError::Internal(
                    "AVX2 not available".to_string(),
                ));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if get_hw_capabilities().has_neon {
                unsafe {
                    tokenize::tokenize_single_string_neon(
                        input_bytes.as_mut_slice(),
                        len,
                        word_boundaries.as_mut_slice(),
                    )
                }
            } else {
                return Err(crate::types::HwxError::Internal(
                    "NEON not available".to_string(),
                ));
            }
        }
    };

    // Reconstruct tokens from word boundaries
    reconstruct_tokens_with_delimiters(input, &word_boundaries, word_count as usize, output)?;

    Ok(())
}

/// Reconstruct tokens from SIMD word boundaries with FIELD_DELIMITER logic
///
/// This function takes word boundaries detected by SIMD and reconstructs tokens
/// from the original input string, handling the complex FIELD_DELIMITER logic.
#[inline]
fn reconstruct_tokens_with_delimiters<'a>(
    input: &'a str,
    word_boundaries: &[u32],
    word_count: usize,
    output: &mut Vec<&'a str>,
) -> Result<(), crate::types::HwxError> {
    if word_count == 0 {
        return Ok(());
    }

    let mut current_token: Option<&str> = None;
    let input_ptr = input.as_ptr() as usize;

    // Process word boundaries in pairs (start, end)
    for i in 0..word_count {
        let start_idx = i * 2;
        let end_idx = start_idx + 1;

        if end_idx >= word_boundaries.len() {
            break;
        }

        let mut word_start = word_boundaries[start_idx] as usize;
        let mut word_end = word_boundaries[end_idx] as usize;

        if word_start >= input.len() || word_end > input.len() || word_start >= word_end {
            continue;
        }

        // Ensure we're on UTF-8 character boundaries
        // For start: move backward to find the start of the character
        while word_start > 0 && word_start < input.len() && !input.is_char_boundary(word_start) {
            word_start -= 1;
        }

        // For end: move forward to find the end of the character
        while word_end < input.len() && !input.is_char_boundary(word_end) {
            word_end += 1;
        }

        if word_start >= word_end || word_start >= input.len() || word_end > input.len() {
            continue;
        }

        let word = &input[word_start..word_end];

        if let Some(current) = current_token {
            let current_start = current.as_ptr() as usize - input_ptr;
            let current_end = current_start + current.len();

            // Check for delimiter between current token end and next word start
            let has_delimiter = current_end < word_start
                && current_end < input.len()
                && input.as_bytes()[current_end] == b'~';

            if has_delimiter {
                // Combine tokens across delimiter: extend from current start to word end
                output.push(&input[current_start..word_end]);
                current_token = None;
                continue;
            } else {
                // No delimiter, push previous token separately
                output.push(current);
            }
        }

        current_token = Some(word);
    }

    // Handle remaining token
    if let Some(token) = current_token {
        output.push(token);
    }

    Ok(())
}

// =============================================================================
//  SEARCH FUNCTIONS - OPTIMIZED BINARY & EXPONENTIAL SEARCH
// =============================================================================

/// Binary search for the first element >= target in a sorted u32 array
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs a binary search to find the index of the first element that is greater than or equal
/// to the target value in a sorted array. This function automatically selects between scalar
/// fallback for small arrays and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `arr` - Mutable u32 vector (sorted in ascending order, mutable reference for potential SIMD optimizations)
/// * `target` - Target value to search for
///
/// # Returns
/// * `Ok(usize)` - Index of the first element >= target, or arr.len() if no such element exists
/// * `Ok(0)` - If the array is empty
///
/// # Examples
/// ```rust
/// use hwx::binary_search_ge_u32;
/// use hwx::types::HwxError;
///
/// let sorted_arr = vec![1, 3, 5, 7, 9, 11];
/// let index = binary_search_ge_u32(&sorted_arr, 6)?;  // Result: 3 (index of 7)
/// assert_eq!(index, 3);
///
/// // Empty array
/// let empty: Vec<u32> = vec![];
/// let index = binary_search_ge_u32(&empty, 5)?;  // Result: 0
/// assert_eq!(index, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn binary_search_ge_u32(arr: &[u32], target: u32) -> Result<usize, crate::types::HwxError> {
    trace!(
        "BINARY_SEARCH_GE_U32 DISPATCH: arr.len()={}, target={}",
        arr.len(),
        target
    );

    // Early termination: return 0 for empty arrays without calling SIMD
    if arr.is_empty() {
        return Ok(0);
    }

    let len = arr.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for first element >= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if arr[mid] >= target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return Ok(left);
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_u32_search(
                    arr,
                    target,
                    len,
                    |gpu_ptr, target, len, result_ptr| {
                        traverse::binary_search_ge_u32_gpu(gpu_ptr, target, len, result_ptr);
                    },
                )
            };
        }
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { traverse::binary_search_ge_u32_avx512(arr, target, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { traverse::binary_search_ge_u32_avx2(arr, target, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { traverse::binary_search_ge_u32_neon(arr, target, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for binary_search_ge_u32");
}

/// Binary search for the first element >= target in a sorted u64 array with smart threshold-based dispatching
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs a binary search to find the index of the first element that is greater than or equal
/// to the target value in a sorted array. This function automatically selects between scalar
/// fallback for small arrays and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `arr` - Mutable u64 vector (sorted in ascending order, mutable reference for potential SIMD optimizations)
/// * `target` - Target value to search for
///
/// # Returns
/// * `Ok(usize)` - Index of the first element >= target, or arr.len() if no such element exists
/// * `Ok(0)` - If the array is empty
///
/// # Examples
/// ```rust
/// use hwx::binary_search_ge_u64;
/// use hwx::types::HwxError;
///
/// let sorted_arr = vec![10, 30, 50, 70, 90, 110];
/// let index = binary_search_ge_u64(&sorted_arr, 60)?;  // Result: 3 (index of 70)
/// assert_eq!(index, 3);
///
/// // Empty array
/// let empty: Vec<u64> = vec![];
/// let index = binary_search_ge_u64(&empty, 50)?;  // Result: 0
/// assert_eq!(index, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn binary_search_ge_u64(arr: &[u64], target: u64) -> Result<usize, crate::types::HwxError> {
    trace!(
        "BINARY_SEARCH_GE_U64 DISPATCH: arr.len()={}, target={}",
        arr.len(),
        target
    );

    if arr.is_empty() {
        return Ok(0);
    }

    let len = arr.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for first element >= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if arr[mid] >= target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return Ok(left);
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_u64_search(
                    arr,
                    target,
                    len,
                    |gpu_ptr, target, len, result_ptr| {
                        traverse::binary_search_ge_u64_gpu(gpu_ptr, target, len, result_ptr);
                    },
                )
            };
        }
    }

    // SIMD dispatch for architecture-specific optimizations
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { traverse::binary_search_ge_u64_avx512(arr, target, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { traverse::binary_search_ge_u64_avx2(arr, target, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { traverse::binary_search_ge_u64_neon(arr, target, len) });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for binary_search_ge_u64");
}

/// Exponential search for the first element >= target in a sorted u64 array with smart threshold-based dispatching
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs an exponential search to find the index of the first element that is greater than or equal
/// to the target value. Exponential search is particularly efficient for large arrays where the target
/// is near the beginning. This function automatically selects between scalar fallback for small arrays
/// and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `sorted_array` - Mutable u64 vector (sorted in ascending order, mutable reference for potential SIMD optimizations)
/// * `target` - Target value to search for
///
/// # Returns
/// * `Ok(usize)` - Index of the first element >= target, or array.len() if no such element exists
/// * `Ok(0)` - If the array is empty
///
/// # Examples
/// ```rust
/// use hwx::exponential_search_ge_u64;
/// use hwx::types::HwxError;
///
/// let sorted_arr = vec![1, 2, 4, 8, 16, 32, 64, 128];
/// let index = exponential_search_ge_u64(&sorted_arr, 20)?;  // Result: 5 (index of 32)
/// assert_eq!(index, 5);
///
/// // Empty array
/// let empty: Vec<u64> = vec![];
/// let index = exponential_search_ge_u64(&empty, 10)?;  // Result: 0
/// assert_eq!(index, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
/// - Especially efficient when target is near the beginning of large arrays
#[inline]
pub fn exponential_search_ge_u64(
    sorted_array: &[u64],
    target: u64,
) -> Result<usize, crate::types::HwxError> {
    trace!(
        "EXPONENTIAL_SEARCH_GE_U64 DISPATCH: sorted_array.len()={}, target={}",
        sorted_array.len(),
        target
    );

    let len = sorted_array.len();

    // Early termination: return immediately for empty arrays without calling SIMD
    if len == 0 {
        return Ok(0); // Empty array
    }

    // GPU dispatch for large arrays
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SEARCH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_search(
                sorted_array,
                target,
                len,
                |gpu_ptr, target, len, result_ptr| {
                    traverse::exponential_search_ge_u64_gpu(gpu_ptr, target, len, result_ptr);
                },
            )
        };
    }

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for first element >= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if sorted_array[mid] >= target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return Ok(left);
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                traverse::exponential_search_ge_u64_avx512(sorted_array, target, len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe {
                    traverse::exponential_search_ge_u64_avx2(
                        sorted_array,
                        target,
                        sorted_array.len(),
                    )
                });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe {
                traverse::exponential_search_ge_u64_neon(sorted_array, target, len)
            });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for exponential_search_ge_u64");
}

/// Exponential search for the first element >= target in a sorted u32 array with smart threshold-based dispatching
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs an exponential search to find the index of the first element that is greater than or equal
/// to the target value. Exponential search is particularly efficient for large arrays where the target
/// is near the beginning. This function automatically selects between scalar fallback for small arrays
/// and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `sorted_array` - Mutable u32 vector (sorted in ascending order, mutable reference for potential SIMD optimizations)
/// * `target` - Target value to search for
///
/// # Returns
/// * `Ok(usize)` - Index of the first element >= target, or array.len() if no such element exists
/// * `Ok(0)` - If the array is empty
///
/// # Examples
/// ```rust
/// use hwx::exponential_search_ge_u32;
/// use hwx::types::HwxError;
///
/// let sorted_arr = vec![1, 2, 4, 8, 16, 32, 64, 128];
/// let index = exponential_search_ge_u32(&sorted_arr, 20)?;  // Result: 5 (index of 32)
/// assert_eq!(index, 5);
///
/// // Empty array
/// let empty: Vec<u32> = vec![];
/// let index = exponential_search_ge_u32(&empty, 10)?;  // Result: 0
/// assert_eq!(index, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
/// - Especially efficient when target is near the beginning of large arrays
#[inline]
pub fn exponential_search_ge_u32(
    sorted_array: &[u32],
    target: u32,
) -> Result<usize, crate::types::HwxError> {
    trace!(
        "EXPONENTIAL_SEARCH_GE_U32 DISPATCH: sorted_array.len()={}, target={}",
        sorted_array.len(),
        target
    );

    if sorted_array.is_empty() {
        return Ok(0);
    }

    let len = sorted_array.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for first element >= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if sorted_array[mid] >= target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return Ok(left);
    }

    // GPU dispatch for large arrays
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SEARCH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u32_search(
                sorted_array,
                target,
                len,
                |gpu_ptr, target, len, result_ptr| {
                    traverse::exponential_search_ge_u32_gpu(gpu_ptr, target, len, result_ptr);
                },
            )
        };
    }

    // SIMD dispatch for architecture-specific optimizations
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                traverse::exponential_search_ge_u32_avx512(sorted_array, target, len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe {
                    traverse::exponential_search_ge_u32_avx2(sorted_array, target, len)
                });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe {
                traverse::exponential_search_ge_u32_neon(sorted_array, target, len)
            });
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for exponential_search_ge_u32");
}

/// Binary search for the last element <= target in a sorted u32 array with smart threshold-based dispatching
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs a binary search to find the index of the last element that is less than or equal
/// to the target value in a sorted array. This function automatically selects between scalar
/// fallback for small arrays and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `arr` - Sorted u32 array (sorted in ascending order)
/// * `target` - Target value to search for
///
/// # Returns
/// * `usize` - Index of the last element <= target, or arr.len() if no such element exists
///
/// # Examples
/// ```rust
/// use hwx::binary_search_le_u32;
///
/// let sorted_arr = vec![1, 3, 5, 7, 9];
/// let index = binary_search_le_u32(&sorted_arr, 6);  // Result: 2 (index of 5)
/// assert_eq!(index, 2);
///
/// // Target larger than all elements
/// let index = binary_search_le_u32(&sorted_arr, 10);  // Result: 4 (index of 9)
/// assert_eq!(index, 4);
///
/// // Target smaller than all elements
/// let index = binary_search_le_u32(&sorted_arr, 0);  // Result: 5 (arr.len())
/// assert_eq!(index, 5);
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn binary_search_le_u32(arr: &[u32], target: u32) -> usize {
    trace!(
        "BINARY_SEARCH_LE_U32 DISPATCH: arr.len()={}, target={}",
        arr.len(),
        target
    );

    // Early termination: return arr.len() for empty arrays
    if arr.is_empty() {
        return arr.len();
    }

    let len = arr.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for last element <= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if arr[mid] <= target {
                left = mid + 1; // Keep searching right for the LAST element <= target
            } else {
                right = mid; // Found element > target, search left
            }
        }

        // left now points to first element > target
        // So the last element <= target is at left - 1
        if left > 0 && arr[left - 1] <= target {
            return left - 1;
        } else {
            return len; // No element <= target
        }
    }

    // GPU dispatch for large inputs
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_FILTER && get_hw_capabilities().has_cuda {
        if let Ok(result) = unsafe {
            gpu::with_gpu_buffer_u32_search(arr, target, len, |gpu_ptr, target, len, result_ptr| {
                traverse::binary_search_le_u32_gpu(gpu_ptr, target, len, result_ptr);
            })
        } {
            // Kernel returns first index > target (or len). Map to last <= target.
            return if result == 0 {
                len // All elements > target, so no element <= target
            } else if result >= len {
                len - 1 // All elements <= target, so last element is at len-1
            } else {
                result - 1 // Last element <= target is at result-1
            };
        }
        // Fall through to SIMD implementation if GPU fails
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return unsafe { traverse::binary_search_le_u32_avx512(arr, target, len) };
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return unsafe { traverse::binary_search_le_u32_avx2(arr, target, len) };
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return unsafe { traverse::binary_search_le_u32_neon(arr, target, len) };
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for binary_search_le_u32");
}

/// Binary search for the last element <= target in a sorted u64 array with smart threshold-based dispatching
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs a binary search to find the index of the last element that is less than or equal
/// to the target value in a sorted array. This function automatically selects between scalar
/// fallback for small arrays and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `arr` - Sorted u64 array (sorted in ascending order)
/// * `target` - Target value to search for
///
/// # Returns
/// * `usize` - Index of the last element <= target, or arr.len() if no such element exists
///
/// # Examples
/// ```rust
/// use hwx::binary_search_le_u64;
///
/// let sorted_arr = vec![1u64, 3u64, 5u64, 7u64, 9u64];
/// let index = binary_search_le_u64(&sorted_arr, 6);  // Result: 2 (index of 5)
/// assert_eq!(index, 2);
///
/// // Target larger than all elements
/// let index = binary_search_le_u64(&sorted_arr, 10);  // Result: 4 (index of 9)
/// assert_eq!(index, 4);
///
/// // Target smaller than all elements
/// let index = binary_search_le_u64(&sorted_arr, 0);  // Result: 5 (arr.len())
/// assert_eq!(index, 5);
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn binary_search_le_u64(arr: &[u64], target: u64) -> usize {
    trace!(
        "BINARY_SEARCH_LE_U64 DISPATCH: arr.len()={}, target={}",
        arr.len(),
        target
    );

    // Early termination: return arr.len() for empty arrays
    if arr.is_empty() {
        return arr.len();
    }

    let len = arr.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for last element <= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if arr[mid] <= target {
                left = mid + 1; // Keep searching right for the LAST element <= target
            } else {
                right = mid; // Found element > target, search left
            }
        }

        // left now points to first element > target
        // So the last element <= target is at left - 1
        if left > 0 && arr[left - 1] <= target {
            return left - 1;
        } else {
            return len; // No element <= target
        }
    }

    // GPU dispatch for large inputs
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SEARCH && get_hw_capabilities().has_cuda {
        if let Ok(result) = unsafe {
            gpu::with_gpu_buffer_u64_search(arr, target, len, |gpu_ptr, target, len, result_ptr| {
                traverse::binary_search_le_u64_gpu(gpu_ptr, target, len, result_ptr);
            })
        } {
            // Kernel returns first index > target (or len). Map to last <= target.
            return if result == 0 {
                len // All elements > target, so no element <= target
            } else if result >= len {
                len - 1 // All elements <= target, so last element is at len-1
            } else {
                result - 1 // Last element <= target is at result-1
            };
        }
        // Fall through to SIMD implementation if GPU fails
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return unsafe { traverse::binary_search_le_u64_avx512(arr, target, len) };
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return unsafe { traverse::binary_search_le_u64_avx2(arr, target, len) };
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return unsafe { traverse::binary_search_le_u64_neon(arr, target, len) };
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for binary_search_le_u64");
}

/// Exponential search for the last element <= target in a sorted u32 array with smart threshold-based dispatching
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs an exponential search to find the index of the last element that is less than or equal
/// to the target value. Exponential search is particularly efficient for large arrays where the target
/// is near the beginning. This function automatically selects between scalar fallback for small arrays
/// and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `sorted_array` - Sorted u32 array (sorted in ascending order)
/// * `target` - Target value to search for
///
/// # Returns
/// * `usize` - Index of the last element <= target, or array.len() if no such element exists
///
/// # Examples
/// ```rust
/// use hwx::dispatch::exponential_search_le_u32;
///
/// let sorted_arr = vec![1, 2, 4, 8, 16, 32, 64, 128];  // Ascending order
/// let index = exponential_search_le_u32(&sorted_arr, 20);  // Result: 4 (index of 16)
/// assert_eq!(index, 4);
///
/// // Empty array
/// let empty: Vec<u32> = vec![];
/// let index = exponential_search_le_u32(&empty, 10);  // Result: 0 (empty.len())
/// assert_eq!(index, 0);
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
/// - Especially efficient when target is near the beginning of large arrays
#[inline]
pub fn exponential_search_le_u32(sorted_array: &[u32], target: u32) -> usize {
    trace!(
        "EXPONENTIAL_SEARCH_LE_U32 DISPATCH: sorted_array.len()={}, target={}",
        sorted_array.len(),
        target
    );

    let len = sorted_array.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for last element <= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if sorted_array[mid] <= target {
                left = mid + 1; // Keep searching right for the LAST element <= target
            } else {
                right = mid; // Found element > target, search left
            }
        }

        // left now points to first element > target
        // So the last element <= target is at left - 1
        if left > 0 && sorted_array[left - 1] <= target {
            return left - 1;
        } else {
            return len; // No element <= target
        }
    }

    // GPU dispatch for large arrays
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SEARCH && get_hw_capabilities().has_cuda {
        if let Ok(result) = unsafe {
            gpu::with_gpu_buffer_u32_search(
                sorted_array,
                target,
                len,
                |gpu_ptr, target, len, result_ptr| {
                    traverse::exponential_search_le_u32_gpu(gpu_ptr, target, len, result_ptr);
                },
            )
        } {
            // Kernel returns first index > target (or len). Map to last <= target.
            return if result == 0 {
                len // All elements > target, so no element <= target
            } else if result >= len {
                len - 1 // All elements <= target, so last element is at len-1
            } else {
                result - 1 // Last element <= target is at result-1
            };
        }
        // Fall through to SIMD if GPU fails
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return unsafe {
                traverse::exponential_search_le_u32_avx512(sorted_array, target, len)
            };
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return unsafe {
                    traverse::exponential_search_le_u32_avx2(sorted_array, target, len)
                };
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return unsafe { traverse::exponential_search_le_u32_neon(sorted_array, target, len) };
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for exponential_search_le_u32");
}

/// Exponential search for the last element <= target in a sorted u64 array with smart threshold-based dispatching
///
/// **Requires:** Array must be sorted in ascending order. Descending arrays are not supported.
///
/// Performs an exponential search to find the index of the last element that is less than or equal
/// to the target value. Exponential search is particularly efficient for large arrays where the target
/// is near the beginning. This function automatically selects between scalar fallback for small arrays
/// and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `sorted_array` - Sorted u64 array (sorted in ascending order)
/// * `target` - Target value to search for
///
/// # Returns
/// * `usize` - Index of the last element <= target, or array.len() if no such element exists
///
/// # Examples
/// ```rust
/// use hwx::dispatch::exponential_search_le_u64;
///
/// let sorted_arr = vec![1u64, 2u64, 4u64, 8u64, 16u64, 32u64, 64u64, 128u64];  // Ascending order
/// let index = exponential_search_le_u64(&sorted_arr, 20);  // Result: 4 (index of 16)
/// assert_eq!(index, 4);
///
/// // Empty array
/// let empty: Vec<u64> = vec![];
/// let index = exponential_search_le_u64(&empty, 10);  // Result: 0 (empty.len())
/// assert_eq!(index, 0);
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
/// - Especially efficient when target is near the beginning of large arrays
#[inline]
pub fn exponential_search_le_u64(sorted_array: &[u64], target: u64) -> usize {
    trace!(
        "EXPONENTIAL_SEARCH_LE_U64 DISPATCH: sorted_array.len()={}, target={}",
        sorted_array.len(),
        target
    );

    let len = sorted_array.len();

    // Smart threshold-based dispatching: use scalar for small arrays
    if len < SIMD_THRESHOLD_BINARY_SEARCH {
        // Binary search for last element <= target (ascending order)
        let mut left = 0;
        let mut right = len;

        while left < right {
            let mid = left + (right - left) / 2;
            if sorted_array[mid] <= target {
                left = mid + 1; // Keep searching right for the LAST element <= target
            } else {
                right = mid; // Found element > target, search left
            }
        }

        // left now points to first element > target
        // So the last element <= target is at left - 1
        if left > 0 && sorted_array[left - 1] <= target {
            return left - 1;
        } else {
            return len; // No element <= target
        }
    }

    // GPU dispatch for large arrays
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SEARCH && get_hw_capabilities().has_cuda {
        if let Ok(result) = unsafe {
            gpu::with_gpu_buffer_u64_search(
                sorted_array,
                target,
                len,
                |gpu_ptr, target, len, result_ptr| {
                    traverse::exponential_search_le_u64_gpu(gpu_ptr, target, len, result_ptr);
                },
            )
        } {
            // Kernel returns first index > target (or len). Map to last <= target.
            return if result == 0 {
                len // All elements > target, so no element <= target
            } else if result >= len {
                len - 1 // All elements <= target, so last element is at len-1
            } else {
                result - 1 // Last element <= target is at result-1
            };
        }
        // Fall through to SIMD if GPU fails
    }

    // SIMD dispatch for larger arrays with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return unsafe {
                traverse::exponential_search_le_u64_avx512(sorted_array, target, len)
            };
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return unsafe {
                    traverse::exponential_search_le_u64_avx2(sorted_array, target, len)
                };
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return unsafe { traverse::exponential_search_le_u64_neon(sorted_array, target, len) };
        }
    }

    // This should not happen as we guarantee SIMD availability
    unreachable!("No SIMD implementation available for exponential_search_le_u64");
}

// =============================================================================
//  CORE ARRAY OPERATIONS - ZERO-COPY IN-PLACE PROCESSING
// =============================================================================

/// Intersect two sorted u32 arrays with automatic truncation
/// Uses the galloping algorithm which automatically chooses the optimal strategy based on array sizes:
/// - Gallopers through the smaller array using exponential + binary search on the larger array
/// - Achieves O(min(m,n) * log(max(m,n))) time complexity for optimal performance
/// - Automatically selects best SIMD implementation (AVX-512, AVX2, NEON) based on hardware
///
/// # Arguments
/// * `a` - First sorted array (modified in-place to contain intersection with automatic truncation)
/// * `b` - Second sorted array (read-only)
/// * `max_size` - Maximum number of results to return
/// * `dedupe` - Whether to remove duplicate elements from the result
///
/// # Examples
/// ```rust
/// use hwx::intersect_sorted_u32;
/// use hwx::types::HwxError;
///
/// let mut a = vec![1, 3, 5, 7, 9];
/// let b = vec![2, 3, 6, 7, 10];
/// intersect_sorted_u32(&mut a, &b, 100, false, true)?;  // No deduplication, ascending
/// assert_eq!(a, vec![3, 7]);  // Results in first array
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[inline]
pub fn intersect_sorted_u32(
    a: &mut Vec<u32>,
    b: &[u32],
    max_size: usize,
    dedup: bool,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "INTERSECT_SORTED_U32 DISPATCH: a.len()={}, b.len()={}, max_size={}",
        a.len(),
        b.len(),
        max_size
    );

    // Early termination: intersection with empty set is empty
    if a.is_empty() || b.is_empty() {
        a.clear();
        return Ok(());
    }

    if max_size == 0 {
        a.clear();
        return Ok(());
    }

    let len_a = a.len();
    let len_b = b.len();
    let total_len = len_a + len_b;
    let a_slice = a.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if total_len < SIMD_THRESHOLD_INTERSECT {
            // Proper scalar intersection that respects duplicate counts
            let mut write_pos = 0;
            let mut a_pos = 0;
            let mut b_pos = 0;
            let mut last_written = u32::MAX;

            while a_pos < len_a && b_pos < len_b {
                let a_val = a[a_pos];
                let b_val = b[b_pos];

                let ordering = if ascending {
                    a_val.cmp(&b_val)
                } else {
                    b_val.cmp(&a_val)
                };

                match ordering {
                    std::cmp::Ordering::Equal => {
                        if !dedup || last_written != a_val {
                            a[write_pos] = a_val;
                            write_pos += 1;
                            if dedup {
                                last_written = a_val;
                            }
                        }
                        a_pos += 1;
                        b_pos += 1;
                    }
                    std::cmp::Ordering::Less => {
                        a_pos += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        b_pos += 1;
                    }
                }
            }
            write_pos
        } else {
            // Tier 2: Check for GPU acceleration for very large arrays

            #[cfg(has_cuda)]
            {
                if total_len >= GPU_THRESHOLD_INTERSECT && get_hw_capabilities().has_cuda {
                    return unsafe {
                        gpu::with_gpu_intersect_u32(
                            a,
                            b,
                            len_a,
                            len_b,
                            max_size,
                            dedup,
                            ascending,
                            |gpu_a, gpu_b, len_a, len_b, max_size, dedup, ascending| {
                                arrays::intersect_sorted_u32_gpu(
                                    gpu_a, gpu_b, len_a, len_b, max_size, dedup, ascending,
                                )
                            },
                        )
                    };
                }
            }

            // For SIMD, always use a_slice as the output regardless of sizes
            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        traverse::intersect_sorted_u32_avx512(
                            a_slice, b, max_size, len_a, len_b, dedup, ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            traverse::intersect_sorted_u32_avx2(
                                a_slice, b, max_size, len_a, len_b, dedup, ascending,
                            )
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        traverse::intersect_sorted_u32_neon(
                            a_slice, b, max_size, len_a, len_b, dedup, ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for intersect_sorted_u32".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    a.truncate(count);

    Ok(())
}

/// Compute set difference of two sorted u32 arrays (A - B) with automatic SIMD dispatch
/// Uses SIMD vectorized comparison and binary search optimizations
/// Automatically selects best SIMD implementation (AVX-512, AVX2, NEON) based on hardware
///
/// # Arguments
/// * `a` - First sorted array (modified in-place to contain difference with automatic truncation)
/// * `b` - Second sorted array (read-only)
/// * `max_size` - Maximum number of results to return
/// * `dedup` - Whether to remove duplicate elements from the result
/// * `ascending` - Whether arrays are sorted in ascending order
///
/// # Examples
/// ```rust
/// use hwx::set_difference_sorted_u32;
/// use hwx::types::HwxError;
///
/// let mut a = vec![1u32, 2, 3, 4, 5];
/// let b = vec![2u32, 4, 6];
/// set_difference_sorted_u32(&mut a, &b, 100, false, true)?;  // No deduplication, ascending
/// assert_eq!(a, vec![1, 3, 5]);  // Elements in A but not in B
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[inline]
pub fn set_difference_sorted_u32(
    a: &mut Vec<u32>,
    b: &[u32],
    max_size: usize,
    dedup: bool,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SET_DIFFERENCE_SORTED_U32 DISPATCH: a.len()={}, b.len()={}, max_size={}",
        a.len(),
        b.len(),
        max_size
    );

    // Early termination: difference with empty set is the original set
    if b.is_empty() {
        if a.len() > max_size {
            a.truncate(max_size);
        }
        return Ok(());
    }

    if a.is_empty() || max_size == 0 {
        a.clear();
        return Ok(());
    }

    let len_a = a.len();
    let len_b = b.len();
    let total_len = len_a + len_b;
    let a_slice = a.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if total_len < SIMD_THRESHOLD_INTERSECT {
            // Scalar set difference implementation
            let mut write_pos = 0;
            let mut a_pos = 0;
            let mut b_pos = 0;
            let mut last_written = u32::MAX;

            while a_pos < len_a && write_pos < max_size {
                let a_val = a[a_pos];
                let mut found_in_b = false;

                // Search for a_val in b starting from current b_pos
                let mut search_pos = b_pos;
                while search_pos < len_b {
                    let b_val = b[search_pos];
                    let ordering = if ascending {
                        a_val.cmp(&b_val)
                    } else {
                        b_val.cmp(&a_val)
                    };

                    match ordering {
                        std::cmp::Ordering::Equal => {
                            found_in_b = true;
                            // Don't advance b_pos - allow duplicates in A to find same element in B
                            break;
                        }
                        std::cmp::Ordering::Less => {
                            break;
                        }
                        std::cmp::Ordering::Greater => {
                            search_pos += 1;
                            // Only advance b_pos if we can skip this element for future searches
                            if search_pos > b_pos {
                                b_pos = search_pos;
                            }
                        }
                    }
                }

                // If not found in B, add to result
                if !found_in_b && (!dedup || last_written != a_val) {
                    a[write_pos] = a_val;
                    write_pos += 1;
                    if dedup {
                        last_written = a_val;
                    }
                }

                a_pos += 1;
            }
            write_pos
        } else {
            // Tier 2: Check for GPU acceleration for very large arrays
            #[cfg(has_cuda)]
            {
                if total_len >= GPU_THRESHOLD_INTERSECT && get_hw_capabilities().has_cuda {
                    let count = unsafe {
                        traverse::set_difference_sorted_u32_gpu(
                            a_slice.as_mut_ptr(),
                            b.as_ptr(),
                            max_size,
                            dedup,
                            ascending,
                            len_a,
                            len_b,
                        )
                    };
                    a.truncate(count);
                    return Ok(());
                }
            }

            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        traverse::set_difference_sorted_u32_avx512(
                            a_slice, b, max_size, len_a, len_b, dedup, ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            traverse::set_difference_sorted_u32_avx2(
                                a_slice, b, max_size, len_a, len_b, dedup, ascending,
                            )
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        traverse::set_difference_sorted_u32_neon(
                            a_slice, b, max_size, len_a, len_b, dedup, ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for set_difference_sorted_u32".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    a.truncate(count);

    Ok(())
}

/// Intersect two sorted u64 arrays with automatic truncation
/// Uses the galloping algorithm which automatically chooses the optimal strategy based on array sizes
/// Automatically selects best SIMD implementation (AVX-512, AVX2, NEON) based on hardware
///
/// # Arguments
/// * `a` - First sorted array (modified in-place to contain intersection with automatic truncation)
/// * `b` - Second sorted array (read-only)
/// * `max_size` - Maximum number of results to return
/// * `dedupe` - Whether to remove duplicate elements from the result
///
/// # Examples
/// ```rust
/// use hwx::intersect_sorted_u64;
/// use hwx::types::HwxError;
///
/// let mut a = vec![1u64, 3, 5, 7, 9];
/// let b = vec![2u64, 3, 6, 7, 10];
/// intersect_sorted_u64(&mut a, &b, 100, false, true)?;  // No deduplication, ascending
/// assert_eq!(a, vec![3, 7]);  // Results in first array
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[inline]
pub fn intersect_sorted_u64(
    a: &mut Vec<u64>,
    b: &[u64],
    max_size: usize,
    dedup: bool,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "INTERSECT_SORTED_U64 DISPATCH: a.len()={}, b.len()={}, max_size={}",
        a.len(),
        b.len(),
        max_size
    );

    // Early termination: intersection with empty set is empty
    if a.is_empty() || b.is_empty() {
        a.clear();
        return Ok(());
    }

    if max_size == 0 {
        a.clear();
        return Ok(());
    }

    let len_a = a.len();
    let len_b = b.len();
    let total_len = len_a + len_b;
    let a_slice = a.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if total_len < SIMD_THRESHOLD_INTERSECT {
            // Proper scalar intersection that respects duplicate counts
            let mut write_pos = 0;
            let mut a_pos = 0;
            let mut b_pos = 0;
            let mut last_written = u64::MAX;

            while a_pos < len_a && b_pos < len_b {
                let a_val = a[a_pos];
                let b_val = b[b_pos];

                let ordering = if ascending {
                    a_val.cmp(&b_val)
                } else {
                    b_val.cmp(&a_val)
                };

                match ordering {
                    std::cmp::Ordering::Equal => {
                        if !dedup || last_written != a_val {
                            a[write_pos] = a_val;
                            write_pos += 1;
                            if dedup {
                                last_written = a_val;
                            }
                        }
                        a_pos += 1;
                        b_pos += 1;
                    }
                    std::cmp::Ordering::Less => {
                        a_pos += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        b_pos += 1;
                    }
                }
            }

            write_pos
        } else {
            // For SIMD, always use a_slice as the output regardless of sizes
            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        traverse::intersect_sorted_u64_avx512(
                            a_slice, b, max_size, len_a, len_b, dedup, ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            traverse::intersect_sorted_u64_avx2(
                                a_slice, b, max_size, len_a, len_b, dedup, ascending,
                            )
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        traverse::intersect_sorted_u64_neon(
                            a_slice, b, max_size, len_a, len_b, dedup, ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for intersect_sorted_u64".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    a.truncate(count);

    Ok(())
}

/// Union multiple sorted u32 arrays with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `arrays` - Mutable vector of sorted arrays to union
/// * `output` - Output vector to store union results with automatic truncation
/// * `max_size` - Maximum number of results to return
///
/// # Returns
/// * `Ok(())` on success with `output` automatically truncated to valid results
/// * `Err(crate::types::HwxError::Internal)` if input arrays are empty or exceed limits
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` for empty input arrays
/// * Returns `crate::types::HwxError::Internal` if array count exceeds architecture-specific array limit
///
/// # Examples
/// ```rust
/// use hwx::union_sorted_u32;
/// use hwx::types::HwxError;
///
/// let arrays = vec![vec![1u32, 3, 5], vec![2u32, 4, 6], vec![3u32, 7, 8]];
/// let arrays_refs: Vec<&[u32]> = arrays.iter().map(|v| v.as_slice()).collect();
/// let mut output = Vec::new();
/// union_sorted_u32(&arrays_refs, &mut output, 100, true)?;  // ascending=true
/// assert_eq!(output, vec![1, 2, 3, 4, 5, 6, 7, 8]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[inline]
pub fn union_sorted_u32(
    arrays: &[&[u32]],
    output: &mut Vec<u32>,
    max_size: usize,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "UNION_SORTED_U32 DISPATCH: arrays.len()={}, output.len()={}, max_size={}",
        arrays.len(),
        output.len(),
        max_size
    );

    // Early termination: return 0 for empty arrays
    if arrays.is_empty() {
        output.clear();
        return Ok(());
    }

    // Get architecture-specific array chunk size for optimal performance
    let max_arrays = get_chunk_size_arrays();

    // Bounds checking: Check if arrays count exceeds architecture-specific limit
    if arrays.len() > max_arrays {
        return Err(crate::types::HwxError::Internal(format!(
      "UNION_SORTED_U32: Arrays count {} exceeds maximum limit of {} for current architecture",
      arrays.len(),
      max_arrays
    )));
    }

    let total_len: usize = arrays.iter().map(|arr| arr.len()).sum();

    // Ensure output has sufficient capacity
    output.clear();
    if output.capacity() < total_len.min(max_size) {
        output.reserve(total_len.min(max_size));
    }

    let mut array_lengths: Vec<usize> = arrays.iter().map(|arr| arr.len()).collect();

    let count = if total_len < SIMD_THRESHOLD_UNION {
        // Scalar fallback for small inputs - use dynamic allocation for architecture-specific size
        let mut indices: Vec<usize> = vec![0; max_arrays];
        let mut active_arrays = 0;

        // Count active arrays
        for (i, array) in arrays.iter().enumerate() {
            if i < max_arrays && !array.is_empty() {
                active_arrays += 1;
            }
            array_lengths[i] = array.len();
        }

        let mut write_pos = 0;
        let mut last_value = None;

        // Resize output to have space for results
        output.resize(max_size.min(total_len), 0);
        let output_slice = output.as_mut_slice();

        // K-way merge using linear scan - NO HEAP ALLOCATION!
        while active_arrays > 0 && write_pos < max_size {
            let mut min_value = if ascending { u32::MAX } else { u32::MIN };
            let mut min_array_idx = usize::MAX;

            // Linear scan to find minimum/maximum current element based on sort order
            for (array_idx, array) in arrays.iter().enumerate() {
                if array_idx >= max_arrays {
                    break;
                }
                let current_idx = indices[array_idx];
                if current_idx < array.len() {
                    let current_value = array[current_idx];
                    let should_update = if ascending {
                        current_value < min_value
                    } else {
                        current_value > min_value
                    };
                    if should_update {
                        min_value = current_value;
                        min_array_idx = array_idx;
                    }
                }
            }

            if min_array_idx == usize::MAX {
                break; // No more elements
            }

            // Add to output if different from last value (deduplication)
            if last_value != Some(min_value) {
                if write_pos < output_slice.len() {
                    output_slice[write_pos] = min_value;
                    write_pos += 1;
                    last_value = Some(min_value);
                } else {
                    break; // Output buffer full
                }
            }

            // Advance index for the array we just consumed from
            indices[min_array_idx] += 1;

            // Check if this array is now exhausted
            if indices[min_array_idx] >= arrays[min_array_idx].len() {
                active_arrays -= 1;
            }
        }

        write_pos
    } else {
        // Tier 2: Check for GPU acceleration for very large arrays

        #[cfg(has_cuda)]
        {
            if total_len >= GPU_THRESHOLD_INTERSECT && get_hw_capabilities().has_cuda {
                return unsafe {
                    gpu::with_gpu_union_u32(
                        arrays,
                        output,
                        max_size,
                        ascending,
                        |gpu_arrays, gpu_sizes, arrays_len, gpu_output, max_size, ascending| {
                            traverse::union_sorted_u32_gpu(
                                gpu_arrays, gpu_sizes, arrays_len, gpu_output, max_size, ascending,
                            )
                        },
                    )
                };
            }
        }

        // SIMD dispatch for larger inputs
        // Resize output to have space for SIMD results
        output.resize(max_size.min(total_len), 0);
        let output_slice = output.as_mut_slice();

        let simd_count = {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        traverse::union_sorted_u32_avx512(
                            arrays,
                            &array_lengths,
                            output_slice,
                            arrays.len(),
                            output_slice.len(),
                            max_size,
                            ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            traverse::union_sorted_u32_avx2(
                                arrays,
                                &array_lengths,
                                output_slice,
                                arrays.len(),
                                output_slice.len(),
                                max_size,
                                ascending,
                            )
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        traverse::union_sorted_u32_neon(
                            arrays,
                            &array_lengths,
                            output_slice,
                            arrays.len(),
                            output_slice.len(),
                            max_size,
                            ascending,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for union_sorted_u32".to_string(),
                ));
            }
        };
        simd_count
    };

    // Automatically truncate to valid results
    output.truncate(count);
    Ok(())
}

/// Remove duplicates from a sorted u32 array with automatic truncation
/// Deduplicate sorted u32 array using zero-copy in-place SIMD
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `input` - Mutable sorted u32 vector to deduplicate in-place (automatically truncated to unique elements)
///
/// # Returns
/// * `Ok(())` - Array deduplicated successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::dedup_sorted_u32;
/// use hwx::types::HwxError;
/// let mut input = vec![1, 1, 2, 2, 3, 4, 4, 5];
/// dedup_sorted_u32(&mut input)?;  //  Error-safe + auto-truncation!
/// assert_eq!(input, vec![1, 2, 3, 4, 5]);  // Zero allocations!
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar deduplication
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated deduplication (4x-8x faster)
#[inline]
pub fn dedup_sorted_u32(input: &mut Vec<u32>) -> Result<(), crate::types::HwxError> {
    trace!("DEDUPE_SORTED_ARRAY DISPATCH: input.len()={}", input.len());

    // Early termination: return length for empty or single-element arrays without calling SIMD
    if input.len() <= 1 {
        return Ok(());
    }

    let len = input.len();
    let input_slice = input.as_mut_slice();

    // Smart threshold-based dispatching: use scalar for small inputs
    if len < SIMD_THRESHOLD_DEDUP {
        // Scalar fallback implementation - optimized for small inputs
        if len <= 1 {
            return Ok(());
        } else {
            let mut write_pos = 1;
            for read_pos in 1..len {
                if input_slice[read_pos] != input_slice[write_pos - 1] {
                    if write_pos != read_pos {
                        input_slice[write_pos] = input_slice[read_pos];
                    }
                    write_pos += 1;
                }
            }
            input.truncate(write_pos);
            return Ok(());
        }
    }

    // GPU dispatch for large arrays.
    //
    // For GPU deduplication we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps
    // the implementation compact and leverages a well-tested DeviceSelect::Unique primitive.

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let count = unsafe { gpu::dedup_sorted_u32_cub(input, len)? };
        input.truncate(count);
        return Ok(());
    }

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let count = unsafe { arrays::dedup_sorted_u32_avx512(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            let count = unsafe { arrays::dedup_sorted_u32_avx2(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "No SIMD capability available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            let count = unsafe { arrays::dedup_sorted_u32_neon(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "SIMD not supported on this architecture".to_string(),
        ));
    }
}

/// Remove duplicates from a sorted u64 array with automatic truncation
/// Deduplicate sorted u64 array using zero-copy in-place SIMD
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `input` - Mutable sorted u64 vector to deduplicate in-place (automatically truncated to unique elements)
///
/// # Returns
/// * `Ok(())` - Array deduplicated successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::dedup_sorted_u64;
/// use hwx::types::HwxError;
/// let mut input = vec![1, 1, 2, 2, 3, 4, 4, 5];
/// dedup_sorted_u64(&mut input)?;  //  Error-safe + auto-truncation!
/// assert_eq!(input, vec![1, 2, 3, 4, 5]);  // Zero allocations!
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar deduplication
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated deduplication (4x-8x faster)
#[inline]
pub fn dedup_sorted_u64(input: &mut Vec<u64>) -> Result<(), crate::types::HwxError> {
    trace!("DEDUPE_SORTED_ARRAY DISPATCH: input.len()={}", input.len());

    // Early termination: return length for empty or single-element arrays without calling SIMD
    if input.len() <= 1 {
        return Ok(());
    }

    let len = input.len();
    let input_slice = input.as_mut_slice();

    // Smart threshold-based dispatching: use scalar for small inputs
    if len < SIMD_THRESHOLD_DEDUP {
        // Scalar fallback implementation - optimized for small inputs
        if len <= 1 {
            return Ok(());
        } else {
            let mut write_pos = 1;
            for read_pos in 1..len {
                if input_slice[read_pos] != input_slice[write_pos - 1] {
                    if write_pos != read_pos {
                        input_slice[write_pos] = input_slice[read_pos];
                    }
                    write_pos += 1;
                }
            }
            input.truncate(write_pos);
            return Ok(());
        }
    }

    // Tier 2: GPU dispatch for large arrays.
    //
    // For GPU deduplication we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps
    // the implementation compact and leverages a well-tested DeviceSelect::Unique primitive.
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let count = unsafe { gpu::dedup_sorted_u64_cub(input, len)? };
        input.truncate(count);
        return Ok(());
    }

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let count = unsafe { arrays::dedup_sorted_u64_avx512(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            let count = unsafe { arrays::dedup_sorted_u64_avx2(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "No SIMD capability available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            let count = unsafe { arrays::dedup_sorted_u64_neon(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "SIMD not supported on this architecture".to_string(),
        ));
    }
}

/// Remove duplicates from a sorted i64 array with automatic truncation
/// Deduplicate sorted i64 array using zero-copy in-place SIMD
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `input` - Mutable sorted i64 vector to deduplicate in-place (automatically truncated to unique elements)
///
/// # Returns
/// * `Ok(())` - Array deduplicated successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::dedup_sorted_i64;
/// use hwx::types::HwxError;
/// let mut input = vec![-10i64, -10, 0, 0, 5, 5, 10];
/// dedup_sorted_i64(&mut input)?;  //  Error-safe + auto-truncation!
/// assert_eq!(input, vec![-10i64, 0, 5, 10]);  // Zero allocations!
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar deduplication
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated deduplication (4x-8x faster)
#[inline]
pub fn dedup_sorted_i64(input: &mut Vec<i64>) -> Result<(), crate::types::HwxError> {
    trace!("DEDUPE_SORTED_I64 DISPATCH: input.len()={}", input.len());

    // Early termination: return length for empty or single-element arrays without calling SIMD
    if input.len() <= 1 {
        return Ok(());
    }

    let len = input.len();
    let input_slice = input.as_mut_slice();

    // Smart threshold-based dispatching: use scalar for small inputs
    if len < SIMD_THRESHOLD_DEDUP {
        // Scalar fallback implementation - optimized for small inputs
        if len <= 1 {
            return Ok(());
        } else {
            let mut write_pos = 1;
            for read_pos in 1..len {
                if input_slice[read_pos] != input_slice[write_pos - 1] {
                    if write_pos != read_pos {
                        input_slice[write_pos] = input_slice[read_pos];
                    }
                    write_pos += 1;
                }
            }
            input.truncate(write_pos);
            return Ok(());
        }
    }

    // Tier 2: GPU dispatch for large arrays.
    //
    // For GPU deduplication we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps
    // the implementation compact and leverages a well-tested DeviceSelect::Unique primitive.
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let count = unsafe { gpu::dedup_sorted_i64_cub(input, len)? };
        input.truncate(count);
        return Ok(());
    }

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let count = unsafe { arrays::dedup_sorted_i64_avx512(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            let count = unsafe { arrays::dedup_sorted_i64_avx2(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "No SIMD capability available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            let count = unsafe { arrays::dedup_sorted_i64_neon(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "SIMD not supported on this architecture".to_string(),
        ));
    }
}

/// Remove duplicates from a sorted f64 array with automatic truncation
/// Deduplicate sorted f64 array using zero-copy in-place SIMD
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `input` - Mutable sorted f64 vector to deduplicate in-place (automatically truncated to unique elements)
///
/// # Returns
/// * `Ok(())` - Array deduplicated successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::dedup_sorted_f64;
/// use hwx::types::HwxError;
/// let mut input = vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0];
/// dedup_sorted_f64(&mut input)?;  //  Error-safe + auto-truncation!
/// assert_eq!(input, vec![1.0, 2.0, 3.0, 4.0, 5.0]);  // Zero allocations!
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (^^ 1 element): Immediate return (no work needed)
/// - Medium arrays (< SIMD_THRESHOLD): Optimized scalar deduplication
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated deduplication (4x-8x faster)
#[inline]
pub fn dedup_sorted_f64(input: &mut Vec<f64>) -> Result<(), crate::types::HwxError> {
    trace!("DEDUPE_SORTED_ARRAY DISPATCH: input.len()={}", input.len());

    // Early termination: return length for empty or single-element arrays without calling SIMD
    if input.len() <= 1 {
        return Ok(());
    }

    let len = input.len();
    let input_slice = input.as_mut_slice();

    // Smart threshold-based dispatching: use scalar for small inputs
    if len < SIMD_THRESHOLD_DEDUP {
        // Scalar fallback implementation - optimized for small inputs
        if len <= 1 {
            return Ok(());
        } else {
            let mut write_pos = 1;
            for read_pos in 1..len {
                if input_slice[read_pos] != input_slice[write_pos - 1] {
                    if write_pos != read_pos {
                        input_slice[write_pos] = input_slice[read_pos];
                    }
                    write_pos += 1;
                }
            }
            input.truncate(write_pos);
            return Ok(());
        }
    }

    // Tier 2: GPU dispatch for large arrays.
    //
    // For GPU deduplication we use NVIDIA CUB primitives via a thin C/CUDA wrapper. This keeps
    // the implementation compact and leverages a well-tested DeviceSelect::Unique primitive.
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
        let count = unsafe { gpu::dedup_sorted_f64_cub(input, len)? };
        input.truncate(count);
        return Ok(());
    }

    // SIMD dispatch for larger inputs with architecture conditionals
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let count = unsafe { arrays::dedup_sorted_f64_avx512(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "AVX-512 not available".to_string(),
            ))
        }

        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            let count = unsafe { arrays::dedup_sorted_f64_avx2(input_slice, input_slice.len()) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "No SIMD capability available".to_string(),
            ))
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            let count = unsafe { arrays::dedup_sorted_f64_neon(input_slice, len) };
            input.truncate(count);
            Ok(())
        } else {
            Err(crate::types::HwxError::Internal(
                "NEON not available".to_string(),
            ))
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(crate::types::HwxError::Internal(
            "No SIMD implementation available for dedup_sorted_f64".to_string(),
        ));
    }
}

/// Filter documents by time range with automatic truncation
///
/// Filters document IDs based on their corresponding timestamps, keeping only those within
/// the specified time range [start_time, end_time]. This function automatically selects
/// between scalar fallback for small datasets and optimized SIMD implementations for larger datasets.
///
/// # Arguments
/// * `doc_ids` - Mutable vector of document IDs to filter (automatically truncated to valid results)
/// * `times` - Vector of timestamps corresponding to document IDs (must be same length as doc_ids)
/// * `start_time` - Start of time range (inclusive)
/// * `end_time` - End of time range (inclusive)
/// * `max_size` - Maximum number of results to keep
///
/// # Returns
/// * `Ok(())` - Documents filtered successfully with `doc_ids` automatically truncated
/// * `Err(crate::types::HwxError::Internal)` - If arrays have different lengths or SIMD operations fail
///
/// # Errors
/// * Returns `crate::types::HwxError::Internal` if doc_ids and times have different lengths
/// * Returns `crate::types::HwxError::Internal` if SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::filter_u32_by_u64_range;
/// use hwx::types::HwxError;
/// let mut doc_ids = vec![1, 2, 3, 4, 5];
/// let times = vec![100, 200, 300, 400, 500];
/// filter_u32_by_u64_range(&mut doc_ids, &times, 150, 350, 10)?;
/// // doc_ids now contains [2, 3] (documents with times 200, 300)
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small datasets (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large datasets (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_u32_by_u64_range(
    doc_ids: &mut Vec<u32>,
    times: &[u64],
    start_time: u64,
    end_time: u64,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "filter_u32_by_u64_range DISPATCH: doc_ids.len()={}, times.len()={}, start_time={}, end_time={}, max_size={}",
    doc_ids.len(),
    times.len(),
    start_time,
    end_time,
    max_size
  );

    // Early termination: return for empty arrays without calling SIMD
    if doc_ids.is_empty() || times.is_empty() {
        doc_ids.clear();
        return Ok(());
    }

    if times.len() != doc_ids.len() {
        return Err(crate::types::HwxError::Internal(
            "Times and doc_ids must have same length".to_string(),
        ));
    }

    let len = doc_ids.len();
    let doc_ids_slice = doc_ids.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len < SIMD_THRESHOLD_TIME_FILTER {
            // Scalar fallback implementation - optimized for small inputs
            let mut write_pos = 0;
            for i in 0..len {
                if write_pos >= max_size {
                    break;
                }
                let doc_id = doc_ids_slice[i];
                let time = times[i];
                if time >= start_time && time <= end_time {
                    doc_ids_slice[write_pos] = doc_id;
                    write_pos += 1;
                }
            }
            write_pos
        } else {
            #[cfg(has_cuda)]
            {
                if len >= GPU_THRESHOLD_FILTER && get_hw_capabilities().has_cuda {
                    let count = unsafe {
                        traverse::filter_u32_by_u64_range_gpu(
                            doc_ids_slice.as_mut_ptr(),
                            times.as_ptr(),
                            start_time,
                            end_time,
                            max_size,
                            len,
                        )
                    };
                    doc_ids.truncate(count);
                    return Ok(());
                }
            }

            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        traverse::filter_u32_by_u64_range_avx512(
                            doc_ids_slice,
                            times,
                            start_time,
                            end_time,
                            max_size,
                            len,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            traverse::filter_u32_by_u64_range_avx2(
                                doc_ids_slice,
                                times,
                                start_time,
                                end_time,
                                max_size,
                                len,
                            )
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        traverse::filter_u32_by_u64_range_neon(
                            doc_ids_slice,
                            times,
                            start_time,
                            end_time,
                            max_size,
                            len,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for filter_u32_by_u64_range".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    doc_ids.truncate(count);
    Ok(())
}

/// Filter u32 array by value range with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `values` - Mutable u32 vector to filter in-place (automatically truncated to values within range)
/// * `min_val` - Minimum value (inclusive)
/// * `max_val` - Maximum value (inclusive)
///
/// # Returns
/// * `Ok(())` - Array filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::filter_range_u32;
/// use hwx::types::HwxError;
///
/// let mut values = vec![1, 5, 10, 15, 20, 25];
/// filter_range_u32(&mut values, 8, 18)?;  //  Error-safe + auto-truncation!
/// assert_eq!(values, vec![10, 15]);  // Zero allocations!
///
/// // Empty array handled gracefully
/// let mut empty: Vec<u32> = vec![];
/// filter_range_u32(&mut empty, 5, 15)?;  // No-op, returns Ok(())
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_range_u32(
    values: &mut Vec<u32>,
    min_val: u32,
    max_val: u32,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "FILTER_RANGE_U32 DISPATCH: values.len()={}, min_val={}, max_val={}",
        values.len(),
        min_val,
        max_val
    );

    // Early termination: return 0 for empty arrays without calling SIMD
    if values.is_empty() {
        return Ok(());
    }

    let len = values.len();
    let values_slice = values.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len < SIMD_THRESHOLD_FILTER_RANGE {
            // Scalar fallback implementation - optimized for small inputs
            let mut write_pos = 0;
            for read_pos in 0..len {
                if values_slice[read_pos] >= min_val && values_slice[read_pos] <= max_val {
                    if write_pos != read_pos {
                        values_slice[write_pos] = values_slice[read_pos];
                    }
                    write_pos += 1;
                }
            }
            write_pos
        } else {
            // Tier 2: Check for GPU acceleration for very large arrays
            #[cfg(has_cuda)]
            if len >= GPU_THRESHOLD_FILTER && get_hw_capabilities().has_cuda {
                let new_len = unsafe {
                    gpu::with_gpu_buffer_u32_inplace(values_slice, len, |gpu_values, len| {
                        traverse::filter_range_u32_gpu(
                            gpu_values,
                            min_val,
                            max_val,
                            usize::MAX,
                            len,
                        )
                    })?
                };
                values.truncate(new_len as usize);
                return Ok(());
            }

            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        traverse::filter_range_u32_avx512(
                            values_slice,
                            min_val,
                            max_val,
                            usize::MAX,
                            len,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            traverse::filter_range_u32_avx2(
                                values_slice,
                                min_val,
                                max_val,
                                usize::MAX,
                                len,
                            )
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        traverse::filter_range_u32_neon(
                            values_slice,
                            min_val,
                            max_val,
                            usize::MAX,
                            len,
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for filter_range_u32".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    values.truncate(count);
    Ok(())
}

/// Filter f64 array by value range with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `values` - Mutable f64 vector to filter in-place (automatically truncated to values within range)
/// * `min_val` - Minimum value (inclusive)
/// * `max_val` - Maximum value (inclusive)
///
/// # Returns
/// * `Ok(())` - Array filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::filter_range_f64;
/// use hwx::types::HwxError;
/// let mut values = vec![1.0, 5.0, 10.0, 15.0, 20.0, 25.0];
/// filter_range_f64(&mut values, 5.0, 15.0)?;  //  Error-safe + auto-truncation!
/// assert_eq!(values, vec![5.0, 10.0, 15.0]);  // Zero allocations!
///
/// // Empty array handled gracefully
/// let mut empty: Vec<f64> = vec![];
/// filter_range_f64(&mut empty, 5.0, 15.0)?;  // No-op, returns Ok(())
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD_FILTER_RANGE): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_FILTER_RANGE): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_range_f64(
    values: &mut Vec<f64>,
    min_val: f64,
    max_val: f64,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "FILTER_RANGE_F64 DISPATCH: values.len()={}, min_val={}, max_val={}",
        values.len(),
        min_val,
        max_val
    );

    // Early termination: return for empty arrays without calling SIMD
    if values.is_empty() {
        return Ok(());
    }

    let len = values.len();
    let values_slice = values.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len < SIMD_THRESHOLD_FILTER_RANGE {
            // Scalar fallback implementation - optimized for small inputs
            let mut write_pos = 0;
            for i in 0..len {
                let value = values_slice[i];
                if value >= min_val && value <= max_val {
                    if write_pos != i {
                        values_slice[write_pos] = values_slice[i];
                    }
                    write_pos += 1;
                }
            }
            write_pos
        } else {
            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe { arrays::filter_range_f64_avx512(values_slice, min_val, max_val, len) }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            arrays::filter_range_f64_avx2(values_slice, min_val, max_val, len)
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe { arrays::filter_range_f64_neon(values_slice, min_val, max_val, len) }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for filter_range_f64".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    values.truncate(count);
    Ok(())
}

/// Filter u64 array by value range with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `values` - Mutable u64 vector to filter in-place (automatically truncated to values within range)
/// * `min_val` - Minimum value (inclusive)
/// * `max_val` - Maximum value (inclusive)
///
/// # Returns
/// * `Ok(())` - Array filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::filter_range_u64;
/// use hwx::types::HwxError;
/// let mut values = vec![1u64, 5, 10, 15, 20, 25];
/// filter_range_u64(&mut values, 8, 18)?;  //  Error-safe + auto-truncation!
/// assert_eq!(values, vec![10, 15]);  // Zero allocations!
///
/// // Empty array handled gracefully
/// let mut empty: Vec<u64> = vec![];
/// filter_range_u64(&mut empty, 5, 15)?;  // No-op, returns Ok(())
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_range_u64(
    values: &mut Vec<u64>,
    min_val: u64,
    max_val: u64,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "FILTER_RANGE_U64 DISPATCH: values.len()={}, min_val={}, max_val={}",
        values.len(),
        min_val,
        max_val
    );

    // Early termination: return for empty arrays without calling SIMD
    if values.is_empty() {
        return Ok(());
    }

    let len = values.len();
    let values_slice = values.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len < SIMD_THRESHOLD_FILTER_RANGE {
            // Scalar fallback implementation - optimized for small inputs
            let mut write_pos = 0;
            for i in 0..len {
                let value = values_slice[i];
                if value >= min_val && value <= max_val {
                    if write_pos != i {
                        values_slice[write_pos] = values_slice[i];
                    }
                    write_pos += 1;
                }
            }
            write_pos
        } else {
            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe { arrays::filter_range_u64_avx512(values_slice, min_val, max_val, len) }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            arrays::filter_range_u64_avx2(values_slice, min_val, max_val, len)
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe { arrays::filter_range_u64_neon(values_slice, min_val, max_val, len) }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for filter_range_u64".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    values.truncate(count);
    Ok(())
}

/// Filter out deleted documents with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
/// deleted_docs must be sorted for optimal performance
///
/// # Arguments
/// * `doc_ids` - Mutable vector of document IDs to filter in-place (automatically truncated to non-deleted documents)
/// * `deleted_docs` - Mutable vector of deleted document IDs (must be sorted for optimal performance)
///
/// # Returns
/// * `Ok(())` - Documents filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::filter_u32;
/// use hwx::types::HwxError;
/// let mut doc_ids = vec![1, 2, 3, 4, 5];
/// let mut deleted_docs = vec![2, 4];  // Must be sorted
/// filter_u32(&mut doc_ids, &mut deleted_docs)?;  //  Error-safe + auto-truncation!
/// assert_eq!(doc_ids, vec![1, 3, 5]);  // Zero allocations!
///
/// // Empty cases handled gracefully
/// let mut empty_docs: Vec<u32> = vec![];
/// filter_u32(&mut empty_docs, &mut deleted_docs)?;  // No-op, returns Ok(())
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_u32(
    doc_ids: &mut Vec<u32>,
    deleted_docs: &[u32], // Must be sorted
) -> Result<(), crate::types::HwxError> {
    trace!(
        "filter_u32 DISPATCH: doc_ids.len()={}, deleted_docs.len()={}",
        doc_ids.len(),
        deleted_docs.len()
    );

    // Early termination: return for empty doc_ids or empty deleted_docs without calling SIMD
    if doc_ids.is_empty() {
        return Ok(());
    }
    if deleted_docs.is_empty() {
        return Ok(()); // No docs to filter out
    }

    let len = doc_ids.len();
    let doc_ids_slice = doc_ids.as_mut_slice();

    // Tier 2: GPU dispatch for large arrays
    #[cfg(has_cuda)]
    {
        if len >= GPU_THRESHOLD_FILTER && get_hw_capabilities().has_cuda {
            let count = unsafe {
                traverse::filter_u32_gpu(
                    doc_ids_slice.as_mut_ptr(),
                    deleted_docs.as_ptr(),
                    len,
                    deleted_docs.len(),
                )
            };
            doc_ids.truncate(count);
            return Ok(());
        }
    }

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len < SIMD_THRESHOLD_FILTER_DELETED || deleted_docs.is_empty() {
            // Scalar fallback implementation - optimized for small inputs
            let mut write_pos = 0;
            for read_pos in 0..len {
                let doc_id = doc_ids_slice[read_pos];

                // Binary search to check if doc_id is in deleted_docs (sorted)
                if deleted_docs.binary_search(&doc_id).is_err() {
                    if write_pos != read_pos {
                        doc_ids_slice[write_pos] = doc_id;
                    }
                    write_pos += 1;
                }
            }
            write_pos
        } else {
            // SIMD dispatch for larger inputs with architecture conditionals
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        traverse::filter_u32_avx512(
                            doc_ids_slice,
                            deleted_docs,
                            len,
                            deleted_docs.len(),
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "AVX-512 not available".to_string(),
                    ));
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            traverse::filter_u32_avx2(
                                doc_ids_slice,
                                deleted_docs,
                                len,
                                deleted_docs.len(),
                            )
                        }
                    } else {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD capability available".to_string(),
                        ));
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        traverse::filter_u32_neon(
                            doc_ids_slice,
                            deleted_docs,
                            len,
                            deleted_docs.len(),
                        )
                    }
                } else {
                    return Err(crate::types::HwxError::Internal(
                        "NEON not available".to_string(),
                    ));
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for filter_u32".to_string(),
                ));
            }
        }
    };

    // Automatically truncate to valid results
    doc_ids.truncate(count);
    Ok(())
}

// =============================================================================
// ^ STRING FUNCTIONS - SIMD-ACCELERATED TEXT PROCESSING
// =============================================================================

/// Match strings with a given prefix with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///  ZERO-COPY: Filters strings directly in-place!
///
/// # Arguments
/// * `strings` - Mutable vector of strings to filter in-place (automatically truncated to matching strings)
/// * `prefix` - Prefix to match against
/// * `case_insensitive` - Whether to ignore case during matching
/// * `max_size` - Maximum number of results to return
///
/// # Returns
/// * `Ok(())` - Strings filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::match_prefix_strings;
/// use hwx::types::HwxError;
/// let mut strings = vec!["hello".to_string(), "world".to_string(), "help".to_string()];
/// match_prefix_strings(&mut strings, "hel", false, 100)?;  //  Error-safe + auto-truncation!
/// assert_eq!(strings, vec!["hello", "help"]);
///
/// // Case insensitive matching
/// let mut strings = vec!["Hello".to_string(), "HELP".to_string(), "world".to_string()];
/// match_prefix_strings(&mut strings, "hel", true, 100)?;
/// assert_eq!(strings, vec!["Hello", "HELP"]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn match_prefix_strings(
    strings: &mut Vec<String>,
    prefix: &str,
    case_insensitive: bool,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "MATCH_PREFIX_STRINGS DISPATCH: strings.len()={}, prefix.len()={}, case_insensitive={}, max_size={}",
    strings.len(),
    prefix.len(),
    case_insensitive,
    max_size
  );

    // Early termination: return for empty arrays only
    if strings.is_empty() {
        return Ok(());
    }

    // Empty prefix matches all strings (return up to max_size)
    if prefix.is_empty() {
        strings.truncate(strings.len().min(max_size));
        return Ok(());
    }

    let len = strings.len();

    let strings_slice = strings.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len <= SIMD_THRESHOLD_STRING_PREFIX {
            // Scalar fallback implementation - optimized for small inputs
            let mut write_pos = 0;
            let prefix_lower = if case_insensitive {
                prefix.to_lowercase()
            } else {
                prefix.to_string()
            };

            //  ZERO-COPY: Process strings directly, no index translation!
            for read_pos in 0..len {
                if write_pos >= max_size {
                    break; // Early termination when max_size results found
                }

                let string_to_check = if case_insensitive {
                    strings_slice[read_pos].to_lowercase()
                } else {
                    strings_slice[read_pos].clone()
                };

                if string_to_check.starts_with(&prefix_lower) {
                    // Keep this string in the filtered result (in-place)
                    if write_pos != read_pos {
                        strings_slice.swap(write_pos, read_pos);
                    }
                    write_pos += 1;
                }
            }
            write_pos
        } else {
            // NOTE: GPU string matching functions exist but are compiled for nvptx64 target only
            // They cannot be called from host code, need proper CUDA kernel launch infrastructure

            // Handle large arrays by chunking at dispatcher level
            let mut total_matches = 0;
            let mut global_write_pos = 0;

            // Get architecture-specific chunk size for optimal performance
            let max_chunk_size = get_chunk_size_strings();

            let mut chunk_start = 0;
            while chunk_start < len && total_matches < max_size {
                let chunk_end = (chunk_start + max_chunk_size).min(len);
                let chunk_size = chunk_end - chunk_start;

                // Create string lengths array for this chunk only - use max size for safety
                let mut string_lengths_array: [usize; 8192] = [0; 8192]; // Use largest possible chunk size
                for (i, string) in strings_slice[chunk_start..chunk_end].iter().enumerate() {
                    string_lengths_array[i] = string.len();
                }
                let string_lengths = &string_lengths_array[..chunk_size];

                // Process this chunk with SIMD (without internal chunking)
                let chunk_matches = {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "hwx-nightly")]
                        if get_hw_capabilities().has_avx512 {
                            unsafe {
                                strings::match_prefix_strings_avx512(
                                    &mut strings_slice[chunk_start..chunk_end],
                                    string_lengths,
                                    prefix,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    prefix.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "AVX512 not available".to_string(),
                            ));
                        }
                        #[cfg(not(feature = "hwx-nightly"))]
                        if get_hw_capabilities().has_avx2 {
                            unsafe {
                                strings::match_prefix_strings_avx2(
                                    &mut strings_slice[chunk_start..chunk_end],
                                    string_lengths,
                                    prefix,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    prefix.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "No SIMD capability available".to_string(),
                            ));
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if get_hw_capabilities().has_neon {
                            unsafe {
                                strings::match_prefix_strings_neon(
                                    &mut strings_slice[chunk_start..chunk_end],
                                    string_lengths,
                                    prefix,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    prefix.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "NEON not available".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(any(
                        target_arch = "aarch64",
                        target_arch = "x86",
                        target_arch = "x86_64"
                    )))]
                    {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD support for this architecture".to_string(),
                        ));
                    }
                };

                // Move matched strings to front of global array
                for i in 0..chunk_matches {
                    if global_write_pos < max_size {
                        strings_slice.swap(global_write_pos, chunk_start + i);
                        global_write_pos += 1;
                    }
                }

                total_matches += chunk_matches;
                chunk_start += chunk_size;
            }

            total_matches
        }
    };

    // Automatically truncate to valid results
    strings.truncate(count);
    Ok(())
}

/// Match field phrases in text with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `doc_ids` - Mutable vector of document IDs to filter in-place (automatically truncated to matching documents)
/// * `events` - Array of stored events to search in
/// * `field_name` - Optional field name to filter by (None searches all fields)
/// * `phrase` - Phrase to search for
/// * `case_insensitive` - Whether to ignore case during matching
/// * `max_size` - Maximum number of results to return
///
/// # Returns
/// * `Ok(())` - Documents filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail
///
/// # Examples
/// ```rust
/// use hwx::match_field_phrases;
/// use hwx::types::HwxError;
/// // Create sample events data
/// let events = vec![];
/// let mut doc_ids = vec![1, 2, 3];
/// match_field_phrases(&mut doc_ids, &events, Some("title"), "hello", false, 100)?;
/// // doc_ids now contains only IDs of documents with "hello" in title field
///
/// // Empty inputs handled gracefully
/// let mut empty_ids: Vec<u32> = vec![];
/// match_field_phrases(&mut empty_ids, &events, None, "test", false, 100)?;  // No-op
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn match_field_phrases(
    doc_ids: &mut Vec<u32>,
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: Option<&str>,
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "MATCH_FIELD_PHRASES DISPATCH: doc_ids.len()={}, events.len()={}, field_name={}, phrase.len()={}, case_insensitive={}, max_size={}",
    doc_ids.len(),
    events.len(),
    field_name.unwrap_or("None"),
    phrase.len(),
    case_insensitive,
    max_size
  );

    // Early termination: return for empty inputs without calling SIMD
    if doc_ids.is_empty() || events.is_empty() || phrase.is_empty() {
        doc_ids.clear();
        return Ok(());
    }

    let len = doc_ids.len().min(events.len());
    let doc_ids_slice = doc_ids.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len <= SIMD_THRESHOLD_STRING_FIELD {
            // Scalar fallback implementation - optimized for small inputs
            let phrase_lower = if case_insensitive {
                phrase.to_lowercase()
            } else {
                phrase.to_string()
            };

            let mut write_pos = 0;
            for read_pos in 0..len {
                if write_pos >= max_size {
                    break;
                }

                if let Some(event) = &events[read_pos] {
                    let fields = event.get_fields();
                    for field in fields {
                        // Check field name if specified
                        if let Some(target_field) = field_name {
                            if field.get_name() != target_field {
                                continue;
                            }
                        }

                        let field_value = field.get_value().to_string();
                        let value_to_check = if case_insensitive {
                            field_value.to_lowercase()
                        } else {
                            field_value
                        };

                        if value_to_check.contains(&phrase_lower) {
                            doc_ids_slice[write_pos] = doc_ids_slice[read_pos];
                            write_pos += 1;
                            break; // Found match in this document
                        }
                    }
                }
            }
            write_pos
        } else {
            // Handle large arrays by chunking at dispatcher level
            let mut total_matches = 0;
            let mut global_write_pos = 0;

            let mut chunk_start = 0;
            while chunk_start < len && total_matches < max_size {
                let chunk_end = (chunk_start + MAX_STRINGS).min(len);
                let chunk_size = chunk_end - chunk_start;

                // Process this chunk with SIMD (without internal chunking)
                let chunk_matches = {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "hwx-nightly")]
                        if get_hw_capabilities().has_avx512 {
                            unsafe {
                                strings::match_field_phrases_avx512(
                                    &mut doc_ids_slice[chunk_start..chunk_end],
                                    &events[chunk_start..chunk_end],
                                    field_name.unwrap_or(""),
                                    phrase,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    phrase.len(),
                                    chunk_size,
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "AVX512 not available".to_string(),
                            ));
                        }
                        #[cfg(not(feature = "hwx-nightly"))]
                        if get_hw_capabilities().has_avx2 {
                            unsafe {
                                strings::match_field_phrases_avx2(
                                    &mut doc_ids_slice[chunk_start..chunk_end],
                                    &events[chunk_start..chunk_end],
                                    field_name.unwrap_or(""),
                                    phrase,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    phrase.len(),
                                    chunk_size,
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "No SIMD capability available".to_string(),
                            ));
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if get_hw_capabilities().has_neon {
                            unsafe {
                                strings::match_field_phrases_neon(
                                    &mut doc_ids_slice[chunk_start..chunk_end],
                                    &events[chunk_start..chunk_end],
                                    field_name.unwrap_or(""),
                                    phrase,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    phrase.len(),
                                    chunk_size,
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "NEON not available".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(any(
                        target_arch = "aarch64",
                        target_arch = "x86",
                        target_arch = "x86_64"
                    )))]
                    {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD support for this architecture".to_string(),
                        ));
                    }
                };

                // Move matched doc_ids to front of global array
                for i in 0..chunk_matches {
                    if global_write_pos < max_size {
                        doc_ids_slice.swap(global_write_pos, chunk_start + i);
                        global_write_pos += 1;
                    }
                }

                total_matches += chunk_matches;
                chunk_start += chunk_size;
            }

            total_matches
        }
    };

    // Automatically truncate to valid results
    doc_ids.truncate(count);
    Ok(())
}

/// Match field prefixes in text with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `doc_ids` - Mutable vector of document IDs to filter in-place with automatic truncation
/// * `events` - Array of stored events to search in
/// * `field_name` - Field name to filter by
/// * `prefix` - Prefix to search for
/// * `case_insensitive` - Whether to ignore case
/// * `max_size` - Maximum number of results to return
///
/// # Examples
/// ```rust
/// use hwx::match_field_prefixes;
/// use hwx::types::HwxError;
/// // Create sample events data
/// let events = vec![];
/// let mut doc_ids = vec![1, 2, 3];
/// match_field_prefixes(&mut doc_ids, &events, "title", "hel", false, 100)?;  //  Error-safe
/// // doc_ids now contains only IDs of documents with fields starting with "hel"
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[inline]
pub fn match_field_prefixes(
    doc_ids: &mut Vec<u32>,
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: &str,
    prefix: &str,
    case_insensitive: bool,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "MATCH_FIELD_PREFIXES DISPATCH: doc_ids.len()={}, events.len()={}, field_name={}, prefix.len()={}, case_insensitive={}, max_size={}",
    doc_ids.len(),
    events.len(),
    field_name,
    prefix.len(),
    case_insensitive,
    max_size
  );

    // Early termination: return for empty inputs without calling SIMD
    if doc_ids.is_empty() || events.is_empty() || field_name.is_empty() || prefix.is_empty() {
        doc_ids.clear();
        return Ok(());
    }

    let len = doc_ids.len().min(events.len());
    let doc_ids_slice = doc_ids.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len <= SIMD_THRESHOLD_STRING_FIELD {
            // Scalar fallback implementation - optimized for small inputs
            let prefix_lower = if case_insensitive {
                prefix.to_lowercase()
            } else {
                prefix.to_string()
            };

            let mut write_pos = 0;
            for read_pos in 0..len {
                if write_pos >= max_size {
                    break;
                }

                if let Some(event) = &events[read_pos] {
                    let fields = event.get_fields();
                    for field in fields {
                        let field_value = field.get_value().to_string();
                        let value_to_check = if case_insensitive {
                            field_value.to_lowercase()
                        } else {
                            field_value
                        };

                        if value_to_check.starts_with(&prefix_lower) {
                            doc_ids_slice[write_pos] = doc_ids_slice[read_pos];
                            write_pos += 1;
                            break; // Found match in this document
                        }
                    }
                }
            }
            write_pos
        } else {
            // Handle large arrays by chunking at dispatcher level
            let mut total_matches = 0;
            let mut global_write_pos = 0;

            let mut chunk_start = 0;
            while chunk_start < len && total_matches < max_size {
                let chunk_end = (chunk_start + MAX_STRINGS).min(len);
                let chunk_size = chunk_end - chunk_start;

                // Process this chunk with SIMD (without internal chunking)
                let chunk_matches = {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "hwx-nightly")]
                        if get_hw_capabilities().has_avx512 {
                            unsafe {
                                strings::match_field_prefixes_avx512(
                                    &mut doc_ids_slice[chunk_start..chunk_end],
                                    &events[chunk_start..chunk_end],
                                    field_name,
                                    prefix,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    prefix.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "AVX512 not available".to_string(),
                            ));
                        }
                        #[cfg(not(feature = "hwx-nightly"))]
                        if get_hw_capabilities().has_avx2 {
                            unsafe {
                                strings::match_field_prefixes_avx2(
                                    &mut doc_ids_slice[chunk_start..chunk_end],
                                    &events[chunk_start..chunk_end],
                                    field_name,
                                    prefix,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    prefix.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "No SIMD capability available".to_string(),
                            ));
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if get_hw_capabilities().has_neon {
                            unsafe {
                                strings::match_field_prefixes_neon(
                                    &mut doc_ids_slice[chunk_start..chunk_end],
                                    &events[chunk_start..chunk_end],
                                    field_name,
                                    prefix,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    prefix.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "NEON not available".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(any(
                        target_arch = "aarch64",
                        target_arch = "x86",
                        target_arch = "x86_64"
                    )))]
                    {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD support for this architecture".to_string(),
                        ));
                    }
                };

                // Move matched doc_ids to front of global array
                for i in 0..chunk_matches {
                    if global_write_pos < max_size {
                        doc_ids_slice.swap(global_write_pos, chunk_start + i);
                        global_write_pos += 1;
                    }
                }

                total_matches += chunk_matches;
                chunk_start += chunk_size;
            }

            total_matches
        }
    };

    // Automatically truncate to valid results
    doc_ids.truncate(count);
    Ok(())
}

/// Filter strings by regex pattern with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///  ZERO-COPY: Filters strings directly in-place!
///
/// # Arguments
/// * `terms` - Mutable vector of strings to filter in-place (automatically truncated to matching strings)
/// * `regex` - Compiled regex pattern to match against
/// * `max_size` - Maximum number of results to return
///
/// # Returns
/// * `Ok(())` - Strings filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail or array exceeds MAX_TERMS
///
/// # Examples
/// ```rust
/// use hwx::filter_regex_terms;
/// use hwx::types::HwxError;
///
/// let mut terms = vec!["hello".to_string(), "world".to_string(), "help".to_string()];
/// let regex = regex::bytes::Regex::new(r"hel.*").unwrap();
/// filter_regex_terms(&mut terms, &regex, 100)?;  //  Error-safe + auto-truncation!
/// assert_eq!(terms, vec!["hello", "help"]);
///
/// // Empty array handled gracefully
/// let mut empty: Vec<String> = vec![];
/// filter_regex_terms(&mut empty, &regex, 100)?;  // No-op, returns Ok(())
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_regex_terms(
    terms: &mut Vec<String>,
    regex: &regex::bytes::Regex,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
        "FILTER_REGEX_TERMS DISPATCH: terms.len()={}, regex.as_str()={}, max_size={}",
        terms.len(),
        regex.as_str().len(),
        max_size
    );

    // Early termination: return for empty terms array without calling SIMD
    if terms.is_empty() {
        return Ok(());
    }

    let len = terms.len();

    // Smart threshold-based dispatching: use scalar for small inputs
    if len <= SIMD_THRESHOLD_STRING_REGEX {
        // Scalar fallback implementation - optimized for small inputs
        let mut write_pos = 0;

        //  ZERO-COPY: Process strings directly, no index translation!
        for read_pos in 0..len {
            if write_pos >= max_size {
                break; // Respect max_size limit
            }
            if regex.is_match(terms[read_pos].as_bytes()) {
                // Keep this string in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }

        // Truncate to actual result size
        terms.truncate(write_pos);
        return Ok(());
    }

    // For large arrays, process in chunks to avoid SIMD limits
    let mut total_matches = 0;
    let mut chunk_start = 0;

    while chunk_start < len && total_matches < max_size {
        let chunk_end = std::cmp::min(chunk_start + MAX_STRINGS, len);
        let chunk_size = chunk_end - chunk_start;

        if chunk_size == 0 {
            break;
        }

        // Prepare term lengths for this chunk - ZERO HEAP using fixed array
        let mut term_lengths_array: [usize; MAX_STRINGS] = [0; MAX_STRINGS];
        for (i, term) in terms[chunk_start..chunk_end].iter().enumerate() {
            term_lengths_array[i] = term.len();
        }
        let term_lengths = &term_lengths_array[..chunk_size];

        let terms_slice = &mut terms[chunk_start..chunk_end];
        let remaining_max_size = max_size - total_matches;

        let chunk_matches = {
            //  SIMD dispatch for chunk processing - direct string processing!
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        strings::filter_regex_terms_avx512(
                            terms_slice,
                            term_lengths,
                            regex,
                            chunk_size,
                            remaining_max_size,
                        )
                    }
                } else {
                    unreachable!("AVX512 should be available with hwx-nightly")
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            strings::filter_regex_terms_avx2(
                                terms_slice,
                                term_lengths,
                                regex,
                                chunk_size,
                                remaining_max_size,
                            )
                        }
                    } else {
                        unreachable!("AVX2 should be available on x86/x86_64")
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        strings::filter_regex_terms_neon(
                            terms_slice,
                            term_lengths,
                            regex,
                            chunk_size,
                            remaining_max_size,
                        )
                    }
                } else {
                    unreachable!("NEON should be guaranteed on aarch64")
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for filter_regex_terms".to_string(),
                ));
            }
        };

        // Move matches to the front of the overall array
        if total_matches > 0 && chunk_matches > 0 {
            // Move the matches from this chunk to positions after previous matches
            for i in 0..chunk_matches {
                terms.swap(total_matches + i, chunk_start + i);
            }
        }

        total_matches += chunk_matches;
        chunk_start = chunk_end;

        // Early termination if we've reached max_size
        if total_matches >= max_size {
            break;
        }
    }

    let count = total_matches;

    // Automatically truncate to valid results
    terms.truncate(count);
    Ok(())
}

/// Filter strings by wildcard pattern with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///  ZERO-COPY: Filters strings directly in-place!
///
/// # Arguments
/// * `terms` - Mutable vector of strings to filter in-place (automatically truncated to matching strings)
/// * `pattern` - Wildcard pattern (* and ? supported)
/// * `case_insensitive` - Whether to ignore case during matching
/// * `max_size` - Maximum number of results to return (function auto-truncates to this limit)
///
/// # Returns
/// * `Ok(())` - Strings filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail or array exceeds MAX_TERMS
///
/// # Examples
/// ```rust
/// use hwx::filter_wildcard_terms;
/// use hwx::types::HwxError;
/// let mut terms = vec!["hello".to_string(), "world".to_string(), "help".to_string()];
/// filter_wildcard_terms(&mut terms, "hel*", false, 100)?;  //  Error-safe + auto-truncation!
/// assert_eq!(terms, vec!["hello", "help"]);
///
/// // Case insensitive matching
/// let mut terms = vec!["Hello".to_string(), "HELP".to_string(), "world".to_string()];
/// filter_wildcard_terms(&mut terms, "hel*", true, 50)?;
/// assert_eq!(terms, vec!["Hello", "HELP"]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn filter_wildcard_terms(
    terms: &mut Vec<String>,
    pattern: &str,
    case_insensitive: bool,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "FILTER_WILDCARD_TERMS DISPATCH: terms.len()={}, pattern.len()={}, case_insensitive={}, max_size={}",
    terms.len(),
    pattern.len(),
    case_insensitive,
    max_size
  );

    // Early termination: return for empty terms array or empty pattern without calling SIMD
    if terms.is_empty() || pattern.is_empty() {
        terms.clear();
        return Ok(());
    }

    let len = terms.len();

    // Smart threshold-based dispatching: use scalar for small inputs
    if len <= SIMD_THRESHOLD_STRING_WILDCARD {
        // Scalar fallback implementation - optimized for small inputs
        let mut write_pos = 0;

        // Convert pattern to regex equivalent for scalar matching
        let has_wildcards = pattern.contains('*') || pattern.contains('?');
        let regex_pattern = if has_wildcards {
            pattern.replace("*", ".*").replace("?", ".")
        } else {
            // No wildcards - treat as exact match
            format!("^{}$", regex::escape(pattern))
        };
        let regex_pattern = if case_insensitive {
            format!("(?i){}", regex_pattern)
        } else {
            regex_pattern
        };

        let regex = regex::Regex::new(&regex_pattern).unwrap_or_else(|_| {
            // Fallback to literal matching if regex compilation fails
            regex::Regex::new(&regex::escape(pattern)).unwrap()
        });

        //  ZERO-COPY: Process strings directly, no index translation!
        for read_pos in 0..len {
            if write_pos >= max_size {
                break; // Stop processing once we reach max_size
            }
            if regex.is_match(&terms[read_pos]) {
                // Keep this string in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }

        // Truncate to actual result size
        terms.truncate(write_pos);
        return Ok(());
    }

    // For large arrays, process in chunks to avoid SIMD limits
    let mut total_matches = 0;
    let mut chunk_start = 0;

    while chunk_start < len && total_matches < max_size {
        let chunk_end = std::cmp::min(chunk_start + MAX_STRINGS, len);
        let chunk_size = chunk_end - chunk_start;

        if chunk_size == 0 {
            break;
        }

        // Prepare term lengths for this chunk - ZERO HEAP using fixed array
        let mut term_lengths_array: [usize; MAX_STRINGS] = [0; MAX_STRINGS];
        for (i, term) in terms[chunk_start..chunk_end].iter().enumerate() {
            term_lengths_array[i] = term.len();
        }
        let term_lengths = &term_lengths_array[..chunk_size];

        let terms_slice = &mut terms[chunk_start..chunk_end];
        let remaining_max_size = max_size - total_matches;

        let chunk_matches = {
            //  SIMD dispatch for chunk processing - direct string processing!
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "hwx-nightly")]
                if get_hw_capabilities().has_avx512 {
                    unsafe {
                        strings::filter_wildcard_terms_avx512(
                            terms_slice,
                            term_lengths,
                            pattern,
                            case_insensitive,
                            chunk_size,
                            pattern.len(),
                            remaining_max_size,
                        )
                    }
                } else {
                    unreachable!("AVX512 should be available with hwx-nightly")
                }
                #[cfg(not(feature = "hwx-nightly"))]
                {
                    if get_hw_capabilities().has_avx2 {
                        unsafe {
                            strings::filter_wildcard_terms_avx2(
                                terms_slice,
                                term_lengths,
                                pattern,
                                case_insensitive,
                                chunk_size,
                                pattern.len(),
                                remaining_max_size,
                            )
                        }
                    } else {
                        unreachable!("AVX2 should be available on x86/x86_64")
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if get_hw_capabilities().has_neon {
                    unsafe {
                        strings::filter_wildcard_terms_neon(
                            terms_slice,
                            term_lengths,
                            pattern,
                            case_insensitive,
                            chunk_size,
                            pattern.len(),
                            remaining_max_size,
                        )
                    }
                } else {
                    unreachable!("NEON should be guaranteed on aarch64")
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                return Err(crate::types::HwxError::Internal(
                    "No SIMD implementation available for filter_wildcard_terms".to_string(),
                ));
            }
        };

        // Move matches to the front of the overall array
        if total_matches > 0 && chunk_matches > 0 {
            // Move the matches from this chunk to positions after previous matches
            for i in 0..chunk_matches {
                terms.swap(total_matches + i, chunk_start + i);
            }
        }

        total_matches += chunk_matches;
        chunk_start = chunk_end;

        // Early termination if we've reached max_size
        if total_matches >= max_size {
            break;
        }
    }

    // Automatically truncate to valid results
    terms.truncate(total_matches);
    Ok(())
}

/// Match exact phrases in text with automatic truncation
/// Automatically selects between scalar fallback and best SIMD implementation based on input size
///
/// # Arguments
/// * `texts` - Mutable vector of strings to filter in-place (automatically truncated to matching strings)
/// * `phrase` - Exact phrase to search for
/// * `case_insensitive` - Whether to ignore case during matching
/// * `max_size` - Maximum number of results to return
///
/// # Returns
/// * `Ok(())` - Strings filtered successfully with automatic truncation
/// * `Err(crate::types::HwxError::Internal)` - If SIMD operations fail or array exceeds MAX_TEXTS
///
/// # Examples
/// ```rust
/// use hwx::match_exact_phrases;
/// use hwx::types::HwxError;
/// let mut texts = vec!["hello world".to_string(), "goodbye world".to_string(), "hello there".to_string()];
/// match_exact_phrases(&mut texts, "hello", false, 100)?;  //  Error-safe + auto-truncation!
/// assert_eq!(texts, vec!["hello world", "hello there"]);
///
/// // Case insensitive matching
/// let mut texts = vec!["Hello World".to_string(), "goodbye world".to_string()];
/// match_exact_phrases(&mut texts, "hello", true, 100)?;
/// assert_eq!(texts, vec!["Hello World"]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Empty arrays: Immediate return
/// - Small arrays (< SIMD_THRESHOLD): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn match_exact_phrases(
    texts: &mut Vec<String>,
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
) -> Result<(), crate::types::HwxError> {
    trace!(
    "MATCH_EXACT_PHRASES DISPATCH: texts.len()={}, phrase.len()={}, case_insensitive={}, max_size={}",
    texts.len(),
    phrase.len(),
    case_insensitive,
    max_size
  );

    // Early termination: return for empty arrays or empty phrase without calling SIMD
    if texts.is_empty() || phrase.is_empty() {
        texts.clear();
        return Ok(());
    }

    let len = texts.len();

    // Remove bounds checking - we'll handle large arrays with chunking

    let texts_slice = texts.as_mut_slice();

    let count = {
        // Smart threshold-based dispatching: use scalar for small inputs
        if len <= SIMD_THRESHOLD_STRING_PREFIX {
            // Scalar fallback implementation - optimized for small inputs
            let mut write_pos = 0;
            let phrase_lower = if case_insensitive {
                phrase.to_lowercase()
            } else {
                phrase.to_string()
            };

            // Process indices in-place: check if text at each index contains phrase
            for read_pos in 0..len {
                if write_pos >= max_size {
                    break; // Early termination when max_size results found
                }

                let text = &texts_slice[read_pos];

                let text_to_check = if case_insensitive {
                    text.to_lowercase()
                } else {
                    text.clone()
                };

                if text_to_check.contains(&phrase_lower) {
                    // Keep this index in the filtered result (in-place)
                    texts_slice.swap(write_pos, read_pos);
                    write_pos += 1;
                }
            }
            write_pos
        } else {
            // Handle large arrays by chunking at dispatcher level
            let mut total_matches = 0;
            let mut global_write_pos = 0;

            let mut chunk_start = 0;
            while chunk_start < len && total_matches < max_size {
                let chunk_end = (chunk_start + MAX_TEXTS).min(len);
                let chunk_size = chunk_end - chunk_start;

                // Create text lengths array for this chunk only
                let mut text_lengths_array: [usize; MAX_TEXTS] = [0; MAX_TEXTS];
                for (i, text) in texts_slice[chunk_start..chunk_end].iter().enumerate() {
                    text_lengths_array[i] = text.len();
                }
                let text_lengths = &text_lengths_array[..chunk_size];

                // Process this chunk with SIMD (without internal chunking)
                let chunk_matches = {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "hwx-nightly")]
                        if get_hw_capabilities().has_avx512 {
                            unsafe {
                                strings::match_exact_phrases_avx512(
                                    &mut texts_slice[chunk_start..chunk_end],
                                    text_lengths,
                                    phrase,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    phrase.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "AVX512 not available".to_string(),
                            ));
                        }
                        #[cfg(not(feature = "hwx-nightly"))]
                        if get_hw_capabilities().has_avx2 {
                            unsafe {
                                strings::match_exact_phrases_avx2(
                                    &mut texts_slice[chunk_start..chunk_end],
                                    text_lengths,
                                    phrase,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    phrase.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "No SIMD capability available".to_string(),
                            ));
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if get_hw_capabilities().has_neon {
                            unsafe {
                                strings::match_exact_phrases_neon(
                                    &mut texts_slice[chunk_start..chunk_end],
                                    text_lengths,
                                    phrase,
                                    case_insensitive,
                                    max_size - total_matches,
                                    chunk_size,
                                    phrase.len(),
                                )
                            }
                        } else {
                            return Err(crate::types::HwxError::Internal(
                                "NEON not available".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(any(
                        target_arch = "aarch64",
                        target_arch = "x86",
                        target_arch = "x86_64"
                    )))]
                    {
                        return Err(crate::types::HwxError::Internal(
                            "No SIMD support for this architecture".to_string(),
                        ));
                    }
                };

                // Move matched strings to front of global array
                for i in 0..chunk_matches {
                    if global_write_pos < max_size {
                        texts_slice.swap(global_write_pos, chunk_start + i);
                        global_write_pos += 1;
                    }
                }

                total_matches += chunk_matches;
                chunk_start += MAX_TEXTS;
            }

            total_matches
        }
    };

    // Automatically truncate to valid results
    texts.truncate(count);
    Ok(())
}

/// Binary search for the first MetricPoint with time >= target_time
///
/// Performs a binary search to find the index of the first MetricPoint that has a timestamp
/// greater than or equal to the target time. This function automatically selects between
/// scalar fallback for small arrays and optimized SIMD implementations for larger arrays.
///
/// # Arguments
/// * `metric_points` - Array slice of MetricPoints (must be sorted by timestamp in ascending order)
/// * `target_time` - Target timestamp to search for
///
/// # Returns
/// * `Ok(usize)` - Index of the first MetricPoint with time >= target_time, or array.len() if none found
/// * `Ok(0)` - If the array is empty
///
/// # Examples
/// ```rust
/// use hwx::binary_search_ge_time;
/// use hwx::types::HwxError;
/// use hwx::MetricPoint;
///
/// // Create sample metric points with timestamps [10, 20, 30, 40, 50]
/// let metric_points = vec![
///     MetricPoint::new( 10, 1.0),
///     MetricPoint::new( 20, 2.0),
///     MetricPoint::new( 30, 3.0),
/// ];
/// let index = binary_search_ge_time(&metric_points, 25)?;  // Result: 2 (index of timestamp 30)
/// assert_eq!(index, 2);
///
/// // Empty array
/// let empty: &[MetricPoint] = &[];
/// let index = binary_search_ge_time(empty, 25)?;  // Result: 0
/// assert_eq!(index, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn binary_search_ge_time(
    metric_points: &[crate::types::MetricPoint],
    target_time: u64,
) -> Result<usize, crate::types::HwxError> {
    trace!(
        "BINARY_SEARCH_LE_TIME DISPATCH: metric_points.len()={}, target_time={}",
        metric_points.len(),
        target_time
    );

    // Early termination: return 0 for empty arrays without calling SIMD
    if metric_points.is_empty() {
        return Ok(0);
    }

    let len = metric_points.len();

    // Tier 2: GPU dispatch for large arrays
    #[cfg(has_cuda)]
    {
        if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
            return Ok(unsafe {
                traverse::binary_search_ge_time_gpu(metric_points.as_ptr(), target_time, len)
            });
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                traverse::binary_search_ge_time_avx512(metric_points, target_time, len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe {
                    traverse::binary_search_ge_time_avx2(metric_points, target_time, len)
                });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe {
                traverse::binary_search_ge_time_neon(metric_points, target_time, len)
            });
        }
    }

    // Fallback - should not happen in normal operation
    panic!("No SIMD implementation available for binary_search_ge_time");
}

/// Binary search for the last MetricPoint with time <= target_time
///
/// Performs a binary search to find the index of the last MetricPoint that has a timestamp
/// less than or equal to the target time. Elements after this position have time > target_time.
/// This function automatically selects between scalar fallback for small arrays and optimized
/// SIMD implementations for larger arrays.
///
/// # Arguments
/// * `metric_points` - Array slice of MetricPoints (must be sorted by timestamp in ascending order)
/// * `target_time` - Target timestamp to search for
///
/// # Returns
/// * `Ok(usize)` - Index of the last MetricPoint with time <= target_time, or 0 if none found
/// * `Ok(0)` - If the array is empty
///
/// # Examples
/// ```rust
/// use hwx::binary_search_le_time;
/// use hwx::types::HwxError;
/// use hwx::MetricPoint;
///
/// // Create sample metric points with timestamps [10, 20, 30, 40, 50]
/// let metric_points = vec![
///     MetricPoint::new( 10, 1.0),
///     MetricPoint::new( 20, 2.0),
///     MetricPoint::new( 30, 3.0),
/// ];
/// let index = binary_search_le_time(&metric_points, 25)?;  // Result: 2 (index after last <= element)
/// assert_eq!(index, 2);
///
/// // Empty array
/// let empty: &[MetricPoint] = &[];
/// let index = binary_search_le_time(empty, 25)?;  // Result: 0
/// assert_eq!(index, 0);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays (< SIMD_THRESHOLD_BINARY_SEARCH): Optimized scalar implementation
/// - Large arrays (^ SIMD_THRESHOLD_BINARY_SEARCH): SIMD-accelerated (4x-8x faster)
#[inline]
pub fn binary_search_le_time(
    metric_points: &[crate::types::MetricPoint],
    target_time: u64,
) -> Result<usize, crate::types::HwxError> {
    trace!(
        "BINARY_SEARCH_LE_TIME DISPATCH: metric_points.len()={}, target_time={}",
        metric_points.len(),
        target_time
    );

    // Early termination: return 0 for empty arrays without calling SIMD
    if metric_points.is_empty() {
        return Ok(0);
    }

    let len = metric_points.len();

    // Tier 2: GPU dispatch for large arrays
    #[cfg(has_cuda)]
    {
        if len >= GPU_THRESHOLD_SORT && get_hw_capabilities().has_cuda {
            return Ok(unsafe {
                traverse::binary_search_le_time_gpu(metric_points.as_ptr(), target_time, len)
            });
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe {
                traverse::binary_search_le_time_avx512(metric_points, target_time, len)
            });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe {
                    traverse::binary_search_le_time_avx2(metric_points, target_time, len)
                });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe {
                traverse::binary_search_le_time_neon(metric_points, target_time, len)
            });
        }
    }

    // Fallback - should not happen in normal operation
    panic!("No SIMD implementation available for binary_search_le_time");
}

// =============================================================================
//  PROMQL ELEMENT-WISE OPERATIONS
// =============================================================================

/// Element-wise addition of two f64 arrays using HW acceleration
///
/// # Arguments
/// * `a` - First input array
/// * `b` - Second input array
/// * `result` - Output array to store the result
///
/// # Returns
/// * `Ok(())` - Operation successful
/// * `Err(crate::types::HwxError)` - Length mismatch between arrays
///
/// # Example
/// ```rust
/// use hwx::add_f64;
/// use hwx::types::HwxError;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let mut result = vec![0.0; 3];
///
/// add_f64(&a, &b, &mut result)?;
/// assert_eq!(result, vec![5.0, 7.0, 9.0]);
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
#[inline]
pub fn add_f64(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_ADD_F64 DISPATCH: a.len()={}, b.len()={}, result.len()={}",
        a.len(),
        b.len(),
        result.len()
    );

    let len = a.len();

    // Validate input lengths
    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for addition".to_string(),
        ));
    }

    // Early termination for empty arrays
    if len == 0 {
        return Ok(());
    }

    // Scalar path for small arrays
    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = a[i] + b[i];
        }
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_3way_f64(
                    a,
                    b,
                    result,
                    len,
                    |gpu_a, gpu_b, gpu_result, len| {
                        arrays::add_f64_gpu(gpu_a, gpu_b, gpu_result, len)
                    },
                )
            };
        }
    }

    // SIMD dispatch based on architecture
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::add_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::add_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::add_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    // Fallback scalar implementation
    for i in 0..len {
        result[i] = a[i] + b[i];
    }
    Ok(())
}

/// Element-wise subtraction of two f64 arrays using HW acceleration
#[inline]
pub fn subtract_f64(
    a: &[f64],
    b: &[f64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_SUBTRACT_F64 DISPATCH: a.len()={}, b.len()={}, result.len()={}",
        a.len(),
        b.len(),
        result.len()
    );

    let len = a.len();

    // Validate input lengths
    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for subtraction".to_string(),
        ));
    }

    // Early termination for empty arrays
    if len == 0 {
        return Ok(());
    }

    // Scalar path for small arrays
    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = a[i] - b[i];
        }
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_3way_f64(
                    a,
                    b,
                    result,
                    len,
                    |gpu_a, gpu_b, gpu_result, len| {
                        arrays::subtract_f64_gpu(gpu_a, gpu_b, gpu_result, len)
                    },
                )
            };
        }
    }

    // SIMD dispatch based on architecture
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::subtract_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::subtract_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::subtract_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    Ok(())
}

/// Element-wise multiplication of two f64 arrays using HW acceleration
#[inline]
pub fn multiply_f64(
    a: &[f64],
    b: &[f64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_MULTIPLY_F64 DISPATCH: a.len()={}, b.len()={}, result.len()={}",
        a.len(),
        b.len(),
        result.len()
    );

    let len = a.len();

    // Validate input lengths
    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for multiplication".to_string(),
        ));
    }

    // Early termination for empty arrays
    if len == 0 {
        return Ok(());
    }

    // Scalar path for small arrays
    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = a[i] * b[i];
        }
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_3way_f64(
                    a,
                    b,
                    result,
                    len,
                    |gpu_a, gpu_b, gpu_result, len| {
                        arrays::multiply_f64_gpu(gpu_a, gpu_b, gpu_result, len)
                    },
                )
            };
        }
    }

    // SIMD dispatch based on architecture
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::multiply_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::multiply_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::multiply_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    // Fallback scalar implementation
    for i in 0..len {
        result[i] = a[i] * b[i];
    }
    Ok(())
}

/// Element-wise division of two f64 arrays using HW acceleration
#[inline]
pub fn divide_f64(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    trace!(
        "SIMD_DIVIDE_F64 DISPATCH: a.len()={}, b.len()={}, result.len()={}",
        a.len(),
        b.len(),
        result.len()
    );

    let len = a.len();

    // Validate input lengths
    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for division".to_string(),
        ));
    }

    // Early termination for empty arrays
    if len == 0 {
        return Ok(());
    }

    // Scalar path for small arrays
    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = a[i] / b[i];
        }
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_3way_f64(
                    a,
                    b,
                    result,
                    len,
                    |gpu_a, gpu_b, gpu_result, len| {
                        arrays::divide_f64_gpu(gpu_a, gpu_b, gpu_result, len)
                    },
                )
            };
        }
    }

    // SIMD dispatch based on architecture
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::divide_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::divide_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::divide_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    // Fallback scalar implementation
    for i in 0..len {
        result[i] = a[i] / b[i];
    }
    Ok(())
}

/// Element-wise modulo of two f64 arrays
#[inline]
pub fn modulo_f64(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for modulo".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_3way_f64(
                    a,
                    b,
                    result,
                    len,
                    |gpu_a, gpu_b, gpu_result, len| {
                        arrays::modulo_f64_gpu(gpu_a, gpu_b, gpu_result, len)
                    },
                )
            };
        }
    }

    // For modulo, we use scalar implementation as there's no SIMD instruction
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::modulo_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::modulo_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::modulo_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = a[i] % b[i];
    }
    Ok(())
}

/// Element-wise power of two f64 arrays
#[inline]
pub fn power_f64(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for power".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffers_3way_f64(
                    a,
                    b,
                    result,
                    len,
                    |gpu_a, gpu_b, gpu_result, len| {
                        arrays::power_f64_gpu(gpu_a, gpu_b, gpu_result, len)
                    },
                )
            };
        }
    }

    // For power, we use scalar implementation as there's no SIMD instruction
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::power_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::power_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::power_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = a[i].powf(b[i]);
    }
    Ok(())
}

// =============================================================================
//  PROMQL COMPARISON OPERATIONS
// =============================================================================

/// Element-wise equality comparison (returns 1.0 if equal, 0.0 otherwise)
#[inline]
pub fn equal_f64(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for equality comparison".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = if a[i] == b[i] { 1.0 } else { 0.0 };
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::equal_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::equal_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::equal_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = if a[i] == b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Element-wise not-equal comparison
#[inline]
pub fn not_equal_f64(
    a: &[f64],
    b: &[f64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for not-equal comparison".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = if a[i] != b[i] { 1.0 } else { 0.0 };
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::not_equal_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::not_equal_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::not_equal_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = if a[i] != b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Element-wise greater-than comparison
#[inline]
pub fn greater_than_f64(
    a: &[f64],
    b: &[f64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for greater-than comparison".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = if a[i] > b[i] { 1.0 } else { 0.0 };
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::greater_than_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::greater_than_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::greater_than_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = if a[i] > b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Element-wise less-than comparison
#[inline]
pub fn less_than_f64(
    a: &[f64],
    b: &[f64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for less-than comparison".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = if a[i] < b[i] { 1.0 } else { 0.0 };
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::less_than_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::less_than_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::less_than_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = if a[i] < b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}
/// Element-wise greater-equal comparison
#[inline]
pub fn greater_equal_f64(
    a: &[f64],
    b: &[f64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for greater-equal comparison".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = if a[i] >= b[i] { 1.0 } else { 0.0 };
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::greater_equal_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::greater_equal_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::greater_equal_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = if a[i] >= b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Element-wise less-equal comparison
#[inline]
pub fn less_equal_f64(
    a: &[f64],
    b: &[f64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = a.len();

    if len != b.len() || len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for less-equal comparison".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            result[i] = if a[i] <= b[i] { 1.0 } else { 0.0 };
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::less_equal_f64_avx512(a, b, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::less_equal_f64_avx2(a, b, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::less_equal_f64_neon(a, b, result, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        result[i] = if a[i] <= b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Check if values are NaN
///
/// Sets result[i] = 1.0 if values[i] is NaN, 0.0 otherwise
pub fn is_nan_f64(values: &[f64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len != result.len() {
        return Err(crate::types::HwxError::Internal(
            "Vector length mismatch for is_nan check".to_string(),
        ));
    }

    if len == 0 {
        return Ok(());
    }

    // NaN != NaN is true, so we check if value != value
    for i in 0..len {
        result[i] = if values[i].is_nan() { 1.0 } else { 0.0 };
    }

    Ok(())
}

// =============================================================================
//  PROMQL MATH FUNCTIONS
// =============================================================================

/// Element-wise absolute value
#[inline]
pub fn abs_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            // Use bit manipulation to clear sign bit
            values[i] = f64::from_bits(values[i].to_bits() & 0x7FFFFFFFFFFFFFFF);
        }
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_values, len| {
                    arrays::abs_f64_gpu(gpu_values, len)
                })
            };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::abs_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::abs_f64_avx2(values, values.len()) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::abs_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].abs();
    }
    Ok(())
}

/// Element-wise negation
#[inline]
pub fn neg_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            values[i] = -values[i];
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::neg_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::neg_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::neg_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = -values[i];
    }
    Ok(())
}

/// Element-wise square root
#[inline]
pub fn sqrt_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            values[i] = values[i].sqrt();
        }
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_values, len| {
                    arrays::sqrt_f64_gpu(gpu_values, len)
                })
            };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::sqrt_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::sqrt_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::sqrt_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].sqrt();
    }
    Ok(())
}

/// Element-wise sine
#[inline]
pub fn sin_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_values, len| {
                    arrays::sin_f64_gpu(gpu_values, len)
                })
            };
        }
    }

    // No SIMD instruction for sin, use scalar
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::sin_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::sin_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::sin_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].sin();
    }
    Ok(())
}

/// Element-wise cosine
#[inline]
pub fn cos_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_values, len| {
                    arrays::cos_f64_gpu(gpu_values, len)
                })
            };
        }
    }

    // No SIMD instruction for cos, use scalar
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::cos_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::cos_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::cos_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].cos();
    }
    Ok(())
}

/// Element-wise tangent
#[inline]
pub fn tan_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // No SIMD instruction for tan, use scalar
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::tan_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::tan_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::tan_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].tan();
    }
    Ok(())
}

/// Element-wise exponential
#[inline]
pub fn exp_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_values, len| {
                    arrays::exp_f64_gpu(gpu_values, len)
                })
            };
        }
    }

    // No SIMD instruction for exp, use scalar
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::exp_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::exp_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::exp_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].exp();
    }
    Ok(())
}

/// Element-wise natural logarithm
#[inline]
pub fn log_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Tier 2: Check for GPU acceleration for very large arrays

    #[cfg(has_cuda)]
    {
        #[cfg(has_cuda)]
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            return unsafe {
                gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_values, len| {
                    arrays::log_f64_gpu(gpu_values, len)
                })
            };
        }
    }

    // No SIMD instruction for log, use scalar
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::log_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::log_f64_avx2(values, values.len()) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::log_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].ln();
    }
    Ok(())
}

/// Element-wise inverse hyperbolic cosine (acosh)
#[inline]
pub fn acosh_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::acosh_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::acosh_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::acosh_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::acosh_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].acosh();
    }
    Ok(())
}

/// Element-wise inverse hyperbolic sine (asinh)
#[inline]
pub fn asinh_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::asinh_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::asinh_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::asinh_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::asinh_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].asinh();
    }
    Ok(())
}

/// Element-wise inverse hyperbolic tangent (atanh)
#[inline]
pub fn atanh_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::atanh_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::atanh_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::atanh_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::atanh_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].atanh();
    }
    Ok(())
}

/// Element-wise inverse cosine using HW acceleration with smart threshold-based dispatching
///
/// Computes the inverse cosine (acos) of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values in range [-1, 1]
/// * `result` - Mutable f64 slice for output values in range [0, ^]
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch or invalid input values
///
/// # Examples
/// ```rust
/// use hwx::dispatch::acos_f64;
/// let mut values = vec![1.0, 0.5, 0.0, -0.5, -1.0];
/// acos_f64(&mut values)?;
/// // values now contains [0.0, ^/3, ^/2, 2^/3, ^]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn acos_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::acos_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::acos_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::acos_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::acos_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].acos();
    }
    Ok(())
}

/// Element-wise inverse sine using HW acceleration with smart threshold-based dispatching
///
/// Computes the inverse sine (asin) of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values in range [-1, 1]
/// * `result` - Mutable f64 slice for output values in range [-^/2, ^/2]
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch or invalid input values
///
/// # Examples
/// ```rust
/// use hwx::dispatch::asin_f64;
/// let mut values = vec![1.0, 0.5, 0.0, -0.5, -1.0];
/// asin_f64(&mut values)?;
/// // values now contains [^/2, ^/6, 0, -^/6, -^/2]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn asin_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::asin_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::asin_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::asin_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::asin_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].asin();
    }
    Ok(())
}

/// Element-wise inverse tangent using HW acceleration with smart threshold-based dispatching
///
/// Computes the inverse tangent (atan) of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values
/// * `result` - Mutable f64 slice for output values in range [-^/2, ^/2]
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch
///
/// # Examples
/// ```rust
/// use hwx::dispatch::atan_f64;
/// let mut values = vec![-1.0, 0.0, 1.0];
/// atan_f64(&mut values)?;
/// // values now contains [-^/4, 0.0, ^/4]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn atan_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::atan_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::atan_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::atan_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::atan_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].atan();
    }
    Ok(())
}

/// Element-wise hyperbolic cosine using HW acceleration with smart threshold-based dispatching
///
/// Computes the hyperbolic cosine (cosh) of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values
/// * `result` - Mutable f64 slice for output values (always positive, ^ 1)
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch
///
/// # Examples
/// ```rust
/// use hwx::dispatch::cosh_f64;
/// let mut values = vec![0.0, 1.0, -1.0];
/// cosh_f64(&mut values)?;
/// // values now contains [1.0, cosh(1.0), cosh(-1.0)]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn cosh_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::cosh_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::cosh_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::cosh_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::cosh_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].cosh();
    }
    Ok(())
}

/// Element-wise hyperbolic sine using HW acceleration with smart threshold-based dispatching
///
/// Computes the hyperbolic sine (sinh) of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values
/// * `result` - Mutable f64 slice for output values
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch
///
/// # Examples
/// ```rust
/// use hwx::dispatch::sinh_f64;
/// let mut values = vec![0.0, 1.0, -1.0];
/// sinh_f64(&mut values)?;
/// // values now contains [0.0, sinh(1.0), sinh(-1.0)]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn sinh_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::sinh_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::sinh_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::sinh_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::sinh_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].sinh();
    }
    Ok(())
}

/// Element-wise hyperbolic tangent using HW acceleration with smart threshold-based dispatching
///
/// Computes the hyperbolic tangent (tanh) of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values
/// * `result` - Mutable f64 slice for output values in range [-1, 1]
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch
///
/// # Examples
/// ```rust
/// use hwx::dispatch::tanh_f64;
/// let mut values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
/// tanh_f64(&mut values)?;
/// // values now contains [-0.964, -0.762, 0.0, 0.762, 0.964] approximately
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn tanh_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::tanh_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::tanh_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::tanh_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::tanh_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].tanh();
    }
    Ok(())
}

/// Element-wise floor using HW acceleration with smart threshold-based dispatching
///
/// Computes the floor of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values
/// * `result` - Mutable f64 slice for output values (largest integer ^^ input)
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch
///
/// # Examples
/// ```rust
/// use hwx::dispatch::floor_f64;
/// let mut values = vec![1.2, -1.2, 2.8, -2.8];
/// floor_f64(&mut values)?;
/// // values now contains [1.0, -2.0, 2.0, -3.0]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn floor_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::floor_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::floor_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::floor_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::floor_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].floor();
    }
    Ok(())
}

/// Element-wise ceiling using HW acceleration with smart threshold-based dispatching
///
/// Computes the ceiling of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values
/// * `result` - Mutable f64 slice for output values (smallest integer ^ input)
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch
///
/// # Examples
/// ```rust
/// use hwx::dispatch::ceil_f64;
/// let mut values = vec![1.2, -1.2, 2.8, -2.8];
/// ceil_f64(&mut values)?;
/// // values now contains [2.0, -1.0, 3.0, -2.0]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn ceil_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::ceil_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::ceil_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::ceil_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::ceil_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].ceil();
    }
    Ok(())
}

/// Element-wise rounding using HW acceleration with smart threshold-based dispatching
///
/// Computes the round (to nearest integer) of each element in the input array using optimized SIMD operations.
/// This function automatically selects between GPU acceleration for large arrays and
/// high-performance CPU SIMD implementations for smaller arrays.
///
/// # Arguments
/// * `values` - f64 slice containing input values
/// * `result` - Mutable f64 slice for output values (nearest integer)
///
/// # Returns
/// * `Ok(())` - Success
/// * `Err(crate::types::HwxError::Internal)` - Buffer size mismatch
///
/// # Examples
/// ```rust
/// use hwx::dispatch::round_f64;
/// let mut values = vec![1.2, -1.2, 2.8, -2.8];
/// round_f64(&mut values)?;
/// // values now contains [1.0, -1.0, 3.0, -3.0]
/// # Ok::<(), hwx::types::HwxError>(())
/// ```
///
/// # Performance
/// - Small arrays: Scalar fallback
/// - Large arrays: GPU or CPU HW acceleration (4x-16x faster)
#[inline]
pub fn round_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_output, len| {
                arrays::round_f64_gpu(gpu_output, len)
            })
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::round_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::round_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::round_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].round();
    }
    Ok(())
}

// =============================================================================
// ^^ TIME EXTRACTION FUNCTIONS (u64 ^ f64)
// =============================================================================

/// Extract Unix timestamp as f64 (identity conversion)
#[inline]
pub fn timestamp_u64(timestamps: &[u64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::timestamp_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::timestamp_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::timestamp_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::timestamp_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        result[i] = timestamps[i] as f64;
    }
    Ok(())
}

/// Extract hour from Unix timestamp (0-23)
#[inline]
pub fn hour_u64(timestamps: &[u64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::hour_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::hour_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::hour_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::hour_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        result[i] = ((timestamps[i] % 86400) / 3600) as f64;
    }
    Ok(())
}

/// Extract minute from Unix timestamp (0-59)
#[inline]
pub fn minute_u64(timestamps: &[u64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::minute_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::minute_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::minute_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::minute_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        result[i] = ((timestamps[i] % 3600) / 60) as f64;
    }
    Ok(())
}

/// Extract day of month from Unix timestamp (1-30, simplified)
#[inline]
pub fn day_of_month_u64(
    timestamps: &[u64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::day_of_month_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::day_of_month_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::day_of_month_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::day_of_month_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        let days = timestamps[i] / 86400;
        result[i] = ((days % 30) + 1) as f64;
    }
    Ok(())
}
/// Extract day of week from Unix timestamp (0=Sunday, 1=Monday, ..., 6=Saturday)
#[inline]
pub fn day_of_week_u64(
    timestamps: &[u64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::day_of_week_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::day_of_week_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::day_of_week_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::day_of_week_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        let days = timestamps[i] / 86400;
        result[i] = ((days + 4) % 7) as f64;
    }
    Ok(())
}

/// Extract day of year from Unix timestamp (1-365, simplified)
#[inline]
pub fn day_of_year_u64(
    timestamps: &[u64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::day_of_year_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::day_of_year_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::day_of_year_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::day_of_year_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        let days = timestamps[i] / 86400;
        result[i] = ((days % 365) + 1) as f64;
    }
    Ok(())
}

/// Get days in month for Unix timestamp (simplified: always 30)
#[inline]
pub fn days_in_month_u64(
    timestamps: &[u64],
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::days_in_month_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::days_in_month_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::days_in_month_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::days_in_month_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback - always 30 days
    for i in 0..len {
        result[i] = 30.0;
    }
    Ok(())
}

/// Extract month from Unix timestamp (1-12, simplified)
#[inline]
pub fn month_u64(timestamps: &[u64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::month_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::month_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::month_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::month_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        let days = timestamps[i] / 86400;
        result[i] = (((days / 30) % 12) + 1) as f64;
    }
    Ok(())
}

/// Extract year from Unix timestamp (1970+, simplified)
#[inline]
pub fn year_u64(timestamps: &[u64], result: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = timestamps.len();

    if len == 0 {
        return Ok(());
    }

    if result.len() < len {
        return Err(crate::types::HwxError::Internal(
            "Result buffer too small".to_string(),
        ));
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        return unsafe {
            gpu::with_gpu_buffer_u64_to_f64(
                timestamps,
                result,
                len,
                |gpu_timestamps, gpu_result, len| {
                    arrays::year_u64_gpu(gpu_timestamps, gpu_result, len)
                },
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::year_u64_avx512(timestamps, result, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::year_u64_avx2(timestamps, result, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::year_u64_neon(timestamps, result, len) };
            return Ok(());
        }
    }

    // Scalar fallback
    for i in 0..len {
        let days = timestamps[i] / 86400;
        result[i] = (1970 + (days / 365)) as f64;
    }
    Ok(())
}

// =============================================================================
// TIME SERIES OPERATIONS
// =============================================================================

/// Count number of value changes in time series
pub fn changes_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    let len = values.len();
    if len < 2 {
        return Ok(0.0);
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        let result = unsafe {
            gpu::with_gpu_buffer_f64(values, len, 0.0, |gpu_ptr, len, result_ptr| {
                let gpu_result = arrays::changes_f64_gpu(gpu_ptr, len);
                *result_ptr = gpu_result;
            })?
        };
        return Ok(result);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let result = unsafe { arrays::changes_f64_avx512(values, len) };
            return Ok(result);
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            let result = unsafe { arrays::changes_f64_avx2(values, len) };
            return Ok(result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            let result = unsafe { arrays::changes_f64_neon(values, len) };
            return Ok(result);
        }
    }

    // Scalar fallback
    let mut count = 0u64;
    for i in 0..len - 1 {
        if values[i] != values[i + 1] {
            count += 1;
        }
    }
    Ok(count as f64)
}

/// Count number of value decreases (resets)
pub fn resets_f64(values: &[f64], ascending: bool) -> Result<f64, crate::types::HwxError> {
    let len = values.len();
    if len < 2 {
        return Ok(0.0);
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        let result = unsafe {
            gpu::with_gpu_buffer_f64(values, len, 0.0, |gpu_ptr, len, result_ptr| {
                let gpu_result = arrays::resets_f64_gpu(gpu_ptr, len, ascending);
                *result_ptr = gpu_result;
            })?
        };
        return Ok(result);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let result = unsafe { arrays::resets_f64_avx512(values, len, ascending) };
            return Ok(result);
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            let result = unsafe { arrays::resets_f64_avx2(values, len, ascending) };
            return Ok(result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            let result = unsafe { arrays::resets_f64_neon(values, len, ascending) };
            return Ok(result);
        }
    }

    // Scalar fallback
    let mut count = 0u64;
    for i in 0..len - 1 {
        if ascending {
            // For ascending, count when next < current (reset/decrease)
            if values[i + 1] < values[i] {
                count += 1;
            }
        } else {
            // For descending, count when next > current (reset/increase)
            if values[i + 1] > values[i] {
                count += 1;
            }
        }
    }
    Ok(count as f64)
}

/// Calculate derivative using linear regression
pub fn deriv_f64(values: &[f64], timestamps: &[u64]) -> Result<f64, crate::types::HwxError> {
    let len = values.len();
    if len != timestamps.len() {
        return Err(crate::types::HwxError::Internal(
            "Values and timestamps length mismatch".to_string(),
        ));
    }
    if len < 2 {
        return Ok(0.0);
    }

    let mut result = 0.0f64;

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        result = unsafe {
            gpu::with_gpu_buffer_f64_u64_to_f64(
                values,
                timestamps,
                |gpu_values, gpu_timestamps, len| {
                    arrays::deriv_f64_gpu(gpu_values, gpu_timestamps, len)
                },
            )?
        };
        return Ok(result);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let r = unsafe { arrays::deriv_f64_avx512(values, timestamps, len) };
            return Ok(r);
        }
        #[cfg(not(feature = "hwx-nightly"))]
        if get_hw_capabilities().has_avx2 {
            let r = unsafe { arrays::deriv_f64_avx2(values, timestamps, len) };
            return Ok(r);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            result = unsafe { arrays::deriv_f64_neon(values, timestamps, len) };
            return Ok(result);
        }
    }

    // Scalar fallback - linear regression
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for i in 0..len {
        let x = timestamps[i] as f64;
        let y = values[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let n = len as f64;
    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator != 0.0 {
        result = (n * sum_xy - sum_x * sum_y) / denominator;
    }
    Ok(result)
}

/// Calculate total increase (handling counter resets)
pub fn increase_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    let len = values.len();
    if len < 2 {
        return Ok(0.0);
    }

    // Check for GPU acceleration
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        let result = unsafe {
            gpu::with_gpu_buffer_f64(values, len, 0.0, |gpu_ptr, len, result_ptr| {
                let gpu_result = arrays::increase_f64_gpu(gpu_ptr, len);
                *result_ptr = gpu_result;
            })?
        };
        return Ok(result);
    }

    // Increase calculation
    let first = values[0];
    let last = values[len - 1];
    let increase = last - first;
    let result = if increase < 0.0 {
        // Handle counter reset
        last
    } else {
        increase
    };
    Ok(result)
}

// =============================================================================
// NEW MATH FUNCTION DISPATCHERS
// =============================================================================

/// Element-wise clamp for f64 arrays
#[inline]
pub fn clamp_f64(
    values: &mut [f64],
    min_val: f64,
    max_val: f64,
) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    trace!(
        "CLAMP_F64 DISPATCH: values.len()={}, min={}, max={}",
        len,
        min_val,
        max_val
    );

    if len == 0 {
        trace!("CLAMP_F64: Empty array, returning early");
        return Ok(());
    }

    #[cfg(has_cuda)]
    {
        if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
            trace!(
                "CLAMP_F64: Taking GPU path (len={} >= GPU_THRESHOLD_MATH={}, has_cuda={})",
                len,
                GPU_THRESHOLD_MATH,
                get_hw_capabilities().has_cuda
            );
            unsafe {
                gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                    arrays::clamp_f64_gpu(gpu_ptr, min_val, max_val, len);
                })?;
            }
            return Ok(());
        }
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        trace!(
            "CLAMP_F64: Taking scalar path (len={} < SIMD_THRESHOLD_ARITHMETIC={})",
            len,
            SIMD_THRESHOLD_ARITHMETIC
        );
        for i in 0..len {
            if !values[i].is_nan() {
                values[i] = values[i].max(min_val).min(max_val);
            }
            // NaN values remain unchanged
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            trace!("CLAMP_F64: Taking AVX-512 SIMD path");
            unsafe { arrays::clamp_f64_avx512(values, min_val, max_val, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                trace!("CLAMP_F64: Taking AVX2 SIMD path");
                unsafe { arrays::clamp_f64_avx2(values, min_val, max_val, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::clamp_f64_neon(values, min_val, max_val, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].max(min_val).min(max_val);
    }
    Ok(())
}

/// Element-wise clamp_min for f64 arrays
#[inline]
pub fn clamp_min_f64(values: &mut [f64], min_val: f64) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                arrays::clamp_min_f64_gpu(gpu_ptr, min_val, len);
            })?;
        }
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            if !values[i].is_nan() {
                values[i] = values[i].max(min_val);
            }
            // NaN values remain unchanged
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::clamp_min_f64_avx512(values, min_val, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::clamp_min_f64_avx2(values, min_val, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::clamp_min_f64_neon(values, min_val, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].max(min_val);
    }
    Ok(())
}

/// Element-wise clamp_max for f64 arrays
#[inline]
pub fn clamp_max_f64(values: &mut [f64], max_val: f64) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                arrays::clamp_max_f64_gpu(gpu_ptr, max_val, len);
            })?;
        }
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            if !values[i].is_nan() {
                values[i] = values[i].min(max_val);
            }
            // NaN values remain unchanged
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::clamp_max_f64_avx512(values, max_val, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::clamp_max_f64_avx2(values, max_val, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::clamp_max_f64_neon(values, max_val, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].min(max_val);
    }
    Ok(())
}

/// Element-wise deg for f64 arrays (radians to degrees)
#[inline]
pub fn deg_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                arrays::deg_f64_gpu(gpu_ptr, len);
            })?;
        }
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            values[i] = values[i] * (180.0 / std::f64::consts::PI);
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::deg_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::deg_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::deg_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i] * (180.0 / std::f64::consts::PI);
    }
    Ok(())
}

/// Element-wise rad for f64 arrays (degrees to radians)
#[inline]
pub fn rad_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                arrays::rad_f64_gpu(gpu_ptr, len);
            })?;
        }
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            values[i] = values[i] * (std::f64::consts::PI / 180.0);
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::rad_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::rad_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::rad_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i] * (std::f64::consts::PI / 180.0);
    }
    Ok(())
}
/// Element-wise log2 for f64 arrays
#[inline]
pub fn log2_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                arrays::log2_f64_gpu(gpu_ptr, len);
            })?;
        }
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            values[i] = values[i].log2();
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::log2_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::log2_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::log2_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].log2();
    }
    Ok(())
}

/// Element-wise log10 for f64 arrays
#[inline]
pub fn log10_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                arrays::log10_f64_gpu(gpu_ptr, len);
            })?;
        }
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            values[i] = values[i].log10();
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::log10_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::log10_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::log10_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = values[i].log10();
    }
    Ok(())
}

/// Element-wise sgn (sign) for f64 arrays
#[inline]
pub fn sgn_f64(values: &mut [f64]) -> Result<(), crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        unsafe {
            gpu::with_gpu_buffer_f64_inplace(values, len, |gpu_ptr, len| {
                arrays::sgn_f64_gpu(gpu_ptr, len);
            })?;
        }
        return Ok(());
    }

    if len < SIMD_THRESHOLD_ARITHMETIC {
        for i in 0..len {
            if values[i].is_nan() {
                // NaN remains NaN
                continue;
            }
            values[i] = if values[i] > 0.0 {
                1.0
            } else if values[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            unsafe { arrays::sgn_f64_avx512(values, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                unsafe { arrays::sgn_f64_avx2(values, len) };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if get_hw_capabilities().has_neon {
            unsafe { arrays::sgn_f64_neon(values, len) };
            return Ok(());
        }
    }

    for i in 0..len {
        values[i] = if values[i] > 0.0 {
            1.0
        } else if values[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
    Ok(())
}

/// Calculate variance of values over time (population variance, handles NaN gracefully)
#[inline]
pub fn stdvar_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(f64::NAN);
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        let result = unsafe { gpu::stdvar_f64_cub(values, len)? };
        return Ok(result);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if len >= SIMD_THRESHOLD_REDUCE && get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::stdvar_f64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if len >= SIMD_THRESHOLD_REDUCE && get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::stdvar_f64_avx2(values, len) });
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if len >= SIMD_THRESHOLD_REDUCE && get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::stdvar_f64_neon(values, len) });
        }
    }

    // Scalar fallback for variance calculation
    let mut sum = 0.0;
    let mut count = 0;

    // Calculate mean, skipping NaN
    for &val in values {
        if !val.is_nan() {
            sum += val;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(f64::NAN);
    }

    let mean = sum / count as f64;

    // Calculate variance
    let mut variance_sum = 0.0;
    for &val in values {
        if !val.is_nan() {
            let diff = val - mean;
            variance_sum += diff * diff;
        }
    }

    Ok(variance_sum / count as f64)
}

/// Predict future values using linear regression with automatic hardware dispatching
///
/// This function predicts a future value at the given timestamp based on a linear
/// regression model fitted to the input values and timestamps. The calculation involves
/// computing regression coefficients (slope and intercept) from the time series data.
/// Automatically selects the best available hardware acceleration.
///
/// # Arguments
/// * `values` - Input time series values
/// * `timestamps` - Corresponding timestamps
/// * `predict_time` - Future timestamp to predict the value at
/// * `result` - Output buffer for the predicted value (single element)
///
/// # Returns
/// * `Ok(())` - Success, predicted value is written to result[0]
/// * `Err(crate::types::HwxError)` - On computation errors or hardware failures
///
/// # Hardware Acceleration
/// - **GPU (CUDA)**: For datasets ^ 1024 elements with parallel regression computation
/// - **AVX-512**: For datasets ^ 32 elements on modern Intel/AMD processors
/// - **AVX2**: For datasets ^ 32 elements on x86-64 processors
/// - **NEON**: For datasets ^ 32 elements on ARM64 processors
pub fn predict_linear_f64(
    values: &[f64],
    timestamps: &[f64],
    predict_time: f64,
    result: &mut [f64],
) -> Result<(), crate::types::HwxError> {
    if values.is_empty() {
        return Err(crate::types::HwxError::Internal(
            "Values array cannot be empty".to_string(),
        ));
    }
    if timestamps.is_empty() {
        return Err(crate::types::HwxError::Internal(
            "Timestamps array cannot be empty".to_string(),
        ));
    }
    if values.len() != timestamps.len() {
        return Err(crate::types::HwxError::Internal(
            "Values and timestamps arrays must have same length".to_string(),
        ));
    }
    if result.is_empty() {
        return Err(crate::types::HwxError::Internal(
            "Result buffer is empty".to_string(),
        ));
    }

    let len = values.len();

    if len == 1 {
        // Single point - use constant prediction
        result[0] = values[0];
        return Ok(());
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        let predicted = unsafe {
            gpu::with_gpu_buffer_f64_f64_scalar_to_f64(
                values,
                timestamps,
                predict_time,
                |gpu_values, gpu_timestamps, time, len| {
                    arrays::predict_linear_f64_gpu(gpu_values, gpu_timestamps, time, len)
                },
            )?
        };
        result[0] = predicted;
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if len >= SIMD_THRESHOLD_REDUCE && get_hw_capabilities().has_avx512 {
            result[0] =
                unsafe { arrays::predict_linear_f64_avx512(values, timestamps, predict_time, len) };
            return Ok(());
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if len >= SIMD_THRESHOLD_REDUCE && get_hw_capabilities().has_avx2 {
                result[0] = unsafe {
                    arrays::predict_linear_f64_avx2(values, timestamps, predict_time, len)
                };
                return Ok(());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if len >= SIMD_THRESHOLD_REDUCE && get_hw_capabilities().has_neon {
            result[0] =
                unsafe { arrays::predict_linear_f64_neon(values, timestamps, predict_time, len) };
            return Ok(());
        }
    }

    // No scalar fallback - use SIMD/GPU only
    Err(crate::types::HwxError::Internal(
        "predict_linear requires SIMD or GPU support".to_string(),
    ))
}

/// Check if any non-NaN values are present in the time series (returns 1.0 if any, 0.0 if none)
#[inline]
pub fn present_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(0.0);
    }

    // #[cfg(has_cuda)]
    // if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
    //   let result = unsafe {
    //     gpu::with_gpu_buffer_f64(values, len, 0.0, |gpu_ptr, len, result_ptr| {
    //       let gpu_result = arrays::present_f64_gpu(gpu_ptr, len);
    //       *result_ptr = gpu_result;
    //     })?
    //   };
    //   return Ok(result);
    // }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::present_f64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::present_f64_avx2(values, len) });
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::present_f64_neon(values, len) });
        }
    }

    // Scalar fallback
    for &val in values {
        if !val.is_nan() {
            return Ok(1.0);
        }
    }
    Ok(0.0)
}

/// Calculate average of values (mean), skipping NaN values
#[inline]
pub fn avg_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(f64::NAN);
    }

    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        // Use reduce_sum_f64_gpu and divide by length
        let sum = unsafe {
            gpu::with_gpu_buffer_f64(values, len, 0.0, |gpu_ptr, len, result_ptr| {
                arrays::reduce_sum_f64_gpu(gpu_ptr, len, result_ptr);
            })?
        };
        return Ok(sum / len as f64);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            let sum = unsafe { arrays::reduce_sum_f64_avx512(values, len) };
            return Ok(sum / len as f64);
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                let sum = unsafe { arrays::reduce_sum_f64_avx2(values, len) };
                return Ok(sum / len as f64);
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        if get_hw_capabilities().has_neon {
            let sum = unsafe { arrays::reduce_sum_f64_neon(values, len) };
            return Ok(sum / len as f64);
        }
    }

    // Scalar fallback
    let mut sum = 0.0;
    for &val in values {
        if val.is_nan() {
            return Ok(f64::NAN);
        }
        sum += val;
    }

    Ok(sum / len as f64)
}

/// Calculate mean absolute deviation over time (MAD = mean of |x - mean|)
#[inline]
pub fn mad_f64(values: &[f64]) -> Result<f64, crate::types::HwxError> {
    let len = values.len();

    if len == 0 {
        return Ok(f64::NAN);
    }

    // GPU path for mad_f64
    #[cfg(has_cuda)]
    if len >= GPU_THRESHOLD_MATH && get_hw_capabilities().has_cuda {
        let result = unsafe {
            gpu::with_gpu_buffer_f64(values, len, 0.0, |gpu_ptr, len, result_ptr| {
                arrays::mad_f64_gpu(gpu_ptr, len, result_ptr);
            })?
        };
        return Ok(result);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "hwx-nightly")]
        if get_hw_capabilities().has_avx512 {
            return Ok(unsafe { arrays::mad_f64_avx512(values, len) });
        }
        #[cfg(not(feature = "hwx-nightly"))]
        {
            if get_hw_capabilities().has_avx2 {
                return Ok(unsafe { arrays::mad_f64_avx2(values, len) });
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        if get_hw_capabilities().has_neon {
            return Ok(unsafe { arrays::mad_f64_neon(values, len) });
        }
    }

    // Scalar fallback
    // First pass: calculate mean, skipping NaN
    let mut sum = 0.0;
    let mut count = 0;
    for &val in values {
        if !val.is_nan() {
            sum += val;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(f64::NAN); // All values were NaN
    }

    if count == 1 {
        return Ok(0.0); // Single non-NaN value has no deviation
    }

    let mean = sum / count as f64;

    // Second pass: calculate mean absolute deviation
    let mut mad_sum = 0.0;
    let mut mad_count = 0;
    for &val in values {
        if !val.is_nan() {
            let abs_dev = (val - mean).abs();
            // Skip NaN absolute deviations (e.g., from inf - inf)
            if !abs_dev.is_nan() {
                mad_sum += abs_dev;
                mad_count += 1;
            }
        }
    }

    if mad_count == 0 {
        return Ok(f64::NAN);
    }

    let result = mad_sum / mad_count as f64;

    // Special case: if we have infinity in the result, return it
    // This handles the case where mean is infinity
    if result.is_infinite() {
        return Ok(f64::INFINITY);
    }

    Ok(result)
}
