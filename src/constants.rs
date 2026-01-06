// SPDX-License-Identifier: Apache-2.0

//! Common constants used across implementations
//!
//! This module centralizes lane counts, thresholds, and related constants used by
//! scalar/SIMD/CUDA paths.

// =============================================================================
// VECTOR DIMENSIONS
// =============================================================================
pub const HNSW_DIMENSION: usize = 1536;

// =============================================================================
// NUMERIC PARSING CONSTANTS
// =============================================================================

/// Maximum digits to parse as actual integer value
/// Beyond this, we classify as Integer but don't compute numeric value
/// u64::MAX = 18,446,744,073,709,551,615 (20 digits)
/// i64::MAX = 9,223,372,036,854,775,807 (19 digits)
/// Most databases cap integers at 19-20 digits, matching 64-bit limits
pub const MAX_INTEGER_DIGITS: usize = 20;

// =============================================================================
// MAX TOKENS PER STRING
// =============================================================================
pub const MAX_TOKENS_PER_STRING: usize = 16384;

// =============================================================================
// SIMD Lowercase Threshold
pub const SIMD_LOWERCASE_THRESHOLD: usize = 8;
// =============================================================================

// =============================================================================
// SIMD Lane Counts by Architecture
// =============================================================================

// AVX-512 Constants (Nightly feature only)
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub use avx512_constants::*;
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod avx512_constants {
    pub const LANES_AVX512_F64: usize = 8; // 512/64 = 8 f64 elements
    pub const LANES_AVX512_U64: usize = 8; // 512/64 = 8 u64 elements
    pub const LANES_AVX512_U32: usize = 16; // 512/32 = 16 u32 elements
    pub const LANES_AVX512_U16: usize = 32; // 512/16 = 32 u16 elements
    pub const LANES_AVX512_BYTES: usize = 64; // 512/8 = 64 byte elements
}

// x86/x86_64 Stable Constants (AVX2 only)
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub use x86_stable_constants::*;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
mod x86_stable_constants {
    // AVX2 (256-bit registers)
    pub const LANES_AVX2_F64: usize = 4; // 256/64 = 4 f64 elements
    pub const LANES_AVX2_U64: usize = 4; // 256/64 = 4 u64 elements
    pub const LANES_AVX2_U32: usize = 8; // 256/32 = 8 u32 elements
    pub const LANES_AVX2_U16: usize = 16; // 256/16 = 16 u16 elements
    pub const LANES_AVX2_BYTES: usize = 32; // 256/8 = 32 byte elements
}

// NEON Constants (ARM64 only)
#[cfg(target_arch = "aarch64")]
pub use neon_constants::*;
#[cfg(target_arch = "aarch64")]
mod neon_constants {
    pub const LANES_NEON_F64: usize = 2; // 128/64 = 2 f64 elements
    pub const LANES_NEON_U64: usize = 2; // 128/64 = 2 u64 elements
    pub const LANES_NEON_U32: usize = 4; // 128/32 = 4 u32 elements
    pub const LANES_NEON_U16: usize = 8; // 128/16 = 8 u16 elements
    pub const LANES_NEON_BYTES: usize = 16; // 128/8 = 16 byte elements
}

// x86/x86_64 Stable Constants (AVX2 only)
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "hwx-nightly",
))]
pub const MAX_POSITIONS_AVX512: usize = 64;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
// Architecture-specific constants for word boundary tracking
pub const MAX_POSITIONS_AVX2: usize = 32;
#[cfg(target_arch = "aarch64")]
pub const MAX_POSITIONS_NEON: usize = 32;

// GPU constants for CUDA acceleration
// GPU can handle much larger buffers than SIMD
pub const MAX_POSITIONS_GPU: usize = 4096;

// =============================================================================
// SIMD Performance Thresholds - Scientifically Tuned
// =============================================================================

// Infinite Loop Prevention
pub const MAX_ITERATIONS: usize = 1000000; // Prevent infinite loops in SIMD functions - reduced for faster debugging

// When disable-hwx feature is enabled, set all thresholds to usize::MAX to force scalar implementations
#[cfg(feature = "disable-hwx")]
mod thresholds {
    // Traversal Operations Thresholds
    pub const SIMD_THRESHOLD_FILTER_DELETED: usize = usize::MAX; // Filter deleted docs
    pub const SIMD_THRESHOLD_INTERSECT: usize = usize::MAX; // Array intersection
    pub const SIMD_THRESHOLD_UNION: usize = usize::MAX; // Array union
    pub const SIMD_THRESHOLD_DEDUP: usize = usize::MAX; // Array deduplication
    pub const SIMD_THRESHOLD_FILTER_RANGE: usize = usize::MAX; // Range filtering
    pub const SIMD_THRESHOLD_POSTINGS: usize = usize::MAX; // Postings processing

    // String Operations Thresholds
    pub const SIMD_THRESHOLD_STRING_PREFIX: usize = usize::MAX; // Prefix matching
    pub const SIMD_THRESHOLD_STRING_EXACT: usize = usize::MAX; // Exact phrase matching
    pub const SIMD_THRESHOLD_STRING_FIELD: usize = usize::MAX; // Field matching
    pub const SIMD_THRESHOLD_STRING_REGEX: usize = usize::MAX; // Regex filtering
    pub const SIMD_THRESHOLD_STRING_WILDCARD: usize = usize::MAX; // Wildcard matching

    // Time Operations Thresholds
    pub const SIMD_THRESHOLD_TIME_FILTER: usize = usize::MAX; // Time range operations

    // Distance Operations Thresholds
    pub const SIMD_THRESHOLD_DISTANCE: usize = usize::MAX; // Vector distance calculations

    // Array Operations Thresholds
    pub const SIMD_THRESHOLD_REDUCE: usize = usize::MAX; // Sum, min, max operations
    pub const SIMD_THRESHOLD_REDUCE_WEIGHTED: usize = usize::MAX; // Weighted operations
    pub const SIMD_THRESHOLD_VECTORIZED: usize = usize::MAX; // Vectorized operations (subtract, quantize)
    pub const SIMD_THRESHOLD_ARITHMETIC: usize = usize::MAX; // Element-wise arithmetic operations (add, sub, mul, div)

    // Merge Operations Thresholds
    pub const SIMD_THRESHOLD_MERGE: usize = usize::MAX; // Merge operations

    // Math Operations Thresholds
    pub const SIMD_THRESHOLD_MATH_FMA: usize = usize::MAX; // Fused multiply-add
    pub const SIMD_THRESHOLD_MATH_INTERPOLATE: usize = usize::MAX; // Interpolation
    pub const SIMD_THRESHOLD_MATH_PERCENTILE: usize = usize::MAX; // Percentile calculations

    // Search Operations Thresholds
    pub const SIMD_THRESHOLD_BINARY_SEARCH: usize = usize::MAX; // Binary search operations
}

// Normal thresholds when SIMD is enabled (default)
#[cfg(not(feature = "disable-hwx"))]
mod thresholds {
    // Traversal Operations Thresholds
    pub const SIMD_THRESHOLD_FILTER_DELETED: usize = 32; // Filter deleted docs
    pub const SIMD_THRESHOLD_INTERSECT: usize = 16; // Array intersection
    pub const SIMD_THRESHOLD_UNION: usize = 16; // Array union
    pub const SIMD_THRESHOLD_DEDUP: usize = 24; // Array deduplication
    pub const SIMD_THRESHOLD_FILTER_RANGE: usize = 32; // Range filtering
    pub const SIMD_THRESHOLD_POSTINGS: usize = 8; // Postings processing

    // String Operations Thresholds
    pub const SIMD_THRESHOLD_STRING_PREFIX: usize = 8; // Prefix matching
    pub const SIMD_THRESHOLD_STRING_EXACT: usize = 4; // Exact phrase matching
    pub const SIMD_THRESHOLD_STRING_FIELD: usize = 8; // Field matching
    pub const SIMD_THRESHOLD_STRING_REGEX: usize = 12; // Regex filtering
    pub const SIMD_THRESHOLD_STRING_WILDCARD: usize = 8; // Wildcard matching

    // Time Operations Thresholds
    pub const SIMD_THRESHOLD_TIME_FILTER: usize = 24; // Time range operations

    // Distance Operations Thresholds
    pub const SIMD_THRESHOLD_DISTANCE: usize = 32; // Vector distance calculations

    // Array Operations Thresholds
    pub const SIMD_THRESHOLD_REDUCE: usize = 32; // Sum, min, max operations
    pub const SIMD_THRESHOLD_REDUCE_WEIGHTED: usize = 16; // Weighted operations
    pub const SIMD_THRESHOLD_VECTORIZED: usize = 16; // Vectorized operations (subtract, quantize)
    pub const SIMD_THRESHOLD_ARITHMETIC: usize = 8; // Element-wise arithmetic operations (add, sub, mul, div)

    // Merge Operations Thresholds
    pub const SIMD_THRESHOLD_MERGE: usize = 16; // Merge operations

    // Math Operations Thresholds
    pub const SIMD_THRESHOLD_MATH_FMA: usize = 16; // Fused multiply-add
    pub const SIMD_THRESHOLD_MATH_INTERPOLATE: usize = 8; // Interpolation
    pub const SIMD_THRESHOLD_MATH_PERCENTILE: usize = 64; // Percentile calculations

    // Search Operations Thresholds
    pub const SIMD_THRESHOLD_BINARY_SEARCH: usize = 32; // Binary search operations

    // HWX general threshold for operations
    pub const HWX_THRESHOLD: usize = 32; // General HWX operation threshold
}

// Re-export the thresholds at the module level
pub use thresholds::*;

// =============================================================================
// SIMD Processing Constants - Standardized Across Architectures
// =============================================================================

// SIMD unroll factors for performance optimization
pub const UNROLL_FACTOR_PREFIX: usize = 4; // Prefix matching operations
pub const UNROLL_FACTOR_PHRASE: usize = 2; // Phrase search operations
pub const UNROLL_FACTOR_COMPARE: usize = 4; // String comparison operations

// SIMD mask constants
pub const FULL_MATCH_MASK_AVX2: i32 = -1i32; // All 32 bits set for AVX2
pub const FULL_MATCH_MASK_AVX512: i64 = -1i64; // All 64 bits set for AVX-512

// Architecture-specific chunk sizes for classification
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const CHUNK_SIZE_AVX512: usize = 64; // Matches AVX-512 64-byte processing

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub const CHUNK_SIZE_AVX2: usize = 32; // Matches AVX2 32-byte processing

#[cfg(target_arch = "aarch64")]
pub const CHUNK_SIZE_NEON: usize = 16; // Matches NEON 16-byte processing

// =============================================================================
// Architecture-Specific Chunking Sizes
// =============================================================================

// AVX-512 Constants (512-bit registers = 64 bytes)
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const MAX_STRINGS_AVX512: usize = 2048; // Larger chunks for wider registers
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const MAX_DATASETS_AVX512: usize = 2048;
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const MAX_ARRAYS_AVX512: usize = 2048;

// AVX2 Constants (256-bit registers = 32 bytes)
#[cfg(all(
    not(feature = "hwx-nightly"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const MAX_STRINGS_AVX2: usize = 1024; // Medium chunks for AVX2
#[cfg(all(
    not(feature = "hwx-nightly"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const MAX_DATASETS_AVX2: usize = 1024;
#[cfg(all(
    not(feature = "hwx-nightly"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const MAX_ARRAYS_AVX2: usize = 1024;

// NEON Constants (128-bit registers = 16 bytes)
#[cfg(target_arch = "aarch64")]
pub const MAX_STRINGS_NEON: usize = 512; // Smaller chunks for ARM NEON
#[cfg(target_arch = "aarch64")]
pub const MAX_DATASETS_NEON: usize = 512;
#[cfg(target_arch = "aarch64")]
pub const MAX_ARRAYS_NEON: usize = 512;

// GPU Constants (massive parallelism)

pub const MAX_STRINGS_GPU: usize = 8192; // Very large chunks for GPU

pub const MAX_DATASETS_GPU: usize = 8192;

pub const MAX_ARRAYS_GPU: usize = 8192;

// Fallback Constants (scalar)
pub const MAX_STRINGS: usize = 256; // Conservative default for scalar
pub const MAX_DATASETS: usize = 256;
pub const MAX_ARRAYS: usize = 256;
pub const MAX_TERMS: usize = 256; // Term processing arrays

// =============================================================================
// Architecture-Specific Stack and Buffer Sizes
// =============================================================================

// Quicksort Stack Sizes - deeper stacks for wider SIMD
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const QUICKSORT_STACK_SIZE_AVX512: usize = 2048; // Deeper for 512-bit processing
#[cfg(all(
    not(feature = "hwx-nightly"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const QUICKSORT_STACK_SIZE_AVX2: usize = 1024; // Medium depth for 256-bit
#[cfg(target_arch = "aarch64")]
pub const QUICKSORT_STACK_SIZE_NEON: usize = 512; // Shallower for 128-bit NEON

pub const QUICKSORT_STACK_SIZE_GPU: usize = 4096; // Very deep for GPU parallelism
pub const QUICKSORT_STACK_SIZE: usize = 256; // Conservative scalar fallback

// SIMD Unroll Factors - more unrolling for wider registers
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const UNROLL_FACTOR_AVX512: usize = 8; // Aggressive unrolling for 512-bit
#[cfg(all(
    not(feature = "hwx-nightly"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub const UNROLL_FACTOR_AVX2: usize = 4; // Medium unrolling for 256-bit
#[cfg(target_arch = "aarch64")]
pub const UNROLL_FACTOR_NEON: usize = 2; // Conservative for 128-bit NEON

pub const UNROLL_FACTOR_GPU: usize = 16; // Very aggressive for GPU parallelism
pub const UNROLL_FACTOR: usize = 2; // Conservative scalar fallback
pub const MAX_TEXTS: usize = 1024; // Text processing arrays

// =============================================================================
// GPU/CUDA Constants
// =============================================================================

pub use gpu_constants::*;

mod gpu_constants {
    // GPU warp/block sizes - similar to SIMD lanes but for GPU threads
    pub const GPU_WARP_SIZE: usize = 32; // NVIDIA warp size (like SIMD lanes)
    pub const GPU_BLOCK_SIZE_SMALL: usize = 128; // For small kernels
    pub const GPU_BLOCK_SIZE_MEDIUM: usize = 256; // Standard block size
    pub const GPU_BLOCK_SIZE_LARGE: usize = 512; // For memory-bound kernels

    // GPU grid dimensions
    pub const GPU_MAX_GRID_SIZE: usize = 65535; // Max blocks per dimension
    pub const GPU_MAX_THREADS: usize = 1024; // Max threads per block

    // GPU shared memory sizes (per SM)
    pub const GPU_SHARED_MEM_SIZE: usize = 49152; // 48KB shared memory
    pub const GPU_L1_CACHE_SIZE: usize = 128; // 128 byte cache line

    // GPU memory alignment
    pub const GPU_MEM_ALIGN: usize = 256; // Coalesced access alignment

    // GPU thresholds - optimized for GPU acceleration sweet spot
    pub const GPU_THRESHOLD_DISTANCE: usize = 1024; // Distance calculations
    pub const GPU_THRESHOLD_INTERSECT: usize = 8192; // Array intersection
    pub const GPU_THRESHOLD_SORT: usize = 4096; // Sorting operations
    pub const GPU_THRESHOLD_FILTER: usize = 8192; // Filtering operations
    pub const GPU_THRESHOLD_REDUCE: usize = 2048; // Reduction operations
    pub const GPU_THRESHOLD_MATH: usize = 1024; // Mathematical operations
    pub const GPU_THRESHOLD_STRING: usize = 4096; // String operations
    pub const GPU_THRESHOLD_TERMS: usize = 1000; // Terms aggregation
    pub const GPU_THRESHOLD_SEARCH: usize = 2048; // Search operations (binary, exponential)
}

// =============================================================================
// Mathematical Constants
// =============================================================================

pub const EPSILON_F32: f32 = 1e-10; // Small value to prevent division by zero
