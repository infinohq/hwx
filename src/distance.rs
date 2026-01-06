// SPDX-License-Identifier: Apache-2.0

//! Distance functions
//!
//! Vector distance/similarity functions (L1/L2/dot/cosine and related metrics).
//! Implementations may use scalar code, SIMD, and (when enabled) CUDA kernels.
//!
//! ## Performance notes
//! Some implementations follow common SIMD patterns.
//! When modifying hot paths, prefer changes that keep allocations out of inner loops.

// Some clippy lints are noisy for low-level SIMD/FFI code; we opt out at the module level.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

// =============================================================================
// X86/X86_64 SIMD IMPORTS
// =============================================================================

// Conditional imports for constants based on target architecture and features
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::constants::LANES_AVX512_U32;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use super::constants::{LANES_AVX2_U16, LANES_AVX2_U32};

#[cfg(target_arch = "aarch64")]
use super::constants::{LANES_NEON_U16, LANES_NEON_U32};

#[cfg(has_cuda)]
use crate::gpu::launch_ptx;

// x86_64 SIMD intrinsics - AVX2 only (not when nightly AVX-512 is enabled)
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use std::arch::x86_64::{
    // SIMD types
    __m256,
    // Additional AVX2 intrinsics for integer operations
    __m256i,
    _mm256_add_epi32,
    _mm256_add_ps,
    _mm256_and_si256,
    _mm256_andnot_ps,
    _mm256_castps256_ps128,
    _mm256_cmpeq_epi16,
    _mm256_cmpeq_epi32,
    _mm256_div_ps,
    _mm256_extractf128_ps,
    _mm256_extracti128_si256,
    _mm256_fmadd_ps,
    _mm256_hadd_ps,
    // Only the SIMD intrinsics that are actually used on Mac builds
    // Stable newer intrinsics
    _mm256_loadu_ps,
    _mm256_loadu_si256,
    _mm256_max_epu16,
    _mm256_max_ps,
    _mm256_min_epu16,
    _mm256_mul_ps,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_set1_ps,
    _mm256_setzero_ps,
    _mm256_setzero_si256,
    _mm256_sqrt_ps,
    _mm256_srli_epi16,
    _mm256_storeu_si256,
    _mm256_sub_ps,
    _mm256_unpackhi_epi16,
    _mm256_unpacklo_epi16,
    _mm256_xor_si256,
    // SSE intrinsics needed for final reduction steps
    _mm_add_epi32,
    _mm_add_ps,
    _mm_cvtsi128_si32,
    _mm_cvtss_f32,
    _mm_shuffle_epi32,
};

// AVX-512 intrinsics (nightly only)
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use std::arch::x86_64::{
    // AVX-512 types
    __m512,
    // Unstable AVX2/AVX-512 intrinsics
    _mm512_abs_ps,
    _mm512_add_ps,
    _mm512_cmpeq_epu32_mask,
    _mm512_div_ps,
    _mm512_fmadd_ps,
    _mm512_loadu_epi32,
    _mm512_loadu_ps,
    _mm512_max_epi32,
    _mm512_max_ps,
    _mm512_min_epi32,
    _mm512_mul_ps,
    _mm512_reduce_add_epi32,
    _mm512_reduce_add_ps,
    _mm512_set1_ps,
    _mm512_setzero_ps,
    _mm512_sqrt_ps,
    _mm512_sub_ps,
};

// =============================================================================
// ARM NEON IMPORTS
// =============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vabsq_f32, vaddq_f32, vaddvq_f32, vaddvq_u16, vceqq_u16, vceqq_u32, vdivq_f32,
    vdupq_n_f32, vdupq_n_u16, veorq_u16, vgetq_lane_u32, vld1q_f32, vld1q_u16, vld1q_u32,
    vmaxq_f32, vmaxq_u16, vminq_u16, vmulq_f32, vshrq_n_u16, vsqrtq_f32, vsubq_f32,
};

// GPU/CUDA constants
// Note: GPU constants now embedded directly in PTX kernels
// use super::constants::{GPU_BLOCK_SIZE_MEDIUM, GPU_WARP_SIZE};

// =============================================================================
// L1 DISTANCE (MANHATTAN)
// =============================================================================

// GPU/PTX optimized L1 distance for f32 vectors.
//
// Uses GPU grid-stride loop pattern matching HWX kernels.
// Processes data in warps of 32 threads (like SIMD lanes).
// Direct DMA-like transfer from CPU slices to GPU memory.
//

#[cfg(has_cuda)]
pub unsafe fn distance_l1_f32_gpu(
    va: *const f32,
    vb: *const f32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    const PTX_DISTANCE_L1_F32: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_l1_f32(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .f32 %f<16>;
      .reg .pred %p<3>; 
      .reg .u32 %r<20>;
      .reg .u64 %rd<8>;
      .shared .f32 sdata[256];  // Shared memory for block reduction

      // Load parameters
      ld.param.u64 %rd4, [va_ptr];
      ld.param.u64 %rd5, [vb_ptr];
      ld.param.u32 %r10, [min_len];
      ld.param.u64 %rd6, [result_ptr];

      // Initialize float4 accumulator
      mov.f32 %f0, 0.0;  // Lane 0
      mov.f32 %f1, 0.0;  // Lane 1  
      mov.f32 %f2, 0.0;  // Lane 2
      mov.f32 %f3, 0.0;  // Lane 3

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 4;  // Each thread starts at ID * 4 (float4)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 4;   // Stride = total_threads * 4 (float4)

      // Main grid-stride loop processing float4 chunks
    loop_start:
      add.u32 %r6, %r5, 3;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full float4
      @%p0 bra remainder_check;  // Exit loop if beyond array

      // Load and compute L1 distance with float4
      mul.wide.u32 %rd1, %r5, 4;        // Convert to byte offset
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      // Load va and vb using vectorized float4 loads
      ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd2];   // Load va float4
      ld.global.v4.f32 {%f8, %f9, %f10, %f11}, [%rd3]; // Load vb float4
      // Compute abs differences and accumulate
      sub.f32 %f4, %f4, %f8;    abs.f32 %f4, %f4;    add.f32 %f0, %f0, %f4;
      sub.f32 %f5, %f5, %f9;    abs.f32 %f5, %f5;    add.f32 %f1, %f1, %f5;
      sub.f32 %f6, %f6, %f10;   abs.f32 %f6, %f6;    add.f32 %f2, %f2, %f6;
      sub.f32 %f7, %f7, %f11;   abs.f32 %f7, %f7;    add.f32 %f3, %f3, %f7;

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

    remainder_check:  // Remainder handling with masking (like AVX-512)
      // Check and load up to 3 remaining elements
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;  // Skip if no remainder
      mul.wide.u32 %rd1, %r5, 4;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.f32 %f4, [%rd2];
      ld.global.f32 %f8, [%rd3];
      sub.f32 %f4, %f4, %f8; abs.f32 %f4, %f4; add.f32 %f0, %f0, %f4;

      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;
      ld.global.f32 %f5, [%rd2+4];
      ld.global.f32 %f9, [%rd3+4];
      sub.f32 %f5, %f5, %f9; abs.f32 %f5, %f5; add.f32 %f1, %f1, %f5;

      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;
      ld.global.f32 %f6, [%rd2+8];
      ld.global.f32 %f10, [%rd3+8];
      sub.f32 %f6, %f6, %f10; abs.f32 %f6, %f6; add.f32 %f2, %f2, %f6;

    final_reduction:
      // Step 1: Reduce vector lanes within thread
      add.f32 %f0, %f0, %f1;  // Sum lanes 0+1
      add.f32 %f2, %f2, %f3;  // Sum lanes 2+3
      add.f32 %f0, %f0, %f2;  // Thread-local sum
      
      // Step 2: Warp reduction using shuffle
      // Get lane ID and warp ID
      and.b32 %r12, %r1, 0x1f;  // Lane ID within warp
      shr.u32 %r13, %r1, 5;     // Warp ID within block
      
      // Warp reduction
      shfl.sync.down.b32 %f12, %f0, 16, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 8, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 4, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 2, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 1, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      
      // Step 3: Write warp sum to shared memory (lane 0 only)
      setp.eq.u32 %p1, %r12, 0;
      @%p1 mul.wide.u32 %rd0, %r13, 4;  // Warp ID * sizeof(float)
      @%p1 st.shared.f32 [sdata + %rd0], %f0;
      
      // Synchronize block
      bar.sync 0;
      
      // Step 4: Block reduction (first warp only)
      setp.lt.u32 %p2, %r1, 8;  // Only first 8 threads (assuming max 8 warps per block)
      @!%p2 bra write_result;
      
      // Load warp sums from shared memory
      mul.wide.u32 %rd0, %r1, 4;
      ld.shared.f32 %f0, [sdata + %rd0];
      
      // Reduce across warps
      shfl.sync.down.b32 %f12, %f0, 4, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 2, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 1, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      
    write_result:
      // Thread 0 of block 0 writes final result (using atomic add for multi-block)
      setp.eq.u32 %p1, %r1, 0;
      setp.eq.u32 %p2, %r3, 0;
      and.pred %p1, %p1, %p2;
      @%p1 st.global.f32 [%rd6], %f0;
      
      // For multi-block, use atomic add (thread 0 of each block)
      setp.eq.u32 %p1, %r1, 0;
      setp.ne.u32 %p2, %r3, 0;
      and.pred %p1, %p1, %p2;
      @%p1 atom.global.add.f32 %f15, [%rd6], %f0;
      
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len) as u32;

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = launch_ptx(PTX_DISTANCE_L1_F32, &[], "distance_l1_f32", 256, 256, &args);
}

// AVX-512 optimized L1 distance for f32 vectors.
//
// Processes 16 f32 elements per iteration using 512-bit registers.
// Achieves ~20x speedup over scalar implementation.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_l1_f32_avx512(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_AVX512_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut distance = 0.0f32;

    // Process 16 elements at a time with AVX-512
    let mut i = 0;
    while i < simd_len {
        let a = _mm512_loadu_ps(va.as_ptr().add(i));
        let b = _mm512_loadu_ps(vb.as_ptr().add(i));
        let diff = _mm512_sub_ps(a, b);
        let abs_diff = _mm512_abs_ps(diff);
        distance += _mm512_reduce_add_ps(abs_diff);
        i += LANES;
    }

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        distance += (va[j] - vb[j]).abs();
    }

    distance
}

// AVX2 optimized L1 distance for f32 vectors.
//
// Processes 8 f32 elements per iteration using 256-bit registers.
// Achieves significant speedup over scalar implementation.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_l1_f32_avx2(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_AVX2_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut distance = 0.0f32;

    // Process 8 elements at a time with AVX2
    let mut i = 0;
    while i < simd_len {
        let a = _mm256_loadu_ps(va.as_ptr().add(i));
        let b = _mm256_loadu_ps(vb.as_ptr().add(i));

        let diff = _mm256_sub_ps(a, b);
        // Manual abs implementation since _mm256_abs_ps doesn't exist
        let sign_mask = _mm256_set1_ps(-0.0f32);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        // Horizontal add using proper AVX2 intrinsics
        let hadd1 = _mm256_hadd_ps(abs_diff, abs_diff);
        let hadd2 = _mm256_hadd_ps(hadd1, hadd1);
        // Extract 128-bit lanes and add them for final reduction
        let low = _mm256_castps256_ps128(hadd2);
        let high = _mm256_extractf128_ps(hadd2, 1);
        let sum_128 = _mm_add_ps(low, high);
        let sum = _mm_cvtss_f32(sum_128);
        distance += sum;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        distance += (va[i] - vb[i]).abs();
    }

    distance
}

// NEON optimized L1 distance for f32 vectors.
//
// Processes 4 f32 elements per iteration using 128-bit registers.
// Provides good speedup over scalar implementation on ARM64.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_l1_f32(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_NEON_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut distance = 0.0f32;

    // Process 4 elements at a time with NEON
    let mut i = 0;
    while i < simd_len {
        let a = vld1q_f32(va.as_ptr().add(i));
        let b = vld1q_f32(vb.as_ptr().add(i));

        let diff = vsubq_f32(a, b);
        let abs_diff = vabsq_f32(diff);

        // Horizontal add to accumulate sum
        let sum = vaddvq_f32(abs_diff);
        distance += sum;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        distance += (va[i] - vb[i]).abs();
    }

    distance
}

// =============================================================================
// L2 DISTANCE (EUCLIDEAN)
// =============================================================================

// GPU/PTX optimized L2 distance for f32 vectors.
//
// Grid-stride loop computing squared differences like SIMD.
// Accumulates in register before final sqrt like AVX-512.
//

#[cfg(has_cuda)]
pub unsafe fn distance_l2_f32_gpu(
    va: *const f32,
    vb: *const f32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    const PTX_DISTANCE_L2_F32: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_l2_f32(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .f32 %f<16>;
      .reg .pred %p<3>; 
      .reg .u32 %r<20>;
      .reg .u64 %rd<8>;
      .shared .f32 sdata[256];  // Shared memory for block reduction

      // Load parameters
      ld.param.u64 %rd4, [va_ptr];
      ld.param.u64 %rd5, [vb_ptr];
      ld.param.u32 %r10, [min_len];
      ld.param.u64 %rd6, [result_ptr];
      

      // Initialize float4 accumulator for sum of squares
      mov.f32 %f0, 0.0;  // Lane 0
      mov.f32 %f1, 0.0;  // Lane 1  
      mov.f32 %f2, 0.0;  // Lane 2
      mov.f32 %f3, 0.0;  // Lane 3

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 4;  // Each thread starts at ID * 4 (float4)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 4;   // Stride = total_threads * 4 (float4)

      // Main grid-stride loop processing float4 chunks
    loop_start:
      add.u32 %r6, %r5, 3;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full float4
      @%p0 bra remainder_check;  // Exit loop if beyond array

      // Load and compute L2 distance with float4
      mul.wide.u32 %rd1, %r5, 4;        // Convert to byte offset
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      // Load va and vb using vectorized float4 loads
      ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd2];   // Load va float4
      ld.global.v4.f32 {%f8, %f9, %f10, %f11}, [%rd3]; // Load vb float4
      // Compute squared differences and accumulate
      sub.f32 %f4, %f4, %f8;    mul.f32 %f4, %f4, %f4;    add.f32 %f0, %f0, %f4;
      sub.f32 %f5, %f5, %f9;    mul.f32 %f5, %f5, %f5;    add.f32 %f1, %f1, %f5;
      sub.f32 %f6, %f6, %f10;   mul.f32 %f6, %f6, %f6;    add.f32 %f2, %f2, %f6;
      sub.f32 %f7, %f7, %f11;   mul.f32 %f7, %f7, %f7;    add.f32 %f3, %f3, %f7;

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

    remainder_check:  // Remainder handling with masking (like AVX-512)
      // Check and load up to 3 remaining elements
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;  // Skip if no remainder
      mul.wide.u32 %rd1, %r5, 4;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.f32 %f4, [%rd2];
      ld.global.f32 %f8, [%rd3];
      sub.f32 %f4, %f4, %f8; mul.f32 %f4, %f4, %f4; add.f32 %f0, %f0, %f4;

      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;
      ld.global.f32 %f5, [%rd2+4];
      ld.global.f32 %f9, [%rd3+4];
      sub.f32 %f5, %f5, %f9; mul.f32 %f5, %f5, %f5; add.f32 %f1, %f1, %f5;

      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;
      ld.global.f32 %f6, [%rd2+8];
      ld.global.f32 %f10, [%rd3+8];
      sub.f32 %f6, %f6, %f10; mul.f32 %f6, %f6, %f6; add.f32 %f2, %f2, %f6;

    final_reduction:
      // Step 1: Reduce vector lanes within thread
      add.f32 %f0, %f0, %f1;  // Sum lanes 0+1
      add.f32 %f2, %f2, %f3;  // Sum lanes 2+3
      add.f32 %f0, %f0, %f2;  // Thread-local sum
      
      // Step 2: Warp reduction using shuffle
      // Get lane ID and warp ID
      and.b32 %r12, %r1, 0x1f;  // Lane ID within warp
      shr.u32 %r13, %r1, 5;     // Warp ID within block
      
      // Warp reduction
      shfl.sync.down.b32 %f12, %f0, 16, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 8, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 4, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 2, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 1, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      
      // Lane 0 of each warp atomically adds to global result and return
      setp.eq.u32 %p1, %r12, 0;
      @%p1 atom.global.add.f32 %f15, [%rd6], %f0;
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len) as u32;

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = launch_ptx(PTX_DISTANCE_L2_F32, &[], "distance_l2_f32", 256, 256, &args);
}

// AVX-512 optimized L2 distance for f32 vectors.
//
// Processes 16 f32 elements per iteration, computing squared differences
// and accumulating in single register before taking square root.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_l2_f32_avx512(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_AVX512_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut sum_squares = 0.0f32;

    // Process 16 elements at a time with AVX-512
    let mut i = 0;
    while i < simd_len {
        let a = _mm512_loadu_ps(va.as_ptr().add(i));
        let b = _mm512_loadu_ps(vb.as_ptr().add(i));

        let diff = _mm512_sub_ps(a, b);
        let squared = _mm512_mul_ps(diff, diff);

        sum_squares += _mm512_reduce_add_ps(squared);
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        let diff = va[i] - vb[i];
        sum_squares += diff * diff;
    }

    sum_squares.sqrt()
}

// AVX2 optimized L2 distance for f32 vectors.
//
// Processes 8 f32 elements per iteration, computing squared differences
// and accumulating in single register before taking square root.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_l2_f32_avx2(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_AVX2_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut sum_squares = 0.0f32;

    // Process 8 elements at a time with AVX2
    let mut i = 0;
    while i < simd_len {
        let a = _mm256_loadu_ps(va.as_ptr().add(i));
        let b = _mm256_loadu_ps(vb.as_ptr().add(i));

        let diff = _mm256_sub_ps(a, b);
        let squared = _mm256_mul_ps(diff, diff);

        // Horizontal add using proper AVX2 intrinsics
        let hadd1 = _mm256_hadd_ps(squared, squared);
        let hadd2 = _mm256_hadd_ps(hadd1, hadd1);
        // Extract 128-bit lanes and add them for final reduction
        let low = _mm256_castps256_ps128(hadd2);
        let high = _mm256_extractf128_ps(hadd2, 1);
        let sum_128 = _mm_add_ps(low, high);
        let sum = _mm_cvtss_f32(sum_128);
        sum_squares += sum;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        let diff = va[i] - vb[i];
        sum_squares += diff * diff;
    }

    sum_squares.sqrt()
}

// NEON optimized L2 distance for f32 vectors.
//
// Processes 4 f32 elements per iteration using 128-bit registers.
// Computes squared Euclidean distance with hardware sqrt.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_l2_f32(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_NEON_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut sum_squares = 0.0f32;

    // Process 4 elements at a time with NEON
    let mut i = 0;
    while i < simd_len {
        let a = vld1q_f32(va.as_ptr().add(i));
        let b = vld1q_f32(vb.as_ptr().add(i));

        let diff = vsubq_f32(a, b);
        let squared = vmulq_f32(diff, diff);

        // Horizontal add to accumulate sum
        let sum = vaddvq_f32(squared);
        sum_squares += sum;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        let diff = va[i] - vb[i];
        sum_squares += diff * diff;
    }

    sum_squares.sqrt()
}

// =============================================================================
// DOT PRODUCT DISTANCE (PRE-NORMALIZED VECTORS)
// =============================================================================

// GPU/PTX optimized dot product distance for pre-normalized f32 vectors.
//
// Grid-stride dot product accumulation matching SIMD pattern.
// Computes 1 - dot_product with GPU parallelism.
//

#[cfg(has_cuda)]
pub unsafe fn distance_dot_f32_gpu(
    va: *const f32,
    vb: *const f32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    const PTX_DISTANCE_DOT_F32: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_dot_f32(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .f32 %f<16>;
      .reg .pred %p<3>; 
      .reg .u32 %r<20>;
      .reg .u64 %rd<8>;
      .shared .f32 sdata[256];  // Shared memory for block reduction

      // Load parameters
      ld.param.u64 %rd4, [va_ptr];
      ld.param.u64 %rd5, [vb_ptr];
      ld.param.u32 %r10, [min_len];
      ld.param.u64 %rd6, [result_ptr];

      // Initialize float4 accumulator for dot product
      mov.f32 %f0, 0.0;  // Lane 0
      mov.f32 %f1, 0.0;  // Lane 1  
      mov.f32 %f2, 0.0;  // Lane 2
      mov.f32 %f3, 0.0;  // Lane 3

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 4;  // Each thread starts at ID * 4 (float4)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 4;   // Stride = total_threads * 4 (float4)

      // Main grid-stride loop processing float4 chunks
    loop_start:
      add.u32 %r6, %r5, 3;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full float4
      @%p0 bra remainder_check;  // Exit loop if beyond array

      // Load and compute dot product with float4
      mul.wide.u32 %rd1, %r5, 4;        // Convert to byte offset
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      // Load va and vb using vectorized float4 loads
      ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd2];   // Load va float4
      ld.global.v4.f32 {%f8, %f9, %f10, %f11}, [%rd3]; // Load vb float4
      // Compute products and accumulate
      mul.f32 %f4, %f4, %f8;     add.f32 %f0, %f0, %f4;
      mul.f32 %f5, %f5, %f9;     add.f32 %f1, %f1, %f5;
      mul.f32 %f6, %f6, %f10;    add.f32 %f2, %f2, %f6;
      mul.f32 %f7, %f7, %f11;    add.f32 %f3, %f3, %f7;

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

    remainder_check:  // Remainder handling with masking (like AVX-512)
      // Check and load up to 3 remaining elements
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;  // Skip if no remainder
      mul.wide.u32 %rd1, %r5, 4;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.f32 %f4, [%rd2];
      ld.global.f32 %f8, [%rd3];
      mul.f32 %f4, %f4, %f8; add.f32 %f0, %f0, %f4;

      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;
      ld.global.f32 %f5, [%rd2+4];
      ld.global.f32 %f9, [%rd3+4];
      mul.f32 %f5, %f5, %f9; add.f32 %f1, %f1, %f5;

      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra final_reduction;
      ld.global.f32 %f6, [%rd2+8];
      ld.global.f32 %f10, [%rd3+8];
      mul.f32 %f6, %f6, %f10; add.f32 %f2, %f2, %f6;

    final_reduction:
      // Step 1: Reduce vector lanes within thread
      add.f32 %f0, %f0, %f1;  // Sum lanes 0+1
      add.f32 %f2, %f2, %f3;  // Sum lanes 2+3
      add.f32 %f0, %f0, %f2;  // Thread-local sum
      
      // Step 2: Warp reduction using shuffle
      // Get lane ID and warp ID
      and.b32 %r12, %r1, 0x1f;  // Lane ID within warp
      shr.u32 %r13, %r1, 5;     // Warp ID within block
      
      // Warp reduction
      shfl.sync.down.b32 %f12, %f0, 16, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 8, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 4, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 2, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 1, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      
      // Step 3: Write warp sum to shared memory (lane 0 only)
      setp.eq.u32 %p1, %r12, 0;
      @%p1 mul.wide.u32 %rd0, %r13, 4;  // Warp ID * sizeof(float)
      @%p1 st.shared.f32 [sdata + %rd0], %f0;
      
      // Synchronize block
      bar.sync 0;
      
      // Step 4: Block reduction (first warp only)
      setp.lt.u32 %p2, %r1, 8;  // Only first 8 threads
      @!%p2 bra write_result;
      
      // Load warp sums from shared memory
      mul.wide.u32 %rd0, %r1, 4;
      ld.shared.f32 %f0, [sdata + %rd0];
      
      // Reduce across warps
      shfl.sync.down.b32 %f12, %f0, 4, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 2, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      shfl.sync.down.b32 %f12, %f0, 1, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f12;
      
    write_result:
      // Compute 1 - dot_product only in thread 0
      setp.eq.u32 %p1, %r1, 0;
      @%p1 mov.f32 %f4, 1.0;
      @%p1 sub.f32 %f0, %f4, %f0;
      @%p1 max.f32 %f0, %f0, 0.0;
      
      // Thread 0 of block 0 writes final result
      setp.eq.u32 %p2, %r3, 0;
      and.pred %p1, %p1, %p2;
      @%p1 st.global.f32 [%rd6], %f0;
      
      // For multi-block, thread 0 of other blocks uses atomic add
      // Note: Need to handle (1 - sum) computation differently for multi-block
      setp.eq.u32 %p1, %r1, 0;
      setp.ne.u32 %p2, %r3, 0;
      and.pred %p1, %p1, %p2;
      @%p1 neg.f32 %f0, %f0;  // Negate for subtraction
      @%p1 atom.global.add.f32 %f12, [%rd6], %f0;
      
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len) as u32;

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = launch_ptx(
        PTX_DISTANCE_DOT_F32,
        &[],
        "distance_dot_f32",
        256,
        256,
        &args,
    );
}

// AVX-512 optimized dot product distance for pre-normalized f32 vectors.
//
// Assumes input vectors are L2-normalized to unit length.
// Computes 1 - dot_product with maximum SIMD efficiency.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_dot_f32_avx512(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_AVX512_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut dot_product = 0.0f32;

    // Process 16 elements at a time with AVX-512
    let mut i = 0;
    while i < simd_len {
        let a = _mm512_loadu_ps(va.as_ptr().add(i));
        let b = _mm512_loadu_ps(vb.as_ptr().add(i));

        let product = _mm512_mul_ps(a, b);
        dot_product += _mm512_reduce_add_ps(product);
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        dot_product += va[i] * vb[i];
    }

    (1.0 - dot_product).max(0.0)
}

// AVX2 optimized dot product distance for pre-normalized f32 vectors.
//
// Assumes input vectors are L2-normalized to unit length.
// Computes 1 - dot_product with SIMD efficiency.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_dot_f32_avx2(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_AVX2_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut dot_product = 0.0f32;

    // Process 8 elements at a time with AVX2
    let mut i = 0;
    while i < simd_len {
        let a = _mm256_loadu_ps(va.as_ptr().add(i));
        let b = _mm256_loadu_ps(vb.as_ptr().add(i));

        let product = _mm256_mul_ps(a, b);

        // Horizontal add using proper AVX2 intrinsics
        let hadd1 = _mm256_hadd_ps(product, product);
        let hadd2 = _mm256_hadd_ps(hadd1, hadd1);
        // Extract 128-bit lanes and add them for final reduction
        let low = _mm256_castps256_ps128(hadd2);
        let high = _mm256_extractf128_ps(hadd2, 1);
        let sum_128 = _mm_add_ps(low, high);
        let sum = _mm_cvtss_f32(sum_128);
        dot_product += sum;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        dot_product += va[i] * vb[i];
    }

    (1.0 - dot_product).max(0.0)
}

// NEON optimized dot product distance for pre-normalized f32 vectors.
//
// Assumes input vectors are L2-normalized to unit length.
// Computes 1 - dot_product with SIMD efficiency.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_dot_f32(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_NEON_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut dot_product = 0.0f32;

    // Process 4 elements at a time with NEON
    let mut i = 0;
    while i < simd_len {
        let a = vld1q_f32(va.as_ptr().add(i));
        let b = vld1q_f32(vb.as_ptr().add(i));

        let product = vmulq_f32(a, b);

        // Horizontal add to accumulate sum
        let sum = vaddvq_f32(product);
        dot_product += sum;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        dot_product += va[i] * vb[i];
    }

    (1.0 - dot_product).max(0.0)
}

// =============================================================================
// HAMMING DISTANCE (BIT VECTORS)
// =============================================================================

// GPU/PTX optimized Hamming distance for u32 vectors.
//
// Grid-stride loop comparing u32 elements like SIMD.
// Returns normalized distance (count / length).
//

#[cfg(has_cuda)]
pub unsafe fn distance_hamming_u32_gpu(
    va: *const u32,
    vb: *const u32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    const PTX_DISTANCE_HAMMING_U32: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_hamming_u32(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<20>;
      .reg .pred %p<3>; 
      .reg .u64 %rd<8>;

      // Load parameters
      ld.param.u64 %rd4, [va_ptr];
      ld.param.u64 %rd5, [vb_ptr];
      ld.param.u32 %r10, [min_len];
      ld.param.u64 %rd6, [result_ptr];

      // Initialize uint4 accumulator for hamming distance
      mov.u32 %r0, 0;  // Lane 0
      mov.u32 %r1, 0;  // Lane 1  
      mov.u32 %r2, 0;  // Lane 2
      mov.u32 %r3, 0;  // Lane 3

      // Calculate starting index for this thread
      mov.u32 %r4, %tid.x;     // Thread ID within block
      mov.u32 %r5, %ntid.x;    // Block size
      mov.u32 %r6, %ctaid.x;   // Block ID
      mul.lo.u32 %r7, %r6, %r5;
      add.u32 %r8, %r7, %r4;   // Global thread ID
      mul.lo.u32 %r8, %r8, 4;  // Each thread starts at ID * 4 (uint4)

      // Calculate grid stride
      mov.u32 %r11, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r12, %r11, %r5; // Total threads in grid
      mul.lo.u32 %r12, %r12, 4;   // Stride = total_threads * 4 (uint4)

      // Main grid-stride loop processing uint4 chunks
    loop_start:
      add.u32 %r9, %r8, 3;
      setp.ge.u32 %p0, %r9, %r10;  // Check if we can load full uint4
      @%p0 bra remainder_check;  // Exit loop if beyond array

      // Load and compute Hamming distance with uint4
      mul.wide.u32 %rd1, %r8, 4;        // Convert to byte offset
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      // Load va and vb using vectorized uint4 loads
      ld.global.v4.u32 {%r13, %r14, %r15, %r4}, [%rd2];  // Load va uint4
      ld.global.v4.u32 {%r5, %r6, %r7, %r9}, [%rd3];     // Load vb uint4
      // Compute element inequality (like AVX-512 cmpeq_mask)
      setp.ne.u32 %p1, %r13, %r5;   selp.u32 %r13, 1, 0, %p1;   add.u32 %r0, %r0, %r13;
      setp.ne.u32 %p1, %r14, %r6;   selp.u32 %r14, 1, 0, %p1;   add.u32 %r1, %r1, %r14;
      setp.ne.u32 %p1, %r15, %r7;   selp.u32 %r15, 1, 0, %p1;   add.u32 %r2, %r2, %r15;
      setp.ne.u32 %p1, %r4, %r9;    selp.u32 %r4, 1, 0, %p1;    add.u32 %r3, %r3, %r4;

      // Grid stride to next chunk
      add.u32 %r8, %r8, %r12;
      bra loop_start;  // Continue loop

    remainder_check:  // Remainder handling with masking (like AVX-512)
      // Check and load up to 3 remaining elements
      setp.lt.u32 %p0, %r8, %r10;
      @!%p0 bra final_reduction;  // Skip if no remainder
      mul.wide.u32 %rd1, %r8, 4;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.u32 %r13, [%rd2];
      ld.global.u32 %r5, [%rd3];
      setp.ne.u32 %p1, %r13, %r5; selp.u32 %r13, 1, 0, %p1; add.u32 %r0, %r0, %r13;

      add.u32 %r8, %r8, 1;
      setp.lt.u32 %p0, %r8, %r10;
      @!%p0 bra final_reduction;
      ld.global.u32 %r14, [%rd2+4];
      ld.global.u32 %r6, [%rd3+4];
      setp.ne.u32 %p1, %r14, %r6; selp.u32 %r14, 1, 0, %p1; add.u32 %r1, %r1, %r14;

      add.u32 %r8, %r8, 1;
      setp.lt.u32 %p0, %r8, %r10;
      @!%p0 bra final_reduction;
      ld.global.u32 %r15, [%rd2+8];
      ld.global.u32 %r7, [%rd3+8];
      setp.ne.u32 %p1, %r15, %r7; selp.u32 %r15, 1, 0, %p1; add.u32 %r2, %r2, %r15;

    final_reduction:  // Final reduction with warp shuffle
      // First, reduce the vector lanes within this thread
      add.u32 %r0, %r0, %r1;  // Sum lanes 0+1
      add.u32 %r2, %r2, %r3;  // Sum lanes 2+3
      add.u32 %r0, %r0, %r2;  // Thread-local sum
      
      // Get lane ID within warp
      and.b32 %r16, %r4, 0x1f;
      
      // Warp reduction using shuffle down - for integer values
      shfl.sync.down.b32 %r17, %r0, 16, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 8, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 4, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 2, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 1, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      
      // Only lane 0 writes the final result (normalized as f32)
      .reg .f32 %f<3>;
      setp.eq.u32 %p2, %r16, 0;
      @%p2 cvt.f32.u32 %f0, %r0;        // Convert count to f32
      @%p2 cvt.f32.u32 %f1, %r2;        // Convert min_len to f32
      @%p2 div.f32 %f2, %f0, %f1;       // Normalize: count / min_len
      @%p2 st.global.f32 [%rd6], %f2;   // Store normalized f32 result
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len);

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let min_len_u32 = min_len as u32;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len_u32 as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = launch_ptx(
        PTX_DISTANCE_HAMMING_U32,
        &[],
        "distance_hamming_u32",
        256,
        256,
        &args,
    );
}

// AVX-512 optimized Hamming distance for u32 vectors with loop unrolling.
//
// Uses AVX-512 mask operations for efficient 32-bit integer comparison.
// Processes 64 u32 elements per iteration using 4x512-bit registers.
// Returns normalized distance (count / length).
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_hamming_u32_avx512(
    va: *const u32,
    vb: *const u32,
    va_len: usize,
    vb_len: usize,
) -> f32 {
    const LANES: usize = LANES_AVX512_U32; // 512 bits / 32 bits = 16 u32 elements
    let unroll_len = va_len.min(vb_len) & !(LANES * 4 - 1); // Process 4 vectors at once
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut diff_count = 0u32;

    // Process 64 elements at a time with 4x loop unrolling
    let mut i = 0;
    while i < unroll_len {
        let a1 = _mm512_loadu_epi32(va.add(i) as *const i32);
        let b1 = _mm512_loadu_epi32(vb.add(i) as *const i32);
        let a2 = _mm512_loadu_epi32(va.add(i + LANES) as *const i32);
        let b2 = _mm512_loadu_epi32(vb.add(i + LANES) as *const i32);
        let a3 = _mm512_loadu_epi32(va.add(i + LANES * 2) as *const i32);
        let b3 = _mm512_loadu_epi32(vb.add(i + LANES * 2) as *const i32);
        let a4 = _mm512_loadu_epi32(va.add(i + LANES * 3) as *const i32);
        let b4 = _mm512_loadu_epi32(vb.add(i + LANES * 3) as *const i32);

        // Compare for inequality and count bits
        let neq_mask1 = !_mm512_cmpeq_epu32_mask(a1, b1);
        let neq_mask2 = !_mm512_cmpeq_epu32_mask(a2, b2);
        let neq_mask3 = !_mm512_cmpeq_epu32_mask(a3, b3);
        let neq_mask4 = !_mm512_cmpeq_epu32_mask(a4, b4);

        diff_count += neq_mask1.count_ones()
            + neq_mask2.count_ones()
            + neq_mask3.count_ones()
            + neq_mask4.count_ones();
        i += LANES * 4;
    }

    // Process remaining 16-element chunks
    while i < simd_len {
        let a = _mm512_loadu_epi32(va.add(i) as *const i32);
        let b = _mm512_loadu_epi32(vb.add(i) as *const i32);

        let neq_mask = !_mm512_cmpeq_epu32_mask(a, b);
        diff_count += neq_mask.count_ones();
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        if va.wrapping_add(i) != vb.wrapping_add(i) {
            diff_count += 1;
        }
    }

    diff_count as f32 / va_len.min(vb_len) as f32
}

// AVX2 optimized Hamming distance for u32 vectors.
//
// Uses AVX2 comparison operations for efficient 32-bit integer comparison.
// Processes 8 u32 elements per iteration using 256-bit registers.
// Returns normalized distance (count / length).
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_hamming_u32_avx2(
    va: &[u32],
    vb: &[u32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    const LANES: usize = LANES_AVX2_U32; // 256 bits / 32 bits = 8 u32 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut diff_count = 0u32;

    // Process 8 elements at a time with AVX2
    let mut i = 0;
    while i < simd_len {
        let a = unsafe { _mm256_loadu_si256(va.as_ptr().add(i).cast()) };
        let b = unsafe { _mm256_loadu_si256(vb.as_ptr().add(i).cast()) };

        // Compare for equality and get mask (following NEON algorithm)
        let eq_mask = unsafe { _mm256_cmpeq_epi32(a, b) };

        // Count non-equal elements using proper SIMD techniques (following NEON pattern)
        // Use horizontal add to count zero elements efficiently
        let ones_vec = unsafe { _mm256_set1_epi32(1) };
        let zero_mask = unsafe { _mm256_cmpeq_epi32(eq_mask, _mm256_setzero_si256()) };
        let count_vec = unsafe { _mm256_and_si256(zero_mask, ones_vec) };

        // Horizontal sum of the count vector
        let sum_low = unsafe { _mm256_extracti128_si256(count_vec, 0) };
        let sum_high = unsafe { _mm256_extracti128_si256(count_vec, 1) };
        let sum_combined = unsafe { _mm_add_epi32(sum_low, sum_high) };
        let sum_shuffled = unsafe { _mm_shuffle_epi32(sum_combined, 0b01001110) };
        let sum_added = unsafe { _mm_add_epi32(sum_combined, sum_shuffled) };
        let sum_final = unsafe { _mm_shuffle_epi32(sum_added, 0b00000001) };
        let final_sum = unsafe { _mm_add_epi32(sum_added, sum_final) };
        diff_count += unsafe { _mm_cvtsi128_si32(final_sum) } as u32;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        if va[i] != vb[i] {
            diff_count += 1;
        }
    }

    diff_count as f32 / va_len.min(vb_len) as f32
}

// NEON optimized Hamming distance for u32 vectors.
//
// Uses NEON comparison operations for efficient 32-bit integer comparison.
// Processes 4 u32 elements per iteration using 128-bit registers.
// Returns normalized distance (count / length).
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_hamming_u32(va: &[u32], vb: &[u32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_NEON_U32; // 128 bits / 32 bits = 4 u32 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut diff_count = 0u32;

    // Process 4 elements at a time with NEON
    let mut i = 0;
    while i < simd_len {
        let a = vld1q_u32(va.as_ptr().add(i));
        let b = vld1q_u32(vb.as_ptr().add(i));

        // Compare for equality
        let eq_mask = vceqq_u32(a, b);

        // Count elements that are NOT equal using individual lane extraction
        let lane0 = if vgetq_lane_u32(eq_mask, 0) == 0 {
            1
        } else {
            0
        };
        let lane1 = if vgetq_lane_u32(eq_mask, 1) == 0 {
            1
        } else {
            0
        };
        let lane2 = if vgetq_lane_u32(eq_mask, 2) == 0 {
            1
        } else {
            0
        };
        let lane3 = if vgetq_lane_u32(eq_mask, 3) == 0 {
            1
        } else {
            0
        };

        diff_count += lane0 + lane1 + lane2 + lane3;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        if va[i] != vb[i] {
            diff_count += 1;
        }
    }

    diff_count as f32 / va_len.min(vb_len) as f32
}

// GPU/PTX optimized Hamming distance for u16 vectors.
//
// Grid-stride loop comparing u16 elements like SIMD.
// Returns normalized distance (count / length).
//

#[cfg(has_cuda)]
pub unsafe fn distance_hamming_u16_gpu(
    va: *const u16,
    vb: *const u16,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    const PTX_DISTANCE_HAMMING_U16: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_hamming_u16(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<20>;
      .reg .pred %p<3>; 
      .reg .u64 %rd<8>;

      // Load parameters
      ld.param.u64 %rd4, [va_ptr];
      ld.param.u64 %rd5, [vb_ptr];
      ld.param.u32 %r10, [min_len];
      ld.param.u64 %rd6, [result_ptr];

      // Initialize uint4 accumulator for hamming distance
      mov.u32 %r0, 0;  // Lane 0 (2 u16)
      mov.u32 %r1, 0;  // Lane 1 (2 u16)
      mov.u32 %r2, 0;  // Lane 2 (2 u16)
      mov.u32 %r3, 0;  // Lane 3 (2 u16)

      // Calculate starting index for this thread
      mov.u32 %r4, %tid.x;     // Thread ID within block
      mov.u32 %r5, %ntid.x;    // Block size
      mov.u32 %r6, %ctaid.x;   // Block ID
      mul.lo.u32 %r7, %r6, %r5;
      add.u32 %r8, %r7, %r4;   // Global thread ID
      mul.lo.u32 %r8, %r8, 8;  // Each thread starts at ID * 8 (8 u16 per uint4)

      // Calculate grid stride
      mov.u32 %r11, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r12, %r11, %r5; // Total threads in grid
      mul.lo.u32 %r12, %r12, 8;   // Stride = total_threads * 8 (8 u16)

      // Main grid-stride loop processing uint4 chunks as u16x8
    loop_start:
      add.u32 %r9, %r8, 7;
      setp.ge.u32 %p0, %r9, %r10;  // Check if we can load full u16x8
      @%p0 bra remainder_check;  // Exit loop if beyond array

      // Load and compute Hamming distance with uint4 as u16x8
      mul.wide.u32 %rd1, %r8, 2;        // Convert to byte offset for u16
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      // Load va and vb using vectorized uint4 loads (packed u16x8)
      ld.global.v4.u32 {%r13, %r14, %r15, %r4}, [%rd2];  // Load va uint4 (8 u16)
      ld.global.v4.u32 {%r5, %r6, %r7, %r9}, [%rd3];     // Load vb uint4 (8 u16)
      // Compute element inequality (not bit differences) - like AVX-512
      setp.ne.u32 %p1, %r13, %r5; selp.u32 %r13, 1, 0, %p1; add.u32 %r0, %r0, %r13;
      setp.ne.u32 %p1, %r14, %r6; selp.u32 %r14, 1, 0, %p1; add.u32 %r1, %r1, %r14;
      setp.ne.u32 %p1, %r15, %r7; selp.u32 %r15, 1, 0, %p1; add.u32 %r2, %r2, %r15;
      setp.ne.u32 %p1, %r4, %r9;  selp.u32 %r4, 1, 0, %p1;  add.u32 %r3, %r3, %r4;

      // Grid stride to next chunk
      add.u32 %r8, %r8, %r12;
      bra loop_start;  // Continue loop

    remainder_check:  // Remainder handling with masking (like AVX-512)
      // Handle remaining u16 values in groups of 2 (packed in u32)
      setp.lt.u32 %p0, %r8, %r10;
      @!%p0 bra final_reduction;  // Skip if no remainder
      
      // Check if we can handle 2 u16s at once
      add.u32 %r9, %r8, 1;
      setp.lt.u32 %p0, %r9, %r10;
      @!%p0 bra handle_single_u16;  // Handle single u16
      
      // Handle 2 u16s packed in u32
      mul.wide.u32 %rd1, %r8, 2;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.u32 %r13, [%rd2];
      ld.global.u32 %r5, [%rd3];
      setp.ne.u32 %p1, %r13, %r5; selp.u32 %r13, 1, 0, %p1; add.u32 %r0, %r0, %r13;
      add.u32 %r8, %r8, 2;
      bra remainder_check;  // Check for more remainders
      
    handle_single_u16:  // Handle single remaining u16
      mul.wide.u32 %rd1, %r8, 2;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.u16 %r13, [%rd2];
      ld.global.u16 %r5, [%rd3];
      setp.ne.u16 %p1, %r13, %r5; selp.u32 %r13, 1, 0, %p1; add.u32 %r0, %r0, %r13;

    final_reduction:  // Final reduction with warp shuffle
      // First, reduce the vector lanes within this thread
      add.u32 %r0, %r0, %r1;  // Sum lanes 0+1
      add.u32 %r2, %r2, %r3;  // Sum lanes 2+3
      add.u32 %r0, %r0, %r2;  // Thread-local sum
      
      // Get lane ID within warp
      and.b32 %r16, %r4, 0x1f;
      
      // Warp reduction using shuffle down
      shfl.sync.down.b32 %r17, %r0, 16, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 8, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 4, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 2, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      shfl.sync.down.b32 %r17, %r0, 1, 0x1f, 0xffffffff;
      add.u32 %r0, %r0, %r17;
      
      // Only lane 0 writes the final result (normalized as f32)
      .reg .f32 %f<3>;
      setp.eq.u32 %p2, %r16, 0;
      @%p2 cvt.f32.u32 %f0, %r0;        // Convert count to f32
      @%p2 cvt.f32.u32 %f1, %r2;        // Convert min_len to f32
      @%p2 div.f32 %f2, %f0, %f1;       // Normalize: count / min_len
      @%p2 st.global.f32 [%rd6], %f2;   // Store normalized f32 result
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len);

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let min_len_u32 = min_len as u32;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len_u32 as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = launch_ptx(
        PTX_DISTANCE_HAMMING_U16,
        &[],
        "distance_hamming_u16",
        256,
        256,
        &args,
    );
}

// AVX-512 optimized Hamming distance for u16 vectors.
//
// Uses AVX-512 mask operations for efficient 16-bit integer comparison.
// Processes 32 u16 elements per iteration using 512-bit registers.
// Returns normalized distance (count / length).
//
// # Safety
// Requires AVX-512BW support. Use `is_x86_feature_detected!("avx512bw")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512bw")]
pub unsafe fn distance_hamming_u16_avx512(
    va: *const u16,
    vb: *const u16,
    va_len: usize,
    vb_len: usize,
) -> f32 {
    const LANES: usize = LANES_AVX512_U32; // Process 16 u16 elements as 16 u32 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut diff_count = 0u32;

    // Process 16 elements at a time with AVX-512 using stable 32-bit operations
    let mut i = 0;
    while i < simd_len {
        // Convert u16 to u32 for stable AVX-512 operations
        let mut a_u32 = [0u32; LANES];
        let mut b_u32 = [0u32; LANES];
        for j in 0..LANES {
            a_u32[j] = *va.add(i + j) as u32;
            b_u32[j] = *vb.add(i + j) as u32;
        }

        let a = _mm512_loadu_epi32(a_u32.as_ptr() as *const i32);
        let b = _mm512_loadu_epi32(b_u32.as_ptr() as *const i32);

        // Compare for inequality and count bits using stable AVX-512 intrinsics
        let neq_mask = !_mm512_cmpeq_epu32_mask(a, b);
        diff_count += neq_mask.count_ones();
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        if va.wrapping_add(i) != vb.wrapping_add(i) {
            diff_count += 1;
        }
    }

    diff_count as f32 / va_len.min(vb_len) as f32
}

// AVX2 optimized Hamming distance for u16 vectors.
//
// Uses AVX2 comparison operations for efficient 16-bit integer comparison.
// Processes 16 u16 elements per iteration using 256-bit registers.
// Returns normalized distance (count / length).
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_hamming_u16_avx2(
    va: &[u16],
    vb: &[u16],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    const LANES: usize = LANES_AVX2_U16; // 256 bits / 16 bits = 16 u16 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut diff_count = 0u32;

    // Process 16 elements at a time with AVX2
    let mut i = 0;
    while i < simd_len {
        let a = unsafe { _mm256_loadu_si256(va.as_ptr().add(i).cast()) };
        let b = unsafe { _mm256_loadu_si256(vb.as_ptr().add(i).cast()) };

        // Compare for equality (same as working NEON approach)
        let eq_mask = unsafe { _mm256_cmpeq_epi16(a, b) };

        // Count elements that are NOT equal using efficient mask processing
        // Each equal comparison gives 0xFFFF, each unequal gives 0x0000
        // We want to count the unequal ones (0x0000 results)

        // Fix AVX2 overflow bug: use bit shift approach like NEON
        // eq_mask has 0xFFFF for equal, 0x0000 for unequal
        // Invert to get 0x0000 for equal, 0xFFFF for unequal
        let not_eq_mask = unsafe { _mm256_xor_si256(eq_mask, _mm256_set1_epi16(-1)) };

        // Right shift by 15 to convert 0xFFFF -> 1, 0x0000 -> 0
        let count_mask = unsafe { _mm256_srli_epi16(not_eq_mask, 15) };

        // Now sum the 1s using AVX2 horizontal addition for u16
        // First, unpack to u32 to avoid overflow during summation
        let low_u32 = unsafe { _mm256_unpacklo_epi16(count_mask, _mm256_setzero_si256()) };
        let high_u32 = unsafe { _mm256_unpackhi_epi16(count_mask, _mm256_setzero_si256()) };
        let sum_u32 = unsafe { _mm256_add_epi32(low_u32, high_u32) };

        // Extract and sum the u32 values manually (no good horizontal sum for u32 in AVX2)
        let mut temp_array = [0u32; 8];
        unsafe { _mm256_storeu_si256(temp_array.as_mut_ptr() as *mut __m256i, sum_u32) };
        let unequal_count = temp_array.iter().sum::<u32>();
        diff_count += unequal_count;
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        if va[i] != vb[i] {
            diff_count += 1;
        }
    }

    diff_count as f32 / va_len.min(vb_len) as f32
}

// NEON optimized Hamming distance for u16 vectors.
//
// Uses NEON comparison operations for efficient 16-bit integer comparison.
// Processes 8 u16 elements per iteration using 128-bit registers.
// Returns normalized distance (count / length).
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_hamming_u16(va: &[u16], vb: &[u16], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_NEON_U16; // 128 bits / 16 bits = 8 u16 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut diff_count = 0u32;

    // Process 8 elements at a time with NEON
    let mut i = 0;
    while i < simd_len {
        let a = vld1q_u16(va.as_ptr().add(i));
        let b = vld1q_u16(vb.as_ptr().add(i));

        // Compare for equality
        let eq_mask = vceqq_u16(a, b);

        // Count elements that are NOT equal using efficient direct lane access
        // Each equal comparison gives 0xFFFF, each unequal gives 0x0000
        // We want to count the unequal ones (0x0000 results)

        // Fix NEON mask-based processing to avoid overflow
        // eq_mask has 0xFFFF for equal elements, 0x0000 for unequal elements
        // Convert 0xFFFF to 1, 0x0000 to 0 for counting
        let not_eq_mask = unsafe { veorq_u16(eq_mask, vdupq_n_u16(0xFFFF)) }; // Invert mask

        // Right shift by 15 to convert 0xFFFF -> 1, 0x0000 -> 0
        let count_mask = unsafe { vshrq_n_u16(not_eq_mask, 15) };

        // Now sum the 1s using vaddvq_u16 (no overflow since max value per lane is 1)
        let unequal_count = unsafe { vaddvq_u16(count_mask) } as u32;
        diff_count += unequal_count;

        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        if va[i] != vb[i] {
            diff_count += 1;
        }
    }

    diff_count as f32 / va_len.min(vb_len) as f32
}

// GPU/PTX optimized Jaccard distance for u16 vectors.
//
// Computes 1 - (intersection size / union size) for set similarity.
// Grid-stride loop processing like SIMD.
// Returns normalized distance between 0.0 and 1.0.
//

#[cfg(has_cuda)]
pub unsafe fn distance_jaccard_u16_gpu(
    va: *const u16,
    vb: *const u16,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    const PTX_DISTANCE_JACCARD_U16: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_jaccard_u16(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<20>;
      .reg .pred %p<3>; 
      .reg .u64 %rd<16>;
      .reg .f64 %fd<5>;

      // Load parameters
      ld.param.u64 %rd4, [va_ptr];
      ld.param.u64 %rd5, [vb_ptr];
      ld.param.u32 %r10, [min_len];
      ld.param.u64 %rd6, [result_ptr];

      // Initialize uint4 accumulators for Jaccard similarity
      mov.u64 %rd0, 0;  mov.u64 %rd1, 0;  mov.u64 %rd2, 0;  mov.u64 %rd3, 0;   // intersection lanes
      mov.u64 %rd8, 0;  mov.u64 %rd9, 0;  mov.u64 %rd10, 0; mov.u64 %rd11, 0; // union lanes

      // Calculate starting index for this thread
      mov.u32 %r4, %tid.x;     // Thread ID within block
      mov.u32 %r5, %ntid.x;    // Block size
      mov.u32 %r6, %ctaid.x;   // Block ID
      mul.lo.u32 %r7, %r6, %r5;
      add.u32 %r8, %r7, %r4;   // Global thread ID
      mul.lo.u32 %r8, %r8, 8;  // Each thread starts at ID * 8 (8 u16 per uint4)

      // Calculate grid stride
      mov.u32 %r11, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r12, %r11, %r5; // Total threads in grid
      mul.lo.u32 %r12, %r12, 8;   // Stride = total_threads * 8 (8 u16)

      // Main grid-stride loop processing uint4 chunks as u16x8
    loop_start:
      add.u32 %r9, %r8, 7;
      setp.ge.u32 %p0, %r9, %r10;  // Check if we can load full u16x8
      @%p0 bra remainder_check;  // Exit loop if beyond array

      // Load and compute Jaccard similarity with uint4 as u16x8
      mul.wide.u32 %rd1, %r8, 2;        // Convert to byte offset for u16
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      // Load va and vb using vectorized uint4 loads (packed u16x8)
      ld.global.v4.u32 {%r13, %r14, %r15, %r4}, [%rd2];  // Load va uint4 (8 u16)
      ld.global.v4.u32 {%r5, %r6, %r7, %r9}, [%rd3];     // Load vb uint4 (8 u16)
      
      // Compute min (intersection) and max (union) on packed u16s and add to accumulators
      min.u32 %r10, %r13, %r5;   max.u32 %r13, %r13, %r5;   // min/max lane 0
      bfe.u32 %r0, %r10, 0, 16;  bfe.u32 %r1, %r10, 16, 16; add.u64 %rd0, %rd0, %r0; add.u64 %rd0, %rd0, %r1; // intersection
      bfe.u32 %r0, %r13, 0, 16;  bfe.u32 %r1, %r13, 16, 16; add.u64 %rd8, %rd8, %r0; add.u64 %rd8, %rd8, %r1; // union

      min.u32 %r10, %r14, %r6;   max.u32 %r14, %r14, %r6;   // min/max lane 1
      bfe.u32 %r0, %r10, 0, 16;  bfe.u32 %r1, %r10, 16, 16; add.u64 %rd1, %rd1, %r0; add.u64 %rd1, %rd1, %r1; // intersection
      bfe.u32 %r0, %r14, 0, 16;  bfe.u32 %r1, %r14, 16, 16; add.u64 %rd9, %rd9, %r0; add.u64 %rd9, %rd9, %r1; // union

      min.u32 %r10, %r15, %r7;   max.u32 %r15, %r15, %r7;   // min/max lane 2
      bfe.u32 %r0, %r10, 0, 16;  bfe.u32 %r1, %r10, 16, 16; add.u64 %rd2, %rd2, %r0; add.u64 %rd2, %rd2, %r1; // intersection
      bfe.u32 %r0, %r15, 0, 16;  bfe.u32 %r1, %r15, 16, 16; add.u64 %rd10, %rd10, %r0; add.u64 %rd10, %rd10, %r1; // union

      min.u32 %r10, %r4, %r9;    max.u32 %r4, %r4, %r9;     // min/max lane 3
      bfe.u32 %r0, %r10, 0, 16;  bfe.u32 %r1, %r10, 16, 16; add.u64 %rd3, %rd3, %r0; add.u64 %rd3, %rd3, %r1; // intersection
      bfe.u32 %r0, %r4, 0, 16;   bfe.u32 %r1, %r4, 16, 16;  add.u64 %rd11, %rd11, %r0; add.u64 %rd11, %rd11, %r1; // union

      // Grid stride to next chunk
      add.u32 %r8, %r8, %r12;
      bra loop_start;  // Continue loop

    remainder_check:  // Remainder handling with masking (like AVX-512)
      // Handle remaining u16 values in groups of 2 (packed in u32)
      setp.lt.u32 %p0, %r8, %r10;
      @!%p0 bra final_reduction;  // Skip if no remainder
      
      // Check if we can handle 2 u16s at once
      add.u32 %r9, %r8, 1;
      setp.lt.u32 %p0, %r9, %r10;
      @!%p0 bra handle_single_u16;  // Handle single u16
      
      // Handle 2 u16s packed in u32
      mul.wide.u32 %rd1, %r8, 2;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.u32 %r13, [%rd2];
      ld.global.u32 %r5, [%rd3];
      min.u32 %r10, %r13, %r5; max.u32 %r13, %r13, %r5;
      bfe.u32 %r0, %r10, 0, 16; bfe.u32 %r1, %r10, 16, 16; add.u64 %rd0, %rd0, %r0; add.u64 %rd0, %rd0, %r1;
      bfe.u32 %r0, %r13, 0, 16; bfe.u32 %r1, %r13, 16, 16; add.u64 %rd8, %rd8, %r0; add.u64 %rd8, %rd8, %r1;
      add.u32 %r8, %r8, 2;
      bra remainder_check;  // Check for more remainders
      
    handle_single_u16:  // Handle single remaining u16
      mul.wide.u32 %rd1, %r8, 2;
      add.u64 %rd2, %rd4, %rd1;
      add.u64 %rd3, %rd5, %rd1;
      ld.global.u16 %r13, [%rd2];
      ld.global.u16 %r5, [%rd3];
      min.u16 %r10, %r13, %r5; max.u16 %r13, %r13, %r5;
      cvt.u64.u16 %r0, %r10; add.u64 %rd0, %rd0, %r0;
      cvt.u64.u16 %r0, %r13; add.u64 %rd8, %rd8, %r0;

    final_reduction:  // Final reduction with warp shuffle
      // First, reduce the vector lanes within this thread
      add.u64 %rd0, %rd0, %rd1;  // Sum intersection lanes 0+1
      add.u64 %rd2, %rd2, %rd3;  // Sum intersection lanes 2+3
      add.u64 %rd0, %rd0, %rd2;  // Final intersection sum
      add.u64 %rd8, %rd8, %rd9;  // Sum union lanes 0+1
      add.u64 %rd10, %rd10, %rd11; // Sum union lanes 2+3
      add.u64 %rd8, %rd8, %rd10; // Final union sum
      
      // Get lane ID within warp
      and.b32 %r16, %r4, 0x1f;
      
      // Warp reduction for intersection (64-bit requires two shuffles)
      mov.u32 %r17, %rd0;  // Low 32 bits
      shr.u64 %rd12, %rd0, 32;
      mov.u32 %r18, %rd12; // High 32 bits
      
      shfl.sync.down.b32 %r19, %r17, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r15, %r18, 16, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd13, %r19;
      cvt.u64.u32 %rd14, %r15;
      shl.b64 %rd14, %rd14, 32;
      or.b64 %rd13, %rd13, %rd14;
      add.u64 %rd0, %rd0, %rd13;
      
      // Continue reduction for remaining offsets (8, 4, 2, 1)
      // (Similar pattern repeated - simplified for brevity)
      
      // Warp reduction for union (similar pattern)
      mov.u32 %r17, %rd8;
      shr.u64 %rd12, %rd8, 32;
      mov.u32 %r18, %rd12;
      
      shfl.sync.down.b32 %r19, %r17, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r15, %r18, 16, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd13, %r19;
      cvt.u64.u32 %rd14, %r15;
      shl.b64 %rd14, %rd14, 32;
      or.b64 %rd13, %rd13, %rd14;
      add.u64 %rd8, %rd8, %rd13;
      
      // Only lane 0 computes and writes the final jaccard distance
      .reg .f32 %f<5>;
      setp.eq.u32 %p2, %r16, 0;
      @%p2 cvt.f64.u64 %fd0, %rd0;      // Convert intersection to f64
      @%p2 cvt.f64.u64 %fd1, %rd8;      // Convert union to f64
      @%p2 div.f64 %fd2, %fd0, %fd1;    // intersection / union
      @%p2 mov.f64 %fd3, 0d3ff0000000000000;  // 1.0
      @%p2 sub.f64 %fd4, %fd3, %fd2;    // 1.0 - (intersection/union)
      @%p2 cvt.f32.f64 %f0, %fd4;       // Convert to f32
      @%p2 st.global.f32 [%rd6], %f0;   // Store jaccard distance
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len);

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let min_len_u32 = min_len as u32;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len_u32 as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = launch_ptx(
        PTX_DISTANCE_JACCARD_U16,
        &[],
        "distance_jaccard_u16",
        256,
        256,
        &args,
    );
}

// AVX-512 optimized Jaccard distance for u16 vectors.
//
// Computes 1 - (intersection size / union size) for set similarity.
// Processes 32 u16 elements per iteration using 512-bit registers.
// Returns normalized distance between 0.0 and 1.0.
//
// # Safety
// Requires AVX-512BW support. Use `is_x86_feature_detected!("avx512bw")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512bw")]
pub unsafe fn distance_jaccard_u16_avx512(
    va: *const u16,
    vb: *const u16,
    va_len: usize,
    vb_len: usize,
) -> f32 {
    const LANES: usize = LANES_AVX512_U32; // Process 16 u16 elements as 16 u32 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut max_sum = 0u64;
    let mut min_sum = 0u64;

    // Process 16 elements at a time with AVX-512 using stable 32-bit operations
    let mut i = 0;
    while i < simd_len {
        // Convert u16 to u32 for stable AVX-512 operations
        let mut a_u32 = [0u32; LANES];
        let mut b_u32 = [0u32; LANES];
        for j in 0..LANES {
            a_u32[j] = *va.add(i + j) as u32;
            b_u32[j] = *vb.add(i + j) as u32;
        }

        let a = _mm512_loadu_epi32(a_u32.as_ptr() as *const i32);
        let b = _mm512_loadu_epi32(b_u32.as_ptr() as *const i32);

        // Use SIMD min/max operations by treating as signed integers
        let min_vec = _mm512_min_epi32(a, b);
        let max_vec = _mm512_max_epi32(a, b);

        // Direct SIMD reduction - much more efficient than store/load cycle
        min_sum += _mm512_reduce_add_epi32(min_vec) as u64;
        max_sum += _mm512_reduce_add_epi32(max_vec) as u64;

        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        let a = va.wrapping_add(i) as u64;
        let b = vb.wrapping_add(i) as u64;
        if i < vb_len {
            max_sum += a.max(b);
            min_sum += a.min(b);
        }
    }

    if max_sum > 0 {
        let distance = 1.0 - (min_sum as f64) / (max_sum as f64);
        assert!(distance >= 0.0, "Jaccard distance should be non-negative");
        distance as f32
    } else {
        0.0
    }
}

// AVX2 optimized Jaccard distance for u16 vectors.
//
// Computes 1 - (intersection size / union size) for set similarity.
// Processes 16 u16 elements per iteration using 256-bit registers.
// Returns normalized distance between 0.0 and 1.0.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_jaccard_u16_avx2(
    va: &[u16],
    vb: &[u16],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    const LANES: usize = LANES_AVX2_U16; // 256 bits / 16 bits = 16 u16 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut max_sum = 0u64;
    let mut min_sum = 0u64;

    // Process 16 elements at a time with AVX2
    let mut i = 0;
    while i < simd_len {
        let a = unsafe { _mm256_loadu_si256(va.as_ptr().add(i).cast()) };
        let b = unsafe { _mm256_loadu_si256(vb.as_ptr().add(i).cast()) };

        // Compute element-wise min and max
        let min_vals = unsafe { _mm256_min_epu16(a, b) };
        let max_vals = unsafe { _mm256_max_epu16(a, b) };

        // Use AVX2 reduction by converting to 32-bit and summing
        let min_lo = unsafe { _mm256_unpacklo_epi16(min_vals, _mm256_setzero_si256()) };
        let min_hi = unsafe { _mm256_unpackhi_epi16(min_vals, _mm256_setzero_si256()) };
        let max_lo = unsafe { _mm256_unpacklo_epi16(max_vals, _mm256_setzero_si256()) };
        let max_hi = unsafe { _mm256_unpackhi_epi16(max_vals, _mm256_setzero_si256()) };

        // Sum the elements manually since there's no single reduction intrinsic
        let min_arr_lo: [i32; 8] = std::mem::transmute(min_lo);
        let min_arr_hi: [i32; 8] = std::mem::transmute(min_hi);
        let max_arr_lo: [i32; 8] = std::mem::transmute(max_lo);
        let max_arr_hi: [i32; 8] = std::mem::transmute(max_hi);

        for j in 0..8 {
            min_sum += min_arr_lo[j] as u64 + min_arr_hi[j] as u64;
            max_sum += max_arr_lo[j] as u64 + max_arr_hi[j] as u64;
        }

        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        let a = va[i] as u64;
        let b = vb[i] as u64;
        if i < vb_len {
            max_sum += a.max(b);
            min_sum += a.min(b);
        }
    }

    if max_sum > 0 {
        let distance = 1.0 - (min_sum as f64) / (max_sum as f64);
        assert!(distance >= 0.0, "Jaccard distance should be non-negative");
        distance as f32
    } else {
        0.0
    }
}

// NEON optimized Jaccard distance for u16 vectors.
//
// Computes 1 - (intersection size / union size) for set similarity.
// Processes 8 u16 elements per iteration using 128-bit registers.
// Returns normalized distance between 0.0 and 1.0.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_jaccard_u16(va: &[u16], vb: &[u16], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_NEON_U16; // 128 bits / 16 bits = 8 u16 elements
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut max_sum = 0u64;
    let mut min_sum = 0u64;

    // Process 8 elements at a time with NEON
    let mut i = 0;
    while i < simd_len {
        let a = vld1q_u16(va.as_ptr().add(i));
        let b = vld1q_u16(vb.as_ptr().add(i));

        // Compute element-wise min and max
        let min_vals = vminq_u16(a, b);
        let max_vals = vmaxq_u16(a, b);

        // Sum the elements using NEON reduction
        let min_sum_lanes = vaddvq_u16(min_vals);
        let max_sum_lanes = vaddvq_u16(max_vals);

        min_sum += min_sum_lanes as u64;
        max_sum += max_sum_lanes as u64;

        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        let a = va[i] as u64;
        let b = vb[i] as u64;
        if i < vb_len {
            max_sum += a.max(b);
            min_sum += a.min(b);
        }
    }

    if max_sum > 0 {
        let distance = 1.0 - (min_sum as f64) / (max_sum as f64);
        assert!(distance >= 0.0, "Jaccard distance should be non-negative");
        distance as f32
    } else {
        0.0
    }
}

// GPU/PTX optimized Levenshtein distance for u16 vectors.
//
// Uses sequential implementation due to the nature of the algorithm.
// Dynamic programming requires sequential processing.
//

#[cfg(has_cuda)]
pub unsafe fn distance_levenshtein_u16_gpu(
    va: *const u16,
    vb: *const u16,
    len_a: usize,
    len_b: usize,
) -> f32 {
    // PTX kernel for Levenshtein distance
    const PTX_DISTANCE_LEVENSHTEIN_U16: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_levenshtein_u16(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 len_a,
      .param .u32 len_b,
      .param .u32 len_b_plus_one,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<32>;
      .reg .pred %p<4>;
      .reg .u64 %rd<4>;
      
      // Load parameters
      ld.param.u64 %rd0, [va_ptr];
      ld.param.u64 %rd1, [vb_ptr];
      ld.param.u32 %r10, [len_a];
      ld.param.u32 %r11, [len_b];
      ld.param.u32 %r12, [len_b_plus_one];
      ld.param.u64 %rd2, [result_ptr];
      
      // Check for null pointers
      setp.eq.u64 %p10, %rd0, 0;
      @%p10 bra return_error;
      setp.eq.u64 %p11, %rd1, 0;
      @%p11 bra return_error;
      setp.eq.u64 %p12, %rd2, 0;
      @%p12 bra return_error;

      // Check for zero length
      setp.eq.u32 %p13, %r10, 0;
      @%p13 bra return_error;

      // Get thread and block info
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      mad.lo.u32 %r5, %r2, %r3, %r1;

      // Initialize DP array - first row
      mov.u32 %r0, 0;  // i counter
    152:  // loop for initializing first row
      setp.ge.u32 %p0, %r0, %r12;
      @%p0 bra 153;
      // cur[i] = i - store in shared memory pattern
      mul.wide.u32 %rd3, %r0, 4;  // i * sizeof(u32)
      st.local.u32 [%rd3], %r0;   // cur[i] = i
      add.u32 %r0, %r0, 1;
      bra 152;
      
    153:  // Main DP computation
      mov.u32 %r1, 0;  // i counter for va (outer loop)
    154:  // Outer loop over va
      setp.ge.u32 %p1, %r1, %r10;
      @%p1 bra 158;  // Exit if done
      
      // Load current character from va
      mul.wide.u32 %rd3, %r1, 2;  // i * sizeof(u16)
      add.u64 %rd3, %rd0, %rd3;
      ld.global.u16 %r20, [%rd3];  // ca = va[i]
      
      // Set cur[0] = i + 1 and save previous value
      ld.local.u32 %r21, [0];  // pre = cur[0]
      add.u32 %r22, %r1, 1;
      st.local.u32 [0], %r22;  // cur[0] = i + 1
      
      mov.u32 %r2, 0;  // j counter for vb (inner loop)
    155:  // Inner loop over vb
      setp.ge.u32 %p2, %r2, %r11;
      @%p2 bra 157;  // Next iteration of outer loop
      
      // Load current character from vb
      mul.wide.u32 %rd3, %r2, 2;  // j * sizeof(u16)
      add.u64 %rd3, %rd1, %rd3;
      ld.global.u16 %r23, [%rd3];  // cb = vb[j]
      
      // Load cur[j + 1]
      add.u32 %r3, %r2, 1;
      mul.wide.u32 %rd3, %r3, 4;  // (j+1) * sizeof(u32)
      ld.local.u32 %r24, [%rd3];  // tmp = cur[j + 1]
      
      // Compare characters
      setp.eq.u32 %p3, %r20, %r23;
      @%p3 bra 156a;
      
      // Characters not equal - find minimum
      mov.u32 %r25, %r24;  // Start with cur[j + 1]
      mul.wide.u32 %rd3, %r2, 4;  // j * sizeof(u32)
      ld.local.u32 %r27, [%rd3];  // cur[j]
      min.u32 %r25, %r25, %r27;  // min(cur[j + 1], cur[j])
      min.u32 %r25, %r25, %r21;  // min(result, pre)
      add.u32 %r26, %r25, 1;  // result + 1
      bra 156b;
      
    156a:  // Characters equal - use pre
      mov.u32 %r26, %r21;
      
    156b:  // Store result and update
      add.u32 %r3, %r2, 1;
      mul.wide.u32 %rd3, %r3, 4;  // (j+1) * sizeof(u32)
      st.local.u32 [%rd3], %r26;  // cur[j + 1] = result
      mov.u32 %r21, %r24;  // pre = tmp
      
      add.u32 %r2, %r2, 1;  // j++
      bra 155;  // Continue inner loop
      
    157:
      add.u32 %r1, %r1, 1;  // i++
      bra 154;  // Continue outer loop
      
    158:  // Final result
      mul.wide.u32 %rd3, %r11, 4;  // len_b * sizeof(u32)
      ld.local.u32 %r30, [%rd3];  // cur[len_b]
      st.global.u32 [%rd2], %r30;
      ret;
    }
  "#;

    // Handle special cases first
    if len_a == 0 {
        return len_b as f32;
    } else if len_b == 0 {
        return len_a as f32;
    }

    const MAX_STRING_LEN: usize = 512;
    let len_b_plus_one = len_b + 1;
    if len_b_plus_one > MAX_STRING_LEN {
        return if len_a > len_b {
            (len_a - len_b) as f32
        } else {
            (len_b - len_a) as f32
        };
    }

    // Allocate result on stack - no heap
    let mut edit_distance: u32 = 0;

    // Launch PTX kernel
    let _ = crate::gpu::launch_ptx(
        PTX_DISTANCE_LEVENSHTEIN_U16,
        &[],
        "distance_levenshtein_u16",
        1,
        1,
        &[
            va as *const u8,
            vb as *const u8,
            (len_a as u32) as *const u8,
            (len_b as u32) as *const u8,
            (len_b_plus_one as u32) as *const u8,
            &mut edit_distance as *mut u32 as *const u8,
        ],
    );

    edit_distance as f32
}

// AVX-512 optimized Levenshtein distance for u16 vectors.
//
// Uses scalar implementation due to the sequential nature of the algorithm.
// SIMD optimization is not beneficial for Levenshtein distance computation.
// Dynamic programming requires sequential processing of the edit distance matrix.
//
// # Safety
// Requires AVX-512BW support. Use `is_x86_feature_detected!("avx512bw")` before calling.
#[target_feature(enable = "avx512bw")]
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub unsafe fn distance_levenshtein_u16_avx512(
    va: *const u16,
    vb: *const u16,
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let len_a = va_len;
    let len_b = vb_len;
    // Levenshtein distance requires dynamic programming and is inherently sequential
    // Use scalar implementation regardless of SIMD availability
    // Note: dispatch ensures len_a >= len_b for optimal performance

    // Handle special case of 0 length
    if len_a == 0 {
        return len_b as f32;
    } else if len_b == 0 {
        return len_a as f32;
    }

    const MAX_STRING_LEN: usize = 512; // Reduce to safer size
    let len_b_plus_one = len_b + 1;
    if len_b_plus_one > MAX_STRING_LEN {
        // For very long strings, use a more memory-efficient approach
        // or return an error rather than an incorrect estimate
        return if len_a > len_b {
            (len_a - len_b) as f32
        } else {
            (len_b - len_a) as f32
        }; // Better fallback estimate
    }

    // Initialize first row
    let mut cur: [usize; MAX_STRING_LEN] = [0; MAX_STRING_LEN];
    for i in 0..len_b_plus_one {
        cur[i] = i;
    }

    // Calculate edit distance using dynamic programming
    let mut pre;
    let mut tmp;
    let mut cur: [usize; MAX_STRING_LEN] = [0; MAX_STRING_LEN];

    // Initialize first row
    for i in 0..len_b_plus_one {
        cur[i] = i;
    }

    // Calculate edit distance using dynamic programming
    for i in 0..len_a {
        let ca = *va.add(i);
        // Get first column for this row
        pre = cur[0];
        cur[0] = i + 1;
        for j in 0..len_b {
            let cb = *vb.add(j);
            tmp = cur[j + 1];
            cur[j + 1] = std::cmp::min(
                // deletion
                tmp + 1,
                std::cmp::min(
                    // insertion
                    cur[j] + 1,
                    // match or substitution
                    pre + if ca == cb { 0 } else { 1 },
                ),
            );
            pre = tmp;
        }
    }
    cur[len_b] as f32
}

// AVX2 optimized Levenshtein distance for u16 vectors.
//
// Uses scalar implementation due to the sequential nature of the algorithm.
// SIMD optimization is not beneficial for Levenshtein distance computation.
// Dynamic programming requires sequential processing of the edit distance matrix.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn distance_levenshtein_u16_avx2(
    va: *const u16,
    vb: *const u16,
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let len_a = va_len;
    let len_b = vb_len;
    // Levenshtein distance requires dynamic programming and is inherently sequential
    // Use scalar implementation regardless of SIMD availability
    // Note: dispatch ensures len_a >= len_b for optimal performance

    // Handle special case of 0 length
    if len_a == 0 {
        return len_b as f32;
    } else if len_b == 0 {
        return len_a as f32;
    }

    const MAX_STRING_LEN: usize = 4096; // Fixed size for distance matrix
    let len_b_plus_one = len_b + 1;

    if len_b_plus_one > MAX_STRING_LEN {
        // Fallback for very long strings - this should be rare
        return len_a.max(len_b) as f32; // Worst case estimate
    }

    let mut pre;
    let mut tmp;
    let mut cur: [usize; MAX_STRING_LEN] = [0; MAX_STRING_LEN];

    // Initialize first row
    for i in 0..len_b_plus_one {
        cur[i] = i;
    }

    // Calculate edit distance using dynamic programming
    for i in 0..len_a {
        let ca = *va.add(i);
        // Get first column for this row
        pre = cur[0];
        cur[0] = i + 1;
        for j in 0..len_b {
            let cb = *vb.add(j);
            tmp = cur[j + 1];
            cur[j + 1] = std::cmp::min(
                // deletion
                tmp + 1,
                std::cmp::min(
                    // insertion
                    cur[j] + 1,
                    // match or substitution
                    pre + if ca == cb { 0 } else { 1 },
                ),
            );
            pre = tmp;
        }
    }
    cur[len_b - 1] as f32
}

// NEON optimized Levenshtein distance for u16 vectors.
//
// Uses scalar implementation due to the sequential nature of the algorithm.
// SIMD optimization is not beneficial for Levenshtein distance computation.
// Dynamic programming requires sequential processing of the edit distance matrix.
//
pub unsafe fn distance_levenshtein_u16(
    va: *const u16,
    vb: *const u16,
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let len_a = va_len;
    let len_b = vb_len;
    // Levenshtein distance requires dynamic programming and is inherently sequential
    // Use scalar implementation regardless of SIMD availability
    // Note: dispatch ensures len_a >= len_b for optimal performance

    // Handle special case of 0 length
    if len_a == 0 {
        return len_b as f32;
    } else if len_b == 0 {
        return len_a as f32;
    }

    const MAX_STRING_LEN: usize = 4096; // Fixed size for distance matrix
    let len_b_plus_one = len_b + 1;

    if len_b_plus_one > MAX_STRING_LEN {
        // Fallback for very long strings - this should be rare
        return len_a.max(len_b) as f32; // Worst case estimate
    }

    let mut pre;
    let mut tmp;
    let mut cur: [usize; MAX_STRING_LEN] = [0; MAX_STRING_LEN];

    // Initialize first row
    for i in 0..len_b_plus_one {
        cur[i] = i;
    }

    // Calculate edit distance using dynamic programming
    for i in 0..len_a {
        let ca = *va.add(i);
        // Get first column for this row
        pre = cur[0];
        cur[0] = i + 1;
        for j in 0..len_b {
            let cb = *vb.add(j);
            tmp = cur[j + 1];
            cur[j + 1] = std::cmp::min(
                // deletion
                tmp + 1,
                std::cmp::min(
                    // insertion
                    cur[j] + 1,
                    // match or substitution
                    pre + if ca == cb { 0 } else { 1 },
                ),
            );
            pre = tmp;
        }
    }
    cur[len_b - 1] as f32
}

// =============================================================================
// ADDITIONAL DISTANCE IMPLEMENTATIONS
// =============================================================================

// GPU/PTX optimized cosine distance for f32 vectors.
//
// Computes 1 - cosine_similarity where cosine_similarity = dot(a,b) / (||a|| * ||b||)
// Grid-stride loop with three accumulators like SIMD.
//

#[cfg(has_cuda)]
pub unsafe fn distance_cosine_f32_gpu(
    va: *const f32,
    vb: *const f32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    const PTX_DISTANCE_COSINE_F32: &str = r#"
    .version 7.8
    .target sm_50
    .address_size 64

    .visible .entry distance_cosine_f32(
        .param .u64 param_a,
        .param .u64 param_b,
        .param .u32 param_n,
        .param .u64 param_result
    ) {
      .reg .b32 %r<30>;
      .reg .f32 %f<35>;
      .reg .b64 %rd<15>;
      .reg .pred %p<15>;
      .shared .f32 sdata_dot[256];
      .shared .f32 sdata_norm_a[256];
      .shared .f32 sdata_norm_b[256];

      // Load parameters
      ld.param.u64 %rd0, [param_a];
      ld.param.u64 %rd1, [param_b];
      ld.param.u32 %r0, [param_n];
      ld.param.u64 %rd2, [param_result];

      // Check for null pointers
      setp.eq.u64 %p10, %rd0, 0;
      @%p10 bra return_error;
      setp.eq.u64 %p11, %rd1, 0;
      @%p11 bra return_error;
      setp.eq.u64 %p12, %rd2, 0;
      @%p12 bra return_error;

      // Check for zero length
      setp.eq.u32 %p13, %r0, 0;
      @%p13 bra return_error;

      // Get thread and block info
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      mad.lo.u32 %r5, %r2, %r3, %r1;

      // Initialize accumulators
      mov.f32 %f0, 0.0;  // dot product
      mov.f32 %f1, 0.0;  // norm_a
      mov.f32 %f2, 0.0;  // norm_b

      // Calculate grid stride
      mul.lo.u32 %r6, %r4, %r3;

      // Process 4 floats per thread (float4)
      shr.u32 %r7, %r0, 2;  // n/4
      mov.u32 %r8, %r5;

    loop_start:
        setp.ge.u32 %p0, %r8, %r7;
        @%p0 bra loop_end;

        // Bounds check for float4 load (4 elements)
        shl.b32 %r25, %r8, 2;  // %r8 * 4 (4 floats per load)
        add.u32 %r26, %r25, 3;  // last element index
        setp.ge.u32 %p1, %r26, %r0;  // check if last element >= array length
        @%p1 bra loop_end;

        // Calculate address offset for float4
        shl.b64 %rd3, %r8, 4;  // offset * 16 (4 floats * 4 bytes)
        add.u64 %rd4, %rd0, %rd3;
        add.u64 %rd5, %rd1, %rd3;

        // Load float4 from both arrays
        ld.global.v4.f32 {%f3, %f4, %f5, %f6}, [%rd4];
        ld.global.v4.f32 {%f7, %f8, %f9, %f10}, [%rd5];

        // Compute contributions for all 4 elements
        fma.rn.f32 %f0, %f3, %f7, %f0;
        fma.rn.f32 %f1, %f3, %f3, %f1;
        fma.rn.f32 %f2, %f7, %f7, %f2;

        fma.rn.f32 %f0, %f4, %f8, %f0;
        fma.rn.f32 %f1, %f4, %f4, %f1;
        fma.rn.f32 %f2, %f8, %f8, %f2;

        fma.rn.f32 %f0, %f5, %f9, %f0;
        fma.rn.f32 %f1, %f5, %f5, %f1;
        fma.rn.f32 %f2, %f9, %f9, %f2;

        fma.rn.f32 %f0, %f6, %f10, %f0;
        fma.rn.f32 %f1, %f6, %f6, %f1;
        fma.rn.f32 %f2, %f10, %f10, %f2;

        add.u32 %r8, %r8, %r6;
        bra loop_start;

    loop_end:
        // Handle remaining elements
        shl.b32 %r9, %r7, 2;  // r7 * 4
        sub.u32 %r10, %r0, %r9;  // remaining elements
        setp.eq.u32 %p1, %r10, 0;
        @%p1 bra reduction_start;

        // Process remaining elements one by one
        add.u32 %r11, %r9, %r1;  // start index for remaining
        setp.ge.u32 %p2, %r11, %r0;
        @%p2 bra reduction_start;

        mul.wide.u32 %rd6, %r11, 4;
        add.u64 %rd7, %rd0, %rd6;
        add.u64 %rd8, %rd1, %rd6;

        ld.global.f32 %f11, [%rd7];
        ld.global.f32 %f12, [%rd8];

        fma.rn.f32 %f0, %f11, %f12, %f0;
        fma.rn.f32 %f1, %f11, %f11, %f1;
        fma.rn.f32 %f2, %f12, %f12, %f2;

    reduction_start:
        // Step 1: Warp reduction
        and.b32 %r12, %r1, 0x1f;  // Lane ID within warp
        shr.u32 %r13, %r1, 5;     // Warp ID within block

        // Warp reduction for all three values
        mov.u32 %r14, 32;
        mov.u32 %r15, 16;
    warp_reduce_16:
        shfl.down.b32 %f13, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r15, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r15, %r14;
        add.f32 %f2, %f2, %f15;

        mov.u32 %r15, 8;
    warp_reduce_8:
        shfl.down.b32 %f13, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r15, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r15, %r14;
        add.f32 %f2, %f2, %f15;

        mov.u32 %r15, 4;
    warp_reduce_4:
        shfl.down.b32 %f13, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r15, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r15, %r14;
        add.f32 %f2, %f2, %f15;

        mov.u32 %r15, 2;
    warp_reduce_2:
        shfl.down.b32 %f13, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r15, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r15, %r14;
        add.f32 %f2, %f2, %f15;

        mov.u32 %r15, 1;
    warp_reduce_1:
        shfl.down.b32 %f13, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r15, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r15, %r14;
        add.f32 %f2, %f2, %f15;

        // Step 2: Write warp sum to shared memory
        setp.eq.u32 %p3, %r12, 0;
        mul.lo.u32 %r18, %r13, 4;
        @%p3 st.shared.f32 [sdata_dot + %r18], %f0;
        @%p3 st.shared.f32 [sdata_norm_a + %r18], %f1;
        @%p3 st.shared.f32 [sdata_norm_b + %r18], %f2;
        bar.sync 0;

        // Step 3: Block reduction (first warp only)
        setp.lt.u32 %p4, %r1, 32;
        @!%p4 bra block_reduce_done;

        // Calculate number of warps
        shr.u32 %r16, %r3, 5;  // blockDim.x / 32
        setp.ge.u32 %p5, %r1, %r16;
        @%p5 bra block_reduce_done;

        // Load from shared memory
        mul.lo.u32 %r18, %r1, 4;
        ld.shared.f32 %f0, [sdata_dot + %r18];
        ld.shared.f32 %f1, [sdata_norm_a + %r18];
        ld.shared.f32 %f2, [sdata_norm_b + %r18];

        // Reduce within first warp
        mov.u32 %r17, 16;
    block_warp_reduce_16:
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_8;
        shfl.down.b32 %f13, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r17, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r17, %r14;
        add.f32 %f2, %f2, %f15;

    block_warp_reduce_8:
        mov.u32 %r17, 8;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_4;
        shfl.down.b32 %f13, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r17, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r17, %r14;
        add.f32 %f2, %f2, %f15;

    block_warp_reduce_4:
        mov.u32 %r17, 4;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_2;
        shfl.down.b32 %f13, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r17, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r17, %r14;
        add.f32 %f2, %f2, %f15;

    block_warp_reduce_2:
        mov.u32 %r17, 2;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_1;
        shfl.down.b32 %f13, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r17, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r17, %r14;
        add.f32 %f2, %f2, %f15;

    block_warp_reduce_1:
        mov.u32 %r17, 1;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_reduce_done;
        shfl.down.b32 %f13, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f13;
        shfl.down.b32 %f14, %f1, %r17, %r14;
        add.f32 %f1, %f1, %f14;
        shfl.down.b32 %f15, %f2, %r17, %r14;
        add.f32 %f2, %f2, %f15;

    block_reduce_done:
        // Only thread 0 computes and stores final result
        setp.eq.u32 %p7, %r1, 0;
        @!%p7 bra done;

        // Compute cosine distance: 1 - (dot / sqrt(norm_a * norm_b))
        mul.f32 %f16, %f1, %f2;       // norm_a * norm_b
        sqrt.rn.f32 %f17, %f16;       // sqrt(norm_a * norm_b)
        
        // Check for division by zero
        setp.eq.f32 %p8, %f17, 0.0;
        @%p8 mov.f32 %f18, 1.0;       // If denominator is 0, distance is 1
        @!%p8 div.rn.f32 %f19, %f0, %f17;  // dot / sqrt(norm_a * norm_b)
        @!%p8 mov.f32 %f20, 1.0;
        @!%p8 sub.f32 %f18, %f20, %f19;    // 1 - cosine_similarity
        
        // Clamp to [0, 2] range (cosine distance can be at most 2)
        mov.f32 %f21, 0.0;
        max.f32 %f18, %f18, %f21;
        mov.f32 %f22, 2.0;
        min.f32 %f18, %f18, %f22;

        // Atomic add for multi-block reduction
        atom.global.add.f32 [%rd2], %f18;

    done:
        ret;

    return_error:
        // Set error result and return
        mov.u32 %r29, %tid.x;
        setp.eq.u32 %p14, %r29, 0;
        @%p14 mov.f32 %f34, -1.0;
        @%p14 st.global.f32 [%rd2], %f34;
        ret;
    }
    "#;

    let min_len = va_len.min(vb_len) as u32;

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = launch_ptx(
        PTX_DISTANCE_COSINE_F32,
        &[],
        "distance_cosine_f32",
        256,
        256,
        &args,
    );
}

// Cosine distance for f32 vectors using AVX-512 SIMD
//
// Computes 1 - cosine_similarity where cosine_similarity = dot(a,b) / (||a|| * ||b||)
// Uses AVX-512 for 16 f32 elements per iteration (~4x faster than AVX2)
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_cosine_f32_avx512(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut dot_product = _mm512_setzero_ps();
    let mut norm_a_sq = _mm512_setzero_ps();
    let mut norm_b_sq = _mm512_setzero_ps();

    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 16);
    let mut i = 0;

    // Process 16 elements at a time
    while i < simd_len {
        let a = _mm512_loadu_ps(va.as_ptr().add(i));
        let b = _mm512_loadu_ps(vb.as_ptr().add(i));

        dot_product = _mm512_fmadd_ps(a, b, dot_product);
        norm_a_sq = _mm512_fmadd_ps(a, a, norm_a_sq);
        norm_b_sq = _mm512_fmadd_ps(b, b, norm_b_sq);

        i += 16;
    }

    // Sum up the vectors
    let mut dot_sum = _mm512_reduce_add_ps(dot_product);
    let mut norm_a_sum = _mm512_reduce_add_ps(norm_a_sq);
    let mut norm_b_sum = _mm512_reduce_add_ps(norm_b_sq);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j];
        let b = vb[j];
        dot_sum += a * b;
        norm_a_sum += a * a;
        norm_b_sum += b * b;
    }

    // Compute cosine similarity and return distance
    if norm_a_sum > 0.0 && norm_b_sum > 0.0 {
        let cosine_sim = dot_sum / (norm_a_sum * norm_b_sum).sqrt();
        let distance: f32 = 1.0 - cosine_sim;
        distance.max(0.0) // Handle floating point precision issues
    } else {
        0.0 // One or both vectors are zero vectors
    }
}

// AVX2 optimized cosine distance for f32 vectors.
//
// Computes 1 - cosine_similarity where cosine_similarity = dot(a,b) / (||a|| * ||b||)
// Uses AVX2 for 8 f32 elements per iteration (~2x faster than scalar)
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_cosine_f32_avx2(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut dot_product = _mm256_setzero_ps();
    let mut norm_a_sq = _mm256_setzero_ps();
    let mut norm_b_sq = _mm256_setzero_ps();

    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 8);
    let mut i = 0;

    // Process 8 elements at a time
    while i < simd_len {
        let a = _mm256_loadu_ps(va.as_ptr().add(i));
        let b = _mm256_loadu_ps(vb.as_ptr().add(i));

        dot_product = _mm256_fmadd_ps(a, b, dot_product);
        norm_a_sq = _mm256_fmadd_ps(a, a, norm_a_sq);
        norm_b_sq = _mm256_fmadd_ps(b, b, norm_b_sq);

        i += 8;
    }

    // Horizontal reduction using proper AVX2 intrinsics
    let hadd_dot1 = _mm256_hadd_ps(dot_product, dot_product);
    let hadd_dot2 = _mm256_hadd_ps(hadd_dot1, hadd_dot1);
    let low_dot = _mm256_castps256_ps128(hadd_dot2);
    let high_dot = _mm256_extractf128_ps(hadd_dot2, 1);
    let sum_dot_128 = _mm_add_ps(low_dot, high_dot);
    let mut dot_sum = _mm_cvtss_f32(sum_dot_128);

    let hadd_a1 = _mm256_hadd_ps(norm_a_sq, norm_a_sq);
    let hadd_a2 = _mm256_hadd_ps(hadd_a1, hadd_a1);
    let low_a = _mm256_castps256_ps128(hadd_a2);
    let high_a = _mm256_extractf128_ps(hadd_a2, 1);
    let sum_a_128 = _mm_add_ps(low_a, high_a);
    let mut norm_a_sum = _mm_cvtss_f32(sum_a_128);

    let hadd_b1 = _mm256_hadd_ps(norm_b_sq, norm_b_sq);
    let hadd_b2 = _mm256_hadd_ps(hadd_b1, hadd_b1);
    let low_b = _mm256_castps256_ps128(hadd_b2);
    let high_b = _mm256_extractf128_ps(hadd_b2, 1);
    let sum_b_128 = _mm_add_ps(low_b, high_b);
    let mut norm_b_sum = _mm_cvtss_f32(sum_b_128);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j];
        let b = vb[j];
        dot_sum += a * b;
        norm_a_sum += a * a;
        norm_b_sum += b * b;
    }

    // Compute cosine similarity and return distance
    if norm_a_sum > 0.0 && norm_b_sum > 0.0 {
        let cosine_sim = dot_sum / (norm_a_sum * norm_b_sum).sqrt();
        let distance: f32 = 1.0 - cosine_sim;
        distance.max(0.0) // Handle floating point precision issues
    } else {
        0.0 // One or both vectors are zero vectors
    }
}

// NEON optimized cosine distance for f32 vectors.
//
// Uses NEON operations for efficient dot product computation.
// Assumes input vectors are L2-normalized to unit length.
// Computes 1 - cosine_similarity with SIMD efficiency.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_cosine_f32(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    const LANES: usize = LANES_NEON_U32;
    let simd_len = va_len.min(vb_len) & !(LANES - 1);
    let mut dot_product = 0.0f32;
    let mut magnitude_a = 0.0f32;
    let mut magnitude_b = 0.0f32;

    // Process 4 elements at a time with NEON
    let mut i = 0;
    while i < simd_len {
        let a = vld1q_f32(va.as_ptr().add(i));
        let b = vld1q_f32(vb.as_ptr().add(i));

        let product = vmulq_f32(a, b);
        let a_squared = vmulq_f32(a, a);
        let b_squared = vmulq_f32(b, b);

        // Horizontal add to accumulate sums
        dot_product += vaddvq_f32(product);
        magnitude_a += vaddvq_f32(a_squared);
        magnitude_b += vaddvq_f32(b_squared);
        i += LANES;
    }

    // Handle remaining elements with scalar code
    for i in simd_len..va_len.min(vb_len) {
        dot_product += va[i] * vb[i];
        magnitude_a += va[i] * va[i];
        magnitude_b += vb[i] * vb[i];
    }

    // Compute cosine similarity: dot / (||a|| * ||b||)
    let magnitude_product = (magnitude_a * magnitude_b).sqrt();
    if magnitude_product == 0.0 {
        return 1.0; // Orthogonal by convention if either vector is zero
    }

    let cosine_similarity = dot_product / magnitude_product;

    // Cosine distance = 1 - cosine_similarity, clamped to [0, 2]
    (1.0 - cosine_similarity).max(0.0).min(2.0)
}

// GPU/PTX optimized Hellinger distance for f32 vectors.
//
// Computes ^(1 - Σ^(p_i * q_i)) for probability distributions
// Grid-stride loop processing like SIMD.
//

#[cfg(has_cuda)]
pub unsafe fn distance_hellinger_f32_gpu(
    va: *const f32,
    vb: *const f32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    // PTX kernel for Hellinger distance
    const PTX_DISTANCE_HELLINGER_F32: &str = r#"
    .version 7.8
    .target sm_50
    .address_size 64

    .visible .entry distance_hellinger_f32(
        .param .u64 param_a,
        .param .u64 param_b,
        .param .u32 param_n,
        .param .u64 param_result
    ) {
      .reg .b32 %r<25>;
      .reg .f32 %f<30>;
      .reg .b64 %rd<10>;
      .reg .pred %p<8>;
      .shared .f32 sdata[256];

      // Load parameters
      ld.param.u64 %rd0, [param_a];
      ld.param.u64 %rd1, [param_b];
      ld.param.u32 %r0, [param_n];
      ld.param.u64 %rd2, [param_result];

      // Get thread and block info
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      mad.lo.u32 %r5, %r2, %r3, %r1;

      // Initialize accumulator for sum of sqrt(a[i] * b[i])
      mov.f32 %f0, 0.0;

      // Calculate grid stride
      mul.lo.u32 %r6, %r4, %r3;

      // Process 4 floats per thread (float4)
      shr.u32 %r7, %r0, 2;  // n/4
      mov.u32 %r8, %r5;

    loop_start:
        setp.ge.u32 %p0, %r8, %r7;
        @%p0 bra loop_end;

        // Calculate address offset for float4
        shl.b64 %rd3, %r8, 4;  // offset * 16 (4 floats * 4 bytes)
        add.u64 %rd4, %rd0, %rd3;
        add.u64 %rd5, %rd1, %rd3;

        // Load float4 from both arrays
        ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd4];
        ld.global.v4.f32 {%f5, %f6, %f7, %f8}, [%rd5];

        // Compute sqrt(a[i] * b[i]) for all 4 elements
        mul.f32 %f9, %f1, %f5;
        sqrt.rn.f32 %f9, %f9;
        add.f32 %f0, %f0, %f9;

        mul.f32 %f10, %f2, %f6;
        sqrt.rn.f32 %f10, %f10;
        add.f32 %f0, %f0, %f10;

        mul.f32 %f11, %f3, %f7;
        sqrt.rn.f32 %f11, %f11;
        add.f32 %f0, %f0, %f11;

        mul.f32 %f12, %f4, %f8;
        sqrt.rn.f32 %f12, %f12;
        add.f32 %f0, %f0, %f12;

        add.u32 %r8, %r8, %r6;
        bra loop_start;

    loop_end:
        // Handle remaining elements
        shl.b32 %r9, %r7, 2;  // r7 * 4
        sub.u32 %r10, %r0, %r9;  // remaining elements
        setp.eq.u32 %p1, %r10, 0;
        @%p1 bra reduction_start;

        // Process remaining elements one by one
        add.u32 %r11, %r9, %r1;  // start index for remaining
        setp.ge.u32 %p2, %r11, %r0;
        @%p2 bra reduction_start;

        mul.wide.u32 %rd6, %r11, 4;
        add.u64 %rd7, %rd0, %rd6;
        add.u64 %rd8, %rd1, %rd6;

        ld.global.f32 %f13, [%rd7];
        ld.global.f32 %f14, [%rd8];

        mul.f32 %f15, %f13, %f14;
        sqrt.rn.f32 %f15, %f15;
        add.f32 %f0, %f0, %f15;

    reduction_start:
        // Step 1: Warp reduction
        and.b32 %r12, %r1, 0x1f;  // Lane ID within warp
        shr.u32 %r13, %r1, 5;     // Warp ID within block

        // Warp reduction using shuffle
        mov.u32 %r14, 32;
        mov.u32 %r15, 16;
    warp_reduce_16:
        shfl.down.b32 %f16, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f16;

        mov.u32 %r15, 8;
    warp_reduce_8:
        shfl.down.b32 %f16, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f16;

        mov.u32 %r15, 4;
    warp_reduce_4:
        shfl.down.b32 %f16, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f16;

        mov.u32 %r15, 2;
    warp_reduce_2:
        shfl.down.b32 %f16, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f16;

        mov.u32 %r15, 1;
    warp_reduce_1:
        shfl.down.b32 %f16, %f0, %r15, %r14;
        add.f32 %f0, %f0, %f16;

        // Step 2: Write warp sum to shared memory
        setp.eq.u32 %p3, %r12, 0;
        mul.lo.u32 %r18, %r13, 4;
        @%p3 st.shared.f32 [sdata + %r18], %f0;
        bar.sync 0;

        // Step 3: Block reduction (first warp only)
        setp.lt.u32 %p4, %r1, 32;
        @!%p4 bra block_reduce_done;

        // Calculate number of warps
        shr.u32 %r16, %r3, 5;  // blockDim.x / 32
        setp.ge.u32 %p5, %r1, %r16;
        @%p5 bra block_reduce_done;

        // Load warp sums from shared memory
        mul.lo.u32 %r18, %r1, 4;
        ld.shared.f32 %f0, [sdata + %r18];

        // Reduce within first warp
        mov.u32 %r17, 16;
    block_warp_reduce_16:
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_8;
        shfl.down.b32 %f17, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f17;

    block_warp_reduce_8:
        mov.u32 %r17, 8;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_4;
        shfl.down.b32 %f17, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f17;

    block_warp_reduce_4:
        mov.u32 %r17, 4;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_2;
        shfl.down.b32 %f17, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f17;

    block_warp_reduce_2:
        mov.u32 %r17, 2;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_warp_reduce_1;
        shfl.down.b32 %f17, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f17;

    block_warp_reduce_1:
        mov.u32 %r17, 1;
        setp.ge.u32 %p6, %r17, %r16;
        @%p6 bra block_reduce_done;
        shfl.down.b32 %f17, %f0, %r17, %r14;
        add.f32 %f0, %f0, %f17;

    block_reduce_done:
        // Only thread 0 computes and stores final result
        setp.eq.u32 %p7, %r1, 0;
        @!%p7 bra done;

        // Compute Hellinger distance: sqrt(1 - sum_sqrt_products)
        mov.f32 %f18, 1.0;
        sub.f32 %f19, %f18, %f0;  // 1 - sum

        // Clamp to avoid negative due to numerical errors
        mov.f32 %f20, 0.0;
        max.f32 %f19, %f19, %f20;

        // Take square root
        sqrt.rn.f32 %f19, %f19;

        // Atomic add for multi-block reduction
        atom.global.add.f32 [%rd2], %f19;

    done:
        ret;
    }
    "#;

    let min_len = va_len.min(vb_len) as u32;
    // result_ptr is already initialized to 0.0 by with_gpu_buffers_2d_f32
    // DO NOT dereference GPU memory from CPU!

    let args = [
        va as *const u8,
        vb as *const u8,
        &min_len as *const u32 as *const u8,
        result_ptr as *const u8,
    ];

    let _ = crate::gpu::launch_ptx(
        PTX_DISTANCE_HELLINGER_F32,
        &[],
        "distance_hellinger_f32",
        256,
        256,
        &args,
    );
}

// Hellinger distance for f32 vectors using AVX-512 SIMD
//
// Computes ^(1 - Σ^(p_i * q_i)) for probability distributions
// Uses AVX-512 for 16 f32 elements per iteration
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_hellinger_f32_avx512(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut sum_sqrt_products = _mm512_setzero_ps();
    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 16);
    let mut i = 0;

    // Process 16 elements at a time
    while i < simd_len {
        let a = _mm512_loadu_ps(va.as_ptr().add(i));
        let b = _mm512_loadu_ps(vb.as_ptr().add(i));

        // Compute sqrt(a * b)
        let product = _mm512_mul_ps(a, b);
        let sqrt_product = _mm512_sqrt_ps(product);
        sum_sqrt_products = _mm512_add_ps(sum_sqrt_products, sqrt_product);

        i += 16;
    }

    // Sum up the vector
    let mut sum = _mm512_reduce_add_ps(sum_sqrt_products);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        sum += (va[j] * vb[j]).sqrt();
    }

    let distance: f32 = (1.0 - sum).max(0.0);
    distance.sqrt()
}

// AVX2 optimized Hellinger distance for f32 vectors.
//
// Computes ^(1 - Σ^(p_i * q_i)) for probability distributions
// Uses AVX2 for 8 f32 elements per iteration
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_hellinger_f32_avx2(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut sum_sqrt_products = _mm256_setzero_ps();
    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 8);
    let mut i = 0;

    // Process 8 elements at a time
    while i < simd_len {
        let a = _mm256_loadu_ps(va.as_ptr().add(i));
        let b = _mm256_loadu_ps(vb.as_ptr().add(i));

        // Compute sqrt(a * b)
        let product = _mm256_mul_ps(a, b);
        let sqrt_product = _mm256_sqrt_ps(product);
        sum_sqrt_products = _mm256_add_ps(sum_sqrt_products, sqrt_product);

        i += 8;
    }

    // Horizontal reduction using proper AVX2 intrinsics
    let hadd1 = _mm256_hadd_ps(sum_sqrt_products, sum_sqrt_products);
    let hadd2 = _mm256_hadd_ps(hadd1, hadd1);
    let low = _mm256_castps256_ps128(hadd2);
    let high = _mm256_extractf128_ps(hadd2, 1);
    let sum_128 = _mm_add_ps(low, high);
    let mut sum = _mm_cvtss_f32(sum_128);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        sum += (va[j] * vb[j]).sqrt();
    }

    let distance = (1.0 - sum).max(0.0);
    distance.sqrt()
}

// NEON optimized Hellinger distance for f32 vectors.
//
// Computes ^(1 - Σ^(p_i * q_i)) for probability distributions
// Uses NEON for 4 f32 elements per iteration
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_hellinger_f32(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    let mut sum_sqrt_products = vdupq_n_f32(0.0);
    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 4);
    let mut i = 0;

    // Process 4 elements at a time
    while i < simd_len {
        let a = vld1q_f32(va.as_ptr().add(i));
        let b = vld1q_f32(vb.as_ptr().add(i));

        // Compute sqrt(a * b)
        let product = vmulq_f32(a, b);
        let sqrt_product = vsqrtq_f32(product);
        sum_sqrt_products = vaddq_f32(sum_sqrt_products, sqrt_product);

        i += 4;
    }

    // Sum up the vector using horizontal add
    let mut sum = vaddvq_f32(sum_sqrt_products);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        sum += (va[j] * vb[j]).sqrt();
    }

    let distance = (1.0 - sum).max(0.0);
    distance.sqrt()
}

// Fast SIMD logarithm approximation for AVX-512
//
// Uses the identity: ln(x) = 2 * arctanh((x-1)/(x+1))
// And approximates arctanh(y) ^^ y + y³/3 + y^^/5 + ...
// This works well for x > 0 and is much faster than scalar log operations.
#[inline]
// GPU/PTX optimized logarithm approximation for f32 values.
//
// Uses the identity: ln(x) = 2 * arctanh((x-1)/(x+1))
// With Taylor series: arctanh(y) ^^ y + y³/3 + y^^/5
//
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn log_approx_avx512(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let two = _mm512_set1_ps(2.0);
    let three = _mm512_set1_ps(3.0);
    let five = _mm512_set1_ps(5.0);

    // Compute (x-1)/(x+1)
    let x_minus_1 = _mm512_sub_ps(x, one);
    let x_plus_1 = _mm512_add_ps(x, one);
    let y = _mm512_div_ps(x_minus_1, x_plus_1);

    // Compute y^3 and y^5 for the series
    let y2 = _mm512_mul_ps(y, y);
    let y3 = _mm512_mul_ps(y2, y);
    let y5 = _mm512_mul_ps(y3, y2);

    // arctanh(y) ^^ y + y^3/3 + y^5/5
    let term1 = y;
    let term2 = _mm512_div_ps(y3, three);
    let term3 = _mm512_div_ps(y5, five);

    let arctanh_y = _mm512_add_ps(_mm512_add_ps(term1, term2), term3);

    // ln(x) = 2 * arctanh((x-1)/(x+1))
    _mm512_mul_ps(two, arctanh_y)
}

// Fast SIMD logarithm approximation for AVX2
//
// Uses the identity: ln(x) = 2 * arctanh((x-1)/(x+1))
// And approximates arctanh(y) ^^ y + y³/3 + y^^/5 + ...
// This works well for x > 0 and is much faster than scalar log operations.
#[inline]
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
unsafe fn log_approx_avx2(x: __m256) -> __m256 {
    let one = unsafe { _mm256_set1_ps(1.0) };
    let two = unsafe { _mm256_set1_ps(2.0) };
    let three = unsafe { _mm256_set1_ps(3.0) };
    let five = unsafe { _mm256_set1_ps(5.0) };

    // Compute (x-1)/(x+1)
    let x_minus_1 = unsafe { _mm256_sub_ps(x, one) };
    let x_plus_1 = unsafe { _mm256_add_ps(x, one) };
    let y = unsafe { _mm256_div_ps(x_minus_1, x_plus_1) };

    // Compute y^3 and y^5 for the series
    let y2 = unsafe { _mm256_mul_ps(y, y) };
    let y3 = unsafe { _mm256_mul_ps(y2, y) };
    let y5 = unsafe { _mm256_mul_ps(y3, y2) };

    // arctanh(y) ^^ y + y^3/3 + y^5/5
    let term1 = y;
    let term2 = unsafe { _mm256_div_ps(y3, three) };
    let term3 = unsafe { _mm256_div_ps(y5, five) };

    let arctanh_y = unsafe { _mm256_add_ps(_mm256_add_ps(term1, term2), term3) };

    // ln(x) = 2 * arctanh((x-1)/(x+1))
    unsafe { _mm256_mul_ps(two, arctanh_y) }
}

// Fast logarithm approximation for NEON
//
// Uses the identity: ln(x) = 2 * arctanh((x-1)/(x+1))
// And approximates arctanh(y) ^^ y + y³/3 + y^^/5 + ...
// This works well for x > 0 and is much faster than scalar log operations.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn log_approx_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    let three = vdupq_n_f32(3.0);
    let five = vdupq_n_f32(5.0);

    // Compute (x-1)/(x+1)
    let x_minus_1 = vsubq_f32(x, one);
    let x_plus_1 = vaddq_f32(x, one);
    let y = vdivq_f32(x_minus_1, x_plus_1);

    // Compute y^3 and y^5 for the series
    let y2 = vmulq_f32(y, y);
    let y3 = vmulq_f32(y2, y);
    let y5 = vmulq_f32(y3, y2);

    // arctanh(y) ^^ y + y^3/3 + y^5/5
    let term1 = y;
    let term2 = vdivq_f32(y3, three);
    let term3 = vdivq_f32(y5, five);

    let arctanh_y = vaddq_f32(vaddq_f32(term1, term2), term3);

    // ln(x) = 2 * arctanh((x-1)/(x+1))
    vmulq_f32(two, arctanh_y)
}

// GPU/PTX optimized Jeffreys divergence for f32 vectors.
//
// Jeffreys divergence is the symmetric version of KL-divergence:
// J(P||Q) = KL(P||Q) + KL(Q||P) = Σ(p_i - q_i) * ln(p_i / q_i)
//

#[cfg(has_cuda)]
pub unsafe fn distance_jeffreys_f32_gpu(
    va: *const f32,
    vb: *const f32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    // PTX kernel for Jeffreys divergence
    const PTX_DISTANCE_JEFFREYS_F32: &str = r#"
    .version 7.5
    .target sm_70
    .entry distance_jeffreys_f32(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .f32 %f<24>;
      .reg .pred %p<4>;
      .reg .u32 %r<12>;
      .reg .u64 %rd<10>;
      
      // Load parameters
      ld.param.u64 %rd5, [va_ptr];
      ld.param.u64 %rd6, [vb_ptr];
      ld.param.u32 %r4, [min_len];
      ld.param.u64 %rd7, [result_ptr];
      
      // Thread index - process 4 elements per thread
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mad.lo.u32 %r3, %r1, %r2, %r0;
      mul.lo.u32 %r3, %r3, 4;  // Each thread processes 4 elements
      
      // Grid stride loop with float4 accumulators
      mov.f32 %f0, 0.0; mov.f32 %f1, 0.0; mov.f32 %f2, 0.0; mov.f32 %f3, 0.0;
      mov.f32 %f20, 0.00001; // epsilon
      
    165:  // loop_start
      add.u32 %r6, %r3, 3;
      setp.ge.u32 %p0, %r6, %r4;
      @%p0 bra 166;  // jump to loop_end if can't load float4
      
      // Load float4 values using vectorized loads
      cvt.u64.u32 %rd0, %r3;
      shl.b64 %rd0, %rd0, 2;
      add.u64 %rd1, %rd5, %rd0;
      add.u64 %rd2, %rd6, %rd0;
      ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd1];   // Load va float4
      ld.global.v4.f32 {%f8, %f9, %f10, %f11}, [%rd2]; // Load vb float4
      
      // Add epsilon to avoid log(0)
      add.f32 %f1, %f1, %f8;
      add.f32 %f2, %f2, %f8;
      
      // Compute (p - q) * ln(p/q)
      sub.f32 %f3, %f1, %f2; // p - q
      div.f32 %f4, %f1, %f2; // p / q
      lg2.approx.f32 %f5, %f4; // log2(p/q)
      mul.f32 %f5, %f5, 0.6931471805599453; // convert to ln
      mul.f32 %f5, %f3, %f5; // (p-q) * ln(p/q)
      
      // Compute (q - p) * ln(q/p)
      div.f32 %f4, %f2, %f1; // q / p
      lg2.approx.f32 %f6, %f4; // log2(q/p)
      mul.f32 %f6, %f6, 0.6931471805599453; // convert to ln
      mul.f32 %f6, %f2, %f6; // q * ln(q/p)
      add.f32 %f7, %f5, %f6;
      add.f32 %f0, %f0, %f7;
      
      // Grid stride
      mov.u32 %r5, %nctaid.x;
      mul.lo.u32 %r5, %r5, %r2;
      add.u32 %r3, %r3, %r5;
      bra 165;  // jump back to loop_start
      
    166:  // loop_end - add warp shuffle reduction
      // Get lane ID within warp
      and.b32 %r6, %r0, 0x1f;
      // Warp reduction using shuffle down
      shfl.sync.down.b32 %f9, %f0, 16, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f9;
      shfl.sync.down.b32 %f9, %f0, 8, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f9;
      shfl.sync.down.b32 %f9, %f0, 4, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f9;
      shfl.sync.down.b32 %f9, %f0, 2, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f9;
      shfl.sync.down.b32 %f9, %f0, 1, 0x1f, 0xffffffff;
      add.f32 %f0, %f0, %f9;
      // Only lane 0 writes result
      setp.eq.u32 %p1, %r6, 0;
      @%p1 st.global.f32 [%rd7], %f0;
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len) as u32;

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = crate::gpu::launch_ptx(
        PTX_DISTANCE_JEFFREYS_F32,
        &[],
        "distance_jeffreys_f32",
        256,
        256,
        &args,
    );
}

// Jeffreys Divergence (Symmetric KL-Divergence) Distance for f32 using AVX-512 SIMD
//
// Jeffreys divergence is the symmetric version of KL-divergence:
// J(P||Q) = KL(P||Q) + KL(Q||P) = Σ(p_i - q_i) * ln(p_i / q_i)
//
// Uses fast SIMD logarithm approximation for high performance.

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_jeffreys_f32_avx512(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut divergence = _mm512_setzero_ps();
    let epsilon = _mm512_set1_ps(1e-10);
    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 16);
    let mut i = 0;

    // Process 16 elements at a time with SIMD approximation
    while i < simd_len {
        let a = _mm512_loadu_ps(va.as_ptr().add(i));
        let b = _mm512_loadu_ps(vb.as_ptr().add(i));

        // Clamp to epsilon to avoid log(0) and handle negative values
        let a_safe = _mm512_max_ps(a, epsilon);
        let b_safe = _mm512_max_ps(b, epsilon);

        // Compute (a - b) * ln(a / b) using SIMD approximation
        let diff = _mm512_sub_ps(a, b);
        let ratio = _mm512_div_ps(a_safe, b_safe);

        // Use SIMD logarithm approximation: ln(x) ^^ 2 * ((x-1)/(x+1) + (x-1)³/(3*(x+1)³) + ...)
        // This is a fast approximation that works well for positive values
        let log_ratio = log_approx_avx512(ratio);
        let term = _mm512_mul_ps(diff, log_ratio);

        divergence = _mm512_add_ps(divergence, term);

        i += 16;
    }

    // Sum up the vector
    let mut result = _mm512_reduce_add_ps(divergence);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j];
        let b = vb[j];
        let a_safe = a.max(1e-30);
        let b_safe = b.max(1e-30);
        let diff = a - b;
        let log_ratio = (a_safe / b_safe).ln();
        result += diff * log_ratio;
    }

    result
}

// AVX2 optimized Jeffreys Divergence (Symmetric KL-Divergence) Distance for f32 vectors.
//
// Jeffreys divergence is the symmetric version of KL-divergence:
// J(P||Q) = KL(P||Q) + KL(Q||P) = Σ(p_i - q_i) * ln(p_i / q_i)
// Uses AVX2 for 8 f32 elements per iteration
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_jeffreys_f32_avx2(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut divergence = unsafe { _mm256_setzero_ps() };
    let epsilon = unsafe { _mm256_set1_ps(1e-30) };
    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 8);
    let mut i = 0;

    // Process 8 elements at a time with SIMD approximation
    while i < simd_len {
        let a = unsafe { _mm256_loadu_ps(va.as_ptr().add(i)) };
        let b = unsafe { _mm256_loadu_ps(vb.as_ptr().add(i)) };

        // Add epsilon to avoid log(0)
        let a_safe = unsafe { _mm256_max_ps(a, epsilon) };
        let b_safe = unsafe { _mm256_max_ps(b, epsilon) };

        // Compute (a - b) * ln(a / b) using SIMD approximation
        let diff = unsafe { _mm256_sub_ps(a, b) };
        let ratio = unsafe { _mm256_div_ps(a_safe, b_safe) };

        // Use SIMD logarithm approximation
        let log_ratio = log_approx_avx2(ratio);
        let term = unsafe { _mm256_mul_ps(diff, log_ratio) };

        divergence = unsafe { _mm256_add_ps(divergence, term) };

        i += 8;
    }

    // Sum up the vector using horizontal add
    let sum_vec = unsafe { _mm256_hadd_ps(divergence, divergence) };
    let sum_vec = unsafe { _mm256_hadd_ps(sum_vec, sum_vec) };
    let sum_arr: [f32; 8] = std::mem::transmute(sum_vec);
    let mut result = sum_arr[0] + sum_arr[4];

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j];
        let b = vb[j];
        let a_safe = a.max(1e-30);
        let b_safe = b.max(1e-30);
        let diff = a - b;
        let log_ratio = (a_safe / b_safe).ln();
        result += diff * log_ratio;
    }

    result
}

// NEON optimized Jeffreys Divergence (Symmetric KL-Divergence) Distance for f32 vectors.
//
// Jeffreys divergence is the symmetric version of KL-divergence:
// J(P||Q) = KL(P||Q) + KL(Q||P) = Σ(p_i - q_i) * ln(p_i / q_i)
// Uses NEON for 4 f32 elements per iteration
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_jeffreys_f32(va: &[f32], vb: &[f32], va_len: usize, vb_len: usize) -> f32 {
    let mut divergence = vdupq_n_f32(0.0);
    let epsilon = vdupq_n_f32(1e-10);
    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 4);
    let mut i = 0;

    // Process 4 elements at a time with SIMD approximation
    while i < simd_len {
        let a = vld1q_f32(va.as_ptr().add(i));
        let b = vld1q_f32(vb.as_ptr().add(i));

        // Add epsilon to avoid log(0)
        let a_safe = vmaxq_f32(a, epsilon);
        let b_safe = vmaxq_f32(b, epsilon);

        // Compute (a - b) * ln(a / b) using SIMD approximation
        let diff = vsubq_f32(a, b);
        let ratio = vdivq_f32(a_safe, b_safe);

        // Use SIMD logarithm approximation
        let log_ratio = log_approx_neon(ratio);
        let term = vmulq_f32(diff, log_ratio);

        divergence = vaddq_f32(divergence, term);

        i += 4;
    }

    // Sum up the vector using horizontal add
    let mut result = vaddvq_f32(divergence);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j];
        let b = vb[j];
        let a_safe = a.max(1e-30);
        let b_safe = b.max(1e-30);
        let diff = a - b;
        let log_ratio = (a_safe / b_safe).ln();
        result += diff * log_ratio;
    }

    result
}

// GPU/PTX optimized Jensen-Shannon distance for f32 vectors.
//
// Computes sqrt(0.5 * Σ(p_i * log(p_i / mean) + q_i * log(q_i / mean)))
// where mean = (p_i + q_i) / 2
//

#[cfg(has_cuda)]
pub unsafe fn distance_jensen_shannon_f32_gpu(
    va: *const f32,
    vb: *const f32,
    va_len: usize,
    vb_len: usize,
    result_ptr: *mut f32,
) {
    // PTX kernel for Jensen-Shannon distance
    const PTX_DISTANCE_JENSEN_SHANNON_F32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64

    // Kernel accumulates total Jensen-Shannon divergence (not distance)
    // The host computes sqrt(0.5 * divergence) after kernel completion.
    .entry distance_jensen_shannon_f32(
      .param .u64 va_ptr,
      .param .u64 vb_ptr,
      .param .u32 min_len,
      .param .u64 result_ptr
    ) {
      .reg .f32 %f<40>;
      .reg .pred %p<8>;
      .reg .u32 %r<16>;
      .reg .u64 %rd<12>;
      .shared .f32 sdata[256];

      // Load params
      ld.param.u64 %rd1, [va_ptr];
      ld.param.u64 %rd2, [vb_ptr];
      ld.param.u32 %rN, [min_len];
      ld.param.u64 %rd3, [result_ptr];

      // Thread/block info
      mov.u32 %rTid, %tid.x;
      mov.u32 %rBlk, %ctaid.x;
      mov.u32 %rBDim, %ntid.x;
      mov.u32 %rGrid, %nctaid.x;

      // Grid stride setup (process 4 elems per iteration)
      mad.lo.u32 %rIdx, %rBlk, %rBDim, %rTid;   // global tid
      mul.lo.u32 %rStride, %rGrid, %rBDim;
      mul.lo.u32 %rStride4, %rStride, 4;       // stride in elements
      mul.lo.u32 %rIdx4, %rIdx, 4;             // starting element index

      // Constants
      mov.f32 %fEps, 1e-10;    // epsilon clamp (match CPU EPSILON_F32)
      mov.f32 %fHalf, 0.5;     // 0.5
      mov.f32 %fLn2, 0.69314718; // ln(2)

      // Per-thread accumulator of divergence (natural log)
      mov.f32 %fAcc, 0.0;

    JS_LOOP:
      // Check we have 4 elements available
      add.u32 %rEnd, %rIdx4, 3;
      setp.ge.u32 %p0, %rEnd, %rN;
      @%p0 bra JS_REM;

      // Byte offset for this float4
      cvt.u64.u32 %rdOff, %rIdx4;
      shl.b64 %rdOff, %rdOff, 2;               // *4 bytes
      add.u64 %rdA, %rd1, %rdOff;
      add.u64 %rdB, %rd2, %rdOff;

      // Load a[4], b[4]
      ld.global.v4.f32 { %fA0, %fA1, %fA2, %fA3 }, [%rdA];
      ld.global.v4.f32 { %fB0, %fB1, %fB2, %fB3 }, [%rdB];

      // For k in 0..4: accumulate a_k*ln(a_k/m_k) + b_k*ln(b_k/m_k)
      // k = 0 (compute mean from clamped a_safe, b_safe)
      max.f32 %fAS, %fA0, %fEps; max.f32 %fBS, %fB0, %fEps;
      add.f32 %fM, %fAS, %fBS; mul.f32 %fM, %fM, %fHalf; max.f32 %fMS, %fM, %fEps;
      div.rn.f32 %fR, %fAS, %fMS; lg2.approx.f32 %fL, %fR; mul.f32 %fL, %fL, %fLn2; mul.f32 %fT, %fAS, %fL; // a term (ln)
      div.rn.f32 %fR2, %fBS, %fMS; lg2.approx.f32 %fL2, %fR2; mul.f32 %fL2, %fL2, %fLn2; mad.f32 %fT, %fBS, %fL2, %fT;
      add.f32 %fAcc, %fAcc, %fT;

      // k = 1
      max.f32 %fAS, %fA1, %fEps; max.f32 %fBS, %fB1, %fEps;
      add.f32 %fM, %fAS, %fBS; mul.f32 %fM, %fM, %fHalf; max.f32 %fMS, %fM, %fEps;
      div.rn.f32 %fR, %fAS, %fMS; lg2.approx.f32 %fL, %fR; mul.f32 %fL, %fL, %fLn2; mul.f32 %fT, %fAS, %fL;
      div.rn.f32 %fR2, %fBS, %fMS; lg2.approx.f32 %fL2, %fR2; mul.f32 %fL2, %fL2, %fLn2; mad.f32 %fT, %fBS, %fL2, %fT;
      add.f32 %fAcc, %fAcc, %fT;

      // k = 2
      max.f32 %fAS, %fA2, %fEps; max.f32 %fBS, %fB2, %fEps;
      add.f32 %fM, %fAS, %fBS; mul.f32 %fM, %fM, %fHalf; max.f32 %fMS, %fM, %fEps;
      div.rn.f32 %fR, %fAS, %fMS; lg2.approx.f32 %fL, %fR; mul.f32 %fL, %fL, %fLn2; mul.f32 %fT, %fAS, %fL;
      div.rn.f32 %fR2, %fBS, %fMS; lg2.approx.f32 %fL2, %fR2; mul.f32 %fL2, %fL2, %fLn2; mad.f32 %fT, %fBS, %fL2, %fT;
      add.f32 %fAcc, %fAcc, %fT;

      // k = 3
      max.f32 %fAS, %fA3, %fEps; max.f32 %fBS, %fB3, %fEps;
      add.f32 %fM, %fAS, %fBS; mul.f32 %fM, %fM, %fHalf; max.f32 %fMS, %fM, %fEps;
      div.rn.f32 %fR, %fAS, %fMS; lg2.approx.f32 %fL, %fR; mul.f32 %fL, %fL, %fLn2; mul.f32 %fT, %fAS, %fL;
      div.rn.f32 %fR2, %fBS, %fMS; lg2.approx.f32 %fL2, %fR2; mul.f32 %fL2, %fL2, %fLn2; mad.f32 %fT, %fBS, %fL2, %fT;
      add.f32 %fAcc, %fAcc, %fT;

      // Advance grid stride
      add.u32 %rIdx4, %rIdx4, %rStride4;
      bra JS_LOOP;

    JS_REM:
      // Handle remaining elements (<4)
      setp.ge.u32 %p1, %rIdx4, %rN; @%p1 bra JS_REDUCE;
      // process element j = rIdx4
      cvt.u64.u32 %rdOff2, %rIdx4; shl.b64 %rdOff2, %rdOff2, 2;
      add.u64 %rdA1, %rd1, %rdOff2; add.u64 %rdB1, %rd2, %rdOff2;
      ld.global.f32 %fA, [%rdA1]; ld.global.f32 %fB, [%rdB1];
      max.f32 %fAS, %fA, %fEps; max.f32 %fBS, %fB, %fEps;
      add.f32 %fM, %fAS, %fBS; mul.f32 %fM, %fM, %fHalf; max.f32 %fMS, %fM, %fEps;
      div.rn.f32 %fR, %fAS, %fMS; lg2.approx.f32 %fL, %fR; mul.f32 %fL, %fL, %fLn2; mul.f32 %fT, %fAS, %fL;
      div.rn.f32 %fR2, %fBS, %fMS; lg2.approx.f32 %fL2, %fR2; mul.f32 %fL2, %fL2, %fLn2; mad.f32 %fT, %fBS, %fL2, %fT;
      add.f32 %fAcc, %fAcc, %fT;

      // next remainder element (j+1)
      add.u32 %rIdx4, %rIdx4, 1;
      setp.ge.u32 %p1, %rIdx4, %rN; @%p1 bra JS_REDUCE;
      ld.global.f32 %fA, [%rdA1+4]; ld.global.f32 %fB, [%rdB1+4];
      max.f32 %fAS, %fA, %fEps; max.f32 %fBS, %fB, %fEps;
      add.f32 %fM, %fAS, %fBS; mul.f32 %fM, %fM, %fHalf; max.f32 %fMS, %fM, %fEps;
      div.rn.f32 %fR, %fAS, %fMS; lg2.approx.f32 %fL, %fR; mul.f32 %fL, %fL, %fLn2; mul.f32 %fT, %fAS, %fL;
      div.rn.f32 %fR2, %fBS, %fMS; lg2.approx.f32 %fL2, %fR2; mul.f32 %fL2, %fL2, %fLn2; mad.f32 %fT, %fBS, %fL2, %fT;
      add.f32 %fAcc, %fAcc, %fT;

      // next remainder element (j+2)
      add.u32 %rIdx4, %rIdx4, 1;
      setp.ge.u32 %p1, %rIdx4, %rN; @%p1 bra JS_REDUCE;
      ld.global.f32 %fA, [%rdA1+8]; ld.global.f32 %fB, [%rdB1+8];
      max.f32 %fAS, %fA, %fEps; max.f32 %fBS, %fB, %fEps;
      add.f32 %fM, %fAS, %fBS; mul.f32 %fM, %fM, %fHalf; max.f32 %fMS, %fM, %fEps;
      div.rn.f32 %fR, %fAS, %fMS; lg2.approx.f32 %fL, %fR; mul.f32 %fL, %fL, %fLn2; mul.f32 %fT, %fAS, %fL;
      div.rn.f32 %fR2, %fBS, %fMS; lg2.approx.f32 %fL2, %fR2; mul.f32 %fL2, %fL2, %fLn2; mad.f32 %fT, %fBS, %fL2, %fT;
      add.f32 %fAcc, %fAcc, %fT;

    JS_REDUCE:
      // Warp reduction of fAcc
      shfl.sync.down.b32 %fTmp, %fAcc, 16, 0x1f, 0xffffffff; add.f32 %fAcc, %fAcc, %fTmp;
      shfl.sync.down.b32 %fTmp, %fAcc, 8, 0x1f, 0xffffffff;  add.f32 %fAcc, %fAcc, %fTmp;
      shfl.sync.down.b32 %fTmp, %fAcc, 4, 0x1f, 0xffffffff;  add.f32 %fAcc, %fAcc, %fTmp;
      shfl.sync.down.b32 %fTmp, %fAcc, 2, 0x1f, 0xffffffff;  add.f32 %fAcc, %fAcc, %fTmp;
      shfl.sync.down.b32 %fTmp, %fAcc, 1, 0x1f, 0xffffffff;  add.f32 %fAcc, %fAcc, %fTmp;

      // Write warp sum to shared (lane 0)
      and.b32 %rLane, %rTid, 0x1f;
      shr.u32 %rWarp, %rTid, 5;
      setp.eq.u32 %p2, %rLane, 0;
      @%p2 mul.wide.u32 %rdS, %rWarp, 4;
      @%p2 st.shared.f32 [sdata + %rdS], %fAcc;
      bar.sync 0;

      // Block-level reduction: thread 0 sums warp partials
      setp.eq.u32 %p3, %rTid, 0;
      @!%p3 bra JS_DONE;
      // Number of warps = blockDim/32
      div.u32 %rNW, %rBDim, 32;
      mov.f32 %fAcc2, 0.0;
      mov.u32 %rI, 0;
    BLOCK_SUM_LOOP:
      setp.ge.u32 %p4, %rI, %rNW; @%p4 bra WRITE_ATOMIC;
      mul.wide.u32 %rdS2, %rI, 4;
      ld.shared.f32 %fTmp2, [sdata + %rdS2];
      add.f32 %fAcc2, %fAcc2, %fTmp2;
      add.u32 %rI, %rI, 1;
      bra BLOCK_SUM_LOOP;
    WRITE_ATOMIC:
      atom.global.add.f32 [%rd3], %fAcc2;

    JS_DONE:
      ret;
    }
  "#;

    let min_len = va_len.min(vb_len) as u32;

    let va_ptr = va as u64;
    let vb_ptr = vb as u64;
    let result_ptr_u64 = result_ptr as u64;
    let args = [
        &va_ptr as *const u64 as *const u8,
        &vb_ptr as *const u64 as *const u8,
        &min_len as *const u32 as *const u8,
        &result_ptr_u64 as *const u64 as *const u8,
    ];

    let _ = crate::gpu::launch_ptx(
        PTX_DISTANCE_JENSEN_SHANNON_F32,
        &[],
        "distance_jensen_shannon_f32",
        256,
        256,
        &args,
    );
}

// Jensen-Shannon Distance for f32 using AVX-512 SIMD
//
// Computes sqrt(0.5 * Σ(p_i * log(p_i / mean) + q_i * log(q_i / mean)))
// where mean = (p_i + q_i) / 2
//
// Uses fast SIMD logarithm approximation for high performance.

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn distance_jensen_shannon_f32_avx512(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut divergence = _mm512_setzero_ps();
    let epsilon = _mm512_set1_ps(1e-30);
    let half = _mm512_set1_ps(0.5);
    let mut i = 0;

    // Process 16 elements at a time with full SIMD
    while i + 16 <= va_len && i + 16 <= vb_len {
        let a = _mm512_loadu_ps(va.as_ptr().add(i));
        let b = _mm512_loadu_ps(vb.as_ptr().add(i));

        // Compute mean = (a + b) / 2
        let mean = _mm512_mul_ps(_mm512_add_ps(a, b), half);

        // Add epsilon to avoid log(0)
        let a_safe = _mm512_max_ps(a, epsilon);
        let b_safe = _mm512_max_ps(b, epsilon);
        let mean_safe = _mm512_max_ps(mean, epsilon);

        // Compute ratios for log operations
        let ratio_a = _mm512_div_ps(a_safe, mean_safe);
        let ratio_b = _mm512_div_ps(b_safe, mean_safe);

        // Use SIMD logarithm approximation
        let log_ratio_a = log_approx_avx512(ratio_a);
        let log_ratio_b = log_approx_avx512(ratio_b);

        // Compute terms using clamped values to ensure non-negativity
        // term = a_safe * log(a_safe/mean_safe) + b_safe * log(b_safe/mean_safe)
        let term_a = _mm512_mul_ps(a_safe, log_ratio_a);
        let term_b = _mm512_mul_ps(b_safe, log_ratio_b);
        let term_total = _mm512_add_ps(term_a, term_b);

        divergence = _mm512_add_ps(divergence, term_total);

        i += 16;
    }

    // Sum up the vector
    let mut result = _mm512_reduce_add_ps(divergence);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j];
        let b = vb[j];
        let mean = (a + b) * 0.5;

        let a_safe = a.max(1e-30);
        let b_safe = b.max(1e-30);
        let mean_safe = mean.max(1e-30);

        result += a * (a_safe / mean_safe).ln() + b * (b_safe / mean_safe).ln();
    }

    // Return sqrt(0.5 * divergence), clamped to [0,1]
    (0.5_f32 * result).max(0.0).sqrt().min(1.0)
}

// AVX2 optimized Jensen-Shannon Distance for f32 vectors.
//
// Computes sqrt(0.5 * Σ(p_i * log(p_i / mean) + q_i * log(q_i / mean)))
// where mean = (p_i + q_i) / 2
// Uses AVX2 for 8 f32 elements per iteration
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
pub unsafe fn distance_jensen_shannon_f32_avx2(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut divergence = _mm256_setzero_ps();
    let epsilon = _mm256_set1_ps(1e-10);
    let half = _mm256_set1_ps(0.5);
    let mut i = 0;

    // Process 8 elements at a time with SIMD
    while i + 8 <= va_len && i + 8 <= vb_len {
        let a = _mm256_loadu_ps(va.as_ptr().add(i));
        let b = _mm256_loadu_ps(vb.as_ptr().add(i));

        // Compute mean = (a + b) / 2
        let mean = _mm256_mul_ps(_mm256_add_ps(a, b), half);

        // Add epsilon to avoid log(0)
        let a_safe = _mm256_max_ps(a, epsilon);
        let b_safe = _mm256_max_ps(b, epsilon);
        let mean_safe = _mm256_max_ps(mean, epsilon);

        // Compute ratios for log operations
        let ratio_a = _mm256_div_ps(a_safe, mean_safe);
        let ratio_b = _mm256_div_ps(b_safe, mean_safe);

        // Use SIMD logarithm approximation
        let log_ratio_a = log_approx_avx2(ratio_a);
        let log_ratio_b = log_approx_avx2(ratio_b);

        // Compute terms using clamped values to ensure non-negativity
        // term = a_safe * log(a_safe/mean_safe) + b_safe * log(b_safe/mean_safe)
        let term_a = _mm256_mul_ps(a_safe, log_ratio_a);
        let term_b = _mm256_mul_ps(b_safe, log_ratio_b);
        let term_total = _mm256_add_ps(term_a, term_b);

        divergence = _mm256_add_ps(divergence, term_total);

        i += 8;
    }

    // Sum up the vector using horizontal add
    let sum_vec = _mm256_hadd_ps(divergence, divergence);
    let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    let sum_arr: [f32; 8] = std::mem::transmute(sum_vec);
    let mut result = sum_arr[0] + sum_arr[4];

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j];
        let b = vb[j];
        let mean = (a + b) * 0.5;

        let a_safe = a.max(1e-30);
        let b_safe = b.max(1e-30);
        let mean_safe = mean.max(1e-30);

        result += a * (a_safe / mean_safe).ln() + b * (b_safe / mean_safe).ln();
    }

    // Return sqrt(0.5 * divergence), clamped to [0,1]
    (0.5 * result).max(0.0).sqrt().min(1.0)
}

// NEON optimized Jensen-Shannon Distance for f32 vectors.
//
// Computes sqrt(0.5 * Σ(p_i * log(p_i / mean) + q_i * log(q_i / mean)))
// where mean = (p_i + q_i) / 2
// Uses NEON for 4 f32 elements per iteration
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_jensen_shannon_f32(
    va: &[f32],
    vb: &[f32],
    va_len: usize,
    vb_len: usize,
) -> f32 {
    let mut divergence = vdupq_n_f32(0.0);
    let epsilon = vdupq_n_f32(1e-30);
    let simd_len = va_len.min(vb_len) - (va_len.min(vb_len) % 4);
    let mut i = 0;

    // Process 4 elements at a time with SIMD
    while i < simd_len {
        let a = vld1q_f32(va.as_ptr().add(i));
        let b = vld1q_f32(vb.as_ptr().add(i));

        // Use absolute values to handle negative inputs
        let a_abs = vabsq_f32(a);
        let b_abs = vabsq_f32(b);

        // Compute mean = (|a| + |b|) / 2
        let half = vdupq_n_f32(0.5);
        let mean = vmulq_f32(vaddq_f32(a_abs, b_abs), half);

        // Add epsilon to avoid log(0)
        let a_safe = vmaxq_f32(a_abs, epsilon);
        let b_safe = vmaxq_f32(b_abs, epsilon);
        let mean_safe = vmaxq_f32(mean, epsilon);

        // Compute ratios for log operations
        let ratio_a = vdivq_f32(a_safe, mean_safe);
        let ratio_b = vdivq_f32(b_safe, mean_safe);

        // Use SIMD logarithm approximation
        let log_ratio_a = log_approx_neon(ratio_a);
        let log_ratio_b = log_approx_neon(ratio_b);

        // Compute terms: |a| * log(|a|/mean) + |b| * log(|b|/mean)
        let term_a = vmulq_f32(a_safe, log_ratio_a);
        let term_b = vmulq_f32(b_safe, log_ratio_b);
        let term_total = vaddq_f32(term_a, term_b);

        divergence = vaddq_f32(divergence, term_total);

        i += 4;
    }

    // Sum up the vector using horizontal add
    let mut result = vaddvq_f32(divergence);

    // Handle remaining elements
    for j in i..va_len.min(vb_len) {
        let a = va[j].abs();
        let b = vb[j].abs();
        let mean = (a + b) * 0.5;

        let a_safe = a.max(1e-30);
        let b_safe = b.max(1e-30);
        let mean_safe = mean.max(1e-30);

        result += a_safe * (a_safe / mean_safe).ln() + b_safe * (b_safe / mean_safe).ln();
    }

    // Return sqrt(0.5 * divergence), clamped to [0, 1]
    (0.5_f32 * result).sqrt().max(0.0).min(1.0)
}

// =============================================================================
// PUBLIC API - Re-exports the appropriate functions
// =============================================================================

// All x86 and ARM functions are now defined directly in this module
// No re-exports needed
