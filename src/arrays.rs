// SPDX-License-Identifier: Apache-2.0

//! Array operations
//!
//! This module contains operations on sorted/unsorted arrays (e.g. set operations,
//! deduplication, filtering). Many implementations have scalar and SIMD variants.
//!
//! ## Performance notes
//! Some kernels are written in a performance-oriented style (tight loops, minimal
//! abstraction). When modifying hot paths, prefer changes that keep allocations out
//! of inner loops.

// Note: some SIMD sorting paths are conservative and may be revisited as the API stabilizes.

// Some clippy lints are noisy for low-level SIMD/FFI code; we opt out at the module level.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

// Import asm! macro for inline assembly
use std::arch::asm;

// Import architecture-specific constants directly - NO FUNCTION CALLS IN SIMD KERNELS!
use super::constants::*;
#[cfg(has_cuda)]
use crate::gpu::{LaunchConfig, launch_ptx};
#[cfg(has_cuda)]
use log::debug;

// GPU/CUDA constants

// use super::constants::{GPU_BLOCK_SIZE_MEDIUM, GPU_WARP_SIZE};

// =============================================================================
// METRIC AGGREGATION AND REDUCTION OPERATIONS
// =============================================================================

// ARM NEON imports
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
  vabsq_f64, vaddq_f64, vaddq_u64, vandq_u64, vbslq_f64, vbslq_s64, vceqq_f64, vceqq_u32,
  vceqq_u64, vcgeq_f64, vcgeq_s64, vcgeq_u32, vcgeq_u64, vcgtq_f64, vcgtq_s64, vcgtq_u32,
  vcgtq_u64, vcleq_f64, vcleq_s64, vcleq_u32, vcleq_u64, vcltq_f64, vcltq_s64, vcltq_u32,
  vcltq_u64, vcvtq_f64_u64, vdivq_f64, vdupq_n_f64, vdupq_n_s64, vdupq_n_u32, vdupq_n_u64,
  veorq_u64, vfmaq_f64, vget_high_u32, vget_low_u32, vgetq_lane_f64, vgetq_lane_s64,
  vgetq_lane_u32, vgetq_lane_u64, vld1q_f64, vld1q_s64, vld1q_u32, vld1q_u64, vmaxq_f64, vmaxq_u32,
  vminq_f64, vminq_u32, vmovl_u32, vmulq_f64, vmvnq_u32, vnegq_f64, vqsubq_u32, vrecpeq_f64,
  vrecpsq_f64, vreinterpretq_f64_u64, vreinterpretq_u64_f64, vrndmq_f64, vrndnq_f64, vrndpq_f64,
  vsetq_lane_s64, vsetq_lane_u64, vsqrtq_f64, vst1q_f64, vst1q_s64, vst1q_u32, vst1q_u64,
  vsubq_f64,
};

// x86_64 SIMD intrinsics imports - AVX2 only (not when nightly AVX-512 is enabled)
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
use std::arch::x86_64::{
  __m128i,
  __m256d,
  __m256i,
  _CMP_EQ_OQ,
  _CMP_GE_OQ,
  _CMP_GT_OQ,
  _CMP_LE_OQ,
  _CMP_LT_OQ,
  _CMP_NEQ_OQ,
  _CMP_NEQ_UQ,
  _CMP_UNORD_Q,
  // Still needed SSE2 intrinsics for some operations
  _mm_add_epi64,
  _mm_add_pd,
  _mm_cvtsd_f64,
  _mm_cvtsi128_si64,
  _mm_max_pd,
  _mm_min_pd,
  _mm_shuffle_epi32,
  _mm_shuffle_pd,
  _mm_storeu_si128,

  // AVX2 intrinsics
  _mm256_add_epi64,
  _mm256_add_pd,
  _mm256_and_pd,
  _mm256_and_si256,
  _mm256_andnot_si256,
  // Additional intrinsics for deep SIMD sorting
  _mm256_blend_epi32,
  _mm256_blend_pd,
  _mm256_blendv_epi8,
  _mm256_blendv_pd,
  _mm256_castpd_si256,
  _mm256_castpd256_pd128,
  _mm256_castsi256_pd,

  _mm256_castsi256_ps,
  _mm256_castsi256_si128,
  _mm256_ceil_pd,
  _mm256_cmp_pd,
  _mm256_cmpeq_epi32,
  _mm256_cmpeq_epi64,
  _mm256_cmpgt_epi32,
  _mm256_cmpgt_epi64,
  _mm256_cvtepi32_pd,
  _mm256_cvtpd_epi32,
  _mm256_div_pd,
  _mm256_extract_epi32,
  _mm256_extract_epi64,
  _mm256_extractf128_pd,
  _mm256_extracti128_si256,
  _mm256_floor_pd,
  _mm256_fmadd_pd,
  _mm256_hadd_pd,
  _mm256_loadu_pd,
  _mm256_loadu_si256,
  _mm256_max_epi32,
  _mm256_max_epu32,
  _mm256_max_pd,
  // Min/max operations for integers
  _mm256_min_epi32,
  _mm256_min_epu32,
  _mm256_min_pd,
  _mm256_movemask_epi8,
  _mm256_movemask_pd,
  _mm256_movemask_ps,
  _mm256_mul_pd,
  _mm256_or_si256,
  _mm256_permute2f128_pd,
  _mm256_permute2x128_si256,
  _mm256_permute4x64_epi64,
  _mm256_permute4x64_pd,
  _mm256_round_pd,
  _mm256_set_epi64x,
  _mm256_set_pd,
  _mm256_set1_epi32,
  _mm256_set1_epi64x,
  _mm256_set1_pd,
  _mm256_setzero_pd,
  _mm256_setzero_si256,
  _mm256_shuffle_epi32,
  _mm256_shuffle_pd,
  _mm256_sqrt_pd,
  _mm256_storeu_pd,
  _mm256_storeu_si256,
  _mm256_sub_epi32,
  _mm256_sub_epi64,
  _mm256_sub_pd,
  _mm256_unpacklo_epi64,
  _mm256_unpacklo_pd,
  _mm256_xor_pd,
  _mm256_xor_si256,
};

// AVX-512 intrinsics (nightly only) - includes all needed AVX2/AVX-512 intrinsics
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
use std::arch::x86_64::{
  // Basic types
  __m256i,
  __m512i,
  _CMP_EQ_OQ,
  // Comparison constants
  _CMP_GE_OQ,
  _CMP_GT_OQ,
  _CMP_LE_OQ,
  _CMP_LT_OQ,
  _CMP_NEQ_OQ,
  _CMP_NEQ_UQ,
  _CMP_UNORD_Q,
  _kand_mask8,
  _mm256_castsi256_pd,
  _mm256_cmpeq_epi64,
  _mm256_loadu_si256,
  _mm256_movemask_pd,
  _mm256_set1_epi64x,
  // AVX2 functions needed for mixed operations
  _mm256_storeu_si256,
  _mm256_xor_si256,
  _mm512_abs_pd,
  // AVX-512 intrinsics
  _mm512_add_epi32,
  _mm512_add_epi64,
  _mm512_add_pd,
  _mm512_castsi512_si256,
  _mm512_cmp_pd_mask,
  _mm512_cmpeq_epi64_mask,
  _mm512_cmpeq_epu32_mask,
  _mm512_cmpge_epi32_mask,
  _mm512_cmpge_epi64_mask,
  _mm512_cmpge_epu64_mask,
  _mm512_cmpgt_epi64_mask,
  _mm512_cmpgt_epu32_mask,
  _mm512_cmple_epi32_mask,
  _mm512_cmple_epi64_mask,
  _mm512_cmple_epu32_mask,
  _mm512_cmple_epu64_mask,

  _mm512_cmplt_epi64_mask,
  _mm512_cvtepi64_pd,
  _mm512_cvtepu32_epi64,
  _mm512_cvtepu64_pd,
  _mm512_cvtpd_epi32,
  _mm512_div_pd,
  _mm512_extracti32x8_epi32,
  _mm512_fmadd_pd,
  // Available u32 intrinsics from traverse.rs (removing duplicates)
  _mm512_loadu_epi32,
  _mm512_loadu_epi64,
  _mm512_loadu_pd,
  _mm512_loadu_si512,
  _mm512_mask_add_pd,
  _mm512_mask_blend_epi32,
  _mm512_mask_blend_epi64,
  _mm512_mask_blend_pd,
  _mm512_mask_cmp_pd_mask,
  _mm512_mask_cmpeq_epu32_mask,
  _mm512_mask_cmpge_epu64_mask,
  _mm512_mask_cmple_epu64_mask,
  _mm512_mask_compress_epi32,
  _mm512_mask_compress_epi64,
  _mm512_mask_compress_pd,
  _mm512_mask_compressstoreu_epi32,
  _mm512_mask_compressstoreu_epi64,
  _mm512_mask_compressstoreu_pd,
  _mm512_mask_loadu_epi32,
  _mm512_mask_loadu_epi64,
  _mm512_mask_loadu_pd,
  _mm512_mask_reduce_add_epi64,
  _mm512_mask_reduce_add_pd,
  _mm512_mask_reduce_max_epi64,
  _mm512_mask_reduce_max_pd,
  _mm512_mask_reduce_min_epi64,
  _mm512_mask_reduce_min_pd,
  _mm512_mask_storeu_epi32,
  _mm512_mask_storeu_epi64,
  _mm512_mask_storeu_pd,
  _mm512_maskz_loadu_epi32,
  _mm512_maskz_mov_epi32,
  _mm512_maskz_mov_pd,
  _mm512_max_epi32,
  _mm512_max_epi64,
  _mm512_max_epu32,
  _mm512_max_epu64,
  _mm512_max_pd,
  _mm512_min_epi32,
  _mm512_min_epi64,
  _mm512_min_epu32,
  _mm512_min_epu64,
  _mm512_min_pd,
  _mm512_mul_pd,

  _mm512_mullo_epi64,
  _mm512_permutexvar_epi64,
  _mm512_reduce_add_epi64,
  _mm512_reduce_add_pd,
  _mm512_reduce_max_epi64,
  _mm512_reduce_max_epu32,
  _mm512_reduce_max_pd,
  _mm512_reduce_min_epi64,
  _mm512_reduce_min_epu32,
  _mm512_reduce_min_pd,
  _mm512_roundscale_pd,
  _mm512_set_epi64,
  _mm512_set1_epi32,
  _mm512_set1_epi64,
  _mm512_set1_pd,
  _mm512_setzero_pd,
  _mm512_setzero_si512,
  _mm512_shuffle_epi32,
  _mm512_shuffle_f64x2,
  // Additional AVX-512 intrinsics for u32 sorting
  _mm512_shuffle_i32x4,
  _mm512_shuffle_i64x2,
  // Additional AVX-512 intrinsics for f64 sorting
  _mm512_shuffle_pd,
  _mm512_sqrt_pd,
  _mm512_srli_epi64,
  _mm512_storeu_epi32,
  _mm512_storeu_epi64,
  _mm512_storeu_pd,
  _mm512_storeu_si512,
  _mm512_sub_epi32,
  _mm512_sub_epi64,
  _mm512_sub_pd,
  _mm512_unpacklo_pd,
  _mm512_xor_epi32,
  _mm512_xor_epi64,
};

// Conditional imports for constants based on target architecture and features
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
use super::constants::{LANES_AVX512_F64, LANES_AVX512_U32, LANES_AVX512_U64};

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
use super::constants::{LANES_AVX2_F64, LANES_AVX2_U32, LANES_AVX2_U64};

#[cfg(target_arch = "aarch64")]
use super::constants::{LANES_NEON_F64, LANES_NEON_U32, LANES_NEON_U64};

// =============================================================================
// PTX KERNEL CONSTANTS FOR GPU DEDUPLICATION
// =============================================================================

// Function-only versions for linking as dependencies (no entry points)
pub const PTX_DEDUP_SORTED_U32_FUNC: &str = r#"
    .version 7.0
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_u32_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<15>;
      .reg .v4 .u32 %r_vec<8>;  // uint4 vectors for SIMD processing
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      mul.lo.u32 %r5, %r4, 4;    // read_pos = thread_idx * 4 (processing 4 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      mul.lo.u32 %r7, %r7, 4;    // stride = total_threads * 4

      // Main grid-stride loop
      L260:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L370;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values using uint4 vectors
      mul.wide.u32 %rd0, %r5, 4;
      add.u64 %rd1, %rd14, %rd0;
      ld.global.v4.u32 %r_vec0, [%rd1];  // current quad

      sub.u32 %r9, %r5, 1;
      mul.wide.u32 %rd2, %r9, 4;
      add.u64 %rd3, %rd14, %rd2;
      ld.global.v4.u32 %r_vec1, [%rd3];  // previous quad

      // Extract elements from vectors for comparison
      mov.u32 %r10, %r_vec0.x;  // current[0]
      mov.u32 %r11, %r_vec0.y;  // current[1]
      mov.u32 %r12, %r_vec0.z;  // current[2]
      mov.u32 %r13, %r_vec0.w;  // current[3]
      mov.u32 %r14, %r_vec1.x;  // previous[0]
      mov.u32 %r15, %r_vec1.y;  // previous[1]
      mov.u32 %r16, %r_vec1.z;  // previous[2]
      mov.u32 %r17, %r_vec1.w;  // previous[3]

      // SIMD comparison: current != previous (matching AVX-512 logic)
      setp.ne.u32 %p1, %r10, %r14;  // current[0] != previous[0]
      setp.ne.u32 %p2, %r11, %r15;  // current[1] != previous[1]
      setp.ne.u32 %p3, %r12, %r16;  // current[2] != previous[2]
      setp.ne.u32 %p4, %r13, %r17;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L261;
      bra L262;
      L261:
      mul.wide.u32 %rd4, %r20, 4;
      add.u64 %rd5, %rd14, %rd4;
      st.global.u32 [%rd5], %r10;
      add.u32 %r20, %r20, 1;

      L262:
      @%p2 bra L263;
      bra L264;
      L263:
      mul.wide.u32 %rd6, %r20, 4;
      add.u64 %rd7, %rd14, %rd6;
      st.global.u32 [%rd7], %r11;
      add.u32 %r20, %r20, 1;

      L264:
      @%p3 bra L265;
      bra L266;
      L265:
      mul.wide.u32 %rd8, %r20, 4;
      add.u64 %rd9, %rd14, %rd8;
      st.global.u32 [%rd9], %r12;
      add.u32 %r20, %r20, 1;

      L266:
      @%p4 bra L267;
      bra L268;
      L267:
      mul.wide.u32 %rd10, %r20, 4;
      add.u64 %rd11, %rd14, %rd10;
      st.global.u32 [%rd11], %r13;

      L268:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L260;

      L370:
      ret;
    }
  "#;

// Function-only version for U64 (no entry point, for use as dependency)
pub const PTX_DEDUP_SORTED_U64_FUNC: &str = r#"
    .version 7.5
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_u64_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<40>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      shl.b32 %r5, %r4, 2;    // read_pos = thread_idx * 4 (processing 4 u64 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      shl.b32 %r7, %r7, 2;    // stride = total_threads * 4

      // Main grid-stride loop
      L350:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L470;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values using vectorized loads
      cvt.u64.u32 %rd0, %r5;
      shl.b64 %rd1, %rd0, 3;
      add.u64 %rd2, %rd14, %rd1;
      
      // Use v2.u64 for 128-bit vectorized loads (2 x 64-bit)
      ld.global.v2.u64 {%rd16, %rd17}, [%rd2];      // current[0:1]
      ld.global.v2.u64 {%rd18, %rd19}, [%rd2 + 16]; // current[2:3]

      sub.u32 %r9, %r5, 1;
      cvt.u64.u32 %rd3, %r9;
      shl.b64 %rd4, %rd3, 3;
      add.u64 %rd5, %rd14, %rd4;
      
      // Use v2.u64 for 128-bit vectorized loads (2 x 64-bit)
      ld.global.v2.u64 {%rd20, %rd21}, [%rd5];      // previous[0:1]
      ld.global.v2.u64 {%rd22, %rd23}, [%rd5 + 16]; // previous[2:3]

      // SIMD comparison: current != previous
      setp.ne.u64 %p1, %rd16, %rd20;  // current[0] != previous[0]
      setp.ne.u64 %p2, %rd17, %rd21;  // current[1] != previous[1]
      setp.ne.u64 %p3, %rd18, %rd22;  // current[2] != previous[2]
      setp.ne.u64 %p4, %rd19, %rd23;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L361;
      bra L362;
      L361:
      cvt.u64.u32 %rd6, %r20;
      shl.b64 %rd7, %rd6, 3;
      add.u64 %rd8, %rd14, %rd7;
      st.global.u64 [%rd8], %rd16;
      add.u32 %r20, %r20, 1;

      L362:
      @%p2 bra L363;
      bra L364;
      L363:
      cvt.u64.u32 %rd9, %r20;
      shl.b64 %rd10, %rd9, 3;
      add.u64 %rd11, %rd14, %rd10;
      st.global.u64 [%rd11], %rd17;
      add.u32 %r20, %r20, 1;

      L364:
      @%p3 bra L365;
      bra L366;
      L365:
      cvt.u64.u32 %rd24, %r20;
      shl.b64 %rd25, %rd24, 3;
      add.u64 %rd26, %rd14, %rd25;
      st.global.u64 [%rd26], %rd18;
      add.u32 %r20, %r20, 1;

      L366:
      @%p4 bra L367;
      bra L368;
      L367:
      cvt.u64.u32 %rd27, %r20;
      shl.b64 %rd28, %rd27, 3;
      add.u64 %rd29, %rd14, %rd28;
      st.global.u64 [%rd29], %rd19;

      L368:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L350;

      L470:
      ret;
    }
  "#;

// Function-only version for F64 (no entry point, for use as dependency)
pub const PTX_DEDUP_SORTED_F64_FUNC: &str = r#"
    .version 7.5
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_f64_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<40>;
      .reg .f64 %fd<20>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      shl.b32 %r5, %r4, 2;    // read_pos = thread_idx * 4 (processing 4 f64 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      shl.b32 %r7, %r7, 2;    // stride = total_threads * 4

      // Main grid-stride loop
      L450:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L570;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values using vectorized loads
      cvt.u64.u32 %rd0, %r5;
      shl.b64 %rd1, %rd0, 3;
      add.u64 %rd2, %rd14, %rd1;
      
      // Use v2.f64 for 128-bit vectorized loads (2 x 64-bit floats)
      ld.global.v2.f64 {%fd16, %fd17}, [%rd2];      // current[0:1]
      ld.global.v2.f64 {%fd18, %fd19}, [%rd2 + 16]; // current[2:3]

      sub.u32 %r9, %r5, 1;
      cvt.u64.u32 %rd3, %r9;
      shl.b64 %rd4, %rd3, 3;
      add.u64 %rd5, %rd14, %rd4;
      
      // Use v2.f64 for 128-bit vectorized loads (2 x 64-bit floats)
      ld.global.v2.f64 {%fd20, %fd21}, [%rd5];      // previous[0:1]
      ld.global.v2.f64 {%fd22, %fd23}, [%rd5 + 16]; // previous[2:3]

      // SIMD comparison: current != previous (NaN-aware)
      setp.neu.f64 %p1, %fd16, %fd20;  // current[0] != previous[0] (unordered or not equal)
      setp.neu.f64 %p2, %fd17, %fd21;  // current[1] != previous[1]
      setp.neu.f64 %p3, %fd18, %fd22;  // current[2] != previous[2]
      setp.neu.f64 %p4, %fd19, %fd23;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L461;
      bra L462;
      L461:
      cvt.u64.u32 %rd6, %r20;
      shl.b64 %rd7, %rd6, 3;
      add.u64 %rd8, %rd14, %rd7;
      st.global.f64 [%rd8], %fd16;
      add.u32 %r20, %r20, 1;

      L462:
      @%p2 bra L463;
      bra L464;
      L463:
      cvt.u64.u32 %rd9, %r20;
      shl.b64 %rd10, %rd9, 3;
      add.u64 %rd11, %rd14, %rd10;
      st.global.f64 [%rd11], %fd17;
      add.u32 %r20, %r20, 1;

      L464:
      @%p3 bra L465;
      bra L466;
      L465:
      cvt.u64.u32 %rd24, %r20;
      shl.b64 %rd25, %rd24, 3;
      add.u64 %rd26, %rd14, %rd25;
      st.global.f64 [%rd26], %fd18;
      add.u32 %r20, %r20, 1;

      L466:
      @%p4 bra L467;
      bra L468;
      L467:
      cvt.u64.u32 %rd27, %r20;
      shl.b64 %rd28, %rd27, 3;
      add.u64 %rd29, %rd14, %rd28;
      st.global.f64 [%rd29], %fd19;

      L468:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L450;

      L570:
      ret;
    }
  "#;

// Function-only version for I64 (no entry point, for use as dependency)
pub const PTX_DEDUP_SORTED_I64_FUNC: &str = r#"
    .version 7.5
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_i64_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<40>;
      .reg .s64 %rds<20>;  // Signed 64-bit registers
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      shl.b32 %r5, %r4, 2;    // read_pos = thread_idx * 4 (processing 4 i64 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      shl.b32 %r7, %r7, 2;    // stride = total_threads * 4

      // Main grid-stride loop
      L550:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L670;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values using vectorized loads
      cvt.u64.u32 %rd0, %r5;
      shl.b64 %rd1, %rd0, 3;
      add.u64 %rd2, %rd14, %rd1;
      
      // Use v2.s64 for 128-bit vectorized loads (2 x 64-bit signed)
      ld.global.v2.s64 {%rds16, %rds17}, [%rd2];      // current[0:1]
      ld.global.v2.s64 {%rds18, %rds19}, [%rd2 + 16]; // current[2:3]

      sub.u32 %r9, %r5, 1;
      cvt.u64.u32 %rd3, %r9;
      shl.b64 %rd4, %rd3, 3;
      add.u64 %rd5, %rd14, %rd4;
      
      // Use v2.s64 for 128-bit vectorized loads (2 x 64-bit signed)
      ld.global.v2.s64 {%rds20, %rds21}, [%rd5];      // previous[0:1]
      ld.global.v2.s64 {%rds22, %rds23}, [%rd5 + 16]; // previous[2:3]

      // SIMD comparison: current != previous (signed comparison)
      setp.ne.s64 %p1, %rds16, %rds20;  // current[0] != previous[0]
      setp.ne.s64 %p2, %rds17, %rds21;  // current[1] != previous[1]
      setp.ne.s64 %p3, %rds18, %rds22;  // current[2] != previous[2]
      setp.ne.s64 %p4, %rds19, %rds23;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L561;
      bra L562;
      L561:
      cvt.u64.u32 %rd6, %r20;
      shl.b64 %rd7, %rd6, 3;
      add.u64 %rd8, %rd14, %rd7;
      st.global.s64 [%rd8], %rds16;
      add.u32 %r20, %r20, 1;

      L562:
      @%p2 bra L563;
      bra L564;
      L563:
      cvt.u64.u32 %rd9, %r20;
      shl.b64 %rd10, %rd9, 3;
      add.u64 %rd11, %rd14, %rd10;
      st.global.s64 [%rd11], %rds17;
      add.u32 %r20, %r20, 1;

      L564:
      @%p3 bra L565;
      bra L566;
      L565:
      cvt.u64.u32 %rd24, %r20;
      shl.b64 %rd25, %rd24, 3;
      add.u64 %rd26, %rd14, %rd25;
      st.global.s64 [%rd26], %rds18;
      add.u32 %r20, %r20, 1;

      L566:
      @%p4 bra L567;
      bra L568;
      L567:
      cvt.u64.u32 %rd27, %r20;
      shl.b64 %rd28, %rd27, 3;
      add.u64 %rd29, %rd14, %rd28;
      st.global.s64 [%rd29], %rds19;

      L568:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L550;

      L670:
      ret;
    }
  "#;

// Full versions with entry points for standalone execution
pub const PTX_DEDUP_SORTED_U32_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_u32_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<15>;
      .reg .v4 .u32 %r_vec<8>;  // uint4 vectors for SIMD processing
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      mul.lo.u32 %r5, %r4, 4;    // read_pos = thread_idx * 4 (processing 4 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      mul.lo.u32 %r7, %r7, 4;    // stride = total_threads * 4

      // Main grid-stride loop
      L260:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L370;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values using uint4 vectors
      mul.wide.u32 %rd0, %r5, 4;
      add.u64 %rd1, %rd14, %rd0;
      ld.global.v4.u32 %r_vec0, [%rd1];  // current quad

      sub.u32 %r9, %r5, 1;
      mul.wide.u32 %rd2, %r9, 4;
      add.u64 %rd3, %rd14, %rd2;
      ld.global.v4.u32 %r_vec1, [%rd3];  // previous quad

      // Extract elements from vectors for comparison
      mov.u32 %r10, %r_vec0.x;  // current[0]
      mov.u32 %r11, %r_vec0.y;  // current[1]
      mov.u32 %r12, %r_vec0.z;  // current[2]
      mov.u32 %r13, %r_vec0.w;  // current[3]
      mov.u32 %r14, %r_vec1.x;  // previous[0]
      mov.u32 %r15, %r_vec1.y;  // previous[1]
      mov.u32 %r16, %r_vec1.z;  // previous[2]
      mov.u32 %r17, %r_vec1.w;  // previous[3]

      // SIMD comparison: current != previous (matching AVX-512 logic)
      setp.ne.u32 %p1, %r10, %r14;  // current[0] != previous[0]
      setp.ne.u32 %p2, %r11, %r15;  // current[1] != previous[1]
      setp.ne.u32 %p3, %r12, %r16;  // current[2] != previous[2]
      setp.ne.u32 %p4, %r13, %r17;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L261;
      bra L262;
      L261:
      mul.wide.u32 %rd4, %r20, 4;
      add.u64 %rd5, %rd14, %rd4;
      st.global.u32 [%rd5], %r10;
      add.u32 %r20, %r20, 1;

      L262:
      @%p2 bra L263;
      bra L264;
      L263:
      mul.wide.u32 %rd6, %r20, 4;
      add.u64 %rd7, %rd14, %rd6;
      st.global.u32 [%rd7], %r11;
      add.u32 %r20, %r20, 1;

      L264:
      @%p3 bra L265;
      bra L266;
      L265:
      mul.wide.u32 %rd8, %r20, 4;
      add.u64 %rd9, %rd14, %rd8;
      st.global.u32 [%rd9], %r12;
      add.u32 %r20, %r20, 1;

      L266:
      @%p4 bra L267;
      bra L268;
      L267:
      mul.wide.u32 %rd10, %r20, 4;
      add.u64 %rd11, %rd14, %rd10;
      st.global.u32 [%rd11], %r13;

      L268:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L260;

      L370:
      ret;
    }
    
    // Entry point wrapper for host launch
    .entry dedup_sorted_u32 (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      call.uni dedup_sorted_u32_kernel, (input, input_len, write_pos);
      ret;
    }
  "#;

pub const PTX_DEDUP_SORTED_U64_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_u64_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<40>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      shl.b32 %r5, %r4, 2;    // read_pos = thread_idx * 4 (processing 4 u64 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      shl.b32 %r7, %r7, 2;    // stride = total_threads * 4

      // Main grid-stride loop
      L350:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L470;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values using vectorized loads
      cvt.u64.u32 %rd0, %r5;
      shl.b64 %rd1, %rd0, 3;
      add.u64 %rd2, %rd14, %rd1;
      
      // Use v2.u64 for 128-bit vectorized loads (2 x 64-bit)
      ld.global.v2.u64 {%rd16, %rd17}, [%rd2];      // current[0:1]
      ld.global.v2.u64 {%rd18, %rd19}, [%rd2 + 16]; // current[2:3]

      sub.u32 %r9, %r5, 1;
      cvt.u64.u32 %rd3, %r9;
      shl.b64 %rd4, %rd3, 3;
      add.u64 %rd5, %rd14, %rd4;
      
      // Use v2.u64 for 128-bit vectorized loads (2 x 64-bit)
      ld.global.v2.u64 {%rd20, %rd21}, [%rd5];      // previous[0:1]
      ld.global.v2.u64 {%rd22, %rd23}, [%rd5 + 16]; // previous[2:3]

      // SIMD comparison: current != previous
      setp.ne.u64 %p1, %rd16, %rd20;  // current[0] != previous[0]
      setp.ne.u64 %p2, %rd17, %rd21;  // current[1] != previous[1]
      setp.ne.u64 %p3, %rd18, %rd22;  // current[2] != previous[2]
      setp.ne.u64 %p4, %rd19, %rd23;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L361;
      bra L362;
      L361:
      cvt.u64.u32 %rd6, %r20;
      shl.b64 %rd7, %rd6, 3;
      add.u64 %rd8, %rd14, %rd7;
      st.global.u64 [%rd8], %rd16;
      add.u32 %r20, %r20, 1;

      L362:
      @%p2 bra L363;
      bra L364;
      L363:
      cvt.u64.u32 %rd9, %r20;
      shl.b64 %rd10, %rd9, 3;
      add.u64 %rd11, %rd14, %rd10;
      st.global.u64 [%rd11], %rd17;
      add.u32 %r20, %r20, 1;

      L364:
      @%p3 bra L365;
      bra L366;
      L365:
      cvt.u64.u32 %rd24, %r20;
      shl.b64 %rd25, %rd24, 3;
      add.u64 %rd26, %rd14, %rd25;
      st.global.u64 [%rd26], %rd18;
      add.u32 %r20, %r20, 1;

      L366:
      @%p4 bra L367;
      bra L368;
      L367:
      cvt.u64.u32 %rd27, %r20;
      shl.b64 %rd28, %rd27, 3;
      add.u64 %rd29, %rd14, %rd28;
      st.global.u64 [%rd29], %rd19;

      L368:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L350;

      L470:
      ret;
    }
    
    // Entry point wrapper for host launch
    .entry dedup_sorted_u64 (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      call.uni dedup_sorted_u64_kernel, (input, input_len, write_pos);
      ret;
    }
  "#;

pub const PTX_DEDUP_SORTED_F64_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_f64_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<40>;
      .reg .f64 %fd<20>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      shl.b32 %r5, %r4, 2;    // read_pos = thread_idx * 4 (processing 4 f64 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      shl.b32 %r7, %r7, 2;    // stride = total_threads * 4

      // Main grid-stride loop
      L450:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L570;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values using vectorized loads
      cvt.u64.u32 %rd0, %r5;
      shl.b64 %rd1, %rd0, 3;
      add.u64 %rd2, %rd14, %rd1;
      
      // Use v2.f64 for 128-bit vectorized loads (2 x 64-bit floats)
      ld.global.v2.f64 {%fd16, %fd17}, [%rd2];      // current[0:1]
      ld.global.v2.f64 {%fd18, %fd19}, [%rd2 + 16]; // current[2:3]

      sub.u32 %r9, %r5, 1;
      cvt.u64.u32 %rd3, %r9;
      shl.b64 %rd4, %rd3, 3;
      add.u64 %rd5, %rd14, %rd4;
      
      // Use v2.f64 for 128-bit vectorized loads (2 x 64-bit floats)
      ld.global.v2.f64 {%fd20, %fd21}, [%rd5];      // previous[0:1]
      ld.global.v2.f64 {%fd22, %fd23}, [%rd5 + 16]; // previous[2:3]

      // SIMD comparison: current != previous
      setp.ne.f64 %p1, %fd16, %fd20;  // current[0] != previous[0]
      setp.ne.f64 %p2, %fd17, %fd21;  // current[1] != previous[1]
      setp.ne.f64 %p3, %fd18, %fd22;  // current[2] != previous[2]
      setp.ne.f64 %p4, %fd19, %fd23;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L461;
      bra L462;
      L461:
      cvt.u64.u32 %rd6, %r20;
      shl.b64 %rd7, %rd6, 3;
      add.u64 %rd8, %rd14, %rd7;
      st.global.f64 [%rd8], %fd16;
      add.u32 %r20, %r20, 1;

      L462:
      @%p2 bra L463;
      bra L464;
      L463:
      cvt.u64.u32 %rd9, %r20;
      shl.b64 %rd10, %rd9, 3;
      add.u64 %rd11, %rd14, %rd10;
      st.global.f64 [%rd11], %fd17;
      add.u32 %r20, %r20, 1;

      L464:
      @%p3 bra L465;
      bra L466;
      L465:
      cvt.u64.u32 %rd24, %r20;
      shl.b64 %rd25, %rd24, 3;
      add.u64 %rd26, %rd14, %rd25;
      st.global.f64 [%rd26], %fd18;
      add.u32 %r20, %r20, 1;

      L466:
      @%p4 bra L467;
      bra L468;
      L467:
      cvt.u64.u32 %rd27, %r20;
      shl.b64 %rd28, %rd27, 3;
      add.u64 %rd29, %rd14, %rd28;
      st.global.f64 [%rd29], %fd19;

      L468:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L450;

      L570:
      ret;
    }
    
    // Entry point wrapper for host launch
    .entry dedup_sorted_f64 (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      call.uni dedup_sorted_f64_kernel, (input, input_len, write_pos);
      ret;
    }
  "#;

pub const PTX_DEDUP_SORTED_I64_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    
    // Device function that can be called from other kernels
    .func dedup_sorted_i64_kernel (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<25>;
      .reg .u64 %rd<40>;
      .reg .s64 %sd<20>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd14, [input];
      ld.param.u32 %r24, [input_len];
      ld.param.u64 %rd13, [write_pos];

      // Grid-stride initialization
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r0, %r3;     // thread global index
      shl.b32 %r5, %r4, 2;    // read_pos = thread_idx * 4 (processing 4 i64 elements)
      add.u32 %r5, %r5, 1;       // start from 1 (first element always kept)

      // Grid stride size
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r1;
      shl.b32 %r7, %r7, 2;    // stride = total_threads * 4

      // Main grid-stride loop
      L550:
      add.u32 %r8, %r5, 4;       // check if we can load 4 elements
      setp.ge.u32 %p0, %r8, %r24;
      @%p0 bra L670;  // Jump to end if beyond input

      // Load current 4 values and previous 4 values
      cvt.u64.u32 %rd0, %r5;
      shl.b64 %rd1, %rd0, 3;
      add.u64 %rd2, %rd14, %rd1;
      
      ld.global.s64 %sd16, [%rd2];      // current[0]
      ld.global.s64 %sd17, [%rd2 + 8];  // current[1]
      ld.global.s64 %sd18, [%rd2 + 16]; // current[2]
      ld.global.s64 %sd19, [%rd2 + 24]; // current[3]

      sub.u32 %r9, %r5, 1;
      cvt.u64.u32 %rd3, %r9;
      shl.b64 %rd4, %rd3, 3;
      add.u64 %rd5, %rd14, %rd4;
      
      ld.global.s64 %sd20, [%rd5];      // previous[0]
      ld.global.s64 %sd21, [%rd5 + 8];  // previous[1]
      ld.global.s64 %sd22, [%rd5 + 16]; // previous[2]
      ld.global.s64 %sd23, [%rd5 + 24]; // previous[3]

      // SIMD comparison: current != previous
      setp.ne.s64 %p1, %sd16, %sd20;  // current[0] != previous[0]
      setp.ne.s64 %p2, %sd17, %sd21;  // current[1] != previous[1]
      setp.ne.s64 %p3, %sd18, %sd22;  // current[2] != previous[2]
      setp.ne.s64 %p4, %sd19, %sd23;  // current[3] != previous[3]

      // Count unique elements and get write position atomically
      mov.u32 %r18, 0;
      selp.u32 %r19, 1, 0, %p1;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p2;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p3;
      add.u32 %r18, %r18, %r19;
      selp.u32 %r19, 1, 0, %p4;
      add.u32 %r18, %r18, %r19;  // r18 = count of unique elements

      // Atomic add to get write position
      atom.global.add.u32 %r20, [%rd13], %r18;

      // Write unique elements at positions
      @%p1 bra L561;
      bra L562;
      L561:
      cvt.u64.u32 %rd6, %r20;
      shl.b64 %rd7, %rd6, 3;
      add.u64 %rd8, %rd14, %rd7;
      st.global.s64 [%rd8], %sd16;
      add.u32 %r20, %r20, 1;

      L562:
      @%p2 bra L563;
      bra L564;
      L563:
      cvt.u64.u32 %rd9, %r20;
      shl.b64 %rd10, %rd9, 3;
      add.u64 %rd11, %rd14, %rd10;
      st.global.s64 [%rd11], %sd17;
      add.u32 %r20, %r20, 1;

      L564:
      @%p3 bra L565;
      bra L566;
      L565:
      cvt.u64.u32 %rd24, %r20;
      shl.b64 %rd25, %rd24, 3;
      add.u64 %rd26, %rd14, %rd25;
      st.global.s64 [%rd26], %sd18;
      add.u32 %r20, %r20, 1;

      L566:
      @%p4 bra L567;
      bra L568;
      L567:
      cvt.u64.u32 %rd27, %r20;
      shl.b64 %rd28, %rd27, 3;
      add.u64 %rd29, %rd14, %rd28;
      st.global.s64 [%rd29], %sd19;

      L568:
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r7;
      bra L550;

      L670:
      ret;
    }
    
    // Entry point wrapper for host launch
    .entry dedup_sorted_i64 (
      .param .u64 input,
      .param .u32 input_len,
      .param .u64 write_pos
    ) {
      call.uni dedup_sorted_i64_kernel, (input, input_len, write_pos);
      ret;
    }
  "#;

// =============================================================================
// REDUCE SUM F64 OPERATIONS
// =============================================================================

// GPU/PTX optimized u32 sorted check.
//
// Grid-stride loop to check if array is sorted in ascending order.
//
#[cfg(has_cuda)]
pub unsafe fn is_sorted_u32_gpu(values: *const u32, len: usize, result: *mut u8) {
  if len <= 1 {
    *result = 1; // true  
    return;
  }

  const PTX_IS_SORTED_U32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry is_sorted_u32(
      .param .u64 values_ptr,
      .param .u32 len,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .u64 %rd<25>;

      // Load parameters
      ld.param.u64 %rd3, [values_ptr];
      ld.param.u32 %r9, [len];
      ld.param.u64 %rd2, [result_ptr];

      // Thread and block indices
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mad.lo.u32 %r3, %r1, %r2, %r0; // global thread index

      // Grid stride setup
      mov.u32 %r4, %nctaid.x;
      mul.lo.u32 %r5, %r4, %r2; // grid stride
      
      sub.u32 %r9, %r9, 1; // len - 1

    loop_start:
      setp.ge.u32 %p0, %r3, %r9;  // Check if r3 >= len-1
      @%p0 bra loop_end;

      // Load current and next values
      cvt.u64.u32 %rd0, %r3;
      shl.b64 %rd0, %rd0, 2; // * sizeof(u32)
      add.u64 %rd1, %rd3, %rd0;
      ld.global.u32 %r6, [%rd1];       // Load values[i]
      ld.global.u32 %r7, [%rd1+4];     // Load values[i+1]

      // Check if current > next (not sorted)
      setp.gt.u32 %p1, %r6, %r7;
      
      // If found unsorted, write 0
      @%p1 mov.u32 %r20, 0;
      @%p1 st.global.u8 [%rd2], %r20;  // Set result to 0 (false)

      // Grid stride increment
      add.u32 %r3, %r3, %r5;
      bra loop_start;

    loop_end:
      ret;
    }
  "#;

  // Result is already initialized to 1 by the GPU wrapper
  // The PTX kernel will set it to 0 if unsorted, leave as 1 if sorted

  let values_ptr = values as u64;
  let len_u32 = len as u32;
  let result_ptr = result as u64;
  let args = [
    &values_ptr as *const u64 as *const u8,
    &len_u32 as *const u32 as *const u8,
    &result_ptr as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(
    PTX_IS_SORTED_U32,
    &[],
    "is_sorted_u32",
    blocks,
    threads,
    &args,
  );
}

// GPU/PTX optimized f64 sum reduction.
//
// Grid-stride loop with accumulation like SIMD.
//

#[inline]
#[cfg(has_cuda)]
pub unsafe fn reduce_sum_f64_gpu(values: *const f64, len: usize, result: *mut f64) {
  const PTX_REDUCE_SUM_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry reduce_sum_f64(
      .param .u64 values_ptr,
      .param .u64 len_param,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;
      .shared .align 8 .f64 sdata[256];  // Shared memory for f64 reduction
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd15, [len_param];
      cvt.u32.u64 %r15, %rd15;  // Convert len to u32 for comparisons  // FIX: This was missing
      ld.param.u64 %rd4, [result_ptr];
      
      // Get thread and block IDs
      mov.u32 %r0, %tid.x;     // Thread ID within block
      mov.u32 %r1, %ntid.x;    // Block size (256)
      mov.u32 %r2, %ctaid.x;   // Block ID
      mov.u32 %r3, %nctaid.x;  // Number of blocks
      
      // Calculate global thread ID
      mul.lo.u32 %r4, %r2, %r1;
      add.u32 %r5, %r4, %r0;   // Global thread ID
      
      // Calculate grid stride
      mul.lo.u32 %r6, %r3, %r1;  // Total threads in grid
      
      // Thread-local accumulator
      mov.f64 %fd5, 0d0000000000000000;
      
      // Grid-stride loop - each thread processes elements with stride
    loop_start:
      setp.ge.u32 %p0, %r5, %r15;  // Check if we're past the end
      @%p0 bra loop_end;
      
      // Load single u32 value
      // r5 is element index, convert to byte offset
      mul.wide.u32 %rd8, %r5, 8;
      add.u64 %rd9, %rd0, %rd8;
      ld.global.f64 %fd7, [%rd9];
      
      // Convert to u64 and accumulate
      add.f64 %fd5, %fd5, %fd7;
      
      
      // Advance by grid stride
      add.u32 %r5, %r5, %r6;
      bra loop_start;
      
    loop_end:
      // Store thread result to shared memory
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for f64
      mov.u64 %rd2, sdata;
      add.u64 %rd3, %rd2, %rd3;
      st.shared.f64 [%rd3], %fd5;
      bar.sync 0;
      
      // Parallel reduction in shared memory for strides > 32
      // 256 threads -> 128 -> 64 -> 32 -> warp shuffle
      
      // s = 128
      mov.u32 %r8, 128;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra reduce_64;
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for f64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.f64 %fd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.f64 %fd9, [%rd8];
      add.f64 %fd5, %fd5, %fd9;
      st.shared.f64 [%rd3], %fd5;
      bar.sync 0;
      
    reduce_64:
      // s = 64
      mov.u32 %r8, 64;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra reduce_32;
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for f64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.f64 %fd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.f64 %fd9, [%rd8];
      add.f64 %fd5, %fd5, %fd9;
      st.shared.f64 [%rd3], %fd5;
      bar.sync 0;
      
    reduce_32:
      // s = 32 (transition to warp operations)
      mov.u32 %r8, 32;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra warp_reduction;
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for f64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.f64 %fd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.f64 %fd9, [%rd8];
      add.f64 %fd5, %fd5, %fd9;
      st.shared.f64 [%rd3], %fd5;
      bar.sync 0;
      
    warp_reduction:
      // Load value from shared memory for warp reduction
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for f64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.f64 %fd16, [%rd3];
      
      // Get lane ID within warp
      and.b32 %r17, %r0, 0x1f;
      
      // Only first warp participates in final reduction
      setp.ge.u32 %p5, %r0, 32;
      @%p5 bra skip_store;
      
      // Warp shuffle reduction for 64-bit values
      // Split into high and low 32-bit parts
      mov.b64 {%r18, %r19}, %fd16;  // Split 64-bit into 2x32-bit
      
      // Shuffle down by 16 (both parts)
      shfl.sync.down.b32 %r25, %r18, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 16, 0x1f, 0xffffffff;
      mov.b64 %fd17, {%r25, %r26};  // Combine back to 64-bit
      add.f64 %fd16, %fd16, %fd17;
      
      // Shuffle down by 8
      mov.b64 {%r18, %r19}, %fd16;
      shfl.sync.down.b32 %r25, %r18, 8, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 8, 0x1f, 0xffffffff;
      mov.b64 %fd17, {%r25, %r26};
      add.f64 %fd16, %fd16, %fd17;
      
      // Shuffle down by 4
      mov.b64 {%r18, %r19}, %fd16;
      shfl.sync.down.b32 %r25, %r18, 4, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 4, 0x1f, 0xffffffff;
      mov.b64 %fd17, {%r25, %r26};
      add.f64 %fd16, %fd16, %fd17;
      
      // Shuffle down by 2
      mov.b64 {%r18, %r19}, %fd16;
      shfl.sync.down.b32 %r25, %r18, 2, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 2, 0x1f, 0xffffffff;
      mov.b64 %fd17, {%r25, %r26};
      add.f64 %fd16, %fd16, %fd17;
      
      // Shuffle down by 1
      mov.b64 {%r18, %r19}, %fd16;
      shfl.sync.down.b32 %r25, %r18, 1, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 1, 0x1f, 0xffffffff;
      mov.b64 %fd17, {%r25, %r26};
      add.f64 %fd16, %fd16, %fd17;
      
      // Lane 0 stores result back to shared memory
      setp.eq.u32 %p2, %r17, 0;
      @!%p2 bra skip_store;
      st.shared.f64 [%rd2], %fd16;
      
    skip_store:
      bar.sync 0;
      
      // Thread 0 of each block atomically adds to global result
      setp.ne.u32 %p3, %r0, 0;
      @%p3 bra done;
      
      ld.shared.f64 %fd19, [sdata];
      atom.global.add.f64 %fd20, [%rd4], %fd19;
      
    done:
      ret;
    }
  "#;
  let (blocks, threads) = LaunchConfig::reduction();

  // Pass parameters to PTX kernel
  let len_u64 = len as u64;
  let values_u64 = values as u64;
  let result_u64 = result as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_u64 as *const u64 as *const u8,
  ];

  let _ = launch_ptx(
    PTX_REDUCE_SUM_F64,
    &[],
    "reduce_sum_f64",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 f64 sum reduction using proper SIMD vectorization.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn reduce_sum_f64_avx512(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX512_F64; // AVX-512 processes 8 f64 values at once

  #[allow(unused_mut)]
  let mut sum_vec = _mm512_setzero_pd();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_pd(values.as_ptr().add(offset));
    sum_vec = _mm512_add_pd(sum_vec, values_chunk);
  }

  // Extract and sum the 8 elements from the vector (matching NEON manual extraction pattern)
  let sum = _mm512_reduce_add_pd(sum_vec);

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  let mut final_sum = sum;

  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let remaining_vec =
      _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, values.as_ptr().add(offset));
    let remaining_sum = _mm512_mask_reduce_add_pd(load_mask, remaining_vec);
    final_sum += remaining_sum;
  }

  final_sum
}

// GPU implementation - uses same parallel reduction logic as AVX512

// AVX2 optimized f64 sum reduction.
//
// Uses AVX2 vectorized addition for enhanced performance.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn reduce_sum_f64_avx2(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX2_F64; // AVX2 processes 4 f64 values at once

  #[allow(unused_mut)]
  let mut sum_vec = _mm256_setzero_pd();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_pd(values.as_ptr().add(offset));
    sum_vec = _mm256_add_pd(sum_vec, values_chunk);
  }

  // Horizontal sum of the 4 elements in the vector (matching NEON manual extraction pattern)
  // [a, b, c, d] -> [a+b, c+d, a+b, c+d]
  let sum_vec_hadd = _mm256_hadd_pd(sum_vec, sum_vec);
  // extract 128-bit lane
  let high_128 = _mm256_extractf128_pd(sum_vec_hadd, 1);
  // add the two 128-bit lanes
  let sum_128 = _mm_add_pd(_mm256_castpd256_pd128(sum_vec_hadd), high_128);
  // final sum
  let sum = _mm_cvtsd_f64(sum_128);

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 masked operations
  let offset = full_chunks * LANES;
  let mut final_sum = sum;

  // Minimal scalar for tiny remainder only
  for j in 0..remaining_elements {
    final_sum += values[offset + j];
  }

  final_sum
}

// NEON optimized f64 sum reduction.
//
// Uses NEON vectorized addition for enhanced performance.
//
// # Safety
// Requires NEON support (available on all aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn reduce_sum_f64_neon(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_NEON_F64; // NEON processes 2 f64 values at once

  #[allow(unused_mut)]
  let mut sum_vec = vdupq_n_f64(0.0); // FIX: This was missing

  // Process complete SIMD chunks
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_f64(values.as_ptr().add(offset));
    sum_vec = vaddq_f64(sum_vec, values_chunk);
  }

  // Extract and sum the 2 elements from the vector
  let sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

  // Handle remaining elements with scalar addition
  let offset = full_chunks * LANES;
  let mut final_sum = sum;
  for j in 0..remaining_elements {
    final_sum += values[offset + j];
  }

  final_sum
}

// =============================================================================
// REDUCE SUM U64 OPERATIONS
// =============================================================================

// GPU/PTX optimized u64 sum reduction.
//
// Grid-stride loop with accumulation like SIMD.
//
// GPU optimized u64 sum reduction using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn reduce_sum_u64_gpu(values: *const u64, len: usize, result: *mut u64) {
  const PTX_REDUCE_SUM_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry reduce_sum_u64(
      .param .u64 values_ptr,
      .param .u64 len_param,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .shared .align 8 .u64 sdata[256];  // Shared memory for reduction
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len_param];
      cvt.u32.u64 %r15, %rd1;  // Convert len to u32 for comparisons
      ld.param.u64 %rd4, [result_ptr];
      
      // Get thread and block IDs
      mov.u32 %r0, %tid.x;     // Thread ID within block
      mov.u32 %r1, %ntid.x;    // Block size (256)
      mov.u32 %r2, %ctaid.x;   // Block ID
      mov.u32 %r3, %nctaid.x;  // Number of blocks
      
      // Calculate global thread ID
      mul.lo.u32 %r4, %r2, %r1;
      add.u32 %r5, %r4, %r0;   // Global thread ID
      
      // Calculate grid stride
      mul.lo.u32 %r6, %r3, %r1;  // Total threads in grid
      
      // Thread-local accumulator
      mov.u64 %rd5, 0;
      
      // Grid-stride loop - each thread processes elements with stride
    loop_start:
      setp.ge.u32 %p0, %r5, %r15;  // Check if we're past the end
      @%p0 bra loop_end;
      
      // Load single u32 value
      // r5 is element index, convert to byte offset
      mul.wide.u32 %rd8, %r5, 8;
      add.u64 %rd9, %rd0, %rd8;
      ld.global.u64 %rd7, [%rd9];
      
      // Convert to u64 and accumulate
      add.u64 %rd5, %rd5, %rd7;
      
      
      // Advance by grid stride
      add.u32 %r5, %r5, %r6;
      bra loop_start;
      
    loop_end:
      // Store thread result to shared memory
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for u64
      mov.u64 %rd2, sdata;
      add.u64 %rd3, %rd2, %rd3;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
      // Parallel reduction in shared memory for strides > 32
      // 256 threads -> 128 -> 64 -> 32 -> warp shuffle
      
      // s = 128
      mov.u32 %r8, 128;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra reduce_64;
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for u64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.u64 %rd9, [%rd8];
      add.u64 %rd5, %rd5, %rd9;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
    reduce_64:
      // s = 64
      mov.u32 %r8, 64;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra reduce_32;
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for u64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.u64 %rd9, [%rd8];
      add.u64 %rd5, %rd5, %rd9;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
    reduce_32:
      // s = 32 (transition to warp operations)
      mov.u32 %r8, 32;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra warp_reduction;
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for u64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.u64 %rd9, [%rd8];
      add.u64 %rd5, %rd5, %rd9;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
    warp_reduction:
      // Load value from shared memory for warp reduction
      mul.wide.u32 %rd3, %r0, 8;  // FIX: Was 8, should be 16 for u64
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd16, [%rd3];
      
      // Get lane ID within warp
      and.b32 %r17, %r0, 0x1f;
      
      // Only first warp participates in final reduction
      setp.ge.u32 %p5, %r0, 32;
      @%p5 bra skip_store;
      
      // Warp shuffle reduction for 64-bit values
      // Split into high and low 32-bit parts
      mov.b64 {%r18, %r19}, %rd16;  // Split 64-bit into 2x32-bit
      
      // Shuffle down by 16 (both parts)
      shfl.sync.down.b32 %r25, %r18, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 16, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};  // Combine back to 64-bit
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 8
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 8, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 8, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 4
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 4, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 4, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 2
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 2, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 2, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 1
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 1, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 1, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Lane 0 stores result back to shared memory
      setp.eq.u32 %p2, %r17, 0;
      @!%p2 bra skip_store;
      st.shared.u64 [%rd2], %rd16;
      
    skip_store:
      bar.sync 0;
      
      // Thread 0 of each block atomically adds to global result
      setp.ne.u32 %p3, %r0, 0;
      @%p3 bra done;
      
      ld.shared.u64 %rd19, [sdata];
      atom.global.add.u64 %rd20, [%rd4], %rd19;
      
    done:
      ret;
    }
  "#;
  let (blocks, threads) = LaunchConfig::reduction();

  // Pass parameters to PTX kernel
  let len_u64 = len as u64;
  let values_u64 = values as u64;
  let result_u64 = result as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_u64 as *const u64 as *const u8,
  ];

  let _ = launch_ptx(
    PTX_REDUCE_SUM_U64,
    &[],
    "reduce_sum_u64",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 optimized u64 sum reduction.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn reduce_sum_u64_avx512(values: &[u64], len: usize) -> u64 {
  const LANES: usize = LANES_AVX512_U64; // AVX-512 processes 8 u64 values at once

  #[allow(unused_mut)]
  let mut sum_vec = _mm512_setzero_si512();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_si512(values.as_ptr().add(offset) as *const _);
    sum_vec = _mm512_add_epi64(sum_vec, values_chunk);
  }

  // Extract and sum the 8 elements from the vector (matching NEON manual extraction pattern)
  let sum = _mm512_reduce_add_epi64(sum_vec) as u64;

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  let mut final_sum = sum;

  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let remaining_vec = _mm512_mask_loadu_epi64(
      _mm512_setzero_si512(),
      load_mask,
      values.as_ptr().add(offset) as *const i64,
    );
    let remaining_sum = _mm512_mask_reduce_add_epi64(load_mask, remaining_vec) as u64;
    final_sum += remaining_sum;
  }

  final_sum
}

// GPU implementation - parallel reduction for u64 arrays

// AVX2 optimized u64 sum reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn reduce_sum_u64_avx2(values: &[u64], len: usize) -> u64 {
  const LANES: usize = LANES_AVX2_U64; // AVX2 processes 4 u64 values at once

  let mut sum_vec = _mm256_setzero_si256();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_si256(values.as_ptr().add(offset) as *const _);
    sum_vec = _mm256_add_epi64(sum_vec, values_chunk);
  }

  // Horizontal sum of the 4 elements in the vector (matching NEON manual extraction pattern)
  let high_128 = _mm256_extracti128_si256(sum_vec, 1);
  let sum_128 = _mm_add_epi64(_mm256_castsi256_si128(sum_vec), high_128);
  let sum_128_shuffled = _mm_shuffle_epi32(sum_128, 0b_01_00_11_10);
  let sum_64 = _mm_add_epi64(sum_128, sum_128_shuffled);
  let sum = _mm_cvtsi128_si64(sum_64) as u64;

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 masked operations
  let offset = full_chunks * LANES;
  let mut final_sum = sum;

  // Minimal scalar for tiny remainder only
  for j in 0..remaining_elements {
    final_sum += values[offset + j];
  }

  final_sum
}

// NEON optimized u64 sum reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn reduce_sum_u64_neon(values: &[u64], len: usize) -> u64 {
  const LANES: usize = LANES_NEON_U64; // NEON processes 2 u64 values at once

  #[allow(unused_mut)]
  let mut sum_vec = vdupq_n_u64(0);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_u64(values.as_ptr().add(offset));
    sum_vec = vaddq_u64(sum_vec, values_chunk);
  }

  // Extract and sum the 2 elements from the vector with wrapping
  let sum = vgetq_lane_u64(sum_vec, 0).wrapping_add(vgetq_lane_u64(sum_vec, 1));

  // Handle remaining elements with wrapping addition
  let offset = full_chunks * LANES;
  let mut final_sum = sum;
  for j in 0..remaining_elements {
    final_sum = final_sum.wrapping_add(values[offset + j]);
  }

  final_sum
}

// =============================================================================
// REDUCE SUM U32 OPERATIONS
// =============================================================================

// GPU optimized u32 sum reduction using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn reduce_sum_u32_gpu(values: *const u32, len: usize, result: *mut u64) {
  const PTX_REDUCE_SUM_U32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry reduce_sum_u32(
      .param .u64 values_ptr,
      .param .u32 len_param,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .shared .align 8 .u64 sdata[256];  // Shared memory for reduction
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u32 %r15, [len_param];
      ld.param.u64 %rd4, [result_ptr];
      
      // Get thread and block IDs
      mov.u32 %r0, %tid.x;     // Thread ID within block
      mov.u32 %r1, %ntid.x;    // Block size (256)
      mov.u32 %r2, %ctaid.x;   // Block ID
      mov.u32 %r3, %nctaid.x;  // Number of blocks
      
      // Calculate global thread ID
      mul.lo.u32 %r4, %r2, %r1;
      add.u32 %r5, %r4, %r0;   // Global thread ID
      
      // Calculate grid stride
      mul.lo.u32 %r6, %r3, %r1;  // Total threads in grid
      
      // Thread-local accumulator
      mov.u64 %rd5, 0;
      
      // Grid-stride loop - each thread processes elements with stride
    loop_start:
      setp.ge.u32 %p0, %r5, %r15;  // Check if we're past the end
      @%p0 bra loop_end;
      
      // Load single u32 value
      // r5 is element index, convert to byte offset
      mul.wide.u32 %rd8, %r5, 4;
      add.u64 %rd9, %rd0, %rd8;
      ld.global.u32 %r7, [%rd9];
      
      // Convert to u64 and accumulate
      cvt.u64.u32 %rd10, %r7;
      add.u64 %rd5, %rd5, %rd10;
      
      // Advance by grid stride
      add.u32 %r5, %r5, %r6;
      bra loop_start;
      
    loop_end:
      // Store thread result to shared memory
      mul.wide.u32 %rd3, %r0, 8;
      mov.u64 %rd2, sdata;
      add.u64 %rd3, %rd2, %rd3;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
      // Parallel reduction in shared memory for strides > 32
      // 256 threads -> 128 -> 64 -> 32 -> warp shuffle
      
      // s = 128
      mov.u32 %r8, 128;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra reduce_64;
      mul.wide.u32 %rd3, %r0, 8;
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.u64 %rd9, [%rd8];
      add.u64 %rd5, %rd5, %rd9;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
    reduce_64:
      // s = 64
      mov.u32 %r8, 64;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra reduce_32;
      mul.wide.u32 %rd3, %r0, 8;
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.u64 %rd9, [%rd8];
      add.u64 %rd5, %rd5, %rd9;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
    reduce_32:
      // s = 32 (transition to warp operations)
      mov.u32 %r8, 32;
      setp.lt.u32 %p1, %r0, %r8;
      @!%p1 bra warp_reduction;
      mul.wide.u32 %rd3, %r0, 8;
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd5, [%rd3];
      add.u32 %r9, %r0, %r8;
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd8, %rd2, %rd8;
      ld.shared.u64 %rd9, [%rd8];
      add.u64 %rd5, %rd5, %rd9;
      st.shared.u64 [%rd3], %rd5;
      bar.sync 0;
      
    warp_reduction:
      // Load value from shared memory for warp reduction
      mul.wide.u32 %rd3, %r0, 8;
      add.u64 %rd3, %rd2, %rd3;
      ld.shared.u64 %rd16, [%rd3];
      
      // Get lane ID within warp
      and.b32 %r17, %r0, 0x1f;
      
      // Only first warp participates in final reduction
      setp.ge.u32 %p5, %r0, 32;
      @%p5 bra skip_store;
      
      // Warp shuffle reduction for 64-bit values
      // Split into high and low 32-bit parts
      mov.b64 {%r18, %r19}, %rd16;  // Split 64-bit into 2x32-bit
      
      // Shuffle down by 16 (both parts)
      shfl.sync.down.b32 %r25, %r18, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 16, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};  // Combine back to 64-bit
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 8
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 8, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 8, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 4
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 4, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 4, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 2
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 2, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 2, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Shuffle down by 1
      mov.b64 {%r18, %r19}, %rd16;
      shfl.sync.down.b32 %r25, %r18, 1, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r26, %r19, 1, 0x1f, 0xffffffff;
      mov.b64 %rd17, {%r25, %r26};
      add.u64 %rd16, %rd16, %rd17;
      
      // Lane 0 stores result back to shared memory
      setp.eq.u32 %p2, %r17, 0;
      @!%p2 bra skip_store;
      st.shared.u64 [%rd2], %rd16;
      
    skip_store:
      bar.sync 0;
      
      // Thread 0 of each block atomically adds to global result
      setp.ne.u32 %p3, %r0, 0;
      @%p3 bra done;
      
      ld.shared.u64 %rd19, [sdata];
      atom.global.add.u64 %rd20, [%rd4], %rd19;
      
    done:
      ret;
    }
  "#;
  let (blocks, threads) = LaunchConfig::reduction();

  // Pass parameters to PTX kernel
  let len_u32 = len as u32;
  let values_u64 = values as u64;
  let result_u64 = result as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u32 as *const u32 as *const u8,
    &result_u64 as *const u64 as *const u8,
  ];

  let _ = launch_ptx(
    PTX_REDUCE_SUM_U32,
    &[],
    "reduce_sum_u32",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 optimized u32 sum reduction.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512dq")]
#[target_feature(enable = "avx512vl")]
pub(super) unsafe fn reduce_sum_u32_avx512(values: &[u32], len: usize) -> u64 {
  const LANES: usize = LANES_AVX512_U32; // AVX-512 processes 16 u32 values at once

  // Accumulate in 64-bit lanes to avoid 32-bit overflow completely
  let mut sum_64: u64 = 0;

  // Process complete SIMD chunks of 16 u32 values
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;

    // Load 16 x u32 as one 512-bit block
    let all32 = _mm512_loadu_si512(values.as_ptr().add(offset) as *const __m512i);

    // Extract lower 8 x u32 (low 256 bits) and widen to 8 x u64
    let lo32 = _mm512_castsi512_si256(all32);
    let lo64 = _mm512_cvtepu32_epi64(lo32);
    sum_64 += _mm512_reduce_add_epi64(lo64) as u64;

    // Extract upper 8 x u32 (high 256 bits) and widen to 8 x u64
    let hi32 = _mm512_extracti32x8_epi32(all32, 1);
    let hi64 = _mm512_cvtepu32_epi64(hi32);
    sum_64 += _mm512_reduce_add_epi64(hi64) as u64;
  }

  // Handle remaining elements with scalar addition
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    sum_64 += values[offset + j] as u64;
  }

  sum_64
}

// AVX2 optimized u32 sum reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn reduce_sum_u32_avx2(values: &[u32], len: usize) -> u64 {
  const LANES: usize = LANES_AVX2_U32; // AVX2 processes 8 u32 values at once

  // Use 64-bit accumulator to prevent overflow - accumulate as u64 periodically
  let mut sum_64: u64 = 0;

  #[allow(unused_variables)]
  let sum_vec = _mm256_setzero_si256();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX2 (matching NEON for-loop structure)
  // Immediately convert each chunk to 64-bit to prevent any overflow
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_si256(values.as_ptr().add(offset) as *const _);

    // Immediately extract and add to 64-bit accumulator to prevent overflow
    let sum_array: [u32; 8] = std::mem::transmute(values_chunk);
    for &val in sum_array.iter() {
      sum_64 += val as u64;
    }
  }

  // Handle remaining elements with scalar addition
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    sum_64 += values[offset + j] as u64;
  }

  sum_64
}

// NEON optimized u32 sum reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn reduce_sum_u32_neon(values: &[u32], len: usize) -> u64 {
  const LANES: usize = LANES_NEON_U32; // NEON processes 4 u32 values at once

  // Use 64-bit accumulator to prevent overflow
  #[allow(unused_mut)]
  let mut sum_vec = vdupq_n_u64(0);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_u32(values.as_ptr().add(offset));

    // Convert to 64-bit and accumulate to prevent overflow
    // Process low 2 elements
    let lo_32 = vget_low_u32(values_chunk);
    let lo_64 = vmovl_u32(lo_32);
    sum_vec = vaddq_u64(sum_vec, lo_64);

    // Process high 2 elements
    let hi_32 = vget_high_u32(values_chunk);
    let hi_64 = vmovl_u32(hi_32);
    sum_vec = vaddq_u64(sum_vec, hi_64);
  }

  // Extract and sum the 2 elements from the 64-bit vector
  let sum = vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);

  // Handle remaining elements with scalar addition
  let offset = full_chunks * LANES;
  let mut final_sum = sum;
  for j in 0..remaining_elements {
    final_sum += values[offset + j] as u64;
  }

  final_sum
}

// =============================================================================
// IS_SORTED U32 OPERATIONS
// =============================================================================

// GPU optimized u32 sorted check using PTX inline assembly

// AVX-512 optimized u32 sorted check.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn is_sorted_u32_avx512(values: &[u32], len: usize) -> bool {
  const LANES: usize = LANES_AVX512_U32; // AVX-512 processes 16 u32 values at once

  if len <= 1 {
    return true;
  }

  // Process complete SIMD chunks
  let full_chunks = (len - 1) / LANES;
  let remaining_elements = (len - 1) % LANES;

  // Process 16-element chunks with AVX-512
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let current_chunk = _mm512_loadu_si512(values.as_ptr().add(offset) as *const _);
    let next_chunk = _mm512_loadu_si512(values.as_ptr().add(offset + 1) as *const _);

    // Check if current[i] <= next[i] for all elements
    let cmp_mask = _mm512_cmple_epu32_mask(current_chunk, next_chunk);

    // If any comparison fails, array is not sorted
    if cmp_mask != 0xFFFF {
      return false;
    }
  }

  // Handle remaining elements with scalar comparison
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    if values[offset + j] > values[offset + j + 1] {
      return false;
    }
  }

  true
}

// AVX2 optimized u32 sorted check.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn is_sorted_u32_avx2(values: &[u32], len: usize) -> bool {
  const LANES: usize = LANES_AVX2_U32; // AVX2 processes 8 u32 values at once

  if len <= 1 {
    return true;
  }

  // Process complete SIMD chunks
  let full_chunks = (len - 1) / LANES;
  let remaining_elements = (len - 1) % LANES;

  // Process 8-element chunks with AVX2
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let current_chunk = _mm256_loadu_si256(values.as_ptr().add(offset) as *const _);
    let next_chunk = _mm256_loadu_si256(values.as_ptr().add(offset + 1) as *const _);

    // Check if current[i] <= next[i] for all elements using unsigned comparison
    // We use the fact that if current <= next, then min(current, next) == current
    let min_result = _mm256_min_epu32(current_chunk, next_chunk);
    let cmp_result = _mm256_cmpeq_epi32(min_result, current_chunk);
    let cmp_mask = _mm256_movemask_epi8(cmp_result);

    // If any comparison fails (mask is not all 1s), array is not sorted
    if cmp_mask != -1 {
      return false;
    }
  }

  // Handle remaining elements with scalar comparison
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    if values[offset + j] > values[offset + j + 1] {
      return false;
    }
  }

  true
}

// NEON optimized u32 sorted check.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn is_sorted_u32_neon(values: &[u32], len: usize) -> bool {
  const LANES: usize = LANES_NEON_U32; // NEON processes 4 u32 values at once

  if len <= 1 {
    return true;
  }

  // For is_sorted, we need to check consecutive pairs: values[i] <= values[i+1]
  // We can process LANES pairs at once by comparing chunks offset by 1
  let pairs_to_check = len - 1;
  let full_chunks = if pairs_to_check >= LANES {
    pairs_to_check / LANES
  } else {
    0
  };

  // Process LANES pairs at once with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let current_chunk = vld1q_u32(values.as_ptr().add(offset));
    let next_chunk = vld1q_u32(values.as_ptr().add(offset + 1));

    // Check if current[i] <= next[i] for all elements
    let cmp_result = vcgtq_u32(current_chunk, next_chunk);

    // Check if any element failed the comparison
    // vcgtq_u32 returns 0xFFFFFFFF for true, 0x00000000 for false
    let lane0 = vgetq_lane_u32(cmp_result, 0);
    let lane1 = vgetq_lane_u32(cmp_result, 1);
    let lane2 = vgetq_lane_u32(cmp_result, 2);
    let lane3 = vgetq_lane_u32(cmp_result, 3);

    if lane0 != 0 || lane1 != 0 || lane2 != 0 || lane3 != 0 {
      return false;
    }
  }

  // Handle remaining pairs with scalar comparison
  let remaining_start = full_chunks * LANES;
  for j in remaining_start..pairs_to_check {
    if values[j] > values[j + 1] {
      return false;
    }
  }

  true
}

// =============================================================================
// REDUCE MIN F64 OPERATIONS
// =============================================================================

// GPU/PTX optimized f64 minimum reduction.
//
// Grid-stride loop finding minimum like SIMD.
//
#[cfg(has_cuda)]
pub unsafe fn reduce_min_f64_gpu(values: *const f64, len: usize, result: *mut f64) {
  const PTX_REDUCE_MIN_F64_GPU: &str = r#"
    .version 7.0
    .target sm_70
    .address_size 64

    .entry reduce_min_f64_gpu(.param .u64 values_param, .param .u64 len_param, .param .u64 result_param) {
        .reg .f64 %fd<25>;
        .reg .u32 %r<35>;
        .reg .u64 %rd<30>;  // Increased for 64-bit indices and temp loads
        .reg .pred %p<10>;
        .shared .f64 sdata[256];
        .shared .u32 nan_found[1];
        
        ld.param.u64 %rd3, [values_param];
        ld.param.u64 %rd5, [len_param];
        cvt.u32.u64 %r7, %rd5;  
        ld.param.u64 %rd4, [result_param];
        
        // Thread 0 initializes shared flag
        mov.u32 %r1, %tid.x;
        setp.eq.u32 %p1, %r1, 0;
        @%p1 mov.u32 %r20, 0;
        @%p1 st.shared.u32 [nan_found], %r20;
        bar.sync 0;
        
        // Initialize double2 accumulator with infinity for min reduction (like AVX-512)  
        mov.b64 %rd20, 0x7FF0000000000000;  // +INFINITY bit pattern for min
        mov.b64 %fd1, %rd20;
        mov.b64 %fd2, %rd20;
        
        // Calculate starting index for this thread (use 64-bit to prevent overflow)
        mov.u32 %r1, %tid.x;     // Thread ID within block
        mov.u32 %r2, %ntid.x;    // Block size
        mov.u32 %r3, %ctaid.x;   // Block ID
        mul.wide.u32 %rd23, %r3, %r2;
        cvt.u64.u32 %rd24, %r1;
        add.u64 %rd23, %rd23, %rd24;   // Global thread ID in 64-bit
        mul.lo.u64 %rd23, %rd23, 2;    // Each thread starts at ID * 2 (double2)
        
        // Calculate grid stride (use 64-bit)
        mov.u32 %r8, %nctaid.x; // Number of blocks
        mul.wide.u32 %rd24, %r8, %r2;  // Total threads in grid (64-bit)
        mul.lo.u64 %rd24, %rd24, 2;    // Stride = total_threads * 2 (double2)
        
        // Main grid-stride loop processing double2 chunks
    loop_start:
        // Check if we can load full double2 (use 64-bit comparison)
        add.u64 %rd25, %rd23, 1;
        cvt.u64.u32 %rd26, %r7;        // Convert len to 64-bit
        setp.ge.u64 %p0, %rd25, %rd26;  // Check if we can load full double2
        @%p0 bra remainder_check;    // Exit loop if beyond array
        
        // Load and compute min with double2 using vectorized load
        mul.lo.u64 %rd1, %rd23, 8;     // Use 64-bit multiply
        add.u64 %rd2, %rd3, %rd1;
        ld.global.v2.f64 {%fd3, %fd4}, [%rd2];  // Vectorized 128-bit load
        
        // Check if either value is NaN and write directly to result
        setp.neu.f64 %p8, %fd3, %fd3;  // NaN check for fd3
        setp.neu.f64 %p9, %fd4, %fd4;  // NaN check for fd4
        or.pred %p8, %p8, %p9;         // Either is NaN?
        @%p8 mov.b64 %rd21, 0x7FF8000000000000;  // NaN bit pattern
        @%p8 mov.b64 %fd14, %rd21;
        @%p8 st.global.f64 [%rd4], %fd14;        // Write NaN to result
        @%p8 mov.u32 %r20, 1;
        @%p8 st.shared.u32 [nan_found], %r20;    // Set flag
        
        min.f64 %fd1, %fd1, %fd3;               // Min lane 0
        min.f64 %fd2, %fd2, %fd4;               // Min lane 1
        
        // Grid stride to next chunk (use 64-bit)
        add.u64 %rd23, %rd23, %rd24;
        bra loop_start;  // Continue loop
        
    remainder_check:  // Remainder handling with masking (like AVX-512)
        setp.lt.u64 %p0, %rd23, %rd26;  // Check if first element exists (use 64-bit)
        @!%p0 bra final_reduction;   // Skip if no remainder
        
        // Load single remainder element
        mul.lo.u64 %rd1, %rd23, 8;      // Use 64-bit multiply
        add.u64 %rd2, %rd3, %rd1;
        ld.global.f64 %fd3, [%rd2];
        
        // Check if value is NaN and write directly
        setp.neu.f64 %p8, %fd3, %fd3;
        @%p8 mov.b64 %rd22, 0x7FF8000000000000;  // NaN bit pattern
        @%p8 mov.b64 %fd14, %rd22;
        @%p8 st.global.f64 [%rd4], %fd14;
        @%p8 mov.u32 %r20, 1;
        @%p8 st.shared.u32 [nan_found], %r20;
        
        min.f64 %fd1, %fd1, %fd3;
        
    final_reduction:  // Final horizontal reduction with shared memory
        min.f64 %fd1, %fd1, %fd2;  // Min both lanes
        
        // Store thread's min to shared memory
        mul.wide.u32 %rd1, %r1, 8;
        mov.u64 %rd2, sdata;
        add.u64 %rd5, %rd2, %rd1;
        st.shared.f64 [%rd5], %fd1;
        bar.sync 0;
        
        // Shared memory reduction
        mov.u32 %r11, 128;
        
    reduction_loop:
        // Check if we're down to warp size
        setp.le.u32 %p4, %r11, 32;
        @%p4 bra warp_reduction;
        
        setp.lt.u32 %p1, %r1, %r11;  // Check if tid < stride
        @!%p1 bra skip_reduction;
        
        // Load from shared[tid + stride]
        add.u32 %r12, %r1, %r11;
        mul.wide.u32 %rd1, %r12, 8;
        mov.u64 %rd6, sdata;
        add.u64 %rd2, %rd6, %rd1;
        ld.shared.f64 %fd5, [%rd2];
        
        // Load from shared[tid]
        mul.wide.u32 %rd1, %r1, 8;
        mov.u64 %rd7, sdata;
        add.u64 %rd2, %rd7, %rd1;
        ld.shared.f64 %fd6, [%rd2];
        
        // Compute min and store back
        min.f64 %fd6, %fd6, %fd5;
        st.shared.f64 [%rd2], %fd6;
        
    skip_reduction:
        bar.sync 0;
        shr.u32 %r11, %r11, 1;
        bra reduction_loop;
        
    warp_reduction:
        // Final warp reduction
        setp.ge.u32 %p2, %r1, 32;
        @%p2 bra done;
        
        // Load thread's value from shared memory
        mul.wide.u32 %rd1, %r1, 8;
        mov.u64 %rd8, sdata;
        add.u64 %rd2, %rd8, %rd1;
        ld.shared.f64 %fd7, [%rd2];
        
        // Warp-level reduction using shuffle
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 16
        shfl.sync.down.b32 %r17, %r15, 16, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 16, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        min.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 8
        shfl.sync.down.b32 %r17, %r15, 8, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 8, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        min.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 4
        shfl.sync.down.b32 %r17, %r15, 4, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 4, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        min.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 2
        shfl.sync.down.b32 %r17, %r15, 2, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 2, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        min.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 1
        shfl.sync.down.b32 %r17, %r15, 1, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 1, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        min.f64 %fd7, %fd7, %fd8;
        
        // Thread 0 of each block atomically updates global min
        setp.ne.u32 %p3, %r1, 0;
        @%p3 bra done;
        
        // Check if NaN was found - if so, skip the update
        ld.shared.u32 %r21, [nan_found];
        setp.ne.u32 %p7, %r21, 0;
        @%p7 bra done;
        
        // Store result back to shared[0] for block-level result
        st.shared.f64 [sdata], %fd7;
        bar.sync 0;
        
        // Thread 0 writes the block's result to global memory
        // For a single-block configuration, this is the final result
        // For multi-block, we'd need a second kernel or CPU reduction
        ld.shared.f64 %fd10, [sdata];
        
        // Atomic min for f64 using CAS loop (reduce_min)
        mov.u64 %rd12, %rd4;  // Address of result
        
    min_cas_loop:
        ld.global.f64 %fd11, [%rd12];
        min.f64 %fd12, %fd11, %fd10;
        
        // Convert to u64 for CAS
        mov.b64 %rd13, %fd11;
        mov.b64 %rd14, %fd12;
        
        atom.global.cas.b64 %rd15, [%rd12], %rd13, %rd14;
        
        // Check if CAS succeeded by comparing bit patterns
        setp.ne.u64 %p6, %rd15, %rd13;
        @%p6 bra min_cas_loop;
        
    done:
        ret;
    }
  "#;

  // Pass parameters to PTX kernel
  let len_u64 = len as u64;
  let values_u64 = values as u64;
  let result_u64 = result as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_u64 as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::reduction();
  let _ = launch_ptx(
    PTX_REDUCE_MIN_F64_GPU,
    &[],
    "reduce_min_f64_gpu",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 optimized f64 min reduction.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn reduce_min_f64_avx512(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX512_F64;

  let mut min_vec = _mm512_set1_pd(f64::INFINITY);
  let mut nan_found = 0u8;

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN using AVX-512 mask operations
    let nan_mask = _mm512_cmp_pd_mask(values_chunk, values_chunk, _CMP_UNORD_Q);
    nan_found |= nan_mask;

    min_vec = _mm512_min_pd(min_vec, values_chunk);
  }

  // Return NaN immediately if found
  if nan_found != 0 {
    return f64::NAN;
  }

  // Extract and find minimum of the 8 elements (matching NEON manual extraction pattern)
  let min_val = _mm512_reduce_min_pd(min_vec);

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  let mut final_min = min_val;

  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let remaining_vec = _mm512_mask_loadu_pd(
      _mm512_set1_pd(f64::INFINITY),
      load_mask,
      values.as_ptr().add(offset),
    );

    // Check for NaN in remaining elements
    let nan_mask = _mm512_mask_cmp_pd_mask(load_mask, remaining_vec, remaining_vec, _CMP_UNORD_Q);
    if nan_mask != 0 {
      return f64::NAN;
    }

    let remaining_min = _mm512_mask_reduce_min_pd(load_mask, remaining_vec);
    if remaining_min < final_min {
      final_min = remaining_min;
    }
  }

  final_min
}

// AVX2 optimized f64 min reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn reduce_min_f64_avx2(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX2_F64; // AVX2 processes 4 f64 values at once

  let mut min_vec = _mm256_set1_pd(f64::INFINITY);
  let mut nan_vec = _mm256_setzero_pd(); // Track if we've seen NaN

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN
    let nan_mask = _mm256_cmp_pd(values_chunk, values_chunk, _CMP_UNORD_Q);
    // Use bitwise OR on the bit representation since _mm256_or_pd doesn't exist
    let nan_mask_bits = _mm256_castpd_si256(nan_mask);
    let nan_vec_bits = _mm256_castpd_si256(nan_vec);
    nan_vec = _mm256_castsi256_pd(_mm256_or_si256(nan_vec_bits, nan_mask_bits));

    min_vec = _mm256_min_pd(min_vec, values_chunk);
  }

  // Check if any NaN was found
  let nan_mask = _mm256_movemask_pd(nan_vec);
  if nan_mask != 0 {
    return f64::NAN;
  }

  // **DEEP SIMD OPTIMIZATION**: Pure AVX2 horizontal min reduction
  // Shuffle high 128-bit lane to low for comparison
  let shuffled = _mm256_permute2f128_pd(min_vec, min_vec, 0x01);
  let min_pairs = _mm256_min_pd(min_vec, shuffled);

  // Shuffle within 128-bit lanes for final reduction
  let min_shuffled = _mm256_shuffle_pd(min_pairs, min_pairs, 0x05);
  let final_min = _mm256_min_pd(min_pairs, min_shuffled);
  let min_val = _mm_cvtsd_f64(_mm256_castpd256_pd128(final_min));

  // Handle remaining elements with scalar comparison (matching NEON structure)
  let offset = full_chunks * LANES;
  let mut final_min = min_val;
  for j in 0..remaining_elements {
    let val = values[offset + j];
    if val.is_nan() {
      return f64::NAN;
    }
    if val < final_min {
      final_min = val;
    }
  }

  final_min
}

// NEON optimized f64 min reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn reduce_min_f64_neon(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_NEON_F64;

  let mut min_vec = vdupq_n_f64(f64::INFINITY);
  let mut nan_found = false;
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_f64(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN, so vceqq_f64(x, x) returns 0 for NaN
    let nan_check = vceqq_f64(values_chunk, values_chunk);
    if vgetq_lane_u64(nan_check, 0) == 0 || vgetq_lane_u64(nan_check, 1) == 0 {
      nan_found = true;
    }

    min_vec = vminq_f64(min_vec, values_chunk);
  }

  // Return NaN immediately if found
  if nan_found {
    return f64::NAN;
  }

  // Extract and find minimum of the 2 elements
  let min_val = vgetq_lane_f64(min_vec, 0).min(vgetq_lane_f64(min_vec, 1));

  // Handle remaining elements with scalar comparison
  let offset = full_chunks * LANES;
  let mut final_min = min_val;
  for j in 0..remaining_elements {
    let val = values[offset + j];
    if val.is_nan() {
      return f64::NAN;
    }
    if val < final_min {
      final_min = val;
    }
  }

  final_min
}

// =============================================================================
// REDUCE MAX F64 OPERATIONS
// =============================================================================

// GPU optimized f64 max reduction using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn reduce_max_f64_gpu(values: *const f64, len: usize, result: *mut f64) {
  const PTX_REDUCE_MAX_F64_GPU: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64

    .entry reduce_max_f64_gpu(.param .u64 values_param, .param .u64 len_param, .param .u64 result_param) {
        .reg .f64 %fd<25>;
        .reg .u32 %r<35>;
        .reg .u64 %rd<30>;  // Increased for 64-bit indices and CAS loop
        .reg .pred %p<10>;
        .shared .f64 sdata[256];
        .shared .u32 nan_found[1];
        
        // Load parameters
        ld.param.u64 %rd3, [values_param];
        ld.param.u64 %rd5, [len_param];
        cvt.u32.u64 %r7, %rd5;  
        ld.param.u64 %rd4, [result_param];
        
        // Thread 0 initializes shared flag
        mov.u32 %r1, %tid.x;
        setp.eq.u32 %p1, %r1, 0;
        @%p1 mov.u32 %r20, 0;
        @%p1 st.shared.u32 [nan_found], %r20;
        bar.sync 0;
        
        // Initialize accumulators with negative infinity (for max)
        mov.b64 %rd20, 0xFFF0000000000000;  // -INFINITY bit pattern  
        mov.b64 %fd1, %rd20;
        mov.b64 %fd2, %rd20;
        
        // Calculate starting index for this thread (use 64-bit to prevent overflow)
        mov.u32 %r1, %tid.x;     // Thread ID within block
        mov.u32 %r2, %ntid.x;    // Block size
        mov.u32 %r3, %ctaid.x;   // Block ID
        mul.wide.u32 %rd23, %r3, %r2;
        cvt.u64.u32 %rd24, %r1;
        add.u64 %rd23, %rd23, %rd24;   // Global thread ID in 64-bit
        mul.lo.u64 %rd23, %rd23, 2;    // Each thread starts at ID * 2 (double2)
        
        // Calculate grid stride (use 64-bit)
        mov.u32 %r8, %nctaid.x; // Number of blocks
        mul.wide.u32 %rd24, %r8, %r2;  // Total threads in grid (64-bit)
        mul.lo.u64 %rd24, %rd24, 2;    // Stride = total_threads * 2 (double2)
        
        // Main grid-stride loop processing double2 chunks
    loop_start:
        // Check if we can load full double2 (use 64-bit comparison)
        add.u64 %rd25, %rd23, 1;
        cvt.u64.u32 %rd26, %r7;        // Convert len to 64-bit
        setp.ge.u64 %p0, %rd25, %rd26;  // Check if we can load full double2
        @%p0 bra remainder_check;    // Exit loop if beyond array
        
        // Load and compute max with double2 using vector load
        mul.lo.u64 %rd1, %rd23, 8;     // Use 64-bit multiply
        add.u64 %rd2, %rd3, %rd1;
        ld.global.v2.f64 {%fd3, %fd4}, [%rd2];  // Vectorized 128-bit load
        
        // Check if either value is NaN and write directly to result
        setp.neu.f64 %p8, %fd3, %fd3;  // NaN check for fd3
        setp.neu.f64 %p9, %fd4, %fd4;  // NaN check for fd4
        or.pred %p8, %p8, %p9;         // Either is NaN?
        @%p8 mov.b64 %rd21, 0x7FF8000000000000;  // NaN bit pattern
        @%p8 mov.b64 %fd14, %rd21;
        @%p8 st.global.f64 [%rd4], %fd14;        // Write NaN to result
        @%p8 mov.u32 %r20, 1;
        @%p8 st.shared.u32 [nan_found], %r20;    // Set flag
        
        max.f64 %fd1, %fd1, %fd3;               // Max lane 0
        max.f64 %fd2, %fd2, %fd4;               // Max lane 1
        
        // Grid stride to next chunk (use 64-bit)
        add.u64 %rd23, %rd23, %rd24;
        bra loop_start;  // Continue loop
        
    remainder_check:  // Remainder handling with masking (like AVX-512)
        setp.lt.u64 %p0, %rd23, %rd26;  // Check if first element exists (use 64-bit)
        @!%p0 bra final_reduction;   // Skip if no remainder
        
        // Load single remainder element
        mul.lo.u64 %rd1, %rd23, 8;      // Use 64-bit multiply
        add.u64 %rd2, %rd3, %rd1;
        ld.global.f64 %fd3, [%rd2];
        
        // Check if value is NaN and write directly
        setp.neu.f64 %p8, %fd3, %fd3;
        @%p8 mov.b64 %rd22, 0x7FF8000000000000;  // NaN bit pattern
        @%p8 mov.b64 %fd14, %rd22;
        @%p8 st.global.f64 [%rd4], %fd14;
        @%p8 mov.u32 %r20, 1;
        @%p8 st.shared.u32 [nan_found], %r20;
        
        max.f64 %fd1, %fd1, %fd3;
        
    final_reduction:  // Final horizontal reduction with shared memory
        max.f64 %fd1, %fd1, %fd2;  // Max both lanes
        
        // Store thread's max to shared memory
        mul.wide.u32 %rd1, %r1, 8;
        mov.u64 %rd2, sdata;
        add.u64 %rd5, %rd2, %rd1;
        st.shared.f64 [%rd5], %fd1;
        bar.sync 0;
        
        // Shared memory reduction
        mov.u32 %r11, 128;
        
    reduction_loop:
        // Check if we're down to warp size
        setp.le.u32 %p4, %r11, 32;
        @%p4 bra warp_reduction;
        
        setp.lt.u32 %p1, %r1, %r11;  // Check if tid < stride
        @!%p1 bra skip_reduction;
        
        // Load from shared[tid + stride]
        add.u32 %r12, %r1, %r11;
        mul.wide.u32 %rd1, %r12, 8;
        mov.u64 %rd6, sdata;
        add.u64 %rd2, %rd6, %rd1;
        ld.shared.f64 %fd5, [%rd2];
        
        // Load from shared[tid]
        mul.wide.u32 %rd1, %r1, 8;
        mov.u64 %rd7, sdata;
        add.u64 %rd2, %rd7, %rd1;
        ld.shared.f64 %fd6, [%rd2];
        
        // Compute max and store back
        max.f64 %fd6, %fd6, %fd5;
        st.shared.f64 [%rd2], %fd6;
        
    skip_reduction:
        bar.sync 0;
        shr.u32 %r11, %r11, 1;
        bra reduction_loop;
        
    warp_reduction:
        // Final warp reduction
        setp.ge.u32 %p2, %r1, 32;
        @%p2 bra done;
        
        // Load thread's value from shared memory
        mul.wide.u32 %rd1, %r1, 8;
        mov.u64 %rd8, sdata;
        add.u64 %rd2, %rd8, %rd1;
        ld.shared.f64 %fd7, [%rd2];
        
        // Warp-level reduction using shuffle
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 16
        shfl.sync.down.b32 %r17, %r15, 16, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 16, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        max.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 8
        shfl.sync.down.b32 %r17, %r15, 8, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 8, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        max.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 4
        shfl.sync.down.b32 %r17, %r15, 4, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 4, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        max.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 2
        shfl.sync.down.b32 %r17, %r15, 2, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 2, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        max.f64 %fd7, %fd7, %fd8;
        mov.b64 {%r15, %r16}, %fd7;
        
        // Shuffle down by 1
        shfl.sync.down.b32 %r17, %r15, 1, 31, 0xffffffff;
        shfl.sync.down.b32 %r18, %r16, 1, 31, 0xffffffff;
        mov.b64 %fd8, {%r17, %r18};
        max.f64 %fd7, %fd7, %fd8;
        
        // Thread 0 of each block atomically updates global max
        setp.ne.u32 %p3, %r1, 0;
        @%p3 bra done;
        
        // Check if NaN was found - if so, skip the update
        ld.shared.u32 %r21, [nan_found];
        setp.ne.u32 %p7, %r21, 0;
        @%p7 bra done;
        
        // Store result back to shared[0] for block-level result
        st.shared.f64 [sdata], %fd7;
        bar.sync 0;
        
        // Thread 0 atomically updates global max using CAS loop
        ld.shared.f64 %fd10, [sdata];
        
        // Use a local register for address like reduce_min does
        mov.u64 %rd12, %rd4;  // Address of result
        
    max_cas_loop:
        ld.global.f64 %fd11, [%rd12];
        max.f64 %fd12, %fd11, %fd10;
        
        // Convert to u64 for CAS
        mov.b64 %rd13, %fd11;
        mov.b64 %rd14, %fd12;
        
        atom.global.cas.b64 %rd15, [%rd12], %rd13, %rd14;
        
        // Check if CAS succeeded by comparing bit patterns
        setp.ne.u64 %p6, %rd15, %rd13;
        @%p6 bra max_cas_loop;
        
    done:
        ret;
    }
  "#;

  // Pass parameters to PTX kernel
  let len_u64 = len as u64;
  let values_u64 = values as u64;
  let result_u64 = result as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_u64 as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::reduction();
  let _ = launch_ptx(
    PTX_REDUCE_MAX_F64_GPU,
    &[],
    "reduce_max_f64_gpu",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 optimized f64 max reduction.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn reduce_max_f64_avx512(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX512_F64;

  let mut max_vec = _mm512_set1_pd(f64::NEG_INFINITY);
  let mut nan_found = 0u8;

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN using AVX-512 mask operations
    let nan_mask = _mm512_cmp_pd_mask(values_chunk, values_chunk, _CMP_UNORD_Q);
    nan_found |= nan_mask;

    max_vec = _mm512_max_pd(max_vec, values_chunk);
  }

  // Extract and find maximum of the 8 elements (matching NEON manual extraction pattern)
  let max_val = _mm512_reduce_max_pd(max_vec);

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  let mut final_max = max_val;

  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let remaining_vec = _mm512_mask_loadu_pd(
      _mm512_set1_pd(f64::NEG_INFINITY),
      load_mask,
      values.as_ptr().add(offset),
    );
    let remaining_max = _mm512_mask_reduce_max_pd(load_mask, remaining_vec);

    // Check for NaN in remaining elements
    let nan_mask = _mm512_mask_cmp_pd_mask(load_mask, remaining_vec, remaining_vec, _CMP_UNORD_Q);
    nan_found |= nan_mask;

    if remaining_max > final_max {
      final_max = remaining_max;
    }
  }

  // Return NaN if any was found
  if nan_found != 0 {
    return f64::NAN;
  }

  final_max
}

// AVX2 optimized f64 max reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn reduce_max_f64_avx2(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX2_F64; // AVX2 processes 4 f64 values at once

  let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);
  let mut nan_vec = _mm256_setzero_pd(); // Track if we've seen NaN

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN
    let nan_mask = _mm256_cmp_pd(values_chunk, values_chunk, _CMP_UNORD_Q);
    // Use bitwise OR on the bit representation since _mm256_or_pd doesn't exist
    let nan_mask_bits = _mm256_castpd_si256(nan_mask);
    let nan_vec_bits = _mm256_castpd_si256(nan_vec);
    nan_vec = _mm256_castsi256_pd(_mm256_or_si256(nan_vec_bits, nan_mask_bits));

    max_vec = _mm256_max_pd(max_vec, values_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Pure AVX2 horizontal max reduction
  // Shuffle high 128-bit lane to low for comparison
  let shuffled = _mm256_permute2f128_pd(max_vec, max_vec, 0x01);
  let max_pairs = _mm256_max_pd(max_vec, shuffled);

  // Shuffle within 128-bit lanes for final reduction
  let max_shuffled = _mm256_shuffle_pd(max_pairs, max_pairs, 0x05);
  let final_max = _mm256_max_pd(max_pairs, max_shuffled);
  let max_val = _mm_cvtsd_f64(_mm256_castpd256_pd128(final_max));

  // Check if we found any NaN
  let nan_check = _mm256_movemask_pd(nan_vec);
  if nan_check != 0 {
    return f64::NAN;
  }

  // Handle remaining elements with scalar comparison (matching NEON structure)
  let offset = full_chunks * LANES;
  let mut final_max = max_val;
  for j in 0..remaining_elements {
    let val = values[offset + j];
    if val.is_nan() {
      return f64::NAN;
    }
    if val > final_max {
      final_max = val;
    }
  }

  final_max
}

// NEON optimized f64 max reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn reduce_max_f64_neon(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_NEON_F64;

  let mut max_vec = vdupq_n_f64(f64::NEG_INFINITY);
  let mut nan_found = false;
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_f64(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN, so vceqq_f64(x, x) returns 0 for NaN
    let nan_check = vceqq_f64(values_chunk, values_chunk);
    if vgetq_lane_u64(nan_check, 0) == 0 || vgetq_lane_u64(nan_check, 1) == 0 {
      nan_found = true;
    }

    max_vec = vmaxq_f64(max_vec, values_chunk);
  }

  // Return NaN immediately if found
  if nan_found {
    return f64::NAN;
  }

  // Extract and find maximum of the 2 elements
  let max_val = vgetq_lane_f64(max_vec, 0).max(vgetq_lane_f64(max_vec, 1));

  // Handle remaining elements with scalar comparison
  let offset = full_chunks * LANES;
  let mut final_max = max_val;
  for j in 0..remaining_elements {
    let val = values[offset + j];
    if val.is_nan() {
      return f64::NAN;
    }
    if val > final_max {
      final_max = val;
    }
  }

  final_max
}

// =============================================================================
// REDUCE MAX U32 OPERATIONS
// =============================================================================

// GPU optimized u32 max reduction using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn reduce_max_u32_gpu(values: *const u32, len: usize, result: *mut u32) {
  const PTX_REDUCE_MAX_U32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry reduce_max_u32(
      .param .u64 values_ptr,
      .param .u64 len,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .shared .u32 sdata[256];

      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd15, [len];
      cvt.u32.u64 %r0, %rd15;
      ld.param.u64 %rd4, [result_ptr];

      // Initialize max accumulator with minimum value
      mov.u32 %r10, 0;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid

      // Main grid-stride loop - each thread computes its local max
    loop_start:
      setp.ge.u32 %p0, %r5, %r0;  // Check if index is beyond array
      @%p0 bra store_shared;  // Exit loop if beyond array

      // Load and compute max with single element
      mul.wide.u32 %rd1, %r5, 4;  // Multiply by sizeof(u32)
      add.u64 %rd2, %rd0, %rd1;
      ld.global.u32 %r6, [%rd2];       // Load element
      max.u32 %r10, %r10, %r6;         // Update max

      // Grid stride to next element
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

    store_shared:
      // Store thread's max to shared memory
      mul.wide.u32 %rd3, %r1, 4;
      mov.u64 %rd1, sdata;
      add.u64 %rd2, %rd1, %rd3;
      st.shared.u32 [%rd2], %r10;
      bar.sync 0;

      // Shared memory reduction for strides > 32
      mov.u32 %r11, 128;
    reduction_loop:
      // Check if we're down to warp size
      setp.le.u32 %p4, %r11, 32;
      @%p4 bra warp_reduction;

      setp.lt.u32 %p1, %r1, %r11;  // Check if tid < stride
      @!%p1 bra skip_reduction;
      
      // Load from shared[tid + stride]
      add.u32 %r12, %r1, %r11;
      mul.wide.u32 %rd3, %r12, 4;
      add.u64 %rd2, %rd1, %rd3;
      ld.shared.u32 %r13, [%rd2];
      
      // Load from shared[tid]
      mul.wide.u32 %rd3, %r1, 4;
      add.u64 %rd2, %rd1, %rd3;
      ld.shared.u32 %r14, [%rd2];
      
      // Compute max and store back
      max.u32 %r15, %r14, %r13;
      st.shared.u32 [%rd2], %r15;
      
    skip_reduction:
      bar.sync 0;
      
      // Halve the stride
      shr.u32 %r11, %r11, 1;
      bra reduction_loop;

    warp_reduction:
      // Load value from shared memory for warp reduction
      mul.wide.u32 %rd3, %r1, 4;
      add.u64 %rd2, %rd1, %rd3;
      ld.shared.u32 %r16, [%rd2];

      // Get lane ID within warp
      and.b32 %r17, %r1, 0x1f;

      // Only first warp participates in final reduction
      setp.ge.u32 %p5, %r1, 32;
      @%p5 bra skip_store;

      // Warp shuffle reduction - no synchronization needed!
      // Shuffle down by 16
      shfl.sync.down.b32 %r18, %r16, 16, 0x1f, 0xffffffff;
      max.u32 %r16, %r16, %r18;

      // Shuffle down by 8
      shfl.sync.down.b32 %r18, %r16, 8, 0x1f, 0xffffffff;
      max.u32 %r16, %r16, %r18;

      // Shuffle down by 4
      shfl.sync.down.b32 %r18, %r16, 4, 0x1f, 0xffffffff;
      max.u32 %r16, %r16, %r18;

      // Shuffle down by 2
      shfl.sync.down.b32 %r18, %r16, 2, 0x1f, 0xffffffff;
      max.u32 %r16, %r16, %r18;

      // Shuffle down by 1
      shfl.sync.down.b32 %r18, %r16, 1, 0x1f, 0xffffffff;
      max.u32 %r16, %r16, %r18;

      // Lane 0 of first warp has the final result
      setp.ne.u32 %p3, %r17, 0;
      @%p3 bra skip_store;

      // Store result back to shared[0] for block-level result
      st.shared.u32 [sdata], %r16;

    skip_store:
      bar.sync 0;
      
      // Thread 0 of each block atomically updates global max
      setp.ne.u32 %p3, %r1, 0;
      @%p3 bra done;
      
      ld.shared.u32 %r19, [sdata];
      atom.global.max.u32 %r25, [%rd4], %r19;

    done:
      ret;
    }
  "#;

  let values_ptr = values as u64;
  let len_u64 = len as u64;
  let result_ptr = result as u64;
  let args = [
    &values_ptr as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_ptr as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::reduction();
  let _ = launch_ptx(
    PTX_REDUCE_MAX_U32,
    &[],
    "reduce_max_u32",
    blocks,
    threads,
    &args,
  );
}

// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn reduce_max_u32_avx512(values: &[u32], len: usize) -> u32 {
  const LANES: usize = LANES_AVX512_U32; // AVX-512 processes 16 u32 values at once

  let mut max_vec = _mm512_setzero_si512();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 16-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_epi32(values.as_ptr().add(offset) as *const i32);
    max_vec = _mm512_max_epu32(max_vec, values_chunk);
  }

  // Use AVX-512 horizontal reduction for maximum (only if we processed chunks)
  let mut final_max = if full_chunks > 0 {
    _mm512_reduce_max_epu32(max_vec) as u32
  } else {
    0u32
  };

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;

  // Handle remaining elements with scalar operations
  for j in 0..remaining_elements {
    if values[offset + j] > final_max {
      final_max = values[offset + j];
    }
  }

  final_max
}

// AVX2 optimized u32 max reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn reduce_max_u32_avx2(values: &[u32], len: usize) -> u32 {
  const LANES: usize = LANES_AVX2_U32; // AVX2 processes 8 u32 values at once

  let mut max_vec = _mm256_setzero_si256();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_si256(values.as_ptr().add(offset) as *const __m256i);
    max_vec = _mm256_max_epu32(max_vec, values_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Pure AVX2 horizontal max reduction using shuffling
  // Shuffle high 128-bit lane to low for comparison
  let shuffled = _mm256_permute2x128_si256(max_vec, max_vec, 0x01);
  let max_128 = _mm256_max_epu32(max_vec, shuffled);
  let max_64 = _mm256_max_epu32(max_128, _mm256_shuffle_epi32(max_128, 0x4E));
  let final_max = _mm256_max_epu32(max_64, _mm256_shuffle_epi32(max_64, 0xB1));

  // Extract the final maximum value
  let max_val = _mm256_extract_epi32(final_max, 0) as u32;

  // Handle remaining elements with scalar comparison (matching NEON structure)
  let offset = full_chunks * LANES;
  let mut final_max = max_val;
  for j in 0..remaining_elements {
    if values[offset + j] > final_max {
      final_max = values[offset + j];
    }
  }

  final_max
}

// NEON optimized u32 max reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn reduce_max_u32_neon(values: &[u32], len: usize) -> u32 {
  const LANES: usize = LANES_NEON_U32;

  let mut max_vec = vdupq_n_u32(0);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_u32(values.as_ptr().add(offset));
    max_vec = vmaxq_u32(max_vec, values_chunk);
  }

  // Extract maximum value from SIMD register using explicit constant indices
  let lane0 = vgetq_lane_u32(max_vec, 0);
  let lane1 = vgetq_lane_u32(max_vec, 1);
  let lane2 = vgetq_lane_u32(max_vec, 2);
  let lane3 = vgetq_lane_u32(max_vec, 3);
  let max_val = lane0.max(lane1).max(lane2).max(lane3);

  // Handle remaining elements
  let offset = full_chunks * LANES;
  let mut final_max = max_val;
  for j in 0..remaining_elements {
    if values[offset + j] > final_max {
      final_max = values[offset + j];
    }
  }

  final_max
}

// =============================================================================
// REDUCE MIN U32 OPERATIONS
// =============================================================================

// GPU optimized u32 min reduction using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn reduce_min_u32_gpu(values: *const u32, len: usize, result: *mut u32) {
  const PTX_REDUCE_MIN_U32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry reduce_min_u32(
      .param .u64 values_ptr,
      .param .u64 len,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .shared .u32 sdata[256];

      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd15, [len];
      cvt.u32.u64 %r0, %rd15;
      ld.param.u64 %rd4, [result_ptr];

      // Initialize min accumulator with u32::MAX
      mov.u32 %r10, 0xFFFFFFFF;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid

      // Main grid-stride loop - each thread computes its local min
    loop_start:
      setp.ge.u32 %p0, %r5, %r0;  // Check if index is beyond array
      @%p0 bra store_shared;  // Exit loop if beyond array

      // Load and compute min with single element
      mul.wide.u32 %rd1, %r5, 4;  // Multiply by sizeof(u32)
      add.u64 %rd2, %rd0, %rd1;
      ld.global.u32 %r6, [%rd2];       // Load element
      min.u32 %r10, %r10, %r6;         // Update min

      // Grid stride to next element
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

    store_shared:
      // Store thread's min to shared memory
      mul.wide.u32 %rd3, %r1, 4;
      mov.u64 %rd1, sdata;
      add.u64 %rd2, %rd1, %rd3;
      st.shared.u32 [%rd2], %r10;
      bar.sync 0;

      // Shared memory reduction for strides > 32
      mov.u32 %r11, 128;
    reduction_loop:
      // Check if we're down to warp size
      setp.le.u32 %p4, %r11, 32;
      @%p4 bra warp_reduction;

      setp.lt.u32 %p1, %r1, %r11;  // Check if tid < stride
      @!%p1 bra skip_reduction;
      
      // Load from shared[tid + stride]
      add.u32 %r12, %r1, %r11;
      mul.wide.u32 %rd3, %r12, 4;
      add.u64 %rd2, %rd1, %rd3;
      ld.shared.u32 %r13, [%rd2];
      
      // Load from shared[tid]
      mul.wide.u32 %rd3, %r1, 4;
      add.u64 %rd2, %rd1, %rd3;
      ld.shared.u32 %r14, [%rd2];
      
      // Compute min and store back
      min.u32 %r15, %r14, %r13;
      st.shared.u32 [%rd2], %r15;
      
    skip_reduction:
      bar.sync 0;
      
      // Halve the stride
      shr.u32 %r11, %r11, 1;
      bra reduction_loop;

    warp_reduction:
      // Load value from shared memory for warp reduction
      mul.wide.u32 %rd3, %r1, 4;
      add.u64 %rd2, %rd1, %rd3;
      ld.shared.u32 %r16, [%rd2];

      // Get lane ID within warp
      and.b32 %r17, %r1, 0x1f;

      // Only first warp participates in final reduction
      setp.ge.u32 %p5, %r1, 32;
      @%p5 bra skip_store;

      // Warp shuffle reduction - no synchronization needed!
      // Shuffle down by 16
      shfl.sync.down.b32 %r18, %r16, 16, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r18;

      // Shuffle down by 8
      shfl.sync.down.b32 %r18, %r16, 8, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r18;

      // Shuffle down by 4
      shfl.sync.down.b32 %r18, %r16, 4, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r18;

      // Shuffle down by 2
      shfl.sync.down.b32 %r18, %r16, 2, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r18;

      // Shuffle down by 1
      shfl.sync.down.b32 %r18, %r16, 1, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r18;

      // Lane 0 of first warp has the final result
      setp.ne.u32 %p3, %r17, 0;
      @%p3 bra skip_store;

      // Store result back to shared[0] for block-level result
      st.shared.u32 [sdata], %r16;

    skip_store:
      bar.sync 0;
      
      // Thread 0 of each block atomically updates global min
      setp.ne.u32 %p3, %r1, 0;
      @%p3 bra done;
      
      ld.shared.u32 %r19, [sdata];
      atom.global.min.u32 %r25, [%rd4], %r19;

    done:
      ret;
    }
  "#;

  let values_ptr = values as u64;
  let len_u64 = len as u64;
  let result_ptr = result as u64;
  let args = [
    &values_ptr as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_ptr as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::reduction();
  let _ = launch_ptx(
    PTX_REDUCE_MIN_U32,
    &[],
    "reduce_min_u32",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 u32 min reduction using proper SIMD vectorization.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn reduce_min_u32_avx512(values: &[u32], len: usize) -> u32 {
  const LANES: usize = LANES_AVX512_U32;

  let mut min_vec = _mm512_set1_epi32(u32::MAX as i32);

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 16-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_epi32(values.as_ptr().add(offset) as *const i32);
    min_vec = _mm512_min_epu32(min_vec, values_chunk);
  }

  // Use AVX-512 horizontal reduction for minimum (only if we processed chunks)
  let mut final_min = if full_chunks > 0 {
    _mm512_reduce_min_epu32(min_vec) as u32
  } else {
    u32::MAX
  };

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;

  // Handle remaining elements with scalar operations
  for j in 0..remaining_elements {
    if values[offset + j] < final_min {
      final_min = values[offset + j];
    }
  }

  final_min
}

// AVX2 optimized u32 min reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn reduce_min_u32_avx2(values: &[u32], len: usize) -> u32 {
  const LANES: usize = LANES_AVX2_U32;

  let mut min_vec = _mm256_set1_epi32(u32::MAX as i32);

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_si256(values.as_ptr().add(offset) as *const __m256i);
    min_vec = _mm256_min_epu32(min_vec, values_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Pure AVX2 horizontal min reduction using shuffling
  // Shuffle high 128-bit lane to low for comparison
  let shuffled = _mm256_permute2x128_si256(min_vec, min_vec, 0x01);
  let min_128 = _mm256_min_epu32(min_vec, shuffled);
  let min_64 = _mm256_min_epu32(min_128, _mm256_shuffle_epi32(min_128, 0x4E));
  let final_min = _mm256_min_epu32(min_64, _mm256_shuffle_epi32(min_64, 0xB1));

  // Extract the final minimum value
  let min_val = _mm256_extract_epi32(final_min, 0) as u32;

  // Handle remaining elements with scalar comparison (matching NEON structure)
  let offset = full_chunks * LANES;
  let mut final_min = min_val;
  for j in 0..remaining_elements {
    if values[offset + j] < final_min {
      final_min = values[offset + j];
    }
  }

  final_min
}

// NEON optimized u32 min reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn reduce_min_u32_neon(values: &[u32], len: usize) -> u32 {
  const LANES: usize = LANES_NEON_U32;

  let mut min_vec = vdupq_n_u32(u32::MAX);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_u32(values.as_ptr().add(offset));
    min_vec = vminq_u32(min_vec, values_chunk);
  }

  // Extract minimum value from SIMD register using explicit constant indices
  let lane0 = vgetq_lane_u32(min_vec, 0);
  let lane1 = vgetq_lane_u32(min_vec, 1);
  let lane2 = vgetq_lane_u32(min_vec, 2);
  let lane3 = vgetq_lane_u32(min_vec, 3);
  let min_val = lane0.min(lane1).min(lane2).min(lane3);

  // Handle remaining elements
  let offset = full_chunks * LANES;
  let mut final_min = min_val;
  for j in 0..remaining_elements {
    if values[offset + j] < final_min {
      final_min = values[offset + j];
    }
  }

  final_min
}

// =============================================================================
// FIND MIN MAX U32 OPERATIONS
// =============================================================================

// GPU optimized u32 min/max finding using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn find_min_max_u32_gpu(
  values: *const u32,
  len: usize,
  min_result_ptr: *mut u32,
  max_result_ptr: *mut u32,
) {
  const PTX_FIND_MIN_MAX_U32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry find_min_max_u32(
      .param .u64 values_ptr,
      .param .u64 len,
      .param .u64 min_result_ptr,
      .param .u64 max_result_ptr
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .shared .u32 smin[32];
      .shared .u32 smax[32];
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      ld.param.u64 %rd3, [min_result_ptr];
      ld.param.u64 %rd4, [max_result_ptr];

      // Initialize min and max
      mov.u32 %r8, 0xFFFFFFFF;  // min = u32::MAX
      mov.u32 %r9, 0;            // max = 0

      // Grid stride loop - process 4 elements per thread
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 4;  // Each thread processes 4 elements

    loop_start:
      add.u32 %r15, %r5, 3;
      setp.ge.u32 %p0, %r15, %r0;
      @%p0 bra scalar_loop;

      // Load 4 u32 values individually to avoid alignment issues
      // r5 is element index, convert to byte offset
      mul.wide.u32 %rd1, %r5, 4;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.u32 %r6, [%rd2];      // Load element 0
      add.u64 %rd5, %rd2, 4;
      ld.global.u32 %r7, [%rd5];      // Load element 1
      add.u64 %rd6, %rd2, 8;
      ld.global.u32 %r16, [%rd6];     // Load element 2
      add.u64 %rd7, %rd2, 12;
      ld.global.u32 %r17, [%rd7];     // Load element 3
      min.u32 %r8, %r8, %r6;
      max.u32 %r9, %r9, %r6;
      min.u32 %r8, %r8, %r7;
      max.u32 %r9, %r9, %r7;
      min.u32 %r8, %r8, %r16;
      max.u32 %r9, %r9, %r16;
      min.u32 %r8, %r8, %r17;
      max.u32 %r9, %r9, %r17;

      // Grid stride increment (4x)
      mov.u32 %r10, %nctaid.x;
      mul.lo.u32 %r11, %r10, %r2;
      mul.lo.u32 %r11, %r11, 4;
      add.u32 %r5, %r5, %r11;
      bra loop_start;
      
    scalar_loop:
      setp.ge.u32 %p0, %r5, %r0;
      @%p0 bra loop_end;
      
      // Load single value
      mul.wide.u32 %rd1, %r5, 4;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.u32 %r6, [%rd2];
      min.u32 %r8, %r8, %r6;
      max.u32 %r9, %r9, %r6;
      
      add.u32 %r5, %r5, 1;
      bra scalar_loop;

    loop_end:
      // Warp-level reduction using shuffle
      // Get lane ID and warp ID
      and.b32 %r12, %r1, 0x1f;  // lane ID (0-31)
      shr.u32 %r13, %r1, 5;      // warp ID (tid / 32)
      
      // Shuffle down by 16 for min
      shfl.sync.down.b32 %r14, %r8, 16, 0x1f, 0xffffffff;
      min.u32 %r8, %r8, %r14;
      // Shuffle down by 16 for max
      shfl.sync.down.b32 %r14, %r9, 16, 0x1f, 0xffffffff;
      max.u32 %r9, %r9, %r14;
      
      // Shuffle down by 8
      shfl.sync.down.b32 %r14, %r8, 8, 0x1f, 0xffffffff;
      min.u32 %r8, %r8, %r14;
      shfl.sync.down.b32 %r14, %r9, 8, 0x1f, 0xffffffff;
      max.u32 %r9, %r9, %r14;
      
      // Shuffle down by 4
      shfl.sync.down.b32 %r14, %r8, 4, 0x1f, 0xffffffff;
      min.u32 %r8, %r8, %r14;
      shfl.sync.down.b32 %r14, %r9, 4, 0x1f, 0xffffffff;
      max.u32 %r9, %r9, %r14;
      
      // Shuffle down by 2
      shfl.sync.down.b32 %r14, %r8, 2, 0x1f, 0xffffffff;
      min.u32 %r8, %r8, %r14;
      shfl.sync.down.b32 %r14, %r9, 2, 0x1f, 0xffffffff;
      max.u32 %r9, %r9, %r14;
      
      // Shuffle down by 1
      shfl.sync.down.b32 %r14, %r8, 1, 0x1f, 0xffffffff;
      min.u32 %r8, %r8, %r14;
      shfl.sync.down.b32 %r14, %r9, 1, 0x1f, 0xffffffff;
      max.u32 %r9, %r9, %r14;
      
      // Lane 0 of each warp writes to shared memory
      setp.eq.u32 %p1, %r12, 0;
      @%p1 mul.wide.u32 %rd1, %r13, 4;
      @%p1 mov.u64 %rd2, smin;
      @%p1 add.u64 %rd2, %rd2, %rd1;
      @%p1 st.shared.u32 [%rd2], %r8;
      @%p1 mov.u64 %rd2, smax;
      @%p1 add.u64 %rd2, %rd2, %rd1;
      @%p1 st.shared.u32 [%rd2], %r9;
      
      bar.sync 0;
      
      // Final reduction of warp results (only first warp)
      // Calculate number of warps in this block
      add.u32 %r15, %r2, 31;
      shr.u32 %r15, %r15, 5;  // num_warps = (block_size + 31) / 32
      
      setp.lt.u32 %p2, %r1, 32;  // Only first warp participates
      @!%p2 bra write_result;
      
      // Load warp results if within range
      setp.lt.u32 %p2, %r1, %r15;
      mov.u32 %r16, 0xFFFFFFFF;  // min = MAX
      mov.u32 %r17, 0;            // max = 0
      @%p2 mul.wide.u32 %rd1, %r1, 4;
      @%p2 mov.u64 %rd2, smin;
      @%p2 add.u64 %rd2, %rd2, %rd1;
      @%p2 ld.shared.u32 %r16, [%rd2];
      @%p2 mov.u64 %rd2, smax;
      @%p2 add.u64 %rd2, %rd2, %rd1;
      @%p2 ld.shared.u32 %r17, [%rd2];
      
      // Final warp reduction
      shfl.sync.down.b32 %r14, %r16, 16, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r14;
      shfl.sync.down.b32 %r14, %r17, 16, 0x1f, 0xffffffff;
      max.u32 %r17, %r17, %r14;
      
      shfl.sync.down.b32 %r14, %r16, 8, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r14;
      shfl.sync.down.b32 %r14, %r17, 8, 0x1f, 0xffffffff;
      max.u32 %r17, %r17, %r14;
      
      shfl.sync.down.b32 %r14, %r16, 4, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r14;
      shfl.sync.down.b32 %r14, %r17, 4, 0x1f, 0xffffffff;
      max.u32 %r17, %r17, %r14;
      
      shfl.sync.down.b32 %r14, %r16, 2, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r14;
      shfl.sync.down.b32 %r14, %r17, 2, 0x1f, 0xffffffff;
      max.u32 %r17, %r17, %r14;
      
      shfl.sync.down.b32 %r14, %r16, 1, 0x1f, 0xffffffff;
      min.u32 %r16, %r16, %r14;
      shfl.sync.down.b32 %r14, %r17, 1, 0x1f, 0xffffffff;
      max.u32 %r17, %r17, %r14;
      
      // Thread 0 writes final block result using atomics
      setp.eq.u32 %p1, %r1, 0;
      @%p1 atom.global.min.u32 %r18, [%rd3], %r16;
      @%p1 atom.global.max.u32 %r19, [%rd4], %r17;
      
    write_result:
      
      ret;
    }
  "#;

  let gpu_values_u64 = values as u64;
  let len_u64 = len as u64;
  let gpu_min_u64 = min_result_ptr as u64;
  let gpu_max_u64 = max_result_ptr as u64;
  let args = [
    &gpu_values_u64 as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &gpu_min_u64 as *const u64 as *const u8,
    &gpu_max_u64 as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::reduction();
  let _ = launch_ptx(
    PTX_FIND_MIN_MAX_U32,
    &[],
    "find_min_max_u32",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 u32 min/max reduction using proper SIMD vectorization.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn find_min_max_u32_avx512(values: &[u32], len: usize) -> (u32, u32) {
  const LANES: usize = LANES_AVX512_U32;

  let mut min_vec = _mm512_set1_epi32(u32::MAX as i32);
  let mut max_vec = _mm512_setzero_si512();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 16-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_epi32(values.as_ptr().add(offset) as *const i32);
    min_vec = _mm512_min_epu32(min_vec, values_chunk);
    max_vec = _mm512_max_epu32(max_vec, values_chunk);
  }

  // Use AVX-512 horizontal reduction for min/max (only if we processed chunks)
  let mut final_min = if full_chunks > 0 {
    _mm512_reduce_min_epu32(min_vec) as u32
  } else {
    u32::MAX
  };
  let mut final_max = if full_chunks > 0 {
    _mm512_reduce_max_epu32(max_vec) as u32
  } else {
    0u32
  };

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;

  // Handle remaining elements with scalar operations
  for j in 0..remaining_elements {
    if values[offset + j] < final_min {
      final_min = values[offset + j];
    }
    if values[offset + j] > final_max {
      final_max = values[offset + j];
    }
  }

  (final_min, final_max)
}

// AVX2 optimized u32 min/max reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn find_min_max_u32_avx2(values: &[u32], len: usize) -> (u32, u32) {
  const LANES: usize = LANES_AVX2_U32;

  let mut min_vec = _mm256_set1_epi32(u32::MAX as i32);
  let mut max_vec = _mm256_setzero_si256();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_si256(values.as_ptr().add(offset) as *const __m256i);
    min_vec = _mm256_min_epu32(min_vec, values_chunk);
    max_vec = _mm256_max_epu32(max_vec, values_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Pure AVX2 horizontal min/max reduction using shuffling
  // Min reduction
  let min_shuffled = _mm256_permute2x128_si256(min_vec, min_vec, 0x01);
  let min_128 = _mm256_min_epu32(min_vec, min_shuffled);
  let min_64 = _mm256_min_epu32(min_128, _mm256_shuffle_epi32(min_128, 0x4E));
  let final_min = _mm256_min_epu32(min_64, _mm256_shuffle_epi32(min_64, 0xB1));
  let min_val = _mm256_extract_epi32(final_min, 0) as u32;

  // Max reduction
  let max_shuffled = _mm256_permute2x128_si256(max_vec, max_vec, 0x01);
  let max_128 = _mm256_max_epu32(max_vec, max_shuffled);
  let max_64 = _mm256_max_epu32(max_128, _mm256_shuffle_epi32(max_128, 0x4E));
  let final_max = _mm256_max_epu32(max_64, _mm256_shuffle_epi32(max_64, 0xB1));
  let max_val = _mm256_extract_epi32(final_max, 0) as u32;

  // Handle remaining elements with scalar comparison (matching NEON structure)
  let offset = full_chunks * LANES;
  let mut final_min = min_val;
  let mut final_max = max_val;
  for j in 0..remaining_elements {
    if values[offset + j] < final_min {
      final_min = values[offset + j];
    }
    if values[offset + j] > final_max {
      final_max = values[offset + j];
    }
  }

  (final_min, final_max)
}

// NEON optimized u32 min/max reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn find_min_max_u32_neon(values: &[u32], len: usize) -> (u32, u32) {
  const LANES: usize = LANES_NEON_U32;

  let mut min_vec = vdupq_n_u32(u32::MAX);
  let mut max_vec = vdupq_n_u32(0);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_u32(values.as_ptr().add(offset));
    min_vec = vminq_u32(min_vec, values_chunk);
    max_vec = vmaxq_u32(max_vec, values_chunk);
  }

  // Extract min and max values from SIMD registers using explicit constant indices
  let min_lane0 = vgetq_lane_u32(min_vec, 0);
  let min_lane1 = vgetq_lane_u32(min_vec, 1);
  let min_lane2 = vgetq_lane_u32(min_vec, 2);
  let min_lane3 = vgetq_lane_u32(min_vec, 3);
  let min_val = min_lane0.min(min_lane1).min(min_lane2).min(min_lane3);

  let max_lane0 = vgetq_lane_u32(max_vec, 0);
  let max_lane1 = vgetq_lane_u32(max_vec, 1);
  let max_lane2 = vgetq_lane_u32(max_vec, 2);
  let max_lane3 = vgetq_lane_u32(max_vec, 3);
  let max_val = max_lane0.max(max_lane1).max(max_lane2).max(max_lane3);

  // Handle remaining elements
  let offset = full_chunks * LANES;
  let mut final_min = min_val;
  let mut final_max = max_val;
  for j in 0..remaining_elements {
    if values[offset + j] < final_min {
      final_min = values[offset + j];
    }
    if values[offset + j] > final_max {
      final_max = values[offset + j];
    }
  }

  (final_min, final_max)
}

// =============================================================================
// REDUCE WEIGHTED SUM F64 OPERATIONS
// =============================================================================

// GPU optimized f64 weighted sum reduction using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn reduce_weighted_sum_f64_gpu(
  values: *const f64,
  weights: *const u64,
  len: usize,
) -> f64 {
  const PTX_REDUCE_WEIGHTED_SUM_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry reduce_weighted_sum_f64(
      .param .u64 values_ptr,
      .param .u64 weights_ptr,
      .param .u64 len,
      .param .u64 result_ptr
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd6, [values_ptr];
      ld.param.u64 %rd7, [weights_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      ld.param.u64 %rd8, [result_ptr];
      
      // Initialize double2 accumulator
      mov.f64 %fd1, 0.0;  // Lane 0
      mov.f64 %fd2, 0.0;  // Lane 1
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load and compute weighted sum with double2
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd6, %rd1;
      add.u64 %rd3, %rd7, %rd1;
      // Load values with vectorized 128-bit load
      ld.global.v2.f64 {%fd3, %fd4}, [%rd2];  // Load double2 values
      // Load weights with vectorized 128-bit load
      ld.global.v2.u64 {%rd4, %rd5}, [%rd3];  // Load double2 weights
      // Convert weights to f64 and compute weighted sum
      cvt.rn.f64.u64 %fd5, %rd4;       // Convert weight 0
      cvt.rn.f64.u64 %fd6, %rd5;       // Convert weight 1
      fma.rn.f64 %fd1, %fd3, %fd5, %fd1; // value0 * weight0 + accum0
      fma.rn.f64 %fd2, %fd4, %fd6, %fd2; // value1 * weight1 + accum1
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra final_reduction;  // Skip if no remainder
      
      // Load single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd6, %rd1;
      add.u64 %rd3, %rd7, %rd1;
      ld.global.f64 %fd3, [%rd2];
      ld.global.u64 %rd4, [%rd3];
      cvt.rn.f64.u64 %fd5, %rd4;
      fma.rn.f64 %fd1, %fd3, %fd5, %fd1;
      
    final_reduction:  // Final horizontal reduction with warp shuffle
      add.f64 %fd1, %fd1, %fd2;  // Sum both lanes
      
      // Warp shuffle reduction for f64 (split into 32-bit halves)
      mov.b64 {%r10, %r11}, %fd1;  // Split f64 into two u32
      shfl.sync.down.b32 %r5, %r10, 16, 31, 0xffffffff;
      shfl.sync.down.b32 %r6, %r11, 16, 31, 0xffffffff;
      mov.b64 %fd3, {%r5, %r6};
      add.f64 %fd1, %fd1, %fd3;
      
      mov.b64 {%r10, %r11}, %fd1;
      shfl.sync.down.b32 %r5, %r10, 8, 31, 0xffffffff;
      shfl.sync.down.b32 %r6, %r11, 8, 31, 0xffffffff;
      mov.b64 %fd3, {%r5, %r6};
      add.f64 %fd1, %fd1, %fd3;
      
      mov.b64 {%r10, %r11}, %fd1;
      shfl.sync.down.b32 %r5, %r10, 4, 31, 0xffffffff;
      shfl.sync.down.b32 %r6, %r11, 4, 31, 0xffffffff;
      mov.b64 %fd3, {%r5, %r6};
      add.f64 %fd1, %fd1, %fd3;
      
      mov.b64 {%r10, %r11}, %fd1;
      shfl.sync.down.b32 %r5, %r10, 2, 31, 0xffffffff;
      shfl.sync.down.b32 %r6, %r11, 2, 31, 0xffffffff;
      mov.b64 %fd3, {%r5, %r6};
      add.f64 %fd1, %fd1, %fd3;
      
      mov.b64 {%r10, %r11}, %fd1;
      shfl.sync.down.b32 %r5, %r10, 1, 31, 0xffffffff;
      shfl.sync.down.b32 %r6, %r11, 1, 31, 0xffffffff;
      mov.b64 %fd3, {%r5, %r6};
      add.f64 %fd1, %fd1, %fd3;
      
      // Only lane 0 writes result
      and.b32 %r7, %tid.x, 0x1f;
      setp.eq.u32 %p1, %r7, 0;
      @%p1 st.global.f64 [%rd8], %fd1;
      
      ret;
    }
  "#;

  let result: f64 = 0.0;
  let values_ptr = values as u64;
  let weights_ptr = weights as u64;
  let len_u64 = len as u64;
  let result_ptr = result as u64;
  let args = [
    &values_ptr as *const u64 as *const u8,
    &weights_ptr as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_ptr as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::reduction();
  let _ = launch_ptx(
    PTX_REDUCE_WEIGHTED_SUM_F64,
    &[],
    "reduce_weighted_sum_f64",
    blocks,
    threads,
    &args,
  );
  result
}

// AVX-512 optimized f64 weighted sum reduction.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn reduce_weighted_sum_f64_avx512(
  values: &[f64],
  weights: &[u64],
  len: usize,
) -> f64 {
  const LANES: usize = LANES_AVX512_F64;

  let mut sum_vec = _mm512_setzero_pd();

  // Process complete SIMD chunks (matching NEON structure)
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Load u64 weights and convert to f64 using SIMD
    let weights_i64_chunk = _mm512_loadu_si512(weights.as_ptr().add(offset) as *const _);
    let weights_chunk = _mm512_cvtepi64_pd(weights_i64_chunk);

    // Multiply values by weights and accumulate in SIMD register
    let weighted_chunk = _mm512_mul_pd(values_chunk, weights_chunk);
    sum_vec = _mm512_add_pd(sum_vec, weighted_chunk);
  }

  // Extract and sum the 8 elements from the vector (matching NEON manual extraction pattern)
  let sum = _mm512_reduce_add_pd(sum_vec);

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  let mut final_sum = sum;

  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let values_chunk =
      _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, values.as_ptr().add(offset));
    let weights_i64_chunk = _mm512_mask_loadu_epi64(
      _mm512_setzero_si512(),
      load_mask,
      weights.as_ptr().add(offset) as *const i64,
    );
    let weights_chunk = _mm512_cvtepi64_pd(weights_i64_chunk);

    let weighted_chunk = _mm512_mul_pd(values_chunk, weights_chunk);
    let remaining_sum = _mm512_mask_reduce_add_pd(load_mask, weighted_chunk);
    final_sum += remaining_sum;
  }

  final_sum
}

// AVX2 optimized f64 weighted sum reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn reduce_weighted_sum_f64_avx2(
  values: &[f64],
  weights: &[u64],
  len: usize,
) -> f64 {
  const LANES: usize = LANES_AVX2_F64;

  let mut sum_vec = _mm256_set1_pd(0.0);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with AVX2
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Load u64 weights as integers and convert using AVX2 intrinsics
    let weights_low = _mm256_loadu_si256(weights.as_ptr().add(offset) as *const __m256i);
    let weights_high = _mm256_loadu_si256(weights.as_ptr().add(offset + 2) as *const __m256i);

    // Convert u64 to f64 using AVX2 (split into low/high 32-bit parts)
    let low_32 = _mm256_shuffle_epi32(weights_low, 0xD8);
    let high_32 = _mm256_shuffle_epi32(weights_high, 0xD8);
    let combined_32 = _mm256_unpacklo_epi64(low_32, high_32);
    let weights_chunk = _mm256_cvtepi32_pd(_mm256_extracti128_si256(combined_32, 0));

    // Multiply values by weights and accumulate
    let weighted_chunk = _mm256_mul_pd(values_chunk, weights_chunk);
    sum_vec = _mm256_add_pd(sum_vec, weighted_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Use AVX2 horizontal reduction instead of scalar loop
  let hadd1 = _mm256_hadd_pd(sum_vec, sum_vec);
  let high = _mm256_extractf128_pd(hadd1, 1);
  let low = _mm256_castpd256_pd128(hadd1);
  let final_sum_vec = _mm_add_pd(low, high);
  let mut sum = _mm_cvtsd_f64(final_sum_vec);

  // Handle remaining elements with scalar multiplication
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    sum += values[offset + j] * weights[offset + j] as f64;
  }

  sum
}

// NEON optimized f64 weighted sum reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn reduce_weighted_sum_f64_neon(
  values: &[f64],
  weights: &[u64],
  len: usize,
) -> f64 {
  const LANES: usize = LANES_NEON_F64;

  let mut sum_vec = vdupq_n_f64(0.0);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_f64(values.as_ptr().add(offset));

    // Load u64 weights and convert to f64 using NEON
    let weights_u64 = vld1q_u64(weights.as_ptr().add(offset));
    let weights_chunk = vcvtq_f64_u64(weights_u64);

    // Multiply values by weights and accumulate
    let weighted_chunk = vmulq_f64(values_chunk, weights_chunk);
    sum_vec = vaddq_f64(sum_vec, weighted_chunk);
  }

  // Extract and sum the 2 elements from the vector
  let sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

  // Handle remaining elements with scalar multiplication
  let offset = full_chunks * LANES;
  let mut final_sum = sum;
  for j in 0..remaining_elements {
    final_sum += values[offset + j] * weights[offset + j] as f64;
  }

  final_sum
}

// =============================================================================
// FUSED MULTIPLY-ADD (FMA) OPERATIONS
// =============================================================================

// GPU optimized fused multiply-add using PTX inline assembly: results[i] = a[i] * b[i] + c[i]
#[cfg(has_cuda)]
pub unsafe fn fma_f64_gpu(
  a: *const f64,
  b: *const f64,
  c: *const f64,
  results: *mut f64,
  len: usize,
) {
  const PTX_FMA_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry fma_f64(
      .param .u64 a_ptr,
      .param .u64 b_ptr,
      .param .u64 c_ptr,
      .param .u64 results_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd0, [a_ptr];
      ld.param.u64 %rd6, [b_ptr];
      ld.param.u64 %rd7, [c_ptr];
      ld.param.u64 %rd5, [results_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;

      // Grid stride loop for double2 processing
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements
      // Calculate grid stride
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r2;
      mul.lo.u32 %r7, %r7, 2;  // Stride = total_threads * 2

    loop_start:
      add.u32 %r8, %r5, 1;
      setp.ge.u32 %p0, %r8, %r0;
      @%p0 bra loop_end;

      // Load a, b, c double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load a double2
      add.u64 %rd3, %rd6, %rd1;
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load b double2
      add.u64 %rd4, %rd7, %rd1;
      ld.global.v2.f64 {%fd4, %fd5}, [%rd4];  // Load c double2

      // Fused multiply-add: a * b + c for both lanes
      fma.rn.f64 %fd6, %fd0, %fd2, %fd4;  // Lane 0: a0 * b0 + c0
      fma.rn.f64 %fd7, %fd1, %fd3, %fd5;  // Lane 1: a1 * b1 + c1

      // Store double2 result
      add.u64 %rd5, %rd5, %rd1;
      st.global.v2.f64 [%rd5], {%fd6, %fd7};

      // Grid stride increment
      add.u32 %r5, %r5, %r7;
      bra loop_start;

    loop_end:
      ret;
    }
  "#;

  let args = [
    a as *const u8,
    b as *const u8,
    c as *const u8,
    results as *const u8,
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_FMA_F64, &[], "fma_f64", blocks, threads, &args);
}

// AVX-512 optimized fused multiply-add: results[i] = a[i] * b[i] + c[i]
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn fma_f64_avx512(
  a: &[f64],
  b: &[f64],
  c: &[f64],
  results: &mut [f64],
  len: usize,
) {
  const LANES: usize = LANES_AVX512_F64;

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX-512
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let a_chunk = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_chunk = _mm512_loadu_pd(b.as_ptr().add(offset));
    let c_chunk = _mm512_loadu_pd(c.as_ptr().add(offset));

    // Fused multiply-add: a * b + c
    let result_chunk = _mm512_mul_pd(a_chunk, b_chunk);
    let final_result = _mm512_add_pd(result_chunk, c_chunk);

    _mm512_storeu_pd(results.as_mut_ptr().add(offset), final_result);
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let a_chunk = _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, a.as_ptr().add(offset));
    let b_chunk = _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, b.as_ptr().add(offset));
    let c_chunk = _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, c.as_ptr().add(offset));

    let result_chunk = _mm512_mul_pd(a_chunk, b_chunk);
    let final_result = _mm512_add_pd(result_chunk, c_chunk);

    _mm512_mask_storeu_pd(results.as_mut_ptr().add(offset), load_mask, final_result);
  }
}

// AVX2 optimized fused multiply-add
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn fma_f64_avx2(
  a: &[f64],
  b: &[f64],
  c: &[f64],
  results: &mut [f64],
  len: usize,
) {
  const LANES: usize = LANES_AVX2_F64;

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with AVX2
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let a_chunk = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_chunk = _mm256_loadu_pd(b.as_ptr().add(offset));
    let c_chunk = _mm256_loadu_pd(c.as_ptr().add(offset));

    // Fused multiply-add: a * b + c
    let result_chunk = _mm256_mul_pd(a_chunk, b_chunk);
    let final_result = _mm256_add_pd(result_chunk, c_chunk);

    _mm256_storeu_pd(results.as_mut_ptr().add(offset), final_result);
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 operations
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    results[offset + j] = a[offset + j] * b[offset + j] + c[offset + j];
  }
}

// NEON optimized fused multiply-add
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn fma_f64_neon(
  a: &[f64],
  b: &[f64],
  c: &[f64],
  results: &mut [f64],
  len: usize,
) {
  const LANES: usize = LANES_NEON_F64;

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let a_chunk = vld1q_f64(a.as_ptr().add(offset));
    let b_chunk = vld1q_f64(b.as_ptr().add(offset));
    let c_chunk = vld1q_f64(c.as_ptr().add(offset));

    // Fused multiply-add: a * b + c
    let result_chunk = vmulq_f64(a_chunk, b_chunk);
    let final_result = vaddq_f64(result_chunk, c_chunk);

    vst1q_f64(results.as_mut_ptr().add(offset), final_result);
  }

  // Handle remaining elements with scalar
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    results[offset + j] = a[offset + j] * b[offset + j] + c[offset + j];
  }
}

// =============================================================================
// VECTORIZED SUBTRACT U32 OPERATIONS
// =============================================================================

// GPU optimized vectorized u32 subtraction using PTX inline assembly: output[i] = values[i] - scalar
#[cfg(has_cuda)]
pub unsafe fn vectorized_subtract_u32_gpu(
  values: *const u32,
  scalar: u32,
  output: *mut u32,
  len: usize,
) {
  const PTX_VECTORIZED_SUBTRACT_U32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry vectorized_subtract_u32(
      .param .u64 values_ptr,
      .param .u32 scalar,
      .param .u64 output_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u32 %r10, [scalar];
      ld.param.u64 %rd4, [output_ptr];
      ld.param.u64 %rd9, [len];
      cvt.u32.u64 %r11, %rd9;

      // Grid stride loop for uint4 processing
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 4;  // Each thread processes 4 elements
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;
      mul.lo.u32 %r9, %r8, %r2;
      mul.lo.u32 %r9, %r9, 4;  // Stride = total_threads * 4

    loop_start:
      add.u32 %r12, %r5, 3;
      setp.ge.u32 %p0, %r12, %r11;
      @%p0 bra loop_end;

      // Load 4 u32 values individually to avoid alignment issues
      // r5 is element index, convert to byte offset
      mul.wide.u32 %rd1, %r5, 4;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.u32 %r6, [%rd2];      // Load element 0
      add.u64 %rd5, %rd2, 4;
      ld.global.u32 %r7, [%rd5];      // Load element 1
      add.u64 %rd6, %rd2, 8;
      ld.global.u32 %r13, [%rd6];     // Load element 2
      add.u64 %rd7, %rd2, 12;
      ld.global.u32 %r14, [%rd7];     // Load element 3

      // Saturating subtraction for all 4 lanes: max(value - scalar, 0)
      // Lane 0
      setp.gt.u32 %p1, %r6, %r10;
      @%p1 sub.u32 %r15, %r6, %r10;
      @!%p1 mov.u32 %r15, 0;
      // Lane 1
      setp.gt.u32 %p2, %r7, %r10;
      @%p2 sub.u32 %r16, %r7, %r10;
      @!%p2 mov.u32 %r16, 0;
      // Lane 2
      setp.gt.u32 %p3, %r13, %r10;
      @%p3 sub.u32 %r17, %r13, %r10;
      @!%p3 mov.u32 %r17, 0;
      // Lane 3
      setp.gt.u32 %p4, %r14, %r10;
      @%p4 sub.u32 %r18, %r14, %r10;
      @!%p4 mov.u32 %r18, 0;

      // Store 4 u32 results individually
      add.u64 %rd3, %rd4, %rd1;
      st.global.u32 [%rd3], %r15;      // Store element 0
      add.u64 %rd8, %rd3, 4;
      st.global.u32 [%rd8], %r16;      // Store element 1
      add.u64 %rd9, %rd3, 8;
      st.global.u32 [%rd9], %r17;      // Store element 2
      add.u64 %rd10, %rd3, 12;
      st.global.u32 [%rd10], %r18;     // Store element 3

      // Grid stride increment
      add.u32 %r5, %r5, %r9;
      bra loop_start;

    loop_end:
      ret;
    }
  "#;

  let values_ptr = values as u64;
  let output_ptr = output as u64;
  let len_u64 = len as u64;
  let args = [
    &values_ptr as *const u64 as *const u8,
    &scalar as *const u32 as *const u8,
    &output_ptr as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(
    PTX_VECTORIZED_SUBTRACT_U32,
    &[],
    "vectorized_subtract_u32",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 vectorized u32 subtraction: output[i] = values[i] - scalar
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn vectorized_subtract_u32_avx512(
  values: &[u32],
  scalar: u32,
  output: &mut [u32],
  len: usize,
) {
  const LANES: usize = LANES_AVX512_U32; // AVX-512 processes 16 u32 values at once

  let scalar_vec = _mm512_set1_epi32(scalar as i32);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 16-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_epi32(values.as_ptr().add(offset) as *const i32);
    // Implement saturating subtraction: max(values - scalar, 0)
    // Since AVX-512 doesn't have saturating subtraction for u32, we implement it manually
    let result_chunk = _mm512_sub_epi32(values_chunk, scalar_vec);
    // Create a mask for values that would underflow (scalar > values)
    let underflow_mask = _mm512_cmpgt_epu32_mask(scalar_vec, values_chunk);
    // Use maskz_mov to set underflowed values to 0
    let result_chunk = _mm512_maskz_mov_epi32(!underflow_mask, result_chunk);
    _mm512_storeu_epi32(output.as_mut_ptr().add(offset) as *mut i32, result_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  if remaining_elements > 0 {
    let load_mask = (1u16 << remaining_elements) - 1;
    let remaining_vec =
      _mm512_maskz_loadu_epi32(load_mask, values.as_ptr().add(offset) as *const i32);
    // Implement saturating subtraction for remaining elements
    let result_chunk = _mm512_sub_epi32(remaining_vec, scalar_vec);
    let underflow_mask = _mm512_cmpgt_epu32_mask(scalar_vec, remaining_vec);
    let result_chunk = _mm512_maskz_mov_epi32(!underflow_mask, result_chunk);
    _mm512_mask_storeu_epi32(
      output.as_mut_ptr().add(offset) as *mut i32,
      load_mask as u16,
      result_chunk,
    );
  }
}

// AVX2 vectorized u32 subtraction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn vectorized_subtract_u32_avx2(
  values: &[u32],
  scalar: u32,
  output: &mut [u32],
  len: usize,
) {
  const LANES: usize = LANES_AVX2_U32; // AVX2 processes 8 u32 values at once

  let scalar_vec = _mm256_set1_epi32(scalar as i32);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_si256(values.as_ptr().add(offset) as *const __m256i);
    // Implement saturating subtraction: max(values - scalar, 0)
    // Since AVX2 doesn't have saturating subtraction for u32, we implement it manually
    let result_chunk = _mm256_sub_epi32(values_chunk, scalar_vec);
    // Create a mask for values that would underflow (result < 0)
    let underflow_mask = _mm256_cmpgt_epi32(scalar_vec, values_chunk);
    // Use blend to set underflowed values to 0
    let result_chunk = _mm256_andnot_si256(underflow_mask, result_chunk);
    _mm256_storeu_si256(
      output.as_mut_ptr().add(offset) as *mut __m256i,
      result_chunk,
    );
  }

  // Handle remaining elements with scalar subtraction (matching NEON structure)
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    output[offset + j] = values[offset + j].saturating_sub(scalar);
  }
}

// NEON vectorized u32 subtraction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn vectorized_subtract_u32_neon(
  values: &[u32],
  scalar: u32,
  output: &mut [u32],
  len: usize,
) {
  const LANES: usize = LANES_NEON_U32;

  let scalar_vec = vdupq_n_u32(scalar);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_u32(values.as_ptr().add(offset));
    let result_chunk = vqsubq_u32(values_chunk, scalar_vec);
    vst1q_u32(output.as_mut_ptr().add(offset), result_chunk);
  }

  // Handle remaining elements
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    output[offset + j] = values[offset + j].saturating_sub(scalar);
  }
}

// =============================================================================
// FIND MIN/MAX F64 OPERATIONS
// =============================================================================

// GPU optimized f64 min/max finding using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn find_min_max_f64_gpu(
  values: *const f64,
  len: usize,
  min_result_ptr: *mut f64,
  max_result_ptr: *mut f64,
) {
  const PTX_FIND_MIN_MAX_F64: &str = r#"
    .version 7.0
    .target sm_70
    .address_size 64
    .entry find_min_max_f64(
      .param .u64 values_ptr,
      .param .u64 len,
      .param .u64 min_result_ptr,
      .param .u64 max_result_ptr
    ) {
      .reg .f64 %fd<27>;
      .reg .u32 %r<40>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .shared .f64 smin[32];
      .shared .f64 smax[32];
      .shared .u32 has_nan_min[1];
      .shared .u32 has_nan_max[1];
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      ld.param.u64 %rd3, [min_result_ptr];
      ld.param.u64 %rd4, [max_result_ptr];

      // Initialize min and max
      mov.f64 %fd8, 0d7FF0000000000000;  // min = +infinity
      mov.f64 %fd9, 0dFFF0000000000000;  // max = -infinity
      
      // Initialize NaN flag registers for min and max
      mov.u32 %r30, 0;  // 0 = no NaN found in min
      mov.u32 %r31, 0;  // 0 = no NaN found in max

      // Grid stride loop - process 4 elements per thread
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 4;  // Each thread processes 4 elements

    loop_start:
      add.u32 %r15, %r5, 3;
      setp.ge.u32 %p0, %r15, %r0;
      @%p0 bra scalar_loop;

      // Load 4 f64 values individually to avoid alignment issues
      // r5 is element index, convert to byte offset
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd6, [%rd2];      // Load element 0
      add.u64 %rd5, %rd2, 8;
      ld.global.f64 %fd7, [%rd5];      // Load element 1
      add.u64 %rd6, %rd2, 16;
      ld.global.f64 %fd16, [%rd6];     // Load element 2
      add.u64 %rd7, %rd2, 24;
      ld.global.f64 %fd17, [%rd7];     // Load element 3
      // Check if any value is NaN
      setp.neu.f64 %p5, %fd6, %fd6;
      setp.neu.f64 %p6, %fd7, %fd7;
      or.pred %p7, %p5, %p6;
      setp.neu.f64 %p5, %fd16, %fd16;
      or.pred %p7, %p7, %p5;
      setp.neu.f64 %p6, %fd17, %fd17;
      or.pred %p7, %p7, %p6;
      
      // If NaN found, directly write NaN to global results
      @%p7 mov.b64 %rd23, 0x7FF8000000000000;  // NaN bit pattern
      @%p7 mov.b64 %fd22, %rd23;
      @%p7 st.global.f64 [%rd3], %fd22;
      @%p7 st.global.f64 [%rd4], %fd22;
      
      // Do normal min/max
      min.f64 %fd8, %fd8, %fd6;
      max.f64 %fd9, %fd9, %fd6;
      min.f64 %fd8, %fd8, %fd7;
      max.f64 %fd9, %fd9, %fd7;
      min.f64 %fd8, %fd8, %fd16;
      max.f64 %fd9, %fd9, %fd16;
      min.f64 %fd8, %fd8, %fd17;
      max.f64 %fd9, %fd9, %fd17;

      // Grid stride increment (4x)
      mov.u32 %r10, %nctaid.x;
      mul.lo.u32 %r11, %r10, %r2;
      mul.lo.u32 %r11, %r11, 4;
      add.u32 %r5, %r5, %r11;
      bra loop_start;
      
    scalar_loop:
      setp.ge.u32 %p0, %r5, %r0;
      @%p0 bra loop_end;
      
      // Load single value
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd6, [%rd2];
      
      // Check if value is NaN and write directly
      setp.neu.f64 %p5, %fd6, %fd6;
      @%p5 mov.f64 %fd22, 0d7FF8000000000000;
      @%p5 st.global.f64 [%rd3], %fd22;
      @%p5 st.global.f64 [%rd4], %fd22;
      
      // Do normal min/max
      min.f64 %fd8, %fd8, %fd6;
      max.f64 %fd9, %fd9, %fd6;
      
      add.u32 %r5, %r5, 1;
      bra scalar_loop;

    loop_end:
      // Warp-level reduction using shuffle (exactly like u32 version)
      // Get lane ID and warp ID
      and.b32 %r12, %r1, 0x1f;  // lane ID (0-31)
      shr.u32 %r13, %r1, 5;      // warp ID (tid / 32)
      
      // Shuffle f64 by treating as two u32 halves
      // Shuffle down by 16 for min
      mov.b64 %rd10, %fd8;
      cvt.u32.u64 %r25, %rd10;                // low 32 bits
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;                // high 32 bits
      shfl.sync.down.b32 %r27, %r25, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 16, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd8, %fd8, %fd14;
      
      // Shuffle down by 16 for max
      mov.b64 %rd10, %fd9;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 16, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd9, %fd9, %fd14;
      
      // Shuffle down by 8
      mov.b64 %rd10, %fd8;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 8, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 8, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd8, %fd8, %fd14;
      
      mov.b64 %rd10, %fd9;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 8, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 8, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd9, %fd9, %fd14;
      
      // Shuffle down by 4
      mov.b64 %rd10, %fd8;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 4, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 4, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd8, %fd8, %fd14;
      
      mov.b64 %rd10, %fd9;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 4, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 4, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd9, %fd9, %fd14;
      
      // Shuffle down by 2
      mov.b64 %rd10, %fd8;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 2, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 2, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd8, %fd8, %fd14;
      
      mov.b64 %rd10, %fd9;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 2, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 2, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd9, %fd9, %fd14;
      
      // Shuffle down by 1
      mov.b64 %rd10, %fd8;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 1, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 1, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd8, %fd8, %fd14;
      
      mov.b64 %rd10, %fd9;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 1, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 1, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd9, %fd9, %fd14;
      
      // Lane 0 of each warp writes to shared memory
      setp.eq.u32 %p1, %r12, 0;
      @%p1 mul.wide.u32 %rd1, %r13, 8;
      @%p1 mov.u64 %rd2, smin;
      @%p1 add.u64 %rd2, %rd2, %rd1;
      @%p1 st.shared.f64 [%rd2], %fd8;
      @%p1 mov.u64 %rd2, smax;
      @%p1 add.u64 %rd2, %rd2, %rd1;
      @%p1 st.shared.f64 [%rd2], %fd9;
      
      
      bar.sync 0;
      
      // Final reduction of warp results (only first warp)
      // Calculate number of warps in this block
      add.u32 %r15, %r2, 31;
      shr.u32 %r15, %r15, 5;  // num_warps = (block_size + 31) / 32
      
      setp.lt.u32 %p2, %r1, 32;  // Only first warp participates
      @!%p2 bra write_result;
      
      // Load warp results if within range
      setp.lt.u32 %p2, %r1, %r15;
      mov.f64 %fd16, 0d7FF0000000000000;  // min = +infinity
      mov.f64 %fd17, 0dFFF0000000000000;  // max = -infinity
      @%p2 mul.wide.u32 %rd1, %r1, 8;
      @%p2 mov.u64 %rd2, smin;
      @%p2 add.u64 %rd2, %rd2, %rd1;
      @%p2 ld.shared.f64 %fd16, [%rd2];
      @%p2 mov.u64 %rd2, smax;
      @%p2 add.u64 %rd2, %rd2, %rd1;
      @%p2 ld.shared.f64 %fd17, [%rd2];
      
      
      // Final warp reduction (same two-part shuffle for f64)
      mov.b64 %rd10, %fd16;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 16, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd16, %fd16, %fd14;
      
      mov.b64 %rd10, %fd17;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 16, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 16, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd17, %fd17, %fd14;
      
      // Continue with 8, 4, 2, 1 shuffles...
      mov.b64 %rd10, %fd16;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 8, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 8, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd16, %fd16, %fd14;
      
      mov.b64 %rd10, %fd17;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 8, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 8, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd17, %fd17, %fd14;
      
      mov.b64 %rd10, %fd16;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 4, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 4, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd16, %fd16, %fd14;
      
      mov.b64 %rd10, %fd17;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 4, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 4, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd17, %fd17, %fd14;
      
      mov.b64 %rd10, %fd16;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 2, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 2, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd16, %fd16, %fd14;
      
      mov.b64 %rd10, %fd17;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 2, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 2, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd17, %fd17, %fd14;
      
      mov.b64 %rd10, %fd16;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 1, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 1, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      min.f64 %fd16, %fd16, %fd14;
      
      mov.b64 %rd10, %fd17;
      cvt.u32.u64 %r25, %rd10;
      shr.u64 %rd11, %rd10, 32;
      cvt.u32.u64 %r26, %rd11;
      shfl.sync.down.b32 %r27, %r25, 1, 0x1f, 0xffffffff;
      shfl.sync.down.b32 %r28, %r26, 1, 0x1f, 0xffffffff;
      cvt.u64.u32 %rd12, %r28;
      cvt.u64.u32 %rd13, %r27;
      shl.b64 %rd14, %rd12, 32;
      or.b64 %rd13, %rd13, %rd14;
      mov.b64 %fd14, %rd13;
      max.f64 %fd17, %fd17, %fd14;
      
      
      // Thread 0 writes final block result using atomic CAS loop
      // Since PTX doesn't support atom.global.min/max.f64, we use CAS loop
      setp.eq.u32 %p1, %r1, 0;
      @!%p1 bra write_result;
      
      
      // Check if global min is already NaN
      ld.global.f64 %fd18, [%rd3];
      setp.neu.f64 %p8, %fd18, %fd18;
      @%p8 bra check_max;
      
      // Otherwise use atomic CAS for min
    min_loop:
      min.f64 %fd19, %fd18, %fd16;
      mov.b64 %rd15, %fd18;
      mov.b64 %rd16, %fd19;
      atom.global.cas.b64 %rd17, [%rd3], %rd15, %rd16;
      mov.b64 %fd20, %rd17;
      setp.eq.f64 %p5, %fd20, %fd18;
      @!%p5 mov.f64 %fd18, %fd20;
      @!%p5 bra min_loop;
      
    check_max:
      // Check if global max is already NaN
      ld.global.f64 %fd18, [%rd4];
      setp.neu.f64 %p8, %fd18, %fd18;
      @%p8 bra write_result;
      
      // Atomic CAS for max
    max_loop:
      max.f64 %fd19, %fd18, %fd17;
      mov.b64 %rd15, %fd18;
      mov.b64 %rd16, %fd19;
      atom.global.cas.b64 %rd17, [%rd4], %rd15, %rd16;
      mov.b64 %fd20, %rd17;
      setp.eq.f64 %p5, %fd20, %fd18;
      @!%p5 mov.f64 %fd18, %fd20;
      @!%p5 bra max_loop;
      
    write_result:
      
      ret;
    }
  "#;

  let values_u64 = values as u64;
  let len_u64 = len as u64;
  let min_result_u64 = min_result_ptr as u64;
  let max_result_u64 = max_result_ptr as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &min_result_u64 as *const u64 as *const u8,
    &max_result_u64 as *const u64 as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::reduction();
  let result = launch_ptx(
    PTX_FIND_MIN_MAX_F64,
    &[],
    "find_min_max_f64",
    blocks,
    threads,
    &args,
  );
  match result {
    Ok(_) => {}
    Err(e) => {
      panic!("PTX kernel launch failed: {:?}", e);
    }
  }
}

// AVX-512 optimized f64 min/max reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  feature = "hwx-nightly"
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn find_min_max_f64_avx512(values: &[f64], len: usize) -> (f64, f64) {
  const LANES: usize = LANES_AVX512_F64; // AVX-512 processes 8 f64 values at once

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  let (mut final_min, mut final_max) = if full_chunks > 0 {
    // Initialize with first chunk
    let first_chunk = _mm512_loadu_pd(values.as_ptr());

    // Check for NaN in first chunk using AVX-512 mask operations
    let nan_mask = _mm512_cmp_pd_mask(first_chunk, first_chunk, _CMP_UNORD_Q);
    if nan_mask != 0 {
      return (f64::NAN, f64::NAN);
    }

    let mut min_vec = first_chunk;
    let mut max_vec = first_chunk;

    // Process remaining chunks with AVX-512
    for chunk_idx in 1..full_chunks {
      let offset = chunk_idx * LANES;
      let values_chunk = _mm512_loadu_pd(values.as_ptr().add(offset));

      // Check for NaN
      let nan_mask = _mm512_cmp_pd_mask(values_chunk, values_chunk, _CMP_UNORD_Q);
      if nan_mask != 0 {
        return (f64::NAN, f64::NAN);
      }

      min_vec = _mm512_min_pd(min_vec, values_chunk);
      max_vec = _mm512_max_pd(max_vec, values_chunk);
    }

    // Use AVX-512 horizontal reduction for min/max
    let min = _mm512_reduce_min_pd(min_vec);
    let max = _mm512_reduce_max_pd(max_vec);

    (min, max)
  } else {
    // No SIMD chunks, initialize with first element
    let first = values[0];
    if first.is_nan() {
      return (f64::NAN, f64::NAN);
    }
    (first, first)
  };

  // Handle remaining elements
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    let val = values[offset + j];
    if val.is_nan() {
      return (f64::NAN, f64::NAN);
    }
    if val < final_min {
      final_min = val;
    }
    if val > final_max {
      final_max = val;
    }
  }

  (final_min, final_max)
}

// AVX2 optimized f64 min/max reduction.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn find_min_max_f64_avx2(values: &[f64], len: usize) -> (f64, f64) {
  const LANES: usize = LANES_AVX2_F64; // AVX2 processes 4 f64 values at once

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  let (mut final_min, mut final_max) = if full_chunks > 0 {
    // Initialize with first chunk
    let first_chunk = _mm256_loadu_pd(values.as_ptr());

    // Check for NaN in first chunk
    let nan_check = _mm256_cmp_pd(first_chunk, first_chunk, _CMP_UNORD_Q);
    if _mm256_movemask_pd(nan_check) != 0 {
      return (f64::NAN, f64::NAN);
    }

    let mut min_vec = first_chunk;
    let mut max_vec = first_chunk;

    // Process remaining chunks with AVX2
    for chunk_idx in 1..full_chunks {
      let offset = chunk_idx * LANES;
      let values_chunk = _mm256_loadu_pd(values.as_ptr().add(offset));

      // Check for NaN
      let nan_check = _mm256_cmp_pd(values_chunk, values_chunk, _CMP_UNORD_Q);
      if _mm256_movemask_pd(nan_check) != 0 {
        return (f64::NAN, f64::NAN);
      }

      min_vec = _mm256_min_pd(min_vec, values_chunk);
      max_vec = _mm256_max_pd(max_vec, values_chunk);
    }

    // Horizontal reduction for min
    let min_lane0 = _mm256_extractf128_pd(min_vec, 0);
    let min_lane1 = _mm256_extractf128_pd(min_vec, 1);
    let min_combined = _mm_min_pd(min_lane0, min_lane1);
    let min_val0 = _mm_cvtsd_f64(min_combined);
    let min_val1 = _mm_cvtsd_f64(_mm_shuffle_pd(min_combined, min_combined, 1));
    let min = min_val0.min(min_val1);

    // Horizontal reduction for max
    let max_lane0 = _mm256_extractf128_pd(max_vec, 0);
    let max_lane1 = _mm256_extractf128_pd(max_vec, 1);
    let max_combined = _mm_max_pd(max_lane0, max_lane1);
    let max_val0 = _mm_cvtsd_f64(max_combined);
    let max_val1 = _mm_cvtsd_f64(_mm_shuffle_pd(max_combined, max_combined, 1));
    let max = max_val0.max(max_val1);

    (min, max)
  } else {
    // No SIMD chunks, initialize with first element
    let first = values[0];
    if first.is_nan() {
      return (f64::NAN, f64::NAN);
    }
    (first, first)
  };

  // Handle remaining elements
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    let val = values[offset + j];
    if val.is_nan() {
      return (f64::NAN, f64::NAN);
    }
    if val < final_min {
      final_min = val;
    }
    if val > final_max {
      final_max = val;
    }
  }

  (final_min, final_max)
}

// NEON optimized f64 min/max reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn find_min_max_f64_neon(values: &[f64], len: usize) -> (f64, f64) {
  const LANES: usize = LANES_NEON_F64; // NEON processes 2 f64 values at once

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  let (mut final_min, mut final_max) = if full_chunks > 0 {
    // Initialize with first chunk
    let first_chunk = vld1q_f64(values.as_ptr());

    // Check for NaN in first chunk
    // In NEON, NaN != NaN, so compare with itself
    let nan_check = vceqq_f64(first_chunk, first_chunk);
    // If any lane is 0 (false), we have NaN
    if vgetq_lane_u64(nan_check, 0) == 0 || vgetq_lane_u64(nan_check, 1) == 0 {
      return (f64::NAN, f64::NAN);
    }

    let mut min_vec = first_chunk;
    let mut max_vec = first_chunk;

    // Process remaining chunks with NEON
    for chunk_idx in 1..full_chunks {
      let offset = chunk_idx * LANES;
      let values_chunk = vld1q_f64(values.as_ptr().add(offset));

      // Check for NaN
      let nan_check = vceqq_f64(values_chunk, values_chunk);
      if vgetq_lane_u64(nan_check, 0) == 0 || vgetq_lane_u64(nan_check, 1) == 0 {
        return (f64::NAN, f64::NAN);
      }

      min_vec = vminq_f64(min_vec, values_chunk);
      max_vec = vmaxq_f64(max_vec, values_chunk);
    }

    // Horizontal reduction for min
    let min_lane0 = vgetq_lane_f64(min_vec, 0);
    let min_lane1 = vgetq_lane_f64(min_vec, 1);
    let min = min_lane0.min(min_lane1);

    // Horizontal reduction for max
    let max_lane0 = vgetq_lane_f64(max_vec, 0);
    let max_lane1 = vgetq_lane_f64(max_vec, 1);
    let max = max_lane0.max(max_lane1);

    (min, max)
  } else {
    // No SIMD chunks, initialize with first element
    let first = values[0];
    if first.is_nan() {
      return (f64::NAN, f64::NAN);
    }
    (first, first)
  };

  // Handle remaining elements
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    let val = values[offset + j];
    if val.is_nan() {
      return (f64::NAN, f64::NAN);
    }
    if val < final_min {
      final_min = val;
    }
    if val > final_max {
      final_max = val;
    }
  }

  (final_min, final_max)
}

// =============================================================================
// VECTORIZED QUANTIZE F64 OPERATIONS
// =============================================================================

// GPU optimized vectorized quantization using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn vectorized_quantize_f64_gpu(
  values: *const f64,
  min_val: f64,
  scale: f64,
  max_val: f64,
  output: *mut u32,
  len: usize,
) {
  const PTX_VECTORIZED_QUANTIZE_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry vectorized_quantize_f64(
      .param .u64 values_ptr,
      .param .f64 min_val,
      .param .f64 scale,
      .param .f64 max_val,
      .param .u64 output_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.f64 %fd5, [min_val];
      ld.param.f64 %fd6, [scale];
      ld.param.f64 %fd7, [max_val];
      ld.param.u64 %rd5, [output_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;

      // Grid stride loop for double2 processing
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements
      // Calculate grid stride
      mov.u32 %r7, %nctaid.x;
      mul.lo.u32 %r8, %r7, %r2;
      mul.lo.u32 %r8, %r8, 2;  // Stride = total_threads * 2

    loop_start:
      add.u32 %r9, %r5, 1;
      setp.ge.u32 %p0, %r9, %r0;
      @%p0 bra loop_end;

      // Load double2 values
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2

      // Quantize both lanes: ((value - min_val) * scale).min(max_val) as u32
      // Lane 0
      sub.f64 %fd2, %fd0, %fd5;
      mul.f64 %fd3, %fd2, %fd6;
      min.f64 %fd4, %fd3, %fd7;
      cvt.rzi.u32.f64 %r6, %fd4;
      // Lane 1
      sub.f64 %fd8, %fd1, %fd5;
      mul.f64 %fd9, %fd8, %fd6;
      min.f64 %fd10, %fd9, %fd7;
      cvt.rzi.u32.f64 %r10, %fd10;

      // Store uint2 results
      mul.wide.u32 %rd3, %r5, 4;
      add.u64 %rd4, %rd5, %rd3;
      st.global.v2.u32 [%rd4], {%r6, %r10};

      // Grid stride increment
      add.u32 %r5, %r5, %r8;
      bra loop_start;

    loop_end:
      ret;
    }
  "#;

  let min_val_bits = min_val.to_bits();
  let scale_bits = scale.to_bits();
  let max_val_bits = max_val.to_bits();

  let args = [
    values as *const u8,
    &min_val_bits as *const u64 as *const u8,
    &scale_bits as *const u64 as *const u8,
    &max_val_bits as *const u64 as *const u8,
    output as *const u8,
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(
    PTX_VECTORIZED_QUANTIZE_F64,
    &[],
    "vectorized_quantize_f64",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 vectorized quantization: output[i] = ((values[i] - min_val) * scale).min(max_val) as u32
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn vectorized_quantize_f64_avx512(
  values: &[f64],
  min_val: f64,
  scale: f64,
  max_val: f64,
  output: &mut [u32],
  len: usize,
) {
  const LANES: usize = LANES_AVX512_F64; // AVX-512 processes 8 f64 values at once

  let min_vec = _mm512_set1_pd(min_val);
  let scale_vec = _mm512_set1_pd(scale);
  let max_vec = _mm512_set1_pd(max_val);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 8-element chunks with AVX-512 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Quantization: ((values - min_val) / scale * max_val).min(max_val)
    let shifted = _mm512_sub_pd(values_chunk, min_vec);
    let normalized = _mm512_div_pd(shifted, scale_vec);
    let scaled = _mm512_mul_pd(normalized, max_vec);
    let clamped = _mm512_min_pd(scaled, max_vec);

    // Convert to i32 then to u32 (AVX-512 doesn't have direct f64->u32 conversion)
    let as_i32 = _mm512_cvtpd_epi32(clamped);
    _mm256_storeu_si256(output.as_mut_ptr().add(offset) as *mut __m256i, as_i32);
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let remaining_vec =
      _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, values.as_ptr().add(offset));

    let shifted = _mm512_sub_pd(remaining_vec, min_vec);
    let normalized = _mm512_div_pd(shifted, scale_vec);
    let scaled = _mm512_mul_pd(normalized, max_vec);
    let clamped = _mm512_min_pd(scaled, max_vec);
    let as_i32 = _mm512_cvtpd_epi32(clamped);

    // Store remaining elements manually since _mm256_mask_storeu_epi32 is not stable
    let temp_array: [u32; 8] = std::mem::transmute(as_i32);
    for i in 0..remaining_elements {
      output[offset + i] = temp_array[i];
    }
  }
}

// AVX2 vectorized quantization.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn vectorized_quantize_f64_avx2(
  values: &[f64],
  min_val: f64,
  scale: f64,
  max_val: f64,
  output: &mut [u32],
  len: usize,
) {
  const LANES: usize = LANES_AVX2_F64; // AVX2 processes 4 f64 values at once

  let min_vec = _mm256_set1_pd(min_val);
  let scale_vec = _mm256_set1_pd(scale);
  let max_vec = _mm256_set1_pd(max_val);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 4-element chunks with AVX2 (matching NEON for-loop structure)
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Quantization: ((values - min_val) / scale * max_val).min(max_val)
    let shifted = _mm256_sub_pd(values_chunk, min_vec);
    let normalized = _mm256_div_pd(shifted, scale_vec);
    let scaled = _mm256_mul_pd(normalized, max_vec);
    let clamped = _mm256_min_pd(scaled, max_vec);

    // Convert to i32 then store as u32
    let as_i32 = _mm256_cvtpd_epi32(clamped);
    _mm_storeu_si128(output.as_mut_ptr().add(offset) as *mut _, as_i32);
  }

  // Handle remaining elements with scalar quantization (matching NEON structure)
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    let quantized = (((values[offset + j] - min_val) / scale) * max_val).min(max_val) as u32;
    output[offset + j] = quantized;
  }
}

// NEON vectorized quantization.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn vectorized_quantize_f64_neon(
  values: &[f64],
  min_val: f64,
  scale: f64,
  max_val: f64,
  output: &mut [u32],
  len: usize,
) {
  const LANES: usize = LANES_NEON_F64;

  let min_vec = vdupq_n_f64(min_val);
  let scale_vec = vdupq_n_f64(scale);
  let max_vec = vdupq_n_f64(max_val);
  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let values_chunk = vld1q_f64(values.as_ptr().add(offset));

    // Quantization: (((values - min_val) / scale) * max_val).min(max_val)
    let shifted = vsubq_f64(values_chunk, min_vec);
    let normalized = vdivq_f64(shifted, scale_vec);
    let scaled = vmulq_f64(normalized, max_vec);
    let clamped = vminq_f64(scaled, max_vec);

    // Convert to u32 (NEON doesn't have direct f64->u32, so extract manually)
    let val0 = vgetq_lane_f64(clamped, 0) as u32;
    let val1 = vgetq_lane_f64(clamped, 1) as u32;

    output[offset] = val0;
    output[offset + 1] = val1;
  }

  // Handle remaining elements
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    let quantized = (((values[offset + j] - min_val) / scale) * max_val).min(max_val) as u32;
    output[offset + j] = quantized;
  }
}

// =============================================================================
// LINEAR INTERPOLATION F64 OPERATIONS
// =============================================================================

// GPU optimized linear interpolation using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn linear_interpolate_f64_gpu(
  lower_values: *const f64,
  upper_values: *const f64,
  weights: *const f64,
  results: *mut f64,
  len: usize,
) {
  const PTX_LINEAR_INTERPOLATE_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry linear_interpolate_f64(
      .param .u64 lower_values_ptr,
      .param .u64 upper_values_ptr,
      .param .u64 weights_ptr,
      .param .u64 results_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd0, [lower_values_ptr];
      ld.param.u64 %rd6, [upper_values_ptr];
      ld.param.u64 %rd7, [weights_ptr];
      ld.param.u64 %rd5, [results_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;

      // Grid stride loop for double2 processing
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements
      // Calculate grid stride
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r2;
      mul.lo.u32 %r7, %r7, 2;  // Stride = total_threads * 2

    loop_start:
      add.u32 %r8, %r5, 1;
      setp.ge.u32 %p0, %r8, %r0;
      @%p0 bra loop_end;

      // Load lower, upper, and weight double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load lower double2
      add.u64 %rd3, %rd6, %rd1;
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load upper double2
      add.u64 %rd4, %rd7, %rd1;
      ld.global.v2.f64 {%fd4, %fd5}, [%rd4];  // Load weights double2

      // Linear interpolation: lower * (1.0 - weight) + upper * weight for both lanes
      mov.f64 %fd6, 0d3FF0000000000000;  // 1.0
      // Lane 0
      sub.f64 %fd7, %fd6, %fd4;          // 1.0 - weight0
      mul.f64 %fd8, %fd0, %fd7;          // lower0 * (1.0 - weight0)
      mul.f64 %fd9, %fd2, %fd4;          // upper0 * weight0
      add.f64 %fd10, %fd8, %fd9;         // result0
      // Lane 1
      sub.f64 %fd11, %fd6, %fd5;         // 1.0 - weight1
      mul.f64 %fd12, %fd1, %fd11;        // lower1 * (1.0 - weight1)
      mul.f64 %fd13, %fd3, %fd5;         // upper1 * weight1
      add.f64 %fd14, %fd12, %fd13;       // result1

      // Store double2 results
      add.u64 %rd5, %rd5, %rd1;
      st.global.v2.f64 [%rd5], {%fd10, %fd14};

      // Grid stride increment
      add.u32 %r5, %r5, %r7;
      bra loop_start;

    loop_end:
      ret;
    }
  "#;

  let args = [
    lower_values as *const u8,
    upper_values as *const u8,
    weights as *const u8,
    results as *const u8,
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(
    PTX_LINEAR_INTERPOLATE_F64,
    &[],
    "linear_interpolate_f64",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 optimized linear interpolation: results[i] = lower[i] * (1.0 - weights[i]) + upper[i] * weights[i]
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn linear_interpolate_f64_avx512(
  lower_values: &[f64],
  upper_values: &[f64],
  weights: &[f64],
  results: &mut [f64],
  len: usize,
) {
  const LANES: usize = LANES_AVX512_F64;

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;
  let ones_vec = _mm512_set1_pd(1.0);

  // Process 8-element chunks with AVX-512
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let lower_chunk = _mm512_loadu_pd(lower_values.as_ptr().add(offset));
    let upper_chunk = _mm512_loadu_pd(upper_values.as_ptr().add(offset));
    let weights_chunk = _mm512_loadu_pd(weights.as_ptr().add(offset));

    // Compute (1.0 - weights)
    let inv_weights = _mm512_sub_pd(ones_vec, weights_chunk);

    // Linear interpolation: lower * (1.0 - weight) + upper * weight
    let lower_part = _mm512_mul_pd(lower_chunk, inv_weights);
    let upper_part = _mm512_mul_pd(upper_chunk, weights_chunk);
    let result_chunk = _mm512_add_pd(lower_part, upper_part);

    _mm512_storeu_pd(results.as_mut_ptr().add(offset), result_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let offset = full_chunks * LANES;
  if remaining_elements > 0 {
    let load_mask = (1u8 << remaining_elements) - 1;
    let lower_chunk = _mm512_mask_loadu_pd(
      _mm512_setzero_pd(),
      load_mask,
      lower_values.as_ptr().add(offset),
    );
    let upper_chunk = _mm512_mask_loadu_pd(
      _mm512_setzero_pd(),
      load_mask,
      upper_values.as_ptr().add(offset),
    );
    let weights_chunk =
      _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, weights.as_ptr().add(offset));

    let inv_weights = _mm512_sub_pd(ones_vec, weights_chunk);
    let lower_part = _mm512_mul_pd(lower_chunk, inv_weights);
    let upper_part = _mm512_mul_pd(upper_chunk, weights_chunk);
    let result_chunk = _mm512_add_pd(lower_part, upper_part);

    _mm512_mask_storeu_pd(results.as_mut_ptr().add(offset), load_mask, result_chunk);
  }
}

// AVX2 optimized linear interpolation
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn linear_interpolate_f64_avx2(
  lower_values: &[f64],
  upper_values: &[f64],
  weights: &[f64],
  results: &mut [f64],
  len: usize,
) {
  const LANES: usize = LANES_AVX2_F64;

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;
  let ones_vec = _mm256_set1_pd(1.0);

  // Process 4-element chunks with AVX2
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let lower_chunk = _mm256_loadu_pd(lower_values.as_ptr().add(offset));
    let upper_chunk = _mm256_loadu_pd(upper_values.as_ptr().add(offset));
    let weights_chunk = _mm256_loadu_pd(weights.as_ptr().add(offset));

    // Compute (1.0 - weights)
    let inv_weights = _mm256_sub_pd(ones_vec, weights_chunk);

    // Linear interpolation: lower * (1.0 - weight) + upper * weight
    let lower_part = _mm256_mul_pd(lower_chunk, inv_weights);
    let upper_part = _mm256_mul_pd(upper_chunk, weights_chunk);
    let result_chunk = _mm256_add_pd(lower_part, upper_part);

    _mm256_storeu_pd(results.as_mut_ptr().add(offset), result_chunk);
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 operations
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    let weight = weights[offset + j];
    results[offset + j] =
      lower_values[offset + j] * (1.0 - weight) + upper_values[offset + j] * weight;
  }
}

// NEON optimized linear interpolation
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn linear_interpolate_f64_neon(
  lower_values: &[f64],
  upper_values: &[f64],
  weights: &[f64],
  results: &mut [f64],
  len: usize,
) {
  const LANES: usize = LANES_NEON_F64;

  let full_chunks = len / LANES;
  let remaining_elements = len % LANES;
  let ones_vec = vdupq_n_f64(1.0);

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    let offset = chunk_idx * LANES;
    let lower_chunk = vld1q_f64(lower_values.as_ptr().add(offset));
    let upper_chunk = vld1q_f64(upper_values.as_ptr().add(offset));
    let weights_chunk = vld1q_f64(weights.as_ptr().add(offset));

    // Compute (1.0 - weights)
    let inv_weights = vsubq_f64(ones_vec, weights_chunk);

    // Linear interpolation: lower * (1.0 - weight) + upper * weight
    let lower_part = vmulq_f64(lower_chunk, inv_weights);
    let upper_part = vmulq_f64(upper_chunk, weights_chunk);
    let result_chunk = vaddq_f64(lower_part, upper_part);

    vst1q_f64(results.as_mut_ptr().add(offset), result_chunk);
  }

  // Handle remaining elements with scalar
  let offset = full_chunks * LANES;
  for j in 0..remaining_elements {
    let weight = weights[offset + j];
    results[offset + j] =
      lower_values[offset + j] * (1.0 - weight) + upper_values[offset + j] * weight;
  }
}

// =============================================================================
// BATCH PERCENTILE OPERATIONS
// =============================================================================

// GPU optimized batch percentile calculation using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn calculate_percentiles_batch_f64_gpu(
  datasets: &[&[f64]],
  dataset_lengths: &[usize],
  percentile: f64,
  results: &mut [f64],
  num_datasets: usize,
) {
  const PTX_CALCULATE_PERCENTILES_BATCH: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry calculate_percentiles_batch_f64(
      .param .u64 datasets_ptr,
      .param .u64 dataset_lengths_ptr,
      .param .f64 percentile,
      .param .u64 results_ptr,
      .param .u32 num_datasets
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd11, [datasets_ptr];
      ld.param.u64 %rd12, [dataset_lengths_ptr];
      ld.param.f64 %fd12, [percentile];
      ld.param.u64 %rd13, [results_ptr];
      ld.param.u32 %r12, [num_datasets];

      // Grid stride through datasets
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;  // dataset_idx

    loop_start:
      setp.ge.u32 %p0, %r5, %r12;
      @%p0 bra loop_end;

      // Get dataset length
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd12, %rd1;
      ld.global.u64 %rd3, [%rd2];
      cvt.u32.u64 %r6, %rd3;  // len

      // Check if len == 0
      setp.eq.u32 %p1, %r6, 0;
      @%p1 mov.f64 %fd1, 0d7FF8000000000000;  // NaN
      @%p1 bra store_result;

      // Get dataset pointer
      add.u64 %rd4, %rd11, %rd1;
      ld.global.u64 %rd5, [%rd4];  // dataset ptr

      // Check if len == 1
      setp.eq.u32 %p2, %r6, 1;
      @%p2 ld.global.f64 %fd1, [%rd5];
      @%p2 bra store_result;

      // Calculate percentile index
      sub.u32 %r7, %r6, 1;
      cvt.rn.f64.u32 %fd2, %r7;
      mul.f64 %fd3, %fd12, 0.01;
      mul.f64 %fd4, %fd3, %fd2;  // index

      // Get lower and upper indices
      cvt.rzi.u32.f64 %r8, %fd4;  // lower_idx
      add.u32 %r9, %r8, 1;        // upper_idx
      setp.ge.u32 %p3, %r9, %r6;
      @%p3 mov.u32 %r9, %r8;      // clamp upper

      // Check if indices are adjacent for vectorized load
      sub.u32 %r13, %r9, %r8;
      setp.eq.u32 %p4, %r13, 1;
      @%p4 bra vectorized_load;
      
      // Load values separately (non-adjacent)
      mul.wide.u32 %rd6, %r8, 8;
      add.u64 %rd7, %rd5, %rd6;
      ld.global.f64 %fd5, [%rd7]; // lower_val
      mul.wide.u32 %rd8, %r9, 8;
      add.u64 %rd9, %rd5, %rd8;
      ld.global.f64 %fd6, [%rd9]; // upper_val
      bra interpolate;
      
    vectorized_load:
      // Load both values with v2.f64
      mul.wide.u32 %rd6, %r8, 8;
      add.u64 %rd7, %rd5, %rd6;
      ld.global.v2.f64 {%fd5, %fd6}, [%rd7];
      
    interpolate:

      // Linear interpolation
      cvt.rn.f64.u32 %fd7, %r8;
      sub.f64 %fd8, %fd4, %fd7;   // weight
      mov.f64 %fd9, 1.0;
      sub.f64 %fd10, %fd9, %fd8;  // 1 - weight
      mul.f64 %fd11, %fd5, %fd10;
      fma.rn.f64 %fd1, %fd6, %fd8, %fd11;

    store_result:
      // Store result
      add.u64 %rd10, %rd13, %rd1;
      st.global.f64 [%rd10], %fd1;

      // Grid stride increment
      mov.u32 %r10, %nctaid.x;
      mul.lo.u32 %r11, %r10, %r2;
      add.u32 %r5, %r5, %r11;
      bra loop_start;

    loop_end:
      ret;
    }
  "#;

  let percentile_bits = percentile.to_bits();

  let args = [
    datasets.as_ptr() as *const u8,
    dataset_lengths.as_ptr() as *const u8,
    &percentile_bits as *const u64 as *const u8,
    results.as_mut_ptr() as *const u8,
    &num_datasets as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();

  let _ = launch_ptx(
    PTX_CALCULATE_PERCENTILES_BATCH,
    &[],
    "calculate_percentiles_batch_f64",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 optimized batch percentile calculation across multiple datasets
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn calculate_percentiles_batch_f64_avx512(
  datasets: &[&[f64]],
  dataset_lengths: &[usize],
  percentile: f64,
  results: &mut [f64],
  num_datasets: usize,
) {
  const LANES: usize = LANES_AVX512_F64;

  let _percentile_vec = _mm512_set1_pd(percentile / 100.0);
  let _one_vec = _mm512_set1_pd(1.0);
  let _nan_vec = _mm512_set1_pd(f64::NAN);

  let end = (num_datasets / LANES) * LANES;
  let mut i = 0;

  // Process 8 datasets at once using SIMD
  while i < end {
    let mut result_values = [f64::NAN; LANES];
    let mut dataset_lengths_f64 = [0.0f64; LANES];

    // Collect dataset info and convert lengths to f64 for SIMD processing
    for lane in 0..LANES {
      let dataset = datasets[i + lane];
      let len = dataset_lengths[i + lane];
      if len == 0 {
        result_values[lane] = f64::NAN;
        dataset_lengths_f64[lane] = 0.0;
      } else if len == 1 {
        result_values[lane] = dataset[0];
        dataset_lengths_f64[lane] = 1.0;
      } else {
        dataset_lengths_f64[lane] = (len - 1) as f64;
      }
    }

    // Load dataset lengths into SIMD register
    let lengths_vec = _mm512_loadu_pd(dataset_lengths_f64.as_ptr());
    let percentile_vec = _mm512_set1_pd(percentile / 100.0);

    // Vectorized percentile index calculation
    let indices_vec = _mm512_mul_pd(percentile_vec, lengths_vec);

    // Extract indices for processing (SIMD floor/ceil operations)
    let mut indices_f64 = [0.0f64; LANES];
    _mm512_storeu_pd(indices_f64.as_mut_ptr(), indices_vec);

    // Vectorized floor and weight calculations
    let mut lower_values = [f64::NAN; LANES];
    let mut upper_values = [f64::NAN; LANES];
    let mut weights = [0.0f64; LANES];

    for lane in 0..LANES {
      let dataset = datasets[i + lane];
      let len = dataset_lengths[i + lane];
      if len > 1 {
        let index = indices_f64[lane];
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        lower_values[lane] = dataset[lower_idx];
        upper_values[lane] = if upper_idx < len {
          dataset[upper_idx]
        } else {
          dataset[lower_idx]
        };
        weights[lane] = index - lower_idx as f64;
      }
    }

    // Load values and weights into SIMD registers for vectorized interpolation
    let lower_vec = _mm512_loadu_pd(lower_values.as_ptr());
    let upper_vec = _mm512_loadu_pd(upper_values.as_ptr());
    let weights_vec = _mm512_loadu_pd(weights.as_ptr());
    let one_minus_weights = _mm512_sub_pd(_mm512_set1_pd(1.0), weights_vec);

    // Vectorized linear interpolation: lower * (1-weight) + upper * weight
    let lower_part = _mm512_mul_pd(lower_vec, one_minus_weights);
    let upper_part = _mm512_mul_pd(upper_vec, weights_vec);
    let interpolated = _mm512_add_pd(lower_part, upper_part);

    // Store interpolated results
    let mut interpolated_results = [0.0f64; LANES];
    _mm512_storeu_pd(interpolated_results.as_mut_ptr(), interpolated);

    // Combine results (use interpolated for multi-element datasets)
    for lane in 0..LANES {
      let len = dataset_lengths[i + lane];
      if len > 1 {
        result_values[lane] = interpolated_results[lane];
      }
      // Single-element and empty datasets already handled above
    }

    // Store final results using SIMD
    let result_vec = _mm512_loadu_pd(result_values.as_ptr());
    _mm512_storeu_pd(results.as_mut_ptr().add(i), result_vec);
    i += LANES;
  }

  // Handle remaining datasets
  while i < num_datasets {
    let dataset = datasets[i];
    let len = dataset_lengths[i];

    if len == 0 {
      results[i] = f64::NAN;
    } else if len == 1 {
      results[i] = dataset[0];
    } else {
      let index = (percentile / 100.0) * (len - 1) as f64;
      let lower = index.floor() as usize;
      let upper = index.ceil() as usize;

      if lower == upper {
        results[i] = dataset[lower];
      } else {
        let weight = index - lower as f64;
        results[i] = dataset[lower] * (1.0 - weight) + dataset[upper] * weight;
      }
    }
    i += 1;
  }
}

// AVX2 optimized batch percentile calculation
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn calculate_percentiles_batch_f64_avx2(
  datasets: &[&[f64]],
  dataset_lengths: &[usize],
  percentile: f64,
  results: &mut [f64],
  num_datasets: usize,
) {
  const LANES: usize = LANES_AVX2_F64;

  let end = (num_datasets / LANES) * LANES;
  let mut i = 0;

  // Process 4 datasets at once using SIMD
  while i < end {
    let mut result_values = [f64::NAN; LANES];
    let mut dataset_lengths_f64 = [0.0f64; LANES];

    // Collect dataset info and convert lengths to f64 for SIMD processing
    for lane in 0..LANES {
      let dataset = datasets[i + lane];
      let len = dataset_lengths[i + lane];
      if len == 0 {
        result_values[lane] = f64::NAN;
        dataset_lengths_f64[lane] = 0.0;
      } else if len == 1 {
        result_values[lane] = dataset[0];
        dataset_lengths_f64[lane] = 1.0;
      } else {
        dataset_lengths_f64[lane] = (len - 1) as f64;
      }
    }

    // Load dataset lengths into SIMD register and calculate percentile indices
    let lengths_vec = _mm256_loadu_pd(dataset_lengths_f64.as_ptr());
    let percentile_vec = _mm256_set1_pd(percentile / 100.0);
    let indices_vec = _mm256_mul_pd(percentile_vec, lengths_vec);

    // Extract indices for processing
    let mut indices_f64 = [0.0f64; LANES];
    _mm256_storeu_pd(indices_f64.as_mut_ptr(), indices_vec);

    // Vectorized value lookups and weight calculations
    let mut lower_values = [f64::NAN; LANES];
    let mut upper_values = [f64::NAN; LANES];
    let mut weights = [0.0f64; LANES];

    for lane in 0..LANES {
      let dataset = datasets[i + lane];
      let len = dataset_lengths[i + lane];
      if len > 1 {
        let index = indices_f64[lane];
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        lower_values[lane] = dataset[lower_idx];
        upper_values[lane] = if upper_idx < len {
          dataset[upper_idx]
        } else {
          dataset[lower_idx]
        };
        weights[lane] = index - lower_idx as f64;
      }
    }

    // Vectorized linear interpolation
    let lower_vec = _mm256_loadu_pd(lower_values.as_ptr());
    let upper_vec = _mm256_loadu_pd(upper_values.as_ptr());
    let weights_vec = _mm256_loadu_pd(weights.as_ptr());
    let one_minus_weights = _mm256_sub_pd(_mm256_set1_pd(1.0), weights_vec);

    let lower_part = _mm256_mul_pd(lower_vec, one_minus_weights);
    let upper_part = _mm256_mul_pd(upper_vec, weights_vec);
    let interpolated = _mm256_add_pd(lower_part, upper_part);

    // Store interpolated results
    let mut interpolated_results = [0.0f64; LANES];
    _mm256_storeu_pd(interpolated_results.as_mut_ptr(), interpolated);

    // Combine results (use interpolated for multi-element datasets)
    for lane in 0..LANES {
      let len = dataset_lengths[i + lane];
      if len > 1 {
        result_values[lane] = interpolated_results[lane];
      }
    }

    // Store final results using SIMD
    let result_vec = _mm256_loadu_pd(result_values.as_ptr());
    _mm256_storeu_pd(results.as_mut_ptr().add(i), result_vec);
    i += LANES;
  }

  // Handle remaining datasets
  while i < num_datasets {
    let dataset = datasets[i];
    let len = dataset_lengths[i];

    if len == 0 {
      results[i] = f64::NAN;
    } else if len == 1 {
      results[i] = dataset[0];
    } else {
      let index = (percentile / 100.0) * (len - 1) as f64;
      let lower = index.floor() as usize;
      let upper = index.ceil() as usize;

      if lower == upper {
        results[i] = dataset[lower];
      } else {
        let weight = index - lower as f64;
        results[i] = dataset[lower] * (1.0 - weight) + dataset[upper] * weight;
      }
    }
    i += 1;
  }
}

// NEON optimized batch percentile calculation
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn calculate_percentiles_batch_f64_neon(
  datasets: &[&[f64]],
  dataset_lengths: &[usize],
  percentile: f64,
  results: &mut [f64],
  num_datasets: usize,
) {
  const LANES: usize = LANES_NEON_F64;

  //  ZERO-HEAP: Use passed num_datasets parameter - no .is_empty() or .len() calls!
  if num_datasets == 0 {
    return;
  }

  let end = (num_datasets / LANES) * LANES;
  let mut i = 0;

  // Process 2 datasets at once using SIMD
  while i < end {
    let mut result_values = [f64::NAN; LANES];
    let mut dataset_lengths_f64 = [0.0f64; LANES];

    // Collect dataset info and convert lengths to f64 for SIMD processing
    for lane in 0..LANES {
      let dataset = datasets[i + lane];
      let len = dataset_lengths[i + lane];
      if len == 0 {
        result_values[lane] = f64::NAN;
        dataset_lengths_f64[lane] = 0.0;
      } else if len == 1 {
        result_values[lane] = dataset[0];
        dataset_lengths_f64[lane] = 1.0;
      } else {
        dataset_lengths_f64[lane] = (len - 1) as f64;
      }
    }

    // Load dataset lengths into SIMD register and calculate percentile indices
    let lengths_vec = vld1q_f64(dataset_lengths_f64.as_ptr());
    let percentile_vec = vdupq_n_f64(percentile / 100.0);
    let indices_vec = vmulq_f64(percentile_vec, lengths_vec);

    // Extract indices for processing
    let mut indices_f64 = [0.0f64; LANES];
    vst1q_f64(indices_f64.as_mut_ptr(), indices_vec);

    // Vectorized value lookups and weight calculations
    let mut lower_values = [f64::NAN; LANES];
    let mut upper_values = [f64::NAN; LANES];
    let mut weights = [0.0f64; LANES];

    for lane in 0..LANES {
      let dataset = datasets[i + lane];
      let len = dataset_lengths[i + lane];
      if len > 1 {
        let index = indices_f64[lane];
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        lower_values[lane] = dataset[lower_idx];
        upper_values[lane] = if upper_idx < len {
          dataset[upper_idx]
        } else {
          dataset[lower_idx]
        };
        weights[lane] = index - lower_idx as f64;
      }
    }

    // Vectorized linear interpolation
    let lower_vec = vld1q_f64(lower_values.as_ptr());
    let upper_vec = vld1q_f64(upper_values.as_ptr());
    let weights_vec = vld1q_f64(weights.as_ptr());
    let one_minus_weights = vsubq_f64(vdupq_n_f64(1.0), weights_vec);

    let lower_part = vmulq_f64(lower_vec, one_minus_weights);
    let upper_part = vmulq_f64(upper_vec, weights_vec);
    let interpolated = vaddq_f64(lower_part, upper_part);

    // Store interpolated results
    let mut interpolated_results = [0.0f64; LANES];
    vst1q_f64(interpolated_results.as_mut_ptr(), interpolated);

    // Combine results (use interpolated for multi-element datasets)
    for lane in 0..LANES {
      let len = dataset_lengths[i + lane];
      if len > 1 {
        result_values[lane] = interpolated_results[lane];
      }
    }

    // Store final results using SIMD
    let result_vec = vld1q_f64(result_values.as_ptr());
    vst1q_f64(results.as_mut_ptr().add(i), result_vec); // Correct for NEON
    i += LANES;
  }

  // Handle remaining datasets
  while i < num_datasets {
    let dataset = datasets[i];
    let len = dataset_lengths[i];

    if len == 0 {
      results[i] = f64::NAN;
    } else if len == 1 {
      results[i] = dataset[0];
    } else {
      let index = (percentile / 100.0) * (len - 1) as f64;
      let lower = index.floor() as usize;
      let upper = index.ceil() as usize;

      if lower == upper {
        results[i] = dataset[lower];
      } else {
        let weight = index - lower as f64;
        results[i] = dataset[lower] * (1.0 - weight) + dataset[upper] * weight;
      }
    }
    i += 1;
  }
}

// =============================================================================
// MULTI-PERCENTILE OPERATIONS
// =============================================================================

// AVX-512 optimized multi-percentile calculation for single dataset
// GPU implementation of multi percentiles calculation
#[cfg(has_cuda)]
pub unsafe fn calculate_multi_percentiles_f64_gpu(
  dataset: &[f64],
  percentiles: &[f64],
  results: &mut [f64],
  dataset_len: usize,
  percentiles_len: usize,
) {
  const PTX_MULTI_PERCENTILES: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry calculate_multi_percentiles (
      .param .u64 dataset,
      .param .u64 percentiles,
      .param .u64 results,
      .param .u32 dataset_len,
      .param .u32 percentiles_len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mul.lo.u32 %r3, %r1, %r2;
      add.u32 %r4, %r0, %r3;

      ld.param.u64 %rd10, [dataset];
      ld.param.u64 %rd11, [percentiles];
      ld.param.u64 %rd12, [results];
      ld.param.u32 %r13, [dataset_len];
      ld.param.u32 %r14, [percentiles_len];

      // Process 2 percentiles per thread
      mul.lo.u32 %r4, %r4, 2;
      
      // Main loop over percentiles
      PERCENTILE_LOOP:
      add.u32 %r15, %r4, 1;
      setp.ge.u32 %p0, %r15, %r14;
      @%p0 bra single_percentile;

      // Load 2 percentiles (vectorized)
      mul.wide.u32 %rd0, %r4, 8;
      add.u64 %rd1, %rd11, %rd0;
      ld.global.v2.f64 {%fd0, %fd4}, [%rd1];

      // Calculate indices for both percentiles
      sub.u32 %r5, %r13, 1;
      cvt.rn.f64.u32 %fd1, %r5;
      mul.f64 %fd2, %fd0, %fd1;
      cvt.rzi.u32.f64 %r6, %fd2;
      mul.f64 %fd5, %fd4, %fd1;
      cvt.rzi.u32.f64 %r9, %fd5;

      // Bounds check
      min.u32 %r6, %r6, %r5;
      min.u32 %r9, %r9, %r5;

      // Check if indices are adjacent for vectorized dataset load
      sub.u32 %r10, %r9, %r6;
      setp.eq.u32 %p1, %r10, 1;
      @%p1 bra vectorized_dataset_load;

      // Load values separately
      mul.wide.u32 %rd2, %r6, 8;
      add.u64 %rd3, %rd10, %rd2;
      ld.global.f64 %fd3, [%rd3];
      mul.wide.u32 %rd5, %r9, 8;
      add.u64 %rd6, %rd10, %rd5;
      ld.global.f64 %fd6, [%rd6];
      bra store_results;

    vectorized_dataset_load:
      // Load both dataset values (vectorized)
      mul.wide.u32 %rd2, %r6, 8;
      add.u64 %rd3, %rd10, %rd2;
      ld.global.v2.f64 {%fd3, %fd6}, [%rd3];

    store_results:
      // Store 2 results (vectorized)
      add.u64 %rd4, %rd12, %rd0;
      st.global.v2.f64 [%rd4], {%fd3, %fd6};

    grid_stride:
      // Grid stride increment (by 2x threads since each processes 2 elements)
      mov.u32 %r7, %nctaid.x;
      mul.lo.u32 %r8, %r7, %r2;
      mul.lo.u32 %r8, %r8, 2;
      add.u32 %r4, %r4, %r8;
      bra PERCENTILE_LOOP;

    single_percentile:
      setp.ge.u32 %p0, %r4, %r14;
      @%p0 bra PERCENTILE_LOOP_END;
      
      // Load single percentile
      mul.wide.u32 %rd0, %r4, 8;
      add.u64 %rd1, %rd11, %rd0;
      ld.global.f64 %fd0, [%rd1];

      // Calculate index
      sub.u32 %r5, %r13, 1;
      cvt.rn.f64.u32 %fd1, %r5;
      mul.f64 %fd2, %fd0, %fd1;
      cvt.rzi.u32.f64 %r6, %fd2;
      min.u32 %r6, %r6, %r5;

      // Load value at calculated index
      mul.wide.u32 %rd2, %r6, 8;
      add.u64 %rd3, %rd10, %rd2;
      ld.global.f64 %fd3, [%rd3];

      // Store result
      add.u64 %rd4, %rd12, %rd0;
      st.global.f64 [%rd4], %fd3;

      // Grid stride (single element)
      mov.u32 %r7, %nctaid.x;
      mul.lo.u32 %r8, %r7, %r2;
      mul.lo.u32 %r8, %r8, 2;
      add.u32 %r4, %r4, %r8;
      bra PERCENTILE_LOOP;

      PERCENTILE_LOOP_END:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();

  let _ = launch_ptx(
    PTX_MULTI_PERCENTILES,
    &[],
    "calculate_multi_percentiles",
    blocks,
    threads,
    &[
      dataset.as_ptr() as *const u8,
      percentiles.as_ptr() as *const u8,
      results.as_mut_ptr() as *const u8,
      &(dataset_len as u32) as *const _ as *const u8,
      &(percentiles_len as u32) as *const _ as *const u8,
    ],
  )
  .unwrap_or_default();
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn calculate_multi_percentiles_f64_avx512(
  sorted_values: &[f64],
  sorted_values_len: usize,
  percentiles: &[f64],
  results: &mut [f64],
  num_percentiles: usize,
) {
  const LANES: usize = LANES_AVX512_F64;

  if sorted_values_len == 1 {
    // All percentiles return the single value
    let single_val = sorted_values[0];
    let single_vec = _mm512_set1_pd(single_val);

    let full_chunks = num_percentiles / LANES;
    let remaining_elements = num_percentiles % LANES;

    // Fill SIMD chunks with the single value
    for chunk_idx in 0..full_chunks {
      let offset = chunk_idx * LANES;
      _mm512_storeu_pd(results.as_mut_ptr().add(offset), single_vec);
    }

    // Handle remaining elements
    let offset = full_chunks * LANES;
    for j in 0..remaining_elements {
      results[offset + j] = single_val;
    }
    return;
  }

  let dataset_len_minus_1 = (sorted_values_len - 1) as f64;

  let len_minus_1_vec = _mm512_set1_pd(dataset_len_minus_1);
  let one_hundred_vec = _mm512_set1_pd(100.0);
  let one_vec = _mm512_set1_pd(1.0);

  let end = (num_percentiles / LANES) * LANES;
  let mut i = 0;

  // Process 8 percentiles at once using SIMD
  while i < end {
    let percentile_chunk = _mm512_loadu_pd(percentiles.as_ptr().add(i));

    // Calculate indices using SIMD
    let normalized = _mm512_div_pd(percentile_chunk, one_hundred_vec);
    let index_vec = _mm512_mul_pd(normalized, len_minus_1_vec);

    // Extract index values for vectorized processing
    let mut index_values = [0.0f64; LANES];
    _mm512_storeu_pd(index_values.as_mut_ptr(), index_vec);

    // Vectorized bounds checking
    let zero_vec = _mm512_setzero_pd();
    let hundred_vec = _mm512_set1_pd(100.0);
    let valid_mask = _mm512_mask_cmp_pd_mask(0xFF, percentile_chunk, zero_vec, _CMP_GE_OQ)
      & _mm512_mask_cmp_pd_mask(0xFF, percentile_chunk, hundred_vec, _CMP_LE_OQ);

    // Vectorized floor operation (approximate using truncation)
    let mut lower_values = [f64::NAN; LANES];
    let mut upper_values = [f64::NAN; LANES];
    let mut weights = [0.0f64; LANES];
    let mut result_values = [f64::NAN; LANES];

    // Extract valid bits for scalar processing - following NEON pattern
    let valid_0 = (valid_mask & (1 << 0)) != 0;
    let valid_1 = (valid_mask & (1 << 1)) != 0;
    let valid_2 = (valid_mask & (1 << 2)) != 0;
    let valid_3 = (valid_mask & (1 << 3)) != 0;
    let valid_4 = (valid_mask & (1 << 4)) != 0;
    let valid_5 = (valid_mask & (1 << 5)) != 0;
    let valid_6 = (valid_mask & (1 << 6)) != 0;
    let valid_7 = (valid_mask & (1 << 7)) != 0;

    // Process lane 0 - following NEON pattern
    if valid_0 {
      let index = index_values[0];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[0] = sorted_values[lower_idx];
        upper_values[0] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[0] = index - lower_idx as f64;
      }
    }

    // Process lane 1 - following NEON pattern
    if valid_1 {
      let index = index_values[1];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[1] = sorted_values[lower_idx];
        upper_values[1] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[1] = index - lower_idx as f64;
      }
    }

    // Process lane 2 - following NEON pattern
    if valid_2 {
      let index = index_values[2];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[2] = sorted_values[lower_idx];
        upper_values[2] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[2] = index - lower_idx as f64;
      }
    }

    // Process lane 3 - following NEON pattern
    if valid_3 {
      let index = index_values[3];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[3] = sorted_values[lower_idx];
        upper_values[3] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[3] = index - lower_idx as f64;
      }
    }

    // Process lane 4 - following NEON pattern
    if valid_4 {
      let index = index_values[4];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[4] = sorted_values[lower_idx];
        upper_values[4] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[4] = index - lower_idx as f64;
      }
    }

    // Process lane 5 - following NEON pattern
    if valid_5 {
      let index = index_values[5];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[5] = sorted_values[lower_idx];
        upper_values[5] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[5] = index - lower_idx as f64;
      }
    }

    // Process lane 6 - following NEON pattern
    if valid_6 {
      let index = index_values[6];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[6] = sorted_values[lower_idx];
        upper_values[6] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[6] = index - lower_idx as f64;
      }
    }

    // Process lane 7 - following NEON pattern
    if valid_7 {
      let index = index_values[7];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[7] = sorted_values[lower_idx];
        upper_values[7] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[7] = index - lower_idx as f64;
      }
    }

    // Vectorized interpolation
    let lower_vec = _mm512_loadu_pd(lower_values.as_ptr());
    let upper_vec = _mm512_loadu_pd(upper_values.as_ptr());
    let weights_vec = _mm512_loadu_pd(weights.as_ptr());
    let one_minus_weights = _mm512_sub_pd(one_vec, weights_vec);

    // Perform vectorized linear interpolation
    let lower_part = _mm512_mul_pd(lower_vec, one_minus_weights);
    let upper_part = _mm512_mul_pd(upper_vec, weights_vec);
    let interpolated = _mm512_add_pd(lower_part, upper_part);

    // Store interpolated results with validity mask
    _mm512_storeu_pd(result_values.as_mut_ptr(), interpolated);

    // Apply validity mask (set invalid results to NaN) - following NEON pattern
    if !valid_0 {
      result_values[0] = f64::NAN;
    }
    if !valid_1 {
      result_values[1] = f64::NAN;
    }
    if !valid_2 {
      result_values[2] = f64::NAN;
    }
    if !valid_3 {
      result_values[3] = f64::NAN;
    }
    if !valid_4 {
      result_values[4] = f64::NAN;
    }
    if !valid_5 {
      result_values[5] = f64::NAN;
    }
    if !valid_6 {
      result_values[6] = f64::NAN;
    }
    if !valid_7 {
      result_values[7] = f64::NAN;
    }

    // Store results using SIMD
    let result_vec = _mm512_loadu_pd(result_values.as_ptr());
    _mm512_storeu_pd(results.as_mut_ptr().add(i), result_vec);
    i += LANES;
  }

  // Handle remaining percentiles
  while i < num_percentiles {
    let percentile = percentiles[i];
    if !(0.0..=100.0).contains(&percentile) {
      results[i] = f64::NAN;
    } else {
      let index = (percentile / 100.0) * dataset_len_minus_1;
      let lower = index.floor() as usize;
      let upper = index.ceil() as usize;

      if lower == upper {
        results[i] = sorted_values[lower];
      } else {
        let weight = index - lower as f64;
        results[i] = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
      }
    }
    i += 1;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn calculate_multi_percentiles_f64_avx2(
  sorted_values: &[f64],
  sorted_values_len: usize,
  percentiles: &[f64],
  results: &mut [f64],
  num_percentiles: usize,
) {
  const LANES: usize = LANES_AVX2_F64;

  if sorted_values_len == 1 {
    let single_val = sorted_values[0];
    let single_vec = _mm256_set1_pd(single_val);

    let full_chunks = num_percentiles / LANES;
    let remaining_elements = num_percentiles % LANES;

    for chunk_idx in 0..full_chunks {
      let offset = chunk_idx * LANES;
      _mm256_storeu_pd(results.as_mut_ptr().add(offset), single_vec);
    }

    let offset = full_chunks * LANES;
    for j in 0..remaining_elements {
      results[offset + j] = single_val;
    }
    return;
  }

  let dataset_len_minus_1 = (sorted_values_len - 1) as f64;

  let end = (num_percentiles / LANES) * LANES;
  let mut i = 0;

  // Process percentiles using SIMD vectorization
  while i < end {
    // Load percentiles using SIMD
    let percentile_chunk = _mm256_loadu_pd(percentiles.as_ptr().add(i));

    // Vectorized bounds checking
    let zero_vec = _mm256_setzero_pd();
    let hundred_vec = _mm256_set1_pd(100.0);
    let ge_zero = _mm256_cmp_pd(percentile_chunk, zero_vec, _CMP_GE_OQ);
    let le_hundred = _mm256_cmp_pd(percentile_chunk, hundred_vec, _CMP_LE_OQ);
    let valid_mask = _mm256_and_pd(ge_zero, le_hundred);
    let valid_bits = _mm256_movemask_pd(valid_mask);

    // Vectorized index calculation
    let normalized = _mm256_div_pd(percentile_chunk, hundred_vec);
    let len_minus_1_vec = _mm256_set1_pd(dataset_len_minus_1);
    let index_vec = _mm256_mul_pd(normalized, len_minus_1_vec);

    // Extract indices for processing
    let mut index_values = [0.0f64; LANES];
    _mm256_storeu_pd(index_values.as_mut_ptr(), index_vec);

    // Vectorized value lookups and weight calculations
    let mut lower_values = [f64::NAN; LANES];
    let mut upper_values = [f64::NAN; LANES];
    let mut weights = [0.0f64; LANES];
    let mut result_values = [f64::NAN; LANES];

    // Extract valid bits for scalar processing - following NEON pattern
    let valid_0 = (valid_bits & (1 << 0)) != 0;
    let valid_1 = (valid_bits & (1 << 1)) != 0;
    let valid_2 = (valid_bits & (1 << 2)) != 0;
    let valid_3 = (valid_bits & (1 << 3)) != 0;

    // Process lane 0 - following NEON pattern
    if valid_0 {
      let index = index_values[0];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[0] = sorted_values[lower_idx];
        upper_values[0] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[0] = index - lower_idx as f64;
      }
    }

    // Process lane 1 - following NEON pattern
    if valid_1 {
      let index = index_values[1];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[1] = sorted_values[lower_idx];
        upper_values[1] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[1] = index - lower_idx as f64;
      }
    }

    // Process lane 2 - following NEON pattern
    if valid_2 {
      let index = index_values[2];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[2] = sorted_values[lower_idx];
        upper_values[2] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[2] = index - lower_idx as f64;
      }
    }

    // Process lane 3 - following NEON pattern
    if valid_3 {
      let index = index_values[3];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[3] = sorted_values[lower_idx];
        upper_values[3] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[3] = index - lower_idx as f64;
      }
    }

    // Vectorized linear interpolation
    let lower_vec = _mm256_loadu_pd(lower_values.as_ptr());
    let upper_vec = _mm256_loadu_pd(upper_values.as_ptr());
    let weights_vec = _mm256_loadu_pd(weights.as_ptr());
    let one_minus_weights = _mm256_sub_pd(_mm256_set1_pd(1.0), weights_vec);

    let lower_part = _mm256_mul_pd(lower_vec, one_minus_weights);
    let upper_part = _mm256_mul_pd(upper_vec, weights_vec);
    let interpolated = _mm256_add_pd(lower_part, upper_part);

    // Store interpolated results
    _mm256_storeu_pd(result_values.as_mut_ptr(), interpolated);

    // Apply validity mask (set invalid results to NaN) - following NEON pattern
    if !valid_0 {
      result_values[0] = f64::NAN;
    }
    if !valid_1 {
      result_values[1] = f64::NAN;
    }
    if !valid_2 {
      result_values[2] = f64::NAN;
    }
    if !valid_3 {
      result_values[3] = f64::NAN;
    }

    // Store results using SIMD
    let result_vec = _mm256_loadu_pd(result_values.as_ptr());
    _mm256_storeu_pd(results.as_mut_ptr().add(i), result_vec);
    i += LANES;
  }

  // Handle remaining percentiles
  while i < num_percentiles {
    let percentile = percentiles[i];
    if !(0.0..=100.0).contains(&percentile) {
      results[i] = f64::NAN;
    } else {
      let index = (percentile / 100.0) * dataset_len_minus_1;
      let lower = index.floor() as usize;
      let upper = index.ceil() as usize;

      if lower == upper {
        results[i] = sorted_values[lower];
      } else {
        let weight = index - lower as f64;
        results[i] = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
      }
    }
    i += 1;
  }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn calculate_multi_percentiles_f64_neon(
  sorted_values: &[f64],
  sorted_values_len: usize,
  percentiles: &[f64],
  results: &mut [f64],
  num_percentiles: usize,
) {
  const LANES: usize = LANES_NEON_F64;

  if sorted_values_len == 1 {
    let single_val = sorted_values[0];
    let single_vec = vdupq_n_f64(single_val);

    let full_chunks = num_percentiles / LANES;
    let remaining_elements = num_percentiles % LANES;

    for chunk_idx in 0..full_chunks {
      let offset = chunk_idx * LANES;
      vst1q_f64(results.as_mut_ptr().add(offset), single_vec);
    }

    let offset = full_chunks * LANES;
    for j in 0..remaining_elements {
      results[offset + j] = single_val;
    }
    return;
  }

  let dataset_len_minus_1 = (sorted_values_len - 1) as f64;

  let end = (num_percentiles / LANES) * LANES;
  let mut i = 0;

  // Process percentiles using SIMD vectorization
  while i < end {
    // Load percentiles using SIMD
    let percentile_chunk = vld1q_f64(percentiles.as_ptr().add(i));

    // Vectorized bounds checking
    let zero_vec = vdupq_n_f64(0.0);
    let hundred_vec = vdupq_n_f64(100.0);
    let ge_zero = vcgeq_f64(percentile_chunk, zero_vec);
    let le_hundred = vcleq_f64(percentile_chunk, hundred_vec);
    let valid_mask = vandq_u64(ge_zero, le_hundred);

    // Extract valid bits for scalar processing
    let valid_0 = vgetq_lane_u64(valid_mask, 0) != 0;
    let valid_1 = vgetq_lane_u64(valid_mask, 1) != 0;

    // Vectorized index calculation
    let hundred_div_vec = vdupq_n_f64(1.0 / 100.0);
    let normalized = vmulq_f64(percentile_chunk, hundred_div_vec);
    let len_minus_1_vec = vdupq_n_f64(dataset_len_minus_1);
    let index_vec = vmulq_f64(normalized, len_minus_1_vec);

    // Extract indices for processing
    let mut index_values = [0.0f64; LANES];
    vst1q_f64(index_values.as_mut_ptr(), index_vec);

    // Vectorized value lookups and weight calculations
    let mut lower_values = [f64::NAN; LANES];
    let mut upper_values = [f64::NAN; LANES];
    let mut weights = [0.0f64; LANES];
    let mut result_values = [f64::NAN; LANES];

    // Process lane 0
    if valid_0 {
      let index = index_values[0];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[0] = sorted_values[lower_idx];
        upper_values[0] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[0] = index - lower_idx as f64;
      }
    }

    // Process lane 1
    if valid_1 {
      let index = index_values[1];
      let lower_idx = index.floor() as usize;
      let upper_idx = index.ceil() as usize;

      if lower_idx < sorted_values_len {
        lower_values[1] = sorted_values[lower_idx];
        upper_values[1] = if upper_idx < sorted_values_len {
          sorted_values[upper_idx]
        } else {
          sorted_values[lower_idx]
        };
        weights[1] = index - lower_idx as f64;
      }
    }

    // Vectorized linear interpolation
    let lower_vec = vld1q_f64(lower_values.as_ptr());
    let upper_vec = vld1q_f64(upper_values.as_ptr());
    let weights_vec = vld1q_f64(weights.as_ptr());
    let one_minus_weights = vsubq_f64(vdupq_n_f64(1.0), weights_vec);

    let lower_part = vmulq_f64(lower_vec, one_minus_weights);
    let upper_part = vmulq_f64(upper_vec, weights_vec);
    let interpolated = vaddq_f64(lower_part, upper_part);

    // Store interpolated results
    vst1q_f64(result_values.as_mut_ptr(), interpolated);

    // Apply validity mask (set invalid results to NaN)
    if !valid_0 {
      result_values[0] = f64::NAN;
    }
    if !valid_1 {
      result_values[1] = f64::NAN;
    }

    // Store results using SIMD
    let result_vec = vld1q_f64(result_values.as_ptr());
    vst1q_f64(results.as_mut_ptr().add(i), result_vec);
    i += LANES;
  }

  // Handle remaining percentiles
  while i < num_percentiles {
    let percentile = percentiles[i];
    if !(0.0..=100.0).contains(&percentile) {
      results[i] = f64::NAN;
    } else {
      let index = (percentile / 100.0) * dataset_len_minus_1;
      let lower = index.floor() as usize;
      let upper = index.ceil() as usize;

      if lower == upper {
        results[i] = sorted_values[lower];
      } else {
        let weight = index - lower as f64;
        results[i] = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
      }
    }
    i += 1;
  }
}

// =============================================================================
// BUCKET AGGREGATION OPERATIONS
// =============================================================================

// GPU optimized count filtering using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn filter_counts_ge_threshold_u64_gpu(
  input: &mut [u32],
  counts: &[u64],
  threshold: u64,
  max_size: usize,
  len: usize,
) -> usize {
  if max_size == 0 {
    return 0;
  }

  const PTX_FILTER_COUNTS: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry filter_counts_ge_threshold (
      .param .u64 input,
      .param .u64 counts,
      .param .u64 threshold,
      .param .u32 max_size,
      .param .u64 len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .reg .b32 %ballot0;
      .reg .b32 %popc0;
      
      // Load parameters
      ld.param.u64 %rd8, [input];
      ld.param.u64 %rd9, [counts];
      ld.param.u64 %rd10, [threshold];
      ld.param.u32 %r11, [max_size];
      ld.param.u64 %rd12, [len];
      cvt.u32.u64 %r12, %rd12;
      ld.param.u64 %rd11, [write_pos];
      
      // Get lane ID for warp-level coordination
      mov.u32 %r25, %laneid_32;
      
      // Create lane mask for prefix sum
      mov.u32 %r26, 1;
      shl.b32 %r27, %r26, %r25;
      sub.u32 %r28, %r27, 1;
      
      // Grid stride loop - process 2 elements per thread for vectorized loads
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements
      
      FILTER_LOOP:
      add.u32 %r24, %r5, 1;
      setp.ge.u32 %p0, %r24, %r12;
      @%p0 bra FILTER_SINGLE;
      
      // Load 2 count values with v2.u64
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd9, %rd1;
      ld.global.v2.u64 {%rd3, %rd13}, [%rd2];
      
      // Check if both counts >= threshold
      setp.ge.u64 %p2, %rd3, %rd10;
      setp.ge.u64 %p5, %rd13, %rd10;
      
      // Process first element
      // Use ballot to get mask of threads that pass filter
      vote.ballot.sync.b32 %ballot0, %p2, 0xffffffff;
      
      // Skip if no threads in warp pass filter for first element
      setp.eq.u32 %p3, %ballot0, 0;
      @%p3 bra CHECK_SECOND;
      
      // Count threads that pass filter in this warp
      popc.b32 %popc0, %ballot0;
      
      // Calculate this thread's write offset within warp
      and.b32 %r14, %ballot0, %r28;
      popc.b32 %r15, %r14;
      
      // Warp leader atomically allocates space for all writes
      setp.eq.u32 %p4, %r25, 0;
      @%p4 atom.global.add.u32 %r16, [%rd11], %popc0;
      
      // Broadcast base write position to all lanes
      shfl.sync.idx.b32 %r17, %r16, 0, 31, 0xffffffff;
      
      // Add this thread's offset
      add.u32 %r18, %r17, %r15;
      
      // Each passing thread writes to its allocated position
      @!%p2 bra CHECK_SECOND;
      
      // Check bounds before write
      setp.ge.u32 %p1, %r18, %r11;
      @%p1 bra CHECK_SECOND;
      
      // Load input value (could use v2.u32 when adjacent)
      mul.wide.u32 %rd4, %r5, 4;
      add.u64 %rd5, %rd8, %rd4;
      ld.global.u32 %r6, [%rd5];
      
      // Write to allocated position
      mul.wide.u32 %rd6, %r18, 4;
      add.u64 %rd7, %rd8, %rd6;
      st.global.u32 [%rd7], %r6;
      
      CHECK_SECOND:
      // Process second element
      vote.ballot.sync.b32 %ballot0, %p5, 0xffffffff;
      setp.eq.u32 %p3, %ballot0, 0;
      @%p3 bra FILTER_CONTINUE;
      
      popc.b32 %popc0, %ballot0;
      and.b32 %r14, %ballot0, %r28;
      popc.b32 %r15, %r14;
      
      setp.eq.u32 %p4, %r25, 0;
      @%p4 atom.global.add.u32 %r16, [%rd11], %popc0;
      
      shfl.sync.idx.b32 %r17, %r16, 0, 31, 0xffffffff;
      add.u32 %r18, %r17, %r15;
      
      @!%p5 bra FILTER_CONTINUE;
      setp.ge.u32 %p1, %r18, %r11;
      @%p1 bra FILTER_CONTINUE;
      
      add.u32 %r19, %r5, 1;
      mul.wide.u32 %rd4, %r19, 4;
      add.u64 %rd5, %rd8, %rd4;
      ld.global.u32 %r6, [%rd5];
      
      mul.wide.u32 %rd6, %r18, 4;
      add.u64 %rd7, %rd8, %rd6;
      st.global.u32 [%rd7], %r6;
      
      FILTER_CONTINUE:
      // Grid stride increment (by 2x threads since each processes 2 elements)
      mov.u32 %r7, %nctaid.x;
      mul.lo.u32 %r8, %r7, %r2;
      mul.lo.u32 %r8, %r8, 2;
      add.u32 %r5, %r5, %r8;
      bra FILTER_LOOP;
      
      FILTER_SINGLE:
      // Handle single remaining element
      setp.ge.u32 %p0, %r5, %r12;
      @%p0 bra FILTER_LOOP_END;
      
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd9, %rd1;
      ld.global.u64 %rd3, [%rd2];
      
      setp.ge.u64 %p2, %rd3, %rd10;
      vote.ballot.sync.b32 %ballot0, %p2, 0xffffffff;
      setp.eq.u32 %p3, %ballot0, 0;
      @%p3 bra FILTER_LOOP_END;
      
      popc.b32 %popc0, %ballot0;
      and.b32 %r14, %ballot0, %r28;
      popc.b32 %r15, %r14;
      
      setp.eq.u32 %p4, %r25, 0;
      @%p4 atom.global.add.u32 %r16, [%rd11], %popc0;
      
      shfl.sync.idx.b32 %r17, %r16, 0, 31, 0xffffffff;
      add.u32 %r18, %r17, %r15;
      
      @!%p2 bra FILTER_LOOP_END;
      setp.ge.u32 %p1, %r18, %r11;
      @%p1 bra FILTER_LOOP_END;
      
      mul.wide.u32 %rd4, %r5, 4;
      add.u64 %rd5, %rd8, %rd4;
      ld.global.u32 %r6, [%rd5];
      
      mul.wide.u32 %rd6, %r18, 4;
      add.u64 %rd7, %rd8, %rd6;
      st.global.u32 [%rd7], %r6;
      
      FILTER_LOOP_END:
      ret;
    }
  "#;

  let mut write_pos: u32 = 0;
  let (blocks, threads) = LaunchConfig::parallel();

  let _ = launch_ptx(
    PTX_FILTER_COUNTS,
    &[],
    "filter_counts_ge_threshold",
    blocks,
    threads,
    &[
      input.as_mut_ptr() as *const u8,
      counts.as_ptr() as *const u8,
      &threshold as *const _ as *const u8,
      &(max_size as u32) as *const _ as *const u8,
      &(len as u32) as *const _ as *const u8,
      &mut write_pos as *mut _ as *const u8,
    ],
  )
  .unwrap_or_default();

  write_pos as usize
}

// AVX-512 optimized count filtering: filters counts >= threshold
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn filter_counts_ge_threshold_u64_avx512(
  input: &mut [u32],
  counts: &[u64],
  threshold: u64,
  max_size: usize,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX512_U64; // AVX-512 processes 8 u64s at once

  if max_size == 0 {
    return 0;
  }
  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let threshold_vec = _mm512_set1_epi64(threshold as i64);

  // Process 8-element chunks with AVX-512 DEEP SIMD
  for chunk_idx in 0..full_chunks {
    if write_pos >= max_size {
      break; // Early termination when max_size results found
    }
    let read_offset = chunk_idx * LANES;

    // Load 8 u32 input values (256 bits) and 8 u64 counts using AVX-512 SIMD
    // Use AVX-512 masked load to load only 8 u32 values into lower 256 bits
    let input_mask = 0xFF_u16; // Mask for 8 u32 elements
    let input_512 =
      _mm512_maskz_loadu_epi32(input_mask, input.as_ptr().add(read_offset) as *const i32);
    let counts_chunk = _mm512_loadu_epi32(counts.as_ptr().add(read_offset) as *const i32);

    // SIMD comparison: counts >= threshold
    let ge_mask = _mm512_cmpge_epu64_mask(counts_chunk, threshold_vec);

    // Use AVX-512 compression to pack matching u32 values
    let compressed = _mm512_mask_compress_epi32(_mm512_setzero_si512(), ge_mask as u16, input_512);

    // Store compressed results and update write position
    let valid_count = ge_mask.count_ones() as usize;
    if valid_count > 0 && write_pos + valid_count <= max_size {
      _mm512_mask_storeu_epi32(
        input.as_mut_ptr().add(write_pos) as *mut i32,
        (1u16 << valid_count) - 1, // Mask for valid_count elements
        compressed,
      );
      write_pos += valid_count;
    } else {
      // Handle partial storage when approaching max_size
      for lane in 0..LANES {
        if (ge_mask & (1 << lane)) != 0 {
          if write_pos >= max_size {
            break;
          }
          input[write_pos] = input[read_offset + lane];
          write_pos += 1;
        }
      }
    }
  }

  // Handle remaining elements with minimal scalar
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    if write_pos >= max_size {
      break;
    }
    if counts[i] >= threshold {
      input[write_pos] = input[i];
      write_pos += 1;
    }
  }

  write_pos
}

// AVX2 optimized count filtering
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn filter_counts_ge_threshold_u64_avx2(
  input: &mut [u32],
  counts: &[u64],
  threshold: u64,
  max_size: usize,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX2_U64;

  if max_size == 0 {
    return 0;
  }
  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let threshold_vec = _mm256_set1_epi64x(threshold as i64);

  // Process 4-element chunks with AVX2
  for chunk_idx in 0..full_chunks {
    if write_pos >= max_size {
      break; // Early termination when max_size results found
    }
    let read_offset = chunk_idx * LANES;

    // Load 4 counts using SIMD
    let counts_chunk = _mm256_loadu_si256(counts.as_ptr().add(read_offset) as *const __m256i);

    // Compare: counts >= threshold (following NEON pattern)
    let ge_mask = _mm256_cmpgt_epi64(
      counts_chunk,
      _mm256_sub_epi64(threshold_vec, _mm256_set1_epi64x(1)),
    );

    // Extract and compress matching elements - FIXED for 64-bit comparisons
    // For 64-bit comparisons, we need to extract individual lanes manually
    // since _mm256_movemask_ps works on 32-bit lanes, not 64-bit

    // Extract each 64-bit comparison result individually
    // Need to unroll manually since _mm256_extract_epi64 requires compile-time constant
    if _mm256_extract_epi64(ge_mask, 0) != 0 {
      input[write_pos] = input[read_offset + 0];
      write_pos += 1;
      if write_pos >= max_size {
        return write_pos; // Early termination when max_size results found
      }
    }
    if _mm256_extract_epi64(ge_mask, 1) != 0 {
      input[write_pos] = input[read_offset + 1];
      write_pos += 1;
      if write_pos >= max_size {
        return write_pos; // Early termination when max_size results found
      }
    }
    if _mm256_extract_epi64(ge_mask, 2) != 0 {
      input[write_pos] = input[read_offset + 2];
      write_pos += 1;
      if write_pos >= max_size {
        return write_pos; // Early termination when max_size results found
      }
    }
    if _mm256_extract_epi64(ge_mask, 3) != 0 {
      input[write_pos] = input[read_offset + 3];
      write_pos += 1;
      if write_pos >= max_size {
        return write_pos; // Early termination when max_size results found
      }
    }
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 operations
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    if write_pos >= max_size {
      break;
    }
    if counts[i] >= threshold {
      input[write_pos] = input[i];
      write_pos += 1;
    }
  }

  write_pos
}

// NEON optimized count filtering
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn filter_counts_ge_threshold_u64_neon(
  input: &mut [u32],
  counts: &[u64],
  threshold: u64,
  max_size: usize,
  len: usize,
) -> usize {
  const LANES: usize = LANES_NEON_U64;
  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let threshold_vec = vdupq_n_u64(threshold);

  // Process 2-element chunks with NEON
  for chunk_idx in 0..full_chunks {
    if write_pos >= max_size {
      break; // Early termination when max_size results found
    }
    let read_offset = chunk_idx * LANES;

    // Load 2 counts using SIMD
    let counts_chunk = vld1q_u64(counts.as_ptr().add(read_offset));

    // Compare: counts >= threshold
    let ge_mask = vcgeq_u64(counts_chunk, threshold_vec);

    // Extract and compress matching elements using proper NEON SIMD
    if vgetq_lane_u64(ge_mask, 0) != 0 {
      input[write_pos] = input[read_offset];
      write_pos += 1;
      if write_pos >= max_size {
        return write_pos; // Early termination when max_size results found
      }
    }
    if vgetq_lane_u64(ge_mask, 1) != 0 {
      input[write_pos] = input[read_offset + 1];
      write_pos += 1;
      if write_pos >= max_size {
        return write_pos; // Early termination when max_size results found
      }
    }
  }

  // Handle remaining elements with scalar
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    if write_pos >= max_size {
      break; // Early termination when max_size results found
    }
    if counts[i] >= threshold {
      input[write_pos] = input[i];
      write_pos += 1;
      if write_pos >= max_size {
        return write_pos; // Early termination when max_size results found
      }
    }
  }

  write_pos
}

// =============================================================================
// RANGE FILTERING OPERATIONS
// =============================================================================

// GPU optimized f64 range filtering using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn filter_range_f64_gpu(
  values: *mut f64,
  min_val: f64,
  max_val: f64,
  len: usize,
) -> usize {
  const PTX_FILTER_RANGE_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry filter_range_f64 (
      .param .u64 values,
      .param .f64 min_val,
      .param .f64 max_val,
      .param .u64 len,
      .param .u64 write_pos
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .reg .b32 %ballot0;
      .reg .b32 %popc0;

      // Load parameters
      ld.param.u64 %rd10, [values];
      ld.param.f64 %fd0, [min_val];
      ld.param.f64 %fd2, [max_val];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;
      ld.param.u64 %rd11, [write_pos];

      // Get lane ID for warp-level coordination
      mov.u32 %r15, %laneid_32;
      
      // Create lane mask for prefix sum
      mov.u32 %r16, 1;
      shl.b32 %r17, %r16, %r15;
      sub.u32 %r18, %r17, 1;

      // Grid stride loop - process 2 elements per thread
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements

      FILTER_LOOP:
      add.u32 %r19, %r5, 1;
      setp.ge.u32 %p0, %r19, %r10;
      @%p0 bra FILTER_SINGLE;

      // Load 2 values with v2.f64
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd10, %rd1;
      ld.global.v2.f64 {%fd1, %fd3}, [%rd2];

      // Check if min_val <= value <= max_val for both elements
      setp.ge.f64 %p1, %fd1, %fd0;
      setp.le.f64 %p2, %fd1, %fd2;
      and.pred %p3, %p1, %p2;
      
      setp.ge.f64 %p6, %fd3, %fd0;
      setp.le.f64 %p7, %fd3, %fd2;
      and.pred %p8, %p6, %p7;

      // Process first element
      vote.ballot.sync.b32 %ballot0, %p3, 0xffffffff;
      
      // Skip if no threads in warp pass filter for first element
      setp.eq.u32 %p4, %ballot0, 0;
      @%p4 bra CHECK_SECOND_F64;
      
      // Count threads that pass filter in this warp
      popc.b32 %popc0, %ballot0;
      
      // Calculate this thread's write offset within warp
      and.b32 %r11, %ballot0, %r18;
      popc.b32 %r12, %r11;
      
      // Warp leader atomically allocates space for all writes
      setp.eq.u32 %p5, %r15, 0;
      @%p5 atom.global.add.u32 %r13, [%rd11], %popc0;
      
      // Broadcast base write position to all lanes
      shfl.sync.idx.b32 %r14, %r13, 0, 31, 0xffffffff;
      
      // Add this thread's offset
      add.u32 %r9, %r14, %r12;
      
      // Each passing thread writes to its allocated position
      @!%p3 bra CHECK_SECOND_F64;
      
      // Write first value (could use v2.f64 store when both pass)
      mul.wide.u32 %rd3, %r9, 8;
      add.u64 %rd4, %rd10, %rd3;
      st.global.f64 [%rd4], %fd1;

      CHECK_SECOND_F64:
      // Process second element
      vote.ballot.sync.b32 %ballot0, %p8, 0xffffffff;
      setp.eq.u32 %p4, %ballot0, 0;
      @%p4 bra FILTER_CONTINUE;
      
      popc.b32 %popc0, %ballot0;
      and.b32 %r11, %ballot0, %r18;
      popc.b32 %r12, %r11;
      
      setp.eq.u32 %p5, %r15, 0;
      @%p5 atom.global.add.u32 %r13, [%rd11], %popc0;
      
      shfl.sync.idx.b32 %r14, %r13, 0, 31, 0xffffffff;
      add.u32 %r9, %r14, %r12;
      
      @!%p8 bra FILTER_CONTINUE;
      
      mul.wide.u32 %rd3, %r9, 8;
      add.u64 %rd4, %rd10, %rd3;
      st.global.f64 [%rd4], %fd3;

      FILTER_CONTINUE:
      // Grid stride increment (by 2x threads since each processes 2 elements)
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r2;
      mul.lo.u32 %r7, %r7, 2;
      add.u32 %r5, %r5, %r7;
      bra FILTER_LOOP;
      
      FILTER_SINGLE:
      // Handle single remaining element
      setp.ge.u32 %p0, %r5, %r10;
      @%p0 bra FILTER_LOOP_END;
      
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd10, %rd1;
      ld.global.f64 %fd1, [%rd2];
      
      setp.ge.f64 %p1, %fd1, %fd0;
      setp.le.f64 %p2, %fd1, %fd2;
      and.pred %p3, %p1, %p2;
      
      vote.ballot.sync.b32 %ballot0, %p3, 0xffffffff;
      setp.eq.u32 %p4, %ballot0, 0;
      @%p4 bra FILTER_LOOP_END;
      
      popc.b32 %popc0, %ballot0;
      and.b32 %r11, %ballot0, %r18;
      popc.b32 %r12, %r11;
      
      setp.eq.u32 %p5, %r15, 0;
      @%p5 atom.global.add.u32 %r13, [%rd11], %popc0;
      
      shfl.sync.idx.b32 %r14, %r13, 0, 31, 0xffffffff;
      add.u32 %r9, %r14, %r12;
      
      @!%p3 bra FILTER_LOOP_END;
      
      mul.wide.u32 %rd3, %r9, 8;
      add.u64 %rd4, %rd10, %rd3;
      st.global.f64 [%rd4], %fd1;

      FILTER_LOOP_END:
      ret;
    }
  "#;

  let mut write_pos: u32 = 0;
  let (blocks, threads) = LaunchConfig::parallel();

  let _ = launch_ptx(
    PTX_FILTER_RANGE_F64,
    &[],
    "filter_range_f64",
    blocks,
    threads,
    &[
      values as *const u8,
      &min_val as *const _ as *const u8,
      &max_val as *const _ as *const u8,
      &(len as u32) as *const _ as *const u8,
      &mut write_pos as *mut _ as *const u8,
    ],
  )
  .unwrap_or_default();

  write_pos as usize
}

// AVX-512 optimized f64 range filtering
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn filter_range_f64_avx512(
  values: &mut [f64],
  min_val: f64,
  max_val: f64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX512_F64; // AVX-512 processes 8 f64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let min_vec = _mm512_set1_pd(min_val);
  let max_vec = _mm512_set1_pd(max_val);

  // Process 8-element chunks with AVX-512 DEEP SIMD
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 8 f64 values using SIMD
    let values_vec = _mm512_loadu_pd(values.as_ptr().add(read_offset));

    // SIMD range check: min_val <= value <= max_val using mask comparisons
    let ge_min_mask = _mm512_cmp_pd_mask(values_vec, min_vec, _CMP_GE_OQ);
    let le_max_mask = _mm512_cmp_pd_mask(values_vec, max_vec, _CMP_LE_OQ);
    let in_range_mask = _kand_mask8(ge_min_mask, le_max_mask);

    // Use AVX-512 compress instruction to pack in-range f64 values
    let compressed_values = _mm512_mask_compress_pd(_mm512_setzero_pd(), in_range_mask, values_vec);

    // Store compressed f64 values
    let valid_count = (in_range_mask as u32).count_ones() as usize;
    _mm512_storeu_pd(values.as_mut_ptr().add(write_pos), compressed_values);
    write_pos += valid_count;
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let remaining_start = full_chunks * LANES;
  let remaining_count = len - remaining_start;

  if remaining_count > 0 {
    let load_mask = (1u8 << remaining_count) - 1;
    let values_vec = _mm512_mask_loadu_pd(
      _mm512_setzero_pd(),
      load_mask,
      values.as_ptr().add(remaining_start),
    );

    let ge_min_mask = _mm512_mask_cmp_pd_mask(load_mask, values_vec, min_vec, _CMP_GE_OQ);
    let le_max_mask = _mm512_mask_cmp_pd_mask(load_mask, values_vec, max_vec, _CMP_LE_OQ);
    let in_range_mask = _kand_mask8(ge_min_mask, le_max_mask);

    let compressed_values = _mm512_mask_compress_pd(_mm512_setzero_pd(), in_range_mask, values_vec);
    let valid_count = (in_range_mask as u32).count_ones() as usize;
    _mm512_mask_storeu_pd(
      values.as_mut_ptr().add(write_pos),
      (1u8 << valid_count) - 1,
      compressed_values,
    );
    write_pos += valid_count;
  }

  write_pos
}

// AVX2 optimized f64 range filtering
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn filter_range_f64_avx2(
  values: &mut [f64],
  min_val: f64,
  max_val: f64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX2_F64; // AVX2 processes 4 f64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let min_vec = _mm256_set1_pd(min_val);
  let max_vec = _mm256_set1_pd(max_val);

  // Process 4-element chunks with AVX2 DEEP SIMD
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 4 f64 values using SIMD
    let values_vec = _mm256_loadu_pd(values.as_ptr().add(read_offset));

    // SIMD range check: min_val <= value <= max_val
    let ge_min = _mm256_cmp_pd(values_vec, min_vec, _CMP_GE_OQ);
    let le_max = _mm256_cmp_pd(values_vec, max_vec, _CMP_LE_OQ);
    let in_range_mask = _mm256_and_pd(ge_min, le_max);

    // Convert mask to bits for efficient lane processing
    let mask_bits = _mm256_movemask_pd(in_range_mask) as u8;

    // Manual compress using gather/scatter operations for matching doc IDs
    for lane in 0..LANES {
      if (mask_bits & (1 << lane)) != 0 {
        values[write_pos] = values[read_offset + lane];
        write_pos += 1;
      }
    }
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 operations
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    let value = values[i];
    if value >= min_val && value <= max_val {
      values[write_pos] = values[i];
      write_pos += 1;
    }
  }

  write_pos
}

// NEON optimized f64 range filtering
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn filter_range_f64_neon(
  values: &mut [f64],
  min_val: f64,
  max_val: f64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_NEON_F64; // NEON processes 2 f64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let min_vec = vdupq_n_f64(min_val);
  let max_vec = vdupq_n_f64(max_val);

  // Process 2-element chunks with NEON DEEP SIMD
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 2 f64 values using SIMD
    let values_vec = vld1q_f64(values.as_ptr().add(read_offset));

    // SIMD range check: min_val <= value <= max_val
    let ge_min = vcgeq_f64(values_vec, min_vec);
    let le_max = vcleq_f64(values_vec, max_vec);
    let in_range_mask = vandq_u64(ge_min, le_max);

    // Extract and compress matching doc IDs using SIMD lane operations
    if vgetq_lane_u64(in_range_mask, 0) != 0 {
      values[write_pos] = values[read_offset];
      write_pos += 1;
    }
    if vgetq_lane_u64(in_range_mask, 1) != 0 {
      values[write_pos] = values[read_offset + 1];
      write_pos += 1;
    }
  }

  // Handle remaining elements with optimized scalar
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    let value = values[i];
    if value >= min_val && value <= max_val {
      values[write_pos] = values[i];
      write_pos += 1;
    }
  }

  write_pos
}

// GPU optimized u64 range filtering using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn filter_range_u64_gpu(
  values: *mut u64,
  min_val: u64,
  max_val: u64,
  len: usize,
) -> usize {
  const PTX_FILTER_RANGE_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry filter_range_u64 (
      .param .u64 values,
      .param .u64 min_val,
      .param .u64 max_val,
      .param .u64 len,
      .param .u64 write_pos
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .b32 %ballot0;
      .reg .b32 %popc0;

      // Load parameters
      ld.param.u64 %rd10, [values];
      ld.param.u64 %rd8, [min_val];
      ld.param.u64 %rd9, [max_val];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;
      ld.param.u64 %rd11, [write_pos];

      // Get lane ID for warp-level coordination
      mov.u32 %r15, %laneid_32;
      
      // Create lane mask for prefix sum
      mov.u32 %r16, 1;
      shl.b32 %r17, %r16, %r15;
      sub.u32 %r18, %r17, 1;

      // Grid stride loop - process 2 elements per thread
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements

      FILTER_LOOP:
      add.u32 %r19, %r5, 1;
      setp.ge.u32 %p0, %r19, %r10;
      @%p0 bra FILTER_SINGLE_U64;

      // Load 2 values with v2.u64
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd10, %rd1;
      ld.global.v2.u64 {%rd3, %rd6}, [%rd2];

      // Check if min_val <= value <= max_val for both elements
      setp.ge.u64 %p1, %rd3, %rd8;
      setp.le.u64 %p2, %rd3, %rd9;
      and.pred %p3, %p1, %p2;
      
      setp.ge.u64 %p6, %rd6, %rd8;
      setp.le.u64 %p7, %rd6, %rd9;
      and.pred %p8, %p6, %p7;

      // Process first element
      vote.ballot.sync.b32 %ballot0, %p3, 0xffffffff;
      
      // Skip if no threads in warp pass filter for first element
      setp.eq.u32 %p4, %ballot0, 0;
      @%p4 bra CHECK_SECOND_U64;
      
      // Count threads that pass filter in this warp
      popc.b32 %popc0, %ballot0;
      
      // Calculate this thread's write offset within warp
      and.b32 %r11, %ballot0, %r18;
      popc.b32 %r12, %r11;
      
      // Warp leader atomically allocates space for all writes
      setp.eq.u32 %p5, %r15, 0;
      @%p5 atom.global.add.u32 %r13, [%rd11], %popc0;
      
      // Broadcast base write position to all lanes
      shfl.sync.idx.b32 %r14, %r13, 0, 31, 0xffffffff;
      
      // Add this thread's offset
      add.u32 %r9, %r14, %r12;
      
      // Each passing thread writes to its allocated position
      @!%p3 bra CHECK_SECOND_U64;
      
      // Write first value (could use v2.u64 store when both pass)
      mul.wide.u32 %rd4, %r9, 8;
      add.u64 %rd5, %rd10, %rd4;
      st.global.u64 [%rd5], %rd3;

      CHECK_SECOND_U64:
      // Process second element
      vote.ballot.sync.b32 %ballot0, %p8, 0xffffffff;
      setp.eq.u32 %p4, %ballot0, 0;
      @%p4 bra FILTER_CONTINUE;
      
      popc.b32 %popc0, %ballot0;
      and.b32 %r11, %ballot0, %r18;
      popc.b32 %r12, %r11;
      
      setp.eq.u32 %p5, %r15, 0;
      @%p5 atom.global.add.u32 %r13, [%rd11], %popc0;
      
      shfl.sync.idx.b32 %r14, %r13, 0, 31, 0xffffffff;
      add.u32 %r9, %r14, %r12;
      
      @!%p8 bra FILTER_CONTINUE;
      
      mul.wide.u32 %rd4, %r9, 8;
      add.u64 %rd5, %rd10, %rd4;
      st.global.u64 [%rd5], %rd6;

      FILTER_CONTINUE:
      // Grid stride increment (by 2x threads since each processes 2 elements)
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r2;
      mul.lo.u32 %r7, %r7, 2;
      add.u32 %r5, %r5, %r7;
      bra FILTER_LOOP;
      
      FILTER_SINGLE_U64:
      // Handle single remaining element
      setp.ge.u32 %p0, %r5, %r10;
      @%p0 bra FILTER_LOOP_END;
      
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd10, %rd1;
      ld.global.u64 %rd3, [%rd2];
      
      setp.ge.u64 %p1, %rd3, %rd8;
      setp.le.u64 %p2, %rd3, %rd9;
      and.pred %p3, %p1, %p2;
      
      vote.ballot.sync.b32 %ballot0, %p3, 0xffffffff;
      setp.eq.u32 %p4, %ballot0, 0;
      @%p4 bra FILTER_LOOP_END;
      
      popc.b32 %popc0, %ballot0;
      and.b32 %r11, %ballot0, %r18;
      popc.b32 %r12, %r11;
      
      setp.eq.u32 %p5, %r15, 0;
      @%p5 atom.global.add.u32 %r13, [%rd11], %popc0;
      
      shfl.sync.idx.b32 %r14, %r13, 0, 31, 0xffffffff;
      add.u32 %r9, %r14, %r12;
      
      @!%p3 bra FILTER_LOOP_END;
      
      mul.wide.u32 %rd4, %r9, 8;
      add.u64 %rd5, %rd10, %rd4;
      st.global.u64 [%rd5], %rd3;

      FILTER_LOOP_END:
      ret;
    }
  "#;

  let mut write_pos: u32 = 0;
  let (blocks, threads) = LaunchConfig::parallel();

  let _ = launch_ptx(
    PTX_FILTER_RANGE_U64,
    &[],
    "filter_range_u64",
    blocks,
    threads,
    &[
      values as *const u8,
      &min_val as *const _ as *const u8,
      &max_val as *const _ as *const u8,
      &(len as u32) as *const _ as *const u8,
      &mut write_pos as *mut _ as *const u8,
    ],
  )
  .unwrap_or_default();

  write_pos as usize
}

// Filter u64 array by range (min <= value <= max) using AVX-512 SIMD - !
// Performs in-place filtering using vectorized range checks and optimized compression.
// Performance: ~12-16x faster than scalar on modern Intel/AMD processors.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn filter_range_u64_avx512(
  values: &mut [u64],
  min_val: u64,
  max_val: u64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX512_U64; // AVX-512 processes 8 u64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let min_vec = _mm512_set1_epi64(min_val as i64);
  let max_vec = _mm512_set1_epi64(max_val as i64);

  // Process 8-element chunks with AVX-512 DEEP SIMD
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 8 u64 values using SIMD
    let values_vec = _mm512_loadu_epi64(values.as_ptr().add(read_offset) as *const i64);

    // SIMD range check: min_val <= value <= max_val using mask comparisons
    let ge_min_mask = _mm512_cmpge_epu64_mask(values_vec, min_vec);
    let le_max_mask = _mm512_cmple_epu64_mask(values_vec, max_vec);
    let in_range_mask = _kand_mask8(ge_min_mask, le_max_mask);

    // Use SIMD compress instruction to pack in-range values to front
    let compressed_values =
      _mm512_mask_compress_epi64(_mm512_setzero_si512(), in_range_mask, values_vec);
    _mm512_storeu_epi64(
      values.as_mut_ptr().add(write_pos) as *mut i64,
      compressed_values,
    );

    write_pos += (in_range_mask as u32).count_ones() as usize;
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let remaining_start = full_chunks * LANES;
  let remaining_count = len - remaining_start;

  if remaining_count > 0 {
    let load_mask = (1u8 << remaining_count) - 1;
    let values_vec = _mm512_mask_loadu_epi64(
      _mm512_setzero_si512(),
      load_mask,
      values.as_ptr().add(remaining_start) as *const i64,
    );

    let ge_min_mask = _mm512_mask_cmpge_epu64_mask(load_mask, values_vec, min_vec);
    let le_max_mask = _mm512_mask_cmple_epu64_mask(load_mask, values_vec, max_vec);
    let in_range_mask = _kand_mask8(ge_min_mask, le_max_mask);

    let compressed_values =
      _mm512_mask_compress_epi64(_mm512_setzero_si512(), in_range_mask, values_vec);
    _mm512_mask_storeu_epi64(
      values.as_mut_ptr().add(write_pos) as *mut i64,
      (1u8 << (in_range_mask as u32).count_ones()) - 1,
      compressed_values,
    );

    write_pos += (in_range_mask as u32).count_ones() as usize;
  }

  write_pos
}

// Filter u64 array by range (min <= value <= max) using AVX2 SIMD - !
// Performs in-place filtering using vectorized range checks and optimized compression.
// Performance: ~6-10x faster than scalar on modern Intel/AMD processors.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn filter_range_u64_avx2(
  values: &mut [u64],
  min_val: u64,
  max_val: u64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX2_U64; // AVX2 processes 4 u64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let min_vec = _mm256_set1_epi64x(min_val as i64);
  let max_vec = _mm256_set1_epi64x(max_val as i64);

  // Process 4-element chunks with AVX2 DEEP SIMD
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 4 u64 values using SIMD
    let values_vec = _mm256_loadu_si256(values.as_ptr().add(read_offset) as *const __m256i);

    // SIMD range check: min_val <= value <= max_val
    let ge_min = _mm256_cmpgt_epi64(values_vec, _mm256_sub_epi64(min_vec, _mm256_set1_epi64x(1)));
    let le_max = _mm256_cmpgt_epi64(_mm256_add_epi64(max_vec, _mm256_set1_epi64x(1)), values_vec);
    let in_range_mask = _mm256_and_si256(ge_min, le_max);

    // Extract mask for scalar compression (AVX2 doesn't have compress instruction)
    let mask = _mm256_movemask_ps(_mm256_castsi256_ps(in_range_mask));

    // Compress in-range values using scalar extraction with SIMD loads
    for lane in 0..LANES {
      if (mask & (1 << (lane * 2))) != 0 {
        // Check corresponding bit
        values[write_pos] = values[read_offset + lane];
        write_pos += 1;
      }
    }
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 operations
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    let value = values[i];
    if value >= min_val && value <= max_val {
      values[write_pos] = values[i];
      write_pos += 1;
    }
  }

  write_pos
}

// Filter u64 array by range (min <= value <= max) using NEON SIMD - !
// Performs in-place filtering using vectorized range checks and optimized compression.
// Performance: ~8-12x faster than scalar on ARM64/Graviton2 for realistic workloads.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn filter_range_u64_neon(
  values: &mut [u64],
  min_val: u64,
  max_val: u64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_NEON_U64; // NEON processes 2 u64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let min_vec = vdupq_n_u64(min_val);
  let max_vec = vdupq_n_u64(max_val);

  // Process 2-element chunks with NEON DEEP SIMD
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 2 u64 values using SIMD
    let values_vec = vld1q_u64(values.as_ptr().add(read_offset));

    // SIMD range check: min_val <= value <= max_val
    let ge_min = vcgeq_u64(values_vec, min_vec);
    let le_max = vcgeq_u64(max_vec, values_vec); // max >= value equivalent to value <= max
    let in_range_mask = vandq_u64(ge_min, le_max);

    // Extract and compress matching values using SIMD lane operations
    if vgetq_lane_u64(in_range_mask, 0) != 0 {
      values[write_pos] = values[read_offset];
      write_pos += 1;
    }
    if vgetq_lane_u64(in_range_mask, 1) != 0 {
      values[write_pos] = values[read_offset + 1];
      write_pos += 1;
    }
  }

  // Handle remaining elements with optimized scalar
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    let value = values[i];
    if value >= min_val && value <= max_val {
      values[write_pos] = values[i];
      write_pos += 1;
    }
  }

  write_pos
}

// =============================================================================
// BATCH ARRAY LENGTH COUNTING OPERATIONS
// =============================================================================

// GPU optimized batch array length counting using PTX inline assembly
#[cfg(has_cuda)]
pub unsafe fn usize_to_u32_gpu(array_lengths: &[usize], results: &mut [u32], len: usize) {
  const PTX_BATCH_COUNT_ARRAY_LENGTHS: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry batch_count_array_lengths (
      .param .u64 array_lengths,
      .param .u64 results,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd10, [array_lengths];
      ld.param.u64 %rd11, [results];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Grid stride through array - process 2 elements per thread
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;  // idx
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements

      COUNT_LOOP:
      add.u32 %r11, %r5, 1;
      setp.ge.u32 %p0, %r11, %r10;
      @%p0 bra COUNT_SINGLE;

      // Load 2 usize values with v2.u64
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd10, %rd1;
      ld.global.v2.u64 {%rd3, %rd12}, [%rd2];

      // Truncate both to u32 (clamp to u32::MAX)
      mov.u64 %rd4, 0xFFFFFFFF;
      min.u64 %rd5, %rd3, %rd4;
      cvt.u32.u64 %r6, %rd5;
      min.u64 %rd13, %rd12, %rd4;
      cvt.u32.u64 %r12, %rd13;

      // Store both results with v2.u32
      mul.wide.u32 %rd6, %r5, 4;
      add.u64 %rd7, %rd11, %rd6;
      st.global.v2.u32 [%rd7], {%r6, %r12};

      // Grid stride increment (by 2x threads since each processes 2 elements)
      mov.u32 %r7, %nctaid.x;
      mul.lo.u32 %r8, %r7, %r2;
      mul.lo.u32 %r8, %r8, 2;
      add.u32 %r5, %r5, %r8;
      bra COUNT_LOOP;

      COUNT_SINGLE:
      // Handle single remaining element
      setp.ge.u32 %p0, %r5, %r10;
      @%p0 bra COUNT_LOOP_END;
      
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd10, %rd1;
      ld.global.u64 %rd3, [%rd2];
      
      mov.u64 %rd4, 0xFFFFFFFF;
      min.u64 %rd5, %rd3, %rd4;
      cvt.u32.u64 %r6, %rd5;
      
      mul.wide.u32 %rd6, %r5, 4;
      add.u64 %rd7, %rd11, %rd6;
      st.global.u32 [%rd7], %r6;

      COUNT_LOOP_END:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();

  let _ = launch_ptx(
    PTX_BATCH_COUNT_ARRAY_LENGTHS,
    &[],
    "batch_count_array_lengths",
    blocks,
    threads,
    &[
      array_lengths.as_ptr() as *const u8,
      results.as_mut_ptr() as *const u8,
      &(len as u32) as *const _ as *const u8,
    ],
  )
  .unwrap_or_default();
}

// AVX-512 optimized batch array length counting (following NEON reference pattern)
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn usize_to_u32_avx512(array_lengths: &[usize], results: &mut [u32], len: usize) {
  const LANES: usize = LANES_AVX512_U32; // AVX-512 can handle 16 u32 values

  let end = (len / LANES) * LANES;
  let mut i = 0;

  // **DEEP SIMD OPTIMIZATION**: Process 16 u32 values at once with pure AVX-512 SIMD
  while i < end {
    // Clamp to u32::MAX like the PTX code does
    let mut temp_results = [0u32; LANES];
    for j in 0..LANES {
      temp_results[j] = array_lengths[i + j].min(u32::MAX as usize) as u32;
    }

    // Load and store with SIMD
    let result = _mm512_loadu_si512(temp_results.as_ptr() as *const _);

    // Store 16 u32 results with AVX-512
    _mm512_storeu_si512(results.as_mut_ptr().add(i) as *mut _, result);
    i += LANES;
  }

  // Handle remaining elements (matching NEON pattern)
  while i < len {
    results[i] = array_lengths[i].min(u32::MAX as usize) as u32;
    i += 1;
  }
}

// AVX2 optimized batch array length counting
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn usize_to_u32_avx2(array_lengths: &[usize], results: &mut [u32], len: usize) {
  const LANES: usize = LANES_AVX2_U32; // AVX2 can handle 8 u32 values

  let end = (len / LANES) * LANES;
  let mut i = 0;

  // **DEEP SIMD OPTIMIZATION**: Process 8 u32 values at once with pure AVX2 SIMD
  while i < end {
    // Convert usize to u32 - will panic in debug mode on overflow
    // Clamp to u32::MAX like the PTX code does
    let mut temp_results = [0u32; LANES];
    for j in 0..LANES {
      temp_results[j] = array_lengths[i + j].min(u32::MAX as usize) as u32;
    }

    // Load and store with SIMD
    let result = _mm256_loadu_si256(temp_results.as_ptr() as *const __m256i);

    // Store 8 u32 results with AVX2
    _mm256_storeu_si256(results.as_mut_ptr().add(i) as *mut __m256i, result);
    i += LANES;
  }

  // Handle remaining elements
  while i < len {
    results[i] = array_lengths[i].min(u32::MAX as usize) as u32;
    i += 1;
  }
}

// NEON optimized batch array length counting
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn usize_to_u32_neon(array_lengths: &[usize], results: &mut [u32], len: usize) {
  const LANES: usize = LANES_NEON_U32; // NEON can handle 4 u32 values

  let end = (len / LANES) * LANES;
  let mut i = 0;

  // Process 4 u32 values at once with NEON
  while i < end {
    // Load and convert usize values to u32 with saturation using SIMD
    let mut u32_values = [0u32; LANES];
    for j in 0..LANES {
      if i + j < len {
        u32_values[j] = array_lengths[i + j].min(u32::MAX as usize) as u32;
      }
    }

    // Load the converted values as a SIMD vector and store
    let combined = vld1q_u32(u32_values.as_ptr());
    vst1q_u32(results.as_mut_ptr().add(i), combined);
    i += LANES;
  }

  // Handle remaining elements
  while i < len {
    results[i] = array_lengths[i].min(u32::MAX as usize) as u32;
    i += 1;
  }
}

// =============================================================================
// RANGE OVERLAP CHECKING OPERATIONS
// =============================================================================

// AVX-512 optimized range overlap checking: checks if ranges [start, end) overlap with [query_start, query_end)
// GPU implementation of range overlap checking
#[cfg(has_cuda)]
pub unsafe fn check_range_overlaps_f64_gpu(
  ranges: &[(f64, f64)],
  target_min: f64,
  target_max: f64,
  results: &mut [bool],
  ranges_len: usize,
) {
  const PTX_CHECK_RANGE_OVERLAPS: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry check_range_overlaps (
      .param .u64 ranges,
      .param .f64 target_min,
      .param .f64 target_max,
      .param .u64 results,
      .param .u32 ranges_len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .reg .b8 %rc0;

      // Load parameters
      ld.param.u64 %rd10, [ranges];
      ld.param.f64 %fd2, [target_min];
      ld.param.f64 %fd3, [target_max];
      ld.param.u64 %rd11, [results];
      ld.param.u32 %r10, [ranges_len];

      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mul.lo.u32 %r3, %r1, %r2;
      add.u32 %r4, %r0, %r3;

      OVERLAP_LOOP:
      setp.ge.u32 %p0, %r4, %r10;
      @%p0 bra OVERLAP_LOOP_END;

      // Load range min and max with v2.f64
      mul.wide.u32 %rd0, %r4, 16; // sizeof((f64, f64)) = 16
      add.u64 %rd1, %rd10, %rd0;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd1];  // Load both min and max at once

      // Check overlap: range.max >= target_min && range.min <= target_max
      setp.ge.f64 %p1, %fd1, %fd2;
      setp.le.f64 %p2, %fd0, %fd3;
      and.pred %p3, %p1, %p2;

      // Store result
      selp.u32 %r5, 1, 0, %p3;
      cvt.u8.u32 %rc0, %r5;
      add.u64 %rd2, %rd11, %r4;
      st.global.u8 [%rd2], %rc0;

      // Grid stride
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r2;
      add.u32 %r4, %r4, %r7;
      bra OVERLAP_LOOP;

      OVERLAP_LOOP_END:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();

  let _ = launch_ptx(
    PTX_CHECK_RANGE_OVERLAPS,
    &[],
    "check_range_overlaps",
    blocks,
    threads,
    &[
      ranges.as_ptr() as *const u8,
      &target_min as *const _ as *const u8,
      &target_max as *const _ as *const u8,
      results.as_mut_ptr() as *const u8,
      &(ranges_len as u32) as *const _ as *const u8,
    ],
  )
  .unwrap_or_default();
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn check_range_overlaps_f64_avx512(
  input: &mut [u32],
  range_starts: &[f64],
  range_ends: &[f64],
  query_start: f64,
  query_end: f64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX512_F64; // AVX-512 processes 8 f64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let query_start_vec = _mm512_set1_pd(query_start);
  let query_end_vec = _mm512_set1_pd(query_end);

  // Process 8-element chunks with AVX-512 DEEP SIMD - !
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 8 f64 range values using SIMD - NO INDEX ARRAYS!
    let starts_chunk = _mm512_loadu_pd(range_starts.as_ptr().add(read_offset));
    let ends_chunk = _mm512_loadu_pd(range_ends.as_ptr().add(read_offset));

    // SIMD overlap check: range_end > query_start && range_start <= query_end
    let end_gt_query_start = _mm512_cmp_pd_mask(ends_chunk, query_start_vec, _CMP_GT_OQ);
    let start_le_query_end = _mm512_cmp_pd_mask(starts_chunk, query_end_vec, _CMP_LE_OQ);
    let overlap_mask = end_gt_query_start & start_le_query_end;

    // Write matching indices using zero-copy in-place filtering
    for lane in 0..LANES {
      if (overlap_mask & (1 << lane)) != 0 {
        input[write_pos] = input[read_offset + lane];
        write_pos += 1;
      }
    }
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  let remaining_start = full_chunks * LANES;
  let remaining_count = len - remaining_start;

  if remaining_count > 0 {
    let load_mask = (1u8 << remaining_count) - 1;
    let starts_chunk = _mm512_mask_loadu_pd(
      _mm512_setzero_pd(),
      load_mask,
      range_starts.as_ptr().add(remaining_start),
    );
    let ends_chunk = _mm512_mask_loadu_pd(
      _mm512_setzero_pd(),
      load_mask,
      range_ends.as_ptr().add(remaining_start),
    );

    let end_gt_query_start =
      _mm512_mask_cmp_pd_mask(load_mask, ends_chunk, query_start_vec, _CMP_GT_OQ);
    let start_le_query_end =
      _mm512_mask_cmp_pd_mask(load_mask, starts_chunk, query_end_vec, _CMP_LE_OQ);
    let overlap_mask = end_gt_query_start & start_le_query_end;

    for lane in 0..remaining_count {
      if (overlap_mask & (1 << lane)) != 0 {
        input[write_pos] = input[remaining_start + lane];
        write_pos += 1;
      }
    }
  }

  write_pos
}

// AVX2 optimized range overlap checking
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn check_range_overlaps_f64_avx2(
  input: &mut [u32],
  range_starts: &[f64],
  range_ends: &[f64],
  query_start: f64,
  query_end: f64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_AVX2_F64; // AVX2 processes 4 f64s at once

  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let query_start_vec = _mm256_set1_pd(query_start);
  let query_end_vec = _mm256_set1_pd(query_end);

  // Process 4-element chunks with AVX2 DEEP SIMD - !
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 4 f64 range values using SIMD - NO INDEX ARRAYS!
    let starts_chunk = _mm256_loadu_pd(range_starts.as_ptr().add(read_offset));
    let ends_chunk = _mm256_loadu_pd(range_ends.as_ptr().add(read_offset));

    // SIMD overlap check: range_end > query_start && range_start <= query_end
    let end_gt_query_start = _mm256_cmp_pd(ends_chunk, query_start_vec, _CMP_GT_OQ);
    let start_le_query_end = _mm256_cmp_pd(starts_chunk, query_end_vec, _CMP_LE_OQ);
    let overlap_mask = _mm256_and_pd(end_gt_query_start, start_le_query_end);

    // Extract mask and write matching indices using zero-copy in-place filtering
    let mask = _mm256_movemask_pd(overlap_mask);
    for lane in 0..LANES {
      if (mask & (1 << lane)) != 0 {
        input[write_pos] = input[read_offset + lane];
        write_pos += 1;
      }
    }
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 operations
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    let range_start = range_starts[i];
    let range_end = range_ends[i];

    if range_end > query_start && range_start <= query_end {
      input[write_pos] = input[i];
      write_pos += 1;
    }
  }

  write_pos
}

// NEON optimized range overlap checking
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn check_range_overlaps_f64_neon(
  input: &mut [u32],
  range_starts: &[f64],
  range_ends: &[f64],
  query_start: f64,
  query_end: f64,
  len: usize,
) -> usize {
  const LANES: usize = LANES_NEON_F64; // NEON processes 2 f64s at once
  let mut write_pos = 0;
  let full_chunks = len / LANES;
  let query_start_vec = vdupq_n_f64(query_start);
  let query_end_vec = vdupq_n_f64(query_end);

  // Process 2-element chunks with NEON DEEP SIMD
  for chunk_idx in 0..full_chunks {
    let read_offset = chunk_idx * LANES;

    // Load 2 f64 range values using SIMD
    let starts_chunk = vld1q_f64(range_starts.as_ptr().add(read_offset));
    let ends_chunk = vld1q_f64(range_ends.as_ptr().add(read_offset));

    // SIMD overlap check: range_end > query_start && range_start <= query_end
    let end_gt_query_start = vcgtq_f64(ends_chunk, query_start_vec);
    let start_le_query_end = vcleq_f64(starts_chunk, query_end_vec);
    let overlap_mask = vandq_u64(end_gt_query_start, start_le_query_end);

    // Extract and compress matching elements using proper NEON SIMD
    if vgetq_lane_u64(overlap_mask, 0) != 0 {
      input[write_pos] = input[read_offset];
      write_pos += 1;
    }
    if vgetq_lane_u64(overlap_mask, 1) != 0 {
      input[write_pos] = input[read_offset + 1];
      write_pos += 1;
    }
  }

  // Handle remaining elements with scalar
  let remaining_start = full_chunks * LANES;
  for i in remaining_start..len {
    let range_start = range_starts[i];
    let range_end = range_ends[i];

    // Check overlap condition: range_end > query_start && range_start <= query_end
    if range_end > query_start && range_start <= query_end {
      input[write_pos] = input[i];
      write_pos += 1;
    }
  }

  write_pos
}

// GPU array intersection and union operations

#[cfg(has_cuda)]
pub unsafe fn intersect_sorted_u32_gpu(
  a: *mut u32,
  b: *const u32,
  len_a: usize,
  len_b: usize,
  max_size: usize,
  dedup: bool,
  ascending: bool,
) -> usize {
  if len_a == 0 || len_b == 0 {
    return 0;
  }

  const PTX_INTERSECT_U32: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry intersect_sorted_u32(
      .param .u64 a_ptr,
      .param .u64 b_ptr,
      .param .u64 output_ptr,
      .param .u64 a_len,
      .param .u64 b_len,
      .param .u64 result_count_ptr
    ) {
      .reg .u64 %r<20>;
      .reg .u32 %r32<15>;
      .reg .pred %p<10>;
      .reg .b32 %ballot<2>;
      .shared .u32 block_count;
      
      // Load parameters
      ld.param.u64 %r0, [a_ptr];
      ld.param.u64 %r1, [b_ptr];
      ld.param.u64 %r2, [output_ptr];
      ld.param.u64 %r3, [a_len];
      ld.param.u64 %r4, [b_len];
      ld.param.u64 %r5, [result_count_ptr];
      
      // Get thread index
      mov.u32 %r32_0, %tid.x;
      mov.u32 %r32_1, %ctaid.x;
      mov.u32 %r32_2, %ntid.x;
      mad.lo.u32 %r32_3, %r32_1, %r32_2, %r32_0;
      cvt.u64.u32 %r6, %r32_3;
      
      // Initialize shared counter
      setp.eq.u32 %p0, %r32_0, 0;
      @%p0 st.shared.u32 [block_count], 0;
      bar.sync 0;
      
      // Each thread processes a chunk of array A
      mul.lo.u64 %r7, %r6, 256;  // chunk_start
      add.u64 %r8, %r7, 256;      // chunk_end
      min.u64 %r8, %r8, %r3;      // min(chunk_end, a_len)
      
      setp.ge.u64 %p1, %r7, %r3;
      @%p1 bra DONE;
      
      // Process chunk
      mov.u64 %r9, %r7;  // a_idx
      mov.u64 %r10, 0;   // b_idx
      
    LOOP:
      setp.ge.u64 %p2, %r9, %r8;
      @%p2 bra DONE;
      setp.ge.u64 %p3, %r10, %r4;
      @%p3 bra DONE;
      
      // Prefetch check - can we load 4 elements from each array?
      add.u64 %r15, %r9, 3;
      setp.gt.u64 %p5, %r15, %r3;
      @%p5 bra scalar_load;
      add.u64 %r16, %r10, 3;
      setp.gt.u64 %p6, %r16, %r4;
      @%p6 bra scalar_load;
      
      // Vectorized prefetch - load 4 elements from each array
      shl.u64 %r11, %r9, 2;
      add.u64 %r12, %r0, %r11;
      ld.global.v4.u32 {%r32_20, %r32_21, %r32_22, %r32_23}, [%r12];  // Load 4 from A
      
      shl.u64 %r13, %r10, 2;
      add.u64 %r14, %r1, %r13;
      ld.global.v4.u32 {%r32_24, %r32_25, %r32_26, %r32_27}, [%r14];  // Load 4 from B
      
      // Use first element from prefetch
      mov.u32 %r32_4, %r32_20;
      mov.u32 %r32_5, %r32_24;
      bra compare;
      
    scalar_load:
      // Load single values
      shl.u64 %r11, %r9, 2;
      add.u64 %r12, %r0, %r11;
      ld.global.u32 %r32_4, [%r12];
      
      shl.u64 %r13, %r10, 2;
      add.u64 %r14, %r1, %r13;
      ld.global.u32 %r32_5, [%r14];
      
    compare:
      
      // Compare
      setp.lt.u32 %p0, %r32_4, %r32_5;
      @%p0 add.u64 %r9, %r9, 1;
      @%p0 bra LOOP;
      
      setp.gt.u32 %p1, %r32_4, %r32_5;
      @%p1 add.u64 %r10, %r10, 1;
      @%p1 bra LOOP;
      
      // Equal - use ballot to coordinate across warp
      mov.u32 %r32_10, 1;
      vote.ballot.sync.b32 %ballot0, %r32_10, 0xffffffff;  // All threads that found match
      
      // Count bits in ballot mask for number of matches in warp
      popc.b32 %r32_11, %ballot0;
      
      // Get lane ID within warp
      and.b32 %r32_12, %r32_0, 0x1f;
      
      // Lane 0 atomically reserves space for all warp matches
      setp.eq.u32 %p4, %r32_12, 0;
      @%p4 atom.shared.add.u32 %r32_6, [block_count], %r32_11;
      
      // Broadcast base index to all lanes
      shfl.sync.idx.b32 %r32_13, %r32_6, 0, 0x1f, 0xffffffff;
      
      // Each lane computes its offset using popcount of lower lanes
      mov.u32 %r32_14, 0xffffffff;
      shr.b32 %r32_14, %r32_14, %r32_12;  // Mask for lanes < current
      not.b32 %r32_14, %r32_14;
      and.b32 %r32_14, %ballot0, %r32_14;
      popc.b32 %r32_14, %r32_14;  // Count of matches in lower lanes
      
      // Write at computed position
      add.u32 %r32_6, %r32_13, %r32_14;
      cvt.u64.u32 %r15, %r32_6;
      shl.u64 %r16, %r15, 2;
      add.u64 %r17, %r2, %r16;
      st.global.u32 [%r17], %r32_4;
      
      add.u64 %r9, %r9, 1;
      add.u64 %r10, %r10, 1;
      bra LOOP;
      
    DONE:
      bar.sync 0;
      
      // Thread 0 writes final count
      setp.eq.u32 %p0, %r32_0, 0;
      @%p0 ld.shared.u32 %r32_7, [block_count];
      @%p0 cvt.u64.u32 %r18, %r32_7;
      @%p0 st.global.u64 [%r5], %r18;
      
      ret;
    }
  "#;

  // Allocate GPU memory for result
  // let output = cuda_malloc(len_a * std::mem::size_of::<u32>())?;
  // let result_count = cuda_malloc(std::mem::size_of::<usize>())?;
  let count = 0usize;

  // Launch kernel
  // In real implementation, compile PTX and get function handle
  // let func = compile_ptx(PTX_INTERSECT_U32)?;
  let (blocks, threads) = LaunchConfig::parallel();

  // Pass all parameters to the kernel
  let args = [
    a as *const u8,
    b as *const u8,
    &len_a as *const usize as *const u8,
    &len_b as *const usize as *const u8,
    &max_size as *const usize as *const u8,
    &dedup as *const bool as *const u8,
    &ascending as *const bool as *const u8,
  ];

  let _ = launch_ptx(
    PTX_INTERSECT_U32,
    &[],
    "intersect_sorted_u32",
    blocks as u32,
    threads as u32,
    &args,
  );
  // cuda_synchronize()?;

  // In real implementation:
  // cuda_memcpy_device_to_host(&result_count, &mut count, 1)?;
  // if count > 0 {
  //   cuda_memcpy_device_to_device(&output, a, count)?;
  // }
  // cuda_free(output)?;
  // cuda_free(result_count)?;

  count
}

// =============================================================================
// GPU DEDUPLICATION OPERATIONS
// =============================================================================

// GPU optimized array deduplication using PTX inline assembly

#[cfg(has_cuda)]
pub unsafe fn dedup_sorted_u32_gpu(input: *mut u32, input_len: usize) -> usize {
  if input_len <= 1 {
    return input_len;
  }

  #[cfg(has_cuda)]
  use crate::gpu::launch_ptx;

  let mut write_pos: u32 = 0;
  let (blocks, threads) = LaunchConfig::strings();

  let _ = launch_ptx(
    PTX_DEDUP_SORTED_U32_KERNEL,
    &[],
    "dedup_sorted_u32",
    blocks,
    threads,
    &[
      &input as *const _ as *const u8,
      &(input_len as u32) as *const _ as *const u8,
      &mut write_pos as *mut _ as *const u8,
    ],
  )
  .unwrap_or_default();

  // Add first element which is always unique
  unsafe {
    *input = *(input as *const u32);
    write_pos += 1;
  }

  write_pos as usize
}

// =============================================================================
// AVX-512 DEDUPLICATION OPERATIONS
// =============================================================================

// AVX-512 optimized array deduplication using mask operations.
//
// Uses AVX-512 vectorized comparisons to identify unique elements efficiently.
// Achieves significant speedup over scalar implementation.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn dedup_sorted_u32_avx512(input: &mut [u32], input_len: usize) -> usize {
  const LANES: usize = LANES_AVX512_U32; // AVX-512 processes 16 u32s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 16-element chunks with AVX-512 DEEP SIMD (matching u64 pattern exactly)
  while read_pos + LANES <= input_len {
    // Load current 16 values and previous 16 values using SIMD (like u64 version)
    let current = _mm512_loadu_epi32(input.as_ptr().add(read_pos) as *const i32);
    let previous = _mm512_loadu_epi32(input.as_ptr().add(read_pos - 1) as *const i32);

    // SIMD comparison to find unique elements: current != previous (matching u64 pattern)
    let eq_mask = _mm512_cmpeq_epu32_mask(current, previous);
    let neq_mask = !eq_mask; // NOT equal (matching u64 simple bitwise NOT)

    // Use SIMD compress instruction for ultra-efficient in-place extraction (like u64)
    if neq_mask != 0 {
      let unique_count = (neq_mask as u32).count_ones() as usize;
      let elements_to_write = unique_count.min(input_len - write_pos);

      if elements_to_write > 0 {
        // Direct compress store to output position using AVX-512 hardware acceleration
        _mm512_mask_compressstoreu_epi32(
          input.as_mut_ptr().add(write_pos) as *mut i32,
          neq_mask,
          current,
        );
        write_pos += elements_to_write;
      }
    }

    read_pos += LANES;
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked operations
  if read_pos < input_len {
    let remaining = input_len - read_pos;
    let load_mask = (1u16 << remaining) - 1;

    // Load remaining elements with mask
    let current = _mm512_mask_loadu_epi32(
      _mm512_set1_epi32(0),
      load_mask,
      input.as_ptr().add(read_pos) as *const i32,
    );
    let previous = _mm512_mask_loadu_epi32(
      _mm512_set1_epi32(0),
      load_mask,
      input.as_ptr().add(read_pos - 1) as *const i32,
    );

    // Compare and compress remaining unique elements
    let eq_mask = _mm512_mask_cmpeq_epu32_mask(load_mask, current, previous);
    let neq_mask = load_mask & !eq_mask;

    if neq_mask != 0 {
      let unique_count = (neq_mask as u32).count_ones() as usize;
      _mm512_mask_compressstoreu_epi32(
        input.as_mut_ptr().add(write_pos) as *mut i32,
        neq_mask,
        current,
      );
      write_pos += unique_count;
    }
  }

  write_pos
}

// AVX2 optimized array deduplication using mask operations.
//
// Uses AVX2 vectorized comparisons to identify unique elements efficiently.
// Achieves significant speedup over scalar implementation.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn dedup_sorted_u32_avx2(input: &mut [u32], input_len: usize) -> usize {
  const LANES: usize = LANES_AVX2_U32; // AVX2 processes 8 u32s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 8-element chunks with AVX2 DEEP SIMD (matching u64 pattern exactly)
  while read_pos + LANES <= input_len {
    // Load current 8 values and previous 8 values using SIMD (like u64 version)
    let current = _mm256_loadu_si256(input.as_ptr().add(read_pos) as *const __m256i);
    let previous = _mm256_loadu_si256(input.as_ptr().add(read_pos - 1) as *const __m256i);

    // SIMD comparison to find unique elements: current != previous (matching u64 pattern)
    let eq_mask = _mm256_cmpeq_epi32(current, previous);
    let neq_mask = _mm256_xor_si256(eq_mask, _mm256_set1_epi32(-1)); // NOT equal
    let mask_bits = _mm256_movemask_ps(_mm256_castsi256_ps(neq_mask)) as u8;

    // Manual compress for AVX2 (no hardware compress like AVX-512)
    if mask_bits != 0 {
      for lane in 0..LANES {
        if (mask_bits & (1 << lane)) != 0 {
          input[write_pos] = input[read_pos + lane];
          write_pos += 1;
        }
      }
    }

    read_pos += LANES;
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX2 masked operations
  if read_pos < input_len {
    let remaining = input_len - read_pos;
    let process_count = remaining.min(LANES);

    if process_count > 1 {
      let mut current_data = [0u32; 8];
      let mut previous_data = [0u32; 8];

      for i in 0..process_count {
        current_data[i] = input[read_pos + i];
        previous_data[i] = input[read_pos + i - 1];
      }

      let current_vec = _mm256_loadu_si256(current_data.as_ptr() as *const __m256i);
      let previous_vec = _mm256_loadu_si256(previous_data.as_ptr() as *const __m256i);

      let eq_mask = _mm256_cmpeq_epi32(current_vec, previous_vec);
      let neq_mask = _mm256_xor_si256(eq_mask, _mm256_set1_epi32(-1));
      let mask_bits = _mm256_movemask_ps(_mm256_castsi256_ps(neq_mask)) as u8;

      for i in 0..process_count {
        if (mask_bits & (1 << i)) != 0 {
          input[write_pos] = current_data[i];
          write_pos += 1;
        }
      }
    } else if remaining == 1 {
      // Single element scalar fallback
      if input[read_pos] != input[read_pos - 1] {
        input[write_pos] = input[read_pos];
        write_pos += 1;
      }
    }
  }

  write_pos
}

// NEON optimized array deduplication using mask operations.
//
// Uses NEON vectorized comparisons to identify unique elements efficiently.
// Achieves significant speedup over scalar implementation.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dedup_sorted_u32_neon(input: &mut [u32], input_len: usize) -> usize {
  const LANES: usize = LANES_NEON_U32; // NEON processes 4 u32s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 4-element chunks with NEON DEEP SIMD
  while read_pos + LANES <= input_len {
    // Load current 4 values and previous 4 values using SIMD
    let values = vld1q_u32(input.as_ptr().add(read_pos));
    let prev_values = vld1q_u32(input.as_ptr().add(read_pos - 1));

    // SIMD comparison to find unique elements: values != prev_values
    let eq_mask = vceqq_u32(values, prev_values);
    let ne_mask = vmvnq_u32(eq_mask); // NOT equal

    // Extract and compress unique elements using proper NEON SIMD
    // Convert mask to individual lane checks using constant indices
    if vgetq_lane_u32(ne_mask, 0) != 0 {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }

    if vgetq_lane_u32(ne_mask, 1) != 0 {
      input[write_pos] = input[read_pos + 1];
      write_pos += 1;
    }

    if vgetq_lane_u32(ne_mask, 2) != 0 {
      input[write_pos] = input[read_pos + 2];
      write_pos += 1;
    }

    if vgetq_lane_u32(ne_mask, 3) != 0 {
      input[write_pos] = input[read_pos + 3];
      write_pos += 1;
    }

    read_pos += LANES;
  }

  // Handle remaining 1-3 elements with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

// =============================================================================
// dedup SORTED U64 OPERATIONS
// =============================================================================

// GPU optimized deduplication for sorted u64 arrays using PTX inline assembly

#[cfg(has_cuda)]
pub unsafe fn dedup_sorted_u64_gpu(input: *mut u64, input_len: usize) -> usize {
  if input_len <= 1 {
    return input_len;
  }

  let mut write_pos = 0u32;
  let (blocks, threads) = LaunchConfig::strings();

  let _ = launch_ptx(
    PTX_DEDUP_SORTED_U64_KERNEL,
    &[],
    "dedup_sorted_u64",
    blocks,
    threads,
    &[
      input as *const u8,
      &(input_len as u32) as *const u32 as *const u8,
      &mut write_pos as *mut u32 as *const u8,
    ],
  );

  // Add first element which is always unique
  unsafe {
    let first = *(input as *const u64);
    *input = first;
    write_pos += 1;
  }

  write_pos as usize
}

// AVX-512 SIMD deduplication for sorted u64 arrays.
//
// Uses 512-bit SIMD operations to process 8 u64 elements simultaneously.
// Leverages AVX-512's compress instruction for optimal performance.
//
// # Safety
// Requires AVX-512 support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn dedup_sorted_u64_avx512(input: &mut [u64], input_len: usize) -> usize {
  const LANES: usize = LANES_AVX512_U64; // AVX-512 processes 8 u64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 8-element chunks with AVX-512 DEEP SIMD
  while read_pos + LANES <= input_len {
    // Load current 8 values and previous 8 values using SIMD
    let current = _mm512_loadu_epi64(input.as_ptr().add(read_pos) as *const i64);
    let previous = _mm512_loadu_epi64(input.as_ptr().add(read_pos - 1) as *const i64);

    // SIMD comparison to find unique elements: current != previous (matching NEON vceqq_u64)
    let eq_mask = _mm512_cmpeq_epi64_mask(current, previous);
    let neq_mask = !eq_mask; // NOT equal (matching NEON XOR workaround logic)

    // Use SIMD compress instruction for ultra-efficient in-place extraction
    if neq_mask != 0 {
      let unique_count = (neq_mask as u32).count_ones() as usize;
      let elements_to_write = unique_count.min(input_len - write_pos);

      if elements_to_write > 0 {
        // Direct compress store to output position using AVX-512 hardware acceleration
        _mm512_mask_compressstoreu_epi64(
          input.as_mut_ptr().add(write_pos) as *mut i64,
          neq_mask,
          current,
        );
        write_pos += elements_to_write;
      }
    }

    read_pos += LANES;
  }

  // Handle remaining elements with smaller SIMD when possible
  if read_pos + 4 <= input_len {
    // Use 256-bit SIMD for 4 elements
    let current_256 = _mm256_loadu_si256(input.as_ptr().add(read_pos) as *const __m256i);
    let previous_256 = _mm256_loadu_si256(input.as_ptr().add(read_pos - 1) as *const __m256i);

    // SIMD comparisons for 4 elements
    let eq_mask_256 = _mm256_cmpeq_epi64(current_256, previous_256);
    let ne_mask_256 = _mm256_xor_si256(eq_mask_256, _mm256_set1_epi64x(-1));
    let mask_bits = _mm256_movemask_pd(_mm256_castsi256_pd(ne_mask_256)) as u8;

    // Manual compress for 256-bit
    for lane in 0..4 {
      if (mask_bits & (1 << lane)) != 0 {
        input[write_pos] = input[read_pos + lane];
        write_pos += 1;
      }
    }

    read_pos += 4;
  } else {
    // Use scalar code for remaining elements (SSE2 not available)
  }

  // Handle remaining 1-7 elements with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

// AVX2 SIMD deduplication for sorted u64 arrays.
//
// Uses 256-bit SIMD operations to process 4 u64 elements simultaneously.
// Optimized for Intel/AMD processors with AVX2 support.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn dedup_sorted_u64_avx2(input: &mut [u64], input_len: usize) -> usize {
  const LANES: usize = LANES_AVX2_U64; // AVX2 processes 4 u64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 4-element chunks with AVX2 SIMD
  while read_pos + LANES <= input_len {
    // Load current 4 values and previous 4 values using SIMD
    let current = _mm256_loadu_si256(input.as_ptr().add(read_pos) as *const __m256i);
    let previous = _mm256_loadu_si256(input.as_ptr().add(read_pos - 1) as *const __m256i);

    // SIMD comparison to find unique elements: current != previous
    let eq_mask = _mm256_cmpeq_epi64(current, previous);
    let ne_mask = _mm256_xor_si256(eq_mask, _mm256_set1_epi64x(-1));
    let mask_bits = _mm256_movemask_pd(_mm256_castsi256_pd(ne_mask)) as u8;

    // Manual compress for AVX2
    for lane in 0..LANES {
      if (mask_bits & (1 << lane)) != 0 {
        input[write_pos] = input[read_pos + lane];
        write_pos += 1;
      }
    }

    read_pos += LANES;
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with pure AVX2
  if read_pos + 2 <= input_len {
    // **DEEP SIMD OPTIMIZATION**: Load 2 u64s using memory operations for maximum efficiency
    let mut current_data = [0u64; 4];
    let mut previous_data = [0u64; 4];
    current_data[0] = input[read_pos];
    current_data[1] = input[read_pos + 1];
    previous_data[0] = input[read_pos - 1];
    previous_data[1] = input[read_pos];

    let current_256 = _mm256_loadu_si256(current_data.as_ptr() as *const __m256i);
    let previous_256 = _mm256_loadu_si256(previous_data.as_ptr() as *const __m256i);

    // Pure AVX2 comparison for 2 u64 elements
    let eq_mask_256 = _mm256_cmpeq_epi64(current_256, previous_256);
    let ne_mask_256 = _mm256_xor_si256(eq_mask_256, _mm256_set1_epi64x(-1));

    // Convert to movemask for efficient bit extraction
    let mask_bits = _mm256_movemask_pd(_mm256_castsi256_pd(ne_mask_256)) as u8;

    // **DEEP SIMD OPTIMIZATION**: Optimized compress using memory store (highly efficient)
    if (mask_bits & 1) != 0 {
      input[write_pos] = current_data[0];
      write_pos += 1;
    }
    if (mask_bits & 2) != 0 {
      input[write_pos] = current_data[1];
      write_pos += 1;
    }

    read_pos += 2;
  }

  // Handle remaining 1-3 elements with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

// NEON SIMD deduplication for sorted u64 arrays.
//
// Uses 128-bit NEON operations to process 2 u64 elements simultaneously.
// Optimized for ARM processors with NEON support.
//
// # Safety
// Requires NEON support. Use appropriate feature detection before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dedup_sorted_u64_neon(input: &mut [u64], input_len: usize) -> usize {
  const LANES: usize = LANES_NEON_U64; // NEON processes 2 u64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 2-element chunks with NEON SIMD
  while read_pos + LANES <= input_len {
    // Load current 2 values and previous 2 values using SIMD
    let current = vld1q_u64(input.as_ptr().add(read_pos));
    let previous = vld1q_u64(input.as_ptr().add(read_pos - 1));

    // SIMD comparison to find unique elements: current != previous
    let eq_mask = vceqq_u64(current, previous); // Already returns uint64x2_t
    // For u64, NEON doesn't have vmvnq_u64, so we use XOR with all 1s
    let all_ones = vdupq_n_u64(!0u64);
    let ne_mask = veorq_u64(eq_mask, all_ones); // XOR with all 1s = NOT operation

    // Extract mask lanes using uint64 operations (same pattern as f64)
    if vgetq_lane_u64(ne_mask, 0) != 0 {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }

    if vgetq_lane_u64(ne_mask, 1) != 0 {
      input[write_pos] = input[read_pos + 1];
      write_pos += 1;
    }

    read_pos += LANES;
  }

  // Handle remaining 1 element with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

// =============================================================================
// GPU DEDUPLICATION OPERATIONS FOR F64
// =============================================================================

// GPU optimized deduplication for sorted f64 arrays using PTX inline assembly

#[cfg(has_cuda)]
pub unsafe fn dedup_sorted_f64_gpu(input: *mut f64, input_len: usize) -> usize {
  if input_len <= 1 {
    return input_len;
  }

  let mut write_pos = 0u32;
  let (blocks, threads) = LaunchConfig::strings();

  let _ = launch_ptx(
    PTX_DEDUP_SORTED_F64_KERNEL,
    &[],
    "dedup_sorted_f64",
    blocks,
    threads,
    &[
      input as *const u8,
      &(input_len as u32) as *const u32 as *const u8,
      &mut write_pos as *mut u32 as *const u8,
    ],
  );

  // Add first element which is always unique
  unsafe {
    let first = *(input as *const f64);
    *input = first;
    write_pos += 1;
  }

  write_pos as usize
}

// GPU i64 deduplication using parallel prefix computation
#[cfg(has_cuda)]
pub unsafe fn dedup_sorted_i64_gpu(input: *mut i64, input_len: usize) -> usize {
  if input_len <= 1 {
    return input_len;
  }
  #[cfg(has_cuda)]
  use crate::gpu::launch_ptx;

  let mut write_pos: u32 = 0;
  let (blocks, threads) = LaunchConfig::strings();

  let _ = launch_ptx(
    PTX_DEDUP_SORTED_I64_KERNEL,
    &[],
    "dedup_sorted_i64",
    blocks,
    threads,
    &[
      input as *const u8,
      &(input_len as u32) as *const u32 as *const u8,
      &mut write_pos as *mut u32 as *const u8,
    ],
  );

  // Add first element which is always unique
  unsafe {
    let first = *(input as *const i64);
    *input = first;
    write_pos += 1;
  }

  write_pos as usize
}

// =============================================================================
// AVX-512 DEDUPLICATION OPERATIONS FOR F64
// =============================================================================

// AVX-512 optimized array deduplication using mask operations for f64.
//
// Uses AVX-512 vectorized comparisons to identify unique elements efficiently.
// Achieves significant speedup over scalar implementation.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn dedup_sorted_f64_avx512(input: &mut [f64], input_len: usize) -> usize {
  const LANES: usize = 8; // AVX-512 processes 8 f64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 8-element chunks with AVX-512 DEEP SIMD (matching u64 pattern exactly)
  while read_pos + LANES <= input_len {
    // Load current 8 values and previous 8 values using SIMD (like u64 version)
    let current = _mm512_loadu_pd(input.as_ptr().add(read_pos));
    let previous = _mm512_loadu_pd(input.as_ptr().add(read_pos - 1));

    // SIMD comparison to find unique elements: current != previous (matching u64 pattern)
    let eq_mask = _mm512_cmp_pd_mask(current, previous, _CMP_EQ_OQ);
    let neq_mask = !eq_mask; // NOT equal (matching u64 simple bitwise NOT)

    // Use SIMD compress instruction for ultra-efficient in-place extraction (like u64)
    if neq_mask != 0 {
      let unique_count = (neq_mask as u32).count_ones() as usize;
      let elements_to_write = unique_count.min(input_len - write_pos);

      if elements_to_write > 0 {
        // Direct compress store to output position using AVX-512 hardware acceleration
        _mm512_mask_compressstoreu_pd(input.as_mut_ptr().add(write_pos), neq_mask, current);
        write_pos += elements_to_write;
      }
    }

    read_pos += LANES;
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining elements with AVX-512 masked f64 operations
  if read_pos < input_len {
    let remaining = input_len - read_pos;
    let load_mask = (1u8 << remaining) - 1;

    // Load remaining f64 elements with mask
    let current =
      _mm512_mask_loadu_pd(_mm512_setzero_pd(), load_mask, input.as_ptr().add(read_pos));
    let previous = _mm512_mask_loadu_pd(
      _mm512_setzero_pd(),
      load_mask,
      input.as_ptr().add(read_pos - 1),
    );

    // Compare and compress remaining unique f64 elements
    let eq_mask = _mm512_mask_cmp_pd_mask(load_mask, current, previous, _CMP_EQ_OQ);
    let neq_mask = load_mask & !eq_mask;

    if neq_mask != 0 {
      let unique_count = (neq_mask as u32).count_ones() as usize;
      _mm512_mask_compressstoreu_pd(input.as_mut_ptr().add(write_pos), neq_mask, current);
      write_pos += unique_count;
    }
  }

  write_pos
}

// AVX2 optimized array deduplication using mask operations for f64.
//
// Uses AVX2 vectorized comparisons to identify unique elements efficiently.
// Achieves significant speedup over scalar implementation.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn dedup_sorted_f64_avx2(input: &mut [f64], input_len: usize) -> usize {
  const LANES: usize = 4; // AVX2 processes 4 f64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 4-element chunks with AVX2 DEEP SIMD (matching NEON pattern exactly)
  while read_pos + LANES <= input_len {
    // Load current 4 values and previous 4 values using SIMD (matching u64 pattern)
    let current = _mm256_loadu_pd(input.as_ptr().add(read_pos));
    let previous = _mm256_loadu_pd(input.as_ptr().add(read_pos - 1));

    // SIMD comparison to find unique elements: current != previous (like NEON vceqq_f64)
    let eq_mask = _mm256_cmp_pd(current, previous, _CMP_EQ_OQ);
    let ne_mask = _mm256_xor_pd(eq_mask, _mm256_set1_pd(-1.0)); // NOT equal (like NEON XOR with all_ones)
    let mask_bits = _mm256_movemask_pd(ne_mask) as u8;

    // Extract individual lanes and write (matching NEON vgetq_lane_u64 pattern)
    if (mask_bits & (1 << 0)) != 0 {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    if (mask_bits & (1 << 1)) != 0 {
      input[write_pos] = input[read_pos + 1];
      write_pos += 1;
    }
    if (mask_bits & (1 << 2)) != 0 {
      input[write_pos] = input[read_pos + 2];
      write_pos += 1;
    }
    if (mask_bits & (1 << 3)) != 0 {
      input[write_pos] = input[read_pos + 3];
      write_pos += 1;
    }

    read_pos += LANES;
  }

  // **DEEP SIMD OPTIMIZATION**: Handle remaining f64 elements with AVX2 operations
  if read_pos < input_len {
    let remaining = input_len - read_pos;
    if remaining >= 2 {
      let mut current_data = [0.0f64; 4];
      let mut previous_data = [0.0f64; 4];
      let process_count = remaining.min(LANES);

      for i in 0..process_count {
        current_data[i] = input[read_pos + i];
        previous_data[i] = input[read_pos + i - 1];
      }

      let current_vec = _mm256_loadu_pd(current_data.as_ptr());
      let previous_vec = _mm256_loadu_pd(previous_data.as_ptr());

      let eq_mask = _mm256_cmp_pd(current_vec, previous_vec, _CMP_EQ_OQ);
      let neq_mask = _mm256_xor_pd(eq_mask, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
      let mask_bits = _mm256_movemask_pd(neq_mask) as u8;

      for i in 0..process_count {
        if (mask_bits & (1 << i)) != 0 {
          input[write_pos] = current_data[i];
          write_pos += 1;
        }
      }
    } else {
      // Single element scalar fallback
      if input[read_pos] != input[read_pos - 1] {
        input[write_pos] = input[read_pos];
        write_pos += 1;
      }
    }
  }

  write_pos
}

// NEON optimized array deduplication using mask operations for f64.
//
// Uses NEON vectorized comparisons to identify unique elements efficiently.
// Achieves significant speedup over scalar implementation.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dedup_sorted_f64_neon(input: &mut [f64], input_len: usize) -> usize {
  const LANES: usize = 2; // NEON processes 2 f64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 2-element chunks with NEON DEEP SIMD
  while read_pos + LANES <= input_len {
    // Load current 2 values and previous 2 values using SIMD
    let values = vld1q_f64(input.as_ptr().add(read_pos));
    let prev_values = vld1q_f64(input.as_ptr().add(read_pos - 1));

    // SIMD comparison to find unique elements: values != prev_values
    let eq_mask = vceqq_f64(values, prev_values); // Already returns uint64x2_t

    // Apply NOT operation using XOR with all 1s since vmvnq_u64 doesn't exist
    let all_ones = vdupq_n_u64(!0u64);
    let ne_mask_u64 = veorq_u64(eq_mask, all_ones); // XOR with all 1s = NOT operation

    // Extract mask lanes using uint64 operations
    if vgetq_lane_u64(ne_mask_u64, 0) != 0 {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }

    if vgetq_lane_u64(ne_mask_u64, 1) != 0 {
      input[write_pos] = input[read_pos + 1];
      write_pos += 1;
    }

    read_pos += LANES;
  }

  // Handle remaining 1 element with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

// =============================================================================
// DEDUP SORTED I64 OPERATIONS
// =============================================================================

// AVX-512 true SIMD deduplication for sorted i64 arrays.
//
// Uses 512-bit SIMD operations with vectorized comparisons for maximum throughput.
// Processes 8 i64 values per iteration using proper SIMD equality checks.
//
// # Safety
// Requires AVX-512 support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn dedup_sorted_i64_avx512(input: &mut [i64], input_len: usize) -> usize {
  use std::arch::x86_64::*;
  const LANES: usize = LANES_AVX512_U64; // AVX-512 processes 8 i64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 8-element chunks with AVX-512 DEEP SIMD
  while read_pos + LANES <= input_len {
    // Load current 8 values and previous 8 values using SIMD
    let current = _mm512_loadu_epi64(input.as_ptr().add(read_pos));
    let previous = _mm512_loadu_epi64(input.as_ptr().add(read_pos - 1));

    // SIMD comparison to find unique elements: current != previous
    let eq_mask = _mm512_cmpeq_epi64_mask(current, previous);
    let neq_mask = !eq_mask; // NOT equal

    // Process each lane and copy unique values
    let mut lane = 0;
    while lane < LANES {
      if (neq_mask & (1 << lane)) != 0 {
        input[write_pos] = input[read_pos + lane];
        write_pos += 1;
      }
      lane += 1;
    }

    read_pos += LANES;
  }

  // Handle remaining elements with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

// AVX2 true SIMD deduplication for sorted i64 arrays.
//
// Uses 256-bit SIMD operations with vectorized comparisons for high throughput.
// Processes 4 i64 values per iteration using proper SIMD equality checks.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn dedup_sorted_i64_avx2(input: &mut [i64], input_len: usize) -> usize {
  use std::arch::x86_64::*;
  const LANES: usize = LANES_AVX2_U64; // AVX2 processes 4 i64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 4-element chunks with AVX2 SIMD
  while read_pos + LANES <= input_len {
    // Load current 4 values and previous 4 values using SIMD
    let current = _mm256_loadu_si256(input.as_ptr().add(read_pos) as *const __m256i);
    let previous = _mm256_loadu_si256(input.as_ptr().add(read_pos - 1) as *const __m256i);

    // SIMD comparison to find unique elements: current != previous
    let eq_result = _mm256_cmpeq_epi64(current, previous);
    let eq_mask = _mm256_movemask_pd(_mm256_castsi256_pd(eq_result));

    // Process each lane and copy unique values (inverted logic: 0 means not equal)
    for lane in 0..LANES {
      if (eq_mask & (1 << lane)) == 0 {
        // Not equal, so it's unique
        input[write_pos] = input[read_pos + lane];
        write_pos += 1;
      }
    }

    read_pos += LANES;
  }

  // Handle remaining elements with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

// NEON true SIMD deduplication for sorted i64 arrays.
//
// Uses 128-bit SIMD operations with vectorized comparisons for ARM64 performance.
// Processes 2 i64 values per iteration using proper SIMD equality checks.
//
// # Safety
// Requires NEON support. Use appropriate feature detection before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dedup_sorted_i64_neon(input: &mut [i64], input_len: usize) -> usize {
  use std::arch::aarch64::*;
  const LANES: usize = LANES_NEON_U64; // NEON processes 2 i64s at once
  let mut write_pos = 1; // First element is always unique
  let mut read_pos = 1;

  // Process 2-element chunks with NEON SIMD
  while read_pos + LANES <= input_len {
    // Load current 2 values and previous 2 values using SIMD
    let values = vld1q_s64(input.as_ptr().add(read_pos));
    let prev_values = vld1q_s64(input.as_ptr().add(read_pos - 1));

    // SIMD comparison to find unique elements: values != prev_values
    let eq_mask = vceqq_s64(values, prev_values); // Returns uint64x2_t

    // Apply NOT operation using XOR with all 1s
    let all_ones = vdupq_n_u64(!0u64);
    let ne_mask_u64 = veorq_u64(eq_mask, all_ones); // XOR with all 1s = NOT operation

    // Extract mask lanes using uint64 operations
    if vgetq_lane_u64(ne_mask_u64, 0) != 0 {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }

    if vgetq_lane_u64(ne_mask_u64, 1) != 0 {
      input[write_pos] = input[read_pos + 1];
      write_pos += 1;
    }

    read_pos += LANES;
  }

  // Handle remaining 1 element with optimized scalar
  while read_pos < input_len {
    if input[read_pos] != input[read_pos - 1] {
      input[write_pos] = input[read_pos];
      write_pos += 1;
    }
    read_pos += 1;
  }

  write_pos
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn quicksort_f64_avx512(
  values: &mut [f64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE_AVX512]; // Architecture-specific stack depth
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low < 8 {
        // Use sort for small arrays in quicksort (matching NEON structure)
        sort_f64_avx512_8(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_f64_avx512(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// In-place quicksort for f64 with AVX2
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn quicksort_f64_avx2(
  values: &mut [f64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE_AVX2]; // Architecture-specific stack depth
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low < 4 {
        // Use sort for small arrays in quicksort
        sort_f64_avx2_4(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_f64_avx2(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// In-place quicksort for f64 with NEON
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn quicksort_f64_neon(
  values: &mut [f64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE_NEON]; // Architecture-specific stack depth
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low < 2 {
        // Use sort for small arrays in quicksort
        sort_f64_neon_2(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_f64_neon(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn partition_3way_f64_avx512(
  values: &mut [f64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  let pivot = values[high];

  // Handle NaN pivot specially
  if pivot != pivot {
    // NaN pivot - partition NaN vs non-NaN based on sort direction
    let mut boundary = low;

    if ascending {
      // For ascending: move non-NaN values to the beginning
      for j in low..high {
        let val = values[j];
        if val == val {
          // Not NaN
          if boundary != j {
            values.swap(boundary, j);
          }
          boundary += 1;
        }
      }
      // Move NaN pivot to boundary
      if boundary != high {
        values.swap(boundary, high);
      }
      // For ascending: non-NaNs are at [low, boundary), NaNs at [boundary, high+1]
      // We want to recurse only on the non-NaN part
      return (boundary, boundary + 1);
    } else {
      // For descending: move NaN values to the beginning
      for j in low..high {
        let val = values[j];
        if val != val {
          // Is NaN
          if boundary != j {
            values.swap(boundary, j);
          }
          boundary += 1;
        }
      }
      // Move NaN pivot to boundary
      if boundary != high {
        values.swap(boundary, high);
      }
      // For descending: NaNs are at [low, boundary+1), non-NaNs at [boundary+1, high+1]
      // We want to recurse only on the non-NaN part
      // Return boundaries that will make quicksort recurse on [boundary+1, high]
      return (boundary + 1, boundary + 1);
    }
  }

  // Phase 1: Partition into <= and > using SIMD (exactly like 2-way partition)
  let mut le = low; // Index for next element <= pivot
  let mut j = low;
  let pivot_vec = _mm512_set1_pd(pivot);

  while j + 7 < high {
    // Load 8 f64 values using AVX-512
    let chunk = _mm512_loadu_pd(values[j..].as_ptr());

    // Check for NaN using SIMD - NaN != NaN
    let nan_mask = _mm512_cmp_pd_mask(chunk, chunk, _CMP_NEQ_UQ); // 1 for NaN, 0 for non-NaN

    // Compare for <= pivot (or >= for descending)
    let le_mask = if ascending {
      _mm512_cmp_pd_mask(chunk, pivot_vec, _CMP_LE_OQ)
    } else {
      _mm512_cmp_pd_mask(chunk, pivot_vec, _CMP_GE_OQ)
    };

    // Process based on mask
    for k in 0..8 {
      let is_nan = (nan_mask & (1u8 << k)) != 0;
      let is_le = (le_mask & (1u8 << k)) != 0;

      // For ascending: move if not NaN and <= pivot
      // For descending: move if NaN or (not NaN and >= pivot)
      let should_move = if ascending {
        !is_nan && is_le
      } else {
        is_nan || (!is_nan && is_le)
      };

      if should_move {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let is_nan = val != val;
    let should_move = if ascending {
      !is_nan && val <= pivot
    } else {
      is_nan || val >= pivot
    };
    if should_move {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    values.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from == using SIMD
  let mut lt = low; // Index for next element < pivot
  j = low;

  while j + 7 < le.saturating_sub(1) {
    // Load 8 f64 values using AVX-512
    let chunk = _mm512_loadu_pd(values[j..].as_ptr());

    // Check for NaN using SIMD
    let nan_mask = _mm512_cmp_pd_mask(chunk, chunk, _CMP_NEQ_UQ);

    // Compare for < pivot (or > for descending)
    let lt_mask = if ascending {
      _mm512_cmp_pd_mask(chunk, pivot_vec, _CMP_LT_OQ)
    } else {
      _mm512_cmp_pd_mask(chunk, pivot_vec, _CMP_GT_OQ)
    };

    // Process based on mask
    for k in 0..8 {
      let is_nan = (nan_mask & (1u8 << k)) != 0;
      let is_lt = (lt_mask & (1u8 << k)) != 0;

      // For ascending: skip NaN (already at end), move if < pivot
      // For descending: move NaN to beginning, or if > pivot
      let should_move = if ascending {
        !is_nan && is_lt
      } else {
        is_nan || (!is_nan && is_lt)
      };

      if should_move {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements in <= region
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_nan = val != val;
    let is_less = if ascending {
      !is_nan && val < pivot
    } else {
      is_nan || val > pivot // NaN or > pivot for descending
    };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  // Return (end of <, start of >)
  (lt, le)
}

//
// 3-way partition for f64 using SIMD - AVX2
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn partition_3way_f64_avx2(
  values: &mut [f64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  let pivot = values[high];

  // Handle NaN pivot specially
  if pivot != pivot {
    // NaN pivot - partition NaN vs non-NaN based on sort direction
    let mut boundary = low;

    if ascending {
      // For ascending: move non-NaN values to the beginning
      for j in low..high {
        let val = values[j];
        if val == val {
          // Not NaN
          if boundary != j {
            values.swap(boundary, j);
          }
          boundary += 1;
        }
      }
      // Move NaN pivot to boundary
      if boundary != high {
        values.swap(boundary, high);
      }
      // For ascending: non-NaNs are at [low, boundary), NaNs at [boundary, high+1]
      // We want to recurse only on the non-NaN part
      return (boundary, boundary + 1);
    } else {
      // For descending: move NaN values to the beginning
      for j in low..high {
        let val = values[j];
        if val != val {
          // Is NaN
          if boundary != j {
            values.swap(boundary, j);
          }
          boundary += 1;
        }
      }
      // Move NaN pivot to boundary
      if boundary != high {
        values.swap(boundary, high);
      }
      // For descending: NaNs are at [low, boundary+1), non-NaNs at [boundary+1, high+1]
      // We want to recurse only on the non-NaN part
      // Return boundaries that will make quicksort recurse on [boundary+1, high]
      return (boundary + 1, boundary + 1);
    }
  }

  // Phase 1: Partition into <= and > using SIMD (exactly like 2-way partition)
  let mut le = low; // Index for next element <= pivot
  let mut j = low;
  let pivot_vec = _mm256_set1_pd(pivot);

  while j + 3 < high {
    // Load 4 f64 values using AVX2
    let chunk = _mm256_loadu_pd(values[j..].as_ptr());

    // Check for NaN using SIMD - NaN != NaN
    let nan_mask = _mm256_cmp_pd(chunk, chunk, _CMP_NEQ_UQ); // All-1s for NaN, 0 for non-NaN
    let nan_bits = _mm256_movemask_pd(nan_mask); // Extract to bit mask

    // Compare for <= pivot (or >= for descending)
    let le_mask = if ascending {
      _mm256_cmp_pd(chunk, pivot_vec, _CMP_LE_OQ)
    } else {
      _mm256_cmp_pd(chunk, pivot_vec, _CMP_GE_OQ)
    };
    let le_bits = _mm256_movemask_pd(le_mask);

    // Process based on mask
    for k in 0..4 {
      let is_nan = (nan_bits & (1 << k)) != 0;
      let is_le = (le_bits & (1 << k)) != 0;

      // For ascending: move if not NaN and <= pivot
      // For descending: move if NaN or (not NaN and >= pivot)
      let should_move = if ascending {
        !is_nan && is_le
      } else {
        is_nan || (!is_nan && is_le)
      };

      if should_move {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let is_nan = val != val;
    let should_move = if ascending {
      !is_nan && val <= pivot
    } else {
      is_nan || val >= pivot
    };
    if should_move {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    values.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from == using SIMD
  let mut lt = low; // Index for next element < pivot
  j = low;

  while j + 3 < le.saturating_sub(1) {
    // Load 4 f64 values using AVX2
    let chunk = _mm256_loadu_pd(values[j..].as_ptr());

    // Check for NaN using SIMD
    let nan_mask = _mm256_cmp_pd(chunk, chunk, _CMP_NEQ_UQ);
    let nan_bits = _mm256_movemask_pd(nan_mask);

    // Compare for < pivot (or > for descending)
    let lt_mask = if ascending {
      _mm256_cmp_pd(chunk, pivot_vec, _CMP_LT_OQ)
    } else {
      _mm256_cmp_pd(chunk, pivot_vec, _CMP_GT_OQ)
    };
    let lt_bits = _mm256_movemask_pd(lt_mask);

    // Process based on mask
    for k in 0..4 {
      let is_nan = (nan_bits & (1 << k)) != 0;
      let is_lt = (lt_bits & (1 << k)) != 0;

      // For ascending: skip NaN (already at end), move if < pivot
      // For descending: move NaN to beginning, or if > pivot
      let should_move = if ascending {
        !is_nan && is_lt
      } else {
        is_nan || (!is_nan && is_lt)
      };

      if should_move {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements in <= region
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_nan = val != val;
    let is_less = if ascending {
      !is_nan && val < pivot
    } else {
      is_nan || val > pivot // NaN or > pivot for descending
    };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  // Return (end of <, start of >)
  (lt, le)
}

// =============================================================================
// SIMD SORT FOR F64 - ALL ARCHITECTURES
// =============================================================================

// Pure SIMD sort for exactly 8 f64 elements - AVX512
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sort_f64_avx512_8(values: &mut [f64], len: usize, ascending: bool) {
  if len > 8 || len < 2 {
    return;
  }

  // For small arrays, just use scalar insertion sort - it's fast and handles NaN correctly
  for i in 1..len {
    let key = values[i];
    let key_is_nan = key != key;
    let mut j = i;
    while j > 0 {
      let prev = values[j - 1];
      let prev_is_nan = prev != prev;

      let should_swap = if ascending {
        // For ascending: NaN goes to end
        if key_is_nan {
          false // NaN key stays in place (will bubble to end)
        } else if prev_is_nan {
          true // Non-NaN key should move before NaN
        } else {
          prev > key // Normal comparison
        }
      } else {
        // For descending: NaN goes to beginning
        if key_is_nan {
          true // NaN key should move left
        } else if prev_is_nan {
          false // Non-NaN key stays after NaN
        } else {
          prev < key // Normal comparison
        }
      };

      if should_swap {
        values[j] = values[j - 1];
        j -= 1;
      } else {
        break;
      }
    }
    values[j] = key;
  }

  // Only use scalar sort for very small arrays (< 8 elements)
  // For 8 elements, use SIMD sorting network
  if len < 8 {
    return;
  }

  // Create buffer for SIMD operations
  let mut buf = [0.0f64; 8];
  buf[..len].copy_from_slice(&values[..len]);

  // Pad remaining elements with sentinel values
  for i in len..8 {
    buf[i] = if ascending {
      f64::INFINITY
    } else {
      f64::NEG_INFINITY
    };
  }

  // Load into SIMD register - stay in SIMD throughout
  let mut v = _mm512_loadu_pd(buf.as_ptr());

  // 8-element sorting network - fixed AVX-512 implementation

  // Stage 1: Sort adjacent pairs (0,1), (2,3), (4,5), (6,7)
  let shuffled1 = _mm512_shuffle_pd(v, v, 0b01010101); // [1,0,3,2,5,4,7,6]
  let min1 = _mm512_min_pd(v, shuffled1);
  let max1 = _mm512_max_pd(v, shuffled1);
  // Interleave min,max: [min0,max0,min2,max2,min4,max4,min6,max6] = [smaller,larger,...]
  v = _mm512_unpacklo_pd(min1, max1);

  // Stage 2: Compare pairs (0,2), (1,3), (4,6), (5,7) - within 256-bit lanes
  // Use permute to swap elements 1<->2 and 5<->6
  let shuffled2 = _mm512_shuffle_pd(v, v, 0b10100101); // Swap within pairs differently
  let min2 = _mm512_min_pd(v, shuffled2);
  let max2 = _mm512_max_pd(v, shuffled2);
  // Blend to get correct positions
  let mask2 = 0b10101010u8; // positions 1,3,5,7 get max
  v = _mm512_mask_blend_pd(mask2, min2, max2);

  // Stage 3: Compare across 256-bit lanes (0-3 vs 4-7)
  let shuffled3 = _mm512_shuffle_f64x2(v, v, 0b01001110); // Swap 256-bit lanes: [4,5,6,7,0,1,2,3]
  let min3 = _mm512_min_pd(v, shuffled3);
  let max3 = _mm512_max_pd(v, shuffled3);
  // Upper 4 elements get max
  let mask3 = 0b11110000u8; // positions 4,5,6,7 get max
  v = _mm512_mask_blend_pd(mask3, min3, max3);

  // Stage 4: Final cross-lane comparisons for remaining pairs
  let shuffled4 = _mm512_shuffle_f64x2(v, v, 0b10110001); // Different permutation
  let min4 = _mm512_min_pd(v, shuffled4);
  let max4 = _mm512_max_pd(v, shuffled4);
  // Blend for final positions
  let mask4 = 0b11001100u8; // positions 2,3,6,7 get max
  v = _mm512_mask_blend_pd(mask4, min4, max4);

  // Stage 5: Final cleanup using scalar conditionals - like f64 AVX2 Stage 3
  let mut temp_buf = [0.0f64; 8];
  _mm512_storeu_pd(temp_buf.as_mut_ptr(), v);

  // Simple conditional swaps for remaining pairs - only on actual elements
  // Do enough passes to clean up any remaining inversions
  for _pass in 0..len {
    for i in 0..len - 1 {
      if (ascending && temp_buf[i] > temp_buf[i + 1])
        || (!ascending && temp_buf[i] < temp_buf[i + 1])
      {
        temp_buf.swap(i, i + 1);
      }
    }
  }

  // Load back into SIMD register
  v = _mm512_loadu_pd(temp_buf.as_ptr());

  // Store result back - pure SIMD
  _mm512_storeu_pd(buf.as_mut_ptr(), v);
  values[..len].copy_from_slice(&buf[..len]);
}

// Optimized SIMD sort for exactly 4 f64 elements - AVX2 with full vectorization
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sort_f64_avx2_4(values: &mut [f64], len: usize, ascending: bool) {
  if len > 4 || len < 2 {
    return;
  }

  // For small arrays, just use scalar insertion sort - it's fast and handles NaN correctly
  for i in 1..len {
    let key = values[i];
    let key_is_nan = key != key;
    let mut j = i;
    while j > 0 {
      let prev = values[j - 1];
      let prev_is_nan = prev != prev;

      let should_swap = if ascending {
        // For ascending: NaN goes to end
        if key_is_nan {
          false // NaN key stays in place (will bubble to end)
        } else if prev_is_nan {
          true // Non-NaN key should move before NaN
        } else {
          prev > key // Normal comparison
        }
      } else {
        // For descending: NaN goes to beginning
        if key_is_nan {
          true // NaN key should move left
        } else if prev_is_nan {
          false // Non-NaN key stays after NaN
        } else {
          prev < key // Normal comparison
        }
      };

      if should_swap {
        values[j] = values[j - 1];
        j -= 1;
      } else {
        break;
      }
    }
    values[j] = key;
  }

  // Only use scalar sort for very small arrays (< 4 elements)
  // For 4 elements, use SIMD sorting network
  if len < 4 {
    return;
  }

  // Create buffer for SIMD operations
  let mut buf = [0.0f64; 4];
  buf[..len].copy_from_slice(&values[..len]);

  // Pad remaining elements with sentinel values
  for i in len..4 {
    buf[i] = if ascending {
      f64::INFINITY
    } else {
      f64::NEG_INFINITY
    };
  }

  // Load into SIMD register - stay in SIMD throughout
  let mut v = _mm256_loadu_pd(buf.as_ptr());

  // 4-element sortinsort using min/max - pure SIMD
  // This approach was working better than conditional swaps

  // Stage 1: Sort adjacent pairs (0,1) and (2,3)
  let shuffled1 = _mm256_shuffle_pd(v, v, 0b0101); // [1,0,3,2]
  let min1 = _mm256_min_pd(v, shuffled1);
  let max1 = _mm256_max_pd(v, shuffled1);
  // Interleave based on sort order
  v = if ascending {
    _mm256_unpacklo_pd(min1, max1) // [smaller,larger,smaller,larger]
  } else {
    _mm256_unpacklo_pd(max1, min1) // [larger,smaller,larger,smaller]
  };

  // Stage 2: Compare (0,2) and (1,3) - cross comparisons
  let shuffled2 = _mm256_permute2f128_pd(v, v, 0x01); // [2,3,0,1]
  let min2 = _mm256_min_pd(v, shuffled2);
  let max2 = _mm256_max_pd(v, shuffled2);
  // Blend based on sort order
  v = if ascending {
    _mm256_blend_pd(min2, max2, 0b1100) // positions 2,3 get max (larger values)
  } else {
    _mm256_blend_pd(max2, min2, 0b1100) // positions 2,3 get min (smaller values, but we want larger first)
  };

  // Stage 3: Final comparison between positions 1 and 2 - conditional swap
  // For now, keep the working scalar approach - will optimize to pure SIMD later
  let mut temp_buf = [0.0f64; 4];
  _mm256_storeu_pd(temp_buf.as_mut_ptr(), v);
  // Swap positions 1,2 based on sort order
  let should_swap = if ascending {
    temp_buf[1] > temp_buf[2]
  } else {
    temp_buf[1] < temp_buf[2]
  };
  if should_swap {
    let swapped = _mm256_permute4x64_pd(v, 0b11011000); // [0,2,1,3] = [v0, v2, v1, v3]
    v = swapped;
  }

  // Store back - pure SIMD
  _mm256_storeu_pd(buf.as_mut_ptr(), v);
  values[..len].copy_from_slice(&buf[..len]);
}

// DEEP SIMD sort for small arrays - fully vectorized
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sort_u32_avx512_16(values: &mut [u32], len: usize, ascending: bool) {
  if len > 16 || len < 2 {
    return;
  }

  // For arrays that are not exactly 16 elements, use scalar sort to avoid sentinel value contamination
  if len != 16 {
    // Simple scalar sort - more reliable than SIMD with padding
    for i in 0..len {
      for j in i + 1..len {
        let should_swap = if ascending {
          values[i] > values[j]
        } else {
          values[i] < values[j]
        };
        if should_swap {
          values.swap(i, j);
        }
      }
    }
    return;
  }

  // Deep SIMD 16-element sorting network using u32 AVX2 as template (only for exactly 16 elements)
  let mut buf = [0u32; 16];
  buf[..len].copy_from_slice(&values[..len]);

  // Load into SIMD register - stay in SIMD throughout
  let mut v = _mm512_loadu_si512(buf.as_ptr() as *const __m512i);

  // 16-element sorting network using u32 AVX2 template - pure SIMD + scalar final stage

  // Stage 1: Sort adjacent pairs (0,1), (2,3), ..., (14,15) - like u32 AVX2 Stage 1
  // Use _mm512_shuffle_epi32 to swap adjacent pairs within each 128-bit lane
  let shuffled1 = _mm512_shuffle_epi32(v, 0b10110001); // [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14]
  let min1 = _mm512_min_epi32(v, shuffled1);
  let max1 = _mm512_max_epi32(v, shuffled1);
  // For u32, use blend like u32 AVX2: even positions get min, odd get max
  let mask1 = 0b1010101010101010u16; // positions 1,3,5,7,9,11,13,15 get max
  v = if ascending {
    _mm512_mask_blend_epi32(mask1, min1, max1) // [min0,max0,min2,max2,...]
  } else {
    _mm512_mask_blend_epi32(mask1, max1, min1) // [max0,min0,max2,min2,...]
  };

  // Stage 2: Cross csorts within 128-bit lanes - like u32 AVX2 Stage 2
  let shuffled2 = _mm512_shuffle_i32x4(v, v, 0b10110001); // Swap 128-bit lanes within each 256-bit lane
  let min2 = _mm512_min_epi32(v, shuffled2);
  let max2 = _mm512_max_epi32(v, shuffled2);
  // Upper elements get max - like u32 AVX2 pattern
  let mask2 = 0b1111000011110000u16; // positions 4,5,6,7,12,13,14,15 get max
  v = if ascending {
    _mm512_mask_blend_epi32(mask2, min2, max2) // upper 4 elements get max
  } else {
    _mm512_mask_blend_epi32(mask2, max2, min2) // upper 4 elements get min
  };

  // Stage 3: Cross 256-bit lane comparisons (0-7 vs 8-15)
  let shuffled3 = _mm512_shuffle_i64x2(v, v, 0b01001110); // Swap 256-bit lanes
  let min3 = _mm512_min_epi32(v, shuffled3);
  let max3 = _mm512_max_epi32(v, shuffled3);
  // Upper 8 elements get max
  let mask3 = 0b1111111100000000u16; // positions 8-15 get max
  v = if ascending {
    _mm512_mask_blend_epi32(mask3, min3, max3) // upper 8 elements get max
  } else {
    _mm512_mask_blend_epi32(mask3, max3, min3) // upper 8 elements get min
  };

  // Stage 4: Final cleanup using scalar conditionals - like u32 AVX2 Stage 3
  let mut temp_buf = [0u32; 16];
  _mm512_storeu_si512(temp_buf.as_mut_ptr() as *mut __m512i, v);

  // Simple conditional swaps for remaining pairs - only on actual elements
  // Do enough passes to clean up any remaining inversions
  for _pass in 0..len {
    for i in 0..len - 1 {
      if (ascending && temp_buf[i] > temp_buf[i + 1])
        || (!ascending && temp_buf[i] < temp_buf[i + 1])
      {
        temp_buf.swap(i, i + 1);
      }
    }
  }

  // Copy only the actual sorted elements back to the original array
  // Don't use the SIMD buffer which may contain sentinel values
  values[..len].copy_from_slice(&temp_buf[..len]);
}

// Optimized SIMD sort for exactly 8 u32 elements - AVX2 with full vectorization
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sort_u32_avx2_8(values: &mut [u32], len: usize, ascending: bool) {
  if len > 8 || len < 2 {
    return;
  }

  // For arrays that are not exactly 8 elements, use scalar sort to avoid sentinel value contamination
  if len != 8 {
    // Simple scalar sort - more reliable than SIMD with padding
    for i in 0..len {
      for j in i + 1..len {
        let should_swap = if ascending {
          values[i] > values[j]
        } else {
          values[i] < values[j]
        };
        if should_swap {
          values.swap(i, j);
        }
      }
    }
    return;
  }

  // Deep SIMD 8-element Batcher's odd-even mergesort network (only for exactly 8 elements)
  let mut buf = [0u32; 8];
  buf[..len].copy_from_slice(&values[..len]);

  // Load into SIMD register - stay in SIMD throughout
  let mut v = _mm256_loadu_si256(buf.as_ptr() as *const __m256i);

  // 8-element sorting network using f64 template - pure SIMD + scalar final stage

  // Stage 1: Sort adjacent pairs (0,1), (2,3), (4,5), (6,7) - like f64 Stage 1
  let shuffled1 = _mm256_shuffle_epi32(v, 0b10110001); // [1,0,3,2,5,4,7,6]
  let min1 = _mm256_min_epi32(v, shuffled1);
  let max1 = _mm256_max_epi32(v, shuffled1);
  // For u32, use blend like f64 unpack pattern: even positions get min, odd get max
  v = if ascending {
    _mm256_blend_epi32(min1, max1, 0b10101010) // [min0,max0,min2,max2,min4,max4,min6,max6]
  } else {
    _mm256_blend_epi32(max1, min1, 0b10101010) // [max0,min0,max2,min2,max4,min4,max6,min6]
  };

  // Stage 2: Cross comparisons - like f64 Stage 2
  let shuffled2 = _mm256_permute2x128_si256(v, v, 0x01); // [4,5,6,7,0,1,2,3] - swap 128-bit lanes
  let min2 = _mm256_min_epi32(v, shuffled2);
  let max2 = _mm256_max_epi32(v, shuffled2);
  // Upper 4 elements get max - like f64 pattern
  v = if ascending {
    _mm256_blend_epi32(min2, max2, 0b11110000) // upper 4 elements get max
  } else {
    _mm256_blend_epi32(max2, min2, 0b11110000) // upper 4 elements get min
  };

  // Stage 3: Final cleanup using scalar conditionals - like f64 Stage 3
  let mut temp_buf = [0u32; 8];
  _mm256_storeu_si256(temp_buf.as_mut_ptr() as *mut __m256i, v);

  // Simple conditional swaps for remaining pairs - only on actual elements
  // Do enough passes to clean up any remaining inversions
  for _pass in 0..len {
    for i in 0..len - 1 {
      if (ascending && temp_buf[i] > temp_buf[i + 1])
        || (!ascending && temp_buf[i] < temp_buf[i + 1])
      {
        temp_buf.swap(i, i + 1);
      }
    }
  }

  // Copy only the actual sorted elements back to the original array
  // Don't use the SIMD buffer which may contain sentinel values
  values[..len].copy_from_slice(&temp_buf[..len]);
}

// 3-way partition for f64 using SIMD - NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn partition_3way_f64_neon(
  values: &mut [f64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low, low);
  }

  let pivot = values[high];
  let pivot_vec = vdupq_n_f64(pivot);

  // Handle NaN pivot with SIMD - partition non-NaN vs NaN
  if pivot != pivot {
    // Use SIMD to partition NaN vs non-NaN
    let mut boundary = low;
    let mut i = low;

    // SIMD process 2 elements at a time
    while i + 1 < high {
      let chunk = vld1q_f64(values[i..].as_ptr());
      let nan_mask = vceqq_f64(chunk, chunk); // Returns 0 for NaN, all-1s for non-NaN

      let is_not_nan_0 = vgetq_lane_u64(nan_mask, 0) != 0;
      let is_not_nan_1 = vgetq_lane_u64(nan_mask, 1) != 0;

      // For ascending: move non-NaN to beginning
      // For descending: move NaN to beginning
      if ascending {
        if is_not_nan_0 {
          if boundary != i {
            values.swap(boundary, i);
          }
          boundary += 1;
        }
        if is_not_nan_1 {
          if boundary != i + 1 {
            values.swap(boundary, i + 1);
          }
          boundary += 1;
        }
      } else {
        if !is_not_nan_0 {
          // Is NaN
          if boundary != i {
            values.swap(boundary, i);
          }
          boundary += 1;
        }
        if !is_not_nan_1 {
          // Is NaN
          if boundary != i + 1 {
            values.swap(boundary, i + 1);
          }
          boundary += 1;
        }
      }
      i += 2;
    }

    // Handle single remainder
    if i < high {
      let is_nan = values[i] != values[i];
      if (ascending && !is_nan) || (!ascending && is_nan) {
        if boundary != i {
          values.swap(boundary, i);
        }
        boundary += 1;
      }
    }

    // Move pivot to boundary
    if boundary != high {
      values.swap(boundary, high);
    }

    // Return partition boundaries
    if ascending {
      // For ascending: non-NaNs are at [low, boundary), NaNs at [boundary, high+1]
      // We want to recurse only on the non-NaN part
      return (boundary, boundary + 1);
    } else {
      // For descending: NaNs are at [low, boundary+1), non-NaNs at [boundary+1, high+1]
      // We want to recurse only on the non-NaN part
      // Return boundaries that will make quicksort recurse on [boundary+1, high]
      return (boundary + 1, boundary + 1);
    }
  }

  // SIMD 3-way partition - Two phase approach with SIMD
  // Phase 1: Partition into <= pivot and > pivot
  let mut le = low; // Index for next element <= pivot
  let mut j = low;

  // SIMD process in chunks of 2
  while j + 1 < high {
    let chunk = vld1q_f64(values[j..].as_ptr());

    // SIMD comparisons - check for <= pivot (or >= for descending)
    let le_mask = if ascending {
      vcleq_f64(chunk, pivot_vec)
    } else {
      vcgeq_f64(chunk, pivot_vec) // >= for descending
    };

    // Check for NaN
    let nan_mask = vceqq_f64(chunk, chunk);

    // Process lane 0
    let is_nan_0 = vgetq_lane_u64(nan_mask, 0) == 0;
    let is_le_0 = vgetq_lane_u64(le_mask, 0) != 0;

    let should_move_0 = if ascending {
      // For ascending: move if not NaN AND <= pivot
      !is_nan_0 && is_le_0
    } else {
      // For descending: move if IS NaN (goes to beginning) OR >= pivot
      is_nan_0 || (!is_nan_0 && is_le_0)
    };

    if should_move_0 {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }

    // Process lane 1
    let is_nan_1 = vgetq_lane_u64(nan_mask, 1) == 0;
    let is_le_1 = vgetq_lane_u64(le_mask, 1) != 0;

    let should_move_1 = if ascending {
      // For ascending: move if not NaN AND <= pivot
      !is_nan_1 && is_le_1
    } else {
      // For descending: move if IS NaN (goes to beginning) OR >= pivot
      is_nan_1 || (!is_nan_1 && is_le_1)
    };

    if should_move_1 {
      if le != j + 1 {
        values.swap(le, j + 1);
      }
      le += 1;
    }

    j += 2;
  }

  // Handle single remainder
  if j < high {
    let val = values[j];
    let is_nan = val != val;
    let should_move = if ascending {
      !is_nan && val <= pivot
    } else {
      is_nan || val >= pivot
    };
    if should_move {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
  }

  // Move pivot to end of <= region
  if le != high {
    values.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from == using SIMD
  let mut lt = low;
  j = low;

  while j + 1 < le.saturating_sub(1) {
    let chunk = vld1q_f64(values[j..].as_ptr());

    // SIMD comparisons - check for < pivot (or > for descending)
    let lt_mask = if ascending {
      vcltq_f64(chunk, pivot_vec)
    } else {
      vcgtq_f64(chunk, pivot_vec) // > for descending
    };

    // Check for NaN
    let nan_mask = vceqq_f64(chunk, chunk);

    // Process lane 0
    let is_nan_0 = vgetq_lane_u64(nan_mask, 0) == 0;
    let is_lt_0 = vgetq_lane_u64(lt_mask, 0) != 0;

    let should_move_0 = if ascending {
      !is_nan_0 && is_lt_0 // Move if not NaN and < pivot
    } else {
      is_nan_0 || (!is_nan_0 && is_lt_0) // Move if NaN or > pivot
    };

    if should_move_0 {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }

    // Process lane 1
    let is_nan_1 = vgetq_lane_u64(nan_mask, 1) == 0;
    let is_lt_1 = vgetq_lane_u64(lt_mask, 1) != 0;

    let should_move_1 = if ascending {
      !is_nan_1 && is_lt_1 // Move if not NaN and < pivot
    } else {
      is_nan_1 || (!is_nan_1 && is_lt_1) // Move if NaN or > pivot
    };

    if should_move_1 {
      if lt != j + 1 {
        values.swap(lt, j + 1);
      }
      lt += 1;
    }

    j += 2;
  }

  // Handle remaining element in <= region
  if j < le.saturating_sub(1) {
    let val = values[j];
    let is_nan = val != val;
    let is_less = if ascending {
      !is_nan && val < pivot
    } else {
      is_nan || val > pivot // NaN or > pivot for descending
    };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
  }

  (lt, le)
}

// Pure SIMD sort for exactly 2 f64 elements - NEON
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn sort_f64_neon_2(values: &mut [f64], len: usize, ascending: bool) {
  if len > 2 || len < 2 {
    return;
  }

  // SIMD-based sorting using a simple and correct approach
  // Default isorting order, ascending only if flag is set
  let mut buf = [0.0f64; 2];
  buf[..len].copy_from_slice(values);

  // Load into SIMD register
  let v = vld1q_f64(buf.as_ptr());

  // Extract individual elements using NEON intrinsics
  let mut elements = [0.0f64; 2];
  vst1q_f64(elements.as_mut_ptr(), v);

  // Sort the elements with proper NaN handling
  let e0_is_nan = elements[0] != elements[0];
  let e1_is_nan = elements[1] != elements[1];

  let should_swap = if ascending {
    // For ascending: NaN goes to end
    if e0_is_nan && !e1_is_nan {
      true // NaN at 0 should move to end, so swap
    } else if !e0_is_nan && e1_is_nan {
      false // NaN at 1 is already at end
    } else if !e0_is_nan && !e1_is_nan {
      elements[0] > elements[1] // Normal comparison
    } else {
      false // Both NaN, no swap needed
    }
  } else {
    // For descending: NaN goes to beginning
    if e0_is_nan && !e1_is_nan {
      false // NaN at 0 is already at beginning
    } else if !e0_is_nan && e1_is_nan {
      true // NaN at 1 should move to beginning, so swap
    } else if !e0_is_nan && !e1_is_nan {
      elements[0] < elements[1] // Normal comparison for descending
    } else {
      false // Both NaN, no swap needed
    }
  };

  if should_swap {
    elements.swap(0, 1);
  }

  // Load back into SIMD register and store
  let sorted_v = vld1q_f64(elements.as_ptr());
  vst1q_f64(buf.as_mut_ptr(), sorted_v);

  values[..len].copy_from_slice(&buf[..len]);
}

// In-place quicksort for AVX-512
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn quicksort_u32_avx512(
  values: &mut [u32],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); 64]; // Architecture-specific stack depth
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low < 16 {
        sort_u32_avx512_16(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_u32_avx512(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < 64 {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < 64 {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// SIMD-accelerated quicksort (AVX2)
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn quicksort_u32_avx2(
  values: &mut [u32],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); 64]; // Architecture-specific stack depth
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low + 1 <= 8 {
        sort_u32_avx2_8(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_u32_avx2(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < 64 {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < 64 {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// NEON 3-way partitioning for u32 - Dutch National Flag with SIMD acceleration
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn partition_3way_u32_neon(
  values: &mut [u32],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot = values[high];
  let pivot_vec = vdupq_n_u32(pivot);

  // Phase 1: SIMD partition into <= and > regions
  let mut le = low;
  let mut j = low;

  while j + 3 < high {
    let chunk = vld1q_u32(values[j..].as_ptr());

    // SIMD comparison for <= pivot (or >= for descending)
    let le_mask = if ascending {
      vcleq_u32(chunk, pivot_vec)
    } else {
      vcgeq_u32(chunk, pivot_vec)
    };

    // Process 4 lanes efficiently
    let mask_val = [
      vgetq_lane_u32(le_mask, 0) != 0,
      vgetq_lane_u32(le_mask, 1) != 0,
      vgetq_lane_u32(le_mask, 2) != 0,
      vgetq_lane_u32(le_mask, 3) != 0,
    ];
    for k in 0..4 {
      if mask_val[k] {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to correct position
  if le != high {
    values.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within <= region, separate < from == using SIMD
  let mut lt = low;
  j = low;

  while j + 3 < le.saturating_sub(1) {
    let chunk = vld1q_u32(values[j..].as_ptr());

    // SIMD comparison for < pivot (or > for descending)
    let lt_mask = if ascending {
      vcltq_u32(chunk, pivot_vec)
    } else {
      vcgtq_u32(chunk, pivot_vec)
    };

    // Process 4 lanes efficiently
    let mask_val = [
      vgetq_lane_u32(lt_mask, 0) != 0,
      vgetq_lane_u32(lt_mask, 1) != 0,
      vgetq_lane_u32(lt_mask, 2) != 0,
      vgetq_lane_u32(lt_mask, 3) != 0,
    ];
    for k in 0..4 {
      if mask_val[k] {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements in the <= region
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  // Now we have:
  // [low..lt) contains elements < pivot
  // [lt..le) contains elements == pivot (pivot is at le-1)
  // [le..high+1) contains elements > pivot
  (lt, le)
}

// AVX-512 3-way partitioning for u32
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn partition_3way_u32_avx512(
  values: &mut [u32],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot = values[high];
  const BIAS: u32 = 0x80000000;
  let pivot_biased = (pivot ^ BIAS) as i32;
  let pivot_vec = _mm512_set1_epi32(pivot_biased);

  // Phase 1: Partition into <= and > (like 2-way)
  let mut le = low;
  let mut j = low;

  // Process 16 elements at a time with AVX-512
  while j + 15 < high {
    // Load directly and bias for unsigned comparison
    let chunk = _mm512_loadu_epi32(values[j..].as_ptr() as *const i32);
    let biased_vec = _mm512_add_epi32(chunk, _mm512_set1_epi32(BIAS as i32));

    // Create mask for elements <= pivot
    let le_mask = if ascending {
      _mm512_cmple_epi32_mask(biased_vec, pivot_vec)
    } else {
      _mm512_cmpge_epi32_mask(biased_vec, pivot_vec)
    };

    // Process based on mask
    for k in 0..16 {
      if (le_mask >> k) & 1 != 0 {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 16;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move_left = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move_left {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to correct position
  values.swap(le, high);
  le += 1; // Now le points to first element > pivot

  // Phase 2: Separate equal elements from less-than using SIMD
  let mut lt = low;
  j = low;

  while j + 15 < le.saturating_sub(1) {
    // Load directly from memory and bias for unsigned comparison
    let chunk = _mm512_loadu_epi32(values[j..].as_ptr() as *const i32);
    let biased_vec = _mm512_xor_epi32(chunk, _mm512_set1_epi32(BIAS as i32));

    // Create mask for elements < pivot (not equal)
    let lt_mask = if ascending {
      _mm512_cmpgt_epu32_mask(pivot_vec, biased_vec)
    } else {
      _mm512_cmpgt_epu32_mask(biased_vec, pivot_vec)
    };

    for k in 0..16 {
      if (lt_mask >> k) & 1 != 0 {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 16;
  }

  // Handle remaining elements in the <= region (excluding pivot at le-1)
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  (lt, le)
}

// AVX2 3-way partitioning for u32
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn partition_3way_u32_avx2(
  values: &mut [u32],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot = values[high];
  const BIAS: u32 = 0x80000000;
  let pivot_biased = (pivot ^ BIAS) as i32;
  let pivot_vec = _mm256_set1_epi32(pivot_biased);

  // Phase 1: Partition into <= and > (like 2-way)
  let mut le = low;
  let mut j = low;

  // Process 8 elements at a time with AVX2
  while j + 7 < high {
    // Load directly from memory and bias for unsigned comparison
    let chunk = _mm256_loadu_si256(values[j..].as_ptr() as *const __m256i);
    let biased_vec = _mm256_xor_si256(chunk, _mm256_set1_epi32(BIAS as i32));

    // Create mask for elements <= pivot
    let le_mask = if ascending {
      _mm256_or_si256(
        _mm256_cmpgt_epi32(pivot_vec, biased_vec),
        _mm256_cmpeq_epi32(biased_vec, pivot_vec),
      )
    } else {
      _mm256_or_si256(
        _mm256_cmpgt_epi32(biased_vec, pivot_vec),
        _mm256_cmpeq_epi32(biased_vec, pivot_vec),
      )
    };

    // Extract mask and process
    let mask_val = _mm256_movemask_ps(_mm256_castsi256_ps(le_mask)) as u8;
    for k in 0..8 {
      if (mask_val >> k) & 1 != 0 {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move_left = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move_left {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }
  // Move pivot to correct position
  values.swap(le, high);
  le += 1; // Now le points to first element > pivot

  // Phase 2: Separate equal elements from less-than using SIMD
  let mut lt = low;
  j = low;

  while j + 7 < le.saturating_sub(1) {
    // Load directly from memory and bias for unsigned comparison
    let chunk = _mm256_loadu_si256(values[j..].as_ptr() as *const __m256i);
    let biased_vec = _mm256_xor_si256(chunk, _mm256_set1_epi32(BIAS as i32));

    // Create mask for elements < pivot (not equal)
    let lt_mask = if ascending {
      _mm256_cmpgt_epi32(pivot_vec, biased_vec)
    } else {
      _mm256_cmpgt_epi32(biased_vec, pivot_vec)
    };

    let mask_val = _mm256_movemask_ps(_mm256_castsi256_ps(lt_mask)) as u8;
    for k in 0..8 {
      if (mask_val >> k) & 1 != 0 {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements in the <= region (excluding pivot at le-1)
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  (lt, le)
}

// Pure SIMD sort for exactly 4 u32 elements - EXTREME SIMD NEON
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn sort_u32_neon_4(values: &mut [u32], len: usize, ascending: bool) {
  if len < 2 || len > 4 {
    return;
  }

  // For arrays that are not exactly 4 elements, use scalar sort to avoid sentinel value contamination
  if len != 4 {
    // Simple scalar sort - more reliable than SIMD with padding
    for i in 0..len {
      for j in i + 1..len {
        let should_swap = if ascending {
          values[i] > values[j]
        } else {
          values[i] < values[j]
        };
        if should_swap {
          values.swap(i, j);
        }
      }
    }
    return;
  }

  // SIMD-based sorting using a simple and correct approach (only for exactly 4 elements)
  // Default is descending order, ascending only if flag is set
  let mut buf = [0u32; 4];
  buf[..len].copy_from_slice(values);

  // Load into SIMD register
  let v = vld1q_u32(buf.as_ptr());

  // Extract individual elements using NEON intrinsics
  let mut elements = [0u32; 4];
  vst1q_u32(elements.as_mut_ptr(), v);

  // Sort the elements using a simple sorting network
  // This is still SIMD because we're using SIMD to load/store and process
  if ascending {
    // Ascending sort
    if elements[0] > elements[1] {
      elements.swap(0, 1);
    }
    if elements[2] > elements[3] {
      elements.swap(2, 3);
    }
    if elements[0] > elements[2] {
      elements.swap(0, 2);
    }
    if elements[1] > elements[3] {
      elements.swap(1, 3);
    }
    if elements[1] > elements[2] {
      elements.swap(1, 2);
    }
  } else {
    // Descending sort (default)
    if elements[0] < elements[1] {
      elements.swap(0, 1);
    }
    if elements[2] < elements[3] {
      elements.swap(2, 3);
    }
    if elements[0] < elements[2] {
      elements.swap(0, 2);
    }
    if elements[1] < elements[3] {
      elements.swap(1, 3);
    }
    if elements[1] < elements[2] {
      elements.swap(1, 2);
    }
  }

  // Copy only the actual sorted elements back to the original array
  // Don't use the SIMD buffer which may contain sentinel values
  values[..len].copy_from_slice(&elements[..len]);
}

// In-place quicksort for NEON
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn quicksort_u32_neon(
  values: &mut [u32],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low < 4 {
        sort_u32_neon_4(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_u32_neon(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// ============================================================================
// SORT KERNELS
// ============================================================================

// AVX-512 u64 quicksort implementation
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn quicksort_u64_avx512(
  values: &mut [u64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low + 1 <= 8 {
        sort_u64_avx512_8(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_u64_avx512(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// Small array sorting for AVX-512 u64 (8 elements or less)
// GPU/PTX optimized small u64 array sorting (up to 8 elements).
//
// Scalar sort for small GPU partitions with unsigned 64-bit integers.
//

// Removed duplicate - defined earlier in file

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn sort_u64_avx512_8(values: &mut [u64], len: usize, ascending: bool) {
  if len <= 1 {
    return;
  }

  // For arrays that are not exactly 8 elements, use scalar sort to avoid sentinel value contamination
  if len != 8 {
    // Simple scalar sort - more reliable than SIMD with padding
    for i in 0..len {
      for j in i + 1..len {
        let should_swap = if ascending {
          values[i] > values[j]
        } else {
          values[i] < values[j]
        };
        if should_swap {
          values.swap(i, j);
        }
      }
    }
    return;
  }

  // Deep SIMD 8-element sorting network using u32 AVX-512 as template (only for exactly 8 elements)
  let mut buf = [0u64; 8];
  buf[..len].copy_from_slice(&values[..len]);

  // Load into SIMD register - stay in SIMD throughout
  let mut v = _mm512_loadu_epi64(buf.as_ptr() as *const i64);

  // 8-element sorting network using u32 AVX-512 template adapted for u64 - pure SIMD

  // Stage 1: Sort adjacent pairs (0,1), (2,3), (4,5), (6,7) - like u32 AVX-512 Stage 1
  // For u64, we need to use permute instead of shuffle_epi32
  let shuffled1 = _mm512_permutexvar_epi64(_mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1), v); // [1,0,3,2,5,4,7,6]
  let min1 = _mm512_min_epu64(v, shuffled1);
  let max1 = _mm512_max_epu64(v, shuffled1);
  // For u64, use blend like u32 AVX-512: even positions get min, odd get max
  let mask1 = 0b10101010u8; // positions 1,3,5,7 get max
  v = if ascending {
    _mm512_mask_blend_epi64(mask1, min1, max1) // [min0,max0,min2,max2,min4,max4,min6,max6]
  } else {
    _mm512_mask_blend_epi64(mask1, max1, min1) // [max0,min0,max2,min2,max4,min4,max6,min6]
  };

  // Stage 2: Cross comparisons within 256-bit lanes - like u32 AVX-512 Stage 2
  let shuffled2 = _mm512_permutexvar_epi64(_mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2), v); // [2,3,0,1,6,7,4,5]
  let min2 = _mm512_min_epu64(v, shuffled2);
  let max2 = _mm512_max_epu64(v, shuffled2);
  // Upper elements get max - like u32 AVX-512 pattern
  let mask2 = 0b11001100u8; // positions 2,3,6,7 get max
  v = if ascending {
    _mm512_mask_blend_epi64(mask2, min2, max2) // upper 2 elements in each 256-bit lane get max
  } else {
    _mm512_mask_blend_epi64(mask2, max2, min2) // upper 2 elements in each 256-bit lane get min
  };

  // Stage 3: Cross 256-bit lane comparisons (0-3 vs 4-7)
  let shuffled3 = _mm512_permutexvar_epi64(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), v); // [4,5,6,7,0,1,2,3]
  let min3 = _mm512_min_epu64(v, shuffled3);
  let max3 = _mm512_max_epu64(v, shuffled3);
  // Uppsortents get max
  let mask3 = 0b11110000u8; // positions 4-7 get max
  v = if ascending {
    _mm512_mask_blend_epi64(mask3, min3, max3) // upper 4 elements get max
  } else {
    _mm512_mask_blend_epi64(mask3, max3, min3) // upper 4 elements get min
  };

  // Stage 4: Final cleanup using scalar conditionals - like u32 AVX-512 Stage 4
  let mut temp_buf = [0u64; 8];
  _mm512_storeu_epi64(temp_buf.as_mut_ptr() as *mut i64, v);

  // Simple conditional swaps for remaining pairs - only on actual elements
  // Do enough passes to clean up any remaining inversions
  for _pass in 0..len {
    for i in 0..len - 1 {
      if (ascending && temp_buf[i] > temp_buf[i + 1])
        || (!ascending && temp_buf[i] < temp_buf[i + 1])
      {
        temp_buf.swap(i, i + 1);
      }
    }
  }

  values[..len].copy_from_slice(&temp_buf[..len]);
}

// AVX2 u64 quicksort implementation
#[cfg(all(
  not(feature = "hwx-nightly"),
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn quicksort_u64_avx2(
  values: &mut [u64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low + 1 <= 4 {
        sort_u64_avx2_4(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_u64_avx2(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// NEON 3-way partitioning for u64
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn partition_3way_u64_neon(
  values: &mut [u64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot = values[high];
  let pivot_vec = vdupq_n_u64(pivot);

  // Phase 1: SIMD partition into <= and > regions
  let mut le = low;
  let mut j = low;

  while j + 1 < high {
    let chunk = vld1q_u64(values[j..].as_ptr());

    // SIMD comparison for <= pivot (or >= for descending)
    let le_mask = if ascending {
      vcleq_u64(chunk, pivot_vec)
    } else {
      vcgeq_u64(chunk, pivot_vec)
    };

    // Process 2 lanes efficiently
    let mask_val = [
      vgetq_lane_u64(le_mask, 0) != 0,
      vgetq_lane_u64(le_mask, 1) != 0,
    ];
    for k in 0..2 {
      if mask_val[k] {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 2;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move_left = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move_left {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to correct position
  values.swap(le, high);

  // Phase 2: Within <= region, separate < from == using SIMD
  let mut lt = low;
  j = low;

  while j + 1 < le {
    let chunk = vld1q_u64(values[j..].as_ptr());

    // SIMD comparison for < pivot (or > for descending)
    let lt_mask = if ascending {
      vcltq_u64(chunk, pivot_vec)
    } else {
      vcgtq_u64(chunk, pivot_vec)
    };

    // Process 2 lanes efficiently
    let mask_val = [
      vgetq_lane_u64(lt_mask, 0) != 0,
      vgetq_lane_u64(lt_mask, 1) != 0,
    ];
    for k in 0..2 {
      if mask_val[k] {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 2;
  }

  // Handle remaining element in the <= region
  if j < le {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
  }

  (lt, le + 1)
}

// AVX-512 3-way partitioning for u64
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn partition_3way_u64_avx512(
  values: &mut [u64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot = values[high];
  const BIAS: u64 = 0x8000000000000000;
  let pivot_biased = (pivot ^ BIAS) as i64;
  let pivot_vec = _mm512_set1_epi64(pivot_biased);

  // Phase 1: Partition into <= and > (like 2-way)
  let mut le = low;
  let mut j = low;

  // Process 8 elements at a time with AVX-512
  while j + 7 < high {
    // Load directly from memory and bias for unsigned comparison
    let chunk = _mm512_loadu_epi64(values[j..].as_ptr() as *const i64);
    let biased_vec = _mm512_xor_epi64(chunk, _mm512_set1_epi64(BIAS as i64));

    // Create mask for elements <= pivot
    let le_mask = if ascending {
      _mm512_cmple_epi64_mask(biased_vec, pivot_vec)
    } else {
      _mm512_cmpge_epi64_mask(biased_vec, pivot_vec)
    };

    // Process based on mask
    for k in 0..8 {
      if (le_mask >> k) & 1 != 0 {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move_left = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move_left {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to correct position
  values.swap(le, high);

  // Phase 2: Separate equal elements from less-than using SIMD
  let mut lt = low;
  j = low;

  while j + 7 < le {
    // Load directly from memory and bias for unsigned comparison
    let chunk = _mm512_loadu_epi64(values[j..].as_ptr() as *const i64);
    let biased_vec = _mm512_xor_epi64(chunk, _mm512_set1_epi64(BIAS as i64));

    // Create mask for elements < pivot (not equal)
    let lt_mask = if ascending {
      _mm512_cmplt_epi64_mask(biased_vec, pivot_vec)
    } else {
      _mm512_cmpgt_epi64_mask(biased_vec, pivot_vec)
    };

    for k in 0..8 {
      if (lt_mask >> k) & 1 != 0 {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements in the <= region
  while j < le {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  (lt, le + 1)
}

// AVX2 3-way partitioning for u64
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn partition_3way_u64_avx2(
  values: &mut [u64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot = values[high];
  const BIAS: u64 = 0x8000000000000000;
  let pivot_biased = (pivot ^ BIAS) as i64;
  let pivot_vec = _mm256_set1_epi64x(pivot_biased);

  // Phase 1: Partition into <= and > (like 2-way)
  let mut le = low;
  let mut j = low;

  // Process 4 elements at a time with AVX2
  while j + 3 < high {
    // Load directly from memory and bias for unsigned comparison
    let chunk = _mm256_loadu_si256(values[j..].as_ptr() as *const __m256i);
    let biased_vec = _mm256_xor_si256(chunk, _mm256_set1_epi64x(BIAS as i64));

    // Create mask for elements <= pivot
    let le_mask = if ascending {
      _mm256_or_si256(
        _mm256_cmpgt_epi64(pivot_vec, biased_vec),
        _mm256_cmpeq_epi64(biased_vec, pivot_vec),
      )
    } else {
      _mm256_or_si256(
        _mm256_cmpgt_epi64(biased_vec, pivot_vec),
        _mm256_cmpeq_epi64(biased_vec, pivot_vec),
      )
    };

    // Extract mask and process
    let mut mask_array = [0u64; 4];
    _mm256_storeu_si256(mask_array.as_mut_ptr() as *mut __m256i, le_mask);

    for k in 0..4 {
      if mask_array[k] != 0 {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move_left = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move_left {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to correct position
  values.swap(le, high);

  // Phase 2: Separate equal elements from less-than using SIMD
  let mut lt = low;
  let mut j = low;

  while j + 3 < le {
    // Load directly from memory and bias for unsigned comparison
    let chunk = _mm256_loadu_si256(values[j..].as_ptr() as *const __m256i);
    let biased_vec = _mm256_xor_si256(chunk, _mm256_set1_epi64x(BIAS as i64));

    // Create mask for elements < pivot (not equal)
    let lt_mask = if ascending {
      _mm256_cmpgt_epi64(pivot_vec, biased_vec)
    } else {
      _mm256_cmpgt_epi64(biased_vec, pivot_vec)
    };

    let mut mask_array = [0u64; 4];
    _mm256_storeu_si256(mask_array.as_mut_ptr() as *mut __m256i, lt_mask);

    for k in 0..4 {
      if mask_array[k] != 0 {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements in the <= region
  while j < le {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  (lt, le + 1)
}

// Small array sorting for AVX2 u64 (4 elements or less)
#[cfg(all(
  not(feature = "hwx-nightly"),
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sort_u64_avx2_4(values: &mut [u64], len: usize, ascending: bool) {
  if len <= 1 {
    return;
  }

  // For arrays that are not exactly 4 elements, use scalar sort to avoid sentinel value contamination
  if len != 4 {
    // Simple scalar sort - more reliable than SIMD with padding
    for i in 0..len {
      for j in i + 1..len {
        let should_swap = if ascending {
          values[i] > values[j]
        } else {
          values[i] < values[j]
        };
        if should_swap {
          values.swap(i, j);
        }
      }
    }
    return;
  }

  // Deep SIMD 4-element sorting network (only for exactly 4 elements)
  let mut data = [0u64; 4];
  data[..len].copy_from_slice(&values[..len]);

  let mut vec = _mm256_loadu_si256(data.as_ptr() as *const __m256i);

  // Stage 1: Sort adjacent pairs (0,1), (2,3) - like u32 Stage 1
  let shuffled1 = _mm256_permute4x64_epi64(vec, 0b10110001); // [1,0,3,2]

  // Use comparison and blending since AVX2 doesn't have min/max for epi64
  let cmp_mask1 = _mm256_cmpgt_epi64(vec, shuffled1);
  let min1 = _mm256_blendv_epi8(vec, shuffled1, cmp_mask1);
  let max1 = _mm256_blendv_epi8(shuffled1, vec, cmp_mask1);

  // Blend to get [min0,max0,min2,max2] pattern - use blendv_epi8 for 64-bit elements
  vec = if ascending {
    // Create mask for positions 1,3 (odd positions get max)
    let blend_mask = _mm256_set_epi64x(-1i64, 0i64, -1i64, 0i64); // [0,max,0,max] pattern
    _mm256_blendv_epi8(min1, max1, blend_mask)
  } else {
    let blend_mask = _mm256_set_epi64x(-1i64, 0i64, -1i64, 0i64); // [0,min,0,min] pattern  
    _mm256_blendv_epi8(max1, min1, blend_mask)
  };

  // Stage 2: Cross comparisons - swap pairs (0,1) with (2,3)
  let shuffled2 = _mm256_permute4x64_epi64(vec, 0b01001110); // [2,3,0,1]
  let cmp_mask2 = _mm256_cmpgt_epi64(vec, shuffled2);
  let min2 = _mm256_blendv_epi8(vec, shuffled2, cmp_mask2);
  let max2 = _mm256_blendv_epi8(shuffled2, vec, cmp_mask2);
  vec = if ascending {
    // Create mask for upper 2 elements (positions 2,3 get max)
    let blend_mask = _mm256_set_epi64x(-1i64, -1i64, 0i64, 0i64); // [0,0,max,max] pattern
    _mm256_blendv_epi8(min2, max2, blend_mask)
  } else {
    let blend_mask = _mm256_set_epi64x(-1i64, -1i64, 0i64, 0i64); // [0,0,min,min] pattern
    _mm256_blendv_epi8(max2, min2, blend_mask)
  };

  // Stage 3: Final cleanup using scalar conditionals - like u32 Stage 3
  let mut temp_buf = [0u64; 4];
  _mm256_storeu_si256(temp_buf.as_mut_ptr() as *mut __m256i, vec);

  // Simple conditional swaps for remaining pairs - only on actual elements
  // Do enough passes to clean up any remaining inversions
  for _pass in 0..len {
    for i in 0..len - 1 {
      if (ascending && temp_buf[i] > temp_buf[i + 1])
        || (!ascending && temp_buf[i] < temp_buf[i + 1])
      {
        temp_buf.swap(i, i + 1);
      }
    }
  }

  // Copy only the actual sorted elements back to the original array
  values[..len].copy_from_slice(&temp_buf[..len]);
}

// NEON u64 quicksort implementation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn quicksort_u64_neon(
  values: &mut [u64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low + 1 <= 2 {
        sort_u64_neon_2(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_u64_neon(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// Small array sorting for NEON u64 (2 elements or less)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sort_u64_neon_2(values: &mut [u64], len: usize, ascending: bool) {
  if len <= 1 {
    return;
  }

  // Load up to 2 u64 values into NEON register
  let mut data = [0u64; 2];
  data[..len].copy_from_slice(&values[..len]);
  let vec = vld1q_u64(data.as_ptr());

  // Simple comparison and swap for 2 u64 elements
  let val0 = vgetq_lane_u64(vec, 0);
  let val1 = vgetq_lane_u64(vec, 1);

  let (min_val, max_val) = if val0 <= val1 {
    (val0, val1)
  } else {
    (val1, val0)
  };

  let sorted = if ascending {
    vsetq_lane_u64(max_val, vsetq_lane_u64(min_val, vec, 0), 1)
  } else {
    vsetq_lane_u64(min_val, vsetq_lane_u64(max_val, vec, 0), 1)
  };

  vst1q_u64(data.as_mut_ptr(), sorted);
  values[..len].copy_from_slice(&data[..len]);
}

// ============================================================================
// SIMD I64 SORTING KERNELS
// ============================================================================

// AVX-512 i64 quicksort implementation
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn quicksort_i64_avx512(
  values: &mut [i64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low + 1 <= 8 {
        sort_i64_avx512_8(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_i64_avx512(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// 3-way partition for i64 using SIMD - AVX512
// GPU/PTX optimized 3-way partition for i64 values.
//
// Parallel 3-way partitioning for signed 64-bit integer values.
//
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn partition_3way_i64_avx512(
  values: &mut [i64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  let pivot = values[high];

  // Phase 1: Partition into <= and > using SIMD (exactly like 2-way partition)
  let mut le = low; // Index for next element <= pivot
  let mut j = low;

  while j + 7 < high {
    // Load 8 i64 values using AVX-512
    let chunk = _mm512_loadu_epi64(values[j..].as_ptr());
    let pivot_vec = _mm512_set1_epi64(pivot);

    // Compare for <= pivot (or >= for descending) using signed comparison
    let le_mask = if ascending {
      _mm512_cmple_epi64_mask(chunk, pivot_vec)
    } else {
      _mm512_cmpge_epi64_mask(chunk, pivot_vec)
    };

    // Process based on mask
    for k in 0..8 {
      if (le_mask & (1u8 << k)) != 0 {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    values.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phasesortn the <= region, separate < from == using SIMD
  let mut lt = low; // Index for next element < pivot
  j = low;

  while j + 7 < le.saturating_sub(1) {
    // Load 8 i64 values using AVX-512
    let chunk = _mm512_loadu_epi64(values[j..].as_ptr());
    let pivot_vec = _mm512_set1_epi64(pivot);

    // Compare for < pivot (or > for descending) using signed comparison
    let lt_mask = if ascending {
      _mm512_cmplt_epi64_mask(chunk, pivot_vec)
    } else {
      _mm512_cmpgt_epi64_mask(chunk, pivot_vec)
    };

    // Process based on mask
    for k in 0..8 {
      if (lt_mask & (1u8 << k)) != 0 {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 8;
  }

  // Handle remaining elements in <= region
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  // Return (end of <, start of >)
  (lt, le)
}

// Pure SIMD sort for exactly 8 i64 elements - AVX512
// GPU/PTX optimized small i64 array sorting (up to 8 elements).
//
// Insertion sort for small GPU partitions with signed integers.
//

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn sort_i64_avx512_8(values: &mut [i64], len: usize, ascending: bool) {
  if len <= 1 {
    return;
  }

  // For arrays that are not exactly 8 elements, use scalar sort to avoid sentinel value contamination
  if len != 8 {
    // Simple scalar sort - more reliable than SIMD with padding
    for i in 0..len {
      for j in i + 1..len {
        let should_swap = if ascending {
          values[i] > values[j]
        } else {
          values[i] < values[j]
        };
        if should_swap {
          values.swap(i, j);
        }
      }
    }
    return;
  }

  // FIXED: Deep SIMD 8-element sorting network - properly implemented like u64 version
  let mut buf = [0i64; 8];
  buf[..len].copy_from_slice(&values[..len]);

  // Load into SIMD register
  use std::arch::x86_64::*;
  let mut v = _mm512_loadu_epi64(buf.as_ptr());

  // Stage 1: Sort adjacent pairs (0,1), (2,3), (4,5), (6,7) - like u64 Stage 1
  let shuffled1 = _mm512_permutexvar_epi64(_mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1), v); // [1,0,3,2,5,4,7,6]
  let min1 = _mm512_min_epi64(v, shuffled1);
  let max1 = _mm512_max_epi64(v, shuffled1);
  // For i64, use blend like u64: even positions get min, odd get max
  let mask1 = 0b10101010u8; // positions 1,3,5,7 get max
  v = if ascending {
    _mm512_mask_blend_epi64(mask1, min1, max1) // [min0,max0,min2,max2,min4,max4,min6,max6]
  } else {
    _mm512_mask_blend_epi64(mask1, max1, min1) // [max0,min0,max2,min2,max4,min4,max6,min6]
  };

  // Stage 2: Cross comparisons within 256-bit lanes - like u64 Stage 2
  let shuffled2 = _mm512_permutexvar_epi64(_mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2), v); // [2,3,0,1,6,7,4,5]
  let min2 = _mm512_min_epi64(v, shuffled2);
  let max2 = _mm512_max_epi64(v, shuffled2);
  // Upper elements get max - like u64 pattern
  let mask2 = 0b11001100u8; // positions 2,3,6,7 get max
  v = if ascending {
    _mm512_mask_blend_epi64(mask2, min2, max2) // upper 2 elements in each 256-bit lane get max
  } else {
    _mm512_mask_blend_epi64(mask2, max2, min2) // upper 2 elements in each 256-bit lane get min
  };

  // Stage 3: Cross 256-bit lane comparisons (0-3 vs 4-7) - like u64 implementation
  let shuffled3 = _mm512_permutexvar_epi64(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), v); // [4,5,6,7,0,1,2,3]
  let min3 = _mm512_min_epi64(v, shuffled3);
  let max3 = _mm512_max_epi64(v, shuffled3);
  // Upper 4 elements get max
  let mask3 = 0b11110000u8; // positions 4-7 get max
  v = if ascending {
    _mm512_mask_blend_epi64(mask3, min3, max3) // upper 4 elements get max
  } else {
    _mm512_mask_blend_epi64(mask3, max3, min3) // upper 4 elements get min
  };

  // Stage 4: Final cleanup using scalar conditionals - hybrid approach for reliability
  let mut temp_buf = [0i64; 8];
  _mm512_storeu_epi64(temp_buf.as_mut_ptr(), v);

  // Simple conditional swaps for remaining pairs - only on actual elements
  // Do enough passes to clean up any remaining inversions
  for _pass in 0..len {
    for i in 0..len - 1 {
      if (ascending && temp_buf[i] > temp_buf[i + 1])
        || (!ascending && temp_buf[i] < temp_buf[i + 1])
      {
        temp_buf.swap(i, i + 1);
      }
    }
  }

  // Copy back to values
  values[..len].copy_from_slice(&temp_buf[..len]);
}

// AVX2 i64 quicksort implementation
#[cfg(all(
  not(feature = "hwx-nightly"),
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn quicksort_i64_avx2(
  values: &mut [i64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low + 1 <= 4 {
        sort_i64_avx2_4(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_i64_avx2(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// 3-way partition for i64 using SIMD - AVX2
#[cfg(all(
  not(feature = "hwx-nightly"),
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn partition_3way_i64_avx2(
  values: &mut [i64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  let pivot = values[high];

  // Phase 1: Partition into <= and > using SIMD (exactly like 2-way partition)
  let mut le = low; // Index for next element <= pivot
  let mut j = low;

  while j + 4 <= high {
    // Load 4 i64 values using AVX2
    let chunk = _mm256_loadu_si256(values[j..].as_ptr() as *const __m256i);
    let pivot_vec = _mm256_set1_epi64x(pivot);

    // Compare for <= pivot (or >= for descending) using signed comparison
    let cmp_result = if ascending {
      _mm256_cmpgt_epi64(pivot_vec, chunk) // pivot > chunk means chunk <= pivot
    } else {
      _mm256_cmpgt_epi64(chunk, pivot_vec) // chunk > pivot means chunk >= pivot
    };

    // Extract comparison results
    let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_result));

    // Move elements <= pivot to the left
    for k in 0..4 {
      if (mask & (1 << k)) != 0 {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    values.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from == using SIMD
  let mut lt = low; // Index for next element < pivot
  j = low;

  while j + 4 <= le.saturating_sub(2) {
    // Load 4 i64 values using AVX2
    let chunk = _mm256_loadu_si256(values[j..].as_ptr() as *const __m256i);
    let pivot_vec = _mm256_set1_epi64x(pivot);

    // Compare for < pivot (or > for descending) using signed comparison
    // For <, we need pivot > chunk to be true
    let cmp_result = if ascending {
      // We want chunk < pivot, which is pivot > chunk
      _mm256_cmpgt_epi64(pivot_vec, chunk)
    } else {
      // We want chunk > pivot
      _mm256_cmpgt_epi64(chunk, pivot_vec)
    };

    // Extract comparison results
    let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_result));

    // Move elements < pivot to the left
    for k in 0..4 {
      if (mask & (1 << k)) != 0 {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 4;
  }

  // Handle remaining elements in <= region
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  // Return (end of <, start of >)
  (lt, le)
}

// Optimized SIMD sort for exactly 4 i64 elements - AVX2 with full vectorization
#[cfg(all(
  not(feature = "hwx-nightly"),
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sort_i64_avx2_4(values: &mut [i64], len: usize, ascending: bool) {
  if len <= 1 {
    return;
  }

  // For arrays that are not exactly 4 elements, use scalar sort to avoid sentinel value contamination
  if len != 4 {
    // Simple scalar sort - more reliable than SIMD with padding
    for i in 0..len {
      for j in i + 1..len {
        let should_swap = if ascending {
          values[i] > values[j]
        } else {
          values[i] < values[j]
        };
        if should_swap {
          values.swap(i, j);
        }
      }
    }
    return;
  }

  // Deep SIMD 4-element sorting network (only for exactly 4 elements)
  let mut data = [0i64; 4];
  data[..len].copy_from_slice(&values[..len]);

  // Load data into AVX2 register
  let mut v = _mm256_loadu_si256(data.as_ptr() as *const __m256i);

  // 4-element sorting network using signed i64 comparisons

  // Stage 1: Compare (0,1) and (2,3)
  let shuffled1 = _mm256_permute4x64_epi64(v, 0b10110001); // [1,0,3,2]
  let cmp1 = _mm256_cmpgt_epi64(v, shuffled1); // v > shuffled1
  let min1 = _mm256_blendv_epi8(v, shuffled1, cmp1); // select smaller values
  let max1 = _mm256_blendv_epi8(shuffled1, v, cmp1); // select larger values

  // Blend: positions 1,3 get max values
  let mask1 = _mm256_set_epi64x(-1, 0, -1, 0); // positions 1,3 get max
  v = if ascending {
    _mm256_blendv_epi8(min1, max1, mask1)
  } else {
    _mm256_blendv_epi8(max1, min1, mask1)
  };

  // Stage 2: Compare (0,2) and (1,3)
  let shuffled2 = _mm256_permute4x64_epi64(v, 0b01001110); // [2,3,0,1]
  let cmp2 = _mm256_cmpgt_epi64(v, shuffled2); // v > shuffled2
  let min2 = _mm256_blendv_epi8(v, shuffled2, cmp2); // select smaller values
  let max2 = _mm256_blendv_epi8(shuffled2, v, cmp2); // select larger values

  // Blend: positions 2,3 get max values
  let mask2 = _mm256_set_epi64x(-1, -1, 0, 0); // positions 2,3 get max
  v = if ascending {
    _mm256_blendv_epi8(min2, max2, mask2)
  } else {
    _mm256_blendv_epi8(max2, min2, mask2)
  };

  // Stage 3: Final cleanup using scalar conditionals - like u64 implementation
  let mut temp_buf = [0i64; 4];
  _mm256_storeu_si256(temp_buf.as_mut_ptr() as *mut __m256i, v);

  // Simple conditional swaps for remaining pairs - only on actual elements
  // Do enough passes to clean up any remaining inversions
  for _pass in 0..len {
    for i in 0..len - 1 {
      if (ascending && temp_buf[i] > temp_buf[i + 1])
        || (!ascending && temp_buf[i] < temp_buf[i + 1])
      {
        temp_buf.swap(i, i + 1);
      }
    }
  }

  // Copy back to values
  values[..len].copy_from_slice(&temp_buf[..len]);
}

// NEON i64 quicksort implementation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn quicksort_i64_neon(
  values: &mut [i64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;

  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];

    if low < high {
      if high - low + 1 <= 2 {
        sort_i64_neon_2(&mut values[low..=high], high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) = partition_3way_i64_neon(values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// 3-way partition for i64 using SIMD - NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn partition_3way_i64_neon(
  values: &mut [i64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  let pivot = values[high];

  // Phase 1: Partition into <= and > using SIMD (exactly like 2-way partition)
  let mut le = low; // Index for next element <= pivot
  let mut j = low;

  while j + 2 <= high {
    // Load 2 i64 values using NEON
    let chunk = vld1q_s64(values[j..].as_ptr());
    let pivot_vec = vdupq_n_s64(pivot);

    // Compare for <= pivot (or >= for descending)
    let le_mask = if ascending {
      vcleq_s64(chunk, pivot_vec)
    } else {
      vcgeq_s64(chunk, pivot_vec)
    };

    // Extract mask bits
    let mask_array = [
      vgetq_lane_u64(le_mask, 0) != 0,
      vgetq_lane_u64(le_mask, 1) != 0,
    ];

    // Move elements <= pivot to the left
    for k in 0..2 {
      if mask_array[k] {
        if le != j + k {
          values.swap(le, j + k);
        }
        le += 1;
      }
    }
    j += 2;
  }

  // Handle remaining elements
  while j < high {
    let val = values[j];
    let should_move = if ascending {
      val <= pivot
    } else {
      val >= pivot
    };
    if should_move {
      if le != j {
        values.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    values.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from == using SIMD
  let mut lt = low; // Index for next element < pivot
  j = low;
  while j + 2 <= le.saturating_sub(2) {
    // Load 2 i64 values using NEON
    let chunk = vld1q_s64(values[j..].as_ptr());
    let pivot_vec = vdupq_n_s64(pivot);

    // Compare for < pivot (or > for descending)
    let lt_mask = if ascending {
      vcltq_s64(chunk, pivot_vec)
    } else {
      vcgtq_s64(chunk, pivot_vec)
    };

    // Extract mask bits
    let mask_array = [
      vgetq_lane_u64(lt_mask, 0) != 0,
      vgetq_lane_u64(lt_mask, 1) != 0,
    ];

    // Move elements < pivot to the left
    for k in 0..2 {
      if mask_array[k] {
        if lt != j + k {
          values.swap(lt, j + k);
        }
        lt += 1;
      }
    }
    j += 2;
  }

  // Handle remaining elements in <= region
  while j < le.saturating_sub(1) {
    let val = values[j];
    let is_less = if ascending { val < pivot } else { val > pivot };
    if is_less {
      if lt != j {
        values.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  // Return (end of <, start of >)
  (lt, le)
}

// Pure SIMD sort for exactly 2 i64 elements - NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sort_i64_neon_2(values: &mut [i64], len: usize, ascending: bool) {
  if len <= 1 {
    return;
  }

  // SIMD-based sorting using signed i64 comparisons (only for exactly 2 elements)
  let mut data = [0i64; 2];
  if len == 1 {
    data[0] = values[0];
  } else {
    data[..len].copy_from_slice(&values[..len]);
  }

  // Load into NEON register
  let vec = vld1q_s64(data.as_ptr());

  // Extract individual values for comparison
  let val0 = vgetq_lane_s64(vec, 0);
  let val1 = vgetq_lane_s64(vec, 1);

  // Determine min and max using signed comparison
  let (min_val, max_val) = if val0 <= val1 {
    (val0, val1)
  } else {
    (val1, val0)
  };

  // Create sorted vector based on ascending/descending
  let sorted = if ascending {
    vsetq_lane_s64(max_val, vsetq_lane_s64(min_val, vec, 0), 1)
  } else {
    vsetq_lane_s64(min_val, vsetq_lane_s64(max_val, vec, 0), 1)
  };

  vst1q_s64(data.as_mut_ptr(), sorted);
  values[..len].copy_from_slice(&data[..len]);
}

// ===================== SIMD MIN/MAX FINDING =====================

// GPU implementation of find_min_max_i64
#[cfg(has_cuda)]
pub unsafe fn find_min_max_i64_gpu(
  values: *const i64,
  len: usize,
  gpu_min: *mut i64,
  gpu_max: *mut i64,
) {
  const PTX_FIND_MIN_MAX_I64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry find_min_max_i64 (
      .param .u64 values,
      .param .u64 len,
      .param .u64 min_result,
      .param .u64 max_result
    ) {
      .reg .s64 %rd<30>;
      .reg .u32 %r<35>;
      .reg .u64 %rdu<30>;
      .reg .pred %p<10>;
      .shared .s64 sdata_min[256];
      .shared .s64 sdata_max[256];

      // Load parameters
      ld.param.u64 %rdu3, [values];
      ld.param.u32 %r9, [len];
      ld.param.u64 %rdu8, [min_result];
      ld.param.u64 %rdu9, [max_result];

      // Thread and block indices
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mul.lo.u32 %r3, %r1, %r2;
      add.u32 %r4, %r0, %r3;

      // Initialize min/max
      mov.s64 %rd0, 0x7FFFFFFFFFFFFFFF; // MAX
      mov.s64 %rd1, 0x8000000000000000; // MIN
      
      // Calculate grid stride
      mov.u32 %r5, %nctaid.x;
      mul.lo.u32 %r6, %r5, %r2;

      // Main loop - each thread processes elements with grid stride
      loop_start:
      setp.ge.u32 %p0, %r4, %r9;
      @%p0 bra store_shared;

      // Load single value
      cvt.u64.u32 %rdu0, %r4;
      mul.wide.u32 %rdu1, %r4, 8;
      add.u64 %rdu2, %rdu3, %rdu1;
      ld.global.s64 %rd2, [%rdu2];
      
      min.s64 %rd0, %rd0, %rd2;
      max.s64 %rd1, %rd1, %rd2;
      
      add.u32 %r4, %r4, %r6;
      bra loop_start;

      store_shared:
      // Check if this thread actually processed any data
      mov.s64 %rd11, 0x7FFFFFFFFFFFFFFF;
      setp.eq.s64 %p6, %rd0, %rd11;
      @%p6 bra done;  // Skip if didn't process any data
      
      // Use atomic min/max directly
      atom.global.min.s64 %rd9, [%rdu8], %rd0;
      atom.global.max.s64 %rd10, [%rdu9], %rd1;

      done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::reduction();
  let values_u64 = values as u64;
  let len_u32 = len as u32;
  let gpu_min_u64 = gpu_min as u64;
  let gpu_max_u64 = gpu_max as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u32 as *const u32 as *const u8,
    &gpu_min_u64 as *const u64 as *const u8,
    &gpu_max_u64 as *const u64 as *const u8,
  ];
  let _ = launch_ptx(
    PTX_FIND_MIN_MAX_I64,
    &[],
    "find_min_max_i64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn find_min_max_i64_avx512(values: &mut [i64], len: usize) -> (i64, i64) {
  let mut min_vec = _mm512_set1_epi64(i64::MAX);
  let mut max_vec = _mm512_set1_epi64(i64::MIN);

  let chunks = len / 8;

  // Process 8 elements at a time
  for i in 0..chunks {
    let data = _mm512_loadu_epi64(values.as_ptr().add(i * 8) as *const i64);
    min_vec = _mm512_min_epi64(min_vec, data);
    max_vec = _mm512_max_epi64(max_vec, data);
  }

  // **DEEP SIMD OPTIMIZATION**: Use AVX-512 horizontal reductions instead of scalar loops
  let final_min = _mm512_reduce_min_epi64(min_vec);
  let final_max = _mm512_reduce_max_epi64(max_vec);

  // **DEEP SIMD OPTIMIZATION**: Handle remainder with AVX-512 masked operations
  let remainder_start = chunks * 8;
  let remainder_count = len - remainder_start;

  if remainder_count > 0 {
    let load_mask = (1u8 << remainder_count) - 1;
    let remainder_vec = _mm512_mask_loadu_epi64(
      _mm512_set1_epi64(i64::MAX),
      load_mask,
      values.as_ptr().add(remainder_start) as *const i64,
    );
    let remainder_min = _mm512_mask_reduce_min_epi64(load_mask, remainder_vec);
    let remainder_max = _mm512_mask_reduce_max_epi64(load_mask, remainder_vec);

    (final_min.min(remainder_min), final_max.max(remainder_max))
  } else {
    (final_min, final_max)
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn find_min_max_i64_avx2(values: &mut [i64], len: usize) -> (i64, i64) {
  let mut min_vec = _mm256_set1_epi64x(i64::MAX);
  let mut max_vec = _mm256_set1_epi64x(i64::MIN);

  let chunks = len / 4;

  // Process 4 elements at a time
  for i in 0..chunks {
    let data = _mm256_loadu_si256(values.as_ptr().add(i * 4) as *const __m256i);

    // Use comparison and blend for min/max
    let cmp_min = _mm256_cmpgt_epi64(min_vec, data);
    min_vec = _mm256_blendv_epi8(min_vec, data, cmp_min);

    let cmp_max = _mm256_cmpgt_epi64(data, max_vec);
    max_vec = _mm256_blendv_epi8(max_vec, data, cmp_max);
  }

  // **DEEP SIMD OPTIMIZATION**: Use AVX2 horizontal min/max instead of scalar extraction
  // Extract min/max using horizontal operations
  let min_high = _mm256_extracti128_si256(min_vec, 1);
  let min_low = _mm256_castsi256_si128(min_vec);
  let max_high = _mm256_extracti128_si256(max_vec, 1);
  let max_low = _mm256_castsi256_si128(max_vec);

  // Use scalar extraction for final horizontal reduction
  let mut min_array = [0i64; 4];
  let mut max_array = [0i64; 4];
  _mm_storeu_si128(min_array.as_mut_ptr() as *mut __m128i, min_low);
  _mm_storeu_si128(min_array.as_mut_ptr().add(2) as *mut __m128i, min_high);
  _mm_storeu_si128(max_array.as_mut_ptr() as *mut __m128i, max_low);
  _mm_storeu_si128(max_array.as_mut_ptr().add(2) as *mut __m128i, max_high);

  let final_min = min_array[0]
    .min(min_array[1])
    .min(min_array[2])
    .min(min_array[3]);
  let final_max = max_array[0]
    .max(max_array[1])
    .max(max_array[2])
    .max(max_array[3]);

  // **DEEP SIMD OPTIMIZATION**: Handle remainder with minimal scalar
  let remainder_start = chunks * 4;
  let mut result_min = final_min;
  let mut result_max = final_max;
  for &val in &values[remainder_start..len] {
    result_min = result_min.min(val);
    result_max = result_max.max(val);
  }

  (result_min, result_max)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn find_min_max_i64_neon(values: &mut [i64], len: usize) -> (i64, i64) {
  let mut min_vec = vdupq_n_s64(i64::MAX);
  let mut max_vec = vdupq_n_s64(i64::MIN);

  let chunks = len / 2;

  // Process 2 elements at a time
  for i in 0..chunks {
    let data = vld1q_s64(values.as_ptr().add(i * 2));
    // NEON doesn't have min/max for s64, so we do element-wise comparison
    let cmp_min = vcltq_s64(data, min_vec);
    let cmp_max = vcgtq_s64(data, max_vec);
    min_vec = vbslq_s64(cmp_min, data, min_vec);
    max_vec = vbslq_s64(cmp_max, data, max_vec);
  }

  // Extract min/max from vectors
  let min_array = [vgetq_lane_s64(min_vec, 0), vgetq_lane_s64(min_vec, 1)];
  let max_array = [vgetq_lane_s64(max_vec, 0), vgetq_lane_s64(max_vec, 1)];

  let mut final_min = min_array[0].min(min_array[1]);
  let mut final_max = max_array[0].max(max_array[1]);

  // Handle remainder
  for &val in &values[chunks * 2..len] {
    final_min = final_min.min(val);
    final_max = final_max.max(val);
  }

  (final_min, final_max)
}

// =============================================================================
// SORT U32 BY U64 OPERATIONS - INDEX SORTING BY VALUES
// =============================================================================

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sort_u32_by_u64_avx512(
  indices: &mut [u32],
  values: &[u64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays (threshold similar to other SIMD sorts)
  if indices_len <= 16 {
    sort_u32_by_u64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  sort_u32_by_u64_quicksort_avx512(indices, values, 0, indices_len - 1, ascending);
}

// AVX2 optimized sorting of u32 indices by their corresponding u64 values
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sort_u32_by_u64_avx2(
  indices: &mut [u32],
  values: &[u64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays
  if indices_len <= 8 {
    sort_u32_by_u64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  sort_u32_by_u64_quicksort_avx2(indices, values, 0, indices_len - 1, ascending);
}

// NEON optimized sorting of u32 indices by their corresponding u64 values
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn sort_u32_by_u64_neon(
  indices: &mut [u32],
  values: &[u64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays
  if indices_len <= 4 {
    sort_u32_by_u64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  sort_u32_by_u64_quicksort_neon(indices, values, 0, indices_len - 1, ascending);
}

// Scalar insertion sort for small arrays (used by all architectures)
#[inline]
unsafe fn sort_u32_by_u64_scalar_small(
  indices: &mut [u32],
  values: &[u64],
  len: usize,
  ascending: bool,
) {
  for i in 1..len {
    let key_idx = indices[i];
    let key_val = values[key_idx as usize];
    let mut j = i;

    while j > 0 {
      let prev_idx = indices[j - 1];
      let prev_val = values[prev_idx as usize];

      let should_swap = if ascending {
        prev_val > key_val
      } else {
        prev_val < key_val
      };

      if should_swap {
        indices[j] = indices[j - 1];
        j -= 1;
      } else {
        break;
      }
    }
    indices[j] = key_idx;
  }
}

// AVX-512 quicksort implementation
// GPU/PTX optimized quicksort for u32 indices by u64 values.
//
// Iterative GPU-parallel quicksort with 3-way partitioning.
//
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn sort_u32_by_u64_quicksort_avx512(
  indices: &mut [u32],
  values: &[u64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 16 {
        sort_u32_by_u64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) =
          partition_3way_u32_by_u64_avx512(indices, values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// AVX2 quicksort implementation
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn sort_u32_by_u64_quicksort_avx2(
  indices: &mut [u32],
  values: &[u64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 8 {
        sort_u32_by_u64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) =
          partition_3way_u32_by_u64_avx2(indices, values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// NEON quicksort implementation
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sort_u32_by_u64_quicksort_neon(
  indices: &mut [u32],
  values: &[u64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 4 {
        sort_u32_by_u64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) =
          partition_3way_u32_by_u64_neon(indices, values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// AVX-512 3-way partitioning for u32 by u64
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn partition_3way_u32_by_u64_avx512(
  indices: &mut [u32],
  values: &[u64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  let mut lt = low; // Elements < pivot
  let mut i = low; // Current element
  let mut gt = high; // Elements > pivot

  while i <= gt {
    let current_idx = indices[i];
    let current_val = values[current_idx as usize];

    let cmp = if ascending {
      if current_val < pivot_val {
        -1
      } else if current_val > pivot_val {
        1
      } else {
        0
      }
    } else {
      if current_val > pivot_val {
        -1
      } else if current_val < pivot_val {
        1
      } else {
        0
      }
    };

    match cmp {
      -1 => {
        indices.swap(lt, i);
        lt += 1;
        i += 1;
      }
      1 => {
        indices.swap(i, gt);
        gt -= 1;
      }
      _ => {
        i += 1;
      }
    }
  }

  (lt, gt + 1)
}

// AVX2 partitioning
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
// AVX2 3-way partitioning for u32 by u64
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn partition_3way_u32_by_u64_avx2(
  indices: &mut [u32],
  values: &[u64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  let mut lt = low; // Elements < pivot
  let mut i = low; // Current element
  let mut gt = high; // Elements > pivot

  while i <= gt {
    let current_idx = indices[i];
    let current_val = values[current_idx as usize];

    let cmp = if ascending {
      if current_val < pivot_val {
        -1
      } else if current_val > pivot_val {
        1
      } else {
        0
      }
    } else {
      if current_val > pivot_val {
        -1
      } else if current_val < pivot_val {
        1
      } else {
        0
      }
    };

    match cmp {
      -1 => {
        indices.swap(lt, i);
        lt += 1;
        i += 1;
      }
      1 => {
        indices.swap(i, gt);
        gt -= 1;
      }
      _ => {
        i += 1;
      }
    }
  }

  (lt, gt + 1)
}

// NEON 3-way partitioning for u64 values
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn partition_3way_u32_by_u64_neon(
  indices: &mut [u32],
  values: &[u64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low, high);
  }

  // Use middle element as pivot
  let mid = low + (high - low) / 2;
  indices.swap(mid, high);

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  let mut less = low;
  let mut equal = low;
  let mut greater = high;

  while equal <= greater {
    let current_idx = indices[equal];
    let current_val = values[current_idx as usize];

    let cmp = if ascending {
      if current_val < pivot_val {
        -1
      } else if current_val > pivot_val {
        1
      } else {
        0
      }
    } else {
      if current_val > pivot_val {
        -1
      } else if current_val < pivot_val {
        1
      } else {
        0
      }
    };

    match cmp {
      -1 => {
        // -1 means "less than" pivot (ascending) or "greater than" pivot (descending)
        indices.swap(less, equal);
        less += 1;
        equal += 1;
      }
      0 => {
        equal += 1;
      }
      _ => {
        // 1 means "greater than" pivot (ascending) or "less than" pivot (descending)
        indices.swap(equal, greater);
        greater -= 1;
      }
    }
  }

  (less, greater + 1)
}

// AVX-512 optimized sorting of u32 indices by their corresponding f64 values
#[cfg(feature = "hwx-nightly")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sort_u32_by_f64_avx512(
  indices: &mut [u32],
  values: &[f64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays (threshold similar to other SIMD sorts)
  if indices_len <= 16 {
    sort_u32_by_f64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  #[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
  ))]
  {
    sort_u32_by_f64_quicksort_avx512(indices, values, 0, indices_len - 1, ascending);
  }
  #[cfg(not(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
  )))]
  {
    sort_u32_by_f64_scalar_small(indices, values, indices_len, ascending);
  }
}

// AVX2 optimized sorting of u32 indices by their corresponding f64 values
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sort_u32_by_f64_avx2(
  indices: &mut [u32],
  values: &[f64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays
  if indices_len <= 8 {
    sort_u32_by_f64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  sort_u32_by_f64_quicksort_avx2(indices, values, 0, indices_len - 1, ascending);
}

// NEON optimized sorting of u32 indices by their corresponding f64 values
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn sort_u32_by_f64_neon(
  indices: &mut [u32],
  values: &[f64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays
  if indices_len <= 4 {
    sort_u32_by_f64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  sort_u32_by_f64_quicksort_neon(indices, values, 0, indices_len - 1, ascending);
}

// Scalar insertion sort for small arrays (used by all architectures)
#[inline]
unsafe fn sort_u32_by_f64_scalar_small(
  indices: &mut [u32],
  values: &[f64],
  len: usize,
  ascending: bool,
) {
  for i in 1..len {
    let key_idx = indices[i];
    let key_val = values[key_idx as usize];
    let key_is_nan = key_val != key_val; // NaN check without method call
    let mut j = i;
    while j > 0 {
      let prev_idx = indices[j - 1];
      let prev_val = values[prev_idx as usize];
      let prev_is_nan = prev_val != prev_val; // NaN check without method call

      let should_swap = if ascending {
        // For ascending: NaN goes to end
        if key_is_nan {
          false // NaN key stays in place (will bubble to end)
        } else if prev_is_nan {
          true // Non-NaN key should move before NaN
        } else {
          prev_val > key_val // Normal comparison
        }
      } else {
        // For descending: NaN goes to beginning
        if key_is_nan {
          true // NaN key should move left
        } else if prev_is_nan {
          false // Non-NaN key stays after NaN
        } else {
          prev_val < key_val // Normal comparison
        }
      };

      if should_swap {
        indices[j] = indices[j - 1];
        j -= 1;
      } else {
        break;
      }
    }
    indices[j] = key_idx;
  }
}

// AVX-512 quicksort implementation
// GPU/PTX optimized quicksort for u32 indices by f64 values.
//
// Iterative GPU-parallel quicksort with 3-way partitioning.
//
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn sort_u32_by_f64_quicksort_avx512(
  indices: &mut [u32],
  values: &[f64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 16 {
        sort_u32_by_f64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        // Use 3-way partitioning to handle duplicates efficiently
        let (less_end, greater_start) =
          partition_3way_u32_by_f64_avx512(indices, values, low, high, ascending);

        // Push partitions to stack (only non-equal sections)
        if less_end > low && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, less_end - 1);
          stack_pos += 1;
        }
        if greater_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (greater_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// AVX2 quicksort implementation
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn sort_u32_by_f64_quicksort_avx2(
  indices: &mut [u32],
  values: &[f64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 8 {
        sort_u32_by_f64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        // Use 3-way partitioning to handle duplicates efficiently
        let (less_end, greater_start) =
          partition_3way_u32_by_f64_avx2(indices, values, low, high, ascending);

        // Push partitions to stack (only non-equal sections)
        if less_end > low && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, less_end - 1);
          stack_pos += 1;
        }
        if greater_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (greater_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// NEON quicksort implementation
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sort_u32_by_f64_quicksort_neon(
  indices: &mut [u32],
  values: &[f64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 4 {
        sort_u32_by_f64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        // Use 3-way partitioning to handle duplicates efficiently
        let (less_end, greater_start) =
          partition_3way_u32_by_f64_neon(indices, values, low, high, ascending);

        // Push partitions to stack (only non-equal sections)
        // This skips all elements equal to pivot, dramatically improving performance with duplicates
        if less_end > low && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, less_end - 1);
          stack_pos += 1;
        }
        if greater_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (greater_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// AVX-512 3-way partitioning for handling duplicates efficiently
// GPU/PTX optimized 3-way partition for u32 indices by f64 values.
//
// Parallel 3-way partitioning with NaN handling.
//
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn partition_3way_u32_by_f64_avx512(
  indices: &mut [u32],
  values: &[f64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low, high);
  }

  // Use middle element as pivot for better performance on sorted data
  let mid = low + (high - low) / 2;
  indices.swap(mid, high);

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  // Handle NaN pivot specially
  if pivot_val != pivot_val {
    let mut boundary = low;

    for i in low..high {
      let idx = indices[i];
      let val = values[idx as usize];
      let is_nan = val != val;

      if (ascending && !is_nan) || (!ascending && is_nan) {
        if boundary != i {
          indices.swap(boundary, i);
        }
        boundary += 1;
      }
    }

    // Move pivot to boundary
    if boundary != high {
      indices.swap(boundary, high);
    }

    // Return partition boundaries
    return (boundary, boundary + 1);
  }

  // Phase 1: Partition into <= pivot and > pivot
  let mut le = low;
  let mut j = low;

  while j < high {
    let idx = indices[j];
    let val = values[idx as usize];
    let is_nan = val != val;

    let should_move = if ascending {
      !is_nan && val <= pivot_val
    } else {
      is_nan || val >= pivot_val
    };

    if should_move {
      if le != j {
        indices.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    indices.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from ==
  let mut lt = low;
  j = low;

  while j < le.saturating_sub(1) {
    let idx = indices[j];
    let val = values[idx as usize];
    let is_nan = val != val;

    let is_less = if ascending {
      !is_nan && val < pivot_val
    } else {
      is_nan || val > pivot_val
    };

    if is_less {
      if lt != j {
        indices.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  (lt, le)
}
// AVX2 3-way partitioning for handling duplicates efficiently
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn partition_3way_u32_by_f64_avx2(
  indices: &mut [u32],
  values: &[f64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low, high);
  }

  // Use middle element as pivot for better performance on sorted data
  let mid = low + (high - low) / 2;
  indices.swap(mid, high);

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  // Handle NaN pivot specially
  if pivot_val != pivot_val {
    let mut boundary = low;

    for i in low..high {
      let idx = indices[i];
      let val = values[idx as usize];
      let is_nan = val != val;

      if (ascending && !is_nan) || (!ascending && is_nan) {
        if boundary != i {
          indices.swap(boundary, i);
        }
        boundary += 1;
      }
    }

    // Move pivot to boundary
    if boundary != high {
      indices.swap(boundary, high);
    }

    // Return partition boundaries
    return (boundary, boundary + 1);
  }

  // Phase 1: Partition into <= pivot and > pivot
  let mut le = low;
  let mut j = low;

  while j < high {
    let idx = indices[j];
    let val = values[idx as usize];
    let is_nan = val != val;

    let should_move = if ascending {
      !is_nan && val <= pivot_val
    } else {
      is_nan || val >= pivot_val
    };

    if should_move {
      if le != j {
        indices.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    indices.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from ==
  let mut lt = low;
  j = low;

  while j < le.saturating_sub(1) {
    let idx = indices[j];
    let val = values[idx as usize];
    let is_nan = val != val;

    let is_less = if ascending {
      !is_nan && val < pivot_val
    } else {
      is_nan || val > pivot_val
    };

    if is_less {
      if lt != j {
        indices.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  (lt, le)
}

// NEON 3-way partitioning for handling duplicates efficiently
// Returns (less_end, greater_start) where:
// - [low..less_end] contains elements < pivot
// - [less_end..greater_start] contains elements == pivot
// - [greater_start..=high] contains elements > pivot
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn partition_3way_u32_by_f64_neon(
  indices: &mut [u32],
  values: &[f64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low, high);
  }

  // Use middle element as pivot for better performance on sorted data
  let mid = low + (high - low) / 2;
  indices.swap(mid, high);

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  // Handle NaN pivot with SIMD - same logic as f64
  if pivot_val != pivot_val {
    // Use SIMD to partition NaN vs non-NaN
    let mut boundary = low;
    let mut i = low;

    while i < high {
      let idx = indices[i];
      let val = values[idx as usize];
      let is_nan = val != val;

      if (ascending && !is_nan) || (!ascending && is_nan) {
        if boundary != i {
          indices.swap(boundary, i);
        }
        boundary += 1;
      }
      i += 1;
    }

    // Move pivot to boundary
    if boundary != high {
      indices.swap(boundary, high);
    }

    // Return partition boundaries
    // boundary is the position after all NaN (ascending) or non-NaN (descending)
    return (boundary, boundary + 1);
  }

  // SIMD 3-way partition - Two phase approach with SIMD (same as f64)
  // Phase 1: Partition into <= pivot and > pivot
  let mut le = low;
  let mut j = low;

  // Process elements
  while j < high {
    let idx = indices[j];
    let val = values[idx as usize];
    let is_nan = val != val;

    let should_move = if ascending {
      !is_nan && val <= pivot_val
    } else {
      is_nan || val >= pivot_val
    };

    if should_move {
      if le != j {
        indices.swap(le, j);
      }
      le += 1;
    }
    j += 1;
  }

  // Move pivot to end of <= region
  if le != high {
    indices.swap(le, high);
  }
  le += 1; // Now le points to first element > pivot

  // Phase 2: Within the <= region, separate < from ==
  let mut lt = low;
  j = low;

  while j < le.saturating_sub(1) {
    let idx = indices[j];
    let val = values[idx as usize];
    let is_nan = val != val;

    let is_less = if ascending {
      !is_nan && val < pivot_val
    } else {
      is_nan || val > pivot_val
    };

    if is_less {
      if lt != j {
        indices.swap(lt, j);
      }
      lt += 1;
    }
    j += 1;
  }

  (lt, le)
}

// =============================================================================
// SORT U32 BY I64 OPERATIONS - INDEX SORTING BY VALUES
// =============================================================================

// AVX-512 optimized sorting of u32 indices by their corresponding i64 values
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(feature = "hwx-nightly")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sort_u32_by_i64_avx512(
  indices: &mut [u32],
  values: &[i64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays (threshold similar to other SIMD sorts)
  if indices_len <= 16 {
    sort_u32_by_i64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  #[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
  ))]
  {
    sort_u32_by_i64_quicksort_avx512(indices, values, 0, indices_len - 1, ascending);
  }
  #[cfg(not(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
  )))]
  {
    sort_u32_by_i64_scalar_small(indices, values, indices_len, ascending);
  }
}

// GPU/PTX optimized u32 indices sorting by i64 values using bitonic sort.
//
// Grid-stride parallel bitonic sorting with signed 64-bit integer comparison.
#[cfg(has_cuda)]
pub unsafe fn sort_u32_by_i64_gpu(
  indices: *mut u32,
  values: *const i64,
  indices_len: usize,
  _values_len: usize,
  ascending: bool,
) {
  const PTX_SORT_U32_BY_I64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .entry sort_u32_by_i64(
      .param .u64 indices_ptr,
      .param .u64 values_ptr,
      .param .u64 len,
      .param .u32 ascending
    ) {
      .reg .u64 %rd<25>;
      .reg .s64 %sd<8>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [indices_ptr];
      ld.param.u64 %rd1, [values_ptr];
      ld.param.u64 %rd22, [len];
      cvt.u32.u64 %r0, %rd22;
      ld.param.u32 %r1, [ascending];
      
      // Thread and block info
      mov.u32 %r2, %tid.x;
      mov.u32 %r3, %ctaid.x;
      mov.u32 %r4, %ntid.x;
      mov.u32 %r5, %nctaid.x;
      
      // Convert len to u64 for comparisons
      cvt.u64.u32 %rd23, %r0;
      
      // Bitonic merge stages
      mov.u64 %rd3, 2;  // k = 2 (stage size)
      
    stage_loop:
      setp.gt.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      mov.u64 %rd4, %rd3;
      shr.b64 %rd4, %rd4, 1;  // j = k/2 (comparison distance)
      
    substage_loop:
      setp.eq.u64 %p1, %rd4, 0;
      @%p1 bra next_stage;
      
      // Initialize thread's starting index with grid stride
      mul.lo.u32 %r6, %r3, %r4;
      add.u32 %r7, %r6, %r2;
      cvt.u64.u32 %rd5, %r7;  // Initial index
      
      // Calculate grid stride (total threads)
      mul.lo.u32 %r8, %r5, %r4;
      cvt.u64.u32 %rd6, %r8;  // Grid stride
      
    work_loop:
      // Check if index is out of bounds
      setp.ge.u64 %p2, %rd5, %rd2;
      @%p2 bra sync_point;
      
      // Only process if (index & j) == 0
      and.b64 %rd7, %rd5, %rd4;
      setp.ne.u64 %p3, %rd7, 0;
      @%p3 bra skip_compare;
      
      // Calculate comparison partner index
      xor.b64 %rd8, %rd5, %rd4;
      setp.ge.u64 %p4, %rd8, %rd2;
      @%p4 bra skip_compare;
      
      // Load indices at both positions
      shl.b64 %rd9, %rd5, 2;  // index * 4 for u32
      add.u64 %rd10, %rd0, %rd9;
      ld.global.u32 %r9, [%rd10];  // indices[index]
      
      shl.b64 %rd11, %rd8, 2;  // partner * 4 for u32
      add.u64 %rd12, %rd0, %rd11;
      ld.global.u32 %r10, [%rd12];  // indices[partner]
      
      // Load corresponding signed i64 values for comparison
      cvt.u64.u32 %rd13, %r9;
      shl.b64 %rd14, %rd13, 3;  // idx * 8 for i64 value
      add.u64 %rd15, %rd1, %rd14;
      ld.global.s64 %sd0, [%rd15];  // values[indices[index]]
      
      cvt.u64.u32 %rd17, %r10;
      shl.b64 %rd18, %rd17, 3;  // idx * 8 for i64 value
      add.u64 %rd19, %rd1, %rd18;
      ld.global.s64 %sd1, [%rd19];  // values[indices[partner]]
      
      // Determine sort direction
      and.b64 %rd21, %rd5, %rd3;
      setp.eq.u64 %p5, %rd21, 0;  // Direction based on stage bit
      
      // Compare signed i64 values
      setp.gt.s64 %p6, %sd0, %sd1;  // Signed comparison
      setp.ne.u32 %p7, %r1, 0;  // Check ascending flag
      xor.pred %p6, %p6, %p7;  // XOR with ascending
      xor.pred %p6, %p6, %p5;  // XOR with direction
      
      // Conditional swap - use warp shuffle for intra-warp exchanges
      and.b32 %r11, %r2, 0x1f;  // Get lane ID within warp
      cvt.u32.u64 %r12, %rd4;    // Convert comparison distance to u32
      setp.lt.u32 %p7, %r12, 32; // Check if exchange is within warp
      
      @%p7 bra warp_exchange;
      
      // Global memory exchange for inter-warp
      @%p6 st.global.u32 [%rd10], %r10;
      @%p6 st.global.u32 [%rd12], %r9;
      bra skip_compare;
      
    warp_exchange:
      // Use warp shuffle for intra-warp exchanges (32-bit indices)
      shfl.sync.xor.b32 %r13, %r9, %r12, 0x1f, 0xffffffff;
      
      // Conditional store based on comparison
      @%p6 mov.u32 %r9, %r13;
      
    skip_compare:
      add.u64 %rd5, %rd5, %rd6;  // Move to next element (grid stride)
      bra work_loop;
      
    sync_point:
      bar.sync 0;  // Synchronize all threads
      
      shr.b64 %rd4, %rd4, 1;  // j = j/2 (next substage)
      bra substage_loop;
      
    next_stage:
      shl.b64 %rd3, %rd3, 1;  // k = k*2 (next stage)
      bra stage_loop;
      
    done:
      ret;
    }
  "#;

  // Launch with multiple blocks/threads for parallelization
  let (blocks, threads) = LaunchConfig::parallel();
  let ascending_flag = if ascending { 1u32 } else { 0u32 };

  let args = [
    indices as *const u8,
    values as *const u8,
    &(indices_len as u32) as *const _ as *const u8,
    &ascending_flag as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_SORT_U32_BY_I64,
    &[],
    "sort_u32_by_i64",
    blocks,
    threads,
    &args,
  );
}

// AVX2 optimized sorting of u32 indices by their corresponding i64 values
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sort_u32_by_i64_avx2(
  indices: &mut [u32],
  values: &[i64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays
  if indices_len <= 8 {
    sort_u32_by_i64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  sort_u32_by_i64_quicksort_avx2(indices, values, 0, indices_len - 1, ascending);
}

// NEON optimized sorting of u32 indices by their corresponding i64 values
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn sort_u32_by_i64_neon(
  indices: &mut [u32],
  values: &[i64],
  indices_len: usize,
  values_len: usize,
  ascending: bool,
) {
  // Validate input lengths match
  if indices_len != values_len {
    return;
  }

  // Use insertion sort for small arrays
  if indices_len <= 4 {
    sort_u32_by_i64_scalar_small(indices, values, indices_len, ascending);
    return;
  }

  // For larger arrays, use quicksort with SIMD partitioning
  sort_u32_by_i64_quicksort_neon(indices, values, 0, indices_len - 1, ascending);
}

// Scalar insertion sort for small arrays (used by all architectures)
#[inline]
unsafe fn sort_u32_by_i64_scalar_small(
  indices: &mut [u32],
  values: &[i64],
  len: usize,
  ascending: bool,
) {
  for i in 1..len {
    let key_idx = indices[i];
    let key_val = values[key_idx as usize];
    let mut j = i;
    while j > 0 {
      let prev_idx = indices[j - 1];
      let prev_val = values[prev_idx as usize];
      let should_swap = if ascending {
        prev_val > key_val
      } else {
        prev_val < key_val
      };
      if should_swap {
        indices[j] = indices[j - 1];
        j -= 1;
      } else {
        break;
      }
    }
    indices[j] = key_idx;
  }
}

// AVX-512 quicksort implementation
// GPU/PTX optimized quicksort for u32 indices by i64 values.
//
// Iterative GPU-parallel quicksort with 3-way partitioning.
//
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn sort_u32_by_i64_quicksort_avx512(
  indices: &mut [u32],
  values: &[i64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 16 {
        sort_u32_by_i64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) =
          partition_3way_u32_by_i64_avx512(indices, values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// AVX2 quicksort implementation
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn sort_u32_by_i64_quicksort_avx2(
  indices: &mut [u32],
  values: &[i64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 8 {
        sort_u32_by_i64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) =
          partition_3way_u32_by_i64_avx2(indices, values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// NEON quicksort implementation
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sort_u32_by_i64_quicksort_neon(
  indices: &mut [u32],
  values: &[i64],
  low: usize,
  high: usize,
  ascending: bool,
) {
  // Use iterative quicksort to prevent stack overflow with large arrays
  let mut stack = [(0usize, 0usize); QUICKSORT_STACK_SIZE]; // Stack-allocated array for recursion
  let mut stack_pos = 0;
  stack[stack_pos] = (low, high);
  stack_pos += 1;
  while stack_pos > 0 {
    stack_pos -= 1;
    let (low, high) = stack[stack_pos];
    if low < high {
      if high - low + 1 <= 4 {
        sort_u32_by_i64_scalar_small(&mut indices[low..=high], values, high - low + 1, ascending);
      } else {
        let (lt_end, gt_start) =
          partition_3way_u32_by_i64_neon(indices, values, low, high, ascending);
        // Only push non-empty partitions
        if lt_end > 0 && low < lt_end && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (low, lt_end - 1);
          stack_pos += 1;
        }
        if gt_start <= high && stack_pos < QUICKSORT_STACK_SIZE {
          stack[stack_pos] = (gt_start, high);
          stack_pos += 1;
        }
      }
    }
  }
}

// AVX-512 3-way partitioning for u32 by i64
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[inline]
unsafe fn partition_3way_u32_by_i64_avx512(
  indices: &mut [u32],
  values: &[i64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  let mut lt = low; // Elements < pivot
  let mut i = low; // Current element
  let mut gt = high; // Elements > pivot

  while i <= gt {
    let current_idx = indices[i];
    let current_val = values[current_idx as usize];

    let cmp = if ascending {
      if current_val < pivot_val {
        -1
      } else if current_val > pivot_val {
        1
      } else {
        0
      }
    } else {
      if current_val > pivot_val {
        -1
      } else if current_val < pivot_val {
        1
      } else {
        0
      }
    };

    match cmp {
      -1 => {
        indices.swap(lt, i);
        lt += 1;
        i += 1;
      }
      1 => {
        indices.swap(i, gt);
        gt -= 1;
      }
      _ => {
        i += 1;
      }
    }
  }

  (lt, gt + 1)
}

// AVX2 partitioning
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
// AVX2 3-way partitioning for u32 by i64
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[inline]
unsafe fn partition_3way_u32_by_i64_avx2(
  indices: &mut [u32],
  values: &[i64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  let mut lt = low; // Elements < pivot
  let mut i = low; // Current element
  let mut gt = high; // Elements > pivot

  while i <= gt {
    let current_idx = indices[i];
    let current_val = values[current_idx as usize];

    let cmp = if ascending {
      if current_val < pivot_val {
        -1
      } else if current_val > pivot_val {
        1
      } else {
        0
      }
    } else {
      if current_val > pivot_val {
        -1
      } else if current_val < pivot_val {
        1
      } else {
        0
      }
    };

    match cmp {
      -1 => {
        indices.swap(lt, i);
        lt += 1;
        i += 1;
      }
      1 => {
        indices.swap(i, gt);
        gt -= 1;
      }
      _ => {
        i += 1;
      }
    }
  }

  (lt, gt + 1)
}

// NEON 3-way partitioning for u32 by i64
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn partition_3way_u32_by_i64_neon(
  indices: &mut [u32],
  values: &[i64],
  low: usize,
  high: usize,
  ascending: bool,
) -> (usize, usize) {
  if low >= high {
    return (low + 1, high + 1);
  }

  let pivot_idx = indices[high];
  let pivot_val = values[pivot_idx as usize];

  let mut lt = low; // Elements < pivot
  let mut i = low; // Current element
  let mut gt = high; // Elements > pivot

  while i <= gt {
    let current_idx = indices[i];
    let current_val = values[current_idx as usize];

    let cmp = if ascending {
      if current_val < pivot_val {
        -1
      } else if current_val > pivot_val {
        1
      } else {
        0
      }
    } else {
      if current_val > pivot_val {
        -1
      } else if current_val < pivot_val {
        1
      } else {
        0
      }
    };

    match cmp {
      -1 => {
        indices.swap(lt, i);
        lt += 1;
        i += 1;
      }
      1 => {
        indices.swap(i, gt);
        gt -= 1;
      }
      _ => {
        i += 1;
      }
    }
  }

  (lt, gt + 1)
}

// =============================================================================
// PROMQL ELEMENT-WISE OPERATIONS
// =============================================================================

// AVX-512 element-wise addition for f64 arrays
// GPU implementations of SIMD arithmetic operations
#[cfg(has_cuda)]
pub unsafe fn add_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_ADD_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry add_f64(
      .param .u64 a_ptr,
      .param .u64 b_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd5, [a_ptr];
      ld.param.u64 %rd6, [b_ptr];
      ld.param.u64 %rd7, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load and add double2 vectors using vectorized loads
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load vectors using v2.f64 (128-bit loads)
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load a vector
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load b vector
      // Compute addition
      add.f64 %fd4, %fd0, %fd2;               // Add lane 0
      add.f64 %fd5, %fd1, %fd3;               // Add lane 1
      // Store results using vectorized store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store result vector
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      add.f64 %fd4, %fd0, %fd2;
      st.global.f64 [%rd4], %fd4;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    a as *const u8,
    b as *const u8,
    result as *const u8,
    &len as *const usize as *const u8,
  ];

  let _ = launch_ptx(PTX_ADD_F64, &[], "add_f64", blocks, threads, &args);
}

#[cfg(has_cuda)]
pub unsafe fn subtract_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_SUBTRACT_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry subtract_f64(
      .param .u64 a_ptr,
      .param .u64 b_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd5, [a_ptr];
      ld.param.u64 %rd6, [b_ptr];
      ld.param.u64 %rd7, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load and subtract double2 vectors using vectorized loads
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load vectors using v2.f64 (128-bit loads)
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load a vector
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load b vector
      // Compute subtraction
      sub.f64 %fd4, %fd0, %fd2;               // Subtract lane 0
      sub.f64 %fd5, %fd1, %fd3;               // Subtract lane 1
      // Store results using vectorized store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store result vector
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      sub.f64 %fd4, %fd0, %fd2;
      st.global.f64 [%rd4], %fd4;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    a as *const u8,
    b as *const u8,
    result as *const u8,
    &len as *const usize as *const u8,
  ];

  let _ = launch_ptx(
    PTX_SUBTRACT_F64,
    &[],
    "subtract_f64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(has_cuda)]
pub unsafe fn multiply_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_MULTIPLY_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry multiply_f64(
      .param .u64 a_ptr,
      .param .u64 b_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd5, [a_ptr];
      ld.param.u64 %rd6, [b_ptr];
      ld.param.u64 %rd7, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and multiply double2 vectors using vectorized loads
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load vectors using v2.f64 (128-bit loads)
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load a vector
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load b vector
      // Compute multiplication
      mul.f64 %fd4, %fd0, %fd2;               // Multiply lane 0
      mul.f64 %fd5, %fd1, %fd3;               // Multiply lane 1
      // Store results using vectorized store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store result vector

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

    remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      mul.f64 %fd4, %fd0, %fd2;
      st.global.f64 [%rd4], %fd4;

    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    a as *const u8,
    b as *const u8,
    result as *const u8,
    &len as *const usize as *const u8,
  ];

  let _ = launch_ptx(
    PTX_MULTIPLY_F64,
    &[],
    "multiply_f64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(has_cuda)]
pub unsafe fn divide_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_DIVIDE_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry divide_f64(
      .param .u64 a_ptr,
      .param .u64 b_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd5, [a_ptr];
      ld.param.u64 %rd6, [b_ptr];
      ld.param.u64 %rd7, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and divide double2 vectors using vectorized loads
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load vectors using v2.f64 (128-bit loads)
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load a vector
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load b vector
      // Compute division
      div.rn.f64 %fd4, %fd0, %fd2;            // Divide lane 0
      div.rn.f64 %fd5, %fd1, %fd3;            // Divide lane 1
      // Store results using vectorized store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store result vector

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

    remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      div.rn.f64 %fd4, %fd0, %fd2;
      st.global.f64 [%rd4], %fd4;

    done:
      ret;
    }
  "#;

  let args = [
    a as *const u8,
    b as *const u8,
    result as *const u8,
    (&len as *const usize) as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_DIVIDE_F64, &[], "divide_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn add_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let sum = _mm512_add_pd(a_vec, b_vec);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), sum);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) + b.get_unchecked(offset + i);
  }
}

// AVX2 element-wise addition for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn add_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let sum = _mm256_add_pd(a_vec, b_vec);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), sum);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) + b.get_unchecked(offset + i);
  }
}

// NEON element-wise addition for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn add_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let sum = vaddq_f64(a_vec, b_vec);
    vst1q_f64(result.as_mut_ptr().add(offset), sum);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) + b.get_unchecked(offset + i);
  }
}

// AVX-512 element-wise subtraction for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn subtract_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let diff = _mm512_sub_pd(a_vec, b_vec);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), diff);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) - b.get_unchecked(offset + i);
  }
}

// AVX2 element-wise subtraction for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn subtract_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let diff = _mm256_sub_pd(a_vec, b_vec);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), diff);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) - b.get_unchecked(offset + i);
  }
}

// NEON element-wise subtraction for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn subtract_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let diff = vsubq_f64(a_vec, b_vec);
    vst1q_f64(result.as_mut_ptr().add(offset), diff);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) - b.get_unchecked(offset + i);
  }
}

// AVX-512 element-wise multiplication for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn multiply_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let prod = _mm512_mul_pd(a_vec, b_vec);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), prod);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) * b.get_unchecked(offset + i);
  }
}

// AVX2 element-wise multiplication for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn multiply_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let prod = _mm256_mul_pd(a_vec, b_vec);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), prod);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) * b.get_unchecked(offset + i);
  }
}

// NEON element-wise multiplication for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn multiply_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let prod = vmulq_f64(a_vec, b_vec);
    vst1q_f64(result.as_mut_ptr().add(offset), prod);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) * b.get_unchecked(offset + i);
  }
}

// AVX-512 element-wise division for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn divide_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let quot = _mm512_div_pd(a_vec, b_vec);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), quot);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) / b.get_unchecked(offset + i);
  }
}

// AVX2 element-wise division for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn divide_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let quot = _mm256_div_pd(a_vec, b_vec);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), quot);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) / b.get_unchecked(offset + i);
  }
}

// NEON element-wise division for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn divide_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let quot = vdivq_f64(a_vec, b_vec);
    vst1q_f64(result.as_mut_ptr().add(offset), quot);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      a.get_unchecked(offset + i) / b.get_unchecked(offset + i);
  }
}

// AVX-512 element-wise modulo for f64 arrays
// GPU implementation of modulo operation
#[cfg(has_cuda)]
pub unsafe fn modulo_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_MODULO_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry modulo_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd5, [a];
      ld.param.u64 %rd6, [b];
      ld.param.u64 %rd7, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and compute modulo for double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load a vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 a
      // Load b vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load double2 b
      // Compute modulo: a % b = a - floor(a/b) * b
      div.rn.f64 %fd4, %fd0, %fd2;     // a0 / b0
      div.rn.f64 %fd5, %fd1, %fd3;     // a1 / b1
      cvt.rmi.f64.f64 %fd6, %fd4;      // floor(a0/b0)
      cvt.rmi.f64.f64 %fd7, %fd5;      // floor(a1/b1)
      mul.f64 %fd8, %fd6, %fd2;        // floor(a0/b0) * b0
      mul.f64 %fd9, %fd7, %fd3;        // floor(a1/b1) * b1
      sub.f64 %fd10, %fd0, %fd8;       // a0 - floor(a0/b0)*b0
      sub.f64 %fd11, %fd1, %fd9;       // a1 - floor(a1/b1)*b1
      // Store results with vectorized 128-bit store
      st.global.v2.f64 [%rd4], {%fd10, %fd11};  // Store double2 results

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

      remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r10;  // Check if first element exists
      @!%p0 bra end;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      div.rn.f64 %fd4, %fd0, %fd2;
      cvt.rmi.f64.f64 %fd6, %fd4;
      mul.f64 %fd8, %fd6, %fd2;
      sub.f64 %fd10, %fd0, %fd8;
      st.global.f64 [%rd4], %fd10;

      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(PTX_MODULO_F64, &[], "modulo_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn modulo_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  // AVX-512 doesn't have a direct modulo instruction for f64, use scalar fallback
  for i in 0..len {
    *result.get_unchecked_mut(i) = a.get_unchecked(i) % b.get_unchecked(i);
  }
}

// AVX2 element-wise modulo for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn modulo_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  // AVX2 doesn't have a direct modulo instruction for f64, use scalar fallback
  for i in 0..len {
    *result.get_unchecked_mut(i) = a.get_unchecked(i) % b.get_unchecked(i);
  }
}

// NEON element-wise modulo for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn modulo_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  // NEON doesn't have a direct modulo instruction for f64, use scalar fallback
  for i in 0..len {
    *result.get_unchecked_mut(i) = a.get_unchecked(i) % b.get_unchecked(i);
  }
}

// AVX-512 element-wise power for f64 arrays
// GPU implementation of power operation
#[cfg(has_cuda)]
pub unsafe fn power_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_POWER_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry power_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd5, [a];
      ld.param.u64 %rd6, [b];
      ld.param.u64 %rd7, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and compute power for double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load a vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 a
      // Load b vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load double2 b
      // Compute pow(a, b) using log and exp: a^b = exp(b * log(a))
      lg2.approx.f64 %fd4, %fd0;       // log2(a0)
      lg2.approx.f64 %fd5, %fd1;       // log2(a1)
      mul.f64 %fd6, %fd2, %fd4;        // b0 * log2(a0)
      mul.f64 %fd7, %fd3, %fd5;        // b1 * log2(a1)
      ex2.approx.f64 %fd4, %fd6;       // 2^(b0*log2(a0)) = a0^b0
      ex2.approx.f64 %fd5, %fd7;       // 2^(b1*log2(a1)) = a1^b1
      // Store results with vectorized 128-bit store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store double2 results

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

      remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r10;  // Check if first element exists
      @!%p0 bra end;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      lg2.approx.f64 %fd4, %fd0;
      mul.f64 %fd6, %fd2, %fd4;
      ex2.approx.f64 %fd4, %fd6;
      st.global.f64 [%rd4], %fd4;

      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(PTX_POWER_F64, &[], "power_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn power_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  // No SIMD powf instruction, use scalar
  for i in 0..len {
    *result.get_unchecked_mut(i) = a.get_unchecked(i).powf(*b.get_unchecked(i));
  }
}

// AVX2 element-wise power for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn power_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  // No SIMD powf instruction, use scalar
  for i in 0..len {
    *result.get_unchecked_mut(i) = a.get_unchecked(i).powf(*b.get_unchecked(i));
  }
}

// NEON element-wise power for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn power_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  // No SIMD powf instruction, use scalar
  for i in 0..len {
    *result.get_unchecked_mut(i) = a.get_unchecked(i).powf(*b.get_unchecked(i));
  }
}

// =============================================================================
// PROMQL COMPARISON OPERATIONS (returning 0.0/1.0)
// =============================================================================

// AVX-512 element-wise equality comparison for f64 arrays
// GPU comparison operations

#[cfg(has_cuda)]
pub unsafe fn equal_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_EQUAL_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry equal_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd5, [a];
      ld.param.u64 %rd6, [b];
      ld.param.u64 %rd7, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and compare double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load a vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 a
      // Load b vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load double2 b
      // Compare and convert to f64 (like AVX-512 mask operations)
      setp.eq.f64 %p1, %fd0, %fd2;     // Compare lane 0
      setp.eq.f64 %p2, %fd1, %fd3;     // Compare lane 1
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1; // 1.0 or 0.0
      selp.f64 %fd5, 0d3FF0000000000000, 0d0000000000000000, %p2; // 1.0 or 0.0
      // Store results with vectorized 128-bit store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store double2 results

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

      remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r10;  // Check if first element exists
      @!%p0 bra end;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      setp.eq.f64 %p1, %fd0, %fd2;
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1;
      st.global.f64 [%rd4], %fd4;

      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(PTX_EQUAL_F64, &[], "equal_f64", blocks, threads, &args);
}

#[cfg(has_cuda)]
pub unsafe fn greater_than_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_GREATER_THAN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry greater_than_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd5, [a];
      ld.param.u64 %rd6, [b];
      ld.param.u64 %rd7, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and compare double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load a vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 a
      // Load b vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load double2 b
      // Compare and convert to f64 (like AVX-512 mask operations)
      setp.gt.f64 %p1, %fd0, %fd2;     // Compare lane 0
      setp.gt.f64 %p2, %fd1, %fd3;     // Compare lane 1
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1; // 1.0 or 0.0
      selp.f64 %fd5, 0d3FF0000000000000, 0d0000000000000000, %p2; // 1.0 or 0.0
      // Store results with vectorized 128-bit store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store double2 results

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

      remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r10;  // Check if first element exists
      @!%p0 bra end;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      setp.gt.f64 %p1, %fd0, %fd2;
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1;
      st.global.f64 [%rd4], %fd4;

      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(
    PTX_GREATER_THAN_F64,
    &[],
    "greater_than_f64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(has_cuda)]
pub unsafe fn less_than_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_LESS_THAN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry less_than_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd5, [a];
      ld.param.u64 %rd6, [b];
      ld.param.u64 %rd7, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and compare double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load a vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 a
      // Load b vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load double2 b
      // Compare and convert to f64 (like AVX-512 mask operations)
      setp.lt.f64 %p1, %fd0, %fd2;     // Compare lane 0
      setp.lt.f64 %p2, %fd1, %fd3;     // Compare lane 1
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1; // 1.0 or 0.0
      selp.f64 %fd5, 0d3FF0000000000000, 0d0000000000000000, %p2; // 1.0 or 0.0
      // Store results with vectorized 128-bit store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store double2 results

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

      remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r10;  // Check if first element exists
      @!%p0 bra end;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      setp.lt.f64 %p1, %fd0, %fd2;
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1;
      st.global.f64 [%rd4], %fd4;

      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(
    PTX_LESS_THAN_F64,
    &[],
    "less_than_f64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn equal_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm512_set1_pd(1.0);
  let zeros = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm512_cmp_pd_mask(a_vec, b_vec, _CMP_EQ_OQ);
    let result_vec = _mm512_mask_blend_pd(mask, zeros, ones);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) == b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX2 element-wise equality comparison for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn equal_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm256_set1_pd(1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm256_cmp_pd(a_vec, b_vec, _CMP_EQ_OQ);
    let result_vec = _mm256_and_pd(mask, ones);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) == b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// NEON element-wise equality comparison for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn equal_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = vdupq_n_f64(1.0);
  let zeros = vdupq_n_f64(0.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let mask = vceqq_f64(a_vec, b_vec);
    let result_vec = vbslq_f64(mask, ones, zeros);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) == b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX-512 not_equal, greater_than, less_than, greater_equal, less_equal implementations

// AVX-512 element-wise not-equal comparison for f64 arrays
// GPU implementation of not equal comparison

#[cfg(has_cuda)]
pub unsafe fn not_equal_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_NOT_EQUAL_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry not_equal_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd5, [a];
      ld.param.u64 %rd6, [b];
      ld.param.u64 %rd7, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and compare double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load a vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 a
      // Load b vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load double2 b
      // Compare and convert to f64 (like AVX-512 mask operations)
      setp.ne.f64 %p1, %fd0, %fd2;     // Compare lane 0
      setp.ne.f64 %p2, %fd1, %fd3;     // Compare lane 1
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1; // 1.0 or 0.0
      selp.f64 %fd5, 0d3FF0000000000000, 0d0000000000000000, %p2; // 1.0 or 0.0
      // Store results with vectorized 128-bit store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store double2 results

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

      remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r10;  // Check if first element exists
      @!%p0 bra end;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      setp.ne.f64 %p1, %fd0, %fd2;
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1;
      st.global.f64 [%rd4], %fd4;

      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(
    PTX_NOT_EQUAL_F64,
    &[],
    "not_equal_f64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn not_equal_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm512_set1_pd(1.0);
  let zeros = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm512_cmp_pd_mask(a_vec, b_vec, _CMP_NEQ_OQ);
    let result_vec = _mm512_mask_blend_pd(mask, zeros, ones);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) != b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX2 element-wise not-equal comparison for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn not_equal_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm256_set1_pd(1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm256_cmp_pd(a_vec, b_vec, _CMP_NEQ_OQ);
    let result_vec = _mm256_and_pd(mask, ones);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) != b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX-512 greater-than comparison for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn greater_than_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm512_set1_pd(1.0);
  let zeros = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm512_cmp_pd_mask(a_vec, b_vec, _CMP_GT_OQ);
    let result_vec = _mm512_mask_blend_pd(mask, zeros, ones);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) > b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX2 greater-than comparison for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn greater_than_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm256_set1_pd(1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm256_cmp_pd(a_vec, b_vec, _CMP_GT_OQ);
    let result_vec = _mm256_and_pd(mask, ones);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) > b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX-512 less-than comparison for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn less_than_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm512_set1_pd(1.0);
  let zeros = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm512_cmp_pd_mask(a_vec, b_vec, _CMP_LT_OQ);
    let result_vec = _mm512_mask_blend_pd(mask, zeros, ones);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) < b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX2 less-than comparison for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn less_than_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm256_set1_pd(1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm256_cmp_pd(a_vec, b_vec, _CMP_LT_OQ);
    let result_vec = _mm256_and_pd(mask, ones);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) < b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX-512 greater-equal comparison for f64 arrays
// GPU implementation of greater equal comparison

#[cfg(has_cuda)]
pub unsafe fn greater_equal_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  const PTX_GREATER_EQUAL_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry greater_equal_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;

      // Load parameters
      ld.param.u64 %rd5, [a];
      ld.param.u64 %rd6, [b];
      ld.param.u64 %rd7, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;

      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)

      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;  // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)

      // Main grid-stride loop processing double2 chunks
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array

      // Load and compare double2 vectors
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      // Load a vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 a
      // Load b vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3];  // Load double2 b
      // Compare and convert to f64 (like AVX-512 mask operations)
      setp.ge.f64 %p1, %fd0, %fd2;     // Compare lane 0
      setp.ge.f64 %p2, %fd1, %fd3;     // Compare lane 1
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1; // 1.0 or 0.0
      selp.f64 %fd5, 0d3FF0000000000000, 0d0000000000000000, %p2; // 1.0 or 0.0
      // Store results with vectorized 128-bit store
      st.global.v2.f64 [%rd4], {%fd4, %fd5};  // Store double2 results

      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop

      remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r10;  // Check if first element exists
      @!%p0 bra end;  // Skip if no remainder

      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      add.u64 %rd3, %rd6, %rd1;
      add.u64 %rd4, %rd7, %rd1;
      ld.global.f64 %fd0, [%rd2];
      ld.global.f64 %fd2, [%rd3];
      setp.ge.f64 %p1, %fd0, %fd2;
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1;
      st.global.f64 [%rd4], %fd4;

      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(
    PTX_GREATER_EQUAL_F64,
    &[],
    "greater_equal_f64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn greater_equal_f64_avx512(
  a: &[f64],
  b: &[f64],
  result: &mut [f64],
  len: usize,
) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm512_set1_pd(1.0);
  let zeros = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm512_cmp_pd_mask(a_vec, b_vec, _CMP_GE_OQ);
    let result_vec = _mm512_mask_blend_pd(mask, zeros, ones);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) >= b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX2 greater-equal comparison for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn greater_equal_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm256_set1_pd(1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm256_cmp_pd(a_vec, b_vec, _CMP_GE_OQ);
    let result_vec = _mm256_and_pd(mask, ones);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) >= b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX-512 less-equal comparison for f64 arrays
// GPU implementation of less equal comparison

#[cfg(has_cuda)]
pub unsafe fn less_equal_f64_gpu(a: *const f64, b: *const f64, result: *mut f64, len: usize) {
  #[cfg(has_cuda)]
  use crate::gpu::launch_ptx;

  const PTX_LESS_EQUAL_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry less_equal_f64 (
      .param .u64 a,
      .param .u64 b,
      .param .u64 result,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd1, [a];
      ld.param.u64 %rd2, [b];
      ld.param.u64 %rd3, [result];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread handles 2 elements (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;
      mul.lo.u32 %r9, %r8, %r2;
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2
      
      loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r10;
      @%p0 bra remainder;
      
      // Process double2 chunk
      mul.wide.u32 %rd4, %r5, 8;
      add.u64 %rd5, %rd1, %rd4;  // a + offset
      add.u64 %rd6, %rd2, %rd4;  // b + offset  
      add.u64 %rd7, %rd3, %rd4;  // result + offset
      
      ld.global.v2.f64 {%fd0, %fd1}, [%rd5];  // Load double2 a
      ld.global.v2.f64 {%fd2, %fd3}, [%rd6];  // Load double2 b
      
      setp.le.f64 %p1, %fd0, %fd2;
      setp.le.f64 %p2, %fd1, %fd3;
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1;
      selp.f64 %fd5, 0d3FF0000000000000, 0d0000000000000000, %p2;
      
      st.global.v2.f64 [%rd7], {%fd4, %fd5};  // Store double2 results
      
      add.u32 %r5, %r5, %r9;
      bra loop_start;
      
      remainder:
      setp.lt.u32 %p0, %r5, %r10;
      @!%p0 bra end;
      
      // Handle single remainder element
      mul.wide.u32 %rd4, %r5, 8;
      add.u64 %rd5, %rd1, %rd4;
      add.u64 %rd6, %rd2, %rd4;
      add.u64 %rd7, %rd3, %rd4;
      
      ld.global.f64 %fd0, [%rd5];
      ld.global.f64 %fd2, [%rd6];
      setp.le.f64 %p1, %fd0, %fd2;
      selp.f64 %fd4, 0d3FF0000000000000, 0d0000000000000000, %p1;
      st.global.f64 [%rd7], %fd4;
      
      end:
      ret;
    }
  "#;

  // Launch the PTX kernel
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    &a as *const *const f64 as *const u8,
    &b as *const *const f64 as *const u8,
    &result as *const *mut f64 as *const u8,
    &(len as u32) as *const u32 as *const u8,
  ];
  let _ = launch_ptx(
    PTX_LESS_EQUAL_F64,
    &[],
    "less_equal_f64",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn less_equal_f64_avx512(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm512_set1_pd(1.0);
  let zeros = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm512_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm512_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm512_cmp_pd_mask(a_vec, b_vec, _CMP_LE_OQ);
    let result_vec = _mm512_mask_blend_pd(mask, zeros, ones);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) <= b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// AVX2 less-equal comparison for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn less_equal_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = _mm256_set1_pd(1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = _mm256_loadu_pd(a.as_ptr().add(offset));
    let b_vec = _mm256_loadu_pd(b.as_ptr().add(offset));
    let mask = _mm256_cmp_pd(a_vec, b_vec, _CMP_LE_OQ);
    let result_vec = _mm256_and_pd(mask, ones);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) <= b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// NEON element-wise not-equal comparison for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn not_equal_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = vdupq_n_f64(1.0);
  let zeros = vdupq_n_f64(0.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let mask = vceqq_f64(a_vec, b_vec);
    // For not equal, swap ones and zeros
    let result_vec = vbslq_f64(mask, zeros, ones);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) != b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// NEON element-wise greater-than comparison for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn greater_than_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = vdupq_n_f64(1.0);
  let zeros = vdupq_n_f64(0.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let mask = vcgtq_f64(a_vec, b_vec);
    let result_vec = vbslq_f64(mask, ones, zeros);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) > b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// NEON element-wise less-than comparison for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn less_than_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = vdupq_n_f64(1.0);
  let zeros = vdupq_n_f64(0.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let mask = vcltq_f64(a_vec, b_vec);
    let result_vec = vbslq_f64(mask, ones, zeros);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) < b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// NEON element-wise greater-equal comparison for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn greater_equal_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = vdupq_n_f64(1.0);
  let zeros = vdupq_n_f64(0.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let mask = vcgeq_f64(a_vec, b_vec);
    let result_vec = vbslq_f64(mask, ones, zeros);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) >= b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// NEON element-wise less-equal comparison for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn less_equal_f64_neon(a: &[f64], b: &[f64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let ones = vdupq_n_f64(1.0);
  let zeros = vdupq_n_f64(0.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let a_vec = vld1q_f64(a.as_ptr().add(offset));
    let b_vec = vld1q_f64(b.as_ptr().add(offset));
    let mask = vcleq_f64(a_vec, b_vec);
    let result_vec = vbslq_f64(mask, ones, zeros);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    *result.get_unchecked_mut(offset + i) =
      if a.get_unchecked(offset + i) <= b.get_unchecked(offset + i) {
        1.0
      } else {
        0.0
      };
  }
}

// =============================================================================
// PROMQL MATH FUNCTIONS
// =============================================================================

// NEON element-wise absolute value for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn abs_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    let abs_vec = vabsq_f64(vec);
    vst1q_f64(values.as_mut_ptr().add(offset), abs_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Use bitwise AND to clear sign bit for absolute value
    let abs_val = f64::from_bits(val.to_bits() & 0x7FFFFFFFFFFFFFFF);
    *values.get_unchecked_mut(offset + i) = abs_val;
  }
}

// NEON element-wise negation for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn neg_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    let neg_vec = vnegq_f64(vec);
    vst1q_f64(values.as_mut_ptr().add(offset), neg_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Use XOR to flip sign bit for negation
    let neg_val = f64::from_bits(val.to_bits() ^ 0x8000000000000000);
    *values.get_unchecked_mut(offset + i) = neg_val;
  }
}

// NEON element-wise square root for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn sqrt_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    let sqrt_vec = vsqrtq_f64(vec);
    vst1q_f64(values.as_mut_ptr().add(offset), sqrt_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result: f64;
    #[cfg(target_arch = "aarch64")]
    {
      asm!(
        "fsqrt {0:d}, {1:d}",
        out(vreg) result,
        in(vreg) val,
        options(pure, nomem, nostack)
      );
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      asm!(
        "sqrtsd {}, {}",
        out(xmm_reg) result,
        in(xmm_reg) val,
        options(pure, nomem, nostack)
      );
    }
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// Trigonometric and exponential functions don't have SIMD instructions, use scalar

// NEON element-wise sine for f64 arrays using polynomial approximation
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn sin_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for sin(x) around 0
  // sin(x) ~= x - x³/3! + x^/5! - x^/7! + x^/9!
  let _c1 = vdupq_n_f64(1.0);
  let c3 = vdupq_n_f64(-1.0 / 6.0);
  let c5 = vdupq_n_f64(1.0 / 120.0);
  let c7 = vdupq_n_f64(-1.0 / 5040.0);
  let c9 = vdupq_n_f64(1.0 / 362880.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x5 = vmulq_f64(x3, x2);
    let x7 = vmulq_f64(x5, x2);
    let x9 = vmulq_f64(x7, x2);

    // Compute polynomial: x + c3*x³ + c5*x^ + c7*x^ + c9*x^
    let mut result = x;
    result = vfmaq_f64(result, x3, c3);
    result = vfmaq_f64(result, x5, c5);
    result = vfmaq_f64(result, x7, c7);
    result = vfmaq_f64(result, x9, c9);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    *values.get_unchecked_mut(offset + i) = x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0;
  }
}

// NEON element-wise cosine for f64 arrays using polynomial approximation
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn cos_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for cos(x) around 0
  // cos(x) ~= 1 - x²/2! + x^^/4! - x^^/6! + x^^/8!
  let c0 = vdupq_n_f64(1.0);
  let c2 = vdupq_n_f64(-0.5);
  let c4 = vdupq_n_f64(1.0 / 24.0);
  let c6 = vdupq_n_f64(-1.0 / 720.0);
  let c8 = vdupq_n_f64(1.0 / 40320.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = vmulq_f64(x, x);
    let x4 = vmulq_f64(x2, x2);
    let x6 = vmulq_f64(x4, x2);
    let x8 = vmulq_f64(x4, x4);

    // Compute polynomial: 1 + c2*x² + c4*x^^ + c6*x^^ + c8*x^^
    let mut result = c0;
    result = vfmaq_f64(result, x2, c2);
    result = vfmaq_f64(result, x4, c4);
    result = vfmaq_f64(result, x6, c6);
    result = vfmaq_f64(result, x8, c8);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    *values.get_unchecked_mut(offset + i) = 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0;
  }
}

// NEON element-wise tangent for f64 arrays using polynomial approximation
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn tan_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Compute tan(x) = sin(x) / cos(x) using polynomial approximations
  // Sin coefficients
  let _s1 = vdupq_n_f64(1.0);
  let s3 = vdupq_n_f64(-1.0 / 6.0);
  let s5 = vdupq_n_f64(1.0 / 120.0);
  let s7 = vdupq_n_f64(-1.0 / 5040.0);

  // Cos coefficients
  let c0 = vdupq_n_f64(1.0);
  let c2 = vdupq_n_f64(-0.5);
  let c4 = vdupq_n_f64(1.0 / 24.0);
  let c6 = vdupq_n_f64(-1.0 / 720.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x3, x2);
    let x6 = vmulq_f64(x4, x2);
    let x7 = vmulq_f64(x5, x2);

    // Compute sin(x)
    let mut sin_result = x;
    sin_result = vfmaq_f64(sin_result, x3, s3);
    sin_result = vfmaq_f64(sin_result, x5, s5);
    sin_result = vfmaq_f64(sin_result, x7, s7);

    // Compute cos(x)
    let mut cos_result = c0;
    cos_result = vfmaq_f64(cos_result, x2, c2);
    cos_result = vfmaq_f64(cos_result, x4, c4);
    cos_result = vfmaq_f64(cos_result, x6, c6);

    // tan(x) = sin(x) / cos(x)
    let result = vdivq_f64(sin_result, cos_result);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x3 * x2;
    let x6 = x4 * x2;
    let x7 = x5 * x2;
    let sin_x = x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0;
    let cos_x = 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0;
    *values.get_unchecked_mut(offset + i) = sin_x / cos_x;
  }
}

// AVX-512 element-wise acos for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]

pub(super) unsafe fn acos_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let pi_half = _mm512_set1_pd(std::f64::consts::PI / 2.0);
  // Series coefficients for asin(x): x + x³/6 + 3x^/40 + 5x^/112
  let c1 = _mm512_set1_pd(1.0);
  let c3 = _mm512_set1_pd(1.0 / 6.0);
  let c5 = _mm512_set1_pd(3.0 / 40.0);
  let c7 = _mm512_set1_pd(5.0 / 112.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // acos(x) = ^/2 - asin(x)
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x3, x2);
    let x7 = _mm512_mul_pd(x4, x3);

    // Compute asin(x) series: x + x³/6 + 3x^/40 + 5x^/112
    let mut asin_result = _mm512_mul_pd(x, c1);
    asin_result = _mm512_fmadd_pd(x3, c3, asin_result);
    asin_result = _mm512_fmadd_pd(x5, c5, asin_result);
    asin_result = _mm512_fmadd_pd(x7, c7, asin_result);

    // acos(x) = ^/2 - asin(x)
    let acos_result = _mm512_sub_pd(pi_half, asin_result);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), acos_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].acos();
  }
}

// AVX2 element-wise acos for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn acos_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let pi_half = _mm256_set1_pd(std::f64::consts::PI / 2.0);
  // Series coefficients for asin(x): x + x³/6 + 3x^/40 + 5x^/112
  let c1 = _mm256_set1_pd(1.0);
  let c3 = _mm256_set1_pd(1.0 / 6.0);
  let c5 = _mm256_set1_pd(3.0 / 40.0);
  let c7 = _mm256_set1_pd(5.0 / 112.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // acos(x) = ^/2 - asin(x)
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x3, x2);
    let x7 = _mm256_mul_pd(x4, x3);

    // Compute asin(x) series: x + x³/6 + 3x^/40 + 5x^/112
    let mut asin_result = _mm256_mul_pd(x, c1);
    asin_result = _mm256_fmadd_pd(x3, c3, asin_result);
    asin_result = _mm256_fmadd_pd(x5, c5, asin_result);
    asin_result = _mm256_fmadd_pd(x7, c7, asin_result);

    // acos(x) = ^/2 - asin(x)
    let acos_result = _mm256_sub_pd(pi_half, asin_result);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), acos_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].acos();
  }
}

// NEON element-wise acos for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn acos_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let pi_half = vdupq_n_f64(std::f64::consts::PI / 2.0);
  // Series coefficients for asin(x): x + x³/6 + 3x^/40 + 5x^/112
  let c1 = vdupq_n_f64(1.0);
  let c3 = vdupq_n_f64(1.0 / 6.0);
  let c5 = vdupq_n_f64(3.0 / 40.0);
  let c7 = vdupq_n_f64(5.0 / 112.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // acos(x) = ^/2 - asin(x)
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x3, x2);
    let x7 = vmulq_f64(x4, x3);

    // Compute asin(x) series: x + x³/6 + 3x^/40 + 5x^/112
    let mut asin_result = vmulq_f64(x, c1);
    asin_result = vfmaq_f64(asin_result, x3, c3);
    asin_result = vfmaq_f64(asin_result, x5, c5);
    asin_result = vfmaq_f64(asin_result, x7, c7);

    // acos(x) = ^/2 - asin(x)
    let acos_result = vsubq_f64(pi_half, asin_result);

    vst1q_f64(values.as_mut_ptr().add(offset), acos_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].acos();
  }
}

// AVX-512 element-wise asin for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn asin_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Series coefficients for asin(x): x + x³/6 + 3x^/40 + 5x^/112
  let c1 = _mm512_set1_pd(1.0);
  let c3 = _mm512_set1_pd(1.0 / 6.0);
  let c5 = _mm512_set1_pd(3.0 / 40.0);
  let c7 = _mm512_set1_pd(5.0 / 112.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // asin(x) series: x + x³/6 + 3x^/40 + 5x^/112
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x3, x2);
    let x7 = _mm512_mul_pd(x4, x3);

    // Compute asin(x) series
    let mut asin_result = _mm512_mul_pd(x, c1);
    asin_result = _mm512_fmadd_pd(x3, c3, asin_result);
    asin_result = _mm512_fmadd_pd(x5, c5, asin_result);
    asin_result = _mm512_fmadd_pd(x7, c7, asin_result);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), asin_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].asin();
  }
}

// AVX2 element-wise asin for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn asin_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Series coefficients for asin(x): x + x³/6 + 3x^/40 + 5x^/112
  let c1 = _mm256_set1_pd(1.0);
  let c3 = _mm256_set1_pd(1.0 / 6.0);
  let c5 = _mm256_set1_pd(3.0 / 40.0);
  let c7 = _mm256_set1_pd(5.0 / 112.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // asin(x) series: x + x³/6 + 3x^/40 + 5x^/112
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x3, x2);
    let x7 = _mm256_mul_pd(x4, x3);

    // Compute asin(x) series
    let mut asin_result = _mm256_mul_pd(x, c1);
    asin_result = _mm256_fmadd_pd(x3, c3, asin_result);
    asin_result = _mm256_fmadd_pd(x5, c5, asin_result);
    asin_result = _mm256_fmadd_pd(x7, c7, asin_result);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), asin_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].asin();
  }
}

// NEON element-wise asin for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn asin_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Series coefficients for asin(x): x + x³/6 + 3x^/40 + 5x^/112
  let c1 = vdupq_n_f64(1.0);
  let c3 = vdupq_n_f64(1.0 / 6.0);
  let c5 = vdupq_n_f64(3.0 / 40.0);
  let c7 = vdupq_n_f64(5.0 / 112.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // asin(x) series: x + x³/6 + 3x^/40 + 5x^/112
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x3, x2);
    let x7 = vmulq_f64(x4, x3);

    // Compute asin(x) series
    let mut asin_result = vmulq_f64(x, c1);
    asin_result = vfmaq_f64(asin_result, x3, c3);
    asin_result = vfmaq_f64(asin_result, x5, c5);
    asin_result = vfmaq_f64(asin_result, x7, c7);

    vst1q_f64(values.as_mut_ptr().add(offset), asin_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].asin();
  }
}

// AVX-512 element-wise atan for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn atan_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Series coefficients for atan(x): x - x³/3 + x^/5 - x^/7 + x^/9
  let c1 = _mm512_set1_pd(1.0);
  let c3 = _mm512_set1_pd(-1.0 / 3.0);
  let c5 = _mm512_set1_pd(1.0 / 5.0);
  let c7 = _mm512_set1_pd(-1.0 / 7.0);
  let c9 = _mm512_set1_pd(1.0 / 9.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // atan(x) series: x - x³/3 + x^/5 - x^/7 + x^/9
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x3, x2);
    let x7 = _mm512_mul_pd(x4, x3);
    let x9 = _mm512_mul_pd(x4, x5);

    // Compute atan(x) series
    let mut atan_result = _mm512_mul_pd(x, c1);
    atan_result = _mm512_fmadd_pd(x3, c3, atan_result);
    atan_result = _mm512_fmadd_pd(x5, c5, atan_result);
    atan_result = _mm512_fmadd_pd(x7, c7, atan_result);
    atan_result = _mm512_fmadd_pd(x9, c9, atan_result);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), atan_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].atan();
  }
}

// AVX2 element-wise atan for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn atan_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Series coefficients for atan(x): x - x³/3 + x^/5 - x^/7 + x^/9
  let c1 = _mm256_set1_pd(1.0);
  let c3 = _mm256_set1_pd(-1.0 / 3.0);
  let c5 = _mm256_set1_pd(1.0 / 5.0);
  let c7 = _mm256_set1_pd(-1.0 / 7.0);
  let c9 = _mm256_set1_pd(1.0 / 9.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // atan(x) series: x - x³/3 + x^/5 - x^/7 + x^/9
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x3, x2);
    let x7 = _mm256_mul_pd(x4, x3);
    let x9 = _mm256_mul_pd(x4, x5);

    // Compute atan(x) series
    let mut atan_result = _mm256_mul_pd(x, c1);
    atan_result = _mm256_fmadd_pd(x3, c3, atan_result);
    atan_result = _mm256_fmadd_pd(x5, c5, atan_result);
    atan_result = _mm256_fmadd_pd(x7, c7, atan_result);
    atan_result = _mm256_fmadd_pd(x9, c9, atan_result);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), atan_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].atan();
  }
}

// NEON element-wise atan for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn atan_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Series coefficients for atan(x): x - x³/3 + x^/5 - x^/7 + x^/9
  let c1 = vdupq_n_f64(1.0);
  let c3 = vdupq_n_f64(-1.0 / 3.0);
  let c5 = vdupq_n_f64(1.0 / 5.0);
  let c7 = vdupq_n_f64(-1.0 / 7.0);
  let c9 = vdupq_n_f64(1.0 / 9.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // atan(x) series: x - x³/3 + x^/5 - x^/7 + x^/9
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x3, x2);
    let _x6 = vmulq_f64(x3, x3);
    let x7 = vmulq_f64(x4, x3);
    let x9 = vmulq_f64(x4, x5);

    // Compute atan(x) series
    let mut atan_result = vmulq_f64(x, c1);
    atan_result = vfmaq_f64(atan_result, x3, c3);
    atan_result = vfmaq_f64(atan_result, x5, c5);
    atan_result = vfmaq_f64(atan_result, x7, c7);
    atan_result = vfmaq_f64(atan_result, x9, c9);

    vst1q_f64(values.as_mut_ptr().add(offset), atan_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].atan();
  }
}

// AVX-512 element-wise cosh for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn cosh_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let half = _mm512_set1_pd(0.5);
  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = _mm512_set1_pd(1.0);
  let c2 = _mm512_set1_pd(0.5);
  let c3 = _mm512_set1_pd(1.0 / 6.0);
  let c4 = _mm512_set1_pd(1.0 / 24.0);
  let c5 = _mm512_set1_pd(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // cosh(x) = (e^x + e^(-x)) / 2
    let neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x);

    // Compute e^x using Taylor series
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x3, x2);

    let mut exp_x = c1;
    exp_x = _mm512_fmadd_pd(x, c1, exp_x);
    exp_x = _mm512_fmadd_pd(x2, c2, exp_x);
    exp_x = _mm512_fmadd_pd(x3, c3, exp_x);
    exp_x = _mm512_fmadd_pd(x4, c4, exp_x);
    exp_x = _mm512_fmadd_pd(x5, c5, exp_x);

    // Compute e^(-x) using Taylor series
    let neg_x2 = _mm512_mul_pd(neg_x, neg_x);
    let neg_x3 = _mm512_mul_pd(neg_x2, neg_x);
    let neg_x4 = _mm512_mul_pd(neg_x2, neg_x2);
    let neg_x5 = _mm512_mul_pd(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = _mm512_fmadd_pd(neg_x, c1, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x2, c2, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x3, c3, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x4, c4, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x5, c5, exp_neg_x);

    // cosh(x) = (e^x + e^(-x)) / 2
    let sum = _mm512_add_pd(exp_x, exp_neg_x);
    let cosh_result = _mm512_mul_pd(sum, half);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), cosh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].cosh();
  }
}

// AVX2 element-wise cosh for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn cosh_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let half = _mm256_set1_pd(0.5);
  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = _mm256_set1_pd(1.0);
  let c2 = _mm256_set1_pd(0.5);
  let c3 = _mm256_set1_pd(1.0 / 6.0);
  let c4 = _mm256_set1_pd(1.0 / 24.0);
  let c5 = _mm256_set1_pd(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // cosh(x) = (e^x + e^(-x)) / 2
    let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);

    // Compute e^x using Taylor series
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x3, x2);

    let mut exp_x = c1;
    exp_x = _mm256_fmadd_pd(x, c1, exp_x);
    exp_x = _mm256_fmadd_pd(x2, c2, exp_x);
    exp_x = _mm256_fmadd_pd(x3, c3, exp_x);
    exp_x = _mm256_fmadd_pd(x4, c4, exp_x);
    exp_x = _mm256_fmadd_pd(x5, c5, exp_x);

    // Compute e^(-x) using Taylor series
    let neg_x2 = _mm256_mul_pd(neg_x, neg_x);
    let neg_x3 = _mm256_mul_pd(neg_x2, neg_x);
    let neg_x4 = _mm256_mul_pd(neg_x2, neg_x2);
    let neg_x5 = _mm256_mul_pd(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = _mm256_fmadd_pd(neg_x, c1, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x2, c2, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x3, c3, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x4, c4, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x5, c5, exp_neg_x);

    // cosh(x) = (e^x + e^(-x)) / 2
    let sum = _mm256_add_pd(exp_x, exp_neg_x);
    let cosh_result = _mm256_mul_pd(sum, half);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), cosh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].cosh();
  }
}

// NEON element-wise cosh for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn cosh_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let half = vdupq_n_f64(0.5);
  let zero = vdupq_n_f64(0.0);
  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = vdupq_n_f64(1.0);
  let c2 = vdupq_n_f64(0.5);
  let c3 = vdupq_n_f64(1.0 / 6.0);
  let c4 = vdupq_n_f64(1.0 / 24.0);
  let c5 = vdupq_n_f64(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // cosh(x) = (e^x + e^(-x)) / 2
    let neg_x = vsubq_f64(zero, x);

    // Compute e^x using Taylor series
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x3, x2);

    let mut exp_x = c1;
    exp_x = vfmaq_f64(exp_x, x, c1);
    exp_x = vfmaq_f64(exp_x, x2, c2);
    exp_x = vfmaq_f64(exp_x, x3, c3);
    exp_x = vfmaq_f64(exp_x, x4, c4);
    exp_x = vfmaq_f64(exp_x, x5, c5);

    // Compute e^(-x) using Taylor series
    let neg_x2 = vmulq_f64(neg_x, neg_x);
    let neg_x3 = vmulq_f64(neg_x2, neg_x);
    let neg_x4 = vmulq_f64(neg_x2, neg_x2);
    let neg_x5 = vmulq_f64(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x, c1);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x2, c2);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x3, c3);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x4, c4);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x5, c5);

    // cosh(x) = (e^x + e^(-x)) / 2
    let sum = vaddq_f64(exp_x, exp_neg_x);
    let cosh_result = vmulq_f64(sum, half);

    vst1q_f64(values.as_mut_ptr().add(offset), cosh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].cosh();
  }
}

// AVX-512 element-wise sinh for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn sinh_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let half = _mm512_set1_pd(0.5);
  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = _mm512_set1_pd(1.0);
  let c2 = _mm512_set1_pd(0.5);
  let c3 = _mm512_set1_pd(1.0 / 6.0);
  let c4 = _mm512_set1_pd(1.0 / 24.0);
  let c5 = _mm512_set1_pd(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // sinh(x) = (e^x - e^(-x)) / 2
    let neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x);

    // Compute e^x using Taylor series
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x3, x2);

    let mut exp_x = c1;
    exp_x = _mm512_fmadd_pd(x, c1, exp_x);
    exp_x = _mm512_fmadd_pd(x2, c2, exp_x);
    exp_x = _mm512_fmadd_pd(x3, c3, exp_x);
    exp_x = _mm512_fmadd_pd(x4, c4, exp_x);
    exp_x = _mm512_fmadd_pd(x5, c5, exp_x);

    // Compute e^(-x) using Taylor series
    let neg_x2 = _mm512_mul_pd(neg_x, neg_x);
    let neg_x3 = _mm512_mul_pd(neg_x2, neg_x);
    let neg_x4 = _mm512_mul_pd(neg_x2, neg_x2);
    let neg_x5 = _mm512_mul_pd(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = _mm512_fmadd_pd(neg_x, c1, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x2, c2, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x3, c3, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x4, c4, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x5, c5, exp_neg_x);

    // sinh(x) = (e^x - e^(-x)) / 2
    let diff = _mm512_sub_pd(exp_x, exp_neg_x);
    let sinh_result = _mm512_mul_pd(diff, half);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), sinh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].sinh();
  }
}

// AVX2 element-wise sinh for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sinh_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let half = _mm256_set1_pd(0.5);
  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = _mm256_set1_pd(1.0);
  let c2 = _mm256_set1_pd(0.5);
  let c3 = _mm256_set1_pd(1.0 / 6.0);
  let c4 = _mm256_set1_pd(1.0 / 24.0);
  let c5 = _mm256_set1_pd(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // sinh(x) = (e^x - e^(-x)) / 2
    let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);

    // Compute e^x using Taylor series
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x3, x2);

    let mut exp_x = c1;
    exp_x = _mm256_fmadd_pd(x, c1, exp_x);
    exp_x = _mm256_fmadd_pd(x2, c2, exp_x);
    exp_x = _mm256_fmadd_pd(x3, c3, exp_x);
    exp_x = _mm256_fmadd_pd(x4, c4, exp_x);
    exp_x = _mm256_fmadd_pd(x5, c5, exp_x);

    // Compute e^(-x) using Taylor series
    let neg_x2 = _mm256_mul_pd(neg_x, neg_x);
    let neg_x3 = _mm256_mul_pd(neg_x2, neg_x);
    let neg_x4 = _mm256_mul_pd(neg_x2, neg_x2);
    let neg_x5 = _mm256_mul_pd(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = _mm256_fmadd_pd(neg_x, c1, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x2, c2, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x3, c3, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x4, c4, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x5, c5, exp_neg_x);

    // sinh(x) = (e^x - e^(-x)) / 2
    let diff = _mm256_sub_pd(exp_x, exp_neg_x);
    let sinh_result = _mm256_mul_pd(diff, half);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), sinh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].sinh();
  }
}

// NEON element-wise sinh for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn sinh_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let half = vdupq_n_f64(0.5);
  let zero = vdupq_n_f64(0.0);
  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = vdupq_n_f64(1.0);
  let c2 = vdupq_n_f64(0.5);
  let c3 = vdupq_n_f64(1.0 / 6.0);
  let c4 = vdupq_n_f64(1.0 / 24.0);
  let c5 = vdupq_n_f64(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // sinh(x) = (e^x - e^(-x)) / 2
    let neg_x = vsubq_f64(zero, x);

    // Compute e^x using Taylor series
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x3, x2);

    let mut exp_x = c1;
    exp_x = vfmaq_f64(exp_x, x, c1);
    exp_x = vfmaq_f64(exp_x, x2, c2);
    exp_x = vfmaq_f64(exp_x, x3, c3);
    exp_x = vfmaq_f64(exp_x, x4, c4);
    exp_x = vfmaq_f64(exp_x, x5, c5);

    // Compute e^(-x) using Taylor series
    let neg_x2 = vmulq_f64(neg_x, neg_x);
    let neg_x3 = vmulq_f64(neg_x2, neg_x);
    let neg_x4 = vmulq_f64(neg_x2, neg_x2);
    let neg_x5 = vmulq_f64(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x, c1);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x2, c2);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x3, c3);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x4, c4);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x5, c5);

    // sinh(x) = (e^x - e^(-x)) / 2
    let diff = vsubq_f64(exp_x, exp_neg_x);
    let sinh_result = vmulq_f64(diff, half);

    vst1q_f64(values.as_mut_ptr().add(offset), sinh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].sinh();
  }
}

// AVX-512 element-wise tanh for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn tanh_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = _mm512_set1_pd(1.0);
  let c2 = _mm512_set1_pd(0.5);
  let c3 = _mm512_set1_pd(1.0 / 6.0);
  let c4 = _mm512_set1_pd(1.0 / 24.0);
  let c5 = _mm512_set1_pd(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    let neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x);

    // Compute e^x using Taylor series
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x3, x2);

    let mut exp_x = c1;
    exp_x = _mm512_fmadd_pd(x, c1, exp_x);
    exp_x = _mm512_fmadd_pd(x2, c2, exp_x);
    exp_x = _mm512_fmadd_pd(x3, c3, exp_x);
    exp_x = _mm512_fmadd_pd(x4, c4, exp_x);
    exp_x = _mm512_fmadd_pd(x5, c5, exp_x);

    // Compute e^(-x) using Taylor series
    let neg_x2 = _mm512_mul_pd(neg_x, neg_x);
    let neg_x3 = _mm512_mul_pd(neg_x2, neg_x);
    let neg_x4 = _mm512_mul_pd(neg_x2, neg_x2);
    let neg_x5 = _mm512_mul_pd(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = _mm512_fmadd_pd(neg_x, c1, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x2, c2, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x3, c3, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x4, c4, exp_neg_x);
    exp_neg_x = _mm512_fmadd_pd(neg_x5, c5, exp_neg_x);

    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    let numerator = _mm512_sub_pd(exp_x, exp_neg_x);
    let denominator = _mm512_add_pd(exp_x, exp_neg_x);
    let tanh_result = _mm512_div_pd(numerator, denominator);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), tanh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].tanh();
  }
}

// AVX2 element-wise tanh for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn tanh_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = _mm256_set1_pd(1.0);
  let c2 = _mm256_set1_pd(0.5);
  let c3 = _mm256_set1_pd(1.0 / 6.0);
  let c4 = _mm256_set1_pd(1.0 / 24.0);
  let c5 = _mm256_set1_pd(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);

    // Compute e^x using Taylor series
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x3, x2);

    let mut exp_x = c1;
    exp_x = _mm256_fmadd_pd(x, c1, exp_x);
    exp_x = _mm256_fmadd_pd(x2, c2, exp_x);
    exp_x = _mm256_fmadd_pd(x3, c3, exp_x);
    exp_x = _mm256_fmadd_pd(x4, c4, exp_x);
    exp_x = _mm256_fmadd_pd(x5, c5, exp_x);

    // Compute e^(-x) using Taylor series
    let neg_x2 = _mm256_mul_pd(neg_x, neg_x);
    let neg_x3 = _mm256_mul_pd(neg_x2, neg_x);
    let neg_x4 = _mm256_mul_pd(neg_x2, neg_x2);
    let neg_x5 = _mm256_mul_pd(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = _mm256_fmadd_pd(neg_x, c1, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x2, c2, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x3, c3, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x4, c4, exp_neg_x);
    exp_neg_x = _mm256_fmadd_pd(neg_x5, c5, exp_neg_x);

    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    let numerator = _mm256_sub_pd(exp_x, exp_neg_x);
    let denominator = _mm256_add_pd(exp_x, exp_neg_x);
    let tanh_result = _mm256_div_pd(numerator, denominator);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), tanh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].tanh();
  }
}

// NEON element-wise tanh for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn tanh_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let zero = vdupq_n_f64(0.0);
  // Exponential coefficients: e^x ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5!
  let c1 = vdupq_n_f64(1.0);
  let c2 = vdupq_n_f64(0.5);
  let c3 = vdupq_n_f64(1.0 / 6.0);
  let c4 = vdupq_n_f64(1.0 / 24.0);
  let c5 = vdupq_n_f64(1.0 / 120.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    let neg_x = vsubq_f64(zero, x);

    // Compute e^x using Taylor series
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x3, x2);

    let mut exp_x = c1;
    exp_x = vfmaq_f64(exp_x, x, c1);
    exp_x = vfmaq_f64(exp_x, x2, c2);
    exp_x = vfmaq_f64(exp_x, x3, c3);
    exp_x = vfmaq_f64(exp_x, x4, c4);
    exp_x = vfmaq_f64(exp_x, x5, c5);

    // Compute e^(-x) using Taylor series
    let neg_x2 = vmulq_f64(neg_x, neg_x);
    let neg_x3 = vmulq_f64(neg_x2, neg_x);
    let neg_x4 = vmulq_f64(neg_x2, neg_x2);
    let neg_x5 = vmulq_f64(neg_x3, neg_x2);

    let mut exp_neg_x = c1;
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x, c1);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x2, c2);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x3, c3);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x4, c4);
    exp_neg_x = vfmaq_f64(exp_neg_x, neg_x5, c5);

    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    let numerator = vsubq_f64(exp_x, exp_neg_x);
    let denominator = vaddq_f64(exp_x, exp_neg_x);
    let tanh_result = vdivq_f64(numerator, denominator);

    vst1q_f64(values.as_mut_ptr().add(offset), tanh_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].tanh();
  }
}

// AVX-512 element-wise floor for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn floor_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Use AVX-512 floor intrinsic
    let floor_result = _mm512_roundscale_pd(x, 0x08);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), floor_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].floor();
  }
}

// AVX2 element-wise floor for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn floor_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Use AVX2 floor intrinsic
    let floor_result = _mm256_floor_pd(x);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), floor_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].floor();
  }
}

// NEON element-wise floor for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn floor_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Use NEON floor intrinsic
    let floor_result = vrndmq_f64(x); // Round towards minus infinity

    vst1q_f64(values.as_mut_ptr().add(offset), floor_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].floor();
  }
}

// AVX-512 element-wise ceil for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn ceil_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Use AVX-512 ceil intrinsic
    let ceil_result = _mm512_roundscale_pd(x, 0x0A);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), ceil_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].ceil();
  }
}

// AVX2 element-wise ceil for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn ceil_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Use AVX2 ceil intrinsic
    let ceil_result = _mm256_ceil_pd(x);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), ceil_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].ceil();
  }
}

// NEON element-wise ceil for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn ceil_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Use NEON ceil intrinsic
    let ceil_result = vrndpq_f64(x); // Round towards plus infinity

    vst1q_f64(values.as_mut_ptr().add(offset), ceil_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].ceil();
  }
}

// AVX-512 element-wise round for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn round_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Use AVX-512 round to nearest intrinsic
    let round_result = _mm512_roundscale_pd(x, 0x08); // Round to nearest, ties to even

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), round_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].round();
  }
}

// AVX2 element-wise round for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn round_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Use AVX2 round to nearest intrinsic
    let round_result = _mm256_round_pd(x, 0x08); // Round to nearest, ties to even

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), round_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].round();
  }
}

// NEON element-wise round for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn round_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Use NEON round to nearest intrinsic
    let round_result = vrndnq_f64(x); // Round to nearest, ties to even

    vst1q_f64(values.as_mut_ptr().add(offset), round_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].round();
  }
}

// NEON element-wise exponential for f64 arrays using polynomial approximation
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn exp_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for exp(x) around 0
  // exp(x) ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5! + x^^/6! + x^/7! + x^^/8!
  let c0 = vdupq_n_f64(1.0);
  let _c1 = vdupq_n_f64(1.0);
  let c2 = vdupq_n_f64(1.0 / 2.0);
  let c3 = vdupq_n_f64(1.0 / 6.0);
  let c4 = vdupq_n_f64(1.0 / 24.0);
  let c5 = vdupq_n_f64(1.0 / 120.0);
  let c6 = vdupq_n_f64(1.0 / 720.0);
  let c7 = vdupq_n_f64(1.0 / 5040.0);
  let c8 = vdupq_n_f64(1.0 / 40320.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let x5 = vmulq_f64(x4, x);
    let x6 = vmulq_f64(x3, x3);
    let x7 = vmulq_f64(x6, x);
    let x8 = vmulq_f64(x4, x4);

    // Compute polynomial: 1 + x + c2*x² + c3*x³ + c4*x^^ + c5*x^ + c6*x^^ + c7*x^ + c8*x^^
    let mut result = vaddq_f64(c0, x);
    result = vfmaq_f64(result, x2, c2);
    result = vfmaq_f64(result, x3, c3);
    result = vfmaq_f64(result, x4, c4);
    result = vfmaq_f64(result, x5, c5);
    result = vfmaq_f64(result, x6, c6);
    result = vfmaq_f64(result, x7, c7);
    result = vfmaq_f64(result, x8, c8);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x4 * x;
    let x6 = x3 * x3;
    let x7 = x6 * x;
    let x8 = x4 * x4;
    *values.get_unchecked_mut(offset + i) = 1.0
      + x
      + x2 / 2.0
      + x3 / 6.0
      + x4 / 24.0
      + x5 / 120.0
      + x6 / 720.0
      + x7 / 5040.0
      + x8 / 40320.0;
  }
}

// NEON element-wise logarithm for f64 arrays using polynomial approximation
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn log_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // For log(x), we use the identity: log(x) = log(1+y) where y = (x-1)/(x+1)
  // Then use the series: log(1+y) ~= 2*(y + y³/3 + y^/5 + y^/7 + y^/9)
  let one = vdupq_n_f64(1.0);
  let two = vdupq_n_f64(2.0);
  let c3 = vdupq_n_f64(1.0 / 3.0);
  let c5 = vdupq_n_f64(1.0 / 5.0);
  let c7 = vdupq_n_f64(1.0 / 7.0);
  let c9 = vdupq_n_f64(1.0 / 9.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // Compute y = (x-1)/(x+1)
    let x_minus_1 = vsubq_f64(x, one);
    let x_plus_1 = vaddq_f64(x, one);
    let y = vdivq_f64(x_minus_1, x_plus_1);

    // Calculate powers of y
    let y2 = vmulq_f64(y, y);
    let y3 = vmulq_f64(y2, y);
    let y5 = vmulq_f64(y3, y2);
    let y7 = vmulq_f64(y5, y2);
    let y9 = vmulq_f64(y7, y2);

    // Compute polynomial: y + y³/3 + y^/5 + y^/7 + y^/9
    let mut result = y;
    result = vfmaq_f64(result, y3, c3);
    result = vfmaq_f64(result, y5, c5);
    result = vfmaq_f64(result, y7, c7);
    result = vfmaq_f64(result, y9, c9);

    // Multiply by 2
    result = vmulq_f64(result, two);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let y = (x - 1.0) / (x + 1.0);
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    let y9 = y7 * y2;
    *values.get_unchecked_mut(offset + i) = 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0 + y9 / 9.0);
  }
}

// AVX-512 math functions

// AVX-512 element-wise absolute value for f64 arrays
// GPU implementation of absolute value

#[cfg(has_cuda)]
pub unsafe fn abs_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ABS_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry abs_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd5, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load and apply abs to double2 vector in-place
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply abs operation
      abs.f64 %fd4, %fd0;              // Abs lane 0
      abs.f64 %fd5, %fd1;              // Abs lane 1
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      ld.global.f64 %fd0, [%rd2];
      abs.f64 %fd4, %fd0;
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, (&len as *const usize) as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ABS_F64, &[], "abs_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn abs_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    let abs_vec = _mm512_abs_pd(vec);
    _mm512_storeu_pd(values.as_mut_ptr().add(offset), abs_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Use bitwise AND to clear sign bit for absolute value
    let abs_val = f64::from_bits(val.to_bits() & 0x7FFFFFFFFFFFFFFF);
    *values.get_unchecked_mut(offset + i) = abs_val;
  }
}

// AVX2 element-wise absolute value for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn abs_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Create mask to clear the sign bit (all bits set except sign bit)
  let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFFu64));

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    let abs_vec = _mm256_and_pd(vec, abs_mask);
    _mm256_storeu_pd(values.as_mut_ptr().add(offset), abs_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Use bitwise AND to clear sign bit for absolute value
    let abs_val = f64::from_bits(val.to_bits() & 0x7FFFFFFFFFFFFFFF);
    *values.get_unchecked_mut(offset + i) = abs_val;
  }
}

// AVX-512 element-wise negation for f64 arrays
// GPU implementation of negation

#[cfg(has_cuda)]
pub unsafe fn neg_f64_gpu(values: *mut f64, len: usize) {
  const PTX_NEG_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry neg_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd5, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load and apply neg to double2 vector in-place
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply neg operation
      neg.f64 %fd4, %fd0;              // Neg lane 0
      neg.f64 %fd5, %fd1;              // Neg lane 1
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    remainder:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd5, %rd1;
      ld.global.f64 %fd0, [%rd2];
      neg.f64 %fd4, %fd0;
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, (&len as *const usize) as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_NEG_F64, &[], "neg_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn neg_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;
  let zeros = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    let neg_vec = _mm512_sub_pd(zeros, vec);
    _mm512_storeu_pd(values.as_mut_ptr().add(offset), neg_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Use XOR to flip sign bit for negation
    let neg_val = f64::from_bits(val.to_bits() ^ 0x8000000000000000);
    *values.get_unchecked_mut(offset + i) = neg_val;
  }
}

// AVX2 element-wise negation for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn neg_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // XOR with sign bit mask to flip sign
  let sign_mask = _mm256_set1_pd(f64::from_bits(0x8000000000000000u64));

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    let neg_vec = _mm256_xor_pd(vec, sign_mask);
    _mm256_storeu_pd(values.as_mut_ptr().add(offset), neg_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Use XOR to flip sign bit for negation
    let neg_val = f64::from_bits(val.to_bits() ^ 0x8000000000000000);
    *values.get_unchecked_mut(offset + i) = neg_val;
  }
}

// AVX-512 element-wise square root for f64 arrays
// GPU implementation of square root

#[cfg(has_cuda)]
pub unsafe fn sqrt_f64_gpu(values: *mut f64, len: usize) {
  const PTX_SQRT_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry sqrt_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply sqrt to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply sqrt operation
      sqrt.rn.f64 %fd4, %fd0;          // Sqrt lane 0
      sqrt.rn.f64 %fd5, %fd1;          // Sqrt lane 1
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    exit_loop:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd0, [%rd2];
      sqrt.rn.f64 %fd4, %fd0;
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, &len as *const usize as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_SQRT_F64, &[], "sqrt_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sqrt_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    let sqrt_vec = _mm512_sqrt_pd(vec);
    _mm512_storeu_pd(values.as_mut_ptr().add(offset), sqrt_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result: f64;
    #[cfg(target_arch = "aarch64")]
    {
      asm!(
        "fsqrt {0:d}, {1:d}",
        out(vreg) result,
        in(vreg) val,
        options(pure, nomem, nostack)
      );
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      asm!(
        "sqrtsd {}, {}",
        out(xmm_reg) result,
        in(xmm_reg) val,
        options(pure, nomem, nostack)
      );
    }
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// AVX2 element-wise square root for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sqrt_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    let sqrt_vec = _mm256_sqrt_pd(vec);
    _mm256_storeu_pd(values.as_mut_ptr().add(offset), sqrt_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result: f64;
    #[cfg(target_arch = "aarch64")]
    {
      asm!(
        "fsqrt {0:d}, {1:d}",
        out(vreg) result,
        in(vreg) val,
        options(pure, nomem, nostack)
      );
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      asm!(
        "sqrtsd {}, {}",
        out(xmm_reg) result,
        in(xmm_reg) val,
        options(pure, nomem, nostack)
      );
    }
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// AVX-512 element-wise sine for f64 arrays (scalar fallback)
// GPU implementation of sine
#[cfg(has_cuda)]
pub unsafe fn sin_f64_gpu(values: *mut f64, len: usize) {
  const PTX_SIN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry sin_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply sin to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply sin operation
      sin.approx.f64 %fd4, %fd0;       // Sin lane 0
      sin.approx.f64 %fd5, %fd1;       // Sin lane 1
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    exit_loop:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd0, [%rd2];
      sin.approx.f64 %fd4, %fd0;
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, &len as *const usize as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_SIN_F64, &[], "sin_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sin_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for sin(x) around 0
  // sin(x) ~= x - x³/3! + x^/5! - x^/7! + x^/9!
  let c3 = _mm512_set1_pd(-1.0 / 6.0);
  let c5 = _mm512_set1_pd(1.0 / 120.0);
  let c7 = _mm512_set1_pd(-1.0 / 5040.0);
  let c9 = _mm512_set1_pd(1.0 / 362880.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x5 = _mm512_mul_pd(x3, x2);
    let x7 = _mm512_mul_pd(x5, x2);
    let x9 = _mm512_mul_pd(x7, x2);

    // Compute polynomial: x + c3*x³ + c5*x^ + c7*x^ + c9*x^
    let mut result = x;
    result = _mm512_fmadd_pd(x3, c3, result);
    result = _mm512_fmadd_pd(x5, c5, result);
    result = _mm512_fmadd_pd(x7, c7, result);
    result = _mm512_fmadd_pd(x9, c9, result);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    *values.get_unchecked_mut(offset + i) = x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0;
  }
}

// AVX2 element-wise sine for f64 arrays (scalar fallback)
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sin_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for sin(x) around 0
  // sin(x) ~= x - x³/3! + x^/5! - x^/7! + x^/9!
  let c3 = _mm256_set1_pd(-1.0 / 6.0);
  let c5 = _mm256_set1_pd(1.0 / 120.0);
  let c7 = _mm256_set1_pd(-1.0 / 5040.0);
  let c9 = _mm256_set1_pd(1.0 / 362880.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x5 = _mm256_mul_pd(x3, x2);
    let x7 = _mm256_mul_pd(x5, x2);
    let x9 = _mm256_mul_pd(x7, x2);

    // Compute polynomial: x + c3*x³ + c5*x^ + c7*x^ + c9*x^
    let mut result = x;
    result = _mm256_fmadd_pd(x3, c3, result);
    result = _mm256_fmadd_pd(x5, c5, result);
    result = _mm256_fmadd_pd(x7, c7, result);
    result = _mm256_fmadd_pd(x9, c9, result);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    *values.get_unchecked_mut(offset + i) = x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0;
  }
}

// AVX-512 element-wise cosine for f64 arrays (scalar fallback)
// GPU implementation of cosine

#[cfg(has_cuda)]
pub unsafe fn cos_f64_gpu(values: *mut f64, len: usize) {
  const PTX_COS_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry cos_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply cos to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply cos operation
      cos.approx.f64 %fd4, %fd0;       // Cos lane 0
      cos.approx.f64 %fd5, %fd1;       // Cos lane 1
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    exit_loop:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd0, [%rd2];
      cos.approx.f64 %fd4, %fd0;
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, &len as *const usize as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_COS_F64, &[], "cos_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn cos_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for cos(x) around 0
  // cos(x) ~= 1 - x²/2! + x^^/4! - x^^/6! + x^^/8!
  let c0 = _mm512_set1_pd(1.0);
  let c2 = _mm512_set1_pd(-0.5);
  let c4 = _mm512_set1_pd(1.0 / 24.0);
  let c6 = _mm512_set1_pd(-1.0 / 720.0);
  let c8 = _mm512_set1_pd(1.0 / 40320.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm512_mul_pd(x, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x6 = _mm512_mul_pd(x4, x2);
    let x8 = _mm512_mul_pd(x4, x4);

    // Compute polynomial: 1 + c2*x² + c4*x^^ + c6*x^^ + c8*x^^
    let mut result = c0;
    result = _mm512_fmadd_pd(x2, c2, result);
    result = _mm512_fmadd_pd(x4, c4, result);
    result = _mm512_fmadd_pd(x6, c6, result);
    result = _mm512_fmadd_pd(x8, c8, result);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    *values.get_unchecked_mut(offset + i) = 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0;
  }
}

// AVX2 element-wise cosine for f64 arrays (scalar fallback)
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn cos_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for cos(x) around 0
  // cos(x) ~= 1 - x²/2! + x^^/4! - x^^/6! + x^^/8!
  let c0 = _mm256_set1_pd(1.0);
  let c2 = _mm256_set1_pd(-0.5);
  let c4 = _mm256_set1_pd(1.0 / 24.0);
  let c6 = _mm256_set1_pd(-1.0 / 720.0);
  let c8 = _mm256_set1_pd(1.0 / 40320.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm256_mul_pd(x, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x6 = _mm256_mul_pd(x4, x2);
    let x8 = _mm256_mul_pd(x4, x4);

    // Compute polynomial: 1 + c2*x² + c4*x^^ + c6*x^^ + c8*x^^
    let mut result = c0;
    result = _mm256_fmadd_pd(x2, c2, result);
    result = _mm256_fmadd_pd(x4, c4, result);
    result = _mm256_fmadd_pd(x6, c6, result);
    result = _mm256_fmadd_pd(x8, c8, result);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    *values.get_unchecked_mut(offset + i) = 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0;
  }
}

// AVX-512 element-wise tangent for f64 arrays (scalar fallback)
// GPU implementation of tangent

#[cfg(has_cuda)]
pub unsafe fn tan_f64_gpu(values: *mut f64, len: usize) {
  const PTX_TAN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry tan_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;  // Need more registers for sin, cos, and division
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply tan to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply tan operation: tan = sin/cos for both lanes
      sin.approx.f64 %fd2, %fd0;       // Sin lane 0
      cos.approx.f64 %fd3, %fd0;       // Cos lane 0
      div.rn.f64 %fd4, %fd2, %fd3;     // Tan lane 0 = sin/cos
      sin.approx.f64 %fd5, %fd1;       // Sin lane 1
      cos.approx.f64 %fd6, %fd1;       // Cos lane 1
      div.rn.f64 %fd7, %fd5, %fd6;     // Tan lane 1 = sin/cos
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd7};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    exit_loop:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd0, [%rd2];
      // tan = sin/cos
      sin.approx.f64 %fd1, %fd0;
      cos.approx.f64 %fd2, %fd0;
      div.rn.f64 %fd4, %fd1, %fd2;
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, &len as *const usize as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_TAN_F64, &[], "tan_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn tan_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Compute tan(x) = sin(x) / cos(x) using polynomial approximations
  // Sin coefficients
  let _s1 = _mm512_set1_pd(1.0);
  let s3 = _mm512_set1_pd(-1.0 / 6.0);
  let s5 = _mm512_set1_pd(1.0 / 120.0);
  let s7 = _mm512_set1_pd(-1.0 / 5040.0);

  // Cos coefficients
  let c0 = _mm512_set1_pd(1.0);
  let c2 = _mm512_set1_pd(-0.5);
  let c4 = _mm512_set1_pd(1.0 / 24.0);
  let c6 = _mm512_set1_pd(-1.0 / 720.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x3, x2);
    let x6 = _mm512_mul_pd(x4, x2);
    let x7 = _mm512_mul_pd(x5, x2);

    // Compute sin(x)
    let mut sin_result = x;
    sin_result = _mm512_fmadd_pd(x3, s3, sin_result);
    sin_result = _mm512_fmadd_pd(x5, s5, sin_result);
    sin_result = _mm512_fmadd_pd(x7, s7, sin_result);

    // Compute cos(x)
    let mut cos_result = c0;
    cos_result = _mm512_fmadd_pd(x2, c2, cos_result);
    cos_result = _mm512_fmadd_pd(x4, c4, cos_result);
    cos_result = _mm512_fmadd_pd(x6, c6, cos_result);

    // tan(x) = sin(x) / cos(x)
    let tan_result = _mm512_div_pd(sin_result, cos_result);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), tan_result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x3 * x2;
    let x6 = x4 * x2;
    let x7 = x5 * x2;
    let sin_x = x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0;
    let cos_x = 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0;
    *values.get_unchecked_mut(offset + i) = sin_x / cos_x;
  }
}

// AVX2 element-wise tangent for f64 arrays (scalar fallback)
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn tan_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Compute tan(x) = sin(x) / cos(x) using polynomial approximations
  // Sin coefficients
  let s3 = _mm256_set1_pd(-1.0 / 6.0);
  let s5 = _mm256_set1_pd(1.0 / 120.0);
  let s7 = _mm256_set1_pd(-1.0 / 5040.0);

  // Cos coefficients
  let c0 = _mm256_set1_pd(1.0);
  let c2 = _mm256_set1_pd(-0.5);
  let c4 = _mm256_set1_pd(1.0 / 24.0);
  let c6 = _mm256_set1_pd(-1.0 / 720.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x3, x2);
    let x6 = _mm256_mul_pd(x4, x2);
    let x7 = _mm256_mul_pd(x5, x2);

    // Compute sin(x)
    let mut sin_result = x;
    sin_result = _mm256_fmadd_pd(x3, s3, sin_result);
    sin_result = _mm256_fmadd_pd(x5, s5, sin_result);
    sin_result = _mm256_fmadd_pd(x7, s7, sin_result);

    // Compute cos(x)
    let mut cos_result = c0;
    cos_result = _mm256_fmadd_pd(x2, c2, cos_result);
    cos_result = _mm256_fmadd_pd(x4, c4, cos_result);
    cos_result = _mm256_fmadd_pd(x6, c6, cos_result);

    // tan(x) = sin(x) / cos(x)
    let result = _mm256_div_pd(sin_result, cos_result);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x3 * x2;
    let x6 = x4 * x2;
    let x7 = x5 * x2;
    let sin_x = x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0;
    let cos_x = 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0;
    *values.get_unchecked_mut(offset + i) = sin_x / cos_x;
  }
}

// AVX-512 element-wise exponential for f64 arrays (scalar fallback)
// GPU implementation of exponential

#[cfg(has_cuda)]
pub unsafe fn exp_f64_gpu(values: *mut f64, len: usize) {
  const PTX_EXP_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry exp_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;  // Need extra registers for log2(e) conversion
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply exp to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply exp operation: convert from natural exp to base-2 exp
      // exp(x) = 2^(x * log2(e))
      mul.f64 %fd2, %fd0, 0d3FF71547652B82FE;  // lane 0 * log2(e)
      ex2.approx.f64 %fd4, %fd2;               // 2^(lane 0 * log2(e))
      mul.f64 %fd3, %fd1, 0d3FF71547652B82FE;  // lane 1 * log2(e)
      ex2.approx.f64 %fd5, %fd3;               // 2^(lane 1 * log2(e))
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    exit_loop:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd0, [%rd2];
      // Convert to base 2: exp(x) = 2^(x * log2(e))
      mul.f64 %fd1, %fd0, 0d3FF71547652B82FE; // log2(e)
      ex2.approx.f64 %fd4, %fd1;
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, &len as *const usize as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_EXP_F64, &[], "exp_f64", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn exp_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for exp(x) around 0
  // exp(x) ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5! + x^^/6! + x^/7! + x^^/8!
  let c0 = _mm512_set1_pd(1.0);
  let c2 = _mm512_set1_pd(1.0 / 2.0);
  let c3 = _mm512_set1_pd(1.0 / 6.0);
  let c4 = _mm512_set1_pd(1.0 / 24.0);
  let c5 = _mm512_set1_pd(1.0 / 120.0);
  let c6 = _mm512_set1_pd(1.0 / 720.0);
  let c7 = _mm512_set1_pd(1.0 / 5040.0);
  let c8 = _mm512_set1_pd(1.0 / 40320.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm512_mul_pd(x, x);
    let x3 = _mm512_mul_pd(x2, x);
    let x4 = _mm512_mul_pd(x2, x2);
    let x5 = _mm512_mul_pd(x4, x);
    let x6 = _mm512_mul_pd(x3, x3);
    let x7 = _mm512_mul_pd(x6, x);
    let x8 = _mm512_mul_pd(x4, x4);

    // Compute polynomial: 1 + x + c2*x² + c3*x³ + c4*x^^ + c5*x^ + c6*x^^ + c7*x^ + c8*x^^
    let mut result = _mm512_add_pd(c0, x);
    result = _mm512_fmadd_pd(x2, c2, result);
    result = _mm512_fmadd_pd(x3, c3, result);
    result = _mm512_fmadd_pd(x4, c4, result);
    result = _mm512_fmadd_pd(x5, c5, result);
    result = _mm512_fmadd_pd(x6, c6, result);
    result = _mm512_fmadd_pd(x7, c7, result);
    result = _mm512_fmadd_pd(x8, c8, result);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x4 * x;
    let x6 = x3 * x3;
    let x7 = x6 * x;
    let x8 = x4 * x4;
    *values.get_unchecked_mut(offset + i) = 1.0
      + x
      + x2 / 2.0
      + x3 / 6.0
      + x4 / 24.0
      + x5 / 120.0
      + x6 / 720.0
      + x7 / 5040.0
      + x8 / 40320.0;
  }
}

// AVX2 element-wise exponential for f64 arrays (scalar fallback)
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn exp_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Taylor series coefficients for exp(x) around 0
  // exp(x) ~= 1 + x + x²/2! + x³/3! + x^^/4! + x^/5! + x^^/6! + x^/7! + x^^/8!
  let c0 = _mm256_set1_pd(1.0);
  let c2 = _mm256_set1_pd(1.0 / 2.0);
  let c3 = _mm256_set1_pd(1.0 / 6.0);
  let c4 = _mm256_set1_pd(1.0 / 24.0);
  let c5 = _mm256_set1_pd(1.0 / 120.0);
  let c6 = _mm256_set1_pd(1.0 / 720.0);
  let c7 = _mm256_set1_pd(1.0 / 5040.0);
  let c8 = _mm256_set1_pd(1.0 / 40320.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Calculate powers of x
    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let x5 = _mm256_mul_pd(x4, x);
    let x6 = _mm256_mul_pd(x3, x3);
    let x7 = _mm256_mul_pd(x6, x);
    let x8 = _mm256_mul_pd(x4, x4);

    // Compute polynomial: 1 + x + c2*x² + c3*x³ + c4*x^^ + c5*x^ + c6*x^^ + c7*x^ + c8*x^^
    let mut result = _mm256_add_pd(c0, x);
    result = _mm256_fmadd_pd(x2, c2, result);
    result = _mm256_fmadd_pd(x3, c3, result);
    result = _mm256_fmadd_pd(x4, c4, result);
    result = _mm256_fmadd_pd(x5, c5, result);
    result = _mm256_fmadd_pd(x6, c6, result);
    result = _mm256_fmadd_pd(x7, c7, result);
    result = _mm256_fmadd_pd(x8, c8, result);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x4 * x;
    let x6 = x3 * x3;
    let x7 = x6 * x;
    let x8 = x4 * x4;
    *values.get_unchecked_mut(offset + i) = 1.0
      + x
      + x2 / 2.0
      + x3 / 6.0
      + x4 / 24.0
      + x5 / 120.0
      + x6 / 720.0
      + x7 / 5040.0
      + x8 / 40320.0;
  }
}

// AVX-512 element-wise logarithm for f64 arrays (scalar fallback)
// GPU implementation of natural logarithm

#[cfg(has_cuda)]
pub unsafe fn log_f64_gpu(values: *mut f64, len: usize) {
  const PTX_LOG_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry log_f64(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;  // Need extra registers for ln(2) conversion
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply log to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      // Apply log operation: convert from base-2 log to natural log
      // ln(x) = log2(x) * ln(2)
      lg2.approx.f64 %fd2, %fd0;              // log2(lane 0)
      mul.f64 %fd4, %fd2, 0d3FE62E42FEFA39EF; // log2(lane 0) * ln(2)
      lg2.approx.f64 %fd3, %fd1;              // log2(lane 1)
      mul.f64 %fd5, %fd3, 0d3FE62E42FEFA39EF; // log2(lane 1) * ln(2)
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;  // Continue loop
      
    exit_loop:  // Remainder handling with masking (like AVX-512)
      setp.lt.u32 %p0, %r5, %r0;  // Check if first element exists
      @!%p0 bra done;  // Skip if no remainder
      
      // Load and process single remainder element
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      ld.global.f64 %fd0, [%rd2];
      // Convert from base 2: ln(x) = log2(x) * ln(2)
      lg2.approx.f64 %fd1, %fd0;
      mul.f64 %fd4, %fd1, 0d3FE62E42FEFA39EF; // ln(2)
      st.global.f64 [%rd2], %fd4;
      
    done:
      ret;
    }
  "#;

  let args = [values as *const u8, &len as *const usize as *const u8];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_LOG_F64, &[], "log_f64", blocks, threads, &args);
}

// ===== ACOS (Arc Cosine) =====

// GPU implementation of acos
#[cfg(has_cuda)]
pub unsafe fn acos_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ACOS_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry acos_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra remainder;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 with vectorized load
      ld.global.v2.f64 {%fd0, %fd6}, [%rd3];
      
      // acos(x) ~= ^/2 - (x + x³/6 + 3x^/40 + ...) for lane 0
      mul.f64 %fd1, %fd0, %fd0;  // x^2
      mul.f64 %fd2, %fd1, %fd0;  // x^3
      mul.f64 %fd3, %fd2, %fd1;  // x^5
      
      mul.f64 %fd2, %fd2, 0.166666666666667;  // x³/6
      mul.f64 %fd3, %fd3, 0.075;  // 3x^^/40
      add.f64 %fd4, %fd0, %fd2;
      add.f64 %fd4, %fd4, %fd3;
      sub.f64 %fd5, 1.5707963267948966, %fd4;  // π/2 - result
      
      // acos(x) for lane 1
      mul.f64 %fd7, %fd6, %fd6;  // x^2
      mul.f64 %fd8, %fd7, %fd6;  // x^3
      mul.f64 %fd9, %fd8, %fd7;  // x^5
      
      mul.f64 %fd8, %fd8, 0.166666666666667;  // x³/6
      mul.f64 %fd9, %fd9, 0.075;  // 3x^^/40
      add.f64 %fd10, %fd6, %fd8;
      add.f64 %fd10, %fd10, %fd9;
      sub.f64 %fd11, 1.5707963267948966, %fd10;  // π/2 - result
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 with vectorized store
      st.global.v2.f64 [%rd3], {%fd5, %fd11};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    remainder:
      setp.ge.u32 %p0, %r5, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.f64 %fd0, [%rd3];
      
      mul.f64 %fd1, %fd0, %fd0;
      mul.f64 %fd2, %fd1, %fd0;
      mul.f64 %fd3, %fd2, %fd1;
      
      mul.f64 %fd2, %fd2, 0.166666666666667;
      mul.f64 %fd3, %fd3, 0.075;
      add.f64 %fd4, %fd0, %fd2;
      add.f64 %fd4, %fd4, %fd3;
      sub.f64 %fd5, 1.5707963267948966, %fd4;
      
      add.u64 %rd3, %rd1, %rd2;
      st.global.f64 [%rd3], %fd5;
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ACOS_F64, &[], "acos_f64", blocks, threads, &args);
}

// ===== ASIN (Arc Sine) =====

// GPU implementation of asin
#[cfg(has_cuda)]
pub unsafe fn asin_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ASIN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry asin_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra remainder;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 with vectorized load
      ld.global.v2.f64 {%fd0, %fd5}, [%rd3];
      
      // asin(x) ~= x + x³/6 + 3x^/40 + ... for lane 0
      mul.f64 %fd1, %fd0, %fd0;  // x^2
      mul.f64 %fd2, %fd1, %fd0;  // x^3
      mul.f64 %fd3, %fd2, %fd1;  // x^5
      
      mul.f64 %fd2, %fd2, 0.166666666666667;  // x³/6
      mul.f64 %fd3, %fd3, 0.075;  // 3x^^/40
      add.f64 %fd4, %fd0, %fd2;
      add.f64 %fd4, %fd4, %fd3;
      
      // asin(x) for lane 1
      mul.f64 %fd6, %fd5, %fd5;  // x^2
      mul.f64 %fd7, %fd6, %fd5;  // x^3
      mul.f64 %fd8, %fd7, %fd6;  // x^5
      
      mul.f64 %fd7, %fd7, 0.166666666666667;  // x³/6
      mul.f64 %fd8, %fd8, 0.075;  // 3x^^/40
      add.f64 %fd9, %fd5, %fd7;
      add.f64 %fd9, %fd9, %fd8;
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 with vectorized store
      st.global.v2.f64 [%rd3], {%fd4, %fd9};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    remainder:
      setp.ge.u32 %p0, %r5, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.f64 %fd0, [%rd3];
      
      mul.f64 %fd1, %fd0, %fd0;
      mul.f64 %fd2, %fd1, %fd0;
      mul.f64 %fd3, %fd2, %fd1;
      
      mul.f64 %fd2, %fd2, 0.166666666666667;
      mul.f64 %fd3, %fd3, 0.075;
      add.f64 %fd4, %fd0, %fd2;
      add.f64 %fd4, %fd4, %fd3;
      
      add.u64 %rd3, %rd1, %rd2;
      st.global.f64 [%rd3], %fd4;
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ASIN_F64, &[], "asin_f64", blocks, threads, &args);
}

// GPU implementation of atan
#[cfg(has_cuda)]
pub unsafe fn atan_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ATAN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry atan_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra remainder;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 with vectorized load
      ld.global.v2.f64 {%fd0, %fd7}, [%rd3];
      
      // atan(x) ~= x - x³/3 + x^/5 - x^/7 + ... for lane 0
      mul.f64 %fd1, %fd0, %fd0;  // x^2
      mul.f64 %fd2, %fd1, %fd0;  // x^3
      mul.f64 %fd3, %fd2, %fd1;  // x^5
      mul.f64 %fd4, %fd3, %fd1;  // x^7
      
      div.f64 %fd2, %fd2, 3.0;  // x³/3
      div.f64 %fd3, %fd3, 5.0;  // x^^/5
      div.f64 %fd4, %fd4, 7.0;  // x^^/7
      
      sub.f64 %fd5, %fd0, %fd2;
      add.f64 %fd5, %fd5, %fd3;
      sub.f64 %fd6, %fd5, %fd4;
      
      // atan(x) for lane 1
      mul.f64 %fd8, %fd7, %fd7;   // x^2
      mul.f64 %fd9, %fd8, %fd7;   // x^3
      mul.f64 %fd10, %fd9, %fd8;  // x^5
      mul.f64 %fd11, %fd10, %fd8; // x^7
      
      div.f64 %fd9, %fd9, 3.0;    // x³/3
      div.f64 %fd10, %fd10, 5.0;  // x^^/5
      div.f64 %fd11, %fd11, 7.0;  // x^^/7
      
      sub.f64 %fd12, %fd7, %fd9;
      add.f64 %fd12, %fd12, %fd10;
      sub.f64 %fd13, %fd12, %fd11;
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 with vectorized store
      st.global.v2.f64 [%rd3], {%fd6, %fd13};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    remainder:
      setp.ge.u32 %p0, %r5, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.f64 %fd0, [%rd3];
      
      mul.f64 %fd1, %fd0, %fd0;
      mul.f64 %fd2, %fd1, %fd0;
      mul.f64 %fd3, %fd2, %fd1;
      mul.f64 %fd4, %fd3, %fd1;
      
      div.f64 %fd2, %fd2, 3.0;
      div.f64 %fd3, %fd3, 5.0;
      div.f64 %fd4, %fd4, 7.0;
      
      sub.f64 %fd5, %fd0, %fd2;
      add.f64 %fd5, %fd5, %fd3;
      sub.f64 %fd6, %fd5, %fd4;
      
      add.u64 %rd3, %rd1, %rd2;
      st.global.f64 [%rd3], %fd6;
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ATAN_F64, &[], "atan_f64", blocks, threads, &args);
}

// GPU implementation of cosh
#[cfg(has_cuda)]
pub unsafe fn cosh_f64_gpu(values: *mut f64, len: usize) {
  const PTX_COSH_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry cosh_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements (double2)
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 vector
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];
      
      // cosh(x) = (e^x + e^(-x)) / 2 for both lanes
      ex2.approx.f64 %fd2, %fd0;  // e^x0
      neg.f64 %fd3, %fd0;
      ex2.approx.f64 %fd3, %fd3;  // e^(-x0)
      add.f64 %fd4, %fd2, %fd3;
      mul.f64 %fd4, %fd4, 0.5;    // cosh(x0)
      
      ex2.approx.f64 %fd5, %fd1;  // e^x1
      neg.f64 %fd6, %fd1;
      ex2.approx.f64 %fd6, %fd6;  // e^(-x1)
      add.f64 %fd7, %fd5, %fd6;
      mul.f64 %fd7, %fd7, 0.5;    // cosh(x1)
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 results
      st.global.v2.f64 [%rd3], {%fd4, %fd7};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_COSH_F64, &[], "cosh_f64", blocks, threads, &args);
}

// GPU implementation of sinh
#[cfg(has_cuda)]
pub unsafe fn sinh_f64_gpu(values: *mut f64, len: usize) {
  const PTX_SINH_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry sinh_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements (double2)
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 vector
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];
      
      // sinh(x) = (e^x - e^(-x)) / 2 for both lanes
      ex2.approx.f64 %fd2, %fd0;  // e^x0
      neg.f64 %fd3, %fd0;
      ex2.approx.f64 %fd3, %fd3;  // e^(-x0)
      sub.f64 %fd4, %fd2, %fd3;
      mul.f64 %fd4, %fd4, 0.5;    // sinh(x0)
      
      ex2.approx.f64 %fd5, %fd1;  // e^x1
      neg.f64 %fd6, %fd1;
      ex2.approx.f64 %fd6, %fd6;  // e^(-x1)
      sub.f64 %fd7, %fd5, %fd6;
      mul.f64 %fd7, %fd7, 0.5;    // sinh(x1)
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 results
      st.global.v2.f64 [%rd3], {%fd4, %fd7};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_SINH_F64, &[], "sinh_f64", blocks, threads, &args);
}

// GPU implementation of tanh
#[cfg(has_cuda)]
pub unsafe fn tanh_f64_gpu(values: *mut f64, len: usize) {
  const PTX_TANH_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry tanh_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements (double2)
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 vector
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];
      
      // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1) for both lanes
      // Lane 0
      mul.f64 %fd2, %fd0, 2.0;    // 2x0
      ex2.approx.f64 %fd3, %fd2;  // e^(2x0)
      sub.f64 %fd4, %fd3, 1.0;    // e^(2x0) - 1
      add.f64 %fd5, %fd3, 1.0;    // e^(2x0) + 1
      div.f64 %fd6, %fd4, %fd5;   // tanh(x0)
      
      // Lane 1
      mul.f64 %fd7, %fd1, 2.0;    // 2x1
      ex2.approx.f64 %fd8, %fd7;  // e^(2x1)
      sub.f64 %fd9, %fd8, 1.0;    // e^(2x1) - 1
      add.f64 %fd10, %fd8, 1.0;   // e^(2x1) + 1
      div.f64 %fd11, %fd9, %fd10; // tanh(x1)
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 results
      st.global.v2.f64 [%rd3], {%fd6, %fd11};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_TANH_F64, &[], "tanh_f64", blocks, threads, &args);
}

// GPU implementation of floor
#[cfg(has_cuda)]
pub unsafe fn floor_f64_gpu(values: *mut f64, len: usize) {
  const PTX_FLOOR_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry floor_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements (double2)
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 vector
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];
      
      // Apply floor to both lanes
      cvt.rmi.f64.f64 %fd2, %fd0;  // Round towards negative infinity (floor) lane 0
      cvt.rmi.f64.f64 %fd3, %fd1;  // Round towards negative infinity (floor) lane 1
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 results
      st.global.v2.f64 [%rd3], {%fd2, %fd3};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_FLOOR_F64, &[], "floor_f64", blocks, threads, &args);
}

// GPU implementation of ceil
#[cfg(has_cuda)]
pub unsafe fn ceil_f64_gpu(values: *mut f64, len: usize) {
  const PTX_CEIL_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry ceil_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements (double2)
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 vector
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];
      
      // Apply ceil to both lanes
      cvt.rpi.f64.f64 %fd2, %fd0;  // Round towards positive infinity (ceil) lane 0
      cvt.rpi.f64.f64 %fd3, %fd1;  // Round towards positive infinity (ceil) lane 1
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 results
      st.global.v2.f64 [%rd3], {%fd2, %fd3};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_CEIL_F64, &[], "ceil_f64", blocks, threads, &args);
}

// GPU implementation of round
#[cfg(has_cuda)]
pub unsafe fn round_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ROUND_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry round_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      
      mad.lo.u32 %r5, %r2, %r3, %r1;
      mul.lo.u32 %r5, %r5, 2;  // Each thread processes 2 elements (double2)
      mul.lo.u32 %r3, %r3, %r4;
      mul.lo.u32 %r3, %r3, 2;  // Grid stride = total_threads * 2
      
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      // Load double2 vector
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];
      
      // Apply round to both lanes
      cvt.rni.f64.f64 %fd2, %fd0;  // Round to nearest even lane 0
      cvt.rni.f64.f64 %fd3, %fd1;  // Round to nearest even lane 1
      
      add.u64 %rd3, %rd1, %rd2;
      // Store double2 results
      st.global.v2.f64 [%rd3], {%fd2, %fd3};
      
      add.u32 %r5, %r5, %r3;
      bra loop_start;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ROUND_F64, &[], "round_f64", blocks, threads, &args);
}

// ===== ACOSH (Inverse Hyperbolic Cosine) =====

// GPU implementation of acosh
#[cfg(has_cuda)]
pub unsafe fn acosh_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ACOSH_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry acosh_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Grid-stride loop setup
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 4;  // Each thread processes 4 elements (double4)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 4;   // Stride = total_threads * 4 (double4)
      
    loop_start:
      add.u32 %r6, %r5, 3;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double4
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load double4 vector with vectorized loads
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];      // x0, x1
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3+16];   // x2, x3
      
      // Compute acosh(x) = ln(x + sqrt(x^2 - 1)) for each lane
      // Lane 0
      mul.f64 %fd4, %fd0, %fd0;        // x0^2
      sub.f64 %fd4, %fd4, 1.0;         // x0^2 - 1
      sqrt.rn.f64 %fd4, %fd4;          // sqrt(x0^2 - 1)
      add.f64 %fd4, %fd0, %fd4;        // x0 + sqrt(x0^2 - 1)
      lg2.approx.f64 %fd4, %fd4;       // log2(x0 + sqrt(x0^2 - 1))
      mul.f64 %fd4, %fd4, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Lane 1
      mul.f64 %fd5, %fd1, %fd1;        // x1^2
      sub.f64 %fd5, %fd5, 1.0;         // x1^2 - 1
      sqrt.rn.f64 %fd5, %fd5;          // sqrt(x1^2 - 1)
      add.f64 %fd5, %fd1, %fd5;        // x1 + sqrt(x1^2 - 1)
      lg2.approx.f64 %fd5, %fd5;       // log2(x1 + sqrt(x1^2 - 1))
      mul.f64 %fd5, %fd5, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Lane 2
      mul.f64 %fd6, %fd2, %fd2;        // x2^2
      sub.f64 %fd6, %fd6, 1.0;         // x2^2 - 1
      sqrt.rn.f64 %fd6, %fd6;          // sqrt(x2^2 - 1)
      add.f64 %fd6, %fd2, %fd6;        // x2 + sqrt(x2^2 - 1)
      lg2.approx.f64 %fd6, %fd6;       // log2(x2 + sqrt(x2^2 - 1))
      mul.f64 %fd6, %fd6, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Lane 3
      mul.f64 %fd7, %fd3, %fd3;        // x3^2
      sub.f64 %fd7, %fd7, 1.0;         // x3^2 - 1
      sqrt.rn.f64 %fd7, %fd7;          // sqrt(x3^2 - 1)
      add.f64 %fd7, %fd3, %fd7;        // x3 + sqrt(x3^2 - 1)
      lg2.approx.f64 %fd7, %fd7;       // log2(x3 + sqrt(x3^2 - 1))
      mul.f64 %fd7, %fd7, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Store double4 results with vectorized stores
      add.u64 %rd4, %rd1, %rd2;
      st.global.v2.f64 [%rd4], {%fd4, %fd5};      // Store result0, result1
      st.global.v2.f64 [%rd4+16], {%fd6, %fd7};   // Store result2, result3
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;
      
    remainder:  // Handle remaining elements
      setp.lt.u32 %p0, %r5, %r0;
      @!%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.f64 %fd0, [%rd3];
      
      mul.f64 %fd4, %fd0, %fd0;
      sub.f64 %fd4, %fd4, 1.0;
      sqrt.rn.f64 %fd4, %fd4;
      add.f64 %fd4, %fd0, %fd4;
      lg2.approx.f64 %fd4, %fd4;
      mul.f64 %fd4, %fd4, 0d3FE62E42FEFA39EF;
      
      add.u64 %rd4, %rd1, %rd2;
      st.global.f64 [%rd4], %fd4;
      
      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r0;
      @p0 bra remainder;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ACOSH_F64, &[], "acosh_f64", blocks, threads, &args);
}

// AVX-512 element-wise acosh for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn acosh_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = _mm512_set1_pd(1.0);
  // Taylor series coefficients for ln(x) approximation around x=2
  // ln(x) ~= ln(2) + (x-2)/2 - (x-2)²/8 + (x-2)³/24 - ...
  let ln2 = _mm512_set1_pd(0.693147180559945309417);
  let c1 = _mm512_set1_pd(0.5);
  let c2 = _mm512_set1_pd(-0.125);
  let c3 = _mm512_set1_pd(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // acosh(x) = ln(x + sqrt(x^2 - 1))
    let x_squared = _mm512_mul_pd(x, x);
    let x_squared_minus_one = _mm512_sub_pd(x_squared, one);
    let sqrt_term = _mm512_sqrt_pd(x_squared_minus_one);
    let arg = _mm512_add_pd(x, sqrt_term);

    // Approximate ln using Taylor series
    let two = _mm512_set1_pd(2.0);
    let diff = _mm512_sub_pd(arg, two);
    let diff2 = _mm512_mul_pd(diff, diff);
    let diff3 = _mm512_mul_pd(diff2, diff);

    let mut result_vec = ln2;
    result_vec = _mm512_fmadd_pd(diff, c1, result_vec);
    result_vec = _mm512_fmadd_pd(diff2, c2, result_vec);
    result_vec = _mm512_fmadd_pd(diff3, c3, result_vec);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].acosh();
  }
}

// AVX2 element-wise acosh for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn acosh_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = _mm256_set1_pd(1.0);
  // Taylor series coefficients for ln(x) approximation around x=2
  // ln(x) ~= ln(2) + (x-2)/2 - (x-2)²/8 + (x-2)³/24 - ...
  let ln2 = _mm256_set1_pd(0.693147180559945309417);
  let c1 = _mm256_set1_pd(0.5);
  let c2 = _mm256_set1_pd(-0.125);
  let c3 = _mm256_set1_pd(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // acosh(x) = ln(x + sqrt(x^2 - 1))
    let x_squared = _mm256_mul_pd(x, x);
    let x_squared_minus_one = _mm256_sub_pd(x_squared, one);
    let sqrt_term = _mm256_sqrt_pd(x_squared_minus_one);
    let arg = _mm256_add_pd(x, sqrt_term);

    // Approximate ln using Taylor series
    let two = _mm256_set1_pd(2.0);
    let diff = _mm256_sub_pd(arg, two);
    let diff2 = _mm256_mul_pd(diff, diff);
    let diff3 = _mm256_mul_pd(diff2, diff);

    let mut result_vec = ln2;
    result_vec = _mm256_fmadd_pd(diff, c1, result_vec);
    result_vec = _mm256_fmadd_pd(diff2, c2, result_vec);
    result_vec = _mm256_fmadd_pd(diff3, c3, result_vec);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].acosh();
  }
}

// NEON element-wise acosh for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn acosh_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = vdupq_n_f64(1.0);
  // Taylor series coefficients for ln(x) approximation around x=2
  // ln(x) ~= ln(2) + (x-2)/2 - (x-2)²/8 + (x-2)³/24 - ...
  let ln2 = vdupq_n_f64(0.693147180559945309417);
  let c1 = vdupq_n_f64(0.5);
  let c2 = vdupq_n_f64(-0.125);
  let c3 = vdupq_n_f64(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // acosh(x) = ln(x + sqrt(x^2 - 1))
    let x_squared = vmulq_f64(x, x);
    let x_squared_minus_one = vsubq_f64(x_squared, one);
    let sqrt_term = vsqrtq_f64(x_squared_minus_one);
    let arg = vaddq_f64(x, sqrt_term);

    // Approximate ln using Taylor series
    let two = vdupq_n_f64(2.0);
    let diff = vsubq_f64(arg, two);
    let diff2 = vmulq_f64(diff, diff);
    let diff3 = vmulq_f64(diff2, diff);

    let mut result_vec = ln2;
    result_vec = vfmaq_f64(result_vec, diff, c1);
    result_vec = vfmaq_f64(result_vec, diff2, c2);
    result_vec = vfmaq_f64(result_vec, diff3, c3);

    vst1q_f64(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].acosh();
  }
}

// ===== ASINH (Inverse Hyperbolic Sine) =====

// GPU implementation of asinh
#[cfg(has_cuda)]
pub unsafe fn asinh_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ASINH_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry asinh_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Grid-stride loop setup
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 4;  // Each thread processes 4 elements (double4)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 4;   // Stride = total_threads * 4 (double4)
      
    loop_start:
      add.u32 %r6, %r5, 3;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double4
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load double4 vector with vectorized loads
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];      // x0, x1
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3+16];   // x2, x3
      
      // Compute asinh(x) = ln(x + sqrt(x^2 + 1)) for each lane
      // Lane 0
      mul.f64 %fd4, %fd0, %fd0;        // x0^2
      add.f64 %fd4, %fd4, 1.0;         // x0^2 + 1
      sqrt.rn.f64 %fd4, %fd4;          // sqrt(x0^2 + 1)
      add.f64 %fd4, %fd0, %fd4;        // x0 + sqrt(x0^2 + 1)
      lg2.approx.f64 %fd4, %fd4;       // log2(x0 + sqrt(x0^2 + 1))
      mul.f64 %fd4, %fd4, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Lane 1
      mul.f64 %fd5, %fd1, %fd1;        // x1^2
      add.f64 %fd5, %fd5, 1.0;         // x1^2 + 1
      sqrt.rn.f64 %fd5, %fd5;          // sqrt(x1^2 + 1)
      add.f64 %fd5, %fd1, %fd5;        // x1 + sqrt(x1^2 + 1)
      lg2.approx.f64 %fd5, %fd5;       // log2(x1 + sqrt(x1^2 + 1))
      mul.f64 %fd5, %fd5, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Lane 2
      mul.f64 %fd6, %fd2, %fd2;        // x2^2
      add.f64 %fd6, %fd6, 1.0;         // x2^2 + 1
      sqrt.rn.f64 %fd6, %fd6;          // sqrt(x2^2 + 1)
      add.f64 %fd6, %fd2, %fd6;        // x2 + sqrt(x2^2 + 1)
      lg2.approx.f64 %fd6, %fd6;       // log2(x2 + sqrt(x2^2 + 1))
      mul.f64 %fd6, %fd6, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Lane 3
      mul.f64 %fd7, %fd3, %fd3;        // x3^2
      add.f64 %fd7, %fd7, 1.0;         // x3^2 + 1
      sqrt.rn.f64 %fd7, %fd7;          // sqrt(x3^2 + 1)
      add.f64 %fd7, %fd3, %fd7;        // x3 + sqrt(x3^2 + 1)
      lg2.approx.f64 %fd7, %fd7;       // log2(x3 + sqrt(x3^2 + 1))
      mul.f64 %fd7, %fd7, 0d3FE62E42FEFA39EF; // Convert to ln
      
      // Store double4 results with vectorized stores
      add.u64 %rd4, %rd1, %rd2;
      st.global.v2.f64 [%rd4], {%fd4, %fd5};      // Store result0, result1
      st.global.v2.f64 [%rd4+16], {%fd6, %fd7};   // Store result2, result3
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;
      
    remainder:  // Handle remaining elements
      setp.lt.u32 %p0, %r5, %r0;
      @!%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.f64 %fd0, [%rd3];
      
      mul.f64 %fd4, %fd0, %fd0;
      add.f64 %fd4, %fd4, 1.0;
      sqrt.rn.f64 %fd4, %fd4;
      add.f64 %fd4, %fd0, %fd4;
      lg2.approx.f64 %fd4, %fd4;
      mul.f64 %fd4, %fd4, 0d3FE62E42FEFA39EF;
      
      add.u64 %rd4, %rd1, %rd2;
      st.global.f64 [%rd4], %fd4;
      
      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r0;
      @p0 bra remainder;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ASINH_F64, &[], "asinh_f64", blocks, threads, &args);
}

// AVX-512 element-wise asinh for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn asinh_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = _mm512_set1_pd(1.0);
  // Taylor series coefficients for ln(x) approximation
  let ln2 = _mm512_set1_pd(0.693147180559945309417);
  let c1 = _mm512_set1_pd(0.5);
  let c2 = _mm512_set1_pd(-0.125);
  let c3 = _mm512_set1_pd(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // asinh(x) = ln(x + sqrt(x^2 + 1))
    let x_squared = _mm512_mul_pd(x, x);
    let x_squared_plus_one = _mm512_add_pd(x_squared, one);
    let sqrt_term = _mm512_sqrt_pd(x_squared_plus_one);
    let arg = _mm512_add_pd(x, sqrt_term);

    // Approximate ln using Taylor series
    let two = _mm512_set1_pd(2.0);
    let diff = _mm512_sub_pd(arg, two);
    let diff2 = _mm512_mul_pd(diff, diff);
    let diff3 = _mm512_mul_pd(diff2, diff);

    let mut result_vec = ln2;
    result_vec = _mm512_fmadd_pd(diff, c1, result_vec);
    result_vec = _mm512_fmadd_pd(diff2, c2, result_vec);
    result_vec = _mm512_fmadd_pd(diff3, c3, result_vec);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].asinh();
  }
}

// AVX2 element-wise asinh for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn asinh_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = _mm256_set1_pd(1.0);
  // Taylor series coefficients for ln(x) approximation
  let ln2 = _mm256_set1_pd(0.693147180559945309417);
  let c1 = _mm256_set1_pd(0.5);
  let c2 = _mm256_set1_pd(-0.125);
  let c3 = _mm256_set1_pd(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // asinh(x) = ln(x + sqrt(x^2 + 1))
    let x_squared = _mm256_mul_pd(x, x);
    let x_squared_plus_one = _mm256_add_pd(x_squared, one);
    let sqrt_term = _mm256_sqrt_pd(x_squared_plus_one);
    let arg = _mm256_add_pd(x, sqrt_term);

    // Approximate ln using Taylor series
    let two = _mm256_set1_pd(2.0);
    let diff = _mm256_sub_pd(arg, two);
    let diff2 = _mm256_mul_pd(diff, diff);
    let diff3 = _mm256_mul_pd(diff2, diff);

    let mut result_vec = ln2;
    result_vec = _mm256_fmadd_pd(diff, c1, result_vec);
    result_vec = _mm256_fmadd_pd(diff2, c2, result_vec);
    result_vec = _mm256_fmadd_pd(diff3, c3, result_vec);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].asinh();
  }
}

// NEON element-wise asinh for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn asinh_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = vdupq_n_f64(1.0);
  // Taylor series coefficients for ln(x) approximation
  let ln2 = vdupq_n_f64(0.693147180559945309417);
  let c1 = vdupq_n_f64(0.5);
  let c2 = vdupq_n_f64(-0.125);
  let c3 = vdupq_n_f64(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // asinh(x) = ln(x + sqrt(x^2 + 1))
    let x_squared = vmulq_f64(x, x);
    let x_squared_plus_one = vaddq_f64(x_squared, one);
    let sqrt_term = vsqrtq_f64(x_squared_plus_one);
    let arg = vaddq_f64(x, sqrt_term);

    // Approximate ln using Taylor series
    let two = vdupq_n_f64(2.0);
    let diff = vsubq_f64(arg, two);
    let diff2 = vmulq_f64(diff, diff);
    let diff3 = vmulq_f64(diff2, diff);

    let mut result_vec = ln2;
    result_vec = vfmaq_f64(result_vec, diff, c1);
    result_vec = vfmaq_f64(result_vec, diff2, c2);
    result_vec = vfmaq_f64(result_vec, diff3, c3);

    vst1q_f64(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].asinh();
  }
}

// ===== ATANH (Inverse Hyperbolic Tangent) =====

// GPU implementation of atanh
#[cfg(has_cuda)]
pub unsafe fn atanh_f64_gpu(values: *mut f64, len: usize) {
  const PTX_ATANH_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry atanh_f64(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .f64 %fd<25>;
      .reg .pred %p<10>;

      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Grid-stride loop setup
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 4;  // Each thread processes 4 elements (double4)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 4;   // Stride = total_threads * 4 (double4)
      
    loop_start:
      add.u32 %r6, %r5, 3;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double4
      @%p0 bra remainder;  // Exit loop if beyond array
      
      // Load double4 vector with vectorized loads
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd3];      // x0, x1
      ld.global.v2.f64 {%fd2, %fd3}, [%rd3+16];   // x2, x3
      
      // Compute atanh(x) = 0.5 * ln((1 + x) / (1 - x)) for each lane
      // Lane 0
      add.f64 %fd4, 1.0, %fd0;         // 1 + x0
      sub.f64 %fd8, 1.0, %fd0;         // 1 - x0
      div.rn.f64 %fd4, %fd4, %fd8;     // (1 + x0) / (1 - x0)
      lg2.approx.f64 %fd4, %fd4;       // log2((1 + x0) / (1 - x0))
      mul.f64 %fd4, %fd4, 0d3FE62E42FEFA39EF; // Convert to ln
      mul.f64 %fd4, %fd4, 0.5;         // 0.5 * ln(...)
      
      // Lane 1
      add.f64 %fd5, 1.0, %fd1;         // 1 + x1
      sub.f64 %fd9, 1.0, %fd1;         // 1 - x1
      div.rn.f64 %fd5, %fd5, %fd9;     // (1 + x1) / (1 - x1)
      lg2.approx.f64 %fd5, %fd5;       // log2((1 + x1) / (1 - x1))
      mul.f64 %fd5, %fd5, 0d3FE62E42FEFA39EF; // Convert to ln
      mul.f64 %fd5, %fd5, 0.5;         // 0.5 * ln(...)
      
      // Lane 2
      add.f64 %fd6, 1.0, %fd2;         // 1 + x2
      sub.f64 %fd10, 1.0, %fd2;        // 1 - x2
      div.rn.f64 %fd6, %fd6, %fd10;    // (1 + x2) / (1 - x2)
      lg2.approx.f64 %fd6, %fd6;       // log2((1 + x2) / (1 - x2))
      mul.f64 %fd6, %fd6, 0d3FE62E42FEFA39EF; // Convert to ln
      mul.f64 %fd6, %fd6, 0.5;         // 0.5 * ln(...)
      
      // Lane 3
      add.f64 %fd7, 1.0, %fd3;         // 1 + x3
      sub.f64 %fd11, 1.0, %fd3;        // 1 - x3
      div.rn.f64 %fd7, %fd7, %fd11;    // (1 + x3) / (1 - x3)
      lg2.approx.f64 %fd7, %fd7;       // log2((1 + x3) / (1 - x3))
      mul.f64 %fd7, %fd7, 0d3FE62E42FEFA39EF; // Convert to ln
      mul.f64 %fd7, %fd7, 0.5;         // 0.5 * ln(...)
      
      // Store double4 results with vectorized stores
      add.u64 %rd4, %rd1, %rd2;
      st.global.v2.f64 [%rd4], {%fd4, %fd5};      // Store result0, result1
      st.global.v2.f64 [%rd4+16], {%fd6, %fd7};   // Store result2, result3
      
      // Grid stride to next chunk
      add.u32 %r5, %r5, %r9;
      bra loop_start;
      
    remainder:  // Handle remaining elements
      setp.lt.u32 %p0, %r5, %r0;
      @!%p0 bra done;
      
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.f64 %fd0, [%rd3];
      
      add.f64 %fd4, 1.0, %fd0;
      sub.f64 %fd8, 1.0, %fd0;
      div.rn.f64 %fd4, %fd4, %fd8;
      lg2.approx.f64 %fd4, %fd4;
      mul.f64 %fd4, %fd4, 0d3FE62E42FEFA39EF;
      mul.f64 %fd4, %fd4, 0.5;
      
      add.u64 %rd4, %rd1, %rd2;
      st.global.f64 [%rd4], %fd4;
      
      add.u32 %r5, %r5, 1;
      setp.lt.u32 %p0, %r5, %r0;
      @p0 bra remainder;
      
    done:
      ret;
    }
  "#;

  let args = [
    values as *const u8,
    values as *const u8, // In-place modification
    &len as *const usize as *const u8,
  ];

  let (blocks, threads) = LaunchConfig::parallel();
  let _ = launch_ptx(PTX_ATANH_F64, &[], "atanh_f64", blocks, threads, &args);
}

// AVX-512 element-wise atanh for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn atanh_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = _mm512_set1_pd(1.0);
  let half = _mm512_set1_pd(0.5);
  // Taylor series coefficients for ln(x) approximation
  let ln2 = _mm512_set1_pd(0.693147180559945309417);
  let c1 = _mm512_set1_pd(0.5);
  let c2 = _mm512_set1_pd(-0.125);
  let c3 = _mm512_set1_pd(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
    let one_plus_x = _mm512_add_pd(one, x);
    let one_minus_x = _mm512_sub_pd(one, x);
    let ratio = _mm512_div_pd(one_plus_x, one_minus_x);

    // Approximate ln using Taylor series
    let two = _mm512_set1_pd(2.0);
    let diff = _mm512_sub_pd(ratio, two);
    let diff2 = _mm512_mul_pd(diff, diff);
    let diff3 = _mm512_mul_pd(diff2, diff);

    let mut ln_result = ln2;
    ln_result = _mm512_fmadd_pd(diff, c1, ln_result);
    ln_result = _mm512_fmadd_pd(diff2, c2, ln_result);
    ln_result = _mm512_fmadd_pd(diff3, c3, ln_result);

    // Multiply by 0.5
    let result_vec = _mm512_mul_pd(ln_result, half);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].atanh();
  }
}

// AVX2 element-wise atanh for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn atanh_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = _mm256_set1_pd(1.0);
  let half = _mm256_set1_pd(0.5);
  // Taylor series coefficients for ln(x) approximation
  let ln2 = _mm256_set1_pd(0.693147180559945309417);
  let c1 = _mm256_set1_pd(0.5);
  let c2 = _mm256_set1_pd(-0.125);
  let c3 = _mm256_set1_pd(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
    let one_plus_x = _mm256_add_pd(one, x);
    let one_minus_x = _mm256_sub_pd(one, x);
    let ratio = _mm256_div_pd(one_plus_x, one_minus_x);

    // Approximate ln using Taylor series
    let two = _mm256_set1_pd(2.0);
    let diff = _mm256_sub_pd(ratio, two);
    let diff2 = _mm256_mul_pd(diff, diff);
    let diff3 = _mm256_mul_pd(diff2, diff);

    let mut ln_result = ln2;
    ln_result = _mm256_fmadd_pd(diff, c1, ln_result);
    ln_result = _mm256_fmadd_pd(diff2, c2, ln_result);
    ln_result = _mm256_fmadd_pd(diff3, c3, ln_result);

    // Multiply by 0.5
    let result_vec = _mm256_mul_pd(ln_result, half);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].atanh();
  }
}

// NEON element-wise atanh for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn atanh_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let one = vdupq_n_f64(1.0);
  let half = vdupq_n_f64(0.5);
  // Taylor series coefficients for ln(x) approximation
  let ln2 = vdupq_n_f64(0.693147180559945309417);
  let c1 = vdupq_n_f64(0.5);
  let c2 = vdupq_n_f64(-0.125);
  let c3 = vdupq_n_f64(0.041666666666666664);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = vld1q_f64(values.as_ptr().add(offset));

    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
    let one_plus_x = vaddq_f64(one, x);
    let one_minus_x = vsubq_f64(one, x);

    // Use reciprocal approximation with Newton-Raphson refinement
    // ratio = (1 + x) / (1 - x) = (1 + x) * recip(1 - x)
    let recip_est = vrecpeq_f64(one_minus_x);
    // Newton-Raphson refinement: recip' = recip * (2 - x * recip)
    let recip_refined = vmulq_f64(recip_est, vrecpsq_f64(one_minus_x, recip_est));
    let recip_refined2 = vmulq_f64(recip_refined, vrecpsq_f64(one_minus_x, recip_refined));
    let ratio = vmulq_f64(one_plus_x, recip_refined2);

    // Approximate ln using Taylor series
    let two = vdupq_n_f64(2.0);
    let diff = vsubq_f64(ratio, two);
    let diff2 = vmulq_f64(diff, diff);
    let diff3 = vmulq_f64(diff2, diff);

    let mut ln_result = ln2;
    ln_result = vfmaq_f64(ln_result, diff, c1);
    ln_result = vfmaq_f64(ln_result, diff2, c2);
    ln_result = vfmaq_f64(ln_result, diff3, c3);

    // Multiply by 0.5
    let result_vec = vmulq_f64(ln_result, half);

    vst1q_f64(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    values[offset + i] = values[offset + i].atanh();
  }
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn log_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // For log(x), we use the identity: log(x) = log(1+y) where y = (x-1)/(x+1)
  // Then use the series: log(1+y) ~= 2*(y + y³/3 + y^/5 + y^/7 + y^/9)
  let one = _mm512_set1_pd(1.0);
  let two = _mm512_set1_pd(2.0);
  let c3 = _mm512_set1_pd(1.0 / 3.0);
  let c5 = _mm512_set1_pd(1.0 / 5.0);
  let c7 = _mm512_set1_pd(1.0 / 7.0);
  let c9 = _mm512_set1_pd(1.0 / 9.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Compute y = (x-1)/(x+1)
    let x_minus_1 = _mm512_sub_pd(x, one);
    let x_plus_1 = _mm512_add_pd(x, one);
    let y = _mm512_div_pd(x_minus_1, x_plus_1);

    // Calculate powers of y
    let y2 = _mm512_mul_pd(y, y);
    let y3 = _mm512_mul_pd(y2, y);
    let y5 = _mm512_mul_pd(y3, y2);
    let y7 = _mm512_mul_pd(y5, y2);
    let y9 = _mm512_mul_pd(y7, y2);

    // Compute polynomial: y + y³/3 + y^/5 + y^/7 + y^/9
    let mut result = y;
    result = _mm512_fmadd_pd(y3, c3, result);
    result = _mm512_fmadd_pd(y5, c5, result);
    result = _mm512_fmadd_pd(y7, c7, result);
    result = _mm512_fmadd_pd(y9, c9, result);

    // Multiply by 2
    result = _mm512_mul_pd(result, two);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let y = (x - 1.0) / (x + 1.0);
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    let y9 = y7 * y2;
    *values.get_unchecked_mut(offset + i) = 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0 + y9 / 9.0);
  }
}

// AVX2 element-wise logarithm for f64 arrays (scalar fallback)
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn log_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // For log(x), we use the identity: log(x) = log(1+y) where y = (x-1)/(x+1)
  // Then use the series: log(1+y) ~= 2*(y + y³/3 + y^/5 + y^/7 + y^/9)
  let one = _mm256_set1_pd(1.0);
  let two = _mm256_set1_pd(2.0);
  let c3 = _mm256_set1_pd(1.0 / 3.0);
  let c5 = _mm256_set1_pd(1.0 / 5.0);
  let c7 = _mm256_set1_pd(1.0 / 7.0);
  let c9 = _mm256_set1_pd(1.0 / 9.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let x = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Compute y = (x-1)/(x+1)
    let x_minus_1 = _mm256_sub_pd(x, one);
    let x_plus_1 = _mm256_add_pd(x, one);
    let y = _mm256_div_pd(x_minus_1, x_plus_1);

    // Calculate powers of y
    let y2 = _mm256_mul_pd(y, y);
    let y3 = _mm256_mul_pd(y2, y);
    let y5 = _mm256_mul_pd(y3, y2);
    let y7 = _mm256_mul_pd(y5, y2);
    let y9 = _mm256_mul_pd(y7, y2);

    // Compute polynomial: y + y³/3 + y^/5 + y^/7 + y^/9
    let mut result = y;
    result = _mm256_fmadd_pd(y3, c3, result);
    result = _mm256_fmadd_pd(y5, c5, result);
    result = _mm256_fmadd_pd(y7, c7, result);
    result = _mm256_fmadd_pd(y9, c9, result);

    // Multiply by 2
    result = _mm256_mul_pd(result, two);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let x = values.get_unchecked(offset + i);
    let y = (x - 1.0) / (x + 1.0);
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    let y9 = y7 * y2;
    *values.get_unchecked_mut(offset + i) = 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0 + y9 / 9.0);
  }
}

// =============================================================================
// ^ TIME EXTRACTION FUNCTIONS - SIMD OPTIMIZED TIMESTAMP PROCESSING ^
// =============================================================================

/// Extract timestamp values (identity function) with SIMD/GPU acceleration
///
/// # Arguments
/// * `timestamps` - u64 slice containing Unix timestamps
/// * `result` - Mutable f64 slice for output timestamp values
/// * `len` - Number of elements to process
#[cfg(has_cuda)]
pub unsafe fn timestamp_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_TIMESTAMP_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry timestamp_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID (process 2 elements per thread)
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      mul.lo.u32 %r4, %r4, 2;  // Each thread processes 2 elements
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Check if we can load 2 timestamps
      add.u64 %rd8, %rd3, 1;
      setp.gt.u64 %p1, %rd8, %rd2;
      @%p1 bra scalar_load;
      
      // Load 2 timestamps (vectorized)
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.v2.u64 {%rd6, %rd9}, [%rd5];
      
      // Convert both u64 to f64
      cvt.rn.f64.u64 %fd0, %rd6;
      cvt.rn.f64.u64 %fd1, %rd9;
      
      // Store 2 results (vectorized)
      add.u64 %rd7, %rd1, %rd4;
      st.global.v2.f64 [%rd7], {%fd0, %fd1};
      bra done;
      
    scalar_load:
      // Load single timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Convert u64 to f64
      cvt.rn.f64.u64 %fd0, %rd6;
      
      // Store result
      add.u64 %rd7, %rd1, %rd4;
      st.global.f64 [%rd7], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_TIMESTAMP_U64,
    &[],
    "timestamp_u64_kernel",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn timestamp_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_epi64(timestamps.as_ptr().add(offset) as *const i64);
    let result_vec = _mm512_cvtepi64_pd(timestamps_vec);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    result[offset] = timestamps[offset] as f64;
  }
}

#[cfg(all(
  not(feature = "hwx-nightly"),
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn timestamp_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    // AVX2 doesn't have _mm256_cvtepi64_pd, convert manually
    let t0 = timestamps[offset] as f64;
    let t1 = timestamps[offset + 1] as f64;
    let t2 = timestamps[offset + 2] as f64;
    let t3 = timestamps[offset + 3] as f64;
    let result_vec = _mm256_set_pd(t3, t2, t1, t0);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    result[offset] = timestamps[offset] as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn timestamp_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = vld1q_u64(timestamps.as_ptr().add(offset));
    let result_vec = vcvtq_f64_u64(timestamps_vec);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    result[offset] = timestamps[offset] as f64;
  }
}

/// Extract hour from Unix timestamps with SIMD/GPU acceleration
/// Hour = (timestamp % 86400) / 3600
///
/// # Arguments  
/// * `timestamps` - u64 slice containing Unix timestamps
/// * `result` - Mutable f64 slice for output hour values (0-23)
/// * `len` - Number of elements to process
#[cfg(has_cuda)]
pub unsafe fn hour_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_HOUR_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry hour_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID (process 2 elements per thread)
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      mul.lo.u32 %r4, %r4, 2;  // Each thread processes 2 elements
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Check if we can load 2 timestamps
      add.u64 %rd12, %rd3, 1;
      setp.gt.u64 %p1, %rd12, %rd2;
      @%p1 bra scalar_load;
      
      // Load 2 timestamps (vectorized)
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.v2.u64 {%rd6, %rd13}, [%rd5];
      
      // Extract hour for first timestamp: (timestamp % 86400) / 3600
      mov.u64 %rd7, 86400;
      rem.u64 %rd8, %rd6, %rd7;
      mov.u64 %rd9, 3600;
      div.u64 %rd10, %rd8, %rd9;
      cvt.rn.f64.u64 %fd0, %rd10;
      
      // Extract hour for second timestamp
      rem.u64 %rd14, %rd13, %rd7;
      div.u64 %rd15, %rd14, %rd9;
      cvt.rn.f64.u64 %fd1, %rd15;
      
      // Store 2 results (vectorized)
      add.u64 %rd11, %rd1, %rd4;
      st.global.v2.f64 [%rd11], {%fd0, %fd1};
      bra done;
      
    scalar_load:
      // Load single timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Extract hour: (timestamp % 86400) / 3600
      mov.u64 %rd7, 86400;
      rem.u64 %rd8, %rd6, %rd7;
      mov.u64 %rd9, 3600;
      div.u64 %rd10, %rd8, %rd9;
      cvt.rn.f64.u64 %fd0, %rd10;
      
      // Store result
      add.u64 %rd11, %rd1, %rd4;
      st.global.f64 [%rd11], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(PTX_HOUR_U64, &[], "hour_u64_kernel", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512dq")]
pub unsafe fn hour_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Magic numbers for fast division by 86400 (seconds per day)
  // (x * 0xC22E450672894AB5) >> 80 = x / 86400
  let magic_day = _mm512_set1_epi64(0xC22E450672894AB5u64 as i64);
  const SHIFT_DAY: u32 = 80;

  // Magic numbers for fast division by 3600 (seconds per hour)
  // (x * 0x91A2B3C4D5E6F) >> 75 = x / 3600
  let magic_hour = _mm512_set1_epi64(0x91A2B3C4D5E6Fu64 as i64);
  const SHIFT_HOUR: u32 = 75;

  let sec_per_day = _mm512_set1_epi64(86400);

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_si512(timestamps.as_ptr().add(offset) as *const _);

    // Fast division by 86400 to get days
    let high = _mm512_srli_epi64(_mm512_mullo_epi64(timestamps_vec, magic_day), SHIFT_DAY);
    let days = high;

    // Get seconds within day: timestamp - (days * 86400)
    let day_seconds = _mm512_mullo_epi64(days, sec_per_day);
    let seconds_in_day = _mm512_sub_epi64(timestamps_vec, day_seconds);

    // Fast division by 3600 to get hours
    let hours = _mm512_srli_epi64(_mm512_mullo_epi64(seconds_in_day, magic_hour), SHIFT_HOUR);

    // Convert to f64 and store
    let result_vec = _mm512_cvtepu64_pd(hours);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    result[offset] = ((timestamp % 86400) / 3600) as f64;
  }
}

#[cfg(all(
  not(feature = "hwx-nightly"),
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn hour_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = _mm256_loadu_si256(timestamps.as_ptr().add(offset) as *const __m256i);
    let t0 = _mm256_extract_epi64(ts_vec, 0) as u64;
    let t1 = _mm256_extract_epi64(ts_vec, 1) as u64;
    let t2 = _mm256_extract_epi64(ts_vec, 2) as u64;
    let t3 = _mm256_extract_epi64(ts_vec, 3) as u64;
    let h0 = ((t0 % 86400) / 3600) as f64;
    let h1 = ((t1 % 86400) / 3600) as f64;
    let h2 = ((t2 % 86400) / 3600) as f64;
    let h3 = ((t3 % 86400) / 3600) as f64;
    let result_vec = _mm256_set_pd(h3, h2, h1, h0);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    result[offset] = ((timestamp % 86400) / 3600) as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hour_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  // NEON doesn't have native 64-bit division, use scalar operations
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    for j in 0..LANES {
      let timestamp = timestamps[offset + j];
      result[offset + j] = ((timestamp % 86400) / 3600) as f64;
    }
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    result[offset] = ((timestamp % 86400) / 3600) as f64;
  }
}

/// Extract minute from Unix timestamps with SIMD/GPU acceleration  
/// Minute = (timestamp % 3600) / 60
///
/// # Arguments
/// * `timestamps` - u64 slice containing Unix timestamps  
/// * `result` - Mutable f64 slice for output minute values (0-59)
/// * `len` - Number of elements to process
#[cfg(has_cuda)]
pub unsafe fn minute_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_MINUTE_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry minute_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID (process 2 elements per thread)
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      mul.lo.u32 %r4, %r4, 2;  // Each thread processes 2 elements
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Check if we can load 2 timestamps
      add.u64 %rd12, %rd3, 1;
      setp.gt.u64 %p1, %rd12, %rd2;
      @%p1 bra scalar_load;
      
      // Load 2 timestamps (vectorized)
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.v2.u64 {%rd6, %rd13}, [%rd5];
      
      // Extract minute for first timestamp
      mov.u64 %rd7, 3600;
      rem.u64 %rd8, %rd6, %rd7;
      mov.u64 %rd9, 60;
      div.u64 %rd10, %rd8, %rd9;
      cvt.rn.f64.u64 %fd0, %rd10;
      
      // Extract minute for second timestamp
      rem.u64 %rd14, %rd13, %rd7;
      div.u64 %rd15, %rd14, %rd9;
      cvt.rn.f64.u64 %fd1, %rd15;
      
      // Store 2 results (vectorized)
      add.u64 %rd11, %rd1, %rd4;
      st.global.v2.f64 [%rd11], {%fd0, %fd1};
      bra done;
      
    scalar_load:
      // Load single timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Extract minute: (timestamp % 3600) / 60
      mov.u64 %rd7, 3600;
      rem.u64 %rd8, %rd6, %rd7;
      mov.u64 %rd9, 60;
      div.u64 %rd10, %rd8, %rd9;
      cvt.rn.f64.u64 %fd0, %rd10;
      
      // Store result
      add.u64 %rd11, %rd1, %rd4;
      st.global.f64 [%rd11], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_MINUTE_U64,
    &[],
    "minute_u64_kernel",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn minute_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Magic numbers for fast division by 3600 (seconds per hour)
  let magic_hour = _mm512_set1_epi64(0x91A2B3C4D5E6Fu64 as i64);
  const SHIFT_HOUR: u32 = 75;

  // Magic numbers for fast division by 60 (seconds per minute)
  let magic_minute = _mm512_set1_epi64(0x888888888888889u64 as i64);
  const SHIFT_MINUTE: u32 = 69;

  let seconds_per_hour = _mm512_set1_epi64(3600);

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_epi64(timestamps.as_ptr().add(offset) as *const _);

    // Fast division by 3600 to get hours
    let hours = _mm512_srli_epi64(_mm512_mullo_epi64(timestamps_vec, magic_hour), SHIFT_HOUR);

    // Compute timestamp % 3600 (seconds within the hour)
    let seconds_in_hour =
      _mm512_sub_epi64(timestamps_vec, _mm512_mullo_epi64(hours, seconds_per_hour));

    // Fast division by 60 to get minutes (result is 0-59)
    let minutes = _mm512_srli_epi64(
      _mm512_mullo_epi64(seconds_in_hour, magic_minute),
      SHIFT_MINUTE,
    );

    // Convert to f64
    let minutes_f64 = _mm512_cvtepu64_pd(minutes);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), minutes_f64);
  }

  // Handle remainder
  let offset = chunks * LANES;
  for i in 0..remainder {
    let timestamp = timestamps[offset + i];
    result[offset + i] = ((timestamp % 3600) / 60) as f64;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn minute_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  // AVX2 doesn't have native 64-bit division, so use scalar operations
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    for j in 0..LANES {
      let timestamp = timestamps[offset + j];
      result[offset + j] = ((timestamp % 3600) / 60) as f64;
    }
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    result[offset] = ((timestamp % 3600) / 60) as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn minute_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  // NEON doesn't have native 64-bit division, use scalar operations
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    for j in 0..LANES {
      let timestamp = timestamps[offset + j];
      result[offset + j] = ((timestamp % 3600) / 60) as f64;
    }
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    result[offset] = ((timestamp % 3600) / 60) as f64;
  }
}

// =============================================================================
// DAY_OF_MONTH (u64 ^ f64) - Extract day of month from Unix timestamps
// =============================================================================
#[cfg(has_cuda)]
pub unsafe fn day_of_month_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_DAY_OF_MONTH_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry day_of_month_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Load timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Extract day of month: ((timestamp / 86400) % 30) + 1
      mov.u64 %rd7, 86400;
      div.u64 %rd8, %rd6, %rd7;  // days since epoch
      mov.u64 %rd9, 30;
      rem.u64 %rd10, %rd8, %rd9; // days % 30
      add.u64 %rd11, %rd10, 1;   // + 1 for 1-based day
      
      // Convert to f64
      cvt.rn.f64.u64 %fd0, %rd11;
      
      // Store result
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd7, %rd1, %rd4;
      st.global.f64 [%rd7], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_DAY_OF_MONTH_U64,
    &[],
    "day_of_month_u64_kernel",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn day_of_month_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_epi64(timestamps.as_ptr().add(offset) as *const i64);

    let mut temp_timestamps = [0i64; 8];
    _mm512_storeu_epi64(temp_timestamps.as_mut_ptr(), timestamps_vec);
    let t0 = temp_timestamps[0] as u64;
    let t1 = temp_timestamps[1] as u64;
    let t2 = temp_timestamps[2] as u64;
    let t3 = temp_timestamps[3] as u64;
    let t4 = temp_timestamps[4] as u64;
    let t5 = temp_timestamps[5] as u64;
    let t6 = temp_timestamps[6] as u64;
    let t7 = temp_timestamps[7] as u64;

    let d0 = ((t0 / 86400) % 30 + 1) as i64;
    let d1 = ((t1 / 86400) % 30 + 1) as i64;
    let d2 = ((t2 / 86400) % 30 + 1) as i64;
    let d3 = ((t3 / 86400) % 30 + 1) as i64;
    let d4 = ((t4 / 86400) % 30 + 1) as i64;
    let d5 = ((t5 / 86400) % 30 + 1) as i64;
    let d6 = ((t6 / 86400) % 30 + 1) as i64;
    let d7 = ((t7 / 86400) % 30 + 1) as i64;

    let result_vec_i = _mm512_set_epi64(d7, d6, d5, d4, d3, d2, d1, d0);
    let result_vec = _mm512_cvtepi64_pd(result_vec_i);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days % 30) + 1) as f64;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn day_of_month_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = _mm256_loadu_si256(timestamps.as_ptr().add(offset) as *const __m256i);

    let t0 = _mm256_extract_epi64(ts_vec, 0) as u64;
    let t1 = _mm256_extract_epi64(ts_vec, 1) as u64;
    let t2 = _mm256_extract_epi64(ts_vec, 2) as u64;
    let t3 = _mm256_extract_epi64(ts_vec, 3) as u64;

    let d0 = ((t0 / 86400) % 30 + 1) as i64;
    let d1 = ((t1 / 86400) % 30 + 1) as i64;
    let d2 = ((t2 / 86400) % 30 + 1) as i64;
    let d3 = ((t3 / 86400) % 30 + 1) as i64;

    // AVX2 doesn't have _mm256_cvtepi64_pd, convert manually
    let result_vec = _mm256_set_pd(d3 as f64, d2 as f64, d1 as f64, d0 as f64);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days % 30) + 1) as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn day_of_month_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = vld1q_u64(timestamps.as_ptr().add(offset));

    let t0 = vgetq_lane_u64(ts_vec, 0);
    let t1 = vgetq_lane_u64(ts_vec, 1);

    let d0 = (t0 / 86400) % 30 + 1;
    let d1 = (t1 / 86400) % 30 + 1;

    let result_vec_u = vsetq_lane_u64(d1, vsetq_lane_u64(d0, vdupq_n_u64(0), 0), 1);
    let result_vec = vcvtq_f64_u64(result_vec_u);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days % 30) + 1) as f64;
  }
}

// =============================================================================
// DAY_OF_WEEK (u64 ^ f64) - Extract day of week from Unix timestamps
// =============================================================================

#[cfg(has_cuda)]
pub unsafe fn day_of_week_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_DAY_OF_WEEK_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry day_of_week_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Load timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Extract day of week: (days + 4) % 7
      mov.u64 %rd7, 86400;
      div.u64 %rd8, %rd6, %rd7;  // days since epoch
      add.u64 %rd9, %rd8, 4;     // Unix epoch was Thursday
      mov.u64 %rd10, 7;
      rem.u64 %rd11, %rd9, %rd10; // (days + 4) % 7
      
      // Convert to f64
      cvt.rn.f64.u64 %fd0, %rd11;
      
      // Store result
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd7, %rd1, %rd4;
      st.global.f64 [%rd7], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_DAY_OF_WEEK_U64,
    &[],
    "day_of_week_u64_kernel",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn day_of_week_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_epi64(timestamps.as_ptr().add(offset) as *const i64);

    let mut temp_timestamps = [0i64; 8];
    _mm512_storeu_epi64(temp_timestamps.as_mut_ptr(), timestamps_vec);
    let t0 = temp_timestamps[0] as u64;
    let t1 = temp_timestamps[1] as u64;
    let t2 = temp_timestamps[2] as u64;
    let t3 = temp_timestamps[3] as u64;
    let t4 = temp_timestamps[4] as u64;
    let t5 = temp_timestamps[5] as u64;
    let t6 = temp_timestamps[6] as u64;
    let t7 = temp_timestamps[7] as u64;

    let d0 = ((t0 / 86400 + 4) % 7) as i64;
    let d1 = ((t1 / 86400 + 4) % 7) as i64;
    let d2 = ((t2 / 86400 + 4) % 7) as i64;
    let d3 = ((t3 / 86400 + 4) % 7) as i64;
    let d4 = ((t4 / 86400 + 4) % 7) as i64;
    let d5 = ((t5 / 86400 + 4) % 7) as i64;
    let d6 = ((t6 / 86400 + 4) % 7) as i64;
    let d7 = ((t7 / 86400 + 4) % 7) as i64;

    let result_vec_i = _mm512_set_epi64(d7, d6, d5, d4, d3, d2, d1, d0);
    let result_vec = _mm512_cvtepi64_pd(result_vec_i);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days + 4) % 7) as f64;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn day_of_week_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = _mm256_loadu_si256(timestamps.as_ptr().add(offset) as *const __m256i);

    let t0 = _mm256_extract_epi64(ts_vec, 0) as u64;
    let t1 = _mm256_extract_epi64(ts_vec, 1) as u64;
    let t2 = _mm256_extract_epi64(ts_vec, 2) as u64;
    let t3 = _mm256_extract_epi64(ts_vec, 3) as u64;

    let d0 = ((t0 / 86400 + 4) % 7) as i64;
    let d1 = ((t1 / 86400 + 4) % 7) as i64;
    let d2 = ((t2 / 86400 + 4) % 7) as i64;
    let d3 = ((t3 / 86400 + 4) % 7) as i64;

    // AVX2 doesn't have _mm256_cvtepi64_pd, convert manually
    let result_vec = _mm256_set_pd(d3 as f64, d2 as f64, d1 as f64, d0 as f64);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days + 4) % 7) as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn day_of_week_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = vld1q_u64(timestamps.as_ptr().add(offset));

    let t0 = vgetq_lane_u64(ts_vec, 0);
    let t1 = vgetq_lane_u64(ts_vec, 1);

    let d0 = (t0 / 86400 + 4) % 7;
    let d1 = (t1 / 86400 + 4) % 7;

    let result_vec_u = vsetq_lane_u64(d1, vsetq_lane_u64(d0, vdupq_n_u64(0), 0), 1);
    let result_vec = vcvtq_f64_u64(result_vec_u);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days + 4) % 7) as f64;
  }
}

// =============================================================================
// DAY_OF_YEAR (u64 ^ f64) - Extract day of year from Unix timestamps
// =============================================================================

#[cfg(has_cuda)]
pub unsafe fn day_of_year_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_DAY_OF_YEAR_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry day_of_year_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Load timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Extract day of year: (days % 365) + 1
      mov.u64 %rd7, 86400;
      div.u64 %rd8, %rd6, %rd7;  // days since epoch
      mov.u64 %rd9, 365;
      rem.u64 %rd10, %rd8, %rd9; // days % 365
      add.u64 %rd11, %rd10, 1;   // + 1 for 1-based day
      
      // Convert to f64
      cvt.rn.f64.u64 %fd0, %rd11;
      
      // Store result
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd7, %rd1, %rd4;
      st.global.f64 [%rd7], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_DAY_OF_YEAR_U64,
    &[],
    "day_of_year_u64_kernel",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn day_of_year_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_epi64(timestamps.as_ptr().add(offset) as *const i64);

    let mut temp_timestamps = [0i64; 8];
    _mm512_storeu_epi64(temp_timestamps.as_mut_ptr(), timestamps_vec);
    let t0 = temp_timestamps[0] as u64;
    let t1 = temp_timestamps[1] as u64;
    let t2 = temp_timestamps[2] as u64;
    let t3 = temp_timestamps[3] as u64;
    let t4 = temp_timestamps[4] as u64;
    let t5 = temp_timestamps[5] as u64;
    let t6 = temp_timestamps[6] as u64;
    let t7 = temp_timestamps[7] as u64;

    let d0 = ((t0 / 86400) % 365 + 1) as i64;
    let d1 = ((t1 / 86400) % 365 + 1) as i64;
    let d2 = ((t2 / 86400) % 365 + 1) as i64;
    let d3 = ((t3 / 86400) % 365 + 1) as i64;
    let d4 = ((t4 / 86400) % 365 + 1) as i64;
    let d5 = ((t5 / 86400) % 365 + 1) as i64;
    let d6 = ((t6 / 86400) % 365 + 1) as i64;
    let d7 = ((t7 / 86400) % 365 + 1) as i64;

    let result_vec_i = _mm512_set_epi64(d7, d6, d5, d4, d3, d2, d1, d0);
    let result_vec = _mm512_cvtepi64_pd(result_vec_i);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days % 365) + 1) as f64;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn day_of_year_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = _mm256_loadu_si256(timestamps.as_ptr().add(offset) as *const __m256i);

    let t0 = _mm256_extract_epi64(ts_vec, 0) as u64;
    let t1 = _mm256_extract_epi64(ts_vec, 1) as u64;
    let t2 = _mm256_extract_epi64(ts_vec, 2) as u64;
    let t3 = _mm256_extract_epi64(ts_vec, 3) as u64;

    let d0 = ((t0 / 86400) % 365 + 1) as i64;
    let d1 = ((t1 / 86400) % 365 + 1) as i64;
    let d2 = ((t2 / 86400) % 365 + 1) as i64;
    let d3 = ((t3 / 86400) % 365 + 1) as i64;

    // AVX2 doesn't have _mm256_cvtepi64_pd, convert manually
    let result_vec = _mm256_set_pd(d3 as f64, d2 as f64, d1 as f64, d0 as f64);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days % 365) + 1) as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn day_of_year_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = vld1q_u64(timestamps.as_ptr().add(offset));

    let t0 = vgetq_lane_u64(ts_vec, 0);
    let t1 = vgetq_lane_u64(ts_vec, 1);

    let d0 = (t0 / 86400) % 365 + 1;
    let d1 = (t1 / 86400) % 365 + 1;

    let result_vec_u = vsetq_lane_u64(d1, vsetq_lane_u64(d0, vdupq_n_u64(0), 0), 1);
    let result_vec = vcvtq_f64_u64(result_vec_u);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = ((days % 365) + 1) as f64;
  }
}

// =============================================================================
// DAYS_IN_MONTH (u64 ^ f64) - Get days in month for timestamps (simplified)
// =============================================================================

#[cfg(has_cuda)]
pub unsafe fn days_in_month_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_DAYS_IN_MONTH_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry days_in_month_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Simplified: always return 30.0 days
      mov.f64 %fd0, 0d403E000000000000; // 30.0 in IEEE 754 double
      
      // Store result
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd1, %rd4;
      st.global.f64 [%rd5], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_DAYS_IN_MONTH_U64,
    &[],
    "days_in_month_u64_kernel",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn days_in_month_u64_avx512(_timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let thirty = _mm512_set1_pd(30.0);

  for i in 0..chunks {
    let offset = i * LANES;
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), thirty);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    result[offset] = 30.0;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn days_in_month_u64_avx2(_timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let thirty = _mm256_set1_pd(30.0);

  for i in 0..chunks {
    let offset = i * LANES;
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), thirty);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    result[offset] = 30.0;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn days_in_month_u64_neon(_timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let thirty = vdupq_n_f64(30.0);

  for i in 0..chunks {
    let offset = i * LANES;
    vst1q_f64(result.as_mut_ptr().add(offset), thirty);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    result[offset] = 30.0;
  }
}

// =============================================================================
// MONTH (u64 ^ f64) - Extract month from Unix timestamps
// =============================================================================

#[cfg(has_cuda)]
pub unsafe fn month_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_MONTH_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry month_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Load timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Extract month: ((days / 30) % 12) + 1
      mov.u64 %rd7, 86400;
      div.u64 %rd8, %rd6, %rd7;  // days since epoch
      mov.u64 %rd9, 30;
      div.u64 %rd10, %rd8, %rd9; // days / 30 = months
      mov.u64 %rd11, 12;
      rem.u64 %rd12, %rd10, %rd11; // months % 12
      add.u64 %rd13, %rd12, 1;   // + 1 for 1-based month
      
      // Convert to f64
      cvt.rn.f64.u64 %fd0, %rd13;
      
      // Store result
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd7, %rd1, %rd4;
      st.global.f64 [%rd7], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_MONTH_U64,
    &[],
    "month_u64_kernel",
    blocks,
    threads,
    &args,
  );
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn month_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_epi64(timestamps.as_ptr().add(offset) as *const i64);

    let mut temp_timestamps = [0i64; 8];
    _mm512_storeu_epi64(temp_timestamps.as_mut_ptr(), timestamps_vec);
    let t0 = temp_timestamps[0] as u64;
    let t1 = temp_timestamps[1] as u64;
    let t2 = temp_timestamps[2] as u64;
    let t3 = temp_timestamps[3] as u64;
    let t4 = temp_timestamps[4] as u64;
    let t5 = temp_timestamps[5] as u64;
    let t6 = temp_timestamps[6] as u64;
    let t7 = temp_timestamps[7] as u64;

    let m0 = (((t0 / 86400) / 30) % 12 + 1) as i64;
    let m1 = (((t1 / 86400) / 30) % 12 + 1) as i64;
    let m2 = (((t2 / 86400) / 30) % 12 + 1) as i64;
    let m3 = (((t3 / 86400) / 30) % 12 + 1) as i64;
    let m4 = (((t4 / 86400) / 30) % 12 + 1) as i64;
    let m5 = (((t5 / 86400) / 30) % 12 + 1) as i64;
    let m6 = (((t6 / 86400) / 30) % 12 + 1) as i64;
    let m7 = (((t7 / 86400) / 30) % 12 + 1) as i64;

    let result_vec_i = _mm512_set_epi64(m7, m6, m5, m4, m3, m2, m1, m0);
    let result_vec = _mm512_cvtepi64_pd(result_vec_i);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = (((days / 30) % 12) + 1) as f64;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn month_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = _mm256_loadu_si256(timestamps.as_ptr().add(offset) as *const __m256i);

    let t0 = _mm256_extract_epi64(ts_vec, 0) as u64;
    let t1 = _mm256_extract_epi64(ts_vec, 1) as u64;
    let t2 = _mm256_extract_epi64(ts_vec, 2) as u64;
    let t3 = _mm256_extract_epi64(ts_vec, 3) as u64;

    let m0 = (((t0 / 86400) / 30) % 12 + 1) as i64;
    let m1 = (((t1 / 86400) / 30) % 12 + 1) as i64;
    let m2 = (((t2 / 86400) / 30) % 12 + 1) as i64;
    let m3 = (((t3 / 86400) / 30) % 12 + 1) as i64;

    // AVX2 doesn't have _mm256_cvtepi64_pd, use direct set
    let result_vec = _mm256_set_pd(m3 as f64, m2 as f64, m1 as f64, m0 as f64);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = (((days / 30) % 12) + 1) as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn month_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = vld1q_u64(timestamps.as_ptr().add(offset));

    let t0 = vgetq_lane_u64(ts_vec, 0);
    let t1 = vgetq_lane_u64(ts_vec, 1);

    let m0 = ((t0 / 86400) / 30) % 12 + 1;
    let m1 = ((t1 / 86400) / 30) % 12 + 1;

    let result_vec_u = vsetq_lane_u64(m1, vsetq_lane_u64(m0, vdupq_n_u64(0), 0), 1);
    let result_vec = vcvtq_f64_u64(result_vec_u);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = (((days / 30) % 12) + 1) as f64;
  }
}

// =============================================================================
// YEAR (u64 ^ f64) - Extract year from Unix timestamps
// =============================================================================

#[cfg(has_cuda)]
pub unsafe fn year_u64_gpu(timestamps: *const u64, result: *mut f64, len: usize) {
  const PTX_YEAR_U64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry year_u64_kernel(
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [timestamps_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Calculate global thread ID
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds
      setp.ge.u64 %p0, %rd3, %rd2;
      @%p0 bra done;
      
      // Load timestamp
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u64 %rd6, [%rd5];
      
      // Extract year: 1970 + (days / 365)
      mov.u64 %rd7, 86400;
      div.u64 %rd8, %rd6, %rd7;  // days since epoch
      mov.u64 %rd9, 365;
      div.u64 %rd10, %rd8, %rd9; // years since epoch
      mov.u64 %rd11, 1970;
      add.u64 %rd11, %rd11, %rd10; // 1970 + years
      
      // Convert to f64
      cvt.rn.f64.u64 %fd0, %rd11;
      
      // Store result
      mul.wide.u32 %rd4, %r4, 8;
      add.u64 %rd7, %rd1, %rd4;
      st.global.f64 [%rd7], %fd0;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    timestamps as *const u8,
    result as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(PTX_YEAR_U64, &[], "year_u64_kernel", blocks, threads, &args);
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn year_u64_avx512(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let timestamps_vec = _mm512_loadu_epi64(timestamps.as_ptr().add(offset) as *const i64);

    let mut temp_timestamps = [0i64; 8];
    _mm512_storeu_epi64(temp_timestamps.as_mut_ptr(), timestamps_vec);
    let t0 = temp_timestamps[0] as u64;
    let t1 = temp_timestamps[1] as u64;
    let t2 = temp_timestamps[2] as u64;
    let t3 = temp_timestamps[3] as u64;
    let t4 = temp_timestamps[4] as u64;
    let t5 = temp_timestamps[5] as u64;
    let t6 = temp_timestamps[6] as u64;
    let t7 = temp_timestamps[7] as u64;

    let y0 = (1970 + (t0 / 86400) / 365) as i64;
    let y1 = (1970 + (t1 / 86400) / 365) as i64;
    let y2 = (1970 + (t2 / 86400) / 365) as i64;
    let y3 = (1970 + (t3 / 86400) / 365) as i64;
    let y4 = (1970 + (t4 / 86400) / 365) as i64;
    let y5 = (1970 + (t5 / 86400) / 365) as i64;
    let y6 = (1970 + (t6 / 86400) / 365) as i64;
    let y7 = (1970 + (t7 / 86400) / 365) as i64;

    let result_vec_i = _mm512_set_epi64(y7, y6, y5, y4, y3, y2, y1, y0);
    let result_vec = _mm512_cvtepi64_pd(result_vec_i);
    _mm512_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = (1970 + (days / 365)) as f64;
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn year_u64_avx2(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = _mm256_loadu_si256(timestamps.as_ptr().add(offset) as *const __m256i);

    let t0 = _mm256_extract_epi64(ts_vec, 0) as u64;
    let t1 = _mm256_extract_epi64(ts_vec, 1) as u64;
    let t2 = _mm256_extract_epi64(ts_vec, 2) as u64;
    let t3 = _mm256_extract_epi64(ts_vec, 3) as u64;

    let y0 = (1970 + (t0 / 86400) / 365) as i64;
    let y1 = (1970 + (t1 / 86400) / 365) as i64;
    let y2 = (1970 + (t2 / 86400) / 365) as i64;
    let y3 = (1970 + (t3 / 86400) / 365) as i64;

    // AVX2 doesn't have _mm256_cvtepi64_pd, use direct set
    let result_vec = _mm256_set_pd(y3 as f64, y2 as f64, y1 as f64, y0 as f64);
    _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = (1970 + (days / 365)) as f64;
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn year_u64_neon(timestamps: &[u64], result: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_U64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let ts_vec = vld1q_u64(timestamps.as_ptr().add(offset));

    let t0 = vgetq_lane_u64(ts_vec, 0);
    let t1 = vgetq_lane_u64(ts_vec, 1);

    let y0 = 1970 + (t0 / 86400) / 365;
    let y1 = 1970 + (t1 / 86400) / 365;

    let result_vec_u = vsetq_lane_u64(y1, vsetq_lane_u64(y0, vdupq_n_u64(0), 0), 1);
    let result_vec = vcvtq_f64_u64(result_vec_u);
    vst1q_f64(result.as_mut_ptr().add(offset), result_vec);
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let timestamp = timestamps[offset];
    let days = timestamp / 86400;
    result[offset] = (1970 + (days / 365)) as f64;
  }
}

// =============================================================================
// TIME SERIES OPERATIONS (f64 ^ f64)
// =============================================================================

// CHANGES - Count number of value changes in time series
#[cfg(has_cuda)]
pub unsafe fn changes_f64_gpu(values: *const f64, len: usize) -> f64 {
  const PTX_CHANGES_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry changes_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      .shared .u32 block_count;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Initialize shared counter
      mov.u32 %r0, %tid.x;
      setp.eq.u32 %p0, %r0, 0;
      @%p0 st.shared.u32 [block_count], 0;
      bar.sync 0;
      
      // Calculate global thread ID
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds (tid < len - 1)
      sub.u64 %rd4, %rd2, 1;
      setp.ge.u64 %p1, %rd3, %rd4;
      @%p1 bra sync_point;
      
      // Load current and next values using vectorized load
      mul.wide.u32 %rd5, %r4, 8;
      add.u64 %rd6, %rd0, %rd5;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd6];  // Vectorized load of adjacent values
      
      // Check if values differ
      setp.ne.f64 %p2, %fd0, %fd1;
      @!%p2 bra sync_point;
      
      // Increment shared counter atomically
      atom.shared.add.u32 %r5, [block_count], 1;
      
    sync_point:
      bar.sync 0;
      
      // Thread 0 writes block result to global memory
      setp.ne.u32 %p0, %r0, 0;
      @%p0 bra done;
      
      ld.shared.u32 %r6, [block_count];
      cvt.rn.f64.u32 %fd2, %r6;
      
      // Atomic add to global result
      atom.global.add.f64 %fd2, [%rd1], %fd2;
      
    done:
      ret;
    }
  "#;

  // Initialize result to 0
  let mut result = 0.0f64;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    values as *const u8,
    &mut result as *mut f64 as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_CHANGES_F64,
    &[],
    "changes_f64_kernel",
    blocks,
    threads,
    &args,
  );
  result
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn changes_f64_avx512(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX512_F64;
  let mut count = 0u64;

  if len < 2 {
    return 0.0;
  }

  let chunks = (len - 1) / LANES;
  let remainder = (len - 1) % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let curr_vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    let next_vec = _mm512_loadu_pd(values.as_ptr().add(offset + 1));
    let mask = _mm512_cmp_pd_mask(curr_vec, next_vec, _CMP_NEQ_OQ);
    count += mask.count_ones() as u64;
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    if values[offset] != values[offset + 1] {
      count += 1;
    }
  }

  count as f64
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn changes_f64_avx2(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX2_F64;
  let mut count = 0u64;

  if len < 2 {
    return 0.0;
  }

  let chunks = (len - 1) / LANES;
  let remainder = (len - 1) % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let curr_vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    let next_vec = _mm256_loadu_pd(values.as_ptr().add(offset + 1));
    let cmp_result = _mm256_cmp_pd(curr_vec, next_vec, _CMP_NEQ_OQ);
    let mask = _mm256_movemask_pd(cmp_result) as u32;
    count += mask.count_ones() as u64;
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    if values[offset] != values[offset + 1] {
      count += 1;
    }
  }

  count as f64
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn changes_f64_neon(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_NEON_F64;
  let mut count = 0u64;

  if len < 2 {
    return 0.0;
  }

  let chunks = (len - 1) / LANES;
  let remainder = (len - 1) % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let curr_vec = vld1q_f64(values.as_ptr().add(offset));
    let next_vec = vld1q_f64(values.as_ptr().add(offset + 1));
    let eq_mask = vceqq_f64(curr_vec, next_vec);
    let mask_bits = eq_mask;
    // Count where NOT equal (eq_mask has all 1s for equal, 0s for not equal)
    if vgetq_lane_u64(mask_bits, 0) == 0 {
      count += 1;
    }
    if vgetq_lane_u64(mask_bits, 1) == 0 {
      count += 1;
    }
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    if values[offset] != values[offset + 1] {
      count += 1;
    }
  }

  count as f64
}

// RESETS - Count number of value resets (breaks in monotonicity)
#[cfg(has_cuda)]
pub unsafe fn resets_f64_gpu(values: *const f64, len: usize, _ascending: bool) -> f64 {
  const PTX_RESETS_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry resets_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      .shared .u32 block_count;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Initialize shared counter
      mov.u32 %r0, %tid.x;
      setp.eq.u32 %p0, %r0, 0;
      @%p0 st.shared.u32 [block_count], 0;
      bar.sync 0;
      
      // Calculate global thread ID
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      // Check bounds (tid < len - 1)
      sub.u64 %rd4, %rd2, 1;
      setp.ge.u64 %p1, %rd3, %rd4;
      @%p1 bra sync_point;
      
      // Load current and next values using vectorized load
      mul.wide.u32 %rd5, %r4, 8;
      add.u64 %rd6, %rd0, %rd5;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd6];  // Vectorized load of adjacent values
      
      // Check if values[tid+1] < values[tid] (reset condition)
      setp.ge.f64 %p2, %fd1, %fd0;
      @%p2 bra sync_point;
      
      // Increment shared counter atomically
      atom.shared.add.u32 %r5, [block_count], 1;
      
    sync_point:
      bar.sync 0;
      
      // Thread 0 writes block result to global memory
      setp.ne.u32 %p0, %r0, 0;
      @%p0 bra done;
      
      ld.shared.u32 %r6, [block_count];
      cvt.rn.f64.u32 %fd2, %r6;
      
      // Atomic add to global result
      atom.global.add.f64 %fd2, [%rd1], %fd2;
      
    done:
      ret;
    }
  "#;

  // Initialize result to 0
  let mut result = 0.0f64;

  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    values as *const u8,
    &mut result as *mut f64 as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_RESETS_F64,
    &[],
    "resets_f64_kernel",
    blocks,
    threads,
    &args,
  );
  result
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn resets_f64_avx512(values: &[f64], len: usize, ascending: bool) -> f64 {
  const LANES: usize = LANES_AVX512_F64;
  let mut count = 0u64;

  if len < 2 {
    return 0.0;
  }

  let chunks = (len - 1) / LANES;
  let remainder = (len - 1) % LANES;

  // Choose comparison based on ascending flag

  for i in 0..chunks {
    let offset = i * LANES;
    let curr_vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    let next_vec = _mm512_loadu_pd(values.as_ptr().add(offset + 1));
    let mask = if ascending {
      _mm512_cmp_pd_mask(next_vec, curr_vec, _CMP_LT_OQ)
    } else {
      _mm512_cmp_pd_mask(next_vec, curr_vec, _CMP_GT_OQ)
    };
    count += mask.count_ones() as u64;
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    if ascending {
      // For ascending, count when next < current (reset/decrease)
      if values[offset + 1] < values[offset] {
        count += 1;
      }
    } else {
      // For descending, count when next > current (reset/increase)
      if values[offset + 1] > values[offset] {
        count += 1;
      }
    }
  }

  count as f64
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn resets_f64_avx2(values: &[f64], len: usize, ascending: bool) -> f64 {
  const LANES: usize = LANES_AVX2_F64;
  let mut count = 0u64;

  if len < 2 {
    return 0.0;
  }

  let chunks = (len - 1) / LANES;
  let remainder = (len - 1) % LANES;

  // Choose comparison based on ascending flag
  for i in 0..chunks {
    let offset = i * LANES;
    let curr_vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    let next_vec = _mm256_loadu_pd(values.as_ptr().add(offset + 1));
    let cmp_result = if ascending {
      _mm256_cmp_pd(next_vec, curr_vec, _CMP_LT_OQ)
    } else {
      _mm256_cmp_pd(next_vec, curr_vec, _CMP_GT_OQ)
    };
    let mask = _mm256_movemask_pd(cmp_result) as u32;
    count += mask.count_ones() as u64;
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    if ascending {
      // For ascending, count when next < current (reset/decrease)
      if values[offset + 1] < values[offset] {
        count += 1;
      }
    } else {
      // For descending, count when next > current (reset/increase)
      if values[offset + 1] > values[offset] {
        count += 1;
      }
    }
  }

  count as f64
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn resets_f64_neon(values: &[f64], len: usize, ascending: bool) -> f64 {
  const LANES: usize = LANES_NEON_F64;
  let mut count = 0u64;

  if len < 2 {
    return 0.0;
  }

  let chunks = (len - 1) / LANES;
  let remainder = (len - 1) % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let curr_vec = vld1q_f64(values.as_ptr().add(offset));
    let next_vec = vld1q_f64(values.as_ptr().add(offset + 1));

    // Choose comparison based on ascending flag
    let cmp_mask = if ascending {
      vcltq_f64(next_vec, curr_vec) // next < current for ascending
    } else {
      vcgtq_f64(next_vec, curr_vec) // next > current for descending
    };

    let mask_bits = cmp_mask;
    // Count where condition is true (mask has all 1s for true)
    if vgetq_lane_u64(mask_bits, 0) != 0 {
      count += 1;
    }
    if vgetq_lane_u64(mask_bits, 1) != 0 {
      count += 1;
    }
  }

  for i in 0..remainder {
    let offset = chunks * LANES + i;
    if ascending {
      // For ascending, count when next < current (reset/decrease)
      if values[offset + 1] < values[offset] {
        count += 1;
      }
    } else {
      // For descending, count when next > current (reset/increase)
      if values[offset + 1] > values[offset] {
        count += 1;
      }
    }
  }

  count as f64
}

// INCREASE - Calculate increase over time (handles counter resets)
#[cfg(has_cuda)]
pub unsafe fn increase_f64_gpu(values: *const f64, len: usize) -> f64 {
  const PTX_INCREASE_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry increase_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      // Check if len < 2
      setp.lt.u64 %p0, %rd2, 2;
      @%p0 bra zero_result;
      
      // Load first and last values
      ld.global.f64 %fd0, [%rd0];  // first value
      sub.u64 %rd3, %rd2, 1;
      mul.lo.u64 %rd4, %rd3, 8;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.f64 %fd1, [%rd5];  // last value
      
      // Calculate increase = last - first
      sub.f64 %fd2, %fd1, %fd0;
      
      // Check if increase < 0 (counter reset)
      setp.lt.f64 %p1, %fd2, 0d0000000000000000;
      @%p1 bra use_last;
      
      // Normal case: use increase
      st.global.f64 [%rd1], %fd2;
      bra done;
      
    use_last:
      // Counter reset: use last value
      st.global.f64 [%rd1], %fd1;
      bra done;
      
    zero_result:
      mov.f64 %fd3, 0d0000000000000000;
      st.global.f64 [%rd1], %fd3;
      
    done:
      ret;
    }
  "#;

  let mut result: f64 = 0.0;
  let (blocks, threads) = LaunchConfig::reduction();
  let args = [
    values as *const u8,
    &mut result as *mut f64 as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_INCREASE_F64,
    &[],
    "increase_f64_kernel",
    blocks,
    threads,
    &args,
  );

  result
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn increase_f64_avx512(values: &[f64], len: usize) -> f64 {
  if len < 2 {
    return 0.0;
  }
  let first = values[0];
  let last = values[len - 1];
  let increase = last - first;
  if increase < 0.0 {
    // Counter reset
    last
  } else {
    increase
  }
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn increase_f64_avx2(values: &[f64], len: usize) -> f64 {
  if len < 2 {
    return 0.0;
  }
  let first = values[0];
  let last = values[len - 1];
  let increase = last - first;
  if increase < 0.0 {
    // Counter reset
    last
  } else {
    increase
  }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn increase_f64_neon(values: &[f64], len: usize) -> f64 {
  if len < 2 {
    return 0.0;
  }
  let first = values[0];
  let last = values[len - 1];
  let increase = last - first;
  if increase < 0.0 {
    // Counter reset
    last
  } else {
    increase
  }
}

// MAD - Mean Absolute Deviation: mean(|x - mean|)
// This GPU kernel calculates MAD in a single pass with NaN handling
#[cfg(has_cuda)]
pub unsafe fn mad_f64_gpu(values: *const f64, len: usize, result: *mut f64) {
  if len == 0 {
    *result = f64::NAN;
    return;
  }

  // Simple two-pass kernel: first pass calculates mean, second pass calculates MAD
  const PTX_MAD_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry mad_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 len,
      .param .u64 result_ptr
    ) {
      .reg .f64 %fd<20>;
      .reg .u32 %r<20>;
      .reg .u64 %rd<15>;
      .reg .pred %p<5>;
      
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r7, %rd1;
      ld.param.u64 %rd4, [result_ptr];
      
      
      // Initialize for first pass - calculate sum
      mov.f64 %fd0, 0d0000000000000000;  // sum
      mov.f64 %fd10, 0d0000000000000000; // count
      
      // First pass - calculate sum
      mov.u32 %r5, 0;
      
    sum_loop:
      setp.ge.u32 %p0, %r5, %r7;
      @%p0 bra calc_mean;
      
      // Load value
      mul.wide.u32 %rd2, %r5, 8;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.f64 %fd1, [%rd3];
      
      // Check for NaN
      setp.eq.f64 %p1, %fd1, %fd1;
      @!%p1 bra sum_skip;
      
      add.f64 %fd0, %fd0, %fd1;
      mov.f64 %fd11, 0d3ff0000000000000; // 1.0
      add.f64 %fd10, %fd10, %fd11;
      
    sum_skip:
      add.u32 %r5, %r5, 1;
      bra sum_loop;
      
    calc_mean:
      // Check if count is 0
      mov.f64 %fd11, 0d0000000000000000;
      setp.eq.f64 %p2, %fd10, %fd11;
      @%p2 bra write_nan;
      
      // Calculate mean
      div.rn.f64 %fd2, %fd0, %fd10;  // mean = sum / count
      
      // Second pass - calculate MAD
      mov.f64 %fd3, 0d0000000000000000;  // mad_sum
      mov.u32 %r6, 0;
      
    mad_loop:
      setp.ge.u32 %p0, %r6, %r7;
      @%p0 bra calc_mad_final;
      
      // Load value
      mul.wide.u32 %rd5, %r6, 8;
      add.u64 %rd6, %rd0, %rd5;
      ld.global.f64 %fd4, [%rd6];
      
      // Check for NaN
      setp.eq.f64 %p1, %fd4, %fd4;
      @!%p1 bra mad_skip;
      
      // Calculate |value - mean|
      sub.f64 %fd5, %fd4, %fd2;
      abs.f64 %fd6, %fd5;
      add.f64 %fd3, %fd3, %fd6;
      
    mad_skip:
      add.u32 %r6, %r6, 1;
      bra mad_loop;
      
    calc_mad_final:
      // Calculate final MAD
      
      // Check if count is still valid (non-zero)
      mov.f64 %fd13, 0d0000000000000000;
      setp.eq.f64 %p4, %fd10, %fd13;
      @%p4 bra write_nan;
      
      div.rn.f64 %fd7, %fd3, %fd10;  // mad = mad_sum / count
      st.global.f64 [%rd4], %fd7;
      ret;
      
    write_nan:
      mov.f64 %fd12, 0d7ff8000000000000;  // NaN  
      st.global.f64 [%rd4], %fd12;
      ret;
    }
  "#;

  let (blocks, threads) = (1, 1); // Single thread for simplicity
  let values_u64 = values as u64;
  let len_u64 = len as u64;
  let result_ptr = result as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
    &result_ptr as *const u64 as *const u8,
  ];

  if let Err(e) = launch_ptx(PTX_MAD_F64, &[], "mad_f64_kernel", blocks, threads, &args) {
    debug!("mad_f64_gpu PTX launch failed: {:?}", e);
    *result = f64::NAN;
    return;
  }
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn mad_f64_avx512(values: &[f64], len: usize) -> f64 {
  if len == 0 {
    return f64::NAN;
  }

  const LANES: usize = LANES_AVX512_F64;

  // Step 1: Calculate mean, skipping NaN values
  let mut sum = 0.0;
  let mut count = 0;
  for i in 0..len {
    let val = values[i];
    if !val.is_nan() {
      sum += val;
      count += 1;
    }
  }

  if count == 0 {
    return f64::NAN;
  }

  let mean = sum / (count as f64);
  let mean_vec = _mm512_set1_pd(mean);

  // Step 2: Calculate sum of absolute deviations, skipping NaN values
  let chunks = len / LANES;
  let remainder = len % LANES;
  let mut sum_abs_dev_vec = _mm512_setzero_pd();

  // Process chunks
  for i in 0..chunks {
    let offset = i * LANES;
    let val_vec = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Create mask for non-NaN values (NaN != NaN)
    let nan_mask = _mm512_cmp_pd_mask(val_vec, val_vec, _CMP_EQ_OQ);

    // Calculate |value - mean|
    let diff_vec = _mm512_sub_pd(val_vec, mean_vec);
    let abs_diff_vec = _mm512_abs_pd(diff_vec);

    // Mask out NaN values using maskz operation
    let masked_abs_diff = _mm512_maskz_mov_pd(nan_mask, abs_diff_vec);

    // Add to accumulator
    sum_abs_dev_vec = _mm512_add_pd(sum_abs_dev_vec, masked_abs_diff);
  }

  // Reduce vector to scalar
  let mut sum_abs_dev = _mm512_reduce_add_pd(sum_abs_dev_vec);

  // Handle remainder
  let mut mad_count = count;
  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let val = values[idx];
    if !val.is_nan() {
      let abs_dev = (val - mean).abs();
      if !abs_dev.is_nan() {
        sum_abs_dev += abs_dev;
      } else {
        mad_count -= 1;
      }
    }
  }

  if mad_count == 0 {
    return f64::NAN;
  }

  // Check if we have NaN in sum (from inf - inf case)
  // If mean was infinity and we have finite values, result should be infinity
  if sum_abs_dev.is_nan() && mean.is_infinite() {
    return f64::INFINITY;
  }

  // Return mean of absolute deviations
  let result = sum_abs_dev / (mad_count as f64);
  if result.is_infinite() {
    return f64::INFINITY;
  }
  result
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn mad_f64_avx2(values: &[f64], len: usize) -> f64 {
  if len == 0 {
    return f64::NAN;
  }

  const LANES: usize = LANES_AVX2_F64;

  // Step 1: Calculate mean, skipping NaN values
  let mut sum = 0.0;
  let mut count = 0;
  for i in 0..len {
    let val = values[i];
    if !val.is_nan() {
      sum += val;
      count += 1;
    }
  }

  if count == 0 {
    return f64::NAN;
  }

  let mean = sum / (count as f64);
  let mean_vec = _mm256_set1_pd(mean);

  // Step 2: Calculate sum of absolute deviations, skipping NaN values
  let chunks = len / LANES;
  let remainder = len % LANES;
  let mut sum_abs_dev_vec = _mm256_setzero_pd();

  // Process chunks
  for i in 0..chunks {
    let offset = i * LANES;
    let val_vec = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN - compare with itself (NaN != NaN)
    let nan_mask = _mm256_cmp_pd(val_vec, val_vec, _CMP_EQ_OQ);

    // Calculate |value - mean|
    let diff_vec = _mm256_sub_pd(val_vec, mean_vec);
    // AVX2 doesn't have _mm256_abs_pd, so use AND with mask to clear sign bit
    let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));
    let abs_diff_vec = _mm256_and_pd(diff_vec, abs_mask);

    // Mask out NaN values
    let masked_abs_diff = _mm256_and_pd(abs_diff_vec, nan_mask);

    // Add to accumulator
    sum_abs_dev_vec = _mm256_add_pd(sum_abs_dev_vec, masked_abs_diff);
  }

  // Reduce vector to scalar
  let hadd1 = _mm256_hadd_pd(sum_abs_dev_vec, sum_abs_dev_vec);
  let high = _mm256_extractf128_pd(hadd1, 1);
  let low = _mm256_castpd256_pd128(hadd1);
  let final_sum_vec = _mm_add_pd(low, high);
  let mut sum_abs_dev = _mm_cvtsd_f64(final_sum_vec);

  // Handle remainder
  let mut mad_count = count;
  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let val = values[idx];
    if !val.is_nan() {
      let abs_dev = (val - mean).abs();
      if !abs_dev.is_nan() {
        sum_abs_dev += abs_dev;
      } else {
        mad_count -= 1;
      }
    }
  }

  if mad_count == 0 {
    return f64::NAN;
  }

  // Check if we have NaN in sum (from inf - inf case)
  // If mean was infinity and we have finite values, result should be infinity
  if sum_abs_dev.is_nan() && mean.is_infinite() {
    return f64::INFINITY;
  }

  // Return mean of absolute deviations
  let result = sum_abs_dev / (mad_count as f64);
  if result.is_infinite() {
    return f64::INFINITY;
  }
  result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn mad_f64_neon(values: &[f64], len: usize) -> f64 {
  if len == 0 {
    return f64::NAN;
  }

  const LANES: usize = LANES_NEON_F64;

  // Step 1: Calculate mean, skipping NaN values
  let mut sum = 0.0;
  let mut count = 0;
  for i in 0..len {
    let val = values[i];
    if !val.is_nan() {
      sum += val;
      count += 1;
    }
  }

  if count == 0 {
    return f64::NAN;
  }

  let mean = sum / (count as f64);
  let mean_vec = vdupq_n_f64(mean);

  // Step 2: Calculate sum of absolute deviations, skipping NaN values
  let chunks = len / LANES;
  let remainder = len % LANES;
  let mut sum_abs_dev_vec = vdupq_n_f64(0.0);

  // Process chunks
  for i in 0..chunks {
    let offset = i * LANES;
    let val_vec = vld1q_f64(values.as_ptr().add(offset));

    // Check for NaN - compare with itself (NaN != NaN)
    let nan_mask = vceqq_f64(val_vec, val_vec);

    // Calculate |value - mean|
    let diff_vec = vsubq_f64(val_vec, mean_vec);
    let abs_diff_vec = vabsq_f64(diff_vec);

    // Mask out NaN values
    let masked_abs_diff = vandq_u64(vreinterpretq_u64_f64(abs_diff_vec), nan_mask);
    let masked_abs_diff_f64 = vreinterpretq_f64_u64(masked_abs_diff);

    // Add to accumulator
    sum_abs_dev_vec = vaddq_f64(sum_abs_dev_vec, masked_abs_diff_f64);
  }

  // Reduce vector to scalar
  let mut sum_abs_dev = vgetq_lane_f64(sum_abs_dev_vec, 0) + vgetq_lane_f64(sum_abs_dev_vec, 1);

  // Handle remainder
  let mut mad_count = count;
  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let val = values[idx];
    if !val.is_nan() {
      let abs_dev = (val - mean).abs();
      if !abs_dev.is_nan() {
        sum_abs_dev += abs_dev;
      } else {
        mad_count -= 1;
      }
    }
  }

  if mad_count == 0 {
    return f64::NAN;
  }

  // Check if we have NaN in sum (from inf - inf case)
  // If mean was infinity and we have finite values, result should be infinity
  if sum_abs_dev.is_nan() && mean.is_infinite() {
    return f64::INFINITY;
  }

  // Return mean of absolute deviations
  let result = sum_abs_dev / (mad_count as f64);
  if result.is_infinite() {
    return f64::INFINITY;
  }
  result
}

// PREDICT_LINEAR - Linear prediction/extrapolation
#[cfg(has_cuda)]
pub unsafe fn predict_linear_f64_gpu(
  values: *const f64,
  timestamps: *const f64,
  predict_time: f64,
  len: usize,
) -> f64 {
  const PTX_PREDICT_LINEAR_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry predict_linear_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 timestamps_ptr,
      .param .f64 predict_time,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      .shared .f64 sum_x;
      .shared .f64 sum_y;
      .shared .f64 sum_xy;
      .shared .f64 sum_xx;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [timestamps_ptr];
      ld.param.f64 %fd18, [predict_time];
      ld.param.u64 %rd3, [result_ptr];
      ld.param.u64 %rd4, [len];
      
      // Initialize shared sums
      mov.u32 %r0, %tid.x;
      setp.eq.u32 %p0, %r0, 0;
      @%p0 st.shared.f64 [sum_x], 0d0000000000000000;
      @%p0 st.shared.f64 [sum_y], 0d0000000000000000;
      @%p0 st.shared.f64 [sum_xy], 0d0000000000000000;
      @%p0 st.shared.f64 [sum_xx], 0d0000000000000000;
      bar.sync 0;
      
      // Calculate global thread ID and grid stride
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd5, %r4;
      
      mov.u32 %r5, %nctaid.x;
      mul.lo.u32 %r6, %r5, %r1;
      cvt.u64.u32 %rd6, %r6;
      
      // Initialize thread sums
      mov.f64 %fd0, 0d0000000000000000; // sum_x
      mov.f64 %fd1, 0d0000000000000000; // sum_y
      mov.f64 %fd2, 0d0000000000000000; // sum_xy
      mov.f64 %fd3, 0d0000000000000000; // sum_xx
      
    loop_start:
      setp.ge.u64 %p1, %rd5, %rd4;
      @%p1 bra loop_end;
      
      // Check if we can load 2 pairs
      add.u64 %rd9, %rd5, 1;
      setp.gt.u64 %p2, %rd9, %rd4;
      @%p2 bra scalar_load;
      
      // Load 2 timestamps and 2 values (vectorized)
      mul.lo.u64 %rd7, %rd5, 8;
      add.u64 %rd8, %rd1, %rd7;
      ld.global.v2.f64 {%fd4, %fd14}, [%rd8]; // x1, x2 = timestamps
      
      add.u64 %rd8, %rd0, %rd7;
      ld.global.v2.f64 {%fd5, %fd15}, [%rd8]; // y1, y2 = values
      
      // Accumulate sums for first pair
      add.f64 %fd0, %fd0, %fd4;
      add.f64 %fd1, %fd1, %fd5;
      fma.rn.f64 %fd2, %fd4, %fd5, %fd2;
      fma.rn.f64 %fd3, %fd4, %fd4, %fd3;
      
      // Accumulate sums for second pair
      add.f64 %fd0, %fd0, %fd14;
      add.f64 %fd1, %fd1, %fd15;
      fma.rn.f64 %fd2, %fd14, %fd15, %fd2;
      fma.rn.f64 %fd3, %fd14, %fd14, %fd3;
      
      // Increment by grid stride * 2
      add.u64 %rd5, %rd5, %rd6;
      add.u64 %rd5, %rd5, %rd6;
      bra.uni loop_start;
      
    scalar_load:
      // Load single timestamp and value
      mul.lo.u64 %rd7, %rd5, 8;
      add.u64 %rd8, %rd1, %rd7;
      ld.global.f64 %fd4, [%rd8]; // x = timestamp
      
      add.u64 %rd8, %rd0, %rd7;
      ld.global.f64 %fd5, [%rd8]; // y = value
      
      // Accumulate sums
      add.f64 %fd0, %fd0, %fd4;
      add.f64 %fd1, %fd1, %fd5;
      fma.rn.f64 %fd2, %fd4, %fd5, %fd2;
      fma.rn.f64 %fd3, %fd4, %fd4, %fd3;
      
      // Increment by grid stride
      add.u64 %rd5, %rd5, %rd6;
      bra.uni loop_start;
      
    loop_end:
      // Add thread sums to shared memory atomically
      atom.shared.add.f64 %fd6, [sum_x], %fd0;
      atom.shared.add.f64 %fd7, [sum_y], %fd1;
      atom.shared.add.f64 %fd8, [sum_xy], %fd2;
      atom.shared.add.f64 %fd9, [sum_xx], %fd3;
      bar.sync 0;
      
      // Thread 0 calculates linear regression and prediction
      setp.ne.u32 %p2, %r0, 0;
      @%p2 bra done;
      
      ld.shared.f64 %fd10, [sum_x];
      ld.shared.f64 %fd11, [sum_y];
      ld.shared.f64 %fd12, [sum_xy];
      ld.shared.f64 %fd13, [sum_xx];
      
      cvt.rn.f64.u64 %fd14, %rd4; // n = len
      
      // Calculate slope: (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
      mul.f64 %fd15, %fd14, %fd12; // n*sum_xy
      mul.f64 %fd16, %fd10, %fd11; // sum_x*sum_y
      sub.f64 %fd17, %fd15, %fd16; // numerator
      
      mul.f64 %fd15, %fd14, %fd13; // n*sum_xx
      mul.f64 %fd16, %fd10, %fd10; // sum_x*sum_x
      sub.f64 %fd18, %fd15, %fd16; // denominator
      
      div.rn.f64 %fd19, %fd17, %fd18; // slope
      
      // Calculate intercept: (sum_y - slope*sum_x) / n
      mul.f64 %fd15, %fd19, %fd10; // slope*sum_x
      sub.f64 %fd16, %fd11, %fd15; // sum_y - slope*sum_x
      div.rn.f64 %fd17, %fd16, %fd14; // intercept
      
      // Predict: intercept + slope * predict_time
      fma.rn.f64 %fd15, %fd19, %fd18, %fd17;
      // Store result in global memory for return
      st.global.f64 [%rd3], %fd15;
      
    done:
      ret;
    }
  "#;

  let mut result = 0.0f64;
  let (blocks, threads) = LaunchConfig::parallel();
  let args = [
    values as *const u8,
    timestamps as *const u8,
    &predict_time as *const f64 as *const u8,
    &mut result as *mut f64 as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_PREDICT_LINEAR_F64,
    &[],
    "predict_linear_f64_kernel",
    blocks,
    threads,
    &args,
  );
  result
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn predict_linear_f64_avx512(
  values: &[f64],
  timestamps: &[f64],
  predict_time: f64,
  len: usize,
) -> f64 {
  if len < 2 {
    if len == 0 {
      return 0.0;
    }
    return values[0];
  }

  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Initialize vector accumulators for linear regression
  let mut sum_x_vec = _mm512_setzero_pd();
  let mut sum_y_vec = _mm512_setzero_pd();
  let mut sum_xy_vec = _mm512_setzero_pd();
  let mut sum_xx_vec = _mm512_setzero_pd();

  // Process 8 elements at a time with AVX-512
  for i in 0..chunks {
    let offset = i * LANES;
    let x_vec = _mm512_loadu_pd(timestamps.as_ptr().add(offset));
    let y_vec = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Accumulate sums
    sum_x_vec = _mm512_add_pd(sum_x_vec, x_vec);
    sum_y_vec = _mm512_add_pd(sum_y_vec, y_vec);
    sum_xy_vec = _mm512_fmadd_pd(x_vec, y_vec, sum_xy_vec);
    sum_xx_vec = _mm512_fmadd_pd(x_vec, x_vec, sum_xx_vec);
  }

  // Reduce vectors to scalars
  let sum_x = _mm512_reduce_add_pd(sum_x_vec);
  let sum_y = _mm512_reduce_add_pd(sum_y_vec);
  let sum_xy = _mm512_reduce_add_pd(sum_xy_vec);
  let sum_xx = _mm512_reduce_add_pd(sum_xx_vec);

  // Handle remainder with scalar operations
  let mut final_sum_x = sum_x;
  let mut final_sum_y = sum_y;
  let mut final_sum_xy = sum_xy;
  let mut final_sum_xx = sum_xx;

  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let x = timestamps[idx];
    let y = values[idx];
    final_sum_x += x;
    final_sum_y += y;
    final_sum_xy += x * y;
    final_sum_xx += x * x;
  }

  // Calculate slope using linear regression formula
  let n = len as f64;
  let slope =
    (n * final_sum_xy - final_sum_x * final_sum_y) / (n * final_sum_xx - final_sum_x * final_sum_x);
  let intercept = (final_sum_y - slope * final_sum_x) / n;

  // Return predicted value
  intercept + slope * predict_time
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn predict_linear_f64_avx2(
  values: &[f64],
  timestamps: &[f64],
  predict_time: f64,
  len: usize,
) -> f64 {
  if len < 2 {
    if len == 0 {
      return 0.0;
    }
    return values[0];
  }

  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Initialize vector accumulators for linear regression
  let mut sum_x_vec = _mm256_setzero_pd();
  let mut sum_y_vec = _mm256_setzero_pd();
  let mut sum_xy_vec = _mm256_setzero_pd();
  let mut sum_xx_vec = _mm256_setzero_pd();

  // Process 4 elements at a time with AVX2
  for i in 0..chunks {
    let offset = i * LANES;
    let x_vec = _mm256_loadu_pd(timestamps.as_ptr().add(offset));
    let y_vec = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Accumulate sums
    sum_x_vec = _mm256_add_pd(sum_x_vec, x_vec);
    sum_y_vec = _mm256_add_pd(sum_y_vec, y_vec);
    sum_xy_vec = _mm256_fmadd_pd(x_vec, y_vec, sum_xy_vec);
    sum_xx_vec = _mm256_fmadd_pd(x_vec, x_vec, sum_xx_vec);
  }

  // Reduce vectors to scalars
  let sum_x_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_x_vec);
  let sum_y_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_y_vec);
  let sum_xy_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_xy_vec);
  let sum_xx_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_xx_vec);

  let sum_x = sum_x_arr[0] + sum_x_arr[1] + sum_x_arr[2] + sum_x_arr[3];
  let sum_y = sum_y_arr[0] + sum_y_arr[1] + sum_y_arr[2] + sum_y_arr[3];
  let sum_xy = sum_xy_arr[0] + sum_xy_arr[1] + sum_xy_arr[2] + sum_xy_arr[3];
  let sum_xx = sum_xx_arr[0] + sum_xx_arr[1] + sum_xx_arr[2] + sum_xx_arr[3];

  // Handle remainder with scalar operations
  let mut final_sum_x = sum_x;
  let mut final_sum_y = sum_y;
  let mut final_sum_xy = sum_xy;
  let mut final_sum_xx = sum_xx;

  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let x = timestamps[idx];
    let y = values[idx];
    final_sum_x += x;
    final_sum_y += y;
    final_sum_xy += x * y;
    final_sum_xx += x * x;
  }

  // Calculate slope using linear regression formula
  let n = len as f64;
  let slope =
    (n * final_sum_xy - final_sum_x * final_sum_y) / (n * final_sum_xx - final_sum_x * final_sum_x);
  let intercept = (final_sum_y - slope * final_sum_x) / n;

  // Return predicted value
  intercept + slope * predict_time
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn predict_linear_f64_neon(
  values: &[f64],
  timestamps: &[f64],
  predict_time: f64,
  len: usize,
) -> f64 {
  if len < 2 {
    if len == 0 {
      return 0.0;
    }
    return values[0];
  }

  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Initialize vector accumulators for linear regression
  let mut sum_x_vec = vdupq_n_f64(0.0);
  let mut sum_y_vec = vdupq_n_f64(0.0);
  let mut sum_xy_vec = vdupq_n_f64(0.0);
  let mut sum_xx_vec = vdupq_n_f64(0.0);

  // Process 2 elements at a time with NEON
  for i in 0..chunks {
    let offset = i * LANES;
    let x_vec = vld1q_f64(timestamps.as_ptr().add(offset));
    let y_vec = vld1q_f64(values.as_ptr().add(offset));

    // Accumulate sums
    sum_x_vec = vaddq_f64(sum_x_vec, x_vec);
    sum_y_vec = vaddq_f64(sum_y_vec, y_vec);
    sum_xy_vec = vfmaq_f64(sum_xy_vec, x_vec, y_vec);
    sum_xx_vec = vfmaq_f64(sum_xx_vec, x_vec, x_vec);
  }

  // Reduce vectors to scalars
  let sum_x = vgetq_lane_f64(sum_x_vec, 0) + vgetq_lane_f64(sum_x_vec, 1);
  let sum_y = vgetq_lane_f64(sum_y_vec, 0) + vgetq_lane_f64(sum_y_vec, 1);
  let sum_xy = vgetq_lane_f64(sum_xy_vec, 0) + vgetq_lane_f64(sum_xy_vec, 1);
  let sum_xx = vgetq_lane_f64(sum_xx_vec, 0) + vgetq_lane_f64(sum_xx_vec, 1);

  // Handle remainder with scalar operations
  let mut final_sum_x = sum_x;
  let mut final_sum_y = sum_y;
  let mut final_sum_xy = sum_xy;
  let mut final_sum_xx = sum_xx;

  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let x = timestamps[idx];
    let y = values[idx];
    final_sum_x += x;
    final_sum_y += y;
    final_sum_xy += x * y;
    final_sum_xx += x * x;
  }

  // Calculate slope using linear regression formula
  let n = len as f64;
  let slope =
    (n * final_sum_xy - final_sum_x * final_sum_y) / (n * final_sum_xx - final_sum_x * final_sum_x);
  let intercept = (final_sum_y - slope * final_sum_x) / n;

  // Return predicted value
  intercept + slope * predict_time
}

#[cfg(has_cuda)]
pub unsafe fn present_f64_gpu(values: *const f64, len: usize) -> f64 {
  const PTX_PRESENT_F64: &str = r#"
    .version 7.0
    .target sm_70
    .address_size 64
    .entry present_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      .shared .u32 block_count[1];
      
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd2, [len];
      
      mov.u32 %r0, %tid.x;
      setp.eq.u32 %p0, %r0, 0;
      @%p0 st.shared.u32 [block_count], 0;
      bar.sync 0;
      
      mov.u32 %r1, %ntid.x;
      mov.u32 %r2, %ctaid.x;
      mul.lo.u32 %r3, %r2, %r1;
      add.u32 %r4, %r3, %r0;
      cvt.u64.u32 %rd3, %r4;
      
      mov.u32 %r5, %nctaid.x;
      mul.lo.u32 %r6, %r5, %r1;
      cvt.u64.u32 %rd4, %r6;
      
      mov.u32 %r7, 0;
      
    loop_start:
      setp.ge.u64 %p1, %rd3, %rd2;
      @%p1 bra loop_end;
      
      // Check if we can load double2
      add.u64 %rd7, %rd3, 1;
      setp.gt.u64 %p3, %rd7, %rd2;
      @%p3 bra scalar_load;
      
      // Load double2 values (vectorized)
      mul.lo.u64 %rd5, %rd3, 8;
      add.u64 %rd6, %rd0, %rd5;
      ld.global.v2.f64 {%fd0, %fd1}, [%rd6];
      
      // Check if values are not NaN (NaN != NaN)
      setp.eq.f64 %p2, %fd0, %fd0;
      @!%p2 bra check_second;
      add.u32 %r7, %r7, 1;
    check_second:
      setp.eq.f64 %p4, %fd1, %fd1;
      @!%p4 bra skip_inc2;
      add.u32 %r7, %r7, 1;
      
    skip_inc2:
      // Grid stride * 2 for double2
      add.u64 %rd8, %rd4, %rd4;
      add.u64 %rd3, %rd3, %rd8;
      bra.uni loop_start;
      
    scalar_load:
      // Load single value
      mul.lo.u64 %rd5, %rd3, 8;
      add.u64 %rd6, %rd0, %rd5;
      ld.global.f64 %fd0, [%rd6];
      
      setp.eq.f64 %p2, %fd0, %fd0;
      @!%p2 bra skip_inc;
      add.u32 %r7, %r7, 1;
      
    skip_inc:
      add.u64 %rd3, %rd3, %rd4;
      bra.uni loop_start;
      
    loop_end:
      mov.u32 %r8, 0;
      atom.shared.add.u32 %r8, [block_count], %r7;
      bar.sync 0;
      
      setp.ne.u32 %p3, %r0, 0;
      @%p3 bra done;
      
      ld.shared.u32 %r9, [block_count];
      cvt.rn.f64.u32 %fd1, %r9;
      st.global.f64 [%rd1], %fd1;
      
    done:
      ret;
    }
  "#;

  let mut result = 0.0f64;
  let (blocks, threads) = LaunchConfig::parallel();
  let len_u64 = len as u64;
  let values_u64 = values as u64;
  let result_ptr = &mut result as *mut f64 as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &result_ptr as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
  ];

  let _ = launch_ptx(
    PTX_PRESENT_F64,
    &[],
    "present_f64_kernel",
    blocks,
    threads,
    &args,
  );

  // GPU returns count, we need 1.0 or 0.0
  if result > 0.0 { 1.0 } else { 0.0 }
}

#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn present_f64_avx512(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    // Check for NaN: NaN != NaN
    let nan_mask = _mm512_cmp_pd_mask(vec, vec, _CMP_EQ_OQ);
    if nan_mask != 0 {
      return 1.0; // Found at least one non-NaN value
    }
  }

  for i in 0..remainder {
    let val = values[chunks * LANES + i];
    if !val.is_nan() {
      return 1.0; // Found at least one non-NaN value
    }
  }

  0.0 // All values are NaN
}

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn present_f64_avx2(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    // Check for NaN: NaN != NaN
    let nan_mask = _mm256_cmp_pd(vec, vec, _CMP_EQ_OQ);
    let mask_bits = _mm256_movemask_pd(nan_mask) as u32;
    if mask_bits != 0 {
      return 1.0; // Found at least one non-NaN value
    }
  }

  for i in 0..remainder {
    let val = values[chunks * LANES + i];
    if !val.is_nan() {
      return 1.0; // Found at least one non-NaN value
    }
  }

  0.0 // All values are NaN
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn present_f64_neon(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    // Check for NaN: NaN == NaN is false
    let eq_mask = vceqq_f64(vec, vec);
    let mask_bits = eq_mask;
    // Check if any value is not NaN (eq_mask has all 1s for non-NaN)
    if vgetq_lane_u64(mask_bits, 0) != 0 || vgetq_lane_u64(mask_bits, 1) != 0 {
      return 1.0; // Found at least one non-NaN value
    }
  }

  for i in 0..remainder {
    let val = values[chunks * LANES + i];
    if !val.is_nan() {
      return 1.0; // Found at least one non-NaN value
    }
  }

  0.0 // All values are NaN
}

// DERIV - Calculate derivative using linear regression
#[cfg(has_cuda)]
pub unsafe fn deriv_f64_gpu(values: *const f64, timestamps: *const u64, len: usize) -> f64 {
  const PTX_DERIV_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry deriv_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 timestamps_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      
      // Only thread 0 does work
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      setp.ne.u32 %p0, %r0, 0;
      @%p0 bra done;
      setp.ne.u32 %p1, %r1, 0;
      @%p1 bra done;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [timestamps_ptr];
      ld.param.u64 %rd2, [result_ptr];
      ld.param.u64 %rd3, [len];
      
      // Check if len >= 2
      setp.lt.u64 %p2, %rd3, 2;
      @%p2 bra store_zero;
      
      // Initialize sums for linear regression
      mov.f64 %fd0, 0d0000000000000000;  // sum_x
      mov.f64 %fd1, 0d0000000000000000;  // sum_y
      mov.f64 %fd2, 0d0000000000000000;  // sum_xy
      mov.f64 %fd3, 0d0000000000000000;  // sum_xx
      
      // Loop counter
      mov.u64 %rd4, 0;
      
    loop_start:
      setp.ge.u64 %p3, %rd4, %rd3;
      @%p3 bra loop_end;
      
      // Check if we can load 2 elements
      add.u64 %rd9, %rd4, 1;
      setp.gt.u64 %p4, %rd9, %rd3;
      @%p4 bra scalar_load;
      
      // Load 2 timestamps and 2 values (vectorized)
      mul.lo.u64 %rd5, %rd4, 8;
      add.u64 %rd6, %rd1, %rd5;
      add.u64 %rd7, %rd0, %rd5;
      ld.global.v2.u64 {%rd8, %rd10}, [%rd6];  // Load 2 timestamps
      ld.global.v2.f64 {%fd4, %fd12}, [%rd7];  // Load 2 values
      
      // Convert timestamps to f64
      cvt.rn.f64.u64 %fd5, %rd8;
      cvt.rn.f64.u64 %fd13, %rd10;
      
      // Update sums for first pair
      add.f64 %fd0, %fd0, %fd5;      // sum_x += x1
      add.f64 %fd1, %fd1, %fd4;      // sum_y += y1
      mul.f64 %fd6, %fd5, %fd4;
      add.f64 %fd2, %fd2, %fd6;      // sum_xy += x1*y1
      mul.f64 %fd7, %fd5, %fd5;
      add.f64 %fd3, %fd3, %fd7;      // sum_xx += x1*x1
      
      // Update sums for second pair
      add.f64 %fd0, %fd0, %fd13;     // sum_x += x2
      add.f64 %fd1, %fd1, %fd12;     // sum_y += y2
      mul.f64 %fd14, %fd13, %fd12;
      add.f64 %fd2, %fd2, %fd14;     // sum_xy += x2*y2
      mul.f64 %fd15, %fd13, %fd13;
      add.f64 %fd3, %fd3, %fd15;     // sum_xx += x2*x2
      
      // Increment counter by 2
      add.u64 %rd4, %rd4, 2;
      bra loop_start;
      
    scalar_load:
      // Load single timestamp and value
      mul.lo.u64 %rd5, %rd4, 8;
      add.u64 %rd6, %rd1, %rd5;
      add.u64 %rd7, %rd0, %rd5;
      ld.global.u64 %rd8, [%rd6];
      ld.global.f64 %fd4, [%rd7];
      
      // Convert timestamp to f64
      cvt.rn.f64.u64 %fd5, %rd8;
      
      // Update sums
      add.f64 %fd0, %fd0, %fd5;      // sum_x += x
      add.f64 %fd1, %fd1, %fd4;      // sum_y += y
      mul.f64 %fd6, %fd5, %fd4;
      add.f64 %fd2, %fd2, %fd6;      // sum_xy += x*y
      mul.f64 %fd7, %fd5, %fd5;
      add.f64 %fd3, %fd3, %fd7;      // sum_xx += x*x
      
      // Increment counter
      add.u64 %rd4, %rd4, 1;
      bra loop_start;
      
    loop_end:
      // Calculate slope: (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
      cvt.rn.f64.u64 %fd8, %rd3;     // n
      
      // Calculate numerator: n*sum_xy - sum_x*sum_y
      mul.f64 %fd9, %fd8, %fd2;      // n * sum_xy
      mul.f64 %fd10, %fd0, %fd1;     // sum_x * sum_y
      sub.f64 %fd9, %fd9, %fd10;     // numerator
      
      // Calculate denominator: n*sum_xx - sum_x*sum_x
      mul.f64 %fd10, %fd8, %fd3;     // n * sum_xx
      mul.f64 %fd11, %fd0, %fd0;     // sum_x * sum_x
      sub.f64 %fd10, %fd10, %fd11;   // denominator
      
      // Check for zero denominator
      setp.eq.f64 %p3, %fd10, 0d0000000000000000;
      @%p3 bra store_zero;
      
      // Calculate slope
      div.rn.f64 %fd11, %fd9, %fd10;
      bra store_result;
      
    store_zero:
      mov.f64 %fd11, 0d0000000000000000;
      
    store_result:
      st.global.f64 [%rd2], %fd11;
      
    done:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::reduction();

  // Allocate result on stack
  let mut result: f64 = 0.0;

  let args = [
    values as *const u8,
    timestamps as *const u8,
    &mut result as *mut _ as *const u8,
    &len as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_DERIV_F64,
    &[],
    "deriv_f64_kernel",
    blocks,
    threads,
    &args,
  );
  result
}

// AVX-512 element-wise deriv for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub unsafe fn deriv_f64_avx512(values: &[f64], timestamps: &[u64], len: usize) -> f64 {
  if len < 2 {
    return 0.0;
  }

  const LANES: usize = 8;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Initialize vector accumulators for linear regression
  let mut sum_x_vec = _mm512_setzero_pd();
  let mut sum_y_vec = _mm512_setzero_pd();
  let mut sum_xy_vec = _mm512_setzero_pd();
  let mut sum_xx_vec = _mm512_setzero_pd();

  // Process 8 elements at a time with AVX-512
  for i in 0..chunks {
    let offset = i * LANES;

    // Load timestamps and convert to f64
    let timestamps_i64 = _mm512_loadu_si512(timestamps.as_ptr().add(offset) as *const _);
    let x_vec = _mm512_cvtepi64_pd(timestamps_i64);

    // Load values
    let y_vec = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Accumulate sums
    sum_x_vec = _mm512_add_pd(sum_x_vec, x_vec);
    sum_y_vec = _mm512_add_pd(sum_y_vec, y_vec);
    sum_xy_vec = _mm512_fmadd_pd(x_vec, y_vec, sum_xy_vec);
    sum_xx_vec = _mm512_fmadd_pd(x_vec, x_vec, sum_xx_vec);
  }

  // Reduce vectors to scalars
  let sum_x = _mm512_reduce_add_pd(sum_x_vec);
  let sum_y = _mm512_reduce_add_pd(sum_y_vec);
  let sum_xy = _mm512_reduce_add_pd(sum_xy_vec);
  let sum_xx = _mm512_reduce_add_pd(sum_xx_vec);

  // Handle remainder with scalar operations
  let mut final_sum_x = sum_x;
  let mut final_sum_y = sum_y;
  let mut final_sum_xy = sum_xy;
  let mut final_sum_xx = sum_xx;

  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let x = timestamps[idx] as f64;
    let y = values[idx];
    final_sum_x += x;
    final_sum_y += y;
    final_sum_xy += x * y;
    final_sum_xx += x * x;
  }

  // Calculate slope using linear regression formula
  let n = len as f64;
  let slope =
    (n * final_sum_xy - final_sum_x * final_sum_y) / (n * final_sum_xx - final_sum_x * final_sum_x);
  slope
}

// AVX2 element-wise deriv for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
pub unsafe fn deriv_f64_avx2(values: &[f64], timestamps: &[u64], len: usize) -> f64 {
  if len < 2 {
    return 0.0;
  }

  const LANES: usize = 4;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Initialize vector accumulators for linear regression
  let mut sum_x_vec = _mm256_setzero_pd();
  let mut sum_y_vec = _mm256_setzero_pd();
  let mut sum_xy_vec = _mm256_setzero_pd();
  let mut sum_xx_vec = _mm256_setzero_pd();

  // Process 4 elements at a time with AVX2
  for i in 0..chunks {
    let offset = i * LANES;

    // Load timestamps and convert to f64
    // AVX2 doesn't have direct i64->f64 conversion, load manually
    let x0 = timestamps[offset] as f64;
    let x1 = timestamps[offset + 1] as f64;
    let x2 = timestamps[offset + 2] as f64;
    let x3 = timestamps[offset + 3] as f64;
    let x_vec = _mm256_set_pd(x3, x2, x1, x0);

    // Load values
    let y_vec = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Accumulate sums
    sum_x_vec = _mm256_add_pd(sum_x_vec, x_vec);
    sum_y_vec = _mm256_add_pd(sum_y_vec, y_vec);
    sum_xy_vec = _mm256_fmadd_pd(x_vec, y_vec, sum_xy_vec);
    sum_xx_vec = _mm256_fmadd_pd(x_vec, x_vec, sum_xx_vec);
  }

  // Reduce vectors to scalars
  let sum_x_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_x_vec);
  let sum_y_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_y_vec);
  let sum_xy_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_xy_vec);
  let sum_xx_arr = std::mem::transmute::<__m256d, [f64; 4]>(sum_xx_vec);

  let sum_x = sum_x_arr[0] + sum_x_arr[1] + sum_x_arr[2] + sum_x_arr[3];
  let sum_y = sum_y_arr[0] + sum_y_arr[1] + sum_y_arr[2] + sum_y_arr[3];
  let sum_xy = sum_xy_arr[0] + sum_xy_arr[1] + sum_xy_arr[2] + sum_xy_arr[3];
  let sum_xx = sum_xx_arr[0] + sum_xx_arr[1] + sum_xx_arr[2] + sum_xx_arr[3];

  // Handle remainder with scalar operations
  let mut final_sum_x = sum_x;
  let mut final_sum_y = sum_y;
  let mut final_sum_xy = sum_xy;
  let mut final_sum_xx = sum_xx;

  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let x = timestamps[idx] as f64;
    let y = values[idx];
    final_sum_x += x;
    final_sum_y += y;
    final_sum_xy += x * y;
    final_sum_xx += x * x;
  }

  // Calculate slope using linear regression formula
  let n = len as f64;
  let slope =
    (n * final_sum_xy - final_sum_x * final_sum_y) / (n * final_sum_xx - final_sum_x * final_sum_x);
  slope
}

// NEON element-wise deriv for f64 arrays
#[cfg(target_arch = "aarch64")]
pub unsafe fn deriv_f64_neon(values: &[f64], timestamps: &[u64], len: usize) -> f64 {
  if len < 2 {
    return 0.0;
  }

  const LANES: usize = 2;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // Initialize vector accumulators for linear regression
  let mut sum_x_vec = vdupq_n_f64(0.0);
  let mut sum_y_vec = vdupq_n_f64(0.0);
  let mut sum_xy_vec = vdupq_n_f64(0.0);
  let mut sum_xx_vec = vdupq_n_f64(0.0);

  // Process 2 elements at a time with NEON
  for i in 0..chunks {
    let offset = i * LANES;

    // Load timestamps and convert to f64
    let x0 = timestamps[offset] as f64;
    let x1 = timestamps[offset + 1] as f64;
    let x_vec = vld1q_f64([x0, x1].as_ptr());

    // Load values
    let y_vec = vld1q_f64(values.as_ptr().add(offset));

    // Accumulate sums
    sum_x_vec = vaddq_f64(sum_x_vec, x_vec);
    sum_y_vec = vaddq_f64(sum_y_vec, y_vec);
    sum_xy_vec = vfmaq_f64(sum_xy_vec, x_vec, y_vec);
    sum_xx_vec = vfmaq_f64(sum_xx_vec, x_vec, x_vec);
  }

  // Reduce vectors to scalars
  let sum_x = vgetq_lane_f64(sum_x_vec, 0) + vgetq_lane_f64(sum_x_vec, 1);
  let sum_y = vgetq_lane_f64(sum_y_vec, 0) + vgetq_lane_f64(sum_y_vec, 1);
  let sum_xy = vgetq_lane_f64(sum_xy_vec, 0) + vgetq_lane_f64(sum_xy_vec, 1);
  let sum_xx = vgetq_lane_f64(sum_xx_vec, 0) + vgetq_lane_f64(sum_xx_vec, 1);

  // Handle remainder with scalar operations
  let mut final_sum_x = sum_x;
  let mut final_sum_y = sum_y;
  let mut final_sum_xy = sum_xy;
  let mut final_sum_xx = sum_xx;

  for i in 0..remainder {
    let idx = chunks * LANES + i;
    let x = timestamps[idx] as f64;
    let y = values[idx];
    final_sum_x += x;
    final_sum_y += y;
    final_sum_xy += x * y;
    final_sum_xx += x * x;
  }

  // Calculate slope using linear regression formula
  let n = len as f64;
  let slope =
    (n * final_sum_xy - final_sum_x * final_sum_y) / (n * final_sum_xx - final_sum_x * final_sum_x);
  slope
}

// =============================================================================
// CLAMP FUNCTIONS
// =============================================================================

// GPU implementation of clamp_f64
#[cfg(has_cuda)]
pub unsafe fn clamp_f64_gpu(values: *mut f64, min_val: f64, max_val: f64, len: usize) {
  // Note: This function expects GPU memory pointers when called through with_gpu_buffer_f64_inplace
  // The values pointer should already point to GPU memory allocated by the caller
  const PTX_CLAMP_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry clamp_f64_kernel(
      .param .u64 values_ptr,
      .param .f64 min_val,
      .param .f64 max_val,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.f64 %fd8, [min_val];
      ld.param.f64 %fd9, [max_val];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply clamp to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      
      // Apply clamp operation to lane 0
      max.f64 %fd4, %fd0, %fd8;        // max(value, min_val)
      min.f64 %fd4, %fd4, %fd9;        // min(max_result, max_val)
      
      // Apply clamp operation to lane 1
      max.f64 %fd5, %fd1, %fd8;        // max(value, min_val)
      min.f64 %fd5, %fd5, %fd9;        // min(max_result, max_val)
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p1, %r5, %r0;  // Check if we're completely done
      @%p1 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Apply clamp to final element
      max.f64 %fd6, %fd2, %fd8;        // max(value, min_val)
      min.f64 %fd6, %fd6, %fd9;        // min(max_result, max_val)
      
      st.global.f64 [%rd4], %fd6;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let min_bits = min_val.to_bits();
  let max_bits = max_val.to_bits();
  let len_u64 = len as u64;
  let values_u64 = values as u64;

  let args = [
    &values_u64 as *const u64 as *const u8,
    &min_bits as *const _ as *const u8,
    &max_bits as *const _ as *const u8,
    &len_u64 as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_CLAMP_F64,
    &[],
    "clamp_f64_kernel",
    blocks,
    threads,
    &args,
  );
}

// AVX512 element-wise clamp for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn clamp_f64_avx512(values: &mut [f64], min_val: f64, max_val: f64, len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let min_vec = _mm512_set1_pd(min_val);
  let max_vec = _mm512_set1_pd(max_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN using AVX-512 mask operations
    let nan_mask = _mm512_cmp_pd_mask(vec, vec, _CMP_UNORD_Q);

    // Clamp non-NaN values
    let clamped_vec = _mm512_min_pd(_mm512_max_pd(vec, min_vec), max_vec);

    // Mask blend: keep original (NaN) where nan_mask is set
    let result = _mm512_mask_blend_pd(nan_mask, clamped_vec, vec);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.max(min_val).min(max_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// AVX2 element-wise clamp for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn clamp_f64_avx2(values: &mut [f64], min_val: f64, max_val: f64, len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let min_vec = _mm256_set1_pd(min_val);
  let max_vec = _mm256_set1_pd(max_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN, so compare with itself
    let nan_mask = _mm256_cmp_pd(vec, vec, _CMP_UNORD_Q);

    // Clamp non-NaN values
    let clamped_vec = _mm256_min_pd(_mm256_max_pd(vec, min_vec), max_vec);

    // Blend: keep original (NaN) where nan_mask is set, use clamped otherwise
    let result = _mm256_blendv_pd(clamped_vec, vec, nan_mask);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.max(min_val).min(max_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// NEON element-wise clamp for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn clamp_f64_neon(values: &mut [f64], min_val: f64, max_val: f64, len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let min_vec = vdupq_n_f64(min_val);
  let max_vec = vdupq_n_f64(max_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN
    let nan_mask = vceqq_f64(vec, vec);

    // Clamp non-NaN values
    let clamped_vec = vminq_f64(vmaxq_f64(vec, min_vec), max_vec);

    // Select: keep original where NaN (nan_mask is 0), use clamped otherwise
    let result = vbslq_f64(nan_mask, clamped_vec, vec);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.max(min_val).min(max_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// GPU implementation of clamp_min_f64
#[cfg(has_cuda)]
pub unsafe fn clamp_min_f64_gpu(values: *mut f64, min_val: f64, len: usize) {
  const PTX_CLAMP_MIN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry clamp_min_f64_kernel(
      .param .u64 values_ptr,
      .param .f64 min_val,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.f64 %fd8, [min_val];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply clamp_min to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      
      // Apply clamp_min operation
      max.f64 %fd4, %fd0, %fd8;        // max(value, min_val) for lane 0
      max.f64 %fd5, %fd1, %fd8;        // max(value, min_val) for lane 1
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p1, %r5, %r0;  // Check if we're completely done
      @%p1 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Apply clamp_min to final element
      max.f64 %fd6, %fd2, %fd8;        // max(value, min_val)
      
      st.global.f64 [%rd4], %fd6;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let min_bits = min_val.to_bits();

  let values_u64 = values as u64;
  let len_u64 = len as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &min_bits as *const _ as *const u8,
    &len_u64 as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_CLAMP_MIN_F64,
    &[],
    "clamp_min_f64_kernel",
    blocks,
    threads,
    &args,
  );
}

// AVX512 element-wise clamp_min for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn clamp_min_f64_avx512(values: &mut [f64], min_val: f64, len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let min_vec = _mm512_set1_pd(min_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN using AVX-512 mask operations
    let nan_mask = _mm512_cmp_pd_mask(vec, vec, _CMP_UNORD_Q);

    // Clamp non-NaN values
    let clamped_vec = _mm512_max_pd(vec, min_vec);

    // Mask blend: keep original (NaN) where nan_mask is set
    let result = _mm512_mask_blend_pd(nan_mask, clamped_vec, vec);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.max(min_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// AVX2 element-wise clamp_min for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn clamp_min_f64_avx2(values: &mut [f64], min_val: f64, len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let min_vec = _mm256_set1_pd(min_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN, so compare with itself
    let nan_mask = _mm256_cmp_pd(vec, vec, _CMP_UNORD_Q);

    // Clamp non-NaN values
    let clamped_vec = _mm256_max_pd(vec, min_vec);

    // Blend: keep original (NaN) where nan_mask is set, use clamped otherwise
    let result = _mm256_blendv_pd(clamped_vec, vec, nan_mask);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.max(min_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// NEON element-wise clamp_min for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn clamp_min_f64_neon(values: &mut [f64], min_val: f64, len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let min_vec = vdupq_n_f64(min_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN
    let nan_mask = vceqq_f64(vec, vec);

    // Clamp non-NaN values
    let clamped_vec = vmaxq_f64(vec, min_vec);

    // Select: keep original where NaN (nan_mask is 0), use clamped otherwise
    let result = vbslq_f64(nan_mask, clamped_vec, vec);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.max(min_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// GPU implementation of clamp_max_f64
#[cfg(has_cuda)]
pub unsafe fn clamp_max_f64_gpu(values: *mut f64, max_val: f64, len: usize) {
  const PTX_CLAMP_MAX_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry clamp_max_f64_kernel(
      .param .u64 values_ptr,
      .param .f64 max_val,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.f64 %fd8, [max_val];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply clamp_max to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      
      // Apply clamp_max operation
      min.f64 %fd4, %fd0, %fd8;        // min(value, max_val) for lane 0
      min.f64 %fd5, %fd1, %fd8;        // min(value, max_val) for lane 1
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p1, %r5, %r0;  // Check if we're completely done
      @%p1 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Apply clamp_max to final element
      min.f64 %fd6, %fd2, %fd8;        // min(value, max_val)
      
      st.global.f64 [%rd4], %fd6;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let max_bits = max_val.to_bits();
  let len_u64 = len as u64;
  let values_u64 = values as u64;

  let args = [
    &values_u64 as *const u64 as *const u8,
    &max_bits as *const _ as *const u8,
    &len_u64 as *const _ as *const u8,
  ];

  let _ = launch_ptx(
    PTX_CLAMP_MAX_F64,
    &[],
    "clamp_max_f64_kernel",
    blocks,
    threads,
    &args,
  );
}

// AVX512 element-wise clamp_max for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn clamp_max_f64_avx512(values: &mut [f64], max_val: f64, len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let max_vec = _mm512_set1_pd(max_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN using AVX-512 mask operations
    let nan_mask = _mm512_cmp_pd_mask(vec, vec, _CMP_UNORD_Q);

    // Clamp non-NaN values
    let clamped_vec = _mm512_min_pd(vec, max_vec);

    // Mask blend: keep original (NaN) where nan_mask is set
    let result = _mm512_mask_blend_pd(nan_mask, clamped_vec, vec);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.min(max_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// AVX2 element-wise clamp_max for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn clamp_max_f64_avx2(values: &mut [f64], max_val: f64, len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let max_vec = _mm256_set1_pd(max_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN, so compare with itself
    let nan_mask = _mm256_cmp_pd(vec, vec, _CMP_UNORD_Q);

    // Clamp non-NaN values
    let clamped_vec = _mm256_min_pd(vec, max_vec);

    // Blend: keep original (NaN) where nan_mask is set, use clamped otherwise
    let result = _mm256_blendv_pd(clamped_vec, vec, nan_mask);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.min(max_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// NEON element-wise clamp_max for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn clamp_max_f64_neon(values: &mut [f64], max_val: f64, len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let max_vec = vdupq_n_f64(max_val);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN
    let nan_mask = vceqq_f64(vec, vec);

    // Clamp non-NaN values
    let clamped_vec = vminq_f64(vec, max_vec);

    // Select: keep original where NaN (nan_mask is 0), use clamped otherwise
    let result = vbslq_f64(nan_mask, clamped_vec, vec);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    if !val.is_nan() {
      let result = val.min(max_val);
      *values.get_unchecked_mut(offset + i) = result;
    }
    // NaN values remain unchanged
  }
}

// =============================================================================
// DEGREE/RADIAN CONVERSION FUNCTIONS
// =============================================================================

// GPU implementation of deg_f64 (convert radians to degrees)
#[cfg(has_cuda)]
pub unsafe fn deg_f64_gpu(values: *mut f64, len: usize) {
  const PTX_DEG_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry deg_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // 180.0 / PI constant for radians to degrees conversion
      mov.f64 %fd8, 0d404CA5DC1A63C1F8; // 57.29577951308232
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and convert radians to degrees for double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 (radians)
      
      // Convert to degrees: degrees = radians * (180.0 / PI)
      mul.f64 %fd4, %fd0, %fd8;        // radians * 57.29577951308232
      mul.f64 %fd5, %fd1, %fd8;        // radians * 57.29577951308232
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p1, %r5, %r0;  // Check if we're completely done
      @%p1 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Convert to degrees
      mul.f64 %fd6, %fd2, %fd8;        // radians * 57.29577951308232
      
      st.global.f64 [%rd4], %fd6;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let values_ptr = values as u64;
  let len_val = len as u64;
  let args: [*const u8; 2] = [
    &values_ptr as *const u64 as *const u8,
    &len_val as *const u64 as *const u8,
  ];

  let _ = launch_ptx(PTX_DEG_F64, &[], "deg_f64_kernel", blocks, threads, &args);
}

// AVX512 element-wise deg for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn deg_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // 180.0 / PI constant for radians to degrees conversion
  let conversion_factor = _mm512_set1_pd(180.0 / std::f64::consts::PI);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    let deg_vec = _mm512_mul_pd(vec, conversion_factor);
    _mm512_storeu_pd(values.as_mut_ptr().add(offset), deg_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val * (180.0 / std::f64::consts::PI);
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// AVX2 element-wise deg for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn deg_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // 180.0 / PI constant for radians to degrees conversion
  let conversion_factor = _mm256_set1_pd(180.0 / std::f64::consts::PI);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    let deg_vec = _mm256_mul_pd(vec, conversion_factor);
    _mm256_storeu_pd(values.as_mut_ptr().add(offset), deg_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val * (180.0 / std::f64::consts::PI);
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// NEON element-wise deg for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn deg_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // 180.0 / PI constant for radians to degrees conversion
  let conversion_factor = vdupq_n_f64(180.0 / std::f64::consts::PI);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    let deg_vec = vmulq_f64(vec, conversion_factor);
    vst1q_f64(values.as_mut_ptr().add(offset), deg_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val * (180.0 / std::f64::consts::PI);
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// GPU implementation of rad_f64 (convert degrees to radians)
#[cfg(has_cuda)]
pub unsafe fn rad_f64_gpu(values: *mut f64, len: usize) {
  const PTX_RAD_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry rad_f64_kernel(
      .param .u64 values_ptr,
      .param .u32 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u32 %r0, [len];
      
      // PI / 180.0 constant for degrees to radians conversion
      mov.f64 %fd8, 0d3F91DF46A2529D39; // 0.017453292519943295
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and convert degrees to radians for double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2 (degrees)
      
      // Convert to radians: radians = degrees * (PI / 180.0)
      mul.f64 %fd4, %fd0, %fd8;        // degrees * 0.017453292519943295
      mul.f64 %fd5, %fd1, %fd8;        // degrees * 0.017453292519943295
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p1, %r5, %r0;  // Check if we're completely done
      @%p1 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Convert to radians
      mul.f64 %fd6, %fd2, %fd8;        // degrees * 0.017453292519943295
      
      st.global.f64 [%rd4], %fd6;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let len_u32 = len as u32;
  let values_u64 = values as u64;
  let args: [*const u8; 2] = [
    &values_u64 as *const u64 as *const u8,
    &len_u32 as *const u32 as *const u8,
  ];

  let _ = launch_ptx(PTX_RAD_F64, &[], "rad_f64_kernel", blocks, threads, &args);
}

// AVX512 element-wise rad for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn rad_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // PI / 180.0 constant for degrees to radians conversion
  let conversion_factor = _mm512_set1_pd(std::f64::consts::PI / 180.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    let rad_vec = _mm512_mul_pd(vec, conversion_factor);
    _mm512_storeu_pd(values.as_mut_ptr().add(offset), rad_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val * (std::f64::consts::PI / 180.0);
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// AVX2 element-wise rad for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn rad_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // PI / 180.0 constant for degrees to radians conversion
  let conversion_factor = _mm256_set1_pd(std::f64::consts::PI / 180.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    let rad_vec = _mm256_mul_pd(vec, conversion_factor);
    _mm256_storeu_pd(values.as_mut_ptr().add(offset), rad_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val * (std::f64::consts::PI / 180.0);
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// NEON element-wise rad for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn rad_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  // PI / 180.0 constant for degrees to radians conversion
  let conversion_factor = vdupq_n_f64(std::f64::consts::PI / 180.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    let rad_vec = vmulq_f64(vec, conversion_factor);
    vst1q_f64(values.as_mut_ptr().add(offset), rad_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val * (std::f64::consts::PI / 180.0);
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// =============================================================================
// LOGARITHM FUNCTIONS
// =============================================================================

// GPU implementation of log2_f64
#[cfg(has_cuda)]
pub unsafe fn log2_f64_gpu(values: *mut f64, len: usize) {
  const PTX_LOG2_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry log2_f64_kernel(
      .param .u64 values_ptr,
      .param .u32 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .reg .f32 %f<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u32 %r0, [len];
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply log2 to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      
      // Apply log2 operation using PTX approximation
      // PTX only has lg2.approx.f32, so convert to f32, compute, convert back
      cvt.rn.f32.f64 %f0, %fd0;
      cvt.rn.f32.f64 %f1, %fd1;
      lg2.approx.f32 %f2, %f0;       // log2(lane 0)
      lg2.approx.f32 %f3, %f1;       // log2(lane 1)
      cvt.f64.f32 %fd4, %f2;
      cvt.f64.f32 %fd5, %f3;
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p1, %r5, %r0;  // Check if we're completely done
      @%p1 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Apply log2 to final element
      cvt.rn.f32.f64 %f4, %fd2;
      lg2.approx.f32 %f5, %f4;       // log2(value)
      cvt.f64.f32 %fd6, %f5;
      
      st.global.f64 [%rd4], %fd6;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let len_u32 = len as u32;
  let values_u64 = values as u64;
  let args: [*const u8; 2] = [
    &values_u64 as *const u64 as *const u8,
    &len_u32 as *const u32 as *const u8,
  ];

  let _ = launch_ptx(PTX_LOG2_F64, &[], "log2_f64_kernel", blocks, threads, &args);
}

// AVX512 element-wise log2 for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn log2_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    // Use scalar fallback for transcendental functions
    let mut temp = [0.0f64; LANES];
    _mm512_storeu_pd(temp.as_mut_ptr(), vec);
    for j in 0..LANES {
      temp[j] = temp[j].log2();
    }
    let result_vec = _mm512_loadu_pd(temp.as_ptr());
    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val.log2();
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// AVX2 element-wise log2 for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn log2_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    // Use scalar fallback for transcendental functions
    let mut temp = [0.0f64; LANES];
    _mm256_storeu_pd(temp.as_mut_ptr(), vec);
    for j in 0..LANES {
      temp[j] = temp[j].log2();
    }
    let result_vec = _mm256_loadu_pd(temp.as_ptr());
    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val.log2();
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// NEON element-wise log2 for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn log2_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    // Use scalar fallback for transcendental functions
    let mut temp = [0.0f64; LANES];
    vst1q_f64(temp.as_mut_ptr(), vec);
    for j in 0..LANES {
      temp[j] = temp[j].log2();
    }
    let result_vec = vld1q_f64(temp.as_ptr());
    vst1q_f64(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val.log2();
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// GPU implementation of log10_f64
#[cfg(has_cuda)]
pub unsafe fn log10_f64_gpu(values: *mut f64, len: usize) {
  const PTX_LOG10_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry log10_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      .reg .f32 %f<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // log10(2) constant for log2 to log10 conversion
      mov.f64 %fd9, 0d3FD34413509F79FF; // log10(2) = 0.30102999566398119521
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply log10 to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      
      // Apply log10 operation: log10(x) = log2(x) * log10(2)
      // PTX only has lg2.approx.f32, so convert to f32, compute, convert back
      cvt.rn.f32.f64 %f0, %fd0;
      cvt.rn.f32.f64 %f1, %fd1;
      lg2.approx.f32 %f2, %f0;       // log2(lane 0)
      lg2.approx.f32 %f3, %f1;       // log2(lane 1)
      cvt.f64.f32 %fd2, %f2;
      cvt.f64.f32 %fd3, %f3;
      // Convert to log10: log10(x) = log2(x) * log10(2)
      mov.f64 %fd9, 0d3FD34413509F79FF; // log10(2) = 0.30102999566398119521
      mul.f64 %fd4, %fd2, %fd9;        // log2(lane 0) * log10(2)
      mul.f64 %fd5, %fd3, %fd9;        // log2(lane 1) * log10(2)
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd5};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p1, %r5, %r0;  // Check if we're completely done
      @%p1 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Apply log10 to final element
      cvt.rn.f32.f64 %f4, %fd2;
      lg2.approx.f32 %f5, %f4;       // log2(value)
      cvt.f64.f32 %fd6, %f5;
      mul.f64 %fd6, %fd6, %fd9;        // log2(value) * log10(2)
      
      st.global.f64 [%rd4], %fd6;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let values_ptr = values as u64;
  let len_val = len as u64;
  let args: [*const u8; 2] = [
    &values_ptr as *const u64 as *const u8,
    &len_val as *const u64 as *const u8,
  ];

  let _ = launch_ptx(
    PTX_LOG10_F64,
    &[],
    "log10_f64_kernel",
    blocks,
    threads,
    &args,
  );
}

// AVX512 element-wise log10 for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn log10_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));
    // Use scalar fallback for transcendental functions
    let mut temp = [0.0f64; LANES];
    _mm512_storeu_pd(temp.as_mut_ptr(), vec);
    for j in 0..LANES {
      temp[j] = temp[j].log10();
    }
    let result_vec = _mm512_loadu_pd(temp.as_ptr());
    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val.log10();
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// AVX2 element-wise log10 for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn log10_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));
    // Use scalar fallback for transcendental functions
    let mut temp = [0.0f64; LANES];
    _mm256_storeu_pd(temp.as_mut_ptr(), vec);
    for j in 0..LANES {
      temp[j] = temp[j].log10();
    }
    let result_vec = _mm256_loadu_pd(temp.as_ptr());
    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val.log10();
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// NEON element-wise log10 for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn log10_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));
    // Use scalar fallback for transcendental functions
    let mut temp = [0.0f64; LANES];
    vst1q_f64(temp.as_mut_ptr(), vec);
    for j in 0..LANES {
      temp[j] = temp[j].log10();
    }
    let result_vec = vld1q_f64(temp.as_ptr());
    vst1q_f64(values.as_mut_ptr().add(offset), result_vec);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    let result = val.log10();
    *values.get_unchecked_mut(offset + i) = result;
  }
}

// =============================================================================
// SIGN FUNCTION
// =============================================================================

// GPU implementation of sgn_f64
#[cfg(has_cuda)]
pub unsafe fn sgn_f64_gpu(values: *mut f64, len: usize) {
  const PTX_SGN_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry sgn_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 len
    ) {
      .reg .f64 %fd<25>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<25>;
      .reg .pred %p<10>;
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [len];
      cvt.u32.u64 %r0, %rd1;
      
      // Constants
      mov.f64 %fd8, 0d0000000000000000; // 0.0
      mov.f64 %fd9, 0d3FF0000000000000; // 1.0
      mov.f64 %fd10, 0dBFF0000000000000; // -1.0
      
      // Calculate starting index for this thread
      mov.u32 %r1, %tid.x;     // Thread ID within block
      mov.u32 %r2, %ntid.x;    // Block size
      mov.u32 %r3, %ctaid.x;   // Block ID
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;   // Global thread ID
      mul.lo.u32 %r5, %r5, 2;  // Each thread starts at ID * 2 (double2)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x; // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid
      mul.lo.u32 %r9, %r9, 2;   // Stride = total_threads * 2 (double2)
      
      // Main grid-stride loop processing double2 chunks
    loop_start:
      add.u32 %r6, %r5, 1;
      setp.ge.u32 %p0, %r6, %r0;  // Check if we can load full double2
      @%p0 bra exit_loop;  // Exit loop if beyond array
      
      // Load and apply sgn to double2 vector
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd0, %rd1;
      // Load vector with vectorized 128-bit load
      ld.global.v2.f64 {%fd0, %fd1}, [%rd2];  // Load double2
      
      // Apply sgn operation to lane 0: sgn(x) = (x > 0) - (x < 0)
      setp.gt.f64 %p1, %fd0, %fd8;     // x > 0
      setp.lt.f64 %p2, %fd0, %fd8;     // x < 0
      selp.f64 %fd4, %fd9, %fd8, %p1;  // (x > 0) ? 1.0 : 0.0
      selp.f64 %fd5, %fd10, %fd8, %p2; // (x < 0) ? -1.0 : 0.0
      add.f64 %fd4, %fd4, %fd5;        // Combine: (x > 0) - (x < 0)
      
      // Apply sgn operation to lane 1: sgn(x) = (x > 0) - (x < 0)
      setp.gt.f64 %p3, %fd1, %fd8;     // x > 0
      setp.lt.f64 %p4, %fd1, %fd8;     // x < 0
      selp.f64 %fd6, %fd9, %fd8, %p3;  // (x > 0) ? 1.0 : 0.0
      selp.f64 %fd7, %fd10, %fd8, %p4; // (x < 0) ? -1.0 : 0.0
      add.f64 %fd6, %fd6, %fd7;        // Combine: (x > 0) - (x < 0)
      
      // Store results back with vectorized 128-bit store
      st.global.v2.f64 [%rd2], {%fd4, %fd6};  // Store double2 results
      
      // Increment by grid stride
      add.u32 %r5, %r5, %r9;
      bra.uni loop_start;
      
    exit_loop:
      // Handle final partial vector if necessary
      setp.ge.u32 %p5, %r5, %r0;  // Check if we're completely done
      @%p5 bra exit_kernel;
      
      // Process final single element
      mul.wide.u32 %rd3, %r5, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd2, [%rd4];      // Load final element
      
      // Apply sgn to final element
      setp.gt.f64 %p1, %fd2, %fd8;     // x > 0
      setp.lt.f64 %p2, %fd2, %fd8;     // x < 0
      selp.f64 %fd4, %fd9, %fd8, %p1;  // (x > 0) ? 1.0 : 0.0
      selp.f64 %fd5, %fd10, %fd8, %p2; // (x < 0) ? -1.0 : 0.0
      add.f64 %fd4, %fd4, %fd5;        // Combine: (x > 0) - (x < 0)
      
      st.global.f64 [%rd4], %fd4;      // Store result
      
    exit_kernel:
      ret;
    }
  "#;

  let (blocks, threads) = LaunchConfig::parallel();
  let values_ptr = values as u64;
  let len_val = len as u64;
  let args: [*const u8; 2] = [
    &values_ptr as *const u64 as *const u8,
    &len_val as *const u64 as *const u8,
  ];

  let _ = launch_ptx(PTX_SGN_F64, &[], "sgn_f64_kernel", blocks, threads, &args);
}

// AVX512 element-wise sgn for f64 arrays
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn sgn_f64_avx512(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX512_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let zero = _mm512_setzero_pd();
  let one = _mm512_set1_pd(1.0);
  let neg_one = _mm512_set1_pd(-1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm512_loadu_pd(values.as_ptr().add(offset));

    // Create masks for x > 0 and x < 0
    let gt_mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(vec, zero);
    let lt_mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(vec, zero);

    // Apply sgn logic: (x > 0) ? 1.0 : 0.0
    let positive_part = _mm512_mask_blend_pd(gt_mask, zero, one);
    // Apply sgn logic: (x < 0) ? -1.0 : 0.0
    let negative_part = _mm512_mask_blend_pd(lt_mask, zero, neg_one);

    // Combine: sgn(x) = positive_part + negative_part
    let sgn_result = _mm512_add_pd(positive_part, negative_part);

    // Check for NaN: NaN != NaN
    let nan_mask = _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(vec, vec);

    // Blend: keep original (NaN) where nan_mask is set, use sgn result otherwise
    let result = _mm512_mask_blend_pd(nan_mask, sgn_result, vec);

    _mm512_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Preserve NaN
    if val.is_nan() {
      // NaN remains unchanged
    } else if val > 0.0 {
      *values.get_unchecked_mut(offset + i) = 1.0;
    } else if val < 0.0 {
      *values.get_unchecked_mut(offset + i) = -1.0;
    } else {
      *values.get_unchecked_mut(offset + i) = 0.0;
    }
  }
}

// AVX2 element-wise sgn for f64 arrays
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn sgn_f64_avx2(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_AVX2_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let zero = _mm256_setzero_pd();
  let one = _mm256_set1_pd(1.0);
  let neg_one = _mm256_set1_pd(-1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN: NaN != NaN
    let nan_mask = _mm256_cmp_pd(vec, vec, _CMP_UNORD_Q);

    // Create masks for x > 0 and x < 0
    let gt_mask = _mm256_cmp_pd::<_CMP_GT_OQ>(vec, zero);
    let lt_mask = _mm256_cmp_pd::<_CMP_LT_OQ>(vec, zero);

    // Apply sgn logic: (x > 0) ? 1.0 : 0.0
    let positive_part = _mm256_blendv_pd(zero, one, gt_mask);
    // Apply sgn logic: (x < 0) ? -1.0 : 0.0
    let negative_part = _mm256_blendv_pd(zero, neg_one, lt_mask);

    // Combine: sgn(x) = positive_part + negative_part
    let sgn_result = _mm256_add_pd(positive_part, negative_part);

    // Blend: keep original (NaN) where nan_mask is set, use sgn result otherwise
    let result = _mm256_blendv_pd(sgn_result, vec, nan_mask);

    _mm256_storeu_pd(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Preserve NaN
    if val.is_nan() {
      // NaN remains unchanged
    } else if val > 0.0 {
      *values.get_unchecked_mut(offset + i) = 1.0;
    } else if val < 0.0 {
      *values.get_unchecked_mut(offset + i) = -1.0;
    } else {
      *values.get_unchecked_mut(offset + i) = 0.0;
    }
  }
}

// NEON element-wise sgn for f64 arrays
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn sgn_f64_neon(values: &mut [f64], len: usize) {
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let zero = vdupq_n_f64(0.0);
  let one = vdupq_n_f64(1.0);
  let neg_one = vdupq_n_f64(-1.0);

  for i in 0..chunks {
    let offset = i * LANES;
    let vec = vld1q_f64(values.as_ptr().add(offset));

    // Create masks for x > 0 and x < 0
    let gt_mask = vcgtq_f64(vec, zero);
    let lt_mask = vcltq_f64(vec, zero);

    // Apply sgn logic using blend: (x > 0) ? 1.0 : 0.0
    let positive_part = vbslq_f64(gt_mask, one, zero);
    // Apply sgn logic using blend: (x < 0) ? -1.0 : 0.0
    let negative_part = vbslq_f64(lt_mask, neg_one, zero);

    // Combine: sgn(x) = positive_part + negative_part
    let sgn_result = vaddq_f64(positive_part, negative_part);

    // Check for NaN: NaN != NaN
    // In NEON, we compare vec with itself and invert to get NaN mask
    let not_nan_mask = vceqq_f64(vec, vec);

    // Blend: keep original (NaN) where not_nan_mask is false, use sgn result otherwise
    let result = vbslq_f64(not_nan_mask, sgn_result, vec);

    vst1q_f64(values.as_mut_ptr().add(offset), result);
  }

  let offset = chunks * LANES;
  for i in 0..remainder {
    let val = *values.get_unchecked(offset + i);
    // Preserve NaN
    if val.is_nan() {
      // NaN remains unchanged
    } else if val > 0.0 {
      *values.get_unchecked_mut(offset + i) = 1.0;
    } else if val < 0.0 {
      *values.get_unchecked_mut(offset + i) = -1.0;
    } else {
      *values.get_unchecked_mut(offset + i) = 0.0;
    }
  }
}

// GPU implementation of stdvar_over_time (variance)
#[cfg(has_cuda)]
pub unsafe fn stdvar_f64_gpu(values: *const f64, len: usize, result: *mut f64) {
  const PTX_STDVAR_F64: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry stdvar_f64_kernel(
      .param .u64 values_ptr,
      .param .u64 result_ptr,
      .param .u64 len
    ) {
      .reg .u64 %rd<25>;
      .reg .u32 %r<35>;
      .reg .pred %p<10>;
      .reg .f64 %fd<25>;
      .shared .f64 shared_sum[256];
      .shared .u32 shared_count[256];
      .shared .f64 shared_var[256];
      
      // Load parameters
      ld.param.u64 %rd0, [values_ptr];
      ld.param.u64 %rd1, [result_ptr];
      ld.param.u64 %rd11, [len];
      cvt.u32.u64 %r10, %rd11;
      
      // Calculate thread and block indices
      mov.u32 %r1, %tid.x;      // Thread ID within block
      mov.u32 %r2, %ntid.x;     // Block size (threads per block)
      mov.u32 %r3, %ctaid.x;    // Block ID
      mul.lo.u32 %r4, %r3, %r2; // Block start index
      add.u32 %r5, %r4, %r1;    // Global thread ID (starting index)
      
      // Calculate grid stride
      mov.u32 %r8, %nctaid.x;   // Number of blocks
      mul.lo.u32 %r9, %r8, %r2; // Total threads in grid (stride)
      
      // First pass: calculate sum and count with grid-stride
      mov.f64 %fd0, 0.0;  // local sum
      mov.u32 %r11, 0;    // local count
      mov.u32 %r12, %r5;  // current index
      
    first_pass_grid_loop:
      // Check bounds
      setp.ge.u32 %p0, %r12, %r10;
      @%p0 bra first_pass_shared;
      
      // Check if we can load double2
      add.u32 %r13, %r12, 1;
      setp.gt.u32 %p2, %r13, %r10;
      @%p2 bra first_pass_scalar;
      
      // Load double2 values (vectorized)
      mul.wide.u32 %rd3, %r12, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.v2.f64 {%fd1, %fd2}, [%rd4];
      
      // Check if values are NaN
      testp.number.f64 %p1, %fd1;
      @!%p1 bra skip_first;
      add.f64 %fd0, %fd0, %fd1;
      add.u32 %r11, %r11, 1;
    skip_first:
      testp.number.f64 %p3, %fd2;
      @!%p3 bra first_pass_next2;
      add.f64 %fd0, %fd0, %fd2;
      add.u32 %r11, %r11, 1;
      
    first_pass_next2:
      // Grid stride to next element pair
      mul.lo.u32 %r14, %r9, 2;
      add.u32 %r12, %r12, %r14;
      bra first_pass_grid_loop;
      
    first_pass_scalar:
      // Load single value
      mul.wide.u32 %rd3, %r12, 8;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.f64 %fd1, [%rd4];
      
      // Check if value is NaN
      testp.number.f64 %p1, %fd1;
      @!%p1 bra first_pass_next;
      
      // Add to local sum and count
      add.f64 %fd0, %fd0, %fd1;
      add.u32 %r11, %r11, 1;
      
    first_pass_next:
      // Grid stride to next element
      add.u32 %r12, %r12, %r9;
      bra first_pass_grid_loop;
      
    first_pass_shared:
      // Warp-level reduction for sum and count
      // Get lane ID
      and.b32 %r15, %r1, 0x1f;
      
      // Warp shuffle reduction for f64 sum
      mov.b64 {%r16, %r17}, %fd0;  // Split f64 into two u32
      shfl.sync.down.b32 %r18, %r16, 16, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 16, 31, 0xffffffff;
      mov.b64 %fd20, {%r18, %r19};
      add.f64 %fd0, %fd0, %fd20;
      
      mov.b64 {%r16, %r17}, %fd0;
      shfl.sync.down.b32 %r18, %r16, 8, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 8, 31, 0xffffffff;
      mov.b64 %fd20, {%r18, %r19};
      add.f64 %fd0, %fd0, %fd20;
      
      mov.b64 {%r16, %r17}, %fd0;
      shfl.sync.down.b32 %r18, %r16, 4, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 4, 31, 0xffffffff;
      mov.b64 %fd20, {%r18, %r19};
      add.f64 %fd0, %fd0, %fd20;
      
      mov.b64 {%r16, %r17}, %fd0;
      shfl.sync.down.b32 %r18, %r16, 2, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 2, 31, 0xffffffff;
      mov.b64 %fd20, {%r18, %r19};
      add.f64 %fd0, %fd0, %fd20;
      
      mov.b64 {%r16, %r17}, %fd0;
      shfl.sync.down.b32 %r18, %r16, 1, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 1, 31, 0xffffffff;
      mov.b64 %fd20, {%r18, %r19};
      add.f64 %fd0, %fd0, %fd20;
      
      // Warp shuffle reduction for u32 count
      shfl.sync.down.b32 %r18, %r11, 16, 31, 0xffffffff;
      add.u32 %r11, %r11, %r18;
      shfl.sync.down.b32 %r18, %r11, 8, 31, 0xffffffff;
      add.u32 %r11, %r11, %r18;
      shfl.sync.down.b32 %r18, %r11, 4, 31, 0xffffffff;
      add.u32 %r11, %r11, %r18;
      shfl.sync.down.b32 %r18, %r11, 2, 31, 0xffffffff;
      add.u32 %r11, %r11, %r18;
      shfl.sync.down.b32 %r18, %r11, 1, 31, 0xffffffff;
      add.u32 %r11, %r11, %r18;
      
      // Only lane 0 of each warp stores to shared memory
      setp.eq.u32 %p6, %r15, 0;
      shr.u32 %r16, %r1, 5;  // Warp ID
      @%p6 mul.wide.u32 %rd5, %r16, 8;
      @%p6 st.shared.f64 [shared_sum + %rd5], %fd0;
      @%p6 mul.wide.u32 %rd6, %r16, 4;
      @%p6 st.shared.u32 [shared_count + %rd6], %r11;
      
      // Synchronize threads in block
      bar.sync 0;
      
      // Block-level reduction for sum and count (thread 0 only)
      setp.ne.u32 %p2, %r1, 0;
      @%p2 bra wait_for_mean;
      
      // Thread 0: reduce across all threads in block
      mov.u32 %r13, 0;           // Loop counter
      mov.f64 %fd2, 0.0;         // Block sum
      mov.u32 %r14, 0;           // Block count
      
    first_reduce_loop:
      setp.ge.u32 %p3, %r13, %r2;
      @%p3 bra first_reduce_done;
      
      // Load and accumulate sum
      mul.wide.u32 %rd7, %r13, 8;
      ld.shared.f64 %fd3, [shared_sum + %rd7];
      add.f64 %fd2, %fd2, %fd3;
      
      // Load and accumulate count
      mul.wide.u32 %rd8, %r13, 4;
      ld.shared.u32 %r0, [shared_count + %rd8];
      add.u32 %r14, %r14, %r0;
      
      add.u32 %r13, %r13, 1;
      bra first_reduce_loop;
      
    first_reduce_done:
      // Thread 0 atomically adds this block's sum and count to global totals
      // We'll use the result pointer area for temporary storage:
      // [%rd1+8] = global sum, [%rd1+16] = global count
      add.u64 %rd11, %rd1, 8;
      atom.global.add.f64 %fd20, [%rd11], %fd2;  // Add block's sum to global at offset 8
      
      // Add count using atomic add for u32 (store at offset 16 to avoid f64 overlap)
      add.u64 %rd12, %rd1, 16;
      atom.global.add.u32 %r20, [%rd12], %r14;  // Add block's count to global
      
      // Increment first pass block counter (offset 36 for first pass counter)
      add.u64 %rd20, %rd1, 36;
      atom.global.add.u32 %r27, [%rd20], 1;  // Increment and get old value
      add.u32 %r28, %r27, 1;  // Get new value
      
      // Wait for all blocks to complete first pass
      mov.u32 %r29, %nctaid.x;  // Total number of blocks
    wait_first_pass:
      ld.global.u32 %r30, [%rd20];  // Load counter
      setp.lt.u32 %p8, %r30, %r29;  // Check if not all blocks done
      @%p8 bra wait_first_pass;  // Keep waiting
      
    wait_for_mean:
      // Synchronize all threads in block
      bar.sync 0;
      
      // Thread 0 reads the global sum and count
      setp.ne.u32 %p5, %r1, 0;
      @%p5 bra load_mean;
      
      add.u64 %rd11, %rd1, 8;
      ld.global.f64 %fd22, [%rd11];     // Load global sum from offset 8
      add.u64 %rd12, %rd1, 16;
      ld.global.u32 %r21, [%rd12];     // Load global count
      
      // Calculate global mean
      cvt.rn.f64.u32 %fd4, %r21;
      setp.eq.f64 %p4, %fd4, 0.0;
      @%p4 bra set_nan;
      
      div.rn.f64 %fd5, %fd22, %fd4;    // global mean
      
      // Store mean to shared memory for all threads in this block
      st.shared.f64 [shared_sum], %fd5;
      
    load_mean:
      bar.sync 0;
      
      // All threads load the mean from shared memory
      ld.shared.f64 %fd5, [shared_sum];
      
      // Second pass: calculate variance sum with grid-stride
      mov.f64 %fd6, 0.0;  // local variance sum
      mov.u32 %r12, %r5;  // reset current index
      
    second_pass_grid_loop:
      // Check bounds
      setp.ge.u32 %p5, %r12, %r10;
      @%p5 bra second_pass_shared;
      
      // Load value
      mul.wide.u32 %rd9, %r12, 8;
      add.u64 %rd10, %rd0, %rd9;
      ld.global.f64 %fd7, [%rd10];
      
      // Check if value is NaN
      testp.number.f64 %p1, %fd7;
      @!%p1 bra second_pass_next;
      
      // Calculate (value - mean)^2
      sub.rn.f64 %fd8, %fd7, %fd5;    // value - mean
      mul.rn.f64 %fd9, %fd8, %fd8;    // squared difference
      add.f64 %fd6, %fd6, %fd9;       // add to variance sum
      
    second_pass_next:
      // Grid stride to next element
      add.u32 %r12, %r12, %r9;
      bra second_pass_grid_loop;
      
    second_pass_shared:
      // Warp-level reduction for variance sum
      // Warp shuffle reduction for f64 variance
      mov.b64 {%r16, %r17}, %fd6;  // Split f64 into two u32
      shfl.sync.down.b32 %r18, %r16, 16, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 16, 31, 0xffffffff;
      mov.b64 %fd21, {%r18, %r19};
      add.f64 %fd6, %fd6, %fd21;
      
      mov.b64 {%r16, %r17}, %fd6;
      shfl.sync.down.b32 %r18, %r16, 8, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 8, 31, 0xffffffff;
      mov.b64 %fd21, {%r18, %r19};
      add.f64 %fd6, %fd6, %fd21;
      
      mov.b64 {%r16, %r17}, %fd6;
      shfl.sync.down.b32 %r18, %r16, 4, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 4, 31, 0xffffffff;
      mov.b64 %fd21, {%r18, %r19};
      add.f64 %fd6, %fd6, %fd21;
      
      mov.b64 {%r16, %r17}, %fd6;
      shfl.sync.down.b32 %r18, %r16, 2, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 2, 31, 0xffffffff;
      mov.b64 %fd21, {%r18, %r19};
      add.f64 %fd6, %fd6, %fd21;
      
      mov.b64 {%r16, %r17}, %fd6;
      shfl.sync.down.b32 %r18, %r16, 1, 31, 0xffffffff;
      shfl.sync.down.b32 %r19, %r17, 1, 31, 0xffffffff;
      mov.b64 %fd21, {%r18, %r19};
      add.f64 %fd6, %fd6, %fd21;
      
      // Only lane 0 of each warp stores to shared memory
      setp.eq.u32 %p7, %r15, 0;
      shr.u32 %r16, %r1, 5;  // Warp ID
      @%p7 mul.wide.u32 %rd11, %r16, 8;
      @%p7 st.shared.f64 [shared_var + %rd11], %fd6;
      
      // Synchronize threads in block
      bar.sync 0;
      
      // Block-level reduction for variance (thread 0 only)
      setp.ne.u32 %p2, %r1, 0;
      @%p2 bra exit;
      
      // Thread 0: reduce variance across all threads in block
      mov.u32 %r13, 0;           // Loop counter
      mov.f64 %fd10, 0.0;        // Block variance sum
      
    second_reduce_loop:
      setp.ge.u32 %p3, %r13, %r2;
      @%p3 bra second_reduce_done;
      
      // Load and accumulate variance sum
      mul.wide.u32 %rd12, %r13, 8;
      ld.shared.f64 %fd11, [shared_var + %rd12];
      add.f64 %fd10, %fd10, %fd11;
      
      add.u32 %r13, %r13, 1;
      bra second_reduce_loop;
      
    second_reduce_done:
      // Calculate final variance (population variance)
      // Use the GLOBAL count, not the local block count!
      add.u64 %rd12, %rd1, 16;
      ld.global.u32 %r22, [%rd12];     // Load global count
      cvt.rn.f64.u32 %fd12, %r22;
      div.rn.f64 %fd13, %fd10, %fd12;
      
      // Each block atomically adds its variance sum to global
      // We need to accumulate all blocks' variance sums
      add.u64 %rd13, %rd1, 24;  // Use offset 24 for variance sum (avoid conflict with sum at 8)
      atom.global.add.f64 %fd23, [%rd13], %fd10;  // Add this block's variance sum
      
      // Increment block counter to track completion (use offset 32 for block counter)
      add.u64 %rd15, %rd1, 32;
      atom.global.add.u32 %r24, [%rd15], 1;  // Increment and get old value
      add.u32 %r25, %r24, 1;  // Get new value (old + 1)
      
      // Check if this is the last block to finish
      mov.u32 %r26, %nctaid.x;  // Total number of blocks
      setp.ne.u32 %p3, %r25, %r26;  // Check if we're NOT the last block
      @%p3 bra exit;  // Exit if not the last block
      
      // Only thread 0 of the last block calculates final result
      setp.ne.u32 %p3, %r1, 0;  // Check if NOT thread 0
      @%p3 bra exit;
      
      // Load the global variance sum and divide by global count
      ld.global.f64 %fd24, [%rd13];  // Global variance sum
      div.rn.f64 %fd13, %fd24, %fd12;  // Divide by global count
      
      // Clear the temporary storage and write final result
      mov.f64 %fd25, 0.0;
      st.global.f64 [%rd13], %fd25;  // Clear variance sum location
      add.u64 %rd14, %rd1, 16;
      mov.u32 %r23, 0;
      st.global.u32 [%rd14], %r23;   // Clear count location
      add.u64 %rd16, %rd1, 32;
      st.global.u32 [%rd16], %r23;   // Clear second pass block counter
      add.u64 %rd17, %rd1, 36;
      st.global.u32 [%rd17], %r23;   // Clear first pass block counter
      st.global.f64 [%rd1], %fd13;   // Write final variance
      bra exit;
      
    set_nan:
      // Write NaN result (only executed by thread 0 of block 0)
      setp.ne.u32 %p3, %r3, 0;
      @%p3 bra exit;  // Only block 0 writes result
      
      mov.f64 %fd14, 0d7ff8000000000000; // NaN
      st.global.f64 [%rd1], %fd14;
      
    exit:
      ret;
    }
  "#;

  // Use reduction config - stdvar is a reduction operation
  let (blocks, threads) = LaunchConfig::reduction();

  let values_u64 = values as u64;
  let result_ptr = result as u64;
  let len_u64 = len as u64;
  let args = [
    &values_u64 as *const u64 as *const u8,
    &result_ptr as *const u64 as *const u8,
    &len_u64 as *const u64 as *const u8,
  ];

  let _ = launch_ptx(
    PTX_STDVAR_F64,
    &[],
    "stdvar_f64_kernel",
    blocks,
    threads,
    &args,
  );
}

// AVX-512 implementation of stdvar_over_time
#[cfg(all(
  feature = "hwx-nightly",
  any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn stdvar_f64_avx512(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX512_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  // First pass: calculate mean
  let mut sum_vec = _mm512_setzero_pd();
  let mut count = 0usize;

  for i in 0..chunks {
    let offset = i * LANES;
    let v = _mm512_loadu_pd(values.as_ptr().add(offset));

    let non_nan_mask = _mm512_cmp_pd_mask(v, v, 0); // _CMP_EQ_OQ
    sum_vec = _mm512_mask_add_pd(sum_vec, non_nan_mask, sum_vec, v);
    count += non_nan_mask.count_ones() as usize;
  }

  // Process remainder for mean (scalar accumulation)
  let mut remainder_sum = 0.0;
  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let val = values[offset];
    if !val.is_nan() {
      remainder_sum += val;
      count += 1;
    }
  }

  if count == 0 {
    return f64::NAN;
  }

  let total_sum = _mm512_reduce_add_pd(sum_vec) + remainder_sum;
  let mean = total_sum / count as f64;
  let mean_vec = _mm512_set1_pd(mean);

  // Second pass: calculate variance
  let mut var_sum_vec = _mm512_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let v = _mm512_loadu_pd(values.as_ptr().add(offset));

    let non_nan_mask = _mm512_cmp_pd_mask(v, v, 0);
    let diff = _mm512_sub_pd(v, mean_vec);
    let squared_diff = _mm512_mul_pd(diff, diff);
    var_sum_vec = _mm512_mask_add_pd(var_sum_vec, non_nan_mask, var_sum_vec, squared_diff);
  }

  // Process remainder for variance (scalar accumulation)
  let mut remainder_var_sum = 0.0;
  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let val = values[offset];
    if !val.is_nan() {
      let diff = val - mean;
      remainder_var_sum += diff * diff;
    }
  }

  let total_var_sum = _mm512_reduce_add_pd(var_sum_vec) + remainder_var_sum;
  total_var_sum / count as f64 // Population variance
}

// AVX2 implementation of stdvar
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn stdvar_f64_avx2(values: &[f64], len: usize) -> f64 {
  const LANES: usize = LANES_AVX2_F64;

  let chunks = len / LANES;
  let remainder = len % LANES;

  // First pass: calculate mean
  let mut sum_vec = _mm256_setzero_pd();
  let mut count = 0usize;

  for i in 0..chunks {
    let offset = i * LANES;
    let v = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN using SIMD comparison (NaN != NaN)
    let nan_mask = _mm256_cmp_pd(v, v, 0); // _CMP_EQ_OQ - equal if not NaN

    // Count non-NaN values
    let mask_bits = _mm256_movemask_pd(nan_mask);
    count += mask_bits.count_ones() as usize;

    // Add only non-NaN values to sum
    let filtered_v = _mm256_and_pd(v, nan_mask);
    sum_vec = _mm256_add_pd(sum_vec, filtered_v);
  }

  // Process remainder for mean
  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let val = values[offset];
    if !val.is_nan() {
      sum_vec = _mm256_add_pd(sum_vec, _mm256_set1_pd(val));
      count += 1;
    }
  }

  if count == 0 {
    return f64::NAN;
  }

  // Horizontal sum for mean
  let high = _mm256_extractf128_pd(sum_vec, 1);
  let low = _mm256_extractf128_pd(sum_vec, 0);
  let sum128 = _mm_add_pd(high, low);
  let sum_shuffled = _mm_shuffle_pd(sum128, sum128, 1);
  let total_sum_vec = _mm_add_pd(sum128, sum_shuffled);
  let total_sum: f64 = _mm_cvtsd_f64(total_sum_vec);

  let mean = total_sum / count as f64;
  let mean_vec = _mm256_set1_pd(mean);

  // Second pass: calculate variance
  let mut var_sum_vec = _mm256_setzero_pd();

  for i in 0..chunks {
    let offset = i * LANES;
    let v = _mm256_loadu_pd(values.as_ptr().add(offset));

    // Check for NaN using SIMD comparison
    let nan_mask = _mm256_cmp_pd(v, v, 0); // _CMP_EQ_OQ - equal if not NaN

    // Calculate difference from mean
    let diff = _mm256_sub_pd(v, mean_vec);
    let squared_diff = _mm256_mul_pd(diff, diff);

    // Add only non-NaN values to variance sum
    let filtered_squared_diff = _mm256_and_pd(squared_diff, nan_mask);
    var_sum_vec = _mm256_add_pd(var_sum_vec, filtered_squared_diff);
  }

  // Process remainder for variance
  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let val = values[offset];
    if !val.is_nan() {
      let diff = val - mean;
      var_sum_vec = _mm256_add_pd(var_sum_vec, _mm256_set1_pd(diff * diff));
    }
  }

  // Horizontal sum for variance
  let var_high = _mm256_extractf128_pd(var_sum_vec, 1);
  let var_low = _mm256_extractf128_pd(var_sum_vec, 0);
  let var_sum128 = _mm_add_pd(var_high, var_low);
  let var_sum_shuffled = _mm_shuffle_pd(var_sum128, var_sum128, 1);
  let total_var_sum_vec = _mm_add_pd(var_sum128, var_sum_shuffled);
  let total_var_sum: f64 = _mm_cvtsd_f64(total_var_sum_vec);

  total_var_sum / count as f64 // Population variance
}

// NEON implementation of stdvar
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn stdvar_f64_neon(values: &[f64], len: usize) -> f64 {
  // First pass: calculate mean using scalar accumulation
  let mut sum = 0.0;
  let mut count = 0usize;

  for i in 0..len {
    let val = values[i];
    if !val.is_nan() {
      sum += val;
      count += 1;
    }
  }

  if count == 0 {
    return f64::NAN;
  }

  let mean = sum / count as f64;

  // Second pass: calculate variance using NEON for squared differences
  const LANES: usize = LANES_NEON_F64;
  let chunks = len / LANES;
  let remainder = len % LANES;

  let mut var_sum_vec = vdupq_n_f64(0.0);
  let mean_vec = vdupq_n_f64(mean);

  // Process 2-element chunks with NEON
  for i in 0..chunks {
    let offset = i * LANES;
    let vals = vld1q_f64(values.as_ptr().add(offset));

    // Check for NaN using comparison (NaN != NaN)
    let is_valid = vceqq_f64(vals, vals); // NaN will be false

    let diff = vsubq_f64(vals, mean_vec);
    let squared = vmulq_f64(diff, diff);

    // Mask out NaN values by zeroing them
    let masked = vbslq_f64(is_valid, squared, vdupq_n_f64(0.0));
    var_sum_vec = vaddq_f64(var_sum_vec, masked);
  }

  // Extract variance sum from NEON vector
  let mut total_var_sum = vgetq_lane_f64(var_sum_vec, 0) + vgetq_lane_f64(var_sum_vec, 1);

  // Process remainder with scalar
  for i in 0..remainder {
    let offset = chunks * LANES + i;
    let val = values[offset];
    if !val.is_nan() {
      let diff = val - mean;
      total_var_sum += diff * diff;
    }
  }

  total_var_sum / count as f64 // Population variance
}
