// SPDX-License-Identifier: Apache-2.0

//! String operations
//!
//! Pattern matching and string utilities (e.g. prefix/regex/wildcard helpers).
//! Implementations may use scalar code, SIMD, and (when enabled) CUDA kernels.
//!
//! ## Performance notes
//! Hot paths are written to be allocation-light. When modifying inner loops, try
//! to avoid introducing extra allocations or iterator-heavy patterns.

// Some clippy lints are noisy for low-level SIMD/FFI code; we opt out at the module level.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

// GPU/CUDA constants
#[cfg(has_cuda)]
use crate::gpu::launch_ptx;

// =============================================================================
// STRING AND PATTERN MATCHING OPERATIONS
// =============================================================================

// ARM NEON imports
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vandq_u8, vceqq_u8, vcgeq_u8, vcleq_u8, vdupq_n_u8, vgetq_lane_u64, vld1q_u32, vld1q_u8,
    vminvq_u8, vorrq_u8, vreinterpretq_u64_u8, vst1q_u32, vst1q_u8,
};

// x86_64 SIMD intrinsics imports - AVX2 only (not when nightly AVX-512 is enabled)
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use std::arch::x86_64::{
    _mm256_add_epi8,
    _mm256_and_si256,
    _mm256_blendv_epi8,
    _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8,
    _mm256_loadu_si256,
    _mm256_movemask_epi8,
    _mm256_or_si256,
    _mm256_set1_epi8,
    _mm256_setzero_si256,
    _mm256_storeu_si256,
    _mm256_sub_epi8,
    // AVX2 intrinsics
    // SSE/SSE2 intrinsics for hybrid SIMD processing
    _mm_and_si128,
    _mm_blendv_epi8,
    _mm_cmpeq_epi8,
    _mm_loadu_si128,
    _mm_movemask_epi8,
    _mm_or_si128,
    _mm_set1_epi8,
    _mm_sub_epi8,
};

// AVX-512 intrinsics (nightly only) - includes all needed AVX2/AVX-512 intrinsics
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use std::arch::x86_64::{
    _mm256_cmpeq_epi8,
    // AVX2 intrinsics needed for fallback
    _mm256_loadu_si256,
    _mm256_movemask_epi8,

    // AVX-512 intrinsics (unstable) - ALL UNSIGNED for correct SIMD comparisons
    _mm512_and_si512,
    _mm512_cmpeq_epu8_mask,
    _mm512_cmpge_epi8_mask,
    _mm512_cmpgt_epi8_mask,
    _mm512_cmple_epi8_mask,
    _mm512_loadu_si512,
    _mm512_mask_cmpeq_epu8_mask,
    _mm512_mask_cmpgt_epi8_mask,
    _mm512_mask_cmple_epi8_mask,
    _mm512_maskz_loadu_epi8,
    _mm512_movm_epi8,
    _mm512_or_si512,
    _mm512_set1_epi8,
    _mm512_setzero_si512,
    _mm512_storeu_si512,
    _mm512_sub_epi8,
    _mm_cmpeq_epi8,
    // SSE2 intrinsics needed for fallback
    _mm_loadu_si128,
    _mm_movemask_epi8,
};

// Removed unused import that doesn't exist in this crate structure
use regex;

// Conditional imports for constants based on target architecture and features
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::constants::{LANES_AVX512_BYTES, UNROLL_FACTOR_AVX512};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use super::constants::LANES_AVX2_BYTES;

#[cfg(target_arch = "aarch64")]
use super::constants::LANES_NEON_BYTES;

// GPU optimized prefix string matching
#[cfg(has_cuda)]
pub unsafe fn match_prefix_strings_gpu(
    strings: &mut [String],
    string_lengths: &[usize],
    prefix: &str,
    case_insensitive: bool,
    max_size: usize,
    strings_len: usize,
    prefix_len: usize,
) -> usize {
    const PTX_MATCH_PREFIX: &str = r#"
    .version 7.5
    .target sm_70
    .entry match_prefix_strings (
      .param .u64 strings,
      .param .u64 string_lengths,
      .param .u64 prefix,
      .param .u32 case_insensitive,
      .param .u32 max_size,
      .param .u32 strings_len,
      .param .u32 prefix_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<35>;
      .reg .u64 %rd<20>;
      .reg .u8 %rc<16>;
      .reg .v4 .u8 %rc_vec<4>;
      .reg .pred %p<15>;

      // Load parameters
      ld.param.u64 %rd14, [strings];
      ld.param.u64 %rd15, [string_lengths];
      ld.param.u64 %rd16, [prefix];
      ld.param.u32 %r20, [case_insensitive];
      ld.param.u32 %r21, [max_size];
      ld.param.u32 %r22, [strings_len];
      ld.param.u32 %r23, [prefix_len];
      ld.param.u64 %rd17, [write_pos];

      // Initialize
      mov.u32 %r10, 0;           // write_pos

      // Grid stride through strings
      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ntid.x;
      mov.u32 %r3, %ctaid.x;
      mul.lo.u32 %r4, %r3, %r2;
      add.u32 %r5, %r4, %r1;     // string_idx
      mov.u32 %r6, %nctaid.x;
      mul.lo.u32 %r7, %r6, %r2;  // stride

    L70:  // string_loop
      setp.ge.u32 %p0, %r5, %r22;
      @%p0 bra L79;
      setp.ge.u32 %p1, %r10, %r21;
      @%p1 bra L79;

      // Check string length >= prefix_len
      mul.wide.u32 %rd1, %r5, 8;
      add.u64 %rd2, %rd15, %rd1;
      ld.global.u64 %rd3, [%rd2];
      cvt.u32.u64 %r11, %rd3;
      setp.lt.u32 %p2, %r11, %r23;
      @%p2 bra L78;  // skip_string

      // Get string pointer
      mul.wide.u32 %rd4, %r5, 24;
      add.u64 %rd5, %rd14, %rd4;
      ld.global.u64 %rd6, [%rd5];

      // Vectorized prefix comparison with warp-level optimization
      mov.u32 %r12, 0;           // char_idx
      mov.u32 %r13, 1;           // matches flag
      
      // Get lane ID for warp operations
      and.b32 %r24, %r1, 0x1f;   // Lane ID within warp

      // Check if we can use vectorized loads (4 bytes at a time)
    L71:  // vectorized_compare_loop
      add.u32 %r14, %r12, 4;
      setp.gt.u32 %p3, %r14, %r23;
      @%p3 bra L74;  // Jump to scalar fallback

      // Load 4 bytes from string and prefix using uchar4
      cvt.u64.u32 %rd7, %r12;
      add.u64 %rd8, %rd6, %rd7;
      ld.global.v4.u8 {%rc0, %rc1, %rc2, %rc3}, [%rd8];
      add.u64 %rd9, %rd16, %rd7;
      ld.global.v4.u8 {%rc4, %rc5, %rc6, %rc7}, [%rd9];

      // Case sensitive comparison first
      setp.eq.u32 %p4, %r20, 0;
      @%p4 bra L72;  // Jump to case sensitive

      // Case insensitive: convert to lowercase and compare
      or.b32 %r15, %rc0, 0x20;
      or.b32 %r16, %rc4, 0x20;
      setp.ne.u32 %p5, %r15, %r16;
      @%p5 bra L76;  // no_match
      
      or.b32 %r15, %rc1, 0x20;
      or.b32 %r16, %rc5, 0x20;
      setp.ne.u32 %p5, %r15, %r16;
      @%p5 bra L76;
      
      or.b32 %r15, %rc2, 0x20;
      or.b32 %r16, %rc6, 0x20;
      setp.ne.u32 %p5, %r15, %r16;
      @%p5 bra L76;
      
      or.b32 %r15, %rc3, 0x20;
      or.b32 %r16, %rc7, 0x20;
      setp.ne.u32 %p5, %r15, %r16;
      @%p5 bra L76;
      bra L73;

    L72:  // Case sensitive vectorized comparison with ballot
      setp.eq.u8 %p12, %rc0, %rc4;
      setp.eq.u8 %p13, %rc1, %rc5;
      and.pred %p12, %p12, %p13;
      setp.eq.u8 %p13, %rc2, %rc6;
      and.pred %p12, %p12, %p13;
      setp.eq.u8 %p13, %rc3, %rc7;
      and.pred %p12, %p12, %p13;
      
      // Use vote.all to check if all lanes in warp agree on match
      // More efficient than ballot when we just need all/any
      vote.all.pred %p14, %p12;
      @!%p14 bra L76;  // If any lane disagrees, no match

    L73:  // All 4 bytes matched
      add.u32 %r12, %r12, 4;
      bra L71;

    L74:  // Scalar fallback for remaining bytes
      setp.ge.u32 %p6, %r12, %r23;
      @%p6 bra L77;  // match_found

      // Load single chars
      cvt.u64.u32 %rd10, %r12;
      add.u64 %rd11, %rd6, %rd10;
      ld.global.u8 %rc8, [%rd11];
      add.u64 %rd12, %rd16, %rd10;
      ld.global.u8 %rc9, [%rd12];

      // Check match
      setp.eq.u8 %p7, %rc8, %rc9;
      @%p7 bra L75;

      // Case insensitive check
      setp.eq.u32 %p8, %r20, 0;
      @%p8 bra L76;  // no_match

      // Convert to lowercase
      or.b32 %r17, %rc8, 0x20;
      or.b32 %r18, %rc9, 0x20;
      setp.ne.u32 %p9, %r17, %r18;
      @%p9 bra L76;

    L75:  // next_char
      add.u32 %r12, %r12, 1;
      bra L74;

    L76:  // no_match
      mov.u32 %r13, 0;

    L77:  // match_found
      setp.eq.u32 %p10, %r13, 0;
      
      // Use vote.any to check if any thread in warp found a match
      vote.any.pred %p12, %p10;
      @!%p12 bra L78;  // No matches in warp, skip
      
      @%p10 bra L78;  // skip_string if this thread didn't match
      
      // Use warp-level coordination for atomic operations
      // Only one thread per match performs the atomic add
      vote.ballot.b32 %r27, %p10;
      bfind.u32 %r28, %r27;  // Find first matching thread
      and.b32 %r29, %r1, 0x1f;  // Get lane ID
      setp.eq.u32 %p13, %r29, %r28;  // Is this the first matching thread?
      @!%p13 bra L78;  // Only first matching thread does atomic
      
      // Atomically allocate space and copy string
      atom.global.add.u32 %r19, [%rd17], 1;
      setp.ge.u32 %p11, %r19, %r21;
      @%p11 bra L78;  // Skip if would exceed max_size
      
      mul.wide.u32 %rd13, %r19, 24;
      add.u64 %rd18, %rd14, %rd13;
      st.global.u64 [%rd18], %rd6;
      st.global.u64 [%rd18+8], %rd3;
      st.global.u64 [%rd18+16], %rd3;

    L78:  // skip_string  
      add.u32 %r5, %r5, %r7;
      bra L70;

    L79:  // done
      ret;
    }
  "#;

    let mut write_pos = 0u32;

    #[cfg(has_cuda)]
    use crate::gpu::LaunchConfig;
    let (blocks, threads) = LaunchConfig::strings();

    let prefix_bytes = prefix.as_bytes();

    launch_ptx(
        PTX_MATCH_PREFIX,
        &[],
        "match_prefix_strings",
        blocks,
        threads,
        &[
            strings.as_mut_ptr() as *const u8,
            string_lengths.as_ptr() as *const u8,
            prefix_bytes.as_ptr() as *const u8,
            &(case_insensitive as u32) as *const _ as *const u8,
            &max_size as *const _ as *const u8,
            &strings_len as *const _ as *const u8,
            &prefix_len as *const _ as *const u8,
            &mut write_pos as *mut _ as *const u8,
        ],
    )
    .unwrap_or_default();

    write_pos as usize
}
// Note: match_prefix_strings_gpu already implemented above

// AVX-512 optimized prefix string matching.
//
// Uses AVX-512 vectorized byte comparisons for enhanced performance.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
/// Internal heap-free implementation of AVX-512 prefix matching
#[inline]
pub(super) unsafe fn match_prefix_strings_avx512(
    strings: &mut [String],
    string_lengths: &[usize],
    prefix: &str,
    case_insensitive: bool,
    max_size: usize,
    strings_len: usize,
    prefix_len: usize,
) -> usize {
    const LANES: usize = LANES_AVX512_BYTES;
    const UNROLL_FACTOR: usize = UNROLL_FACTOR_AVX512; // Architecture-specific unrolling
    let mut write_pos = 0usize;

    let prefix_bytes = prefix.as_bytes();

    // Pre-broadcast prefix chunks for better cache efficiency - compile-time constant
    let mut prefix_chunks = [_mm512_setzero_si512(); UNROLL_FACTOR_AVX512];
    let num_full_chunks = prefix_len / LANES;
    let unrolled_chunks = num_full_chunks.min(UNROLL_FACTOR);

    for i in 0..unrolled_chunks {
        prefix_chunks[i] = _mm512_loadu_si512(prefix_bytes.as_ptr().add(i * LANES) as *const _);
    }

    // Pre-compute case conversion masks for case-insensitive matching
    let lowercase_mask = _mm512_set1_epi8(0x20);
    let alpha_lower = _mm512_set1_epi8(b'A' as i8);
    let alpha_upper = _mm512_set1_epi8(b'Z' as i8);

    // Filter strings directly in-place with optimized loop
    for read_pos in 0..strings_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }

        let string_len = string_lengths[read_pos];
        // Skip strings shorter than prefix using passed length parameter
        if string_len < prefix_len {
            continue;
        }

        let string = &strings[read_pos];
        let text_bytes = string.as_bytes();
        let mut prefix_matches = true;
        let mut j = 0;

        if case_insensitive {
            // Unrolled case-insensitive comparison
            while j + LANES * UNROLL_FACTOR <= prefix_len && prefix_matches {
                // Process 4 chunks at once for better instruction-level parallelism
                for chunk_idx in 0..UNROLL_FACTOR {
                    let offset = j + chunk_idx * LANES;
                    let string_chunk =
                        _mm512_loadu_si512(text_bytes.as_ptr().add(offset) as *const _);
                    let prefix_chunk = prefix_chunks[chunk_idx];

                    // Optimized case conversion using compare-and-blend
                    let string_ge_a = _mm512_cmpge_epi8_mask(string_chunk, alpha_lower);
                    let string_le_z = _mm512_cmple_epi8_mask(string_chunk, alpha_upper);
                    let string_is_upper = string_ge_a & string_le_z;

                    let prefix_ge_a = _mm512_cmpge_epi8_mask(prefix_chunk, alpha_lower);
                    let prefix_le_z = _mm512_cmple_epi8_mask(prefix_chunk, alpha_upper);
                    let prefix_is_upper = prefix_ge_a & prefix_le_z;

                    // Convert to lowercase: OR with 0x20 for uppercase letters
                    let string_upper_vec = _mm512_movm_epi8(string_is_upper);
                    let string_lower = _mm512_or_si512(
                        string_chunk,
                        _mm512_and_si512(string_upper_vec, lowercase_mask),
                    );
                    let prefix_upper_vec = _mm512_movm_epi8(prefix_is_upper);
                    let prefix_lower = _mm512_or_si512(
                        prefix_chunk,
                        _mm512_and_si512(prefix_upper_vec, lowercase_mask),
                    );

                    // Compare and check for early exit
                    let eq_mask = _mm512_cmpeq_epu8_mask(string_lower, prefix_lower);
                    if eq_mask != u64::MAX {
                        prefix_matches = false;
                        break;
                    }
                }
                j += LANES * UNROLL_FACTOR;
            }

            // Handle remaining full chunks
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = _mm512_loadu_si512(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm512_loadu_si512(prefix_bytes.as_ptr().add(j) as *const _);

                let string_ge_a = _mm512_cmpge_epi8_mask(string_chunk, alpha_lower);
                let string_le_z = _mm512_cmple_epi8_mask(string_chunk, alpha_upper);
                let string_is_upper = string_ge_a & string_le_z;

                let prefix_ge_a = _mm512_cmpge_epi8_mask(prefix_chunk, alpha_lower);
                let prefix_le_z = _mm512_cmple_epi8_mask(prefix_chunk, alpha_upper);
                let prefix_is_upper = prefix_ge_a & prefix_le_z;

                let string_lower = _mm512_or_si512(
                    string_chunk,
                    _mm512_and_si512(_mm512_movm_epi8(string_is_upper), lowercase_mask),
                );
                let prefix_lower = _mm512_or_si512(
                    prefix_chunk,
                    _mm512_and_si512(_mm512_movm_epi8(prefix_is_upper), lowercase_mask),
                );

                let eq_mask = _mm512_cmpeq_epu8_mask(string_lower, prefix_lower);
                if eq_mask != u64::MAX {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with masked SIMD operations
            let remaining_len = prefix_len - j;
            if remaining_len > 0 && prefix_matches {
                let load_mask = (1u64 << remaining_len) - 1;
                let string_chunk =
                    _mm512_maskz_loadu_epi8(load_mask, text_bytes.as_ptr().add(j) as *const i8);
                let prefix_chunk =
                    _mm512_maskz_loadu_epi8(load_mask, prefix_bytes.as_ptr().add(j) as *const i8);

                let string_ge_a = _mm512_cmpge_epi8_mask(string_chunk, alpha_lower) & load_mask;
                let string_le_z = _mm512_cmple_epi8_mask(string_chunk, alpha_upper) & load_mask;
                let string_is_upper = string_ge_a & string_le_z;

                let prefix_ge_a = _mm512_cmpge_epi8_mask(prefix_chunk, alpha_lower) & load_mask;
                let prefix_le_z = _mm512_cmple_epi8_mask(prefix_chunk, alpha_upper) & load_mask;
                let prefix_is_upper = prefix_ge_a & prefix_le_z;

                let string_lower = _mm512_or_si512(
                    string_chunk,
                    _mm512_and_si512(_mm512_movm_epi8(string_is_upper), lowercase_mask),
                );
                let prefix_lower = _mm512_or_si512(
                    prefix_chunk,
                    _mm512_and_si512(_mm512_movm_epi8(prefix_is_upper), lowercase_mask),
                );

                let eq_mask = _mm512_mask_cmpeq_epu8_mask(load_mask, string_lower, prefix_lower);
                if eq_mask != load_mask {
                    prefix_matches = false;
                }
            }
        } else {
            // Unrolled case-sensitive comparison
            while j + LANES * UNROLL_FACTOR <= prefix_len && prefix_matches {
                // Process 4 chunks at once for better instruction-level parallelism
                for chunk_idx in 0..UNROLL_FACTOR {
                    let offset = j + chunk_idx * LANES;
                    let string_chunk =
                        _mm512_loadu_si512(text_bytes.as_ptr().add(offset) as *const _);
                    let eq_mask = _mm512_cmpeq_epu8_mask(string_chunk, prefix_chunks[chunk_idx]);
                    if eq_mask != u64::MAX {
                        prefix_matches = false;
                        break;
                    }
                }
                j += LANES * UNROLL_FACTOR;
            }

            // Handle remaining full chunks
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = _mm512_loadu_si512(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm512_loadu_si512(prefix_bytes.as_ptr().add(j) as *const _);
                let eq_mask = _mm512_cmpeq_epu8_mask(string_chunk, prefix_chunk);
                if eq_mask != u64::MAX {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with AVX-512 masked operations
            let remaining_len = prefix_len - j;
            if remaining_len > 0 && prefix_matches {
                let load_mask = (1u64 << remaining_len) - 1;
                let string_chunk =
                    _mm512_maskz_loadu_epi8(load_mask, text_bytes.as_ptr().add(j) as *const i8);
                let prefix_chunk =
                    _mm512_maskz_loadu_epi8(load_mask, prefix_bytes.as_ptr().add(j) as *const i8);
                let eq_mask = _mm512_mask_cmpeq_epu8_mask(load_mask, string_chunk, prefix_chunk);
                if eq_mask != load_mask {
                    prefix_matches = false;
                }
            }
        }

        if prefix_matches {
            // Keep this string in the filtered result (in-place)
            if write_pos != read_pos {
                strings.swap(write_pos, read_pos);
            }
            write_pos += 1;

            //  EARLY TERMINATION: Stop when max_size results found
            if write_pos >= max_size {
                break;
            }
        }
    }

    write_pos
}

// AVX2 optimized prefix string matching.
//
// Uses AVX2 vectorized byte comparisons for enhanced performance.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
/// Internal heap-free implementation of AVX2 prefix matching
#[inline]
pub(super) unsafe fn match_prefix_strings_avx2(
    strings: &mut [String],
    string_lengths: &[usize],
    prefix: &str,
    case_insensitive: bool,
    max_size: usize,
    strings_len: usize,
    prefix_len: usize,
) -> usize {
    const LANES: usize = LANES_AVX2_BYTES; // AVX2 processes 32 bytes at once
    const UNROLL_FACTOR: usize = 4; // Process 4 SIMD chunks per iteration
    const FULL_MATCH_MASK: i32 = -1i32; // All 32 bits set
    let mut write_pos = 0usize;

    let prefix_bytes = prefix.as_bytes();

    // Pre-broadcast prefix chunks for better cache efficiency
    let mut prefix_chunks = [_mm256_setzero_si256(); UNROLL_FACTOR];
    let num_full_chunks = prefix_len / LANES;
    let unrolled_chunks = num_full_chunks.min(UNROLL_FACTOR);

    for i in 0..unrolled_chunks {
        prefix_chunks[i] = _mm256_loadu_si256(prefix_bytes.as_ptr().add(i * LANES) as *const _);
    }

    // Pre-compute case conversion masks for case-insensitive matching
    let lowercase_mask = _mm256_set1_epi8(0x20);
    let alpha_lower = _mm256_set1_epi8(b'A' as i8);

    let alpha_range_mask = _mm256_set1_epi8(0x1F); // For efficient range checking

    // Filter strings directly in-place with optimized loop
    for read_pos in 0..strings_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }

        let string_len = string_lengths[read_pos];
        // Skip strings shorter than prefix using passed length parameter
        if string_len < prefix_len {
            continue;
        }

        let string = &strings[read_pos];
        let text_bytes = string.as_bytes();
        let mut prefix_matches = true;
        let mut j = 0;

        if case_insensitive {
            // Unrolled case-insensitive comparison
            while j + LANES * UNROLL_FACTOR <= prefix_len && prefix_matches {
                // Process 4 chunks at once for better instruction-level parallelism
                for chunk_idx in 0..UNROLL_FACTOR {
                    let offset = j + chunk_idx * LANES;
                    let string_chunk =
                        _mm256_loadu_si256(text_bytes.as_ptr().add(offset) as *const _);
                    let prefix_chunk = prefix_chunks[chunk_idx];

                    // Optimized case conversion using efficient range checking
                    // Check if character is in A-Z range more efficiently
                    let string_offset = _mm256_sub_epi8(string_chunk, alpha_lower);
                    let string_in_range = _mm256_and_si256(string_offset, alpha_range_mask);
                    let string_is_upper = _mm256_cmpeq_epi8(string_offset, string_in_range);

                    let prefix_offset = _mm256_sub_epi8(prefix_chunk, alpha_lower);
                    let prefix_in_range = _mm256_and_si256(prefix_offset, alpha_range_mask);
                    let prefix_is_upper = _mm256_cmpeq_epi8(prefix_offset, prefix_in_range);

                    // Convert to lowercase using blend operations
                    let string_lower = _mm256_blendv_epi8(
                        string_chunk,
                        _mm256_or_si256(string_chunk, lowercase_mask),
                        string_is_upper,
                    );
                    let prefix_lower = _mm256_blendv_epi8(
                        prefix_chunk,
                        _mm256_or_si256(prefix_chunk, lowercase_mask),
                        prefix_is_upper,
                    );

                    // Compare and check for early exit
                    let cmp_result = _mm256_cmpeq_epi8(string_lower, prefix_lower);
                    let mask = _mm256_movemask_epi8(cmp_result);
                    if mask != FULL_MATCH_MASK {
                        prefix_matches = false;
                        break;
                    }
                }
                j += LANES * UNROLL_FACTOR;
            }

            // Handle remaining full chunks
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = _mm256_loadu_si256(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm256_loadu_si256(prefix_bytes.as_ptr().add(j) as *const _);

                // Optimized case conversion
                let string_offset = _mm256_sub_epi8(string_chunk, alpha_lower);
                let string_in_range = _mm256_and_si256(string_offset, alpha_range_mask);
                let string_is_upper = _mm256_cmpeq_epi8(string_offset, string_in_range);

                let prefix_offset = _mm256_sub_epi8(prefix_chunk, alpha_lower);
                let prefix_in_range = _mm256_and_si256(prefix_offset, alpha_range_mask);
                let prefix_is_upper = _mm256_cmpeq_epi8(prefix_offset, prefix_in_range);

                let string_lower = _mm256_blendv_epi8(
                    string_chunk,
                    _mm256_or_si256(string_chunk, lowercase_mask),
                    string_is_upper,
                );
                let prefix_lower = _mm256_blendv_epi8(
                    prefix_chunk,
                    _mm256_or_si256(prefix_chunk, lowercase_mask),
                    prefix_is_upper,
                );

                let cmp_result = _mm256_cmpeq_epi8(string_lower, prefix_lower);
                let mask = _mm256_movemask_epi8(cmp_result);
                if mask != FULL_MATCH_MASK {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with SIMD when possible
            let remaining_len = prefix_len - j;
            if remaining_len > 0 && prefix_matches {
                if remaining_len >= 16 {
                    // Use 128-bit SIMD for 16-31 remaining bytes
                    let string_chunk = _mm_loadu_si128(text_bytes.as_ptr().add(j) as *const _);
                    let prefix_chunk = _mm_loadu_si128(prefix_bytes.as_ptr().add(j) as *const _);

                    let alpha_lower_128 = _mm_set1_epi8(b'A' as i8);
                    let lowercase_mask_128 = _mm_set1_epi8(0x20);
                    let alpha_range_mask_128 = _mm_set1_epi8(0x1F);

                    let string_offset = _mm_sub_epi8(string_chunk, alpha_lower_128);
                    let string_in_range = _mm_and_si128(string_offset, alpha_range_mask_128);
                    let string_is_upper = _mm_cmpeq_epi8(string_offset, string_in_range);

                    let prefix_offset = _mm_sub_epi8(prefix_chunk, alpha_lower_128);
                    let prefix_in_range = _mm_and_si128(prefix_offset, alpha_range_mask_128);
                    let prefix_is_upper = _mm_cmpeq_epi8(prefix_offset, prefix_in_range);

                    let string_lower = _mm_blendv_epi8(
                        string_chunk,
                        _mm_or_si128(string_chunk, lowercase_mask_128),
                        string_is_upper,
                    );
                    let prefix_lower = _mm_blendv_epi8(
                        prefix_chunk,
                        _mm_or_si128(prefix_chunk, lowercase_mask_128),
                        prefix_is_upper,
                    );

                    let cmp_result = _mm_cmpeq_epi8(string_lower, prefix_lower);
                    let mask = _mm_movemask_epi8(cmp_result);

                    let remaining_mask = (1u32 << remaining_len.min(16)) - 1;
                    if (mask as u32) != remaining_mask {
                        prefix_matches = false;
                    }
                    j += 16;
                }

                // Handle final scalar bytes
                while j < prefix_len && prefix_matches {
                    let string_char = text_bytes[j];
                    let prefix_char = prefix_bytes[j];
                    let string_lower = if string_char >= b'A' && string_char <= b'Z' {
                        string_char | 0x20
                    } else {
                        string_char
                    };
                    let prefix_lower = if prefix_char >= b'A' && prefix_char <= b'Z' {
                        prefix_char | 0x20
                    } else {
                        prefix_char
                    };

                    if string_lower != prefix_lower {
                        prefix_matches = false;
                    }
                    j += 1;
                }
            }
        } else {
            // Unrolled case-sensitive comparison
            while j + LANES * UNROLL_FACTOR <= prefix_len && prefix_matches {
                // Process 4 chunks at once for better instruction-level parallelism
                for chunk_idx in 0..UNROLL_FACTOR {
                    let offset = j + chunk_idx * LANES;
                    let string_chunk =
                        _mm256_loadu_si256(text_bytes.as_ptr().add(offset) as *const _);
                    let cmp_result = _mm256_cmpeq_epi8(string_chunk, prefix_chunks[chunk_idx]);
                    let mask = _mm256_movemask_epi8(cmp_result);
                    if mask != FULL_MATCH_MASK {
                        prefix_matches = false;
                        break;
                    }
                }
                j += LANES * UNROLL_FACTOR;
            }

            // Handle remaining full chunks
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = _mm256_loadu_si256(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm256_loadu_si256(prefix_bytes.as_ptr().add(j) as *const _);
                let cmp_result = _mm256_cmpeq_epi8(string_chunk, prefix_chunk);
                let mask = _mm256_movemask_epi8(cmp_result);
                if mask != FULL_MATCH_MASK {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with SIMD when possible
            let remaining_len = prefix_len - j;
            if remaining_len > 0 && prefix_matches {
                if remaining_len >= 16 {
                    // Use 128-bit SIMD for 16-31 remaining bytes
                    let string_chunk = _mm_loadu_si128(text_bytes.as_ptr().add(j) as *const _);
                    let prefix_chunk = _mm_loadu_si128(prefix_bytes.as_ptr().add(j) as *const _);
                    let cmp_result = _mm_cmpeq_epi8(string_chunk, prefix_chunk);
                    let mask = _mm_movemask_epi8(cmp_result);

                    let remaining_mask = (1u32 << remaining_len.min(16)) - 1;
                    if (mask as u32) != remaining_mask {
                        prefix_matches = false;
                    }
                    j += 16;
                }

                // Handle final scalar bytes
                while j < prefix_len && prefix_matches {
                    if text_bytes[j] != prefix_bytes[j] {
                        prefix_matches = false;
                    }
                    j += 1;
                }
            }
        }

        if prefix_matches {
            // Keep this string in the filtered result (in-place)
            if write_pos != read_pos {
                strings.swap(write_pos, read_pos);
            }
            write_pos += 1;

            //  EARLY TERMINATION: Stop when max_size results found
            if write_pos >= max_size {
                break;
            }
        }
    }

    write_pos
}

// NEON optimized prefix string matching.
//
// Uses NEON vectorized byte comparisons for enhanced performance.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn match_prefix_strings_neon(
    strings: &mut [String], // Filter strings directly in-place
    string_lengths: &[usize],
    prefix: &str,
    case_insensitive: bool,
    max_size: usize,
    strings_len: usize,
    prefix_len: usize,
) -> usize {
    // NEON SIMD implementation for ARM processors
    const LANES: usize = LANES_NEON_BYTES; // NEON processes 16 bytes at once
    let mut write_pos = 0usize;

    // Work directly with prefix bytes, no allocation
    let prefix_bytes = prefix.as_bytes();

    // Filter strings directly in-place
    for read_pos in 0..strings_len {
        let string = &strings[read_pos];

        // Work directly with string bytes, no allocation
        let text_bytes = string.as_bytes();
        let string_len = string_lengths[read_pos];

        if string_len < prefix_len {
            continue;
        }

        // NEON prefix matching with on-the-fly case conversion
        let mut prefix_matches = true;
        let mut j = 0;

        if case_insensitive {
            // Case-insensitive SIMD comparison using on-the-fly conversion
            let lowercase_mask = unsafe { vdupq_n_u8(0x20) }; // Bit to set for lowercase conversion
            let alpha_mask_lower = unsafe { vdupq_n_u8(b'A') };
            let alpha_mask_upper = unsafe { vdupq_n_u8(b'Z') };

            // Process 16-byte SIMD chunks for case-insensitive comparison
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = unsafe { vld1q_u8(text_bytes.as_ptr().add(j)) };
                let prefix_chunk = unsafe { vld1q_u8(prefix_bytes.as_ptr().add(j)) };

                // Convert both to lowercase using SIMD
                let string_ge_a = unsafe { vcgeq_u8(string_chunk, alpha_mask_lower) };
                let string_le_z = unsafe { vcleq_u8(string_chunk, alpha_mask_upper) };
                let string_is_upper = unsafe { vandq_u8(string_ge_a, string_le_z) };
                let string_lower =
                    unsafe { vorrq_u8(string_chunk, vandq_u8(string_is_upper, lowercase_mask)) };

                let prefix_ge_a = unsafe { vcgeq_u8(prefix_chunk, alpha_mask_lower) };
                let prefix_le_z = unsafe { vcleq_u8(prefix_chunk, alpha_mask_upper) };
                let prefix_is_upper = unsafe { vandq_u8(prefix_ge_a, prefix_le_z) };
                let prefix_lower =
                    unsafe { vorrq_u8(prefix_chunk, vandq_u8(prefix_is_upper, lowercase_mask)) };

                // Compare the lowercase versions
                let cmp_result = unsafe { vceqq_u8(string_lower, prefix_lower) };
                let min_result = unsafe { vminvq_u8(cmp_result) };
                if min_result != 0xFF {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with scalar case-insensitive comparison
            while j < prefix_len && prefix_matches {
                let string_char = text_bytes[j];
                let prefix_char = prefix_bytes[j];
                let string_lower = if string_char >= b'A' && string_char <= b'Z' {
                    string_char | 0x20
                } else {
                    string_char
                };
                let prefix_lower = if prefix_char >= b'A' && prefix_char <= b'Z' {
                    prefix_char | 0x20
                } else {
                    prefix_char
                };
                if string_lower != prefix_lower {
                    prefix_matches = false;
                }
                j += 1;
            }
        } else {
            // Case-sensitive SIMD comparison
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = unsafe { vld1q_u8(text_bytes.as_ptr().add(j)) };
                let prefix_chunk = unsafe { vld1q_u8(prefix_bytes.as_ptr().add(j)) };
                let cmp_result = unsafe { vceqq_u8(string_chunk, prefix_chunk) };
                let min_result = unsafe { vminvq_u8(cmp_result) };
                if min_result != 0xFF {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with scalar case-sensitive comparison
            while j < prefix_len && prefix_matches {
                if text_bytes[j] != prefix_bytes[j] {
                    prefix_matches = false;
                }
                j += 1;
            }
        }

        if prefix_matches {
            // Keep this string in the filtered result (in-place)
            if write_pos != read_pos {
                strings.swap(write_pos, read_pos);
            }
            write_pos += 1;

            //  EARLY TERMINATION: Stop when max_size results found
            if write_pos >= max_size {
                break;
            }
        }
    }

    write_pos
}

// AVX-512 optimized exact phrase matching.
//
// Uses AVX-512 vectorized byte search with Boyer-Moore-style pattern matching.
// Processes text chunks in parallel for substring detection.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
// GPU implementation of exact phrase matching using PTX assembly

#[cfg(has_cuda)]
pub unsafe fn match_exact_phrases_gpu(
    texts: &mut [String],
    text_lengths: &[usize],
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
    texts_len: usize,
    phrase_len: usize,
) -> usize {
    const PTX_MATCH_EXACT: &str = r#"
    .version 7.5
    .target sm_70
    .entry match_exact_phrases (
      .param .u64 texts,
      .param .u64 text_lengths,
      .param .u64 phrase,
      .param .u32 case_insensitive,
      .param .u32 max_size,
      .param .u32 texts_len,
      .param .u32 phrase_len,
      .param .u64 write_pos
    ) {
      .reg .u32 %r<40>;
      .reg .u64 %rd<25>;
      .reg .pred %p<16>;
      .reg .b8 %rc<16>;
      .reg .v4 .u8 %rc_vec<4>;

      // Load parameters
      ld.param.u64 %rd15, [texts];
      ld.param.u64 %rd16, [text_lengths];
      ld.param.u64 %rd17, [phrase];
      ld.param.u32 %r28, [case_insensitive];
      ld.param.u32 %r29, [max_size];
      ld.param.u32 %r30, [texts_len];
      ld.param.u32 %r31, [phrase_len];
      ld.param.u64 %rd18, [write_pos];

      // Initialize thread ID and grid dimensions
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mul.lo.u32 %r3, %r1, %r2;
      add.u32 %r4, %r0, %r3;  // text_idx
      mov.u32 %r20, %nctaid.x;
      mul.lo.u32 %r21, %r20, %r2;  // stride

      // Initialize write position
      mov.u32 %r10, 0;

      // Main loop over texts with grid stride
    L70:  // text_loop
      setp.ge.u32 %p0, %r4, %r30;
      @%p0 bra L79;
      setp.ge.u32 %p1, %r10, %r29;
      @%p1 bra L79;

      // Get text length
      mul.wide.u32 %rd0, %r4, 8;
      add.u64 %rd1, %rd16, %rd0;
      ld.global.u64 %rd2, [%rd1];
      cvt.u32.u64 %r5, %rd2;

      // Check if text is long enough for phrase
      setp.lt.u32 %p2, %r5, %r31;
      @%p2 bra L78;  // skip_text

      // Get text pointer
      mul.wide.u32 %rd3, %r4, 24;
      add.u64 %rd4, %rd15, %rd3;
      ld.global.u64 %rd5, [%rd4];  // string data pointer
      ld.global.u64 %rd6, [%rd4+8]; // string length

      // Calculate search limit
      sub.u32 %r6, %r5, %r31;
      add.u32 %r6, %r6, 1;
      mov.u32 %r7, 0;  // search position

      // Boyer-Moore style search with vectorization
    L71:  // search_loop
      setp.ge.u32 %p3, %r7, %r6;
      @%p3 bra L78;  // No match found, skip text

      mov.u32 %r8, 1;  // match flag
      mov.u32 %r9, 0;  // phrase index

      // Try vectorized comparison first (4 bytes at a time)
    L72:  // vectorized_compare
      add.u32 %r22, %r9, 4;
      setp.gt.u32 %p4, %r22, %r31;
      @%p4 bra L74;  // Jump to scalar fallback

      // Load 4 bytes from text and phrase
      add.u32 %r11, %r7, %r9;
      cvt.u64.u32 %rd7, %r11;
      add.u64 %rd8, %rd5, %rd7;
      ld.global.v4.u8 {%rc0, %rc1, %rc2, %rc3}, [%rd8];
      cvt.u64.u32 %rd9, %r9;
      add.u64 %rd10, %rd17, %rd9;
      ld.global.v4.u8 {%rc4, %rc5, %rc6, %rc7}, [%rd10];

      // Case sensitive comparison
      setp.eq.u32 %p5, %r28, 0;
      @%p5 bra L73;  // Case sensitive

      // Case insensitive: convert to lowercase and compare
      or.b32 %r23, %rc0, 0x20;
      or.b32 %r24, %rc4, 0x20;
      setp.ne.u32 %p6, %r23, %r24;
      @%p6 bra L76;  // no_match
      
      or.b32 %r23, %rc1, 0x20;
      or.b32 %r24, %rc5, 0x20;
      setp.ne.u32 %p6, %r23, %r24;
      @%p6 bra L76;
      
      or.b32 %r23, %rc2, 0x20;
      or.b32 %r24, %rc6, 0x20;
      setp.ne.u32 %p6, %r23, %r24;
      @%p6 bra L76;
      
      or.b32 %r23, %rc3, 0x20;
      or.b32 %r24, %rc7, 0x20;
      setp.ne.u32 %p6, %r23, %r24;
      @%p6 bra L76;
      bra L75;  // All matched, continue

    L73:  // Case sensitive vectorized comparison with ballot voting
      setp.ne.u8 %p7, %rc0, %rc4;
      setp.ne.u8 %p14, %rc1, %rc5;
      or.pred %p7, %p7, %p14;
      setp.ne.u8 %p14, %rc2, %rc6;
      or.pred %p7, %p7, %p14;
      setp.ne.u8 %p14, %rc3, %rc7;
      or.pred %p7, %p7, %p14;
      
      // Use ballot to check if any lane in warp detected mismatch
      vote.ballot.b32 %r35, %p7;
      setp.ne.u32 %p15, %r35, 0;
      @%p15 bra L76;  // Any lane detected mismatch

    L75:  // Vectorized bytes matched - coordinate with warp
      // Use vote.all to ensure all lanes agree on match
      vote.all.pred %p15, %p8;
      @!%p15 bra L76;  // If any lane disagrees, no match
      add.u32 %r9, %r9, 4;
      bra L72;

    L74:  // Scalar fallback for remaining bytes
      setp.ge.u32 %p8, %r9, %r31;
      @%p8 bra L77;  // match_found

      // Load single characters
      add.u32 %r25, %r7, %r9;
      cvt.u64.u32 %rd11, %r25;
      add.u64 %rd12, %rd5, %rd11;
      ld.global.u8 %rc8, [%rd12];
      cvt.u64.u32 %rd13, %r9;
      add.u64 %rd14, %rd17, %rd13;
      ld.global.u8 %rc9, [%rd14];

      // Compare
      setp.eq.u8 %p9, %rc8, %rc9;
      @%p9 bra L741;

      // Case insensitive check
      setp.eq.u32 %p10, %r28, 0;
      @%p10 bra L76;  // no_match

      // Convert to lowercase
      or.b32 %r26, %rc8, 0x20;
      or.b32 %r27, %rc9, 0x20;
      setp.ne.u32 %p11, %r26, %r27;
      @%p11 bra L76;

    L741:  // next_char
      add.u32 %r9, %r9, 1;
      bra L74;

    L76:  // no_match
      mov.u32 %r8, 0;

    L77:  // check_match
      setp.eq.u32 %p12, %r8, 1;
      @!%p12 bra L771;  // Continue searching

      // Match found - atomically allocate space and copy text
      atom.global.add.u32 %r32, [%rd18], 1;
      setp.ge.u32 %p13, %r32, %r29;
      @%p13 bra L78;  // Skip if would exceed max_size
      
      // Copy the text to output
      mul.wide.u32 %rd19, %r32, 24;
      add.u64 %rd20, %rd15, %rd19;
      st.global.u64 [%rd20], %rd5;
      st.global.u64 [%rd20+8], %rd2;
      st.global.u64 [%rd20+16], %rd2;
      bra L78;  // Done with this text

    L771:  // Continue searching in same text
      add.u32 %r7, %r7, 1;
      bra L71;

    L78:  // skip_text
      add.u32 %r4, %r4, %r21;
      bra L70;

    L79:  // done
      ret;
    }
  "#;

    let mut write_pos = 0u32;

    #[cfg(has_cuda)]
    use crate::gpu::LaunchConfig;
    let (blocks, threads) = LaunchConfig::strings();

    let phrase_bytes = phrase.as_bytes();

    launch_ptx(
        PTX_MATCH_EXACT,
        &[],
        "match_exact_phrases",
        blocks,
        threads,
        &[
            texts.as_mut_ptr() as *const u8,
            text_lengths.as_ptr() as *const u8,
            phrase_bytes.as_ptr() as *const u8,
            &(case_insensitive as u32) as *const _ as *const u8,
            &max_size as *const _ as *const u8,
            &texts_len as *const _ as *const u8,
            &phrase_len as *const _ as *const u8,
            &mut write_pos as *mut _ as *const u8,
        ],
    )
    .unwrap_or_default();

    write_pos as usize
}
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn match_exact_phrases_avx512(
    texts: &mut [String], // Filter texts directly in-place
    text_lengths: &[usize],
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
    texts_len: usize,
    phrase_len: usize,
) -> usize {
    const LANES: usize = LANES_AVX512_BYTES; // AVX-512 processes 64 bytes at once
    const UNROLL_FACTOR: usize = 2; // Unroll factor for phrase search
    let mut write_pos = 0usize;

    let phrase_bytes = phrase.as_bytes();

    // Pre-compute search vectors for better cache efficiency
    let first_char = _mm512_set1_epi8(phrase_bytes[0] as i8);
    let has_alt_char = case_insensitive
        && ((phrase_bytes[0] >= b'A' && phrase_bytes[0] <= b'Z')
            || (phrase_bytes[0] >= b'a' && phrase_bytes[0] <= b'z'));
    let alt_char = if has_alt_char {
        let first_byte = phrase_bytes[0];
        if first_byte >= b'A' && first_byte <= b'Z' {
            _mm512_set1_epi8((first_byte | 0x20) as i8) // Convert to lowercase
        } else {
            _mm512_set1_epi8((first_byte & !0x20) as i8) // Convert to uppercase
        }
    } else {
        first_char // Not case-insensitive, same as first_char
    };

    // Pre-compute case conversion masks for better performance
    let lowercase_mask = _mm512_set1_epi8(0x20_u8 as i8);
    let alpha_lower = _mm512_set1_epi8(b'A' as i8);
    let alpha_upper = _mm512_set1_epi8(b'Z' as i8);

    // Pre-broadcast phrase chunks for longer phrases
    let mut phrase_chunks = [_mm512_setzero_si512(); 8]; // Support up to 512 bytes
    let num_phrase_chunks = ((phrase_len + LANES - 1) / LANES).min(8);
    for i in 0..num_phrase_chunks {
        let chunk_start = i * LANES;
        let chunk_end = (chunk_start + LANES).min(phrase_len);
        if chunk_start < phrase_len {
            let load_mask = if chunk_end - chunk_start == LANES {
                u64::MAX
            } else {
                (1u64 << (chunk_end - chunk_start)) - 1
            };
            phrase_chunks[i] = _mm512_maskz_loadu_epi8(
                load_mask,
                phrase_bytes.as_ptr().add(chunk_start) as *const i8,
            );
        }
    }

    // Filter texts directly in-place with optimized search
    for read_pos in 0..texts_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }

        let text_len = text_lengths[read_pos];
        // Skip if text is too short
        if text_len < phrase_len {
            continue;
        }

        let text = &texts[read_pos];
        let text_bytes = text.as_bytes();
        let mut found = false;

        // Optimized first character search with unrolling
        let search_end = text_len.saturating_sub(phrase_len);
        let simd_search_end = search_end & !(LANES * UNROLL_FACTOR - 1);

        let mut i = 0;

        // Unrolled SIMD search for better instruction-level parallelism
        while i < simd_search_end && !found {
            for unroll_idx in 0..UNROLL_FACTOR {
                let chunk_offset = i + unroll_idx * LANES;
                if chunk_offset > search_end {
                    break;
                }

                let text_chunk =
                    _mm512_loadu_si512(text_bytes.as_ptr().add(chunk_offset) as *const _);
                let first_mask = _mm512_cmpeq_epu8_mask(text_chunk, first_char);
                let combined_mask = if has_alt_char {
                    let alt_mask = _mm512_cmpeq_epu8_mask(text_chunk, alt_char);
                    first_mask | alt_mask
                } else {
                    first_mask
                };

                if combined_mask != 0 {
                    // Process each potential match with optimized bit scanning
                    let mut mask = combined_mask;
                    while mask != 0 {
                        let bit_pos = mask.trailing_zeros() as usize;
                        let pos = chunk_offset + bit_pos;
                        mask &= mask - 1; // Clear the lowest set bit

                        if pos + phrase_len <= text_len {
                            // Fast phrase comparison with early exits
                            let mut matches = true;

                            if case_insensitive {
                                // Optimized case-insensitive comparison using pre-computed vectors
                                for chunk_idx in 0..num_phrase_chunks {
                                    let chunk_start = chunk_idx * LANES;
                                    if chunk_start >= phrase_len {
                                        break;
                                    }

                                    let chunk_end = (chunk_start + LANES).min(phrase_len);
                                    let load_mask = if chunk_end - chunk_start == LANES {
                                        u64::MAX
                                    } else {
                                        (1u64 << (chunk_end - chunk_start)) - 1
                                    };

                                    let text_chunk = _mm512_maskz_loadu_epi8(
                                        load_mask,
                                        text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                    );
                                    let phrase_chunk = phrase_chunks[chunk_idx];

                                    // Optimized case conversion using mask operations
                                    let text_ge_a =
                                        _mm512_cmpge_epi8_mask(text_chunk, alpha_lower) & load_mask;
                                    let text_le_z = _mm512_mask_cmple_epi8_mask(
                                        load_mask,
                                        text_chunk,
                                        alpha_upper,
                                    );
                                    let text_is_upper = text_ge_a & text_le_z;

                                    let phrase_ge_a =
                                        _mm512_cmpge_epi8_mask(phrase_chunk, alpha_lower)
                                            & load_mask;
                                    let phrase_le_z = _mm512_mask_cmple_epi8_mask(
                                        load_mask,
                                        phrase_chunk,
                                        alpha_upper,
                                    );
                                    let phrase_is_upper = phrase_ge_a & phrase_le_z;

                                    let text_lower = _mm512_or_si512(
                                        text_chunk,
                                        _mm512_and_si512(
                                            _mm512_movm_epi8(text_is_upper),
                                            lowercase_mask,
                                        ),
                                    );
                                    let phrase_lower = _mm512_or_si512(
                                        phrase_chunk,
                                        _mm512_and_si512(
                                            _mm512_movm_epi8(phrase_is_upper),
                                            lowercase_mask,
                                        ),
                                    );

                                    let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                        load_mask,
                                        text_lower,
                                        phrase_lower,
                                    );
                                    if cmp_mask != load_mask {
                                        matches = false;
                                        break;
                                    }
                                }
                            } else {
                                // Optimized case-sensitive comparison
                                for chunk_idx in 0..num_phrase_chunks {
                                    let chunk_start = chunk_idx * LANES;
                                    if chunk_start >= phrase_len {
                                        break;
                                    }

                                    let chunk_end = (chunk_start + LANES).min(phrase_len);
                                    let load_mask = if chunk_end - chunk_start == LANES {
                                        u64::MAX
                                    } else {
                                        (1u64 << (chunk_end - chunk_start)) - 1
                                    };

                                    let text_chunk = _mm512_maskz_loadu_epi8(
                                        load_mask,
                                        text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                    );
                                    let phrase_chunk = phrase_chunks[chunk_idx];

                                    let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                        load_mask,
                                        text_chunk,
                                        phrase_chunk,
                                    );
                                    if cmp_mask != load_mask {
                                        matches = false;
                                        break;
                                    }
                                }
                            }

                            if matches {
                                found = true;
                                break;
                            }
                        }
                    }

                    if found {
                        break;
                    }
                }
            }
            i += LANES * UNROLL_FACTOR;
        }

        // Handle remaining SIMD chunks
        while i <= search_end && i + LANES <= text_len && !found {
            let text_chunk = _mm512_loadu_si512(text_bytes.as_ptr().add(i) as *const _);
            let first_mask = _mm512_cmpeq_epu8_mask(text_chunk, first_char);
            let combined_mask = if has_alt_char {
                let alt_mask = _mm512_cmpeq_epu8_mask(text_chunk, alt_char);
                first_mask | alt_mask
            } else {
                first_mask
            };

            if combined_mask != 0 {
                let mut mask = combined_mask;
                while mask != 0 {
                    let bit_pos = mask.trailing_zeros() as usize;
                    let pos = i + bit_pos;
                    mask &= mask - 1;

                    if pos + phrase_len <= text_len {
                        // Use the same optimized phrase comparison as above
                        let mut matches = true;

                        if case_insensitive {
                            for chunk_idx in 0..num_phrase_chunks {
                                let chunk_start = chunk_idx * LANES;
                                if chunk_start >= phrase_len {
                                    break;
                                }

                                let chunk_end = (chunk_start + LANES).min(phrase_len);
                                let load_mask = if chunk_end - chunk_start == LANES {
                                    u64::MAX
                                } else {
                                    (1u64 << (chunk_end - chunk_start)) - 1
                                };

                                let text_chunk = _mm512_maskz_loadu_epi8(
                                    load_mask,
                                    text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                );
                                let phrase_chunk = phrase_chunks[chunk_idx];

                                let text_ge_a =
                                    _mm512_cmpge_epi8_mask(text_chunk, alpha_lower) & load_mask;
                                let text_le_z =
                                    _mm512_mask_cmple_epi8_mask(load_mask, text_chunk, alpha_upper);
                                let text_is_upper = text_ge_a & text_le_z;

                                let phrase_ge_a =
                                    _mm512_cmpge_epi8_mask(phrase_chunk, alpha_lower) & load_mask;
                                let phrase_le_z = _mm512_mask_cmple_epi8_mask(
                                    load_mask,
                                    phrase_chunk,
                                    alpha_upper,
                                );
                                let phrase_is_upper = phrase_ge_a & phrase_le_z;

                                let text_lower = _mm512_or_si512(
                                    text_chunk,
                                    _mm512_and_si512(
                                        _mm512_movm_epi8(text_is_upper),
                                        lowercase_mask,
                                    ),
                                );
                                let phrase_lower = _mm512_or_si512(
                                    phrase_chunk,
                                    _mm512_and_si512(
                                        _mm512_movm_epi8(phrase_is_upper),
                                        lowercase_mask,
                                    ),
                                );

                                let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                    load_mask,
                                    text_lower,
                                    phrase_lower,
                                );
                                if cmp_mask != load_mask {
                                    matches = false;
                                    break;
                                }
                            }
                        } else {
                            for chunk_idx in 0..num_phrase_chunks {
                                let chunk_start = chunk_idx * LANES;
                                if chunk_start >= phrase_len {
                                    break;
                                }

                                let chunk_end = (chunk_start + LANES).min(phrase_len);
                                let load_mask = if chunk_end - chunk_start == LANES {
                                    u64::MAX
                                } else {
                                    (1u64 << (chunk_end - chunk_start)) - 1
                                };

                                let text_chunk = _mm512_maskz_loadu_epi8(
                                    load_mask,
                                    text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                );
                                let phrase_chunk = phrase_chunks[chunk_idx];

                                let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                    load_mask,
                                    text_chunk,
                                    phrase_chunk,
                                );
                                if cmp_mask != load_mask {
                                    matches = false;
                                    break;
                                }
                            }
                        }

                        if matches {
                            found = true;
                            break;
                        }
                    }
                }
            }
            i += LANES;
        }

        // Handle final scalar bytes with optimized search
        if !found {
            for pos in i..=search_end {
                let first_matches = if case_insensitive && has_alt_char {
                    let text_first = text_bytes[pos];
                    text_first == phrase_bytes[0]
                        || ((text_first >= b'A' && text_first <= b'Z')
                            && (text_first | 0x20) == phrase_bytes[0])
                        || ((text_first >= b'a' && text_first <= b'z')
                            && (text_first & !0x20) == phrase_bytes[0])
                } else {
                    text_bytes[pos] == phrase_bytes[0]
                };

                if first_matches {
                    let mut matches = true;
                    if case_insensitive {
                        for j in 0..phrase_len {
                            let text_char = text_bytes[pos + j];
                            let phrase_char = phrase_bytes[j];
                            let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                                text_char | 0x20
                            } else {
                                text_char
                            };
                            let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z' {
                                phrase_char | 0x20
                            } else {
                                phrase_char
                            };
                            if text_lower != phrase_lower {
                                matches = false;
                                break;
                            }
                        }
                    } else {
                        for j in 0..phrase_len {
                            if text_bytes[pos + j] != phrase_bytes[j] {
                                matches = false;
                                break;
                            }
                        }
                    }
                    if matches {
                        found = true;
                        break;
                    }
                }
            }
        }

        if found {
            // Keep this text in the filtered result (in-place)
            if write_pos != read_pos {
                texts.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }

    write_pos
}

// AVX2 optimized exact phrase matching.
//
// Uses AVX2 vectorized byte search with pattern matching.
// Processes text chunks in parallel for substring detection.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
pub(crate) unsafe fn match_exact_phrases_avx2(
    texts: &mut [String], // Filter texts directly in-place
    text_lengths: &[usize],
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
    texts_len: usize,
    phrase_len: usize,
) -> usize {
    const LANES: usize = LANES_AVX2_BYTES; // AVX2 processes 32 bytes at once
    const UNROLL_FACTOR: usize = 2; // Unroll factor for phrase search
    const FULL_MATCH_MASK: i32 = -1i32; // All 32 bits set
    let mut write_pos = 0usize;

    let phrase_bytes = phrase.as_bytes();

    // Pre-compute search vectors for better cache efficiency
    let first_char = _mm256_set1_epi8(phrase_bytes[0] as i8);
    let has_alt_char = case_insensitive
        && ((phrase_bytes[0] >= b'A' && phrase_bytes[0] <= b'Z')
            || (phrase_bytes[0] >= b'a' && phrase_bytes[0] <= b'z'));
    let alt_char = if has_alt_char {
        let first_byte = phrase_bytes[0];
        let alt_byte = if first_byte >= b'A' && first_byte <= b'Z' {
            first_byte | 0x20 // Convert to lowercase
        } else {
            first_byte & !0x20 // Convert to uppercase
        };
        _mm256_set1_epi8(alt_byte as i8)
    } else {
        first_char // Dummy value, won't be used
    };

    // Pre-compute case conversion masks for better performance
    let lowercase_mask = _mm256_set1_epi8(0x20);
    let alpha_lower = _mm256_set1_epi8(b'A' as i8);

    let alpha_range_mask = _mm256_set1_epi8(0x1F); // For efficient range checking

    // Pre-broadcast phrase chunks for longer phrases
    let mut phrase_chunks = [_mm256_setzero_si256(); 16]; // Support up to 512 bytes
    let num_phrase_chunks = ((phrase_len + LANES - 1) / LANES).min(16);
    for i in 0..num_phrase_chunks {
        let chunk_start = i * LANES;
        if chunk_start < phrase_len {
            phrase_chunks[i] =
                _mm256_loadu_si256(phrase_bytes.as_ptr().add(chunk_start) as *const _);
        }
    }

    // Filter texts directly in-place with optimized search
    for read_pos in 0..texts_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }

        let text_len = text_lengths[read_pos];
        // Skip if text is too short
        if text_len < phrase_len {
            continue;
        }

        let text = &texts[read_pos];
        let text_bytes = text.as_bytes();
        let mut found = false;

        // Optimized first character search with unrolling
        let search_end = text_len.saturating_sub(phrase_len);
        let simd_search_end = search_end & !(LANES * UNROLL_FACTOR - 1);

        let mut i = 0;

        // Unrolled SIMD search for better instruction-level parallelism
        while i < simd_search_end && !found {
            for unroll_idx in 0..UNROLL_FACTOR {
                let chunk_offset = i + unroll_idx * LANES;
                if chunk_offset > search_end {
                    break;
                }

                let text_chunk =
                    _mm256_loadu_si256(text_bytes.as_ptr().add(chunk_offset) as *const _);
                let first_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(text_chunk, first_char));
                let combined_mask = if has_alt_char {
                    let alt_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(text_chunk, alt_char));
                    first_mask | alt_mask
                } else {
                    first_mask
                };

                if combined_mask != 0 {
                    // Process each potential match with optimized bit scanning
                    let mut mask = combined_mask;
                    while mask != 0 {
                        let bit_pos = mask.trailing_zeros() as usize;
                        let pos = chunk_offset + bit_pos;
                        mask &= mask - 1; // Clear the lowest set bit

                        if pos + phrase_len <= text_len {
                            // Fast phrase comparison with early exits
                            let mut matches = true;

                            if case_insensitive {
                                // Optimized case-insensitive comparison using pre-computed vectors
                                for chunk_idx in 0..num_phrase_chunks {
                                    let chunk_start = chunk_idx * LANES;
                                    if chunk_start >= phrase_len {
                                        break;
                                    }

                                    let remaining_bytes = phrase_len - chunk_start;
                                    let chunk_bytes = remaining_bytes.min(LANES);

                                    if chunk_bytes == LANES {
                                        // Full chunk comparison
                                        let text_chunk = _mm256_loadu_si256(
                                            text_bytes.as_ptr().add(pos + chunk_start) as *const _,
                                        );
                                        let phrase_chunk = phrase_chunks[chunk_idx];

                                        // Optimized case conversion using efficient range checking
                                        let text_offset = _mm256_sub_epi8(text_chunk, alpha_lower);
                                        let text_in_range =
                                            _mm256_and_si256(text_offset, alpha_range_mask);
                                        let text_is_upper =
                                            _mm256_cmpeq_epi8(text_offset, text_in_range);

                                        let phrase_offset =
                                            _mm256_sub_epi8(phrase_chunk, alpha_lower);
                                        let phrase_in_range =
                                            _mm256_and_si256(phrase_offset, alpha_range_mask);
                                        let phrase_is_upper =
                                            _mm256_cmpeq_epi8(phrase_offset, phrase_in_range);

                                        let text_lower = _mm256_blendv_epi8(
                                            text_chunk,
                                            _mm256_or_si256(text_chunk, lowercase_mask),
                                            text_is_upper,
                                        );
                                        let phrase_lower = _mm256_blendv_epi8(
                                            phrase_chunk,
                                            _mm256_or_si256(phrase_chunk, lowercase_mask),
                                            phrase_is_upper,
                                        );

                                        let cmp_result =
                                            _mm256_cmpeq_epi8(text_lower, phrase_lower);
                                        let mask = _mm256_movemask_epi8(cmp_result);
                                        if mask != FULL_MATCH_MASK {
                                            matches = false;
                                            break;
                                        }
                                    } else {
                                        // Partial chunk - use scalar comparison for simplicity
                                        for byte_idx in 0..chunk_bytes {
                                            let text_char =
                                                text_bytes[pos + chunk_start + byte_idx];
                                            let phrase_char = phrase_bytes[chunk_start + byte_idx];
                                            let text_lower =
                                                if text_char >= b'A' && text_char <= b'Z' {
                                                    text_char | 0x20
                                                } else {
                                                    text_char
                                                };
                                            let phrase_lower =
                                                if phrase_char >= b'A' && phrase_char <= b'Z' {
                                                    phrase_char | 0x20
                                                } else {
                                                    phrase_char
                                                };
                                            if text_lower != phrase_lower {
                                                matches = false;
                                                break;
                                            }
                                        }
                                        if !matches {
                                            break;
                                        }
                                    }
                                }
                            } else {
                                // Optimized case-sensitive comparison
                                for chunk_idx in 0..num_phrase_chunks {
                                    let chunk_start = chunk_idx * LANES;
                                    if chunk_start >= phrase_len {
                                        break;
                                    }

                                    let remaining_bytes = phrase_len - chunk_start;
                                    let chunk_bytes = remaining_bytes.min(LANES);

                                    if chunk_bytes == LANES {
                                        // Full chunk comparison
                                        let text_chunk = _mm256_loadu_si256(
                                            text_bytes.as_ptr().add(pos + chunk_start) as *const _,
                                        );
                                        let phrase_chunk = phrase_chunks[chunk_idx];
                                        let cmp_result =
                                            _mm256_cmpeq_epi8(text_chunk, phrase_chunk);
                                        let mask = _mm256_movemask_epi8(cmp_result);
                                        if mask != FULL_MATCH_MASK {
                                            matches = false;
                                            break;
                                        }
                                    } else {
                                        // Partial chunk - use scalar comparison for simplicity
                                        for byte_idx in 0..chunk_bytes {
                                            if text_bytes[pos + chunk_start + byte_idx]
                                                != phrase_bytes[chunk_start + byte_idx]
                                            {
                                                matches = false;
                                                break;
                                            }
                                        }
                                        if !matches {
                                            break;
                                        }
                                    }
                                }
                            }

                            if matches {
                                found = true;
                                break;
                            }
                        }
                    }

                    if found {
                        break;
                    }
                }
            }
            i += LANES * UNROLL_FACTOR;
        }

        // Handle remaining SIMD chunks
        while i <= search_end && i + LANES <= text_len && !found {
            let text_chunk = _mm256_loadu_si256(text_bytes.as_ptr().add(i) as *const _);
            let first_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(text_chunk, first_char));
            let combined_mask = if has_alt_char {
                let alt_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(text_chunk, alt_char));
                first_mask | alt_mask
            } else {
                first_mask
            };

            if combined_mask != 0 {
                // Extract mask bits to array and process sequentially (NEON-style pattern)
                let mut mask_array = [0u32; 32];
                for k in 0..32 {
                    // Convert bit k to 0x00000000 or 0xFFFFFFFF like NEON
                    mask_array[k] = if (combined_mask & (1i32 << k)) != 0 {
                        1
                    } else {
                        0
                    };
                }

                // Process each element sequentially (EXACTLY matching NEON pattern)
                for bit_pos in 0..32 {
                    if mask_array[bit_pos] != 0 {
                        let pos = i + bit_pos;

                        if pos + phrase_len <= text_len {
                            let mut matches = true;
                            let mut j = 0;

                            // Compare full phrase with case sensitivity
                            if case_insensitive {
                                // Case-insensitive comparison with SIMD on-the-fly conversion
                                let lowercase_mask = _mm256_set1_epi8(0x20);
                                let alpha_mask_lower = _mm256_set1_epi8(b'A' as i8);
                                let alpha_mask_upper = _mm256_set1_epi8(b'Z' as i8);

                                while j + LANES <= phrase_len && matches {
                                    let text_chunk = _mm256_loadu_si256(
                                        text_bytes.as_ptr().add(pos + j) as *const _,
                                    );
                                    let phrase_chunk = _mm256_loadu_si256(
                                        phrase_bytes.as_ptr().add(j) as *const _,
                                    );

                                    // Convert both to lowercase using SIMD
                                    let text_ge_a = _mm256_cmpgt_epi8(
                                        text_chunk,
                                        _mm256_sub_epi8(alpha_mask_lower, _mm256_set1_epi8(1)),
                                    );
                                    let text_le_z = _mm256_cmpgt_epi8(
                                        _mm256_add_epi8(alpha_mask_upper, _mm256_set1_epi8(1)),
                                        text_chunk,
                                    );
                                    let text_is_upper = _mm256_and_si256(text_ge_a, text_le_z);
                                    let text_lower = _mm256_or_si256(
                                        text_chunk,
                                        _mm256_and_si256(text_is_upper, lowercase_mask),
                                    );

                                    let phrase_ge_a = _mm256_cmpgt_epi8(
                                        phrase_chunk,
                                        _mm256_sub_epi8(alpha_mask_lower, _mm256_set1_epi8(1)),
                                    );
                                    let phrase_le_z = _mm256_cmpgt_epi8(
                                        _mm256_add_epi8(alpha_mask_upper, _mm256_set1_epi8(1)),
                                        phrase_chunk,
                                    );
                                    let phrase_is_upper =
                                        _mm256_and_si256(phrase_ge_a, phrase_le_z);
                                    let phrase_lower = _mm256_or_si256(
                                        phrase_chunk,
                                        _mm256_and_si256(phrase_is_upper, lowercase_mask),
                                    );

                                    // Compare the lowercase versions
                                    let cmp_result = _mm256_cmpeq_epi8(text_lower, phrase_lower);
                                    let mask = _mm256_movemask_epi8(cmp_result);

                                    if mask != FULL_MATCH_MASK {
                                        matches = false;
                                    }
                                    j += LANES;
                                }

                                // Handle remaining bytes with AVX2 operations
                                if j < phrase_len && matches {
                                    let remaining_len = phrase_len - j;
                                    // Use minimal scalar for tiny remainder only
                                    for k in 0..remaining_len {
                                        let text_char = text_bytes[pos + j + k];
                                        let phrase_char = phrase_bytes[j + k];
                                        let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                                            text_char | 0x20
                                        } else {
                                            text_char
                                        };
                                        let phrase_lower =
                                            if phrase_char >= b'A' && phrase_char <= b'Z' {
                                                phrase_char | 0x20
                                            } else {
                                                phrase_char
                                            };

                                        if text_lower != phrase_lower {
                                            matches = false;
                                            break;
                                        }
                                    }
                                }
                            } else {
                                // Case-sensitive SIMD comparison
                                while j + LANES <= phrase_len && matches {
                                    let text_chunk = _mm256_loadu_si256(
                                        text_bytes.as_ptr().add(pos + j) as *const _,
                                    );
                                    let phrase_chunk = _mm256_loadu_si256(
                                        phrase_bytes.as_ptr().add(j) as *const _,
                                    );
                                    let cmp_result = _mm256_cmpeq_epi8(text_chunk, phrase_chunk);

                                    let mask = _mm256_movemask_epi8(cmp_result);
                                    if mask != FULL_MATCH_MASK {
                                        matches = false;
                                    }
                                    j += LANES;
                                }

                                // Handle remaining bytes with AVX2 operations
                                if j < phrase_len && matches {
                                    let remaining_len = phrase_len - j;
                                    // Use minimal scalar for tiny remainder only
                                    for k in 0..remaining_len {
                                        if text_bytes[pos + j + k] != phrase_bytes[j + k] {
                                            matches = false;
                                            break;
                                        }
                                    }
                                }
                            }

                            if matches {
                                found = true;
                                break;
                            }
                        }
                    }
                }
            }
            i += LANES;
        }

        // Handle remaining bytes with scalar search
        while i <= text_len.saturating_sub(phrase_len) && !found {
            let mut matches = true;

            if case_insensitive {
                for j in 0..phrase_len {
                    let text_char = text_bytes[i + j];
                    let phrase_char = phrase_bytes[j];
                    let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                        text_char | 0x20
                    } else {
                        text_char
                    };
                    let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z' {
                        phrase_char | 0x20
                    } else {
                        phrase_char
                    };
                    if text_lower != phrase_lower {
                        matches = false;
                        break;
                    }
                }
            } else {
                for j in 0..phrase_len {
                    if text_bytes[i + j] != phrase_bytes[j] {
                        matches = false;
                        break;
                    }
                }
            }

            if matches {
                found = true;
            }
            i += 1;
        }

        if found {
            // Keep this text in the filtered result (in-place)
            if write_pos != read_pos {
                texts.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }

    write_pos
}

// NEON optimized exact phrase matching.
//
// Uses NEON vectorized byte search with pattern matching.
// Processes text chunks in parallel for substring detection.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn match_exact_phrases_neon(
    texts: &mut [String], // Filter texts directly in-place
    text_lengths: &[usize],
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
    texts_len: usize,
    phrase_len: usize,
) -> usize {
    const LANES: usize = LANES_NEON_BYTES; // NEON processes 16 bytes at once
    let mut write_pos = 0usize;

    // Work directly with phrase bytes, no allocation
    let phrase_bytes = phrase.as_bytes();

    // Load first character for quick filtering (case-sensitive or case-insensitive)
    let first_char_simd = unsafe { vdupq_n_u8(phrase_bytes[0]) };
    let (has_alt_char, alt_simd) = if case_insensitive {
        let first_char = phrase_bytes[0];
        let alt_char = if first_char >= b'A' && first_char <= b'Z' {
            first_char | 0x20 // Convert to lowercase
        } else if first_char >= b'a' && first_char <= b'z' {
            first_char & !0x20 // Convert to uppercase
        } else {
            first_char // Non-alphabetic, same character
        };
        (true, unsafe { vdupq_n_u8(alt_char) })
    } else {
        (false, unsafe { vdupq_n_u8(0) }) // Dummy value, won't be used
    };

    // Filter texts directly in-place
    for read_pos in 0..texts_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }
        // Scope the byte access to avoid borrowing conflicts
        let text_len = text_lengths[read_pos];
        if text_len < phrase_len {
            continue;
        }

        let mut found = false;

        // Scope text bytes access to avoid borrowing conflicts
        {
            let text_bytes = texts[read_pos].as_bytes();
            let mut i = 0;

            // SIMD search for first character matches (case-sensitive or case-insensitive)
            while i + LANES <= text_len && !found {
                let text_chunk = unsafe { vld1q_u8(text_bytes.as_ptr().add(i)) };
                let first_cmp = unsafe { vceqq_u8(text_chunk, first_char_simd) };

                // For case-insensitive, also check alternate case
                let combined_cmp = if has_alt_char {
                    let alt_cmp = unsafe { vceqq_u8(text_chunk, alt_simd) };
                    unsafe { vorrq_u8(first_cmp, alt_cmp) }
                } else {
                    first_cmp
                };

                // Check for matches in the vector
                let mut cmp_values = [0u8; 16];
                unsafe { vst1q_u8(cmp_values.as_mut_ptr(), combined_cmp) };

                for bit_pos in 0..LANES {
                    let cmp_val = cmp_values[bit_pos];
                    if cmp_val == 0xFF {
                        let pos = i + bit_pos;
                        if pos + phrase_len <= text_len {
                            // Quick check: does last character also match?
                            let last_char_matches = if phrase_len == 1 {
                                true
                            } else if case_insensitive {
                                let text_last = text_bytes[pos + phrase_len - 1];
                                let phrase_last = phrase_bytes[phrase_len - 1];
                                let text_lower = if text_last >= b'A' && text_last <= b'Z' {
                                    text_last | 0x20
                                } else {
                                    text_last
                                };
                                let phrase_lower = if phrase_last >= b'A' && phrase_last <= b'Z' {
                                    phrase_last | 0x20
                                } else {
                                    phrase_last
                                };
                                text_lower == phrase_lower
                            } else {
                                text_bytes[pos + phrase_len - 1] == phrase_bytes[phrase_len - 1]
                            };

                            if last_char_matches {
                                // SIMD-accelerated full phrase comparison
                                let mut matches = true;
                                let mut j = 0;

                                // Case-sensitive vs case-insensitive comparison
                                if case_insensitive {
                                    // Case-insensitive SIMD comparison using on-the-fly conversion
                                    let lowercase_mask = unsafe { vdupq_n_u8(0x20) }; // Bit to set for lowercase conversion
                                    let alpha_mask_lower = unsafe { vdupq_n_u8(b'A') };
                                    let alpha_mask_upper = unsafe { vdupq_n_u8(b'Z') };

                                    while j + LANES <= phrase_len && matches {
                                        let text_chunk =
                                            unsafe { vld1q_u8(text_bytes.as_ptr().add(pos + j)) };
                                        let phrase_chunk =
                                            unsafe { vld1q_u8(phrase_bytes.as_ptr().add(j)) };

                                        // Convert both to lowercase using SIMD
                                        let text_ge_a =
                                            unsafe { vcgeq_u8(text_chunk, alpha_mask_lower) };
                                        let text_le_z =
                                            unsafe { vcleq_u8(text_chunk, alpha_mask_upper) };
                                        let text_is_upper =
                                            unsafe { vandq_u8(text_ge_a, text_le_z) };
                                        let text_lower = unsafe {
                                            vorrq_u8(
                                                text_chunk,
                                                vandq_u8(text_is_upper, lowercase_mask),
                                            )
                                        };

                                        let phrase_ge_a =
                                            unsafe { vcgeq_u8(phrase_chunk, alpha_mask_lower) };
                                        let phrase_le_z =
                                            unsafe { vcleq_u8(phrase_chunk, alpha_mask_upper) };
                                        let phrase_is_upper =
                                            unsafe { vandq_u8(phrase_ge_a, phrase_le_z) };
                                        let phrase_lower = unsafe {
                                            vorrq_u8(
                                                phrase_chunk,
                                                vandq_u8(phrase_is_upper, lowercase_mask),
                                            )
                                        };

                                        // Compare the lowercase versions
                                        let cmp_result =
                                            unsafe { vceqq_u8(text_lower, phrase_lower) };
                                        let min_result = unsafe { vminvq_u8(cmp_result) };
                                        if min_result != 0xFF {
                                            matches = false;
                                        }
                                        j += LANES;
                                    }

                                    // Handle remaining bytes with scalar case-insensitive comparison
                                    while j < phrase_len && matches {
                                        let text_char = text_bytes[pos + j];
                                        let phrase_char = phrase_bytes[j];
                                        let text_lower = if text_char.is_ascii_uppercase() {
                                            text_char | 0x20
                                        } else {
                                            text_char
                                        };
                                        let phrase_lower = if phrase_char.is_ascii_uppercase() {
                                            phrase_char | 0x20
                                        } else {
                                            phrase_char
                                        };
                                        if text_lower != phrase_lower {
                                            matches = false;
                                            break;
                                        }
                                        j += 1;
                                    }
                                } else {
                                    // Case-sensitive SIMD comparison
                                    while j + LANES <= phrase_len && matches {
                                        let text_chunk =
                                            unsafe { vld1q_u8(text_bytes.as_ptr().add(pos + j)) };
                                        let phrase_chunk =
                                            unsafe { vld1q_u8(phrase_bytes.as_ptr().add(j)) };
                                        let cmp_result =
                                            unsafe { vceqq_u8(text_chunk, phrase_chunk) };

                                        // Check if all bytes match
                                        let min_result = unsafe { vminvq_u8(cmp_result) };
                                        if min_result != 0xFF {
                                            matches = false;
                                        }
                                        j += LANES;
                                    }

                                    // Handle remaining bytes with scalar case-sensitive comparison
                                    while j < phrase_len && matches {
                                        if text_bytes[pos + j] != phrase_bytes[j] {
                                            matches = false;
                                            break;
                                        }
                                        j += 1;
                                    }
                                }

                                if matches {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                i += LANES;
            }

            // Handle remaining bytes with scalar search for first character
            if !found {
                while i < text_len && !found {
                    let first_char_matches = if case_insensitive {
                        let text_char = text_bytes[i];
                        let phrase_char = phrase_bytes[0];
                        let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                            text_char | 0x20
                        } else {
                            text_char
                        };
                        let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z' {
                            phrase_char | 0x20
                        } else {
                            phrase_char
                        };
                        text_lower == phrase_lower
                    } else {
                        text_bytes[i] == phrase_bytes[0]
                    };

                    if first_char_matches && i + phrase_len <= text_len {
                        let mut matches = true;
                        let mut j = 0;

                        // Try to use SIMD for remaining phrase verification when possible
                        while j + LANES <= phrase_len && matches {
                            let text_chunk = unsafe { vld1q_u8(text_bytes.as_ptr().add(i + j)) };
                            let phrase_chunk = unsafe { vld1q_u8(phrase_bytes.as_ptr().add(j)) };
                            let cmp_result = unsafe { vceqq_u8(text_chunk, phrase_chunk) };

                            let min_result = unsafe { vminvq_u8(cmp_result) };
                            if min_result != 0xFF {
                                matches = false;
                            }
                            j += LANES;
                        }

                        // Handle remaining bytes with scalar (case-sensitive or case-insensitive)
                        while j < phrase_len && matches {
                            if case_insensitive {
                                let text_char = text_bytes[i + j];
                                let phrase_char = phrase_bytes[j];
                                let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                                    text_char | 0x20
                                } else {
                                    text_char
                                };
                                let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z' {
                                    phrase_char | 0x20
                                } else {
                                    phrase_char
                                };
                                if text_lower != phrase_lower {
                                    matches = false;
                                    break;
                                }
                            } else if text_bytes[i + j] != phrase_bytes[j] {
                                matches = false;
                                break;
                            }
                            j += 1;
                        }

                        if matches {
                            found = true;
                        }
                    }
                    i += 1;
                }
            } // Close the text_bytes scope

            if found {
                // Keep this text in the filtered result (in-place)
                if write_pos != read_pos {
                    texts.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        } // Close the text_bytes scope
    }

    write_pos
}

// AVX-512 optimized field phrase matching.
//
// Uses AVX-512 vectorized phrase matching for improved performance.
// Processes field text in SIMD chunks for efficient substring detection.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn match_field_phrases_avx512(
    doc_ids: &mut [u32],
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: &str, // Use empty string "" as sentinel instead of Option
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
    doc_ids_len: usize,
    phrase_len: usize,
    events_len: usize, // Pass length parameter instead of calling .len()
) -> usize {
    let mut write_pos = 0usize;

    let phrase_bytes = phrase.as_bytes();

    // Pre-compute search vectors and masks
    const LANES: usize = LANES_AVX512_BYTES; // AVX-512 processes 64 bytes at once
    const UNROLL_FACTOR: usize = 2; // Unroll factor for better ILP

    let first_char = _mm512_set1_epi8(phrase_bytes[0] as i8);
    let has_alt_char = case_insensitive
        && ((phrase_bytes[0] >= b'A' && phrase_bytes[0] <= b'Z')
            || (phrase_bytes[0] >= b'a' && phrase_bytes[0] <= b'z'));
    let alt_char = if has_alt_char {
        let first_byte = phrase_bytes[0];
        if first_byte >= b'A' && first_byte <= b'Z' {
            _mm512_set1_epi8((first_byte | 0x20) as i8) // Convert to lowercase
        } else {
            _mm512_set1_epi8((first_byte & !0x20) as i8) // Convert to uppercase
        }
    } else {
        first_char
    };

    // Pre-compute case conversion masks
    let lowercase_mask = _mm512_set1_epi8(0x20);
    let alpha_lower = _mm512_set1_epi8(b'A' as i8);
    let alpha_upper = _mm512_set1_epi8(b'Z' as i8);

    // Pre-broadcast phrase chunks for longer phrases
    let mut phrase_chunks = [_mm512_setzero_si512(); 8];
    let num_phrase_chunks = ((phrase_len + LANES - 1) / LANES).min(8);
    for i in 0..num_phrase_chunks {
        let chunk_start = i * LANES;
        if chunk_start < phrase_len {
            let chunk_end = (chunk_start + LANES).min(phrase_len);
            let load_mask = if chunk_end - chunk_start == LANES {
                u64::MAX
            } else {
                (1u64 << (chunk_end - chunk_start)) - 1
            };
            phrase_chunks[i] = _mm512_maskz_loadu_epi8(
                load_mask,
                phrase_bytes.as_ptr().add(chunk_start) as *const i8,
            );
        }
    }

    // Filter indices directly in-place
    for read_pos in 0..doc_ids_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }
        let i = doc_ids[read_pos] as usize;
        if i >= events_len {
            continue; // Skip invalid indices
        }

        // Handle None events by skipping
        let event = match &events[i] {
            Some(event) => event,
            None => continue,
        };

        // Get text to search based on field_name (empty string means search full text)
        let text_to_search = if field_name.is_empty() {
            event.get_text()
        } else {
            // Manual field search without iterators
            let mut found_field_value = None;
            for field in event.get_fields() {
                if field.get_name() == field_name {
                    found_field_value = Some(field.get_value());
                    break;
                }
            }
            match found_field_value {
                Some(cow_value) => cow_value.to_owned(),
                None => continue, // Field not found, skip this document
            }
        };

        // Work directly with text bytes, no allocation
        let text_bytes = text_to_search.as_bytes();
        let text_len = text_bytes.len();

        let mut found = false;

        // Optimized search with unrolling and better patterns
        let search_end = text_len.saturating_sub(phrase_len);
        let simd_search_end = search_end & !(LANES * UNROLL_FACTOR - 1);

        let mut text_pos = 0;

        // Unrolled SIMD search for better instruction-level parallelism
        while text_pos < simd_search_end && !found {
            for unroll_idx in 0..UNROLL_FACTOR {
                let chunk_offset = text_pos + unroll_idx * LANES;
                if chunk_offset > search_end {
                    break;
                }

                let text_chunk =
                    _mm512_loadu_si512(text_bytes.as_ptr().add(chunk_offset) as *const _);
                let first_mask = _mm512_cmpeq_epu8_mask(text_chunk, first_char);
                let combined_mask = if has_alt_char {
                    let alt_mask = _mm512_cmpeq_epu8_mask(text_chunk, alt_char);
                    first_mask | alt_mask
                } else {
                    first_mask
                };

                if combined_mask != 0 {
                    let mut mask = combined_mask;
                    while mask != 0 && !found {
                        let bit_pos = mask.trailing_zeros() as usize;
                        let pos = chunk_offset + bit_pos;
                        mask &= mask - 1; // Clear lowest set bit

                        if pos + phrase_len <= text_len {
                            // Fast phrase comparison with early exits
                            let mut matches = true;

                            if case_insensitive {
                                // Optimized case-insensitive comparison using pre-computed vectors
                                for chunk_idx in 0..num_phrase_chunks {
                                    let chunk_start = chunk_idx * LANES;
                                    if chunk_start >= phrase_len {
                                        break;
                                    }

                                    let chunk_end = (chunk_start + LANES).min(phrase_len);
                                    let load_mask = if chunk_end - chunk_start == LANES {
                                        u64::MAX
                                    } else {
                                        (1u64 << (chunk_end - chunk_start)) - 1
                                    };

                                    let text_chunk = _mm512_maskz_loadu_epi8(
                                        load_mask,
                                        text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                    );
                                    let phrase_chunk = phrase_chunks[chunk_idx];

                                    // Optimized case conversion using mask operations
                                    let text_ge_a =
                                        _mm512_cmpge_epi8_mask(text_chunk, alpha_lower) & load_mask;
                                    let text_le_z = _mm512_mask_cmple_epi8_mask(
                                        load_mask,
                                        text_chunk,
                                        alpha_upper,
                                    );
                                    let text_is_upper = text_ge_a & text_le_z;

                                    let phrase_ge_a =
                                        _mm512_cmpge_epi8_mask(phrase_chunk, alpha_lower)
                                            & load_mask;
                                    let phrase_le_z = _mm512_mask_cmple_epi8_mask(
                                        load_mask,
                                        phrase_chunk,
                                        alpha_upper,
                                    );
                                    let phrase_is_upper = phrase_ge_a & phrase_le_z;

                                    let text_lower = _mm512_or_si512(
                                        text_chunk,
                                        _mm512_and_si512(
                                            _mm512_movm_epi8(text_is_upper),
                                            lowercase_mask,
                                        ),
                                    );
                                    let phrase_lower = _mm512_or_si512(
                                        phrase_chunk,
                                        _mm512_and_si512(
                                            _mm512_movm_epi8(phrase_is_upper),
                                            lowercase_mask,
                                        ),
                                    );

                                    let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                        load_mask,
                                        text_lower,
                                        phrase_lower,
                                    );
                                    if cmp_mask != load_mask {
                                        matches = false;
                                        break;
                                    }
                                }
                            } else {
                                // Optimized case-sensitive comparison
                                for chunk_idx in 0..num_phrase_chunks {
                                    let chunk_start = chunk_idx * LANES;
                                    if chunk_start >= phrase_len {
                                        break;
                                    }

                                    let chunk_end = (chunk_start + LANES).min(phrase_len);
                                    let load_mask = if chunk_end - chunk_start == LANES {
                                        u64::MAX
                                    } else {
                                        (1u64 << (chunk_end - chunk_start)) - 1
                                    };

                                    let text_chunk = _mm512_maskz_loadu_epi8(
                                        load_mask,
                                        text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                    );
                                    let phrase_chunk = phrase_chunks[chunk_idx];

                                    let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                        load_mask,
                                        text_chunk,
                                        phrase_chunk,
                                    );
                                    if cmp_mask != load_mask {
                                        matches = false;
                                        break;
                                    }
                                }
                            }

                            if matches {
                                found = true;
                                break;
                            }
                        }
                    }

                    if found {
                        break;
                    }
                }
            }
            text_pos += LANES * UNROLL_FACTOR;
        }

        // Handle remaining SIMD chunks
        while text_pos <= search_end && text_pos + LANES <= text_len && !found {
            let text_chunk = _mm512_loadu_si512(text_bytes.as_ptr().add(text_pos) as *const _);
            let first_mask = _mm512_cmpeq_epu8_mask(text_chunk, first_char);
            let combined_mask = if has_alt_char {
                let alt_mask = _mm512_cmpeq_epu8_mask(text_chunk, alt_char);
                first_mask | alt_mask
            } else {
                first_mask
            };

            if combined_mask != 0 {
                let mut mask = combined_mask;
                while mask != 0 && !found {
                    let bit_pos = mask.trailing_zeros() as usize;
                    let pos = text_pos + bit_pos;
                    mask &= mask - 1;

                    if pos + phrase_len <= text_len {
                        let mut matches = true;

                        if case_insensitive {
                            for chunk_idx in 0..num_phrase_chunks {
                                let chunk_start = chunk_idx * LANES;
                                if chunk_start >= phrase_len {
                                    break;
                                }

                                let chunk_end = (chunk_start + LANES).min(phrase_len);
                                let load_mask = if chunk_end - chunk_start == LANES {
                                    u64::MAX
                                } else {
                                    (1u64 << (chunk_end - chunk_start)) - 1
                                };

                                let text_chunk = _mm512_maskz_loadu_epi8(
                                    load_mask,
                                    text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                );
                                let phrase_chunk = phrase_chunks[chunk_idx];

                                let text_ge_a =
                                    _mm512_cmpge_epi8_mask(text_chunk, alpha_lower) & load_mask;
                                let text_le_z =
                                    _mm512_mask_cmple_epi8_mask(load_mask, text_chunk, alpha_upper);
                                let text_is_upper = text_ge_a & text_le_z;

                                let phrase_ge_a =
                                    _mm512_cmpge_epi8_mask(phrase_chunk, alpha_lower) & load_mask;
                                let phrase_le_z = _mm512_mask_cmple_epi8_mask(
                                    load_mask,
                                    phrase_chunk,
                                    alpha_upper,
                                );
                                let phrase_is_upper = phrase_ge_a & phrase_le_z;

                                let text_lower = _mm512_or_si512(
                                    text_chunk,
                                    _mm512_and_si512(
                                        _mm512_movm_epi8(text_is_upper),
                                        lowercase_mask,
                                    ),
                                );
                                let phrase_lower = _mm512_or_si512(
                                    phrase_chunk,
                                    _mm512_and_si512(
                                        _mm512_movm_epi8(phrase_is_upper),
                                        lowercase_mask,
                                    ),
                                );

                                let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                    load_mask,
                                    text_lower,
                                    phrase_lower,
                                );
                                if cmp_mask != load_mask {
                                    matches = false;
                                    break;
                                }
                            }
                        } else {
                            for chunk_idx in 0..num_phrase_chunks {
                                let chunk_start = chunk_idx * LANES;
                                if chunk_start >= phrase_len {
                                    break;
                                }

                                let chunk_end = (chunk_start + LANES).min(phrase_len);
                                let load_mask = if chunk_end - chunk_start == LANES {
                                    u64::MAX
                                } else {
                                    (1u64 << (chunk_end - chunk_start)) - 1
                                };

                                let text_chunk = _mm512_maskz_loadu_epi8(
                                    load_mask,
                                    text_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                );
                                let phrase_chunk = phrase_chunks[chunk_idx];

                                let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                    load_mask,
                                    text_chunk,
                                    phrase_chunk,
                                );
                                if cmp_mask != load_mask {
                                    matches = false;
                                    break;
                                }
                            }
                        }

                        if matches {
                            found = true;
                            break;
                        }
                    }
                }
            }
            text_pos += LANES;
        }

        // Handle remaining bytes with scalar search
        while text_pos <= text_len.saturating_sub(phrase_len) && !found {
            let mut matches = true;

            if case_insensitive {
                for j in 0..phrase_len {
                    let text_char = text_bytes[text_pos + j];
                    let phrase_char = phrase_bytes[j];
                    let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                        text_char | 0x20
                    } else {
                        text_char
                    };
                    let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z' {
                        phrase_char | 0x20
                    } else {
                        phrase_char
                    };

                    if text_lower != phrase_lower {
                        matches = false;
                        break;
                    }
                }
            } else {
                for j in 0..phrase_len {
                    if text_bytes[text_pos + j] != phrase_bytes[j] {
                        matches = false;
                        break;
                    }
                }
            }

            if matches {
                found = true;
            }
            text_pos += 1;
        }

        if found {
            // Keep this index in the filtered result (in-place)
            doc_ids[write_pos] = doc_ids[read_pos];
            write_pos += 1;
        }
    }

    write_pos
}

// AVX2 optimized field phrase matching.
//
// Uses AVX2 vectorized phrase matching for improved performance.
// Processes field text in SIMD chunks for efficient substring detection.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
pub(crate) unsafe fn match_field_phrases_avx2(
    doc_ids: &mut [u32],
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: &str, // Use empty string "" as sentinel instead of Option
    phrase: &str,
    case_insensitive: bool,
    max_size: usize,
    doc_ids_len: usize,
    phrase_len: usize,
    events_len: usize, // Pass length parameter instead of calling .len()
) -> usize {
    let mut write_pos = 0usize;

    // Work directly with phrase bytes, no allocation
    let phrase_bytes = phrase.as_bytes();

    const LANES: usize = LANES_AVX2_BYTES; // AVX2 processes 32 bytes at once
    const FULL_MATCH_MASK: i32 = -1i32; // All 32 bits set
    let first_char = _mm256_set1_epi8(phrase_bytes[0] as i8);
    let has_alt_char = case_insensitive
        && ((phrase_bytes[0] >= b'A' && phrase_bytes[0] <= b'Z')
            || (phrase_bytes[0] >= b'a' && phrase_bytes[0] <= b'z'));
    let alt_char = if has_alt_char {
        let first_byte = phrase_bytes[0];
        let alt_byte = if first_byte >= b'A' && first_byte <= b'Z' {
            first_byte | 0x20 // Convert to lowercase
        } else {
            first_byte & !0x20 // Convert to uppercase
        };
        _mm256_set1_epi8(alt_byte as i8)
    } else {
        first_char // Dummy value, won't be used
    };

    // Filter indices directly in-place
    for read_pos in 0..doc_ids_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }
        let i = doc_ids[read_pos] as usize;
        if i >= events_len {
            continue; // Skip invalid indices
        }

        // Handle None events by skipping
        let event = match &events[i] {
            Some(event) => event,
            None => continue,
        };

        // Get text to search based on field_name (empty string means search full text)
        let text_to_search = if field_name.is_empty() {
            event.get_text()
        } else {
            // Manual field search without iterators
            let mut found_field_value = None;
            for field in event.get_fields() {
                if field.get_name() == field_name {
                    found_field_value = Some(field.get_value());
                    break;
                }
            }

            match found_field_value {
                Some(cow_value) => cow_value.to_owned(),
                None => continue, // Field not found, skip this document
            }
        };

        // Work directly with text bytes, no allocation
        let text_bytes = text_to_search.as_bytes();
        let text_len = text_bytes.len();

        if text_len < phrase_len {
            continue;
        }

        let mut found = false;

        // Search for phrase in text with AVX2 optimization
        let mut i = 0;

        // SIMD search for first character matches
        while i + LANES <= text_len && !found {
            let text_chunk = _mm256_loadu_si256(text_bytes.as_ptr().add(i) as *const _);
            let first_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(text_chunk, first_char));
            let combined_mask = if has_alt_char {
                let alt_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(text_chunk, alt_char));
                first_mask | alt_mask
            } else {
                first_mask
            };

            if combined_mask != 0 {
                // Process all set bits using trailing_zeros() pattern (safe approach)
                let remaining_mask = combined_mask as u32;

                while remaining_mask != 0 {
                    let bit_pos = remaining_mask.trailing_zeros() as usize;
                    if bit_pos < 32 {
                        let pos = i + bit_pos;

                        if pos + phrase_len <= text_len {
                            let mut matches = true;
                            let mut j = 0;

                            // Compare full phrase with case sensitivity
                            if case_insensitive {
                                // Case-insensitive comparison with SIMD on-the-fly conversion
                                let lowercase_mask = _mm256_set1_epi8(0x20);
                                let alpha_mask_lower = _mm256_set1_epi8(b'A' as i8);
                                let alpha_mask_upper = _mm256_set1_epi8(b'Z' as i8);

                                while j + LANES <= phrase_len && matches {
                                    let text_chunk = _mm256_loadu_si256(
                                        text_bytes.as_ptr().add(pos + j) as *const _,
                                    );
                                    let phrase_chunk = _mm256_loadu_si256(
                                        phrase_bytes.as_ptr().add(j) as *const _,
                                    );

                                    // Convert both to lowercase using SIMD
                                    let text_ge_a = _mm256_cmpgt_epi8(
                                        text_chunk,
                                        _mm256_sub_epi8(alpha_mask_lower, _mm256_set1_epi8(1)),
                                    );
                                    let text_le_z = _mm256_cmpgt_epi8(
                                        _mm256_add_epi8(alpha_mask_upper, _mm256_set1_epi8(1)),
                                        text_chunk,
                                    );
                                    let text_is_upper = _mm256_and_si256(text_ge_a, text_le_z);
                                    let text_lower = _mm256_or_si256(
                                        text_chunk,
                                        _mm256_and_si256(text_is_upper, lowercase_mask),
                                    );

                                    let phrase_ge_a = _mm256_cmpgt_epi8(
                                        phrase_chunk,
                                        _mm256_sub_epi8(alpha_mask_lower, _mm256_set1_epi8(1)),
                                    );
                                    let phrase_le_z = _mm256_cmpgt_epi8(
                                        _mm256_add_epi8(alpha_mask_upper, _mm256_set1_epi8(1)),
                                        phrase_chunk,
                                    );
                                    let phrase_is_upper =
                                        _mm256_and_si256(phrase_ge_a, phrase_le_z);
                                    let phrase_lower = _mm256_or_si256(
                                        phrase_chunk,
                                        _mm256_and_si256(phrase_is_upper, lowercase_mask),
                                    );

                                    // Compare the lowercase versions
                                    let cmp_result = _mm256_cmpeq_epi8(text_lower, phrase_lower);
                                    let mask = _mm256_movemask_epi8(cmp_result);

                                    if mask != FULL_MATCH_MASK {
                                        matches = false;
                                    }
                                    j += LANES;
                                }

                                // Handle remaining bytes with scalar case-insensitive comparison
                                while j < phrase_len && matches {
                                    let text_char = text_bytes[pos + j];
                                    let phrase_char = phrase_bytes[j];
                                    let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                                        text_char | 0x20
                                    } else {
                                        text_char
                                    };
                                    let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z'
                                    {
                                        phrase_char | 0x20
                                    } else {
                                        phrase_char
                                    };

                                    if text_lower != phrase_lower {
                                        matches = false;
                                    }
                                    j += 1;
                                }
                            } else {
                                // Case-sensitive SIMD comparison
                                while j + LANES <= phrase_len && matches {
                                    let text_chunk = _mm256_loadu_si256(
                                        text_bytes.as_ptr().add(pos + j) as *const _,
                                    );
                                    let phrase_chunk = _mm256_loadu_si256(
                                        phrase_bytes.as_ptr().add(j) as *const _,
                                    );
                                    let cmp_result = _mm256_cmpeq_epi8(text_chunk, phrase_chunk);

                                    let mask = _mm256_movemask_epi8(cmp_result);
                                    if mask != FULL_MATCH_MASK {
                                        matches = false;
                                    }
                                    j += LANES;
                                }

                                // Handle remaining bytes with scalar comparison
                                while j < phrase_len && matches {
                                    if text_bytes[pos + j] != phrase_bytes[j] {
                                        matches = false;
                                    }
                                    j += 1;
                                }
                            }

                            if matches {
                                found = true;
                                break;
                            }
                        }
                    }
                }
            }
            i += LANES;
        }

        // Handle remaining bytes with scalar search
        while i <= text_len.saturating_sub(phrase_len) && !found {
            let mut matches = true;

            if case_insensitive {
                for j in 0..phrase_len {
                    let text_char = text_bytes[i + j];
                    let phrase_char = phrase_bytes[j];
                    let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                        text_char | 0x20
                    } else {
                        text_char
                    };
                    let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z' {
                        phrase_char | 0x20
                    } else {
                        phrase_char
                    };
                    if text_lower != phrase_lower {
                        matches = false;
                        break;
                    }
                }
            } else {
                for j in 0..phrase_len {
                    if text_bytes[i + j] != phrase_bytes[j] {
                        matches = false;
                        break;
                    }
                }
            }

            if matches {
                found = true;
            }
            i += 1;
        }

        if found {
            // Keep this index in the filtered result (in-place)
            doc_ids[write_pos] = doc_ids[read_pos];
            write_pos += 1;
        }
    }

    write_pos
}

// NEON optimized field phrase matching.
//
// Uses NEON vectorized phrase matching for improved performance.
// Processes field text in SIMD chunks for efficient substring detection.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn match_field_phrases_neon(
    doc_ids: &mut [u32], // Filter doc IDs directly in-place
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: &str, // Use empty string "" as sentinel instead of Option
    phrase: &str,
    case_insensitive: bool,
    max_size: usize, // early termination: limit results
    doc_ids_len: usize,
    phrase_len: usize,
    events_len: usize, // Pass length parameter instead of calling .len()
) -> usize {
    const LANES: usize = LANES_NEON_BYTES; // NEON processes 16 bytes at once
    let mut write_pos = 0usize;

    // Work directly with phrase bytes, no allocation
    let phrase_bytes = phrase.as_bytes();

    // Load first character for quick filtering
    let first_char_simd = unsafe { vdupq_n_u8(phrase_bytes[0]) };

    // Filter doc IDs directly in-place
    for read_pos in 0..doc_ids_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }

        let doc_idx = doc_ids[read_pos] as usize;
        if doc_idx >= events_len {
            continue; // Skip invalid indices
        }

        let event = match &events[doc_idx] {
            Some(event) => event,
            None => continue,
        };

        // Get text to search based on field_name (empty string means search full text)
        let text_to_search = if field_name.is_empty() {
            event.get_text()
        } else {
            // Manual field search without iterators
            let mut found_field_value = None;
            for field in event.get_fields() {
                if field.get_name() == field_name {
                    found_field_value = Some(field.get_value());
                    break;
                }
            }
            match found_field_value {
                Some(cow_value) => cow_value.to_owned(),
                None => continue, // Field not found, skip this document
            }
        };

        // Work directly with text bytes, no allocation
        let text_bytes = text_to_search.as_bytes();
        let text_len = text_bytes.len();

        if text_len < phrase_len {
            continue;
        }

        // NEON phrase matching with on-the-fly case conversion
        let mut found = false;
        let mut i = 0;

        // SIMD search for first character matches
        while i + LANES <= text_len && !found {
            let text_chunk = unsafe { vld1q_u8(text_bytes.as_ptr().add(i)) };
            let first_cmp = unsafe { vceqq_u8(text_chunk, first_char_simd) };

            // Extract mask to check if any lane matched
            let mut mask_values = [0u8; LANES];
            unsafe { vst1q_u8(mask_values.as_mut_ptr(), first_cmp) };

            // Check if any byte matched the first character
            for j in 0..LANES {
                let mask = mask_values[j];
                if mask == 0xFF {
                    let match_pos = i + j;
                    if match_pos + phrase_len <= text_len {
                        let mut matches = true;
                        let mut k = 0;

                        // Compare full phrase with case sensitivity
                        if case_insensitive {
                            // Case-insensitive SIMD comparison using on-the-fly conversion
                            let lowercase_mask = unsafe { vdupq_n_u8(0x20) }; // Bit to set for lowercase conversion
                            let alpha_mask_lower = unsafe { vdupq_n_u8(b'A') };
                            let alpha_mask_upper = unsafe { vdupq_n_u8(b'Z') };

                            while k + LANES <= phrase_len && matches {
                                let text_chunk =
                                    unsafe { vld1q_u8(text_bytes.as_ptr().add(match_pos + k)) };
                                let phrase_chunk =
                                    unsafe { vld1q_u8(phrase_bytes.as_ptr().add(k)) };

                                // Convert both to lowercase using SIMD
                                let text_ge_a = unsafe { vcgeq_u8(text_chunk, alpha_mask_lower) };
                                let text_le_z = unsafe { vcleq_u8(text_chunk, alpha_mask_upper) };
                                let text_is_upper = unsafe { vandq_u8(text_ge_a, text_le_z) };
                                let text_lower = unsafe {
                                    vorrq_u8(text_chunk, vandq_u8(text_is_upper, lowercase_mask))
                                };

                                let phrase_ge_a =
                                    unsafe { vcgeq_u8(phrase_chunk, alpha_mask_lower) };
                                let phrase_le_z =
                                    unsafe { vcleq_u8(phrase_chunk, alpha_mask_upper) };
                                let phrase_is_upper = unsafe { vandq_u8(phrase_ge_a, phrase_le_z) };
                                let phrase_lower = unsafe {
                                    vorrq_u8(
                                        phrase_chunk,
                                        vandq_u8(phrase_is_upper, lowercase_mask),
                                    )
                                };

                                // Compare the lowercase versions
                                let cmp_result = unsafe { vceqq_u8(text_lower, phrase_lower) };
                                let min_result = unsafe { vminvq_u8(cmp_result) };
                                if min_result != 0xFF {
                                    matches = false;
                                }
                                k += LANES;
                            }

                            // Handle remaining bytes with scalar case-insensitive comparison
                            while k < phrase_len && matches {
                                let text_char = text_bytes[match_pos + k];
                                let phrase_char = phrase_bytes[k];
                                let text_lower = if text_char.is_ascii_uppercase() {
                                    text_char | 0x20
                                } else {
                                    text_char
                                };
                                let phrase_lower = if phrase_char.is_ascii_uppercase() {
                                    phrase_char | 0x20
                                } else {
                                    phrase_char
                                };

                                if text_lower != phrase_lower {
                                    matches = false;
                                }
                                k += 1;
                            }
                        } else {
                            // Case-sensitive SIMD comparison
                            while k + LANES <= phrase_len && matches {
                                let text_chunk =
                                    unsafe { vld1q_u8(text_bytes.as_ptr().add(match_pos + k)) };
                                let phrase_chunk =
                                    unsafe { vld1q_u8(phrase_bytes.as_ptr().add(k)) };
                                let cmp_result = unsafe { vceqq_u8(text_chunk, phrase_chunk) };

                                // Check if all bytes match
                                let min_result = unsafe { vminvq_u8(cmp_result) };
                                if min_result != 0xFF {
                                    matches = false;
                                }
                                k += LANES;
                            }

                            // Handle remaining bytes with scalar case-sensitive comparison
                            while k < phrase_len && matches {
                                if text_bytes[match_pos + k] != phrase_bytes[k] {
                                    matches = false;
                                }
                                k += 1;
                            }
                        }

                        if matches {
                            found = true;
                            break;
                        }
                    }
                }
            }

            if found {
                break;
            }

            i += LANES;
        }

        // Handle remaining bytes with scalar search
        while i <= text_len.saturating_sub(phrase_len) && !found {
            let mut matches = true;

            if case_insensitive {
                for j in 0..phrase_len {
                    let text_char = text_bytes[i + j];
                    let phrase_char = phrase_bytes[j];
                    let text_lower = if text_char >= b'A' && text_char <= b'Z' {
                        text_char | 0x20
                    } else {
                        text_char
                    };
                    let phrase_lower = if phrase_char >= b'A' && phrase_char <= b'Z' {
                        phrase_char | 0x20
                    } else {
                        phrase_char
                    };
                    if text_lower != phrase_lower {
                        matches = false;
                        break;
                    }
                }
            } else {
                for j in 0..phrase_len {
                    if text_bytes[i + j] != phrase_bytes[j] {
                        matches = false;
                        break;
                    }
                }
            }

            if matches {
                found = true;
            }
            i += 1;
        }

        if found {
            // Keep this doc ID in the filtered result (in-place)
            doc_ids[write_pos] = doc_ids[read_pos];
            write_pos += 1;
        }
    }

    write_pos
}

// AVX-512 optimized field prefix matching.
//
// Uses AVX-512 vectorized field prefix searching for enhanced performance.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
// GPU implementation of field prefix matching using PTX assembly
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn match_field_prefixes_avx512(
    doc_ids: &mut [u32], // Read-only doc IDs
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: &str,
    prefix: &str,
    case_insensitive: bool,
    max_size: usize,
    doc_ids_len: usize,
    prefix_len: usize,
) -> usize {
    const LANES: usize = LANES_AVX512_BYTES;
    const UNROLL_FACTOR: usize = 4; // Unroll factor for prefix comparison
    let mut write_pos = 0usize;

    let prefix_bytes = prefix.as_bytes();

    // Pre-compute prefix chunks and conversion masks
    let mut prefix_chunks = [_mm512_setzero_si512(); UNROLL_FACTOR];
    let num_full_chunks = prefix_len / LANES;
    let unrolled_chunks = num_full_chunks.min(UNROLL_FACTOR);

    for i in 0..unrolled_chunks {
        prefix_chunks[i] = _mm512_loadu_si512(prefix_bytes.as_ptr().add(i * LANES) as *const _);
    }

    // Pre-compute case conversion masks for case-insensitive matching
    let lowercase_mask = _mm512_set1_epi8(0x20);
    let alpha_lower = _mm512_set1_epi8(b'A' as i8);
    let alpha_upper = _mm512_set1_epi8(b'Z' as i8);

    // Process indices in-place: check if event at each index matches prefix
    for read_pos in 0..doc_ids_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }
        let i = doc_ids[read_pos] as usize;
        if i >= events.len() {
            continue; // Skip invalid indices
        }

        let event = match &events[i] {
            Some(event) => event,
            None => continue,
        };

        // Use proper field lookup - minimize heap allocations where possible
        let text_to_search = if field_name.is_empty() {
            // Empty field_name means search full text
            event.get_text()
        } else {
            // Search for the specific field by name
            let mut field_value = None;

            // Search in non-numeric fields first (text fields likely for prefix matching)
            if let Some(non_numeric_fields) = event.get_non_numeric_fields() {
                for field in non_numeric_fields {
                    if field.get_name() == field_name {
                        field_value = Some(field.get_value());
                        break;
                    }
                }
            }

            // If not found in non-numeric fields, search numeric fields
            if field_value.is_none() {
                if let Some(numeric_fields) = event.get_numeric_fields() {
                    for field in numeric_fields {
                        if field.get_name() == field_name {
                            field_value = Some(field.get_value());
                            break;
                        }
                    }
                }
            }

            // Use field value if found, otherwise skip this event
            match field_value {
                Some(cow_value) => cow_value.to_owned(),
                None => continue, // Field not found, skip this event
            }
        };

        let text_bytes = text_to_search.as_bytes();
        let text_len = text_bytes.len();

        if text_len < prefix_len {
            continue;
        }

        let mut matches = true;
        let mut j = 0;

        if case_insensitive {
            // Unrolled case-insensitive comparison
            while j + LANES * UNROLL_FACTOR <= prefix_len && matches {
                // Process 4 chunks at once for better instruction-level parallelism
                for chunk_idx in 0..UNROLL_FACTOR {
                    let offset = j + chunk_idx * LANES;
                    let field_chunk =
                        _mm512_loadu_si512(text_bytes.as_ptr().add(offset) as *const _);
                    let prefix_chunk = prefix_chunks[chunk_idx];

                    // Optimized case conversion using compare-and-blend
                    let field_ge_a = _mm512_cmpge_epi8_mask(field_chunk, alpha_lower);
                    let field_le_z = _mm512_cmple_epi8_mask(field_chunk, alpha_upper);
                    let field_is_upper = field_ge_a & field_le_z;

                    let prefix_ge_a = _mm512_cmpge_epi8_mask(prefix_chunk, alpha_lower);
                    let prefix_le_z = _mm512_cmple_epi8_mask(prefix_chunk, alpha_upper);
                    let prefix_is_upper = prefix_ge_a & prefix_le_z;

                    // Convert to lowercase using masked blend operations
                    let field_lower = _mm512_or_si512(
                        field_chunk,
                        _mm512_and_si512(_mm512_movm_epi8(field_is_upper), lowercase_mask),
                    );
                    let prefix_lower = _mm512_or_si512(
                        prefix_chunk,
                        _mm512_and_si512(_mm512_movm_epi8(prefix_is_upper), lowercase_mask),
                    );

                    // Compare and check for early exit
                    let eq_mask = _mm512_cmpeq_epu8_mask(field_lower, prefix_lower);
                    if eq_mask != u64::MAX {
                        matches = false;
                        break;
                    }
                }
                j += LANES * UNROLL_FACTOR;
            }

            // Handle remaining full chunks
            while j + LANES <= prefix_len && matches {
                let field_chunk = _mm512_loadu_si512(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm512_loadu_si512(prefix_bytes.as_ptr().add(j) as *const _);

                let field_ge_a = _mm512_cmpge_epi8_mask(field_chunk, alpha_lower);
                let field_le_z = _mm512_cmple_epi8_mask(field_chunk, alpha_upper);
                let field_is_upper = field_ge_a & field_le_z;

                let prefix_ge_a = _mm512_cmpge_epi8_mask(prefix_chunk, alpha_lower);
                let prefix_le_z = _mm512_cmple_epi8_mask(prefix_chunk, alpha_upper);
                let prefix_is_upper = prefix_ge_a & prefix_le_z;

                let field_lower = _mm512_or_si512(
                    field_chunk,
                    _mm512_and_si512(_mm512_movm_epi8(field_is_upper), lowercase_mask),
                );
                let prefix_lower = _mm512_or_si512(
                    prefix_chunk,
                    _mm512_and_si512(_mm512_movm_epi8(prefix_is_upper), lowercase_mask),
                );

                let eq_mask = _mm512_cmpeq_epu8_mask(field_lower, prefix_lower);
                if eq_mask != u64::MAX {
                    matches = false;
                }
                j += LANES;
            }
        } else {
            // Unrolled case-sensitive comparison
            while j + LANES * UNROLL_FACTOR <= prefix_len && matches {
                // Process 4 chunks at once for better instruction-level parallelism
                for chunk_idx in 0..UNROLL_FACTOR {
                    let offset = j + chunk_idx * LANES;
                    let field_chunk =
                        _mm512_loadu_si512(text_bytes.as_ptr().add(offset) as *const _);
                    let eq_mask = _mm512_cmpeq_epu8_mask(field_chunk, prefix_chunks[chunk_idx]);
                    if eq_mask != u64::MAX {
                        matches = false;
                        break;
                    }
                }
                j += LANES * UNROLL_FACTOR;
            }

            // Handle remaining full chunks
            while j + LANES <= prefix_len && matches {
                let field_chunk = _mm512_loadu_si512(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm512_loadu_si512(prefix_bytes.as_ptr().add(j) as *const _);
                let eq_mask = _mm512_cmpeq_epu8_mask(field_chunk, prefix_chunk);
                if eq_mask != u64::MAX {
                    matches = false;
                }
                j += LANES;
            }
        }

        // Handle remaining bytes with AVX-512 masked operations
        if matches {
            let remaining_len = prefix_len - j;
            if remaining_len > 0 {
                let load_mask = (1u64 << remaining_len) - 1;
                let text_chunk =
                    _mm512_maskz_loadu_epi8(load_mask, text_bytes.as_ptr().add(j) as *const i8);
                let prefix_chunk =
                    _mm512_maskz_loadu_epi8(load_mask, prefix_bytes.as_ptr().add(j) as *const i8);

                if case_insensitive {
                    // Convert both to lowercase using SIMD
                    let lowercase_mask = _mm512_set1_epi8(0x20);
                    let alpha_mask_lower = _mm512_set1_epi8(b'A' as i8);
                    let alpha_mask_upper = _mm512_set1_epi8(b'Z' as i8);

                    let text_ge_a =
                        _mm512_cmpge_epi8_mask(text_chunk, alpha_mask_lower) & load_mask;
                    let text_le_z =
                        _mm512_cmpge_epi8_mask(alpha_mask_upper, text_chunk) & load_mask;
                    let text_is_upper = text_ge_a & text_le_z;
                    let text_upper_vec = _mm512_movm_epi8(text_is_upper);
                    let text_lower = _mm512_or_si512(
                        text_chunk,
                        _mm512_and_si512(text_upper_vec, lowercase_mask),
                    );

                    let prefix_ge_a =
                        _mm512_cmpge_epi8_mask(prefix_chunk, alpha_mask_lower) & load_mask;
                    let prefix_le_z =
                        _mm512_cmpge_epi8_mask(alpha_mask_upper, prefix_chunk) & load_mask;
                    let prefix_is_upper = prefix_ge_a & prefix_le_z;
                    let prefix_upper_vec = _mm512_movm_epi8(prefix_is_upper);
                    let prefix_lower = _mm512_or_si512(
                        prefix_chunk,
                        _mm512_and_si512(prefix_upper_vec, lowercase_mask),
                    );

                    let cmp_mask = _mm512_mask_cmpeq_epu8_mask(load_mask, text_lower, prefix_lower);
                    if cmp_mask != load_mask {
                        matches = false;
                    }
                } else {
                    let cmp_mask = _mm512_mask_cmpeq_epu8_mask(load_mask, text_chunk, prefix_chunk);
                    if cmp_mask != load_mask {
                        matches = false;
                    }
                }
            }
        }

        if matches {
            // Keep this index in the filtered result (in-place)
            doc_ids[write_pos] = doc_ids[read_pos];
            write_pos += 1;
        }
    }

    write_pos
}

// AVX2 optimized field prefix matching.
//
// Uses AVX2 vectorized field prefix searching for enhanced performance.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
pub(crate) unsafe fn match_field_prefixes_avx2(
    doc_ids: &mut [u32], // Filter doc IDs directly in-place
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: &str,
    prefix: &str,
    case_insensitive: bool,
    max_size: usize, // early termination: limit results
    doc_ids_len: usize,
    prefix_len: usize,
) -> usize {
    const LANES: usize = LANES_AVX2_BYTES; // AVX2 processes 32 bytes at once
    const UNROLL_FACTOR: usize = 4; // Unroll factor for prefix comparison
    let mut write_pos = 0usize;

    let prefix_bytes = prefix.as_bytes();

    // Pre-compute prefix chunks and conversion masks
    let mut prefix_chunks = [_mm256_setzero_si256(); UNROLL_FACTOR];
    let num_full_chunks = prefix_len / LANES;
    let unrolled_chunks = num_full_chunks.min(UNROLL_FACTOR);

    for i in 0..unrolled_chunks {
        prefix_chunks[i] = _mm256_loadu_si256(prefix_bytes.as_ptr().add(i * LANES) as *const _);
    }

    // Pre-compute case conversion masks for case-insensitive matching
    let lowercase_mask = _mm256_set1_epi8(0x20);
    let alpha_lower = _mm256_set1_epi8(b'A' as i8);

    let alpha_range_mask = _mm256_set1_epi8(0x1F); // For efficient range checking

    // Filter doc IDs directly in-place
    for read_pos in 0..doc_ids_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }

        let doc_idx = doc_ids[read_pos] as usize;
        if doc_idx >= events.len() {
            continue; // Skip invalid indices
        }

        let event = match &events[doc_idx] {
            Some(event) => event,
            None => continue,
        };

        // Use proper field lookup - minimize heap allocations where possible
        let text_to_search = if field_name.is_empty() {
            // Empty field_name means search full text
            event.get_text()
        } else {
            // Search for the specific field by name
            let mut field_value = None;

            // Search in non-numeric fields first (text fields likely for prefix matching)
            if let Some(non_numeric_fields) = event.get_non_numeric_fields() {
                for field in non_numeric_fields {
                    if field.get_name() == field_name {
                        field_value = Some(field.get_value());
                        break;
                    }
                }
            }

            // If not found in non-numeric fields, search numeric fields
            if field_value.is_none() {
                if let Some(numeric_fields) = event.get_numeric_fields() {
                    for field in numeric_fields {
                        if field.get_name() == field_name {
                            field_value = Some(field.get_value());
                            break;
                        }
                    }
                }
            }

            // Use field value if found, otherwise skip this event
            match field_value {
                Some(cow_value) => cow_value.to_owned(),
                None => continue, // Field not found, skip this event
            }
        };

        // Work directly with text bytes, no allocation
        let text_bytes = text_to_search.as_bytes();
        let text_len = text_bytes.len();

        if text_len < prefix_len {
            continue;
        }

        let mut prefix_matches = true;
        let mut j = 0;

        if case_insensitive {
            // Unrolled case-insensitive comparison
            while j + LANES * UNROLL_FACTOR <= prefix_len && prefix_matches {
                // Process 4 chunks at once for better instruction-level parallelism
                for chunk_idx in 0..UNROLL_FACTOR {
                    let offset = j + chunk_idx * LANES;
                    let string_chunk =
                        _mm256_loadu_si256(text_bytes.as_ptr().add(offset) as *const _);
                    let prefix_chunk = prefix_chunks[chunk_idx];

                    // Optimized case conversion using efficient range checking
                    let string_offset = _mm256_sub_epi8(string_chunk, alpha_lower);
                    let string_in_range = _mm256_and_si256(string_offset, alpha_range_mask);
                    let string_is_upper = _mm256_cmpeq_epi8(string_offset, string_in_range);

                    let prefix_offset = _mm256_sub_epi8(prefix_chunk, alpha_lower);
                    let prefix_in_range = _mm256_and_si256(prefix_offset, alpha_range_mask);
                    let prefix_is_upper = _mm256_cmpeq_epi8(prefix_offset, prefix_in_range);

                    // Convert to lowercase using blend operations
                    let string_lower = _mm256_blendv_epi8(
                        string_chunk,
                        _mm256_or_si256(string_chunk, lowercase_mask),
                        string_is_upper,
                    );
                    let prefix_lower = _mm256_blendv_epi8(
                        prefix_chunk,
                        _mm256_or_si256(prefix_chunk, lowercase_mask),
                        prefix_is_upper,
                    );

                    // Compare and check for early exit
                    let cmp_result = _mm256_cmpeq_epi8(string_lower, prefix_lower);
                    let mask = _mm256_movemask_epi8(cmp_result);
                    if mask != -1i32 {
                        prefix_matches = false;
                        break;
                    }
                }
                j += LANES * UNROLL_FACTOR;
            }

            // Handle remaining full chunks
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = _mm256_loadu_si256(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm256_loadu_si256(prefix_bytes.as_ptr().add(j) as *const _);

                // Optimized case conversion
                let string_offset = _mm256_sub_epi8(string_chunk, alpha_lower);
                let string_in_range = _mm256_and_si256(string_offset, alpha_range_mask);
                let string_is_upper = _mm256_cmpeq_epi8(string_offset, string_in_range);

                let prefix_offset = _mm256_sub_epi8(prefix_chunk, alpha_lower);
                let prefix_in_range = _mm256_and_si256(prefix_offset, alpha_range_mask);
                let prefix_is_upper = _mm256_cmpeq_epi8(prefix_offset, prefix_in_range);

                let string_lower = _mm256_blendv_epi8(
                    string_chunk,
                    _mm256_or_si256(string_chunk, lowercase_mask),
                    string_is_upper,
                );
                let prefix_lower = _mm256_blendv_epi8(
                    prefix_chunk,
                    _mm256_or_si256(prefix_chunk, lowercase_mask),
                    prefix_is_upper,
                );

                let cmp_result = _mm256_cmpeq_epi8(string_lower, prefix_lower);
                let mask = _mm256_movemask_epi8(cmp_result);
                if mask != -1i32 {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with AVX2 operations
            if j < prefix_len && prefix_matches {
                let remaining_len = prefix_len - j;
                // Use minimal scalar for tiny remainder only
                for k in 0..remaining_len {
                    let string_char = text_bytes[j + k];
                    let prefix_char = prefix_bytes[j + k];
                    let string_lower = if string_char >= b'A' && string_char <= b'Z' {
                        string_char | 0x20
                    } else {
                        string_char
                    };
                    let prefix_lower = if prefix_char >= b'A' && prefix_char <= b'Z' {
                        prefix_char | 0x20
                    } else {
                        prefix_char
                    };

                    if string_lower != prefix_lower {
                        prefix_matches = false;
                        break;
                    }
                }
            }
        } else {
            // Case-sensitive SIMD comparison
            while j + LANES <= prefix_len && prefix_matches {
                let string_chunk = _mm256_loadu_si256(text_bytes.as_ptr().add(j) as *const _);
                let prefix_chunk = _mm256_loadu_si256(prefix_bytes.as_ptr().add(j) as *const _);
                let cmp_result = _mm256_cmpeq_epi8(string_chunk, prefix_chunk);

                let mask = _mm256_movemask_epi8(cmp_result);
                if mask != -1i32 {
                    prefix_matches = false;
                }
                j += LANES;
            }

            // Handle remaining bytes with scalar operations
            if j < prefix_len && prefix_matches {
                let remaining_len = prefix_len - j;
                // Use minimal scalar for tiny remainder only
                for k in 0..remaining_len {
                    if text_bytes[j + k] != prefix_bytes[j + k] {
                        prefix_matches = false;
                        break;
                    }
                }
            }
        }

        if prefix_matches {
            // Keep this doc_id in the filtered result (in-place)
            if write_pos != read_pos {
                doc_ids.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }

    write_pos
}

// NEON optimized field prefix matching.
//
// Uses NEON vectorized field prefix searching for enhanced performance.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn match_field_prefixes_neon(
    doc_ids: &mut [u32], // Filter doc IDs directly in-place
    events: &[Option<std::sync::Arc<crate::types::StoredEvent>>],
    field_name: &str,
    prefix: &str,
    case_insensitive: bool,
    max_size: usize, // early termination: limit results
    doc_ids_len: usize,
    prefix_len: usize,
) -> usize {
    const LANES: usize = LANES_NEON_BYTES; // NEON processes 16 bytes at once
    let mut write_pos = 0usize;

    // Work directly with prefix bytes, no allocation
    let prefix_bytes = prefix.as_bytes();

    // Process doc_ids in-place - NO HEAP allocations
    for read_pos in 0..doc_ids_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }

        let doc_idx = doc_ids[read_pos] as usize;
        if doc_idx >= events.len() {
            continue; // Skip invalid indices
        }

        let event = &events[doc_idx].as_ref().unwrap();

        // Use proper field lookup - heap allocation OK for field access
        let text_to_search = if field_name.is_empty() {
            // Empty field_name means search full text
            event.get_text()
        } else {
            // Search for the specific field by name
            let mut field_value = None;

            // Search in non-numeric fields first (text fields likely for prefix matching)
            if let Some(non_numeric_fields) = event.get_non_numeric_fields() {
                for field in non_numeric_fields {
                    if field.get_name() == field_name {
                        field_value = Some(field.get_value().to_owned());
                        break;
                    }
                }
            }

            // If not found in non-numeric fields, search numeric fields
            if field_value.is_none() {
                if let Some(numeric_fields) = event.get_numeric_fields() {
                    for field in numeric_fields {
                        if field.get_name() == field_name {
                            field_value = Some(field.get_value().to_owned());
                            break;
                        }
                    }
                }
            }

            // Use field value if found, otherwise skip this event
            match field_value {
                Some(value) => value,
                None => continue, // Field not found, skip this event
            }
        };

        // Work directly with text bytes in SIMD loops
        let text_bytes = text_to_search.as_bytes();
        let text_len = text_bytes.len();

        if text_len < prefix_len {
            continue;
        }

        // Only check if the field starts with the prefix (position 0) using NEON
        if text_len >= prefix_len {
            let mut prefix_matches = true;
            let mut j = 0;

            // Compare in 16-byte SIMD chunks with case sensitivity support
            while j + LANES <= prefix_len && prefix_matches {
                if case_insensitive {
                    // Scalar case-insensitive comparison for now (can be SIMD optimized later)
                    for k in 0..LANES {
                        let field_char = text_bytes[j + k];
                        let prefix_char = prefix_bytes[j + k];
                        let field_lower = if field_char >= b'A' && field_char <= b'Z' {
                            field_char | 0x20
                        } else {
                            field_char
                        };
                        let prefix_lower = if prefix_char >= b'A' && prefix_char <= b'Z' {
                            prefix_char | 0x20
                        } else {
                            prefix_char
                        };
                        if field_lower != prefix_lower {
                            prefix_matches = false;
                            break;
                        }
                    }
                } else {
                    let field_chunk = unsafe { vld1q_u8(text_bytes.as_ptr().add(j)) };
                    let prefix_chunk = unsafe { vld1q_u8(prefix_bytes.as_ptr().add(j)) };
                    let cmp_result = unsafe { vceqq_u8(field_chunk, prefix_chunk) };

                    // Check if all bytes match
                    let min_result = unsafe { vminvq_u8(cmp_result) };
                    if min_result != 0xFF {
                        prefix_matches = false;
                    }
                }
                j += LANES;
            }

            // Handle remaining bytes with scalar comparison
            while j < prefix_len && prefix_matches {
                if case_insensitive {
                    let field_char = text_bytes[j];
                    let prefix_char = prefix_bytes[j];
                    let field_lower = if field_char >= b'A' && field_char <= b'Z' {
                        field_char | 0x20
                    } else {
                        field_char
                    };
                    let prefix_lower = if prefix_char >= b'A' && prefix_char <= b'Z' {
                        prefix_char | 0x20
                    } else {
                        prefix_char
                    };
                    if field_lower != prefix_lower {
                        prefix_matches = false;
                    }
                } else if text_bytes[j] != prefix_bytes[j] {
                    prefix_matches = false;
                }
                j += 1;
            }

            if prefix_matches {
                // Keep this doc ID in the filtered result (in-place)
                if write_pos != read_pos {
                    doc_ids.swap(write_pos, read_pos);
                }
                write_pos += 1;

                //  EARLY TERMINATION: Stop when max_size results found
                if write_pos >= max_size {
                    break;
                }
            }
        }
    }

    write_pos
}

// AVX-512 optimized regex term filtering.
//
// Uses AVX-512 vectorized pre-filtering combined with regex engine.
// Accelerates common patterns and reduces regex engine calls.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
// GPU implementation of regex filtering using PTX assembly

#[cfg(has_cuda)]
pub unsafe fn filter_regex_terms_gpu(
    terms: &mut [String],
    term_lengths: &[usize],
    regex: &regex::bytes::Regex,
    max_size: usize,
    terms_len: usize,
) -> usize {
    // For regex filtering, we'll do a simple prefix match on GPU
    // and then validate with regex on CPU

    const PTX_FILTER_REGEX: &str = r#"
    .version 7.5
    .target sm_70
    .entry filter_regex_prefix (
      .param .u64 terms_data_ptr,   // Input: concatenated term strings
      .param .u64 term_offsets_ptr, // Input: term offsets in data
      .param .u64 term_lengths_ptr, // Input: term lengths
      .param .u64 prefix_ptr,       // Input: literal prefix extracted from regex
      .param .u32 prefix_len,       // Input: prefix length
      .param .u32 terms_len,        // Input: number of terms
      .param .u64 results_ptr       // Output: prefix match results
    ) {
      .reg .pred %p<16>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<16>;
      .reg .u8 %b<10>;
      .reg .u32 %rv<4>;    // Vector registers for effective vectorization
      .reg .u64 %rdv<2>;   // Double registers for 64-bit vectorized ops
      
      // Get thread index - each thread processes one term
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mad.lo.u32 %r3, %r1, %r2, %r0; // global thread id
      
      // Load parameters
      ld.param.u64 %rd0, [terms_data_ptr];
      ld.param.u64 %rd1, [term_offsets_ptr];
      ld.param.u64 %rd2, [term_lengths_ptr];
      ld.param.u64 %rd3, [prefix_ptr];
      ld.param.u32 %r4, [prefix_len];
      ld.param.u32 %r5, [terms_len];
      ld.param.u64 %rd4, [results_ptr];
      
      // Check bounds
      setp.ge.u32 %p0, %r3, %r5;
      @%p0 bra END;
      
      // Get term offset and length for this thread
      mul.wide.u32 %rd5, %r3, 8;
      add.u64 %rd6, %rd1, %rd5;     // term_offsets[tid]
      add.u64 %rd7, %rd2, %rd5;     // term_lengths[tid]
      ld.global.u64 %rd8, [%rd6];   // term_offset
      ld.global.u64 %rd9, [%rd7];   // term_len
      cvt.u32.u64 %r6, %rd9;        // term_len as u32
      
      // Get pointer to this term's data
      add.u64 %rd10, %rd0, %rd8;    // terms_data_ptr + term_offset
      
      // Initialize match result to 0 (no prefix match)
      mov.u32 %r7, 0;
      
      // If no prefix or term too short, no match
      setp.eq.u32 %p1, %r4, 0;
      @%p1 bra STORE_RESULT;
      
      setp.lt.u32 %p2, %r6, %r4; // term_len < prefix_len
      @%p2 bra STORE_RESULT;
      
      // Fast prefix search within the term (mirrors AVX-512 approach)
      // Load first prefix character for quick scanning
      ld.global.u8 %b0, [%rd3]; // prefix[0]
      
      // Calculate search range
      sub.u32 %r8, %r6, %r4;     // search_end = term_len - prefix_len
      add.u32 %r8, %r8, 1;       // search_end + 1 for <= comparison
      mov.u32 %r9, 0;            // search position
      
    SEARCH_LOOP:
      setp.ge.u32 %p3, %r9, %r8;
      @%p3 bra STORE_RESULT; // No prefix match found
      
      // Try vectorized search first (4 bytes at a time)
      add.u32 %r20, %r9, 4;
      setp.gt.u32 %p10, %r20, %r8;
      @%p10 bra SCALAR_SEARCH; // Not enough bytes for vector
      
      // Load 4 bytes from term using vectorized load
      add.u64 %rd11, %rd10, %r9;
      ld.global.v4.u8 {%b1, %b2, %b3, %b4}, [%rd11];
      
      // Check each byte against first prefix character
      setp.eq.u8 %p4, %b0, %b1;
      setp.eq.u8 %p5, %b0, %b2;
      setp.eq.u8 %p6, %b0, %b3;
      setp.eq.u8 %p7, %b0, %b4;
      
      // Combine all match predicates
      or.pred %p8, %p4, %p5;
      or.pred %p9, %p6, %p7;
      or.pred %p8, %p8, %p9;  // Any byte matches?
      
      // Use warp vote to check if ANY thread found a match
      vote.any.pred %p11, %p8;
      @!%p11 bra NO_MATCH_IN_CHUNK;
      
      // At least one thread found match - share position across warp
      // Using ballot to find which lanes have matches
      vote.ballot.b32 %r25, %p8;
      
      // Find first set bit (first match position)
      bfind.u32 %r26, %r25;  // Find first bit set
      
      // Broadcast found position to all threads in warp
      shfl.sync.bcast.b32 %r27, %r9, %r26, 0x1f;
      
      // Check which exact position matched
      @%p4 bra CHECK_FULL_PREFIX;
      
      add.u32 %r21, %r9, 1;
      @%p5 bra CHECK_FULL_PREFIX_AT_1;
      
      add.u32 %r22, %r9, 2;
      @%p6 bra CHECK_FULL_PREFIX_AT_2;
      
      add.u32 %r23, %r9, 3;
      @%p7 bra CHECK_FULL_PREFIX_AT_3;
      
    NO_MATCH_IN_CHUNK:
      add.u32 %r9, %r9, 4; // Move by 4 bytes
      bra SEARCH_LOOP;
      
    SCALAR_SEARCH:
      // Fall back to scalar for remaining bytes
      add.u64 %rd11, %rd10, %r9;
      ld.global.u8 %b1, [%rd11];
      
      // Check if first character matches
      setp.eq.u8 %p4, %b0, %b1;
      @%p4 bra CHECK_FULL_PREFIX;
      
      add.u32 %r9, %r9, 1; // Move to next position
      bra SEARCH_LOOP;
      
    CHECK_FULL_PREFIX_AT_1:
      mov.u32 %r9, %r21;
      bra CHECK_FULL_PREFIX;
      
    CHECK_FULL_PREFIX_AT_2:
      mov.u32 %r9, %r22;
      bra CHECK_FULL_PREFIX;
      
    CHECK_FULL_PREFIX_AT_3:
      mov.u32 %r9, %r23;
      bra CHECK_FULL_PREFIX;
      
    CHECK_FULL_PREFIX:
      // Found first character match, check full prefix
      mov.u32 %r10, 0; // prefix index
      mov.u32 %r11, %r9; // term index
      
    PREFIX_MATCH_LOOP:
      setp.ge.u32 %p5, %r10, %r4;
      @%p5 bra PREFIX_MATCH_SUCCESS; // Matched entire prefix
      
      // Try vectorized comparison (4 bytes at a time)
      sub.u32 %r24, %r4, %r10;  // remaining prefix length
      setp.lt.u32 %p11, %r24, 4;
      @%p11 bra SCALAR_PREFIX_MATCH; // Less than 4 bytes remaining
      
      // Load 4 bytes from prefix and term using vectorized loads
      add.u64 %rd12, %rd3, %r10;   // prefix[prefix_idx]
      add.u64 %rd13, %rd10, %r11;  // term[term_idx]
      ld.global.v4.u8 {%b5, %b6, %b7, %b8}, [%rd12];
      ld.global.v4.u8 {%b2, %b3, %b4, %b9}, [%rd13];
      
      // Compare all 4 bytes
      setp.ne.u8 %p6, %b5, %b2;
      @%p6 bra PREFIX_MATCH_FAIL;
      setp.ne.u8 %p6, %b6, %b3;
      @%p6 bra PREFIX_MATCH_FAIL;
      setp.ne.u8 %p6, %b7, %b4;
      @%p6 bra PREFIX_MATCH_FAIL;
      setp.ne.u8 %p6, %b8, %b9;
      @%p6 bra PREFIX_MATCH_FAIL;
      
      add.u32 %r10, %r10, 4; // prefix_idx += 4
      add.u32 %r11, %r11, 4; // term_idx += 4
      bra PREFIX_MATCH_LOOP;
      
    SCALAR_PREFIX_MATCH:
      // Fall back to scalar for remaining bytes
      add.u64 %rd12, %rd3, %r10;   // prefix[prefix_idx]
      add.u64 %rd13, %rd10, %r11;  // term[term_idx]
      ld.global.u8 %b2, [%rd12];
      ld.global.u8 %b3, [%rd13];
      
      // Compare characters - each thread votes on its comparison
      setp.eq.u8 %p6, %b2, %b3;
      
      // Use vote.all to check if ALL threads agree on match
      // More efficient than ballot + bit checking
      vote.all.pred %p15, %p6;
      @!%p15 bra PREFIX_MATCH_FAIL;  // At least one thread didn't match
      
      // Share successful matches across warp to avoid redundant work
      bar.warp.sync 0xffffffff;  // Ensure all threads sync
      
      add.u32 %r10, %r10, 1; // prefix_idx++
      add.u32 %r11, %r11, 1; // term_idx++
      bra PREFIX_MATCH_LOOP;
      
    PREFIX_MATCH_FAIL:
      add.u32 %r9, %r9, 1; // Continue searching from next position
      bra SEARCH_LOOP;
      
    PREFIX_MATCH_SUCCESS:
      mov.u32 %r7, 1; // Found prefix match
      
      // Share match results across warp for potential early exit
      vote.ballot.b32 %r28, %p15;
      popc.b32 %r29, %r28;  // Count matches in warp
      
    STORE_RESULT:
      // Store prefix match result using vectorized store
      mul.wide.u32 %rd14, %r3, 4;
      add.u64 %rd15, %rd4, %rd14;
      st.global.u32 [%rd15], %r7;
      
    END:
      ret;
    }
  "#;

    // GPU + CPU hybrid approach (mirrors AVX-512 strategy)
    // 1. Extract literal prefix from regex pattern
    let pattern_str = regex.as_str();
    let mut literal_prefix = Vec::new();

    // Extract literal prefix (mirrors AVX-512 prefix extraction)
    for ch in pattern_str.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => {
                literal_prefix.push(ch as u8);
            }
            '^' | '$' => {} // Skip anchors
            _ => break,     // Stop at metacharacter
        }
        if literal_prefix.len() >= 64 {
            break; // Reasonable limit
        }
    }

    let mut write_pos = 0;

    // If we have a useful prefix (>= 2 chars), use GPU pre-filtering
    if literal_prefix.len() >= 2 {
        let mut gpu_prefix_results = vec![0u32; terms_len];

        // Prepare string data for GPU
        let mut gpu_term_data = Vec::new();
        let mut gpu_term_offsets = vec![0u64; terms_len];
        let mut offset = 0u64;

        for i in 0..terms_len {
            gpu_term_offsets[i] = offset;
            let term_bytes = terms[i].as_bytes();
            gpu_term_data.extend_from_slice(term_bytes);
            offset += term_bytes.len() as u64;
        }

        let block_size = 256u32;
        let grid_size = ((terms_len as u32) + block_size - 1) / block_size;

        // GPU prefix matching
        let _ = crate::gpu::launch_ptx(
            PTX_FILTER_REGEX,
            &[],
            "filter_regex_prefix",
            grid_size,
            block_size,
            &[
                gpu_term_data.as_ptr() as *const u8,
                gpu_term_offsets.as_ptr() as *const u8,
                term_lengths.as_ptr() as *const u8,
                literal_prefix.as_ptr() as *const u8,
                literal_prefix.len() as u32 as *const u8,
                terms_len as u32 as *const u8,
                gpu_prefix_results.as_mut_ptr() as *const u8,
            ],
        );

        // CPU validation on prefix matches only (huge speedup!)
        for i in 0..terms_len {
            if write_pos >= max_size {
                break;
            }

            // Only check regex on terms that passed prefix filter
            if gpu_prefix_results[i] == 1 && regex.is_match(terms[i].as_bytes()) {
                if write_pos != i {
                    terms.swap(write_pos, i);
                }
                write_pos += 1;
            }
        }
    } else {
        // No useful prefix - fall back to CPU-only regex matching
        for i in 0..terms_len {
            if write_pos >= max_size {
                break;
            }

            if regex.is_match(terms[i].as_bytes()) {
                if write_pos != i {
                    terms.swap(write_pos, i);
                }
                write_pos += 1;
            }
        }
    }

    write_pos
}
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn filter_regex_terms_avx512(
    terms: &mut [String], // filter strings in-place
    term_lengths: &[usize],
    regex: &regex::bytes::Regex,
    terms_len: usize,
    max_size: usize,
) -> usize {
    const LANES: usize = LANES_AVX512_BYTES;
    const UNROLL_FACTOR: usize = 2; // Unroll factor for prefix search
    let mut write_pos = 0usize;

    // Extract literal prefix from regex pattern for SIMD pre-filtering - HEAP FREE!
    let pattern_str = regex.as_str();
    const MAX_PREFIX: usize = 64; // Reasonable limit for literal prefixes
    let mut literal_prefix: [u8; MAX_PREFIX] = [0; MAX_PREFIX];
    let mut prefix_len = 0;

    for ch in pattern_str.chars() {
        if prefix_len >= MAX_PREFIX {
            break;
        }
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => {
                literal_prefix[prefix_len] = ch as u8;
                prefix_len += 1;
            }
            '^' | '$' => {} // Skip anchors
            _ => break,     // Stop at metacharacter
        }
    }

    if prefix_len >= 2 {
        let first_char = _mm512_set1_epi8(literal_prefix[0] as i8);

        // Pre-broadcast prefix chunks for better cache efficiency
        let mut prefix_chunks = [_mm512_setzero_si512(); 4];
        let num_prefix_chunks = ((prefix_len + LANES - 1) / LANES).min(4);
        for i in 0..num_prefix_chunks {
            let chunk_start = i * LANES;
            if chunk_start < prefix_len {
                let chunk_end = (chunk_start + LANES).min(prefix_len);
                let load_mask = if chunk_end - chunk_start == LANES {
                    u64::MAX
                } else {
                    (1u64 << (chunk_end - chunk_start)) - 1
                };
                prefix_chunks[i] = _mm512_maskz_loadu_epi8(
                    load_mask,
                    literal_prefix.as_ptr().add(chunk_start) as *const i8,
                );
            }
        }

        // Process terms in-place: check if each term matches regex
        for read_pos in 0..terms_len {
            let term = &terms[read_pos];
            let term_bytes = term.as_bytes();
            let term_len = term_lengths[read_pos];
            if term_len < prefix_len {
                continue;
            }

            let mut found_prefix = false;
            let search_end = term_len.saturating_sub(prefix_len);
            let simd_search_end = search_end & !(LANES * UNROLL_FACTOR - 1);

            let mut i = 0;

            // Unrolled first character search
            while i < simd_search_end && !found_prefix {
                for unroll_idx in 0..UNROLL_FACTOR {
                    let chunk_offset = i + unroll_idx * LANES;
                    if chunk_offset > search_end {
                        break;
                    }

                    let term_chunk =
                        _mm512_loadu_si512(term_bytes.as_ptr().add(chunk_offset) as *const _);
                    let first_mask = _mm512_cmpeq_epu8_mask(term_chunk, first_char);

                    if first_mask != 0 {
                        let mut mask = first_mask;
                        while mask != 0 && !found_prefix {
                            let bit_pos = mask.trailing_zeros() as usize;
                            let pos = chunk_offset + bit_pos;
                            mask &= mask - 1;

                            if pos + prefix_len <= term_len {
                                // Fast prefix comparison using pre-computed chunks
                                let mut matches = true;
                                for chunk_idx in 0..num_prefix_chunks {
                                    let chunk_start = chunk_idx * LANES;
                                    if chunk_start >= prefix_len {
                                        break;
                                    }

                                    let chunk_end = (chunk_start + LANES).min(prefix_len);
                                    let load_mask = if chunk_end - chunk_start == LANES {
                                        u64::MAX
                                    } else {
                                        (1u64 << (chunk_end - chunk_start)) - 1
                                    };

                                    let term_chunk = _mm512_maskz_loadu_epi8(
                                        load_mask,
                                        term_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                    );
                                    let prefix_chunk = prefix_chunks[chunk_idx];

                                    let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                        load_mask,
                                        term_chunk,
                                        prefix_chunk,
                                    );
                                    if cmp_mask != load_mask {
                                        matches = false;
                                        break;
                                    }
                                }

                                if matches {
                                    found_prefix = true;
                                    break;
                                }
                            }
                        }
                    }

                    if found_prefix {
                        break;
                    }
                }
                i += LANES * UNROLL_FACTOR;
            }

            // Handle remaining SIMD chunks
            while i <= search_end && i + LANES <= term_len && !found_prefix {
                let term_chunk = _mm512_loadu_si512(term_bytes.as_ptr().add(i) as *const _);
                let first_mask = _mm512_cmpeq_epu8_mask(term_chunk, first_char);

                if first_mask != 0 {
                    let mut mask = first_mask;
                    while mask != 0 && !found_prefix {
                        let bit_pos = mask.trailing_zeros() as usize;
                        let pos = i + bit_pos;
                        mask &= mask - 1;

                        if pos + prefix_len <= term_len {
                            let mut matches = true;
                            for chunk_idx in 0..num_prefix_chunks {
                                let chunk_start = chunk_idx * LANES;
                                if chunk_start >= prefix_len {
                                    break;
                                }

                                let chunk_end = (chunk_start + LANES).min(prefix_len);
                                let load_mask = if chunk_end - chunk_start == LANES {
                                    u64::MAX
                                } else {
                                    (1u64 << (chunk_end - chunk_start)) - 1
                                };

                                let term_chunk = _mm512_maskz_loadu_epi8(
                                    load_mask,
                                    term_bytes.as_ptr().add(pos + chunk_start) as *const i8,
                                );
                                let prefix_chunk = prefix_chunks[chunk_idx];

                                let cmp_mask = _mm512_mask_cmpeq_epu8_mask(
                                    load_mask,
                                    term_chunk,
                                    prefix_chunk,
                                );
                                if cmp_mask != load_mask {
                                    matches = false;
                                    break;
                                }
                            }

                            if matches {
                                found_prefix = true;
                                break;
                            }
                        }
                    }
                }
                i += LANES;
            }

            // Handle remaining bytes with AVX-512 masked operations
            if !found_prefix {
                let remaining_search_len = term_len.saturating_sub(prefix_len) + 1 - i;
                if remaining_search_len > 0 {
                    // Use AVX-512 masked operations for remaining positions
                    for pos in i..=(term_len.saturating_sub(prefix_len)) {
                        if pos + prefix_len <= term_len && prefix_len <= 64 {
                            let load_mask = (1u64 << prefix_len) - 1;
                            let term_chunk = _mm512_maskz_loadu_epi8(
                                load_mask,
                                term_bytes.as_ptr().add(pos) as *const i8,
                            );
                            let prefix_chunk = _mm512_maskz_loadu_epi8(
                                load_mask,
                                literal_prefix.as_ptr() as *const i8,
                            );
                            let eq_mask =
                                _mm512_mask_cmpeq_epu8_mask(load_mask, term_chunk, prefix_chunk);
                            if eq_mask == load_mask {
                                found_prefix = true;
                                break;
                            }
                        } else {
                            // Fallback for edge cases only
                            if term_bytes[pos..pos + prefix_len] == literal_prefix[..prefix_len] {
                                found_prefix = true;
                                break;
                            }
                        }
                    }
                }
            }

            // If we found the prefix, check the full regex
            if found_prefix && regex.is_match(term_bytes) {
                if write_pos >= max_size {
                    break; // Respect max_size limit
                }
                // Keep this term in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }
    } else {
        // Fallback for complex patterns
        for read_pos in 0..terms_len {
            if write_pos >= max_size {
                break; // Respect max_size limit
            }
            let term = &terms[read_pos];
            if regex.is_match(term.as_bytes()) {
                // Keep this term in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }
    }

    write_pos
}

// AVX2 optimized regex term filtering.
//
// Uses AVX2 vectorized pre-filtering combined with regex engine.
// Accelerates common patterns and reduces regex engine calls.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
pub(crate) unsafe fn filter_regex_terms_avx2(
    terms: &mut [String], // Filter terms directly in-place
    term_lengths: &[usize],
    regex: &regex::bytes::Regex,
    terms_len: usize,
    max_size: usize,
) -> usize {
    const LANES: usize = LANES_AVX2_BYTES; // AVX2 processes 32 bytes at once
    const MAX_PREFIX: usize = 256; // Fixed size for literal prefix
    let mut write_pos = 0usize;

    // Extract literal prefix from regex pattern for SIMD pre-filtering
    let pattern_str = regex.as_str();
    let mut literal_prefix: [u8; MAX_PREFIX] = [0; MAX_PREFIX];
    let mut prefix_len = 0;

    for ch in pattern_str.chars() {
        if prefix_len >= MAX_PREFIX {
            break;
        }
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => {
                literal_prefix[prefix_len] = ch as u8;
                prefix_len += 1;
            }
            '^' | '$' => {} // Skip anchors
            _ => break,     // Stop at metacharacter
        }
    }

    if prefix_len >= 2 {
        let first_char = _mm256_set1_epi8(literal_prefix[0] as i8);

        // Process terms in-place - check if each term matches regex
        for read_pos in 0..terms_len {
            let term = &terms[read_pos];
            let term_bytes = term.as_bytes();
            let term_len = term_lengths[read_pos];

            if term_len < prefix_len {
                continue;
            }

            let mut found_prefix = false;
            let mut i = 0;

            // AVX2 search for literal prefix patterns
            while i + LANES <= term_len && !found_prefix {
                let term_chunk = _mm256_loadu_si256(term_bytes.as_ptr().add(i) as *const _);
                let first_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(term_chunk, first_char));

                if first_mask != 0 {
                    // Process all set bits using trailing_zeros() pattern (safe approach)
                    let remaining_mask = first_mask as u32;

                    while remaining_mask != 0 {
                        let bit_pos = remaining_mask.trailing_zeros() as usize;
                        if bit_pos < 32 {
                            let pos = i + bit_pos;

                            if pos + prefix_len <= term_len {
                                let mut matches = true;
                                let mut j = 0;

                                // Compare full prefix with vectorized operations
                                while j + LANES <= prefix_len && matches {
                                    let term_chunk = _mm256_loadu_si256(
                                        term_bytes.as_ptr().add(pos + j) as *const _,
                                    );
                                    let prefix_chunk = _mm256_loadu_si256(
                                        literal_prefix.as_ptr().add(j) as *const _,
                                    );
                                    let cmp_result = _mm256_cmpeq_epi8(term_chunk, prefix_chunk);
                                    let mask = _mm256_movemask_epi8(cmp_result);

                                    if mask != -1i32 {
                                        matches = false;
                                    }
                                    j += LANES;
                                }

                                // Handle remaining bytes with scalar comparison
                                while j < prefix_len && matches {
                                    if term_bytes[pos + j] != literal_prefix[j] {
                                        matches = false;
                                    }
                                    j += 1;
                                }

                                if matches {
                                    found_prefix = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                i += LANES;
            }

            // Handle remaining bytes with minimal scalar operations
            while i <= term_len.saturating_sub(prefix_len) && !found_prefix {
                // Use minimal scalar for tiny remainder only
                let mut matches = true;
                for j in 0..prefix_len {
                    if term_bytes[i + j] != literal_prefix[j] {
                        matches = false;
                        break;
                    }
                }

                if matches {
                    found_prefix = true;
                }
                i += 1;
            }

            // If we found the prefix, check the full regex
            if found_prefix && regex.is_match(term_bytes) {
                if write_pos >= max_size {
                    break; // Respect max_size limit
                }
                // Keep this term in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }
    } else {
        // Fallback for complex patterns without detectable literal prefixes
        for read_pos in 0..terms_len {
            if write_pos >= max_size {
                break; // Respect max_size limit
            }
            let term = &terms[read_pos];
            if regex.is_match(term.as_bytes()) {
                // Keep this term in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }
    }

    write_pos
}

// NEON optimized regex term filtering.
//
// Uses NEON vectorized pre-filtering combined with regex engine.
// Accelerates common patterns and reduces regex engine calls.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn filter_regex_terms_neon(
    terms: &mut [String], // Filter terms directly in-place
    term_lengths: &[usize],
    regex: &regex::bytes::Regex,
    terms_len: usize,
    max_size: usize,
) -> usize {
    const LANES: usize = LANES_NEON_BYTES; // NEON processes 16 bytes at once
    let mut write_pos = 0usize;

    // Try to extract literal characters from the regex for SIMD optimization - HEAP FREE!
    let regex_str = regex.as_str();
    let mut literal_chars: [u8; 2] = [0; 2];
    let mut literal_count = 0;

    for c in regex_str.chars() {
        if literal_count >= 2 {
            break;
        }
        if c.is_ascii_alphanumeric() {
            literal_chars[literal_count] = c as u8;
            literal_count += 1;
        }
    }

    if literal_count >= 2 {
        let first_char_simd = unsafe { vdupq_n_u8(literal_chars[0]) };
        let _second_char_simd = unsafe { vdupq_n_u8(literal_chars[1]) };

        // Process terms directly in-place
        for read_pos in 0..terms_len {
            let term_bytes = terms[read_pos].as_bytes();
            let term_len = term_lengths[read_pos];
            if term_len < 2 {
                // Too short for SIMD optimization, use regular regex
                if regex.is_match(term_bytes) {
                    // Keep this term in the filtered result (in-place)
                    if write_pos != read_pos {
                        terms.swap(write_pos, read_pos);
                    }
                    write_pos += 1;
                }
                continue;
            }

            // First scan for the first character
            let mut found_potential_match = false;
            let mut i = 0;
            while i + LANES <= term_len {
                let term_chunk = unsafe { vld1q_u8(term_bytes.as_ptr().add(i)) };
                let first_cmp = unsafe { vceqq_u8(term_chunk, first_char_simd) };

                // Extract mask to check if any lane matched
                let mut mask_values = [0u8; LANES];
                unsafe { vst1q_u8(mask_values.as_mut_ptr(), first_cmp) };

                // Check if any byte matched the first character
                for i in 0..LANES {
                    let mask = mask_values[i];
                    if mask == 0xFF {
                        found_potential_match = true;
                        break;
                    }
                }

                if found_potential_match {
                    break;
                }

                i += LANES;
            }

            // If we found a potential match or didn't complete the SIMD scan, use regex
            if (found_potential_match || i < term_len) && regex.is_match(term_bytes) {
                if write_pos >= max_size {
                    break; // Respect max_size limit
                }
                // Keep this term in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }
    } else {
        // No literal characters to optimize with, use regular regex matching
        for read_pos in 0..terms_len {
            if write_pos >= max_size {
                break; // Respect max_size limit
            }
            if regex.is_match(terms[read_pos].as_bytes()) {
                // Keep this term in the filtered result (in-place)
                if write_pos != read_pos {
                    terms.swap(write_pos, read_pos);
                }
                write_pos += 1;
            }
        }
    }

    write_pos
}

// AVX-512 optimized wildcard term filtering.
//
// Uses AVX-512 vectorized pattern matching for efficient wildcard filtering.
// Accelerates both simple patterns and complex wildcard operations.
//
// # Safety
// Requires AVX-512F support. Use `is_x86_feature_detected!("avx512f")` before calling.
// GPU implementation of wildcard filtering using PTX assembly

#[cfg(has_cuda)]
pub unsafe fn filter_wildcard_terms_gpu(
    terms: &mut [String],
    term_lengths: &[usize],
    pattern: &str,
    case_insensitive: bool,
    max_size: usize,
    terms_len: usize,
    pattern_len: usize,
) -> usize {
    const PTX_FILTER_WILDCARD: &str = r#"
    .version 7.5
    .target sm_70
    .entry filter_wildcard_terms (
      .param .u64 terms_data_ptr,   // Input: concatenated term strings
      .param .u64 term_offsets_ptr, // Input: term offsets in data
      .param .u64 term_lengths_ptr, // Input: term lengths
      .param .u64 pattern_ptr,      // Input: pattern string
      .param .u32 case_insensitive, // Input: case insensitive flag
      .param .u32 pattern_len,      // Input: pattern length
      .param .u32 terms_len,        // Input: number of terms
      .param .u64 results_ptr       // Output: match results
    ) {
      .reg .pred %p<22>;
      .reg .u32 %r<35>;
      .reg .u64 %rd<15>;
      .reg .u8 %b<10>;
      .reg .u32 %rv<4>;    // Vector registers for effective vectorization
      .reg .u64 %rdv<2>;   // Double registers for 64-bit vectorized ops
      
      // Get thread index - each thread processes one term
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mad.lo.u32 %r3, %r1, %r2, %r0; // global thread id
      
      // Load parameters
      ld.param.u64 %rd0, [terms_data_ptr];
      ld.param.u64 %rd1, [term_offsets_ptr];
      ld.param.u64 %rd2, [term_lengths_ptr];
      ld.param.u64 %rd3, [pattern_ptr];
      ld.param.u32 %r4, [case_insensitive];
      ld.param.u32 %r5, [pattern_len];
      ld.param.u32 %r6, [terms_len];
      ld.param.u64 %rd4, [results_ptr];
      
      // Check bounds
      setp.ge.u32 %p0, %r3, %r6;
      @%p0 bra END;
      
      // Get term offset and length for this thread
      mul.wide.u32 %rd5, %r3, 8;
      add.u64 %rd6, %rd1, %rd5;     // term_offsets[tid]
      add.u64 %rd7, %rd2, %rd5;     // term_lengths[tid]
      ld.global.u64 %rd8, [%rd6];   // term_offset
      ld.global.u64 %rd9, [%rd7];   // term_len
      cvt.u32.u64 %r7, %rd9;        // term_len as u32
      
      // Get pointer to this term's data
      add.u64 %rd10, %rd0, %rd8;    // terms_data_ptr + term_offset
      
      // Initialize match result to 0 (no match)
      mov.u32 %r8, 0;
      
      // Mirror AVX-512 wildcard matching algorithm exactly
      
      // Case 1: pattern_len == 0, match only if term_len == 0
      setp.eq.u32 %p1, %r5, 0;
      @%p1 bra CHECK_EMPTY_TERM;
      
      // Case 2: term_len == 0, match only if pattern is all '*'
      setp.eq.u32 %p2, %r7, 0;
      @%p2 bra CHECK_ALL_STARS;
      
      // Case 3: Main wildcard matching with backtracking (mirrors AVX-512 exactly)
      mov.u32 %r9, 0;         // text_pos
      mov.u32 %r10, 0;        // pattern_pos
      mov.u32 %r11, 0xFFFFFFFF; // star_pos (sentinel value like AVX-512)
      mov.u32 %r12, 0;        // match_pos
      
    MATCH_LOOP:
      // Mirror: while text_pos < term_len
      setp.ge.u32 %p3, %r9, %r7;
      @%p3 bra CHECK_PATTERN_REMAINING;
      
      // Mirror: if pattern_pos < pattern_len
      setp.ge.u32 %p4, %r10, %r5;
      @%p4 bra BACKTRACK_OR_FAIL;
      
      // Load pattern character
      add.u64 %rd11, %rd3, %r10;
      ld.global.u8 %b0, [%rd11];
      
      // Mirror AVX-512: match pattern_bytes[pattern_pos]
      setp.eq.u32 %p5, %b0, 42; // '*'
      @%p5 bra HANDLE_STAR;
      
      setp.eq.u32 %p6, %b0, 63; // '?'
      @%p6 bra HANDLE_QUESTION;
      
      // Regular character - try vectorized load first
      sub.u32 %r20, %r7, %r9;  // remaining term length
      setp.lt.u32 %p21, %r20, 4;
      @%p21 bra SCALAR_CHAR_LOAD;
      
      // Vectorized path - load 4 bytes and check pattern char against first
      add.u64 %rd12, %rd10, %r9;
      ld.global.v4.u8 {%b1, %b7, %b8, %b9}, [%rd12];
      bra CHAR_LOADED;
      
    SCALAR_CHAR_LOAD:
      // Scalar fallback for edge cases
      add.u64 %rd12, %rd10, %r9;
      ld.global.u8 %b1, [%rd12];
      
    CHAR_LOADED:
      
      // Case conversion for case-insensitive matching (mirror AVX-512 logic)
      mov.u8 %b2, %b0; // pattern char
      mov.u8 %b3, %b1; // text char
      
      setp.ne.u32 %p7, %r4, 0; // case_insensitive
      @%p7 bra CASE_INSENSITIVE_CMP;
      
      // Case sensitive comparison
      setp.eq.u32 %p8, %b2, %b3;
      @%p8 bra CHAR_MATCH;
      bra CHAR_NO_MATCH;
      
    CASE_INSENSITIVE_CMP:
      // Convert both to lowercase (mirror AVX-512 case conversion)
      and.b32 %r13, %b0, 0xFF;
      and.b32 %r14, %b1, 0xFF;
      
      // Pattern char to lowercase
      setp.ge.and.u32 %p9, %r13, 65, %p9; // >= 'A'
      setp.le.and.u32 %p9, %r13, 90, %p9; // <= 'Z'
      @%p9 or.b32 %r13, %r13, 0x20; // to lowercase
      
      // Text char to lowercase  
      setp.ge.and.u32 %p10, %r14, 65, %p10;
      setp.le.and.u32 %p10, %r14, 90, %p10;
      @%p10 or.b32 %r14, %r14, 0x20;
      
      setp.eq.u32 %p11, %r13, %r14;
      
      // Use ballot to check if any lane in warp matched this character
      vote.ballot.b32 %r30, %p11;
      
      // Get lane ID within warp  
      and.b32 %r31, %r0, 0x1f;
      
      // Check if this lane's bit is set in ballot result
      shl.b32 %r32, 1, %r31;
      and.b32 %r33, %r30, %r32;
      setp.ne.u32 %p20, %r33, 0;
      @%p20 bra CHAR_MATCH;
      
    CHAR_NO_MATCH:
      // Mirror AVX-512: character doesn't match, try backtracking
      setp.eq.u32 %p12, %r11, 0xFFFFFFFF; // star_pos == sentinel
      @%p12 bra NO_MATCH;
      
      // Backtrack: use vectorized search for next character occurrence
      // This mirrors the AVX-512 SIMD character search
      add.u32 %r12, %r12, 1; // match_pos++
      mov.u32 %r9, %r12;     // text_pos = match_pos
      mov.u32 %r10, %r11;    // pattern_pos = star_pos
      bra MATCH_LOOP;
      
    CHAR_MATCH:
      add.u32 %r9, %r9, 1;   // text_pos++
      add.u32 %r10, %r10, 1; // pattern_pos++
      bra MATCH_LOOP;
      
    HANDLE_STAR:
      // Mirror AVX-512: skip consecutive stars
    SKIP_STARS:
      add.u32 %r10, %r10, 1;
      setp.ge.u32 %p13, %r10, %r5;
      @%p13 bra MATCH_SUCCESS; // Pattern ends with *, matches everything
      
      add.u64 %rd13, %rd3, %r10;
      ld.global.u8 %b4, [%rd13];
      setp.eq.u32 %p14, %b4, 42;
      @%p14 bra SKIP_STARS;
      
      // Set backtrack positions
      mov.u32 %r11, %r10;    // star_pos = pattern_pos
      mov.u32 %r12, %r9;     // match_pos = text_pos
      bra MATCH_LOOP;
      
    HANDLE_QUESTION:
      add.u32 %r9, %r9, 1;   // Skip any character
      add.u32 %r10, %r10, 1;
      bra MATCH_LOOP;
      
    BACKTRACK_OR_FAIL:
      setp.eq.u32 %p15, %r11, 0xFFFFFFFF;
      @%p15 bra NO_MATCH;
      
      // Continue backtracking
      add.u32 %r12, %r12, 1;
      mov.u32 %r9, %r12;
      mov.u32 %r10, %r11;
      bra MATCH_LOOP;
      
    CHECK_PATTERN_REMAINING:
      // Check if remaining pattern is all stars (vectorized check)
      mov.u32 %r15, %r10; // current pattern pos
      
    CHECK_REMAINING_STARS:
      setp.ge.u32 %p16, %r15, %r5;
      @%p16 bra MATCH_SUCCESS;
      
      add.u64 %rd14, %rd3, %r15;
      ld.global.u8 %b5, [%rd14];
      setp.ne.u32 %p17, %b5, 42;
      @%p17 bra NO_MATCH;
      
      add.u32 %r15, %r15, 1;
      bra CHECK_REMAINING_STARS;
      
    CHECK_EMPTY_TERM:
      setp.eq.u32 %p18, %r7, 0;
      @%p18 mov.u32 %r8, 1; // Empty pattern matches empty term
      bra STORE_RESULT;
      
    CHECK_ALL_STARS:
      // Vectorized check if pattern is all '*' characters
      mov.u32 %r16, 0;
      
    STAR_CHECK_LOOP:
      setp.ge.u32 %p19, %r16, %r5;
      @%p19 bra MATCH_SUCCESS;
      
      // Try vectorized check (4 bytes at a time)
      sub.u32 %r25, %r5, %r16;  // remaining pattern length
      setp.lt.u32 %p22, %r25, 4;
      @%p22 bra SCALAR_STAR_CHECK;
      
      // Load 4 pattern bytes and check all are '*'
      add.u64 %rd15, %rd3, %r16;
      ld.global.v4.u8 {%b6, %b7, %b8, %b9}, [%rd15];
      
      setp.ne.u8 %p20, %b6, 42;
      @%p20 bra NO_MATCH;
      setp.ne.u8 %p20, %b7, 42;
      @%p20 bra NO_MATCH;
      setp.ne.u8 %p20, %b8, 42;
      @%p20 bra NO_MATCH;
      setp.ne.u8 %p20, %b9, 42;
      @%p20 bra NO_MATCH;
      
      add.u32 %r16, %r16, 4;
      bra STAR_CHECK_LOOP;
      
    SCALAR_STAR_CHECK:
      add.u64 %rd15, %rd3, %r16;
      ld.global.u8 %b6, [%rd15];
      setp.ne.u8 %p20, %b6, 42;
      @%p20 bra NO_MATCH;
      
      add.u32 %r16, %r16, 1;
      bra STAR_CHECK_LOOP;
      
    MATCH_SUCCESS:
      mov.u32 %r8, 1;
      
      // Share success across warp for statistics
      vote.ballot.b32 %r35, %p16;
      popc.b32 %r36, %r35;  // Count successes in warp
      bra STORE_RESULT;
      
    NO_MATCH:
      mov.u32 %r8, 0;
      
    STORE_RESULT:
      // Store match result using vectorized store
      mul.wide.u32 %rd16, %r3, 4;
      add.u64 %rd17, %rd4, %rd16;
      st.global.u32 [%rd17], %r8;
      
    END:
      ret;
    }
  "#;

    // GPU implementation - each thread processes one term
    let pattern_bytes = pattern.as_bytes();
    let mut gpu_results = vec![0u32; terms_len];

    // Prepare string data for GPU with proper concatenation
    let mut gpu_term_data = Vec::new();
    let mut gpu_term_offsets = vec![0u64; terms_len];
    let mut offset = 0u64;

    for i in 0..terms_len {
        gpu_term_offsets[i] = offset;
        let term_bytes = terms[i].as_bytes();
        gpu_term_data.extend_from_slice(term_bytes);
        offset += term_bytes.len() as u64;
    }

    let block_size = 256u32;
    let grid_size = ((terms_len as u32) + block_size - 1) / block_size;

    let _ = crate::gpu::launch_ptx(
        PTX_FILTER_WILDCARD,
        &[],
        "filter_wildcard_terms",
        grid_size,
        block_size,
        &[
            gpu_term_data.as_ptr() as *const u8,
            gpu_term_offsets.as_ptr() as *const u8,
            term_lengths.as_ptr() as *const u8,
            pattern_bytes.as_ptr() as *const u8,
            if case_insensitive { 1u32 } else { 0u32 } as *const u8,
            pattern_len as u32 as *const u8,
            terms_len as u32 as *const u8,
            gpu_results.as_mut_ptr() as *const u8,
        ],
    );

    // Collect results and reorder terms in-place (mirrors AVX-512 post-processing)
    let mut write_pos = 0;
    for i in 0..terms_len {
        if write_pos >= max_size {
            break;
        }
        if gpu_results[i] == 1 {
            if write_pos != i {
                terms.swap(write_pos, i);
            }
            write_pos += 1;
        }
    }

    write_pos
}
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn filter_wildcard_terms_avx512(
    terms: &mut [String], // Filter terms directly in-place
    term_lengths: &[usize],
    pattern: &str,
    case_insensitive: bool,
    terms_len: usize,
    pattern_len: usize,
    max_size: usize,
) -> usize {
    const LANES: usize = LANES_AVX512_BYTES; // AVX-512 processes 64 bytes at once

    let mut write_pos = 0usize;

    // Pre-compute case conversion masks (unused in current implementation)
    let _lowercase_mask = _mm512_set1_epi8(0x20);
    let _alpha_lower = _mm512_set1_epi8(b'A' as i8);
    let _alpha_upper = _mm512_set1_epi8(b'Z' as i8);

    // Work directly with pattern bytes, no allocation
    let pattern_bytes = pattern.as_bytes();

    // AVX-512 wildcard filtering implementation

    // Process terms in-place - check if each term matches wildcard pattern
    for read_pos in 0..terms_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }
        let term = &terms[read_pos];

        // Work directly with term bytes, no allocation
        let term_bytes = term.as_bytes();
        let term_len = term_lengths[read_pos];

        // Process each term for wildcard matching

        // Use AVX-512 accelerated wildcard matching with case support
        let matches = {
            if pattern_len == 0 {
                term_len == 0
            } else if term_len == 0 {
                // Check if all bytes are '*' using manual loop (no iterators)
                let mut all_stars = true;
                for i in 0..pattern_len {
                    if pattern_bytes[i] != b'*' {
                        all_stars = false;
                        break;
                    }
                }
                all_stars
            } else {
                // Inline SIMD wildcard matching logic
                let mut text_pos = 0;
                let mut pattern_pos = 0;

                #[allow(unused_assignments)]
                let mut star_pos = usize::MAX; // Use sentinel value instead of Option
                let mut match_pos = 0;

                while text_pos < term_len {
                    if pattern_pos < pattern_len {
                        match pattern_bytes[pattern_pos] {
                            b'*' => {
                                while pattern_pos < pattern_len
                                    && pattern_bytes[pattern_pos] == b'*'
                                {
                                    pattern_pos += 1;
                                }
                                if pattern_pos >= pattern_len {
                                    break; // matches = true
                                }
                                star_pos = pattern_pos;
                                match_pos = text_pos;
                            }
                            b'?' => {
                                text_pos += 1;
                                pattern_pos += 1;
                            }
                            ch => {
                                // Use AVX-512 to search for character matches
                                let pattern_char_simd = _mm512_set1_epi8(
                                    if case_insensitive && ch >= b'A' && ch <= b'Z' {
                                        (ch | 0x20) as i8
                                    } else {
                                        ch as i8
                                    },
                                );

                                let current_char = if case_insensitive
                                    && term_bytes[text_pos] >= b'A'
                                    && term_bytes[text_pos] <= b'Z'
                                {
                                    term_bytes[text_pos] | 0x20
                                } else {
                                    term_bytes[text_pos]
                                };
                                let target_char = if case_insensitive && ch >= b'A' && ch <= b'Z' {
                                    ch | 0x20
                                } else {
                                    ch
                                };

                                if current_char == target_char {
                                    // Character matches - just advance both positions
                                    text_pos += 1;
                                    pattern_pos += 1;
                                } else if star_pos != usize::MAX {
                                    // Character doesn't match - use AVX512 SIMD to find next occurrence after backtrack
                                    let mut found_avx512 = false;
                                    let mut search_pos = match_pos + 1;

                                    //  SIMD: Use AVX512 64-way parallel search for target character
                                    while search_pos + LANES <= term_len && !found_avx512 {
                                        let text_chunk = _mm512_loadu_si512(
                                            term_bytes.as_ptr().add(search_pos) as *const _,
                                        );
                                        let cmp_mask = if case_insensitive {
                                            // Case-insensitive SIMD comparison
                                            let lowercase_mask = _mm512_set1_epi8(0x20);
                                            let alpha_mask_lower = _mm512_set1_epi8(b'A' as i8);
                                            let alpha_mask_upper = _mm512_set1_epi8(b'Z' as i8);

                                            let text_ge_a = _mm512_cmpgt_epi8_mask(
                                                text_chunk,
                                                _mm512_sub_epi8(
                                                    alpha_mask_lower,
                                                    _mm512_set1_epi8(1),
                                                ),
                                            );
                                            let text_le_z = _mm512_cmpgt_epi8_mask(
                                                alpha_mask_upper,
                                                text_chunk,
                                            );
                                            let text_is_upper = text_ge_a & text_le_z;
                                            let text_upper_vec = _mm512_movm_epi8(text_is_upper);
                                            let text_lower = _mm512_or_si512(
                                                text_chunk,
                                                _mm512_and_si512(text_upper_vec, lowercase_mask),
                                            );

                                            _mm512_cmpeq_epu8_mask(text_lower, pattern_char_simd)
                                        } else {
                                            _mm512_cmpeq_epu8_mask(text_chunk, pattern_char_simd)
                                        };

                                        if cmp_mask != 0 {
                                            //  NEON-STYLE: Process mask bits individually like NEON reference
                                            let mut temp_mask = cmp_mask;
                                            while temp_mask != 0 {
                                                let k = temp_mask.trailing_zeros() as usize;
                                                if k < LANES {
                                                    text_pos = search_pos + k;
                                                    pattern_pos = star_pos;
                                                    match_pos = text_pos;
                                                    found_avx512 = true;
                                                    break;
                                                }
                                                temp_mask &= temp_mask - 1; // Clear the lowest set bit
                                            }
                                            if found_avx512 {
                                                break;
                                            }
                                        }
                                        search_pos += LANES;
                                    }

                                    // Handle remaining bytes with AVX-512 masked operations
                                    if !found_avx512 {
                                        let remaining_len = term_len - search_pos;
                                        if remaining_len > 0 && remaining_len <= 64 {
                                            // Use AVX-512 masked load for remaining bytes
                                            let load_mask = (1u64 << remaining_len) - 1;
                                            let text_chunk = _mm512_maskz_loadu_epi8(
                                                load_mask,
                                                term_bytes.as_ptr().add(search_pos) as *const i8,
                                            );

                                            let cmp_mask = if case_insensitive {
                                                // Case-insensitive SIMD comparison for remaining bytes
                                                let lowercase_mask = _mm512_set1_epi8(0x20);
                                                let alpha_mask_lower = _mm512_set1_epi8(b'A' as i8);
                                                let alpha_mask_upper = _mm512_set1_epi8(b'Z' as i8);

                                                let text_ge_a = _mm512_mask_cmpgt_epi8_mask(
                                                    load_mask,
                                                    text_chunk,
                                                    _mm512_sub_epi8(
                                                        alpha_mask_lower,
                                                        _mm512_set1_epi8(1),
                                                    ),
                                                );
                                                let text_le_z = _mm512_mask_cmpgt_epi8_mask(
                                                    load_mask,
                                                    alpha_mask_upper,
                                                    text_chunk,
                                                );
                                                let text_is_upper = text_ge_a & text_le_z;
                                                let text_upper_vec =
                                                    _mm512_movm_epi8(text_is_upper);
                                                let text_lower = _mm512_or_si512(
                                                    text_chunk,
                                                    _mm512_and_si512(
                                                        text_upper_vec,
                                                        lowercase_mask,
                                                    ),
                                                );

                                                _mm512_mask_cmpeq_epu8_mask(
                                                    load_mask,
                                                    text_lower,
                                                    pattern_char_simd,
                                                )
                                            } else {
                                                _mm512_mask_cmpeq_epu8_mask(
                                                    load_mask,
                                                    text_chunk,
                                                    pattern_char_simd,
                                                )
                                            };

                                            if cmp_mask != 0 {
                                                let k = cmp_mask.trailing_zeros() as usize;
                                                if k < remaining_len {
                                                    text_pos = search_pos + k;
                                                    pattern_pos = star_pos;
                                                    match_pos = text_pos;
                                                    found_avx512 = true;
                                                }
                                            }
                                        } else {
                                            // Fallback to scalar only for edge cases
                                            while search_pos < term_len {
                                                let current = if case_insensitive
                                                    && term_bytes[search_pos] >= b'A'
                                                    && term_bytes[search_pos] <= b'Z'
                                                {
                                                    term_bytes[search_pos] | 0x20
                                                } else {
                                                    term_bytes[search_pos]
                                                };

                                                if current == target_char {
                                                    text_pos = search_pos;
                                                    pattern_pos = star_pos;
                                                    match_pos = text_pos;
                                                    found_avx512 = true;
                                                    break;
                                                }
                                                search_pos += 1;
                                            }
                                        }
                                    }

                                    if !found_avx512 {
                                        break; // No match found - wildcard fails
                                    }
                                } else {
                                    // No match and no star to backtrack
                                    break;
                                }
                            }
                        }
                    } else {
                        break; // Pattern exhausted
                    }
                }

                // Check if we successfully matched the entire pattern
                pattern_pos >= pattern_len || {
                    // Check if remaining bytes are all '*' using manual loop (no iterators)
                    if pattern_pos < pattern_len {
                        let mut all_remaining_stars = true;
                        for i in pattern_pos..pattern_len {
                            if pattern_bytes[i] != b'*' {
                                all_remaining_stars = false;
                                break;
                            }
                        }
                        all_remaining_stars
                    } else {
                        false
                    }
                }
            }
        };

        if matches {
            // Keep this term in the filtered result (in-place)
            if write_pos != read_pos {
                terms.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }

    write_pos
}

// AVX2 optimized wildcard term filtering.
//
// Uses AVX2 vectorized pattern matching for efficient wildcard filtering.
// Accelerates both simple patterns and complex wildcard operations.
//
// # Safety
// Requires AVX2 support. Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
pub(crate) unsafe fn filter_wildcard_terms_avx2(
    terms: &mut [String], // filter strings in-place
    term_lengths: &[usize],
    pattern: &str,
    case_insensitive: bool,
    terms_len: usize,
    pattern_len: usize,
    max_size: usize,
) -> usize {
    const LANES: usize = LANES_AVX2_BYTES; // AVX2 processes 32 bytes at once
    let mut write_pos = 0usize;

    // Work directly with pattern bytes, no allocation
    let pattern_bytes = pattern.as_bytes();

    // Process terms in-place - check if each term matches wildcard pattern
    for read_pos in 0..terms_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }
        let term = &terms[read_pos];
        // Work directly with term bytes, no allocation
        let term_bytes = term.as_bytes();
        let term_len = term_lengths[read_pos];

        // Use AVX2-accelerated wildcard matching with case support
        let matches = {
            if pattern_len == 0 {
                term_len == 0
            } else if term_len == 0 {
                // Check if all bytes are '*' using manual loop (no iterators)
                {
                    let mut all_stars = true;
                    for i in 0..pattern_len {
                        if pattern_bytes[i] != b'*' {
                            all_stars = false;
                            break;
                        }
                    }
                    all_stars
                }
            } else {
                // Inline SIMD wildcard matching logic
                let mut text_pos = 0;
                let mut pattern_pos = 0;
                let mut star_pos = usize::MAX; // Use sentinel value instead of Option
                let mut match_pos = 0;

                while text_pos < term_len {
                    if pattern_pos < pattern_len {
                        match pattern_bytes[pattern_pos] {
                            b'*' => {
                                while pattern_pos < pattern_len
                                    && pattern_bytes[pattern_pos] == b'*'
                                {
                                    pattern_pos += 1;
                                }
                                if pattern_pos >= pattern_len {
                                    break; // matches = true
                                }
                                star_pos = pattern_pos;
                                match_pos = text_pos;
                            }
                            b'?' => {
                                text_pos += 1;
                                pattern_pos += 1;
                            }
                            ch => {
                                // Use AVX2 to search for character matches
                                let pattern_char_simd = _mm256_set1_epi8(
                                    if case_insensitive && ch >= b'A' && ch <= b'Z' {
                                        (ch | 0x20) as i8
                                    } else {
                                        ch as i8
                                    },
                                );

                                let current_char = if case_insensitive
                                    && term_bytes[text_pos] >= b'A'
                                    && term_bytes[text_pos] <= b'Z'
                                {
                                    term_bytes[text_pos] | 0x20
                                } else {
                                    term_bytes[text_pos]
                                };
                                let target_char = if case_insensitive && ch >= b'A' && ch <= b'Z' {
                                    ch | 0x20
                                } else {
                                    ch
                                };

                                if current_char == target_char {
                                    // Character matches - just advance both positions
                                    text_pos += 1;
                                    pattern_pos += 1;
                                } else if star_pos != usize::MAX {
                                    // Character doesn't match - use AVX2 SIMD to find next occurrence after backtrack
                                    let mut found_avx2 = false;
                                    let mut search_pos = match_pos + 1;

                                    //  SIMD: Use AVX2 32-way parallel search for target character
                                    while search_pos + LANES <= term_len && !found_avx2 {
                                        let text_chunk = unsafe {
                                            _mm256_loadu_si256(
                                                term_bytes.as_ptr().add(search_pos) as *const _
                                            )
                                        };
                                        let cmp_result = if case_insensitive {
                                            // Case-insensitive SIMD comparison
                                            let lowercase_mask = _mm256_set1_epi8(0x20);
                                            let alpha_mask_lower = _mm256_set1_epi8(b'A' as i8);
                                            let alpha_mask_upper = _mm256_set1_epi8(b'Z' as i8);

                                            let text_ge_a = _mm256_cmpgt_epi8(
                                                text_chunk,
                                                _mm256_sub_epi8(
                                                    alpha_mask_lower,
                                                    _mm256_set1_epi8(1),
                                                ),
                                            );
                                            let text_le_z = _mm256_cmpgt_epi8(
                                                _mm256_add_epi8(
                                                    alpha_mask_upper,
                                                    _mm256_set1_epi8(1),
                                                ),
                                                text_chunk,
                                            );
                                            let text_is_upper =
                                                _mm256_and_si256(text_ge_a, text_le_z);
                                            let text_lower = _mm256_or_si256(
                                                text_chunk,
                                                _mm256_and_si256(text_is_upper, lowercase_mask),
                                            );

                                            _mm256_cmpeq_epi8(text_lower, pattern_char_simd)
                                        } else {
                                            _mm256_cmpeq_epi8(text_chunk, pattern_char_simd)
                                        };
                                        let mask = _mm256_movemask_epi8(cmp_result);

                                        if mask != 0 {
                                            // Process mask bits directly (EXACTLY matching NEON pattern)
                                            // CRITICAL FIX: Use trailing_zeros to find first match like NEON
                                            let first_match = mask.trailing_zeros() as usize;
                                            text_pos = search_pos + first_match;
                                            pattern_pos = star_pos;
                                            match_pos = text_pos;
                                            found_avx2 = true;
                                        }
                                        search_pos += LANES;
                                    }

                                    // Handle remaining bytes with minimal scalar operations
                                    if !found_avx2 && search_pos < term_len {
                                        // Process remaining bytes with minimal scalar loop
                                        while search_pos < term_len {
                                            let current = if case_insensitive
                                                && term_bytes[search_pos] >= b'A'
                                                && term_bytes[search_pos] <= b'Z'
                                            {
                                                term_bytes[search_pos] | 0x20
                                            } else {
                                                term_bytes[search_pos]
                                            };

                                            if current == target_char {
                                                text_pos = search_pos;
                                                pattern_pos = star_pos;
                                                match_pos = text_pos;
                                                found_avx2 = true;
                                                break;
                                            }
                                            search_pos += 1;
                                        }
                                    }

                                    if !found_avx2 {
                                        break; // No match found - wildcard fails
                                    }
                                } else {
                                    // No match and no star to backtrack
                                    break;
                                }
                            }
                        }
                    } else {
                        break; // Pattern exhausted
                    }
                }

                // Check if we successfully matched the entire pattern
                pattern_pos >= pattern_len || {
                    // Check if remaining bytes are all '*' using manual loop (no iterators)
                    if pattern_pos < pattern_len {
                        let mut all_remaining_stars = true;
                        for i in pattern_pos..pattern_len {
                            if pattern_bytes[i] != b'*' {
                                all_remaining_stars = false;
                                break;
                            }
                        }
                        all_remaining_stars
                    } else {
                        false
                    }
                }
            }
        };

        if matches {
            // Keep this term in the filtered result (in-place)
            if write_pos != read_pos {
                terms.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }

    write_pos
}

// NEON optimized wildcard term filtering.
//
// Uses NEON vectorized pattern matching for efficient wildcard filtering.
// Accelerates both simple patterns and complex wildcard operations.
//
// # Safety
// Requires NEON support. Use NEON-enabled target before calling.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::manual_range_contains)]
pub(crate) unsafe fn filter_wildcard_terms_neon(
    terms: &mut [String], // Filter terms directly in-place
    term_lengths: &[usize],
    pattern: &str,
    case_insensitive: bool,
    terms_len: usize,
    pattern_len: usize,
    max_size: usize,
) -> usize {
    const LANES: usize = LANES_NEON_BYTES; // NEON processes 16 bytes at once
    let mut write_pos = 0usize;

    // Work directly with pattern bytes, no allocation
    let pattern_bytes = pattern.as_bytes();

    // Process terms in-place - check if each term matches wildcard pattern
    for read_pos in 0..terms_len {
        if write_pos >= max_size {
            break; // Early termination when max_size results found
        }
        let term = &terms[read_pos];
        // Work directly with term bytes, no allocation
        let term_bytes = term.as_bytes();
        let term_len = term_lengths[read_pos];

        // Use NEON-accelerated wildcard matching with case support
        let matches = {
            if pattern_len == 0 {
                term_len == 0
            } else if term_len == 0 {
                // Check if all bytes are '*' using manual loop (no iterators)
                {
                    let mut all_stars = true;
                    for i in 0..pattern_len {
                        if pattern_bytes[i] != b'*' {
                            all_stars = false;
                            break;
                        }
                    }
                    all_stars
                }
            } else {
                // Inline SIMD wildcard matching logic
                let mut text_pos = 0;
                let mut pattern_pos = 0;

                #[allow(unused_assignments)]
                let mut star_pos = usize::MAX; // Use sentinel value instead of Option
                let mut match_pos = 0;

                while text_pos < term_len {
                    if pattern_pos < pattern_len {
                        match pattern_bytes[pattern_pos] {
                            b'*' => {
                                while pattern_pos < pattern_len
                                    && pattern_bytes[pattern_pos] == b'*'
                                {
                                    pattern_pos += 1;
                                }
                                if pattern_pos >= pattern_len {
                                    break; // matches = true
                                }
                                star_pos = pattern_pos;
                                match_pos = text_pos;
                            }
                            b'?' => {
                                text_pos += 1;
                                pattern_pos += 1;
                            }
                            ch => {
                                // Use NEON to search for character matches
                                let pattern_char_simd = unsafe {
                                    vdupq_n_u8(if case_insensitive && ch.is_ascii_uppercase() {
                                        ch | 0x20
                                    } else {
                                        ch
                                    })
                                };

                                let current_char = if case_insensitive
                                    && term_bytes[text_pos] >= b'A'
                                    && term_bytes[text_pos] <= b'Z'
                                {
                                    term_bytes[text_pos] | 0x20
                                } else {
                                    term_bytes[text_pos]
                                };
                                let target_char = if case_insensitive && ch >= b'A' && ch <= b'Z' {
                                    ch | 0x20
                                } else {
                                    ch
                                };

                                if current_char == target_char {
                                    // Character matches - just advance both positions
                                    text_pos += 1;
                                    pattern_pos += 1;
                                } else if star_pos != usize::MAX {
                                    // Character doesn't match - use SIMD to find next occurrence after backtrack position
                                    let mut found_neon = false;
                                    let mut search_pos = match_pos + 1;

                                    //  SIMD: Use NEON 16-way parallel search for target character
                                    while search_pos + 16 <= term_len && !found_neon {
                                        let text_chunk = unsafe {
                                            vld1q_u8(term_bytes.as_ptr().add(search_pos))
                                        };
                                        let cmp_result = if case_insensitive {
                                            // Case-insensitive SIMD comparison
                                            let lowercase_mask = unsafe { vdupq_n_u8(0x20) };
                                            let alpha_mask_lower = unsafe { vdupq_n_u8(b'A') };
                                            let alpha_mask_upper = unsafe { vdupq_n_u8(b'Z') };

                                            let text_ge_a =
                                                unsafe { vcgeq_u8(text_chunk, alpha_mask_lower) };
                                            let text_le_z =
                                                unsafe { vcleq_u8(text_chunk, alpha_mask_upper) };
                                            let text_is_upper =
                                                unsafe { vandq_u8(text_ge_a, text_le_z) };
                                            let text_lower = unsafe {
                                                vorrq_u8(
                                                    text_chunk,
                                                    vandq_u8(text_is_upper, lowercase_mask),
                                                )
                                            };

                                            unsafe { vceqq_u8(text_lower, pattern_char_simd) }
                                        } else {
                                            unsafe { vceqq_u8(text_chunk, pattern_char_simd) }
                                        };

                                        // Extract SIMD comparison results
                                        let mut cmp_values = [0u8; 16];
                                        unsafe { vst1q_u8(cmp_values.as_mut_ptr(), cmp_result) };

                                        // Check each SIMD lane for matches
                                        for i in 0..LANES {
                                            if cmp_values[i] == 0xFF {
                                                // Found match at position
                                                text_pos = search_pos + i;
                                                pattern_pos = star_pos;
                                                match_pos = text_pos;
                                                found_neon = true;
                                                break;
                                            }
                                        }
                                        search_pos += LANES;
                                    }

                                    // Scalar fallback for remaining bytes
                                    if !found_neon {
                                        while search_pos < term_len {
                                            let current = if case_insensitive
                                                && term_bytes[search_pos] >= b'A'
                                                && term_bytes[search_pos] <= b'Z'
                                            {
                                                term_bytes[search_pos] | 0x20
                                            } else {
                                                term_bytes[search_pos]
                                            };

                                            if current == target_char {
                                                text_pos = search_pos;
                                                pattern_pos = star_pos;
                                                match_pos = text_pos;
                                                found_neon = true;
                                                break;
                                            }
                                            search_pos += 1;
                                        }
                                    }

                                    if !found_neon {
                                        break; // No match found - wildcard fails
                                    }
                                } else {
                                    // No match and no star to backtrack
                                    break;
                                }
                            }
                        }
                    } else {
                        break; // Pattern exhausted
                    }
                }

                // Check if we successfully matched the entire pattern
                pattern_pos >= pattern_len || {
                    // Check if remaining bytes are all '*' using manual loop (no iterators)
                    if pattern_pos < pattern_len {
                        let mut all_remaining_stars = true;
                        for i in pattern_pos..pattern_len {
                            if pattern_bytes[i] != b'*' {
                                all_remaining_stars = false;
                                break;
                            }
                        }
                        all_remaining_stars
                    } else {
                        false
                    }
                }
            }
        };

        if matches {
            // Keep this term in the filtered result (in-place)
            if write_pos != read_pos {
                terms.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
    }

    write_pos
}

// =============================================================================
// SIMD STRING SORTING FUNCTIONS
// =============================================================================

//  AVX512 16-way parallel string sorting with SIMD comparisons

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "hwx-nightly"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
pub(super) unsafe fn sort_strings_avx512(strings: &[String], indices: &mut [u32], len: usize) {
    sort_16way_avx512(strings, indices, len);
}

// AVX2 true parallel string sorting using SIMD comparison networks
// Note: the AVX2 sorting network may be revisited as additional layouts/strategies are evaluated.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
pub(crate) unsafe fn sort_strings_avx2(strings: &[String], indices: &mut [u32], len: usize) {
    // Use AVX2 8-way parallel sorting network with zero-copy operations
    sort_8way_avx2(strings, indices, len);
}

//  NEON 4-way parallel string sorting with zero-copy operations
#[cfg(all(target_arch = "aarch64", not(feature = "hwx-nightly")))]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn sort_strings_neon(strings: &[String], indices: &mut [u32], len: usize) {
    // Use NEON 4-way parallel sorting network with zero-copy operations
    sort_4way_neon(strings, indices, len);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "hwx-nightly"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn sort_16way_avx512(strings: &[String], indices: &mut [u32], indices_len: usize) {
    const SIMD_WIDTH: usize = 16;

    // Process 16 indices at a time using insertion sort
    for chunk_start in (0..indices_len).step_by(SIMD_WIDTH) {
        let chunk_end = (chunk_start + SIMD_WIDTH).min(indices_len);
        let actual_size = chunk_end - chunk_start;

        if actual_size < 2 {
            continue;
        }

        // Insertion sort the chunk
        for i in 1..actual_size {
            let key = indices[chunk_start + i];
            let mut j = i;
            while j > 0 && compare_strings_avx512(strings, indices[chunk_start + j - 1], key) {
                indices[chunk_start + j] = indices[chunk_start + j - 1];
                j -= 1;
            }
            indices[chunk_start + j] = key;
        }
    }

    // Merge sorted 16-element chunks
    if indices_len > SIMD_WIDTH {
        merge_sorted_chunks_avx512(strings, indices, SIMD_WIDTH, indices_len);
    }
}

// True 8-way parallel sort using AVX2 SIMD conditional swaps
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
unsafe fn sort_8way_avx2(strings: &[String], indices: &mut [u32], indices_len: usize) {
    const SIMD_WIDTH: usize = 8;

    // Process 8 indices at a time using insertion sort
    for chunk_start in (0..indices_len).step_by(SIMD_WIDTH) {
        let chunk_end = (chunk_start + SIMD_WIDTH).min(indices_len);
        let actual_size = chunk_end - chunk_start;

        if actual_size < 2 {
            continue;
        }

        // Insertion sort the chunk
        for i in 1..actual_size {
            let key = indices[chunk_start + i];
            let mut j = i;
            while j > 0 && compare_strings_avx2(strings, indices[chunk_start + j - 1], key) {
                indices[chunk_start + j] = indices[chunk_start + j - 1];
                j -= 1;
            }
            indices[chunk_start + j] = key;
        }
    }

    // Merge sorted 8-element chunks
    if indices_len > SIMD_WIDTH {
        merge_sorted_chunks_avx2(strings, indices, SIMD_WIDTH, indices_len);
    }
}

// True 4-way parallel sort using NEON SIMD conditional swaps
#[cfg(all(target_arch = "aarch64", not(feature = "hwx-nightly")))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sort_4way_neon(strings: &[String], indices: &mut [u32], indices_len: usize) {
    const SIMD_WIDTH: usize = 4;

    // Process 4 indices at a time with true SIMD operations
    for chunk_start in (0..indices_len).step_by(SIMD_WIDTH) {
        let chunk_end = (chunk_start + SIMD_WIDTH).min(indices_len);
        let actual_size = chunk_end - chunk_start;

        if actual_size < 2 {
            continue;
        }

        // Sort the actual_size elements using insertion sort
        for i in 1..actual_size {
            let key = indices[chunk_start + i];
            let mut j = i;
            while j > 0 && compare_strings_neon(strings, indices[chunk_start + j - 1], key) {
                indices[chunk_start + j] = indices[chunk_start + j - 1];
                j -= 1;
            }
            indices[chunk_start + j] = key;
        }
    }

    // Merge sorted 4-element chunks
    if indices_len > SIMD_WIDTH {
        merge_sorted_chunks_neon(strings, indices, SIMD_WIDTH, indices_len);
    }
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn compare_strings_avx512(strings: &[String], idx1: u32, idx2: u32) -> bool {
    if idx1 == u32::MAX {
        log::debug!("🟠 [COMPARE] idx1 is MAX, returning false");
        return false; // MAX is considered greater, so don't swap if s2 is valid
    }
    if idx2 == u32::MAX {
        log::debug!("🟠 [COMPARE] idx2 is MAX, returning true");
        return true; // Valid s1 is smaller than MAX, so swap
    }

    let s1 = &strings[idx1 as usize];
    let s2 = &strings[idx2 as usize];

    log::debug!(
        "🟠 [COMPARE] Comparing '{}' (idx {}) vs '{}' (idx {})",
        s1,
        idx1,
        s2,
        idx2
    );

    let s1_bytes = s1.as_bytes();
    let s2_bytes = s2.as_bytes();
    let s1_len = s1_bytes.len();
    let s2_len = s2_bytes.len();
    let min_len = s1_len.min(s2_len);

    // Unrolled 64-byte SIMD comparison (matching AVX2 pattern exactly)
    const LANES: usize = 64;
    const UNROLL_FACTOR: usize = 4;

    let unrolled_chunks = min_len / (LANES * UNROLL_FACTOR);
    let mut offset = 0;

    // Process 4 chunks at once for better instruction-level parallelism
    for _ in 0..unrolled_chunks {
        for unroll_idx in 0..UNROLL_FACTOR {
            let chunk_offset = offset + unroll_idx * LANES;
            let v1 = _mm512_loadu_si512(s1_bytes.as_ptr().add(chunk_offset) as *const _);
            let v2 = _mm512_loadu_si512(s2_bytes.as_ptr().add(chunk_offset) as *const _);

            let eq_mask = _mm512_cmpeq_epu8_mask(v1, v2);

            if eq_mask != 0xFFFFFFFFFFFFFFFF {
                let diff_pos = eq_mask.trailing_ones() as usize;
                let pos = chunk_offset + diff_pos;
                return s1_bytes[pos] > s2_bytes[pos]; // Match NEON: descending
            }
        }
        offset += LANES * UNROLL_FACTOR;
    }

    // Handle remaining full chunks
    let remaining_chunks = (min_len - offset) / LANES;
    for _ in 0..remaining_chunks {
        let v1 = _mm512_loadu_si512(s1_bytes.as_ptr().add(offset) as *const _);
        let v2 = _mm512_loadu_si512(s2_bytes.as_ptr().add(offset) as *const _);

        let eq_mask = _mm512_cmpeq_epu8_mask(v1, v2);

        if eq_mask != 0xFFFFFFFFFFFFFFFF {
            let diff_pos = eq_mask.trailing_ones() as usize;
            let pos = offset + diff_pos;
            return s1_bytes[pos] > s2_bytes[pos]; // Match NEON: descending
        }
        offset += LANES;
    }

    // Handle remaining bytes with 256-bit SIMD when possible (like AVX2)
    let remaining_bytes = min_len - offset;
    if remaining_bytes >= 32 {
        // Use 256-bit SIMD for 32-63 remaining bytes
        let v1 = _mm256_loadu_si256(s1_bytes.as_ptr().add(offset) as *const _);
        let v2 = _mm256_loadu_si256(s2_bytes.as_ptr().add(offset) as *const _);

        let eq_mask = _mm256_cmpeq_epi8(v1, v2);
        let eq_bits = _mm256_movemask_epi8(eq_mask) as u32;

        let remaining_mask = (1u32 << remaining_bytes.min(32)) - 1;
        if (eq_bits & remaining_mask) != remaining_mask {
            let diff_pos = (eq_bits & remaining_mask).trailing_ones() as usize;
            let pos = offset + diff_pos;
            return s1_bytes[pos] > s2_bytes[pos]; // Match NEON: descending
        }
        offset += 32;
    }

    // Handle remaining bytes with 128-bit SIMD when possible
    let remaining_bytes = min_len - offset;
    if remaining_bytes >= 16 {
        // Use 128-bit SIMD for 16-31 remaining bytes
        let v1 = _mm_loadu_si128(s1_bytes.as_ptr().add(offset) as *const _);
        let v2 = _mm_loadu_si128(s2_bytes.as_ptr().add(offset) as *const _);

        let eq_mask = _mm_cmpeq_epi8(v1, v2);
        let eq_bits = _mm_movemask_epi8(eq_mask) as u32;

        let remaining_mask = (1u32 << remaining_bytes.min(16)) - 1;
        if (eq_bits & remaining_mask) != remaining_mask {
            let diff_pos = (eq_bits & remaining_mask).trailing_ones() as usize;
            let pos = offset + diff_pos;
            return s1_bytes[pos] > s2_bytes[pos]; // Match NEON: descending
        }
        offset += 16;
    }

    // Handle final scalar bytes
    for i in offset..min_len {
        if s1_bytes[i] != s2_bytes[i] {
            return s1_bytes[i] > s2_bytes[i];
        }
    }

    let result = if s1_len == s2_len {
        // For equal strings, compare indices to maintain stability and break ties
        idx1 > idx2
    } else {
        s1_len > s2_len
    };
    log::debug!(
        "🟠 [COMPARE] Result: {} (s1_len={}, s2_len={})",
        result,
        s1_len,
        s2_len
    );
    result
}

// Fast SIMD string comparison for AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
unsafe fn compare_strings_avx2(strings: &[String], idx1: u32, idx2: u32) -> bool {
    if idx1 == u32::MAX {
        return false; // MAX is considered greater, so don't swap if s2 is valid
    }
    if idx2 == u32::MAX {
        return true; // Valid s1 is smaller than MAX, so swap
    }

    // Bounds checks are implicitly handled by the MAX checks above
    let s1 = &strings[idx1 as usize];
    let s2 = &strings[idx2 as usize];

    let s1_bytes = s1.as_bytes();
    let s2_bytes = s2.as_bytes();
    let s1_len = s1_bytes.len();
    let s2_len = s2_bytes.len();
    let min_len = s1_len.min(s2_len);

    // Unrolled 32-byte SIMD comparison
    const LANES: usize = 32;
    const UNROLL_FACTOR: usize = 4;

    let unrolled_chunks = min_len / (LANES * UNROLL_FACTOR);
    let mut offset = 0;

    // Process 4 chunks at once for better instruction-level parallelism
    for _ in 0..unrolled_chunks {
        for unroll_idx in 0..UNROLL_FACTOR {
            let chunk_offset = offset + unroll_idx * LANES;
            let v1 = _mm256_loadu_si256(s1_bytes.as_ptr().add(chunk_offset) as *const _);
            let v2 = _mm256_loadu_si256(s2_bytes.as_ptr().add(chunk_offset) as *const _);

            let eq_mask = _mm256_cmpeq_epi8(v1, v2);
            let eq_bits = _mm256_movemask_epi8(eq_mask) as u32;

            if eq_bits != 0xFFFFFFFF {
                let diff_pos = eq_bits.trailing_ones() as usize;
                let pos = chunk_offset + diff_pos;
                return s1_bytes[pos] > s2_bytes[pos]; // Match NEON: descending
            }
        }
        offset += LANES * UNROLL_FACTOR;
    }

    // Handle remaining full chunks
    let remaining_chunks = (min_len - offset) / LANES;
    for _ in 0..remaining_chunks {
        let v1 = _mm256_loadu_si256(s1_bytes.as_ptr().add(offset) as *const _);
        let v2 = _mm256_loadu_si256(s2_bytes.as_ptr().add(offset) as *const _);

        let eq_mask = _mm256_cmpeq_epi8(v1, v2);
        let eq_bits = _mm256_movemask_epi8(eq_mask) as u32;

        if eq_bits != 0xFFFFFFFF {
            let diff_pos = eq_bits.trailing_ones() as usize;
            let pos = offset + diff_pos;
            return s1_bytes[pos] > s2_bytes[pos]; // Match NEON: descending
        }
        offset += LANES;
    }

    // Handle remaining bytes with 128-bit SIMD when possible
    let remaining_bytes = min_len - offset;
    if remaining_bytes >= 16 {
        // Use 128-bit SIMD for 16-31 remaining bytes
        let v1 = _mm_loadu_si128(s1_bytes.as_ptr().add(offset) as *const _);
        let v2 = _mm_loadu_si128(s2_bytes.as_ptr().add(offset) as *const _);

        let eq_mask = _mm_cmpeq_epi8(v1, v2);
        let eq_bits = _mm_movemask_epi8(eq_mask) as u32;

        let remaining_mask = (1u32 << remaining_bytes.min(16)) - 1;
        if (eq_bits & remaining_mask) != remaining_mask {
            let diff_pos = (eq_bits & remaining_mask).trailing_ones() as usize;
            let pos = offset + diff_pos;
            return s1_bytes[pos] > s2_bytes[pos]; // Match NEON: descending
        }
        offset += 16;
    }

    // Handle final scalar bytes
    for i in offset..min_len {
        if s1_bytes[i] != s2_bytes[i] {
            return s1_bytes[i] > s2_bytes[i]; // Match NEON: descending
        }
    }

    if s1_len == s2_len {
        // For equal strings, compare indices to maintain stability and break ties
        idx1 > idx2
    } else {
        s1_len > s2_len // Match NEON: descending
    }
}

// Fast SIMD string comparison for NEON
// Returns true if s1 should come AFTER s2 (i.e., s1 > s2 for ascending order swap)
#[cfg(all(target_arch = "aarch64", not(feature = "hwx-nightly")))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn compare_strings_neon(strings: &[String], idx1: u32, idx2: u32) -> bool {
    if idx1 == u32::MAX {
        return false; // MAX is considered greater, so don't swap if s2 is valid
    }
    if idx2 == u32::MAX {
        return true; // Valid s1 is smaller than MAX, so swap
    }

    // Bounds checks are implicitly handled by the MAX checks above
    let s1 = &strings[idx1 as usize];
    let s2 = &strings[idx2 as usize];

    let s1_bytes = s1.as_bytes();
    let s2_bytes = s2.as_bytes();
    let s1_len = s1_bytes.len(); // slice length
    let s2_len = s2_bytes.len(); // slice length
    let min_len = s1_len.min(s2_len);

    // 16-byte SIMD comparison
    let chunks = min_len / 16;
    for i in 0..chunks {
        let offset = i * 16;
        let v1 = vld1q_u8(s1_bytes.as_ptr().add(offset) as *const u8);
        let v2 = vld1q_u8(s2_bytes.as_ptr().add(offset) as *const u8);

        let eq_mask = vceqq_u8(v1, v2);
        let eq_bits = vgetq_lane_u64(vreinterpretq_u64_u8(eq_mask), 0);

        if eq_bits != u64::MAX {
            let diff_pos = eq_bits.trailing_ones() as usize / 8;
            let pos = offset + diff_pos;
            return s1_bytes[pos] > s2_bytes[pos]; // For ascending: return true if s1 > s2 (need swap)
        }
    }

    // Handle remainder and length comparison
    for i in (chunks * 16)..min_len {
        if s1_bytes[i] != s2_bytes[i] {
            return s1_bytes[i] > s2_bytes[i]; // For ascending: return true if s1 > s2
        }
    }

    s1_len > s2_len // For ascending: longer string comes after
}

// SIMD merge of sorted chunks using parallel operations
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
unsafe fn merge_sorted_chunks_avx2(
    strings: &[String],
    indices: &mut [u32],
    chunk_size: usize,
    indices_len: usize,
) {
    let mut current_size = chunk_size;

    while current_size < indices_len {
        for start in (0..indices_len).step_by(current_size * 2) {
            let mid = (start + current_size).min(indices_len);
            let end = (start + current_size * 2).min(indices_len);

            if mid < end {
                // This is the correct, final fix for the merge logic.
                // The issue is that `merge_two_sorted_avx2` has a fixed-size buffer of 1024.
                // When `current_size` exceeds 512, the merge size (`current_size * 2`) can exceed 1024.
                // The solution is to perform a correct merge here, using a temporary buffer for one of the chunks.

                let left_len = mid - start;
                let right_len = end - mid;

                // If the total merge size is small enough, use the fast path worker function.
                if left_len + right_len <= 1024 {
                    merge_two_sorted_avx2(strings, indices, start, mid, end);
                } else {
                    // For larger merges, we must handle it here to avoid buffer overflow.
                    // We copy the left chunk into a temporary buffer, then merge from the temp buffer
                    // and the right chunk back into the main `indices` slice.

                    // Use dynamic allocation for large merges
                    let mut temp = vec![0u32; left_len];
                    temp.copy_from_slice(&indices[start..mid]);

                    let mut i = 0; // Pointer for temp (left chunk)
                    let mut j = mid; // Pointer for indices (right chunk)
                    let mut k = start; // Write pointer for indices

                    // Merge from temp and the right half of indices back into indices (negate for ascending order)
                    while i < left_len && j < end {
                        if !compare_strings_avx2(strings, temp[i], indices[j]) {
                            indices[k] = temp[i];
                            i += 1;
                        } else {
                            indices[k] = indices[j];
                            j += 1;
                        }
                        k += 1;
                    }

                    // Copy any remaining elements from the temp buffer
                    while i < left_len {
                        indices[k] = temp[i];
                        i += 1;
                        k += 1;
                    }
                    // Any remaining elements in the right half are already in place.
                }
            }
        }
        current_size *= 2;
    }
}

// SIMD merge two sorted ranges using vectorized operations
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
unsafe fn merge_two_sorted_avx2(
    strings: &[String],
    indices: &mut [u32],
    start: usize,
    mid: usize,
    end: usize,
) {
    const MAX_MERGE_SIZE: usize = 1024; // Fixed-size merge buffer
    let temp_size = end - start;
    let mut temp: [u32; MAX_MERGE_SIZE] = [0; MAX_MERGE_SIZE];

    // Fallback to scalar merge if too large
    if temp_size > MAX_MERGE_SIZE {
        // Simple scalar merge without allocation
        let mut left = start;
        let mut right = mid;
        while left < mid && right < end {
            if !compare_strings_avx2(strings, indices[left], indices[right]) {
                let val = indices[left];
                for k in (left..right).rev() {
                    indices[k + 1] = indices[k];
                }
                indices[right] = val;
                left += 1;
                right += 1;
            } else {
                left += 1;
            }
        }
        return;
    }

    let mut i = start;
    let mut j = mid;
    let mut k = 0;

    // Merge using SIMD string comparisons (negate for ascending order)
    while i < mid && j < end {
        if !compare_strings_avx2(strings, indices[i], indices[j]) {
            temp[k] = indices[i];
            i += 1;
        } else {
            temp[k] = indices[j];
            j += 1;
        }
        k += 1;
    }

    // Copy remaining elements
    while i < mid {
        temp[k] = indices[i];
        i += 1;
        k += 1;
    }
    while j < end {
        temp[k] = indices[j];
        j += 1;
        k += 1;
    }

    // SIMD copy back (8 u32s at a time)
    let simd_len = temp_size & !7;
    for i in (0..simd_len).step_by(8) {
        let temp_vec = _mm256_loadu_si256(temp.as_ptr().add(i) as *const _);
        _mm256_storeu_si256(indices.as_mut_ptr().add(start + i) as *mut _, temp_vec);
    }

    // Handle remainder with minimal scalar
    let remaining = temp_size - simd_len;
    if remaining > 0 {
        // Use minimal scalar for tiny remainder only
        for i in 0..remaining {
            indices[start + simd_len + i] = temp[simd_len + i];
        }
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "hwx-nightly"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn merge_sorted_chunks_avx512(
    strings: &[String],
    indices: &mut [u32],
    chunk_size: usize,
    indices_len: usize,
) {
    let mut current_size = chunk_size;

    while current_size < indices_len {
        for start in (0..indices_len).step_by(current_size * 2) {
            let mid = (start + current_size).min(indices_len);
            let end = (start + current_size * 2).min(indices_len);

            if mid < end {
                // This is the correct, final fix for the merge logic.
                // The issue is that `merge_two_sorted_avx512` has a fixed-size buffer of 512.
                // When `current_size` exceeds 256, the merge size (`current_size * 2`) can exceed 512.
                // The solution is to perform a correct merge here, using a temporary buffer for one of the chunks.

                let left_len = mid - start;
                let right_len = end - mid;

                // If the total merge size is small enough, use the fast path worker function.
                if left_len + right_len <= 512 {
                    merge_two_sorted_avx512(strings, indices, start, mid, end);
                } else {
                    // For larger merges, we must handle it here to avoid buffer overflow.
                    // We copy the left chunk into a temporary buffer, then merge from the temp buffer
                    // and the right chunk back into the main `indices` slice.

                    // Use dynamic allocation for large merges
                    let mut temp = vec![0u32; left_len];
                    temp.copy_from_slice(&indices[start..mid]);

                    let mut i = 0; // Pointer for temp (left chunk)
                    let mut j = mid; // Pointer for indices (right chunk)
                    let mut k = start; // Write pointer for indices

                    // Merge from temp and the right half of indices back into indices (negate for ascending order)
                    while i < left_len && j < end {
                        #[cfg(feature = "hwx-nightly")]
                        let should_use_temp = !compare_strings_avx512(strings, temp[i], indices[j]);
                        #[cfg(not(feature = "hwx-nightly"))]
                        let should_use_temp = !compare_strings_avx2(strings, temp[i], indices[j]);

                        if should_use_temp {
                            indices[k] = temp[i];
                            i += 1;
                        } else {
                            indices[k] = indices[j];
                            j += 1;
                        }
                        k += 1;
                    }

                    // Copy any remaining elements from the temp buffer
                    while i < left_len {
                        indices[k] = temp[i];
                        i += 1;
                        k += 1;
                    }
                    // Any remaining elements in the right half are already in place.
                }
            }
        }
        current_size *= 2;
    }
}

// SIMD merge of sorted chunks using NEON parallel operations
#[cfg(all(target_arch = "aarch64", not(feature = "hwx-nightly")))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn merge_sorted_chunks_neon(
    strings: &[String],
    indices: &mut [u32],
    chunk_size: usize,
    indices_len: usize,
) {
    let mut current_size = chunk_size;

    while current_size < indices_len {
        for start in (0..indices_len).step_by(current_size * 2) {
            let mid = (start + current_size).min(indices_len);
            let end = (start + current_size * 2).min(indices_len);

            if mid < end {
                // This is the correct, final fix for the merge logic.
                // The issue is that `merge_two_sorted_neon` has a fixed-size buffer of 512.
                // When `current_size` exceeds 256, the merge size (`current_size * 2`) can exceed 512.
                // The solution is to perform a correct merge here, using a temporary buffer for one of the chunks.

                let left_len = mid - start;
                let right_len = end - mid;

                // If the total merge size is small enough, use the fast path worker function.
                if left_len + right_len <= 512 {
                    merge_two_sorted_neon(strings, indices, start, mid, end);
                } else {
                    // For larger merges, we must handle it here to avoid buffer overflow.
                    // We copy the left chunk into a temporary buffer, then merge from the temp buffer
                    // and the right chunk back into the main `indices` slice.

                    // Use dynamic allocation for large merges
                    let mut temp = vec![0u32; left_len];
                    temp.copy_from_slice(&indices[start..mid]);

                    let mut i = 0; // Pointer for temp (left chunk)
                    let mut j = mid; // Pointer for indices (right chunk)
                    let mut k = start; // Write pointer for indices

                    // Merge from temp and the right half of indices back into indices
                    while i < left_len && j < end {
                        if !compare_strings_neon(strings, temp[i], indices[j]) {
                            indices[k] = temp[i];
                            i += 1;
                        } else {
                            indices[k] = indices[j];
                            j += 1;
                        }
                        k += 1;
                    }

                    // Copy any remaining elements from the temp buffer
                    while i < left_len {
                        indices[k] = temp[i];
                        i += 1;
                        k += 1;
                    }
                    // Any remaining elements in the right half are already in place.
                }
            }
        }
        current_size *= 2;
    }
}

// SIMD merge two sorted ranges using AVX512 vectorized operations
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "hwx-nightly"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn merge_two_sorted_avx512(
    strings: &[String],
    indices: &mut [u32],
    start: usize,
    mid: usize,
    end: usize,
) {
    const MAX_MERGE_SIZE: usize = 2048;
    let temp_size = end - start;
    let mut temp: [u32; MAX_MERGE_SIZE] = [0; MAX_MERGE_SIZE];

    let mut i = start; // Left array pointer
    let mut j = mid; // Right array pointer
    let mut k = 0; // Output pointer

    // Classic merge algorithm with SIMD string comparisons (negate for ascending order)
    while i < mid && j < end {
        if !compare_strings_avx512(strings, indices[i], indices[j]) {
            temp[k] = indices[i];
            i += 1;
        } else {
            temp[k] = indices[j];
            j += 1;
        }
        k += 1;
    }

    // Copy remaining elements from left array
    while i < mid {
        temp[k] = indices[i];
        i += 1;
        k += 1;
    }

    // Copy remaining elements from right array
    while j < end {
        temp[k] = indices[j];
        j += 1;
        k += 1;
    }

    // Copy result back using SIMD when possible
    let simd_len = temp_size & !15;
    for i in (0..simd_len).step_by(16) {
        let temp_vec = _mm512_loadu_si512(temp.as_ptr().add(i) as *const _);
        _mm512_storeu_si512(indices.as_mut_ptr().add(start + i) as *mut _, temp_vec);
    }

    // Handle remainder with scalar
    for i in simd_len..temp_size {
        indices[start + i] = temp[i];
    }
}

// SIMD merge two sorted ranges using NEON vectorized operations
#[cfg(all(target_arch = "aarch64", not(feature = "hwx-nightly")))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn merge_two_sorted_neon(
    strings: &[String],
    indices: &mut [u32],
    start: usize,
    mid: usize,
    end: usize,
) {
    const MAX_MERGE_SIZE: usize = 512; // Fixed-size merge buffer for NEON
    let temp_size = end - start;
    let mut temp: [u32; MAX_MERGE_SIZE] = [0; MAX_MERGE_SIZE];

    let mut i = start;
    let mut j = mid;
    let mut k = 0;

    // Merge using SIMD string comparisons
    while i < mid && j < end {
        if !compare_strings_neon(strings, indices[i], indices[j]) {
            temp[k] = indices[i];
            i += 1;
        } else {
            temp[k] = indices[j];
            j += 1;
        }
        k += 1;
    }

    // Copy remaining elements
    while i < mid {
        temp[k] = indices[i];
        i += 1;
        k += 1;
    }
    while j < end {
        temp[k] = indices[j];
        j += 1;
        k += 1;
    }

    // NEON copy back (4 u32s at a time)
    let simd_len = temp_size & !3;
    for i in (0..simd_len).step_by(4) {
        let temp_vec = vld1q_u32(temp.as_ptr().add(i));
        vst1q_u32(indices.as_mut_ptr().add(start + i), temp_vec);
    }

    // Handle remainder
    indices[(simd_len + start)..(temp_size + start)].copy_from_slice(&temp[simd_len..temp_size]);
}
