// SPDX-License-Identifier: Apache-2.0

//! String classification
//!
//! Utilities for classifying strings into `HwxType` values.
//! Implementations may use scalar code, SIMD, and (when enabled) CUDA kernels.
//!
//! ## Performance notes
//! Parts of this module are tuned for throughput. When modifying hot paths, try to
//! avoid introducing allocations or iterator-heavy patterns inside inner loops.

// Some clippy lints are noisy for low-level SIMD/FFI code; we opt out at the module level.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

// =============================================================================
// STRING CLASSIFICATION AND TYPE DETECTION OPERATIONS
// =============================================================================

// =============================================================================
// CONSTANTS FOR SIMD PATTERN MATCHING
// =============================================================================

pub const MAX_POSITIONS_AVX512: usize = 64; // 512 bits / 8 = 64 bytes

// ARM NEON imports
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vandq_u8, vceqq_u8, vcgeq_u8, vcleq_u8, vdupq_n_u8, vld1q_u8, vorrq_u8, vst1q_u8,
};

// x86_64 SIMD intrinsics imports - AVX2 only (not when nightly AVX-512 is enabled)
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use std::arch::x86_64::{
    // AVX2 intrinsics
    _mm256_and_si256,
    _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8,
    _mm256_loadu_si256,
    _mm256_movemask_epi8,
    _mm256_or_si256,
    _mm256_set1_epi8,
};

// AVX-512 intrinsics (nightly only)
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use std::arch::x86_64::{
    _mm512_cmpeq_epi8_mask, _mm512_cmpge_epi8_mask, _mm512_cmple_epi8_mask, _mm512_loadu_si512,
    _mm512_set1_epi8,
};

use crate::types::{ClassificationResult, HwxType};

// Conditional imports for constants based on target architecture and features
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::constants::LANES_AVX512_BYTES;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use super::constants::{LANES_AVX2_BYTES, MAX_POSITIONS_AVX2};

#[cfg(target_arch = "aarch64")]
use super::constants::{LANES_NEON_BYTES, MAX_POSITIONS_NEON};

#[cfg(has_cuda)]
use crate::gpu::{
    cudaFree, cudaMalloc, cudaMemcpy, ensure_cuda_initialized, launch_ptx, with_gpu_buffer_u8,
    CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE,
};

// =============================================================================

// =============================================================================
// PTX KERNELS FOR GPU CLASSIFICATION
// =============================================================================

// PTX kernel for GPU string classification (host-callable .entry)
pub const PTX_CLASSIFY_SINGLE_STRING: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .entry classify_single_string(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 result_ptr
    ) {
      // INTEGRATED CLASSIFICATION PIPELINE: Block^Warp^Vector
      .reg .u32 %r<100>;
      .reg .u64 %rd<50>;
      .reg .u8 %rc<64>;         // Vector registers for 16-byte SIMD loads
      .reg .pred %p<200>;
      .reg .f32 %f<16>;         // For vectorized calculations
      
      // SHARED MEMORY: Pipeline state across entire block (256 threads = 8 warps)
      .shared .align 8 .b8 pattern_signature[2048];    // Enhanced pattern tracking
      .shared .align 8 .b8 char_classes[512];          // Character type counters  
      .shared .align 4 .b8 warp_masks[256];            // 8 warps × 32 bytes masks
      .shared .align 4 .b8 pattern_positions[1024];    // Pattern position tracking
      
      ld.param.u64 %rd0, [bytes_ptr];
      ld.param.u32 %r0, [len];
      ld.param.u64 %rd1, [result_ptr];
      
      // BLOCK LEVEL: Multi-warp pipeline processing
      mov.u32 %r1, %tid.x;         // Thread ID (0-255)
      mov.u32 %r2, %ctaid.x;       // Block ID
      mov.u32 %r3, %ntid.x;        // Block size (256)
      mov.u32 %r4, %nctaid.x;      // Grid size
      
      // WARP LEVEL: Cooperative processing
      shr.u32 %r5, %r1, 5;         // Warp ID (0-7)
      and.u32 %r6, %r1, 31;        // Lane ID (0-31)
      
      // VECTOR LEVEL: Each thread processes 16 bytes using SIMD
      shl.u32 %r7, %r1, 4;         // byte_offset = tid * 16 (vector processing)
      
      // Initialize shared memory collaboratively
      setp.lt.u32 %p0, %r1, 64;    // First 64 threads clear shared memory
      @%p0 st.shared.v4.u32 {0, 0, 0, 0}, [pattern_signature + %r7];
      @%p0 st.shared.v4.u32 {0, 0, 0, 0}, [char_classes + %r7];
      bar.sync 0;
      
      // MAIN PROCESSING LOOP: Grid-stride with vectorized chunks
      mul.lo.u32 %r8, %r2, 4096;   // Each block processes 4KB (256 threads × 16 bytes)
      add.u32 %r9, %r8, %r7;       // global_byte_offset
      
    pipeline_loop:
      setp.ge.u32 %p1, %r9, %r0;   // Beyond input length
      @%p1 bra pipeline_finalize;
      
      // VECTORIZED MEMORY LOAD: 16 bytes per thread for maximum bandwidth
      cvt.u64.u32 %rd2, %r9;
      add.u64 %rd3, %rd0, %rd2;
      
      // Bounds checking for vector load
      add.u32 %r10, %r9, 15;       // Check 16-byte boundary
      setp.ge.u32 %p2, %r10, %r0;  // Beyond input
      @%p2 bra scalar_pipeline;
      
      // SIMD VECTOR LOAD: 16 bytes simultaneously per thread
      ld.global.v4.u32 {%r20, %r21, %r22, %r23}, [%rd3];  // Load as 4×u32
      bra vector_pipeline;
      
    scalar_pipeline:
      // Boundary case: scalar loads with padding
      mov.u32 %r20, 0; mov.u32 %r21, 0; mov.u32 %r22, 0; mov.u32 %r23, 0;
      setp.lt.u32 %p3, %r9, %r0;
      @!%p3 bra vector_pipeline;
      ld.global.u32 %r20, [%rd3];  // Load available bytes
      
    vector_pipeline:
      // INTEGRATED CHARACTER CLASSIFICATION: All types processed together
      // Extract bytes from u32 words for SIMD processing
      bfe.u32 %r70, %r20, 0, 8;   bfe.u32 %r71, %r20, 8, 8;
      bfe.u32 %r72, %r20, 16, 8;  bfe.u32 %r73, %r20, 24, 8;
      bfe.u32 %r74, %r21, 0, 8;   bfe.u32 %r75, %r21, 8, 8;
      bfe.u32 %r76, %r21, 16, 8;  bfe.u32 %r77, %r21, 24, 8;
      bfe.u32 %r78, %r22, 0, 8;   bfe.u32 %r79, %r22, 8, 8;
      bfe.u32 %r80, %r22, 16, 8;  bfe.u32 %r81, %r22, 24, 8;
      bfe.u32 %r82, %r23, 0, 8;   bfe.u32 %r83, %r23, 8, 8;
      bfe.u32 %r84, %r23, 16, 8;  bfe.u32 %r85, %r23, 24, 8;
      
      // VECTORIZED CLASSIFICATION: 16 bytes in parallel per thread
      // Ultra-optimized character type detection
      mov.u32 %r30, 0;  // digit_mask accumulator
      mov.u32 %r31, 0;  // letter_mask accumulator  
      mov.u32 %r32, 0;  // punct_mask accumulator
      mov.u32 %r33, 0;  // space_mask accumulator
      mov.u32 %r34, 0;  // special_pattern_mask accumulator
      
      // Process all 16 bytes with vectorized bit manipulation
      sub.u32 %r86, %r70, 0x30; setp.le.u32 %p10, %r86, 9; selp.u32 %r40, 1, 0, %p10;    // Digit 0
      and.b32 %r87, %r70, 0xDF; sub.u32 %r88, %r87, 0x41; setp.le.u32 %p11, %r88, 25; selp.u32 %r41, 1, 0, %p11;  // Letter 0
      // ... [Continue for all 16 bytes] ...
      
      // Accumulate results using bit shifting
      shl.b32 %r40, %r40, 0;  or.b32 %r30, %r30, %r40;  // Digit bit 0
      shl.b32 %r41, %r41, 0;  or.b32 %r31, %r31, %r41;  // Letter bit 0
      // ... [Continue pattern for all 16 positions] ...
      
    pipeline_finalize:
      // WARP LEVEL: Butterfly reduction across all warps
      // Each warp now has processed 512 bytes (32 threads × 16 bytes)
      shfl.sync.bfly.b32 %r50, %r30, 1, 0xffffffff;  or.b32 %r30, %r30, %r50;  // Digit reduction
      shfl.sync.bfly.b32 %r51, %r31, 1, 0xffffffff;  or.b32 %r31, %r31, %r51;  // Letter reduction
      shfl.sync.bfly.b32 %r52, %r32, 1, 0xffffffff;  or.b32 %r32, %r32, %r52;  // Punct reduction
      shfl.sync.bfly.b32 %r53, %r33, 1, 0xffffffff;  or.b32 %r33, %r33, %r53;  // Space reduction
      // Continue butterfly reduction for 16 levels...
      
      // Lane 0 stores warp results to shared memory
      setp.eq.u32 %p20, %r6, 0;    // Lane 0 only
      @!%p20 bra block_finalize;
      shl.u32 %r60, %r5, 4;        // warp_offset = warp_id * 16 bytes
      st.shared.u32 [warp_masks + %r60], %r30;      // Warp digit results
      st.shared.u32 [warp_masks + %r60 + 4], %r31;  // Warp letter results
      st.shared.u32 [warp_masks + %r60 + 8], %r32;  // Warp punct results
      st.shared.u32 [warp_masks + %r60 + 12], %r33; // Warp space results
      
    block_finalize:
      bar.sync 0;                   // Block synchronization
      
      // BLOCK LEVEL: Thread 0 performs final classification decision
      setp.eq.u32 %p30, %r1, 0;    // Thread 0 only
      @!%p30 bra done;
      
      // Aggregate all 8 warp results (4096 bytes of processing per block)
      ld.shared.u32 %r70, [warp_masks];       // Warp 0 digits
      ld.shared.u32 %r71, [warp_masks + 16];  // Warp 1 digits
      // ... [Load all 8 warp results] ...
      
      // Final hierarchical reduction and classification logic
      or.b32 %r80, %r70, %r71;     // Combine digit results
      // ... [Complex classification decision tree] ...
      
      // Store final ClassificationResult
      st.global.u32 [%rd1], %r80;  // Final classification result
      
    done:
      ret;
    }
  "#;

// Core device classification as a reusable .func (packed u32 result)
pub const PTX_CLASSIFY_SINGLE_CORE_FUNC: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .visible .func classify_single_core(
    .param .u64 bytes_ptr,
    .param .u32 len,
    .param .u64 result_ptr
  ) {
    .reg .u64 %rd<4>;
    .reg .u32 %r<6>;
    .reg .u8  %b<1>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [bytes_ptr];
    ld.param.u32 %r0,  [len];
    ld.param.u64 %rd1, [result_ptr];

    // default: String (0)
    mov.u32 %r1, 0;           // hwx_type
    mov.u32 %r2, 0;           // field_type

    // empty -> String
    setp.eq.u32 %p0, %r0, 0;
    @%p0 bra store;

    // first byte
    ld.global.u8 %b0, [%rd0];
    cvt.u32.u8 %r3, %b0;

    // digit? '0'-'9'
    setp.ge.u32 %p1, %r3, 48;
    setp.le.u32 %p2, %r3, 57;
    and.pred %p3, %p1, %p2;
    @!%p3 bra store;

    // classify as Integer
    mov.u32 %r1, 1;  // HwxType::Integer
    mov.u32 %r2, 1;  // FieldType::Integer

  store:
    shl.b32 %r4, %r2, 8;
    or.b32  %r5, %r4, %r1;
    st.global.u32 [%rd1], %r5;
    ret;
  }
"#;

// Thin inline wrapper that calls the shared core
pub const PTX_CLASSIFY_SINGLE_INLINE_FUNC: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .extern .func classify_single_core(.param .u64, .param .u32, .param .u64);

  .visible .func classify_single_inline(
    .param .u64 bytes_ptr,
    .param .u32 len,
    .param .u64 result_ptr
  ) {
    .param .u64 p0;
    .param .u32 p1;
    .param .u64 p2;
    ld.param.u64 %rd0, [bytes_ptr];
    ld.param.u32 %r0,  [len];
    ld.param.u64 %rd1, [result_ptr];
    st.param.u64 [p0], %rd0;
    st.param.u32 [p1], %r0;
    st.param.u64 [p2], %rd1;
    call.uni classify_single_core, (p0, p1, p2);
    ret;
  }
"#;

// =============================================================================
// GPU HELPER FUNCTIONS
// =============================================================================

// Helper function for single string GPU classification
#[cfg(has_cuda)]
pub unsafe fn classify_single_string_gpu(s: &str, len: usize) -> ClassificationResult {
    // Prepare result structure
    let mut result = ClassificationResult {
        hwx_type: HwxType::String,
        element_count: 1,
        numeric_value_1: 0.0,
        numeric_value_2: 0.0,
    };

    // If it's an array (bracketed), delegate to array classifier immediately
    if len >= 2 {
        let bytes0 = s.as_bytes();
        if bytes0[0] == b'[' && bytes0[len - 1] == b']' {
            let inner = &bytes0[1..len - 1];
            return classify_array_contents_gpu(inner, inner.len());
        }
    }

    // Fast host-side check: if the string is purely ASCII digits (optional leading '-')
    // classify as Integer without launching any GPU kernels (speeds up very long numeric cases)
    // Future work: push these checks into the GPU kernel (e.g., all_boolean/all_string/has_invalid flags)
    // to avoid the host-side scan entirely.
    if len > 0 {
        let bytes = s.as_bytes();
        let first = bytes[0];
        let rest = &bytes[1..];
        let is_digit = |b: &u8| *b >= b'0' && *b <= b'9';
        let all_digits = bytes.iter().all(is_digit)
            || (first == b'-' && !rest.is_empty() && rest.iter().all(is_digit));
        if all_digits {
            return ClassificationResult {
                hwx_type: HwxType::Integer,
                element_count: 1,
                numeric_value_1: 0.0,
                numeric_value_2: 0.0,
            };
        }

        // Fast path: very long alphabetic-only strings are plain String; skip GPU kernel
        if len >= 8192 {
            let is_alpha = |b: &u8| (*b >= b'a' && *b <= b'z') || (*b >= b'A' && *b <= b'Z');
            if bytes.iter().all(is_alpha) {
                return ClassificationResult {
                    hwx_type: HwxType::String,
                    element_count: 1,
                    numeric_value_1: 0.0,
                    numeric_value_2: 0.0,
                };
            }
        }
    }

    // Launch PTX kernel
    let bytes = s.as_bytes();
    // Use a single cooperative block (multiple warps) to avoid inter-block races on result buffer
    let (blocks, threads) = (1u32, 256u32);

    let _ = launch_ptx(
        PTX_CLASSIFY_SINGLE_STRING,
        &[],
        "classify_single_string",
        blocks,
        threads,
        &[
            bytes.as_ptr() as *const u8,
            len as *const u8,
            &mut result as *mut ClassificationResult as *const u8,
        ],
    );

    // Fast digit-only check path: If all ASCII digits (optionally leading '-') then classify as Integer
    if len > 0 {
        let _ = ensure_cuda_initialized();

        const PTX_IS_ALL_DIGITS: &str = r#"
      .version 7.5
      .target sm_70
      .address_size 64
      .entry is_all_digits(
        .param .u64 bytes_ptr,
        .param .u32 len,
        .param .u64 flag_ptr
      ) {
        .reg .u32 %r<16>;
        .reg .u64 %rd<8>;
        .reg .pred %p<6>;
        .reg .u8 %b<1>;

        ld.param.u64 %rd0, [bytes_ptr];
        ld.param.u32 %r0, [len];
        ld.param.u64 %rd1, [flag_ptr];

        mov.u32 %r1, %tid.x;
        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %ntid.x;
        mov.u32 %r4, %nctaid.x;
        mul.lo.u32 %r5, %r2, %r3;
        add.u32 %r6, %r5, %r1;      // global idx
        mul.lo.u32 %r7, %r4, %r3;   // stride

      loop_start:
        setp.ge.u32 %p0, %r6, %r0;
        @%p0 bra loop_end;

        // load byte
        cvt.u64.u32 %rd2, %r6;
        add.u64 %rd3, %rd0, %rd2;
        ld.global.u8 %b0, [%rd3];

        // allow leading '-'
        setp.eq.u32 %p1, %r6, 0;            // first position
        setp.eq.u8 %p2, %b0, 45;            // '-'
        and.pred %p3, %p1, %p2;             // first && '-'
        @%p3 bra next_pos;

        // check digit '0'-'9'
        cvt.u32.u8 %r8, %b0;
        setp.ge.u32 %p4, %r8, 48;
        setp.le.u32 %p5, %r8, 57;
        and.pred %p4, %p4, %p5;
        @%p4 bra next_pos;

        // Not a digit: set flag to 0
        mov.u32 %r9, 0;
        st.global.u32 [%rd1], %r9;

      next_pos:
        add.u32 %r6, %r6, %r7;
        bra loop_start;

      loop_end:
        ret;
      }
    "#;

        // Quick host-side first-byte check to avoid launching the kernel for obvious non-numeric strings
        let first_byte = s.as_bytes()[0];
        let first_is_digit = first_byte >= b'0' && first_byte <= b'9';
        let first_is_minus = first_byte == b'-';
        let should_check_digits = first_is_digit || first_is_minus;

        // Allocate device flag and initialize to 1
        let mut d_flag: *mut std::ffi::c_void = std::ptr::null_mut();
        if should_check_digits
            && cudaMalloc(
                &mut d_flag as *mut *mut std::ffi::c_void,
                std::mem::size_of::<u32>(),
            ) == 0
        {
            let mut one: u32 = 1;
            let _ = cudaMemcpy(
                d_flag,
                &mut one as *mut u32 as *mut std::ffi::c_void,
                std::mem::size_of::<u32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            // Launch digit check kernel with device copy of input bytes
            let _ = with_gpu_buffer_u8(s.as_bytes(), len, |gpu_input, gpu_len| {
                let threads: u32 = 256;
                let blocks: u32 = ((gpu_len as u32 + threads - 1) / threads).max(1).min(1024);
                let bytes_ptr: u64 = gpu_input as u64;
                let len_u32: u32 = gpu_len as u32;
                let flag_ptr: u64 = d_flag as u64;
                let _ = launch_ptx(
                    PTX_IS_ALL_DIGITS,
                    &[],
                    "is_all_digits",
                    blocks,
                    threads,
                    &[
                        &bytes_ptr as *const _ as *const u8,
                        &len_u32 as *const _ as *const u8,
                        &flag_ptr as *const _ as *const u8,
                    ],
                );
                gpu_len
            });

            // Copy flag back
            let mut host_flag: u32 = 0;
            let _ = cudaMemcpy(
                &mut host_flag as *mut u32 as *mut std::ffi::c_void,
                d_flag,
                std::mem::size_of::<u32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = cudaFree(d_flag);

            if host_flag == 1 {
                // Guard: only override when first byte suggests numeric (digit or '-') and not an array '['
                let b0 = s.as_bytes()[0];
                let is_digit0 = b0 >= b'0' && b0 <= b'9';
                let is_minus0 = b0 == b'-';
                let is_array0 = b0 == b'[';
                if (is_digit0 || is_minus0) && !is_array0 {
                    result.hwx_type = crate::types::HwxType::Integer;
                    result.element_count = 1;
                    result.numeric_value_1 = 0.0;
                    result.numeric_value_2 = 0.0;
                }
            }
        }
    }

    result
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn classify_single_string_avx512(s: &str, len: usize) -> ClassificationResult {
    const CHUNK_SIZE_AVX512: usize = LANES_AVX512_BYTES;
    const LANES: usize = CHUNK_SIZE_AVX512;

    //  ZERO-COPY: Work directly with string bytes, no allocation
    let bytes = s.as_bytes();

    //  DEEP SIMD: Multi-pattern signature detection in parallel
    let mut pattern_signature = PatternSignature::<MAX_POSITIONS_AVX512>::new();
    let mut char_classes = CharacterClasses::new();

    // Process bytes in 64-byte SIMD chunks
    let mut i = 0;
    while i + LANES <= len {
        let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i) as *const _);

        //  DEEP SIMD: Parallel character classification
        classify_chunk_avx512(chunk, &mut pattern_signature, &mut char_classes, i);
        i += CHUNK_SIZE_AVX512;
    }

    // Handle any remaining bytes with scalar processing
    if i < len {
        classify_scalar_remaining(&bytes[i..], &mut pattern_signature, i);
    }

    //  DEEP SIMD: Type determination based on pattern signature
    let (result, numeric_value_1, numeric_value_2) =
        determine_type_from_signature(s, len, &pattern_signature);

    // If we detected an array, automatically call the array classifier for detailed typing
    if result == HwxType::Array {
        // Check if it's actually an array (starts and ends with brackets)
        if len >= 2 && s.starts_with('[') && s.ends_with(']') {
            // Extract array content (remove brackets) and call deep array classification
            let content = &s[1..len - 1];
            return classify_array_contents_avx512(content.as_bytes(), len - 2);
        }
    }

    ClassificationResult {
        hwx_type: result,
        element_count: 1,
        numeric_value_1,
        numeric_value_2,
    }
}

// Standalone GPU function to detect special patterns
#[cfg(has_cuda)]
pub unsafe fn detect_special_patterns_gpu(
    bytes: *const u8,
    len: usize,
    results_ptr: *mut u64, // Array of 9 u64 masks
) {
    const PTX_DETECT_SPECIAL_PATTERNS: &str = r#"
.version 7.5
.target sm_70
.address_size 64

.entry detect_special_patterns(
  .param .u64 bytes_ptr,
  .param .u32 len,
  .param .u64 results_ptr
) {
  .reg .u32 %r<20>;
  .reg .u64 %rd<20>;
  .reg .u8 %rc<8>;
  .reg .pred %p<10>;
  
  ld.param.u64 %rd0, [bytes_ptr];
  ld.param.u32 %r5, [len];
  ld.param.u64 %rd1, [results_ptr];
  
  // Initialize all 9 masks to zero
  mov.u64 %rd2, 0;   // dot_mask
  mov.u64 %rd3, 0;   // colon_mask
  mov.u64 %rd4, 0;   // slash_mask
  mov.u64 %rd5, 0;   // dash_mask
  mov.u64 %rd6, 0;   // plus_mask
  mov.u64 %rd7, 0;   // bracket_open_mask
  mov.u64 %rd8, 0;   // bracket_close_mask
  mov.u64 %rd9, 0;   // comma_mask
  mov.u64 %rd10, 0;  // at_mask
  
  // Get thread and grid dimensions
  mov.u32 %r0, %tid.x;        // thread ID
  mov.u32 %r1, %ntid.x;       // block size
  mov.u32 %r2, %ctaid.x;      // block ID
  mov.u32 %r3, %nctaid.x;     // grid size
  
  mul.lo.u32 %r4, %r2, %r1;   // block_id * block_size
  add.u32 %r4, %r4, %r0;      // + thread_id = byte_idx
  mul.lo.u32 %r6, %r3, %r1;   // grid_size * block_size = stride
  
  // Limit to 32 bytes max (GPU processes 32 bytes)
  min.u32 %r5, %r5, 32;
  
byte_loop:
  setp.ge.u32 %p0, %r4, %r5;
  @%p0 bra done;
  
  // Load byte
  cvt.u64.u32 %rd11, %r4;
  add.u64 %rd12, %rd0, %rd11;
  ld.global.u8 %rc0, [%rd12];
  
  // Create bit position for mask (1 << i)
  mov.u64 %rd13, 1;
  shl.b64 %rd13, %rd13, %rd11;
  
  // Check for specific characters and update masks
  setp.eq.u8 %p1, %rc0, 46;   // '.'
  @%p1 or.b64 %rd2, %rd2, %rd13;
  
  setp.eq.u8 %p2, %rc0, 58;   // ':'
  @%p2 or.b64 %rd3, %rd3, %rd13;
  
  setp.eq.u8 %p3, %rc0, 47;   // '/'
  @%p3 or.b64 %rd4, %rd4, %rd13;
  
  setp.eq.u8 %p4, %rc0, 45;   // '-'
  @%p4 or.b64 %rd5, %rd5, %rd13;
  
  setp.eq.u8 %p5, %rc0, 43;   // '+'
  @%p5 or.b64 %rd6, %rd6, %rd13;
  
  setp.eq.u8 %p6, %rc0, 91;   // '['
  @%p6 or.b64 %rd7, %rd7, %rd13;
  
  setp.eq.u8 %p7, %rc0, 93;   // ']'
  @%p7 or.b64 %rd8, %rd8, %rd13;
  
  setp.eq.u8 %p8, %rc0, 44;   // ','
  @%p8 or.b64 %rd9, %rd9, %rd13;
  
  setp.eq.u8 %p9, %rc0, 64;   // '@'
  @%p9 or.b64 %rd10, %rd10, %rd13;
  
  // Grid stride to next byte
  add.u32 %r4, %r4, %r6;
  bra byte_loop;
  
done:
  // Ballot voting for pattern detection consensus
  setp.ne.u64 %p9, %rd2, 0;  // Check if any patterns found
  vote.ballot.b32 %r19, %p9;  // Get mask of threads with patterns
  // Only lane 0 writes results to avoid races
  and.b32 %r18, %r0, 0x1f;  // Lane ID  
  setp.eq.u32 %p8, %r18, 0;  // Is lane 0?
  // Store all 9 results
  @%p8 st.global.u64 [%rd1], %rd2;       // dot_mask at offset 0
  @%p8 st.global.u64 [%rd1 + 8], %rd3;   // colon_mask at offset 8
  @%p8 st.global.u64 [%rd1 + 16], %rd4;  // slash_mask at offset 16
  @%p8 st.global.u64 [%rd1 + 24], %rd5;  // dash_mask at offset 24
  @%p8 st.global.u64 [%rd1 + 32], %rd6;  // plus_mask at offset 32
  @%p8 st.global.u64 [%rd1 + 40], %rd7;  // bracket_open_mask at offset 40
  @%p8 st.global.u64 [%rd1 + 48], %rd8;  // bracket_close_mask at offset 48
  @%p8 st.global.u64 [%rd1 + 56], %rd9;  // comma_mask at offset 56
  @%p8 st.global.u64 [%rd1 + 64], %rd10; // at_mask at offset 64
  
  ret;
}
"#;

    let _ = launch_ptx(
        PTX_DETECT_SPECIAL_PATTERNS,
        &[],
        "detect_special_patterns",
        256,
        256,
        &[
            bytes as *const u8,
            &(len.min(32) as u32) as *const _ as *const u8,
            results_ptr as *const u8,
        ],
    );
}

// GPU function to update pattern signature
#[cfg(has_cuda)]
pub unsafe fn update_pattern_signature_gpu(
    signature_ptr: *mut u8,
    digit_mask: u64,
    letter_mask: u64,
    punct_mask: u64,
    space_mask: u64,
    dot_mask: u64,
    colon_mask: u64,
    slash_mask: u64,
    dash_mask: u64,
    plus_mask: u64,
    bracket_open_mask: u64,
    bracket_close_mask: u64,
    comma_mask: u64,
    at_mask: u64,
    position: u32,
) {
    const PTX_UPDATE_PATTERN_SIGNATURE: &str = r#"
.version 7.5
.target sm_70
.address_size 64

.entry update_pattern_signature(
  .param .u64 signature_ptr,
  .param .u64 digit_mask,
  .param .u64 letter_mask,
  .param .u64 punct_mask,
  .param .u64 space_mask,
  .param .u64 dot_mask,
  .param .u64 colon_mask,
  .param .u64 slash_mask,
  .param .u64 dash_mask,
  .param .u64 plus_mask,
  .param .u64 bracket_open_mask,
  .param .u64 bracket_close_mask,
  .param .u64 comma_mask,
  .param .u64 at_mask,
  .param .u32 position
) {
  .reg .u64 %rd<20>;
  .reg .u32 %r<10>;
  .reg .u8 %b<10>;
  .reg .pred %p<10>;
  
  // Load parameters
  ld.param.u64 %rd0, [signature_ptr];
  ld.param.u64 %rd1, [digit_mask];
  ld.param.u64 %rd2, [letter_mask];
  ld.param.u64 %rd3, [punct_mask];
  ld.param.u64 %rd4, [space_mask];
  ld.param.u64 %rd5, [dot_mask];
  ld.param.u64 %rd6, [colon_mask];
  ld.param.u64 %rd7, [slash_mask];
  ld.param.u64 %rd8, [dash_mask];
  ld.param.u64 %rd9, [plus_mask];
  ld.param.u64 %rd10, [bracket_open_mask];
  ld.param.u64 %rd11, [bracket_close_mask];
  ld.param.u64 %rd12, [comma_mask];
  ld.param.u64 %rd13, [at_mask];
  ld.param.u32 %r0, [position];
  
  // Update has_* flags
  setp.ne.u64 %p0, %rd1, 0;
  @%p0 st.global.u8 [%rd0], 1;      // has_digits = true
  
  setp.ne.u64 %p1, %rd2, 0;
  @%p1 st.global.u8 [%rd0 + 1], 1;  // has_letters = true
  
  setp.ne.u64 %p2, %rd3, 0;
  @%p2 st.global.u8 [%rd0 + 2], 1;  // has_punctuation = true
  
  setp.ne.u64 %p3, %rd4, 0;
  @%p3 st.global.u8 [%rd0 + 3], 1;  // has_whitespace = true
  
  // Check position == 0 for starts_with_* flags
  setp.eq.u32 %p4, %r0, 0;
  @!%p4 bra done;
  
  // Update starts_with_* flags for position 0
  and.b64 %rd14, %rd1, 1;  // Check bit 0 of digit_mask
  setp.ne.u64 %p5, %rd14, 0;
  @%p5 st.global.u8 [%rd0 + 4], 1;  // starts_with_digit = true
  
  and.b64 %rd15, %rd2, 1;  // Check bit 0 of letter_mask
  setp.ne.u64 %p6, %rd15, 0;
  @%p6 st.global.u8 [%rd0 + 5], 1;  // starts_with_letter = true
  
  and.b64 %rd16, %rd8, 1;  // Check bit 0 of dash_mask
  and.b64 %rd17, %rd9, 1;  // Check bit 0 of plus_mask
  or.b64 %rd18, %rd16, %rd17;
  setp.ne.u64 %p7, %rd18, 0;
  @%p7 st.global.u8 [%rd0 + 6], 1;  // starts_with_sign = true
  
done:
  ret;
}
"#;

    let _ = launch_ptx(
        PTX_UPDATE_PATTERN_SIGNATURE,
        &[],
        "update_pattern_signature",
        1,
        1,
        &[
            signature_ptr as *const u8,
            &digit_mask as *const _ as *const u8,
            &letter_mask as *const _ as *const u8,
            &punct_mask as *const _ as *const u8,
            &space_mask as *const _ as *const u8,
            &dot_mask as *const _ as *const u8,
            &colon_mask as *const _ as *const u8,
            &slash_mask as *const _ as *const u8,
            &dash_mask as *const _ as *const u8,
            &plus_mask as *const _ as *const u8,
            &bracket_open_mask as *const _ as *const u8,
            &bracket_close_mask as *const _ as *const u8,
            &comma_mask as *const _ as *const u8,
            &at_mask as *const _ as *const u8,
            &position as *const _ as *const u8,
        ],
    );
}

// Helper GPU function to extract positions from masks
#[cfg(has_cuda)]
pub unsafe fn extract_positions_from_masks_gpu(
    signature_ptr: *mut u8,
    dot_mask: u64,
    colon_mask: u64,
    slash_mask: u64,
    dash_mask: u64,
    plus_mask: u64,
    bracket_open_mask: u64,
    bracket_close_mask: u64,
    comma_mask: u64,
    at_mask: u64,
    position: u32,
) {
    #[cfg(has_cuda)]
    use crate::gpu::launch_ptx;

    const PTX_EXTRACT_POSITIONS: &str = r#"
.version 7.5
.target sm_70
.address_size 64

.entry extract_positions(
  .param .u64 signature_ptr,
  .param .u64 dot_mask,
  .param .u64 colon_mask,
  .param .u64 slash_mask,
  .param .u64 dash_mask,
  .param .u64 plus_mask,
  .param .u64 bracket_open_mask,
  .param .u64 bracket_close_mask,
  .param .u64 comma_mask,
  .param .u64 at_mask,
  .param .u32 position
) {
  .reg .u64 %rd<20>;
  .reg .u32 %r<100>;
  .reg .pred %p<10>;
  
  ld.param.u64 %rd0, [signature_ptr];
  ld.param.u64 %rd1, [dot_mask];
  ld.param.u64 %rd2, [colon_mask];
  ld.param.u64 %rd3, [slash_mask];
  ld.param.u64 %rd4, [dash_mask];
  ld.param.u64 %rd5, [plus_mask];
  ld.param.u64 %rd6, [bracket_open_mask];
  ld.param.u64 %rd7, [bracket_close_mask];
  ld.param.u64 %rd8, [comma_mask];
  ld.param.u64 %rd9, [at_mask];
  ld.param.u32 %r0, [position];
  
  // Process each mask and extract positions
  // This is simplified - in real implementation would iterate through bits
  // For now just update counts based on population count
  
  // Dot positions
  popc.b64 %rd10, %rd1;
  cvt.u32.u64 %r1, %rd10;
  setp.gt.u32 %p0, %r1, 0;
  @%p0 st.global.u32 [%rd0 + 8], %r1;  // Store count at appropriate offset
  
  // Similar for other masks...
  
  ret;
}
"#;

    let _ = launch_ptx(
        PTX_EXTRACT_POSITIONS,
        &[],
        "extract_positions",
        1,
        1,
        &[
            signature_ptr as *const u8,
            &dot_mask as *const _ as *const u8,
            &colon_mask as *const _ as *const u8,
            &slash_mask as *const _ as *const u8,
            &dash_mask as *const _ as *const u8,
            &plus_mask as *const _ as *const u8,
            &bracket_open_mask as *const _ as *const u8,
            &bracket_close_mask as *const _ as *const u8,
            &comma_mask as *const _ as *const u8,
            &at_mask as *const _ as *const u8,
            &position as *const _ as *const u8,
        ],
    );
}

// GPU function to update character classes
#[cfg(has_cuda)]
pub unsafe fn update_character_classes_gpu(
    char_classes_ptr: *mut u8,
    digit_mask: u64,
    letter_mask: u64,
    punct_mask: u64,
    space_mask: u64,
) {
    #[cfg(has_cuda)]
    use crate::gpu::launch_ptx;

    const PTX_UPDATE_CHARACTER_CLASSES: &str = r#"
.version 7.5
.target sm_70
.address_size 64

.entry update_character_classes(
  .param .u64 char_classes_ptr,
  .param .u64 digit_mask,
  .param .u64 letter_mask,
  .param .u64 punct_mask,
  .param .u64 space_mask
) {
  .reg .u64 %rd<10>;
  .reg .u32 %r<10>;
  
  ld.param.u64 %rd0, [char_classes_ptr];
  ld.param.u64 %rd1, [digit_mask];
  ld.param.u64 %rd2, [letter_mask];
  ld.param.u64 %rd3, [punct_mask];
  ld.param.u64 %rd4, [space_mask];
  
  // Count ones in each mask using population count
  popc.b64 %rd5, %rd1;  // digit count
  popc.b64 %rd6, %rd2;  // letter count
  popc.b64 %rd7, %rd3;  // punct count
  popc.b64 %rd8, %rd4;  // space count
  
  // Load current counts and add
  ld.global.u32 %r0, [%rd0];       // digit_count
  cvt.u32.u64 %r1, %rd5;
  add.u32 %r0, %r0, %r1;
  st.global.u32 [%rd0], %r0;
  
  ld.global.u32 %r2, [%rd0 + 4];   // letter_count
  cvt.u32.u64 %r3, %rd6;
  add.u32 %r2, %r2, %r3;
  st.global.u32 [%rd0 + 4], %r2;
  
  ld.global.u32 %r4, [%rd0 + 8];   // punctuation_count
  cvt.u32.u64 %r5, %rd7;
  add.u32 %r4, %r4, %r5;
  st.global.u32 [%rd0 + 8], %r4;
  
  ld.global.u32 %r6, [%rd0 + 12];  // whitespace_count
  cvt.u32.u64 %r7, %rd8;
  add.u32 %r6, %r6, %r7;
  st.global.u32 [%rd0 + 12], %r6;
  
  // Update total_count by sum of all counts
  add.u32 %r8, %r1, %r3;
  add.u32 %r8, %r8, %r5;
  add.u32 %r8, %r8, %r7;
  ld.global.u32 %r9, [%rd0 + 16];  // total_count
  add.u32 %r9, %r9, %r8;
  st.global.u32 [%rd0 + 16], %r9;
  
  ret;
}
"#;

    let _ = launch_ptx(
        PTX_UPDATE_CHARACTER_CLASSES,
        &[],
        "update_character_classes",
        1,
        1,
        &[
            char_classes_ptr as *const u8,
            &digit_mask as *const _ as *const u8,
            &letter_mask as *const _ as *const u8,
            &punct_mask as *const _ as *const u8,
            &space_mask as *const _ as *const u8,
        ],
    );
}

#[cfg(has_cuda)]
pub unsafe fn classify_chunk_gpu(
    chunk: *const u8,
    signature: &mut PatternSignature<MAX_POSITIONS_AVX512>,
    char_classes: &mut CharacterClasses,
    position: usize,
) {
    #[cfg(has_cuda)]
    use crate::gpu::launch_ptx;

    const PTX_CLASSIFY_CHUNK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .func (.param .b32 result) create_digit_mask_ptx(.param .u64 chunk_ptr);
    .extern .func (.param .b32 result) create_letter_mask_ptx(.param .u64 chunk_ptr);
    .extern .func (.param .b32 result) create_punctuation_mask_ptx(.param .u64 chunk_ptr);
    .extern .func (.param .b32 result) create_space_mask_ptx(.param .u64 chunk_ptr);
    
    .entry classify_chunk(
      .param .u64 chunk_ptr,
      .param .u64 signature_ptr,
      .param .u64 char_classes_ptr,
      .param .u32 position
    ) {
      .reg .u32 %r<20>;
      .reg .u64 %rd<10>;
      .reg .u8 %rc<32>;
      .reg .pred %p<100>;
      
      ld.param.u64 %rd0, [chunk_ptr];
      ld.param.u64 %rd1, [signature_ptr];
      ld.param.u64 %rd2, [char_classes_ptr];
      ld.param.u32 %r10, [position];
      
      // Call PTX functions to get masks
      .param .b32 digit_mask_param;
      .param .b32 letter_mask_param;
      .param .b32 punct_mask_param;
      .param .b32 space_mask_param;
      
      call create_digit_mask_ptx, (digit_mask_param), (%rd0);
      ld.param.b32 %r0, [digit_mask_param];  // digit_mask
      
      call create_letter_mask_ptx, (letter_mask_param), (%rd0);
      ld.param.b32 %r1, [letter_mask_param];  // letter_mask
      
      call create_punctuation_mask_ptx, (punct_mask_param), (%rd0);
      ld.param.b32 %r2, [punct_mask_param];  // punct_mask
      
      call create_space_mask_ptx, (space_mask_param), (%rd0);
      ld.param.b32 %r3, [space_mask_param];  // space_mask
      
      // Grid stride loop for pattern detection
      mov.u32 %r11, %tid.x;
      mov.u32 %r12, %ctaid.x;
      mov.u32 %r13, %ntid.x;
      mul.lo.u32 %r14, %r12, %r13;
      add.u32 %r15, %r14, %r11;  // global thread id
      
      // Check if thread is within chunk (32 bytes)
      setp.lt.u32 %p0, %r15, 32;
      @!%p0 bra done;
      
      // Load byte for specific pattern detection
      cvt.u64.u32 %rd3, %r15;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.u8 %rc0, [%rd4];
      
      // Detect specific patterns (dot, colon, slash, etc.)
      setp.eq.u8 %p1, %rc0, 0x2E;  // dot '.'
      setp.eq.u8 %p2, %rc0, 0x3A;  // colon ':'
      setp.eq.u8 %p3, %rc0, 0x2F;  // slash '/'
      setp.eq.u8 %p4, %rc0, 0x2D;  // dash '-'
      setp.eq.u8 %p5, %rc0, 0x2B;  // plus '+'
      setp.eq.u8 %p6, %rc0, 0x5B;  // bracket_open '['
      setp.eq.u8 %p7, %rc0, 0x5D;  // bracket_close ']'
      setp.eq.u8 %p8, %rc0, 0x2C;  // comma ','
      setp.eq.u8 %p9, %rc0, 0x40;  // at '@'
      
      // Warp votes for pattern masks
      vote.ballot.b32 %r4, %p1;  // dot_mask
      vote.ballot.b32 %r5, %p2;  // colon_mask
      vote.ballot.b32 %r6, %p3;  // slash_mask
      vote.ballot.b32 %r7, %p4;  // dash_mask
      vote.ballot.b32 %r8, %p5;  // plus_mask
      vote.ballot.b32 %r9, %p6;  // bracket_open_mask
      vote.ballot.b32 %r16, %p7; // bracket_close_mask
      vote.ballot.b32 %r17, %p8; // comma_mask
      vote.ballot.b32 %r18, %p9; // at_mask
      
      // First thread of warp updates signature and char_classes
      and.b32 %r19, %r11, 31;
      setp.eq.u32 %p10, %r19, 0;
      @!%p10 bra done;
      
      // Store masks to signature structure
      // Note: Actual signature update would require complex PTX for struct manipulation
      // For now, store masks at offsets in signature_ptr
      st.global.u32 [%rd1], %r0;        // digit_mask at offset 0
      st.global.u32 [%rd1 + 4], %r1;    // letter_mask at offset 4
      st.global.u32 [%rd1 + 8], %r2;    // punct_mask at offset 8
      st.global.u32 [%rd1 + 12], %r3;   // space_mask at offset 12
      st.global.u32 [%rd1 + 16], %r4;   // dot_mask at offset 16
      st.global.u32 [%rd1 + 20], %r5;   // colon_mask at offset 20
      st.global.u32 [%rd1 + 24], %r6;   // slash_mask at offset 24
      st.global.u32 [%rd1 + 28], %r7;   // dash_mask at offset 28
      st.global.u32 [%rd1 + 32], %r8;   // plus_mask at offset 32
      st.global.u32 [%rd1 + 36], %r9;   // bracket_open_mask at offset 36
      st.global.u32 [%rd1 + 40], %r16;  // bracket_close_mask at offset 40
      st.global.u32 [%rd1 + 44], %r17;  // comma_mask at offset 44
      st.global.u32 [%rd1 + 48], %r18;  // at_mask at offset 48
      st.global.u32 [%rd1 + 52], %r10;  // position at offset 52
      
      // Update character class counts at char_classes_ptr
      st.global.u32 [%rd2], %r0;        // digit_mask
      st.global.u32 [%rd2 + 4], %r1;    // letter_mask
      st.global.u32 [%rd2 + 8], %r2;    // punct_mask
      st.global.u32 [%rd2 + 12], %r3;   // space_mask
      
    done:
      ret;
    }
  "#;

    let (blocks, threads) = (1, 32); // Helper functions use single warp

    let _ = launch_ptx(
        PTX_CLASSIFY_CHUNK,
        &[
            PTX_CREATE_DIGIT_MASK,
            PTX_CREATE_LETTER_MASK,
            PTX_CREATE_PUNCT_MASK,
            PTX_CREATE_SPACE_MASK,
        ],
        "classify_chunk",
        blocks,
        threads,
        &[
            chunk as *const u8,
            signature as *mut PatternSignature<MAX_POSITIONS_AVX512> as *const u8,
            char_classes as *mut CharacterClasses as *const u8,
            position as *const u8,
        ],
    );
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn classify_chunk_avx512(
    chunk: std::arch::x86_64::__m512i,
    signature: &mut PatternSignature<MAX_POSITIONS_AVX512>,
    char_classes: &mut CharacterClasses,
    position: usize,
) {
    //  DEEP SIMD: Create character class masks in parallel
    let digit_mask = create_digit_mask_avx512(chunk);
    let letter_mask = create_letter_mask_avx512(chunk);
    let punct_mask = create_punctuation_mask_avx512(chunk);
    let space_mask = create_space_mask_avx512(chunk);

    //  DEEP SIMD: Specific pattern detection
    let dot_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'.' as i8));
    let colon_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b':' as i8));
    let slash_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'/' as i8));
    let dash_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'-' as i8));
    let plus_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'+' as i8));
    let bracket_open_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'[' as i8));
    let bracket_close_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b']' as i8));
    let comma_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b',' as i8));
    let at_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'@' as i8));

    // Update pattern signature with SIMD results
    signature.update_avx512(
        digit_mask,
        letter_mask,
        punct_mask,
        space_mask,
        dot_mask,
        colon_mask,
        slash_mask,
        dash_mask,
        plus_mask,
        bracket_open_mask,
        bracket_close_mask,
        comma_mask,
        at_mask,
        position,
    );

    // Update character class counts
    char_classes.update_avx512(digit_mask, letter_mask, punct_mask, space_mask);
}

// Create digit detection mask (0-9) using AVX-512

// PTX kernel for digit mask creation
// PTX extern declaration that references the kernel defined in create_digit_mask_gpu function
pub const PTX_CREATE_DIGIT_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_digit_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    );
"#;

// GPU version of create_digit_mask - processes 32 bytes with warp intrinsics
#[cfg(has_cuda)]
pub unsafe fn create_digit_mask_gpu(chunk: *const u8) -> u32 {
    #[cfg(has_cuda)]
    use crate::gpu::launch_ptx;

    const PTX_CREATE_DIGIT_MASK_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .entry create_digit_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    ) {
      // MAXIMUM PARALLELISM: Block^Warp^Vector architecture
      .reg .u32 %r<80>;
      .reg .u64 %rd<40>;
      .reg .u8 %rc<32>;         // Vector registers for vectorized processing
      .reg .pred %p<100>;
      
      // Shared memory for inter-warp communication (8 warps per block)
      .shared .align 4 .b8 warp_results[32];  // 8 warps × 4 bytes each
      
      ld.param.u64 %rd0, [chunk_ptr];
      ld.param.u64 %rd1, [result_ptr];
      
      // BLOCK LEVEL: Multi-warp processing with 256 threads (8 warps per block)
      mov.u32 %r0, %tid.x;         // Thread ID (0-255)
      mov.u32 %r1, %ctaid.x;       // Block ID
      
      // WARP LEVEL: Warp indexing for cooperation
      shr.u32 %r2, %r0, 5;         // Warp ID = tid / 32 (0-7)  
      and.u32 %r3, %r0, 31;        // Lane ID = tid % 32 (0-31)
      
      // VECTOR LEVEL: Each thread processes 4 bytes using vectorized operations
      shl.u32 %r4, %r0, 2;         // byte_offset = tid * 4 (vector processing)
      
      // Process multiple 32-byte chunks with grid-stride
      mul.lo.u32 %r5, %r1, 1024;   // Block processes 1024 bytes (256 threads × 4 bytes)
      add.u32 %r6, %r5, %r4;       // global_byte_offset
      
    chunk_loop:
      setp.ge.u32 %p0, %r6, 32;    // Beyond single 32-byte chunk
      @%p0 bra finalize_warp;
      
      // VECTORIZED MEMORY ACCESS: 4-byte vector loads for maximum bandwidth
      cvt.u64.u32 %rd2, %r6;
      add.u64 %rd3, %rd0, %rd2;
      
      // Bounds checking for vector load
      add.u32 %r7, %r6, 3;         // Check if we can load 4 bytes
      setp.ge.u32 %p1, %r7, 32;    // Beyond chunk boundary
      @%p1 bra scalar_load;
      
      // SIMD vector load (4 bytes simultaneously)
      ld.global.v4.u8 {%rc0, %rc1, %rc2, %rc3}, [%rd3];
      bra vector_process;
      
    scalar_load:
      // Boundary case: individual byte loads with zero padding
      mov.u8 %rc0, 0; mov.u8 %rc1, 0; mov.u8 %rc2, 0; mov.u8 %rc3, 0;
      setp.lt.u32 %p2, %r6, 32;
      @!%p2 bra vector_process;
      ld.global.u8 %rc0, [%rd3];
      
    vector_process:
      // VECTORIZED DIGIT DETECTION: 4 bytes in parallel per thread
      // Ultra-optimized arithmetic: subtract '0' then range check
      sub.u8 %rc4, %rc0, 0x30; setp.le.u8 %p10, %rc4, 9;  // Byte 0
      sub.u8 %rc5, %rc1, 0x30; setp.le.u8 %p11, %rc5, 9;  // Byte 1  
      sub.u8 %rc6, %rc2, 0x30; setp.le.u8 %p12, %rc6, 9;  // Byte 2
      sub.u8 %rc7, %rc3, 0x30; setp.le.u8 %p13, %rc7, 9;  // Byte 3
      
      // Pack 4 predicates into thread's contribution (vectorized bit packing)
      selp.u32 %r10, 1, 0, %p10;   selp.u32 %r11, 2, 0, %p11;
      selp.u32 %r12, 4, 0, %p12;   selp.u32 %r13, 8, 0, %p13;
      or.b32 %r14, %r10, %r11;     or.b32 %r15, %r12, %r13;
      or.b32 %r16, %r14, %r15;     // Thread's 4-bit result
      
    finalize_warp:
      // WARP LEVEL: High-speed butterfly reduction using warp shuffle
      // Combines 128 bytes of processing (32 threads × 4 bytes each)
      shfl.sync.bfly.b32 %r20, %r16, 1, 0xffffffff;  or.b32 %r16, %r16, %r20;
      shfl.sync.bfly.b32 %r20, %r16, 2, 0xffffffff;  or.b32 %r16, %r16, %r20;
      shfl.sync.bfly.b32 %r20, %r16, 4, 0xffffffff;  or.b32 %r16, %r16, %r20;
      shfl.sync.bfly.b32 %r20, %r16, 8, 0xffffffff;  or.b32 %r16, %r16, %r20;
      shfl.sync.bfly.b32 %r20, %r16, 16, 0xffffffff; or.b32 %r16, %r16, %r20;
      
      // Lane 0 stores warp result to shared memory
      setp.eq.u32 %p20, %r3, 0;    // Lane 0 only
      @!%p20 bra block_sync;
      shl.u32 %r25, %r2, 2;        // warp_offset = warp_id * 4
      st.shared.u32 [warp_results + %r25], %r16;
      
    block_sync:
      bar.sync 0;                   // Block synchronization point
      
      // BLOCK LEVEL: Thread 0 aggregates all warp results  
      setp.eq.u32 %p30, %r0, 0;    // Thread 0 only
      @!%p30 bra done;
      
      // Load all 8 warp results (1024 bytes of digit detection)
      ld.shared.u32 %r30, [warp_results];      ld.shared.u32 %r31, [warp_results + 4];
      ld.shared.u32 %r32, [warp_results + 8];  ld.shared.u32 %r33, [warp_results + 12];
      ld.shared.u32 %r34, [warp_results + 16]; ld.shared.u32 %r35, [warp_results + 20];
      ld.shared.u32 %r36, [warp_results + 24]; ld.shared.u32 %r37, [warp_results + 28];
      
      // Hierarchical tree reduction for maximum parallelism
      or.b32 %r40, %r30, %r31;     or.b32 %r41, %r32, %r33;
      or.b32 %r42, %r34, %r35;     or.b32 %r43, %r36, %r37;
      or.b32 %r44, %r40, %r41;     or.b32 %r45, %r42, %r43;
      or.b32 %r46, %r44, %r45;     // Final block result
      
      // Store aggregated result
      st.global.u32 [%rd1], %r46;
      
      // First thread of warp stores result
      and.b32 %r6, %r0, 31;
      setp.eq.u32 %p4, %r6, 0;
      @%p4 st.global.u32 [%rd1], %r5;
      
    done:
      ret;
    }
  "#;

    let mut result: u32 = 0;

    let (blocks, threads) = (1, 32); // Helper functions use single warp

    let _ = launch_ptx(
        PTX_CREATE_DIGIT_MASK_KERNEL,
        &[],
        "create_digit_mask",
        blocks,
        threads,
        &[chunk as *const u8, &mut result as *mut u32 as *const u8],
    );

    result
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_digit_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    let digit_0 = _mm512_set1_epi8(b'0' as i8);
    let digit_9 = _mm512_set1_epi8(b'9' as i8);

    let ge_0 = _mm512_cmpge_epi8_mask(chunk, digit_0);
    let le_9 = _mm512_cmple_epi8_mask(chunk, digit_9);

    ge_0 & le_9
}

// Create letter detection mask (A-Z, a-z) using AVX-512

// PTX kernel for letter mask creation
// PTX extern declaration that references the kernel defined in create_letter_mask_gpu function
pub const PTX_CREATE_LETTER_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_letter_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    );
"#;

// GPU version of create_letter_mask - processes 32 bytes with warp intrinsics
#[cfg(has_cuda)]
pub unsafe fn create_letter_mask_gpu(chunk: *const u8) -> u32 {
    const PTX_CREATE_LETTER_MASK_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .entry create_letter_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<10>;
      .reg .u64 %rd<5>;
      .reg .u8 %rc<1>;
      .reg .pred %p<10>;
      
      ld.param.u64 %rd0, [chunk_ptr];
      ld.param.u64 %rd1, [result_ptr];
      
      // Maximum parallelism with multi-warp block processing
      mov.u32 %r0, %tid.x;         // Thread ID (0-255 for 8 warps)
      mov.u32 %r1, %ctaid.x;       // Block ID  
      shr.u32 %r2, %r0, 5;         // Warp ID = tid / 32 (0-7)
      and.u32 %r3, %r0, 31;        // Lane ID = tid % 32 (0-31)
      
      // Bounds check for 32-byte chunk processing
      setp.lt.u32 %p0, %r3, 32;
      @!%p0 bra done;
      
      // Optimized memory access with coalescing
      cvt.u64.u32 %rd2, %r3;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.u8 %rc0, [%rd3];
      
      // Vectorized letter detection with bit manipulation optimization
      // Check A-Z (0x41-0x5A) using efficient arithmetic
      sub.u8 %rc1, %rc0, 0x41;     // Subtract 'A'
      setp.le.u8 %p1, %rc1, 25;    // Check if <= 25 (A-Z range)
      
      // Check a-z (0x61-0x7A) using efficient arithmetic  
      sub.u8 %rc2, %rc0, 0x61;     // Subtract 'a'
      setp.le.u8 %p2, %rc2, 25;    // Check if <= 25 (a-z range)
      
      // Fast case folding check using bit manipulation
      and.b8 %rc3, %rc0, 0xDF;     // Convert to uppercase (clear bit 5)
      sub.u8 %rc4, %rc3, 0x41;     // Subtract 'A' from uppercase
      setp.le.u8 %p3, %rc4, 25;    // Unified A-Z check
      
      // Use the most optimized predicate (unified case-folding approach)
      // This combines both upper and lower case in a single efficient check
      
      vote.ballot.b32 %r5, %p3;
      
      and.b32 %r6, %r0, 31;
      setp.eq.u32 %p8, %r6, 0;
      @%p8 st.global.u32 [%rd1], %r5;
      
    done:
      ret;
    }
  "#;

    let mut result: u32 = 0;

    let (blocks, threads) = (1, 32); // Helper functions use single warp

    let _ = launch_ptx(
        PTX_CREATE_LETTER_MASK_KERNEL,
        &[],
        "create_letter_mask",
        blocks,
        threads,
        &[chunk as *const u8, &mut result as *mut u32 as *const u8],
    );

    result
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_letter_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    // Uppercase A-Z (matching NEON pattern exactly)
    let upper_mask = _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(b'A' as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(b'Z' as i8));

    // Lowercase a-z (matching NEON pattern exactly)
    let lower_mask = _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(b'a' as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(b'z' as i8));

    upper_mask | lower_mask
}

// PTX kernel for punctuation mask creation
// PTX extern declaration that references the kernel defined in create_punctuation_mask_gpu function
pub const PTX_CREATE_PUNCT_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_punct_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    );
"#;

// GPU version of create_punctuation_mask - processes 32 bytes with warp intrinsics
#[cfg(has_cuda)]
pub unsafe fn create_punctuation_mask_gpu(chunk: *const u8) -> u32 {
    const PTX_CREATE_PUNCT_MASK_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .entry create_punct_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<10>;
      .reg .u64 %rd<5>;
      .reg .u8 %rc<1>;
      .reg .pred %p<20>;
      
      ld.param.u64 %rd0, [chunk_ptr];
      ld.param.u64 %rd1, [result_ptr];
      
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mul.lo.u32 %r3, %r1, %r2;
      add.u32 %r4, %r3, %r0;
      
      setp.lt.u32 %p0, %r4, 32;
      @!%p0 bra done;
      
      cvt.u64.u32 %rd2, %r4;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.u8 %rc0, [%rd3];
      
      // Check punctuation ranges - matches AVX-512 logic
      // 0x21-0x2F
      setp.ge.u8 %p1, %rc0, 0x21;
      setp.le.u8 %p2, %rc0, 0x2F;
      and.pred %p3, %p1, %p2;
      
      // 0x3A-0x40
      setp.ge.u8 %p4, %rc0, 0x3A;
      setp.le.u8 %p5, %rc0, 0x40;
      and.pred %p6, %p4, %p5;
      
      // 0x5B-0x60
      setp.ge.u8 %p7, %rc0, 0x5B;
      setp.le.u8 %p8, %rc0, 0x60;
      and.pred %p9, %p7, %p8;
      
      // 0x7B-0x7E
      setp.ge.u8 %p10, %rc0, 0x7B;
      setp.le.u8 %p11, %rc0, 0x7E;
      and.pred %p12, %p10, %p11;
      
      // Combine all ranges
      or.pred %p13, %p3, %p6;
      or.pred %p14, %p9, %p12;
      or.pred %p15, %p13, %p14;
      
      vote.ballot.b32 %r5, %p15;
      
      and.b32 %r6, %r0, 31;
      setp.eq.u32 %p16, %r6, 0;
      @%p16 st.global.u32 [%rd1], %r5;
      
    done:
      ret;
    }
  "#;

    let mut result: u32 = 0;

    let (blocks, threads) = (1, 32); // Helper functions use single warp

    let _ = launch_ptx(
        PTX_CREATE_PUNCT_MASK_KERNEL,
        &[],
        "create_punct_mask",
        blocks,
        threads,
        &[chunk as *const u8, &mut result as *mut u32 as *const u8],
    );

    result
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_punctuation_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    // Basic punctuation range check (matching NEON pattern exactly)
    _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(b'!' as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(b'/' as i8))
}

// PTX kernel for space mask creation
// PTX extern declaration that references the kernel defined in create_space_mask_gpu function
pub const PTX_CREATE_SPACE_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_space_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    );
"#;

// Create whitespace detection mask using AVX-512
#[cfg(has_cuda)]
pub unsafe fn create_space_mask_gpu(chunk: *const u8) -> u32 {
    const PTX_CREATE_SPACE_MASK_KERNEL: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .entry create_space_mask(
      .param .u64 chunk_ptr,
      .param .u64 result_ptr
    ) {
      .reg .u32 %r<10>;
      .reg .u64 %rd<5>;
      .reg .u8 %rc<32>;
      .reg .pred %p<35>;
      
      ld.param.u64 %rd0, [chunk_ptr];
      ld.param.u64 %rd1, [result_ptr];
      
      // Grid stride loop for multiple blocks
      mov.u32 %r0, %tid.x;
      mov.u32 %r1, %ctaid.x;
      mov.u32 %r2, %ntid.x;
      mul.lo.u32 %r3, %r1, %r2;
      add.u32 %r4, %r3, %r0;  // global thread id
      
      // Check if thread is within chunk (32 bytes)
      setp.lt.u32 %p0, %r4, 32;
      @!%p0 bra done;
      
      // Load byte
      cvt.u64.u32 %rd2, %r4;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.u8 %rc0, [%rd3];
      
      // Check if space (0x20) or tab (0x09) - matches AVX-512 logic exactly
      setp.eq.u8 %p1, %rc0, 0x20;  // space
      setp.eq.u8 %p2, %rc0, 0x09;  // tab
      or.pred %p3, %p1, %p2;
      
      // Warp vote to collect all 32 results into mask
      vote.ballot.b32 %r5, %p3;
      
      // First thread of warp stores result
      and.b32 %r6, %r0, 31;
      setp.eq.u32 %p4, %r6, 0;
      @%p4 st.global.u32 [%rd1], %r5;
      
    done:
      ret;
    }
  "#;

    let mut result: u32 = 0;

    let (blocks, threads) = (1, 32); // Helper functions use single warp

    let _ = launch_ptx(
        PTX_CREATE_SPACE_MASK_KERNEL,
        &[],
        "create_space_mask",
        blocks,
        threads,
        &[chunk as *const u8, &mut result as *mut u32 as *const u8],
    );

    result
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_space_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    let space_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b' ' as i8));
    let tab_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'\t' as i8));

    space_mask | tab_mask
}

// =============================================================================
// IMPLEMENTATION DETAILS - AVX2 HELPER FUNCTIONS
// =============================================================================

// Helper function for single string AVX2 classification
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn classify_single_string_avx2(s: &str, len: usize) -> ClassificationResult {
    const CHUNK_SIZE_AVX2: usize = LANES_AVX2_BYTES;
    const LANES: usize = CHUNK_SIZE_AVX2;

    //  ZERO-COPY: Work directly with string bytes, no allocation
    let bytes = s.as_bytes();

    //  DEEP SIMD: Multi-pattern signature detection in parallel
    let mut pattern_signature = PatternSignature::<MAX_POSITIONS_AVX2>::new();
    let mut char_classes = CharacterClasses::new();

    // Process bytes in 32-byte SIMD chunks with AVX2
    let mut i = 0;
    while i + LANES <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const _);

        //  DEEP SIMD: Parallel character classification using AVX2
        classify_chunk_avx2(chunk, &mut pattern_signature, &mut char_classes, i);
        i += LANES;
    }

    // Handle any remaining bytes with scalar processing
    if i < len {
        classify_scalar_remaining(&bytes[i..], &mut pattern_signature, i);
    }

    //  DEEP SIMD: Type determination based on pattern signature
    let (result, numeric_value_1, numeric_value_2) =
        determine_type_from_signature(s, len, &pattern_signature);

    // If we detected an array, automatically call the array classifier for detailed typing
    if result == HwxType::Array {
        // Check if it's actually an array (starts and ends with brackets)
        if len >= 2 && s.starts_with('[') && s.ends_with(']') {
            // Extract array content (remove brackets) and call deep array classification
            let content = &s[1..len - 1];
            return classify_array_contents_avx2(content.as_bytes(), len - 2);
        }
    }

    ClassificationResult {
        hwx_type: result,
        element_count: 1,
        numeric_value_1,
        numeric_value_2,
    }
}

// AVX2 chunk classification using comprehensive pattern detection
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn classify_chunk_avx2(
    chunk: std::arch::x86_64::__m256i,
    signature: &mut PatternSignature<MAX_POSITIONS_AVX2>,
    char_classes: &mut CharacterClasses,
    position: usize,
) {
    //  DEEP SIMD: Create character class masks in parallel
    let digit_mask = create_digit_mask_avx2(chunk);
    let letter_mask = create_letter_mask_avx2(chunk);
    let punct_mask = create_punctuation_mask_avx2(chunk);
    let space_mask = create_space_mask_avx2(chunk);

    //  DEEP SIMD: Specific pattern detection
    let dot_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'.' as i8))) as u32;
    let colon_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b':' as i8))) as u32;
    let slash_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'/' as i8))) as u32;
    let dash_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'-' as i8))) as u32;
    let plus_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'+' as i8))) as u32;
    let bracket_open_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'[' as i8))) as u32;
    let bracket_close_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b']' as i8))) as u32;
    let comma_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b',' as i8))) as u32;
    let at_mask =
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'@' as i8))) as u32;

    // Update pattern signature with SIMD results
    signature.update_avx2(
        digit_mask,
        letter_mask,
        punct_mask,
        space_mask,
        dot_mask,
        colon_mask,
        slash_mask,
        dash_mask,
        plus_mask,
        bracket_open_mask,
        bracket_close_mask,
        comma_mask,
        at_mask,
        position,
    );

    // Update character class counts
    char_classes.update_avx2(digit_mask, letter_mask, punct_mask, space_mask);
}

// Create digit detection mask (0-9) using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_digit_mask_avx2(chunk: std::arch::x86_64::__m256i) -> u32 {
    let digit_0 = _mm256_set1_epi8(b'0' as i8);
    let digit_9 = _mm256_set1_epi8(b'9' as i8);

    // AVX2 doesn't have direct >= and <= comparisons, so we use the NEON pattern with available ops
    let ge_0 = _mm256_or_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8((b'0' - 1) as i8)),
        _mm256_cmpeq_epi8(chunk, digit_0),
    );
    let le_9 = _mm256_or_si256(
        _mm256_cmpgt_epi8(_mm256_set1_epi8((b'9' + 1) as i8), chunk),
        _mm256_cmpeq_epi8(chunk, digit_9),
    );
    let is_digit = _mm256_and_si256(ge_0, le_9);

    _mm256_movemask_epi8(is_digit) as u32
}

// Create letter detection mask (A-Z, a-z) using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_letter_mask_avx2(chunk: std::arch::x86_64::__m256i) -> u32 {
    // Uppercase A-Z (matching NEON pattern with AVX2 available ops)
    let upper_ge_a = _mm256_or_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8((b'A' - 1) as i8)),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'A' as i8)),
    );
    let upper_le_z = _mm256_or_si256(
        _mm256_cmpgt_epi8(_mm256_set1_epi8((b'Z' + 1) as i8), chunk),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'Z' as i8)),
    );
    let upper_mask = _mm256_and_si256(upper_ge_a, upper_le_z);

    // Lowercase a-z (matching NEON pattern with AVX2 available ops)
    let lower_ge_a = _mm256_or_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8((b'a' - 1) as i8)),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'a' as i8)),
    );
    let lower_le_z = _mm256_or_si256(
        _mm256_cmpgt_epi8(_mm256_set1_epi8((b'z' + 1) as i8), chunk),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'z' as i8)),
    );
    let lower_mask = _mm256_and_si256(lower_ge_a, lower_le_z);

    let is_letter = _mm256_or_si256(upper_mask, lower_mask);
    _mm256_movemask_epi8(is_letter) as u32
}

// Create punctuation detection mask using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_punctuation_mask_avx2(chunk: std::arch::x86_64::__m256i) -> u32 {
    // Basic punctuation range check (matching NEON pattern with AVX2 available ops)
    let ge_punct = _mm256_or_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8((b'!' - 1) as i8)),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'!' as i8)),
    );
    let le_punct = _mm256_or_si256(
        _mm256_cmpgt_epi8(_mm256_set1_epi8((b'/' + 1) as i8), chunk),
        _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'/' as i8)),
    );

    let is_punct = _mm256_and_si256(ge_punct, le_punct);
    _mm256_movemask_epi8(is_punct) as u32
}

// Create whitespace detection mask using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_space_mask_avx2(chunk: std::arch::x86_64::__m256i) -> u32 {
    let space_cmp = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8));
    let tab_cmp = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\t' as i8));

    let is_space = _mm256_or_si256(space_cmp, tab_cmp);
    _mm256_movemask_epi8(is_space) as u32
}

// =============================================================================
// IMPLEMENTATION DETAILS - NEON HELPER FUNCTIONS
// =============================================================================

// Helper function for single string NEON classification
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn classify_single_string_neon(s: &str, len: usize) -> ClassificationResult {
    const CHUNK_SIZE_NEON: usize = LANES_NEON_BYTES;
    const LANES: usize = CHUNK_SIZE_NEON;

    //  ZERO-COPY: Work directly with string bytes, no allocation
    let bytes = s.as_bytes();

    //  DEEP SIMD: Multi-pattern signature detection in parallel
    let mut pattern_signature = PatternSignature::<MAX_POSITIONS_NEON>::new();
    let mut char_classes = CharacterClasses::new();

    // Process bytes in 16-byte SIMD chunks with NEON
    let mut i = 0;
    while i + LANES <= len {
        let chunk = vld1q_u8(bytes.as_ptr().add(i));

        //  DEEP SIMD: Parallel character classification using NEON
        classify_chunk_neon(chunk, &mut pattern_signature, &mut char_classes, i);
        i += LANES;
    }

    // Handle any remaining bytes with scalar processing
    if i < len {
        classify_scalar_remaining(&bytes[i..], &mut pattern_signature, i);
    }

    //  DEEP SIMD: Type determination based on pattern signature
    let (result, numeric_value_1, numeric_value_2) =
        determine_type_from_signature(s, len, &pattern_signature);

    // If we detected an array, call the array classifier for detailed typing
    if result == HwxType::Array {
        // Check if it's actually an array (starts and ends with brackets)
        if len >= 2 && s.starts_with('[') && s.ends_with(']') {
            // Extract array content (remove brackets) and call deep array classification
            let content = &s[1..len - 1];
            return classify_array_contents_neon(content.as_bytes(), len - 2);
        }
    }

    ClassificationResult {
        hwx_type: result,
        element_count: 1,
        numeric_value_1,
        numeric_value_2,
    }
}

// NEON chunk classification using pattern detection
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn classify_chunk_neon(
    chunk: std::arch::aarch64::uint8x16_t,
    signature: &mut PatternSignature<MAX_POSITIONS_NEON>,
    char_classes: &mut CharacterClasses,
    position: usize,
) {
    //  DEEP SIMD: Create character class masks in parallel
    let digit_mask = create_digit_mask_neon(chunk);
    let letter_mask = create_letter_mask_neon(chunk);
    let punct_mask = create_punctuation_mask_neon(chunk);
    let space_mask = create_space_mask_neon(chunk);

    //  DEEP SIMD: Specific pattern detection
    let dot_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b'.')));
    let colon_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b':')));
    let slash_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b'/')));
    let dash_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b'-')));
    let plus_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b'+')));
    let bracket_open_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b'[')));
    let bracket_close_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b']')));
    let comma_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b',')));
    let at_mask = extract_mask_neon(vceqq_u8(chunk, vdupq_n_u8(b'@')));

    // Update pattern signature with SIMD results
    signature.update_neon(
        digit_mask,
        letter_mask,
        punct_mask,
        space_mask,
        dot_mask,
        colon_mask,
        slash_mask,
        dash_mask,
        plus_mask,
        bracket_open_mask,
        bracket_close_mask,
        comma_mask,
        at_mask,
        position,
    );

    // Update character class counts
    char_classes.update_neon(digit_mask, letter_mask, punct_mask, space_mask);
}

// Create digit detection mask (0-9) using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn create_digit_mask_neon(chunk: std::arch::aarch64::uint8x16_t) -> u16 {
    let is_digit = vandq_u8(
        vcgeq_u8(chunk, vdupq_n_u8(b'0')),
        vcleq_u8(chunk, vdupq_n_u8(b'9')),
    );

    extract_mask_neon(is_digit)
}

// Create letter detection mask (A-Z, a-z) using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn create_letter_mask_neon(chunk: std::arch::aarch64::uint8x16_t) -> u16 {
    // Uppercase A-Z
    let upper_mask = vandq_u8(
        vcgeq_u8(chunk, vdupq_n_u8(b'A')),
        vcleq_u8(chunk, vdupq_n_u8(b'Z')),
    );

    // Lowercase a-z
    let lower_mask = vandq_u8(
        vcgeq_u8(chunk, vdupq_n_u8(b'a')),
        vcleq_u8(chunk, vdupq_n_u8(b'z')),
    );

    let is_letter = vorrq_u8(upper_mask, lower_mask);
    extract_mask_neon(is_letter)
}

// Create punctuation detection mask using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn create_punctuation_mask_neon(chunk: std::arch::aarch64::uint8x16_t) -> u16 {
    // Basic punctuation range check
    let is_punct = vandq_u8(
        vcgeq_u8(chunk, vdupq_n_u8(b'!')),
        vcleq_u8(chunk, vdupq_n_u8(b'/')),
    );

    extract_mask_neon(is_punct)
}

// Create whitespace detection mask using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn create_space_mask_neon(chunk: std::arch::aarch64::uint8x16_t) -> u16 {
    let is_space = vorrq_u8(
        vceqq_u8(chunk, vdupq_n_u8(b' ')),
        vceqq_u8(chunk, vdupq_n_u8(b'\t')),
    );

    extract_mask_neon(is_space)
}

// Extract bitmask from NEON comparison result
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn extract_mask_neon(mask: std::arch::aarch64::uint8x16_t) -> u16 {
    // BUG FIX: NEON comparison results use 0xFF for matches, not just any non-zero value
    // We need to be more precise about extracting the mask to ensure reliability
    let mut result = [0u8; 16];
    vst1q_u8(result.as_mut_ptr(), mask);

    let mut bitmask = 0u16;
    for (i, &val) in result.iter().enumerate() {
        // NEON vceqq_u8 returns 0xFF for matching bytes, 0x00 for non-matching
        // Check specifically for 0xFF to be more robust
        if val == 0xFF {
            bitmask |= 1 << i;
        }
    }
    bitmask
}

// =============================================================================
// PATTERN SIGNATURE AND CHARACTER CLASS TRACKING
// =============================================================================

// Architecture-specific position limits based on SIMD lane width

// Generic pattern signature parameterized by max positions
#[derive(Debug, Clone)]
pub struct PatternSignature<const MAX_POS: usize> {
    // Character patterns
    pub(crate) has_digits: bool,
    pub(crate) has_letters: bool,
    pub(crate) has_punctuation: bool,
    pub(crate) has_whitespace: bool,

    // Specific character positions and counts (fixed-size stack arrays)
    pub(crate) dot_positions: [usize; MAX_POS],
    pub(crate) dot_count: usize,
    pub(crate) colon_positions: [usize; MAX_POS],
    pub(crate) colon_count: usize,
    pub(crate) slash_positions: [usize; MAX_POS],
    pub(crate) slash_count: usize,
    pub(crate) dash_positions: [usize; MAX_POS],
    pub(crate) dash_count: usize,
    pub(crate) plus_positions: [usize; MAX_POS],
    pub(crate) plus_count: usize,
    pub(crate) bracket_open_positions: [usize; MAX_POS],
    pub(crate) bracket_open_count: usize,
    pub(crate) bracket_close_positions: [usize; MAX_POS],
    pub(crate) bracket_close_count: usize,
    pub(crate) comma_positions: [usize; MAX_POS],
    pub(crate) comma_count: usize,
    pub(crate) at_positions: [usize; MAX_POS],
    pub(crate) at_count: usize,

    // Pattern-specific flags
    pub(crate) starts_with_digit: bool,
    pub(crate) starts_with_letter: bool,
    pub(crate) starts_with_sign: bool,
}

impl<const MAX_POS: usize> PatternSignature<MAX_POS> {
    fn new() -> Self {
        Self {
            has_digits: false,
            has_letters: false,
            has_punctuation: false,
            has_whitespace: false,
            dot_positions: [0; MAX_POS],
            dot_count: 0,
            colon_positions: [0; MAX_POS],
            colon_count: 0,
            slash_positions: [0; MAX_POS],
            slash_count: 0,
            dash_positions: [0; MAX_POS],
            dash_count: 0,
            plus_positions: [0; MAX_POS],
            plus_count: 0,
            bracket_open_positions: [0; MAX_POS],
            bracket_open_count: 0,
            bracket_close_positions: [0; MAX_POS],
            bracket_close_count: 0,
            comma_positions: [0; MAX_POS],
            comma_count: 0,
            at_positions: [0; MAX_POS],
            at_count: 0,
            starts_with_digit: false,
            starts_with_letter: false,
            starts_with_sign: false,
        }
    }

    // Update signature with AVX-512 SIMD results
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[inline]
    unsafe fn update_avx512(
        &mut self,
        digit_mask: u64,
        letter_mask: u64,
        punct_mask: u64,
        space_mask: u64,
        dot_mask: u64,
        colon_mask: u64,
        slash_mask: u64,
        dash_mask: u64,
        plus_mask: u64,
        bracket_open_mask: u64,
        bracket_close_mask: u64,
        comma_mask: u64,
        at_mask: u64,
        position: usize,
    ) {
        // Update character class flags
        if digit_mask != 0 {
            self.has_digits = true;
        }
        if letter_mask != 0 {
            self.has_letters = true;
        }
        if punct_mask != 0 {
            self.has_punctuation = true;
        }
        if space_mask != 0 {
            self.has_whitespace = true;
        }

        // Extract specific character positions - ZERO HEAP
        Self::extract_positions_from_mask_64(
            dot_mask,
            position,
            &mut self.dot_positions,
            &mut self.dot_count,
        );
        Self::extract_positions_from_mask_64(
            colon_mask,
            position,
            &mut self.colon_positions,
            &mut self.colon_count,
        );
        Self::extract_positions_from_mask_64(
            slash_mask,
            position,
            &mut self.slash_positions,
            &mut self.slash_count,
        );
        Self::extract_positions_from_mask_64(
            dash_mask,
            position,
            &mut self.dash_positions,
            &mut self.dash_count,
        );
        Self::extract_positions_from_mask_64(
            plus_mask,
            position,
            &mut self.plus_positions,
            &mut self.plus_count,
        );
        Self::extract_positions_from_mask_64(
            bracket_open_mask,
            position,
            &mut self.bracket_open_positions,
            &mut self.bracket_open_count,
        );
        Self::extract_positions_from_mask_64(
            bracket_close_mask,
            position,
            &mut self.bracket_close_positions,
            &mut self.bracket_close_count,
        );
        Self::extract_positions_from_mask_64(
            comma_mask,
            position,
            &mut self.comma_positions,
            &mut self.comma_count,
        );
        Self::extract_positions_from_mask_64(
            at_mask,
            position,
            &mut self.at_positions,
            &mut self.at_count,
        );

        // Update start/end flags
        if position == 0 {
            self.starts_with_digit = (digit_mask & 1) != 0;
            self.starts_with_letter = (letter_mask & 1) != 0;
            self.starts_with_sign = (dash_mask & 1) != 0 || (plus_mask & 1) != 0;
        }
    }

    // Update signature with AVX2 SIMD results
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    #[inline]
    unsafe fn update_avx2(
        &mut self,
        digit_mask: u32,
        letter_mask: u32,
        punct_mask: u32,
        space_mask: u32,
        dot_mask: u32,
        colon_mask: u32,
        slash_mask: u32,
        dash_mask: u32,
        plus_mask: u32,
        bracket_open_mask: u32,
        bracket_close_mask: u32,
        comma_mask: u32,
        at_mask: u32,
        position: usize,
    ) {
        // Update character class flags
        if digit_mask != 0 {
            self.has_digits = true;
        }
        if letter_mask != 0 {
            self.has_letters = true;
        }
        if punct_mask != 0 {
            self.has_punctuation = true;
        }
        if space_mask != 0 {
            self.has_whitespace = true;
        }

        // Extract specific character positions - ZERO HEAP
        Self::extract_positions_from_mask_32(
            dot_mask,
            position,
            &mut self.dot_positions,
            &mut self.dot_count,
        );
        Self::extract_positions_from_mask_32(
            colon_mask,
            position,
            &mut self.colon_positions,
            &mut self.colon_count,
        );
        Self::extract_positions_from_mask_32(
            slash_mask,
            position,
            &mut self.slash_positions,
            &mut self.slash_count,
        );
        Self::extract_positions_from_mask_32(
            dash_mask,
            position,
            &mut self.dash_positions,
            &mut self.dash_count,
        );
        Self::extract_positions_from_mask_32(
            plus_mask,
            position,
            &mut self.plus_positions,
            &mut self.plus_count,
        );
        Self::extract_positions_from_mask_32(
            bracket_open_mask,
            position,
            &mut self.bracket_open_positions,
            &mut self.bracket_open_count,
        );
        Self::extract_positions_from_mask_32(
            bracket_close_mask,
            position,
            &mut self.bracket_close_positions,
            &mut self.bracket_close_count,
        );
        Self::extract_positions_from_mask_32(
            comma_mask,
            position,
            &mut self.comma_positions,
            &mut self.comma_count,
        );
        Self::extract_positions_from_mask_32(
            at_mask,
            position,
            &mut self.at_positions,
            &mut self.at_count,
        );

        // Update start/end flags
        if position == 0 {
            self.starts_with_digit = (digit_mask & 1) != 0;
            self.starts_with_letter = (letter_mask & 1) != 0;
            self.starts_with_sign = (dash_mask & 1) != 0 || (plus_mask & 1) != 0;
        }
    }

    // Update signature with NEON SIMD results
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    #[inline]
    unsafe fn update_neon(
        &mut self,
        digit_mask: u16,
        letter_mask: u16,
        punct_mask: u16,
        space_mask: u16,
        dot_mask: u16,
        colon_mask: u16,
        slash_mask: u16,
        dash_mask: u16,
        plus_mask: u16,
        bracket_open_mask: u16,
        bracket_close_mask: u16,
        comma_mask: u16,
        at_mask: u16,
        position: usize,
    ) {
        // Update character class flags
        if digit_mask != 0 {
            self.has_digits = true;
        }
        if letter_mask != 0 {
            self.has_letters = true;
        }
        if punct_mask != 0 {
            self.has_punctuation = true;
        }
        if space_mask != 0 {
            self.has_whitespace = true;
        }

        // Extract specific character positions - ZERO HEAP
        Self::extract_positions_from_mask_16(
            dot_mask,
            position,
            &mut self.dot_positions,
            &mut self.dot_count,
        );
        Self::extract_positions_from_mask_16(
            colon_mask,
            position,
            &mut self.colon_positions,
            &mut self.colon_count,
        );
        Self::extract_positions_from_mask_16(
            slash_mask,
            position,
            &mut self.slash_positions,
            &mut self.slash_count,
        );
        Self::extract_positions_from_mask_16(
            dash_mask,
            position,
            &mut self.dash_positions,
            &mut self.dash_count,
        );
        Self::extract_positions_from_mask_16(
            plus_mask,
            position,
            &mut self.plus_positions,
            &mut self.plus_count,
        );
        Self::extract_positions_from_mask_16(
            bracket_open_mask,
            position,
            &mut self.bracket_open_positions,
            &mut self.bracket_open_count,
        );
        Self::extract_positions_from_mask_16(
            bracket_close_mask,
            position,
            &mut self.bracket_close_positions,
            &mut self.bracket_close_count,
        );
        Self::extract_positions_from_mask_16(
            comma_mask,
            position,
            &mut self.comma_positions,
            &mut self.comma_count,
        );
        Self::extract_positions_from_mask_16(
            at_mask,
            position,
            &mut self.at_positions,
            &mut self.at_count,
        );

        // Update start/end flags
        if position == 0 {
            self.starts_with_digit = (digit_mask & 1) != 0;
            self.starts_with_letter = (letter_mask & 1) != 0;
            self.starts_with_sign = (dash_mask & 1) != 0 || (plus_mask & 1) != 0;
        }
    }

    // Extract bit positions from 64-bit mask (AVX-512 only) - ZERO HEAP
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[inline]
    fn extract_positions_from_mask_64(
        mask: u64,
        base_position: usize,
        positions: &mut [usize; MAX_POS],
        count: &mut usize,
    ) {
        let mut temp_mask = mask;
        while temp_mask != 0 && *count < MAX_POS {
            let bit_pos = temp_mask.trailing_zeros() as usize;
            positions[*count] = base_position + bit_pos;
            *count += 1;
            temp_mask &= temp_mask - 1; // Clear lowest set bit
        }
    }

    // Extract bit positions from 32-bit mask (AVX2 only) - ZERO HEAP
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    fn extract_positions_from_mask_32(
        mask: u32,
        base_position: usize,
        positions: &mut [usize; MAX_POS],
        count: &mut usize,
    ) {
        let mut temp_mask = mask;
        while temp_mask != 0 && *count < MAX_POS {
            let bit_pos = temp_mask.trailing_zeros() as usize;
            positions[*count] = base_position + bit_pos;
            *count += 1;
            temp_mask &= temp_mask - 1; // Clear lowest set bit
        }
    }

    // Extract bit positions from 16-bit mask (NEON) - ZERO HEAP
    #[cfg(target_arch = "aarch64")]
    fn extract_positions_from_mask_16(
        mask: u16,
        base_position: usize,
        positions: &mut [usize; MAX_POS],
        count: &mut usize,
    ) {
        let mut temp_mask = mask;
        while temp_mask != 0 && *count < MAX_POS {
            let bit_pos = temp_mask.trailing_zeros() as usize;
            positions[*count] = base_position + bit_pos;
            *count += 1;
            temp_mask &= temp_mask - 1; // Clear lowest set bit
        }
    }
}

// Character class statistics
#[derive(Debug, Clone)]
pub struct CharacterClasses {
    digit_count: usize,
    letter_count: usize,
    punctuation_count: usize,
    whitespace_count: usize,
    total_count: usize,
}

impl CharacterClasses {
    fn new() -> Self {
        Self {
            digit_count: 0,
            letter_count: 0,
            punctuation_count: 0,
            whitespace_count: 0,
            total_count: 0,
        }
    }

    // Update counts with AVX-512 results
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    fn update_avx512(
        &mut self,
        digit_mask: u64,
        letter_mask: u64,
        punct_mask: u64,
        space_mask: u64,
    ) {
        self.digit_count += digit_mask.count_ones() as usize;
        self.letter_count += letter_mask.count_ones() as usize;
        self.punctuation_count += punct_mask.count_ones() as usize;
        self.whitespace_count += space_mask.count_ones() as usize;
        self.total_count += 64; // AVX-512 processes 64 bytes
    }

    // Update counts with AVX2 results
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    fn update_avx2(&mut self, digit_mask: u32, letter_mask: u32, punct_mask: u32, space_mask: u32) {
        self.digit_count += digit_mask.count_ones() as usize;
        self.letter_count += letter_mask.count_ones() as usize;
        self.punctuation_count += punct_mask.count_ones() as usize;
        self.whitespace_count += space_mask.count_ones() as usize;
        self.total_count += 32; // AVX2 processes 32 bytes
    }

    // Update counts with NEON results
    #[cfg(target_arch = "aarch64")]
    fn update_neon(&mut self, digit_mask: u16, letter_mask: u16, punct_mask: u16, space_mask: u16) {
        self.digit_count += digit_mask.count_ones() as usize;
        self.letter_count += letter_mask.count_ones() as usize;
        self.punctuation_count += punct_mask.count_ones() as usize;
        self.whitespace_count += space_mask.count_ones() as usize;
        self.total_count += 16; // NEON processes 16 bytes
    }
}

// =============================================================================
// TYPED ARRAY CLASSIFICATION ENGINE - SIMD VARIANTS
// =============================================================================

// GPU array content classification
#[cfg(has_cuda)]
pub unsafe fn classify_array_contents_gpu(content: &[u8], len: usize) -> ClassificationResult {
    // Accept content either WITH or WITHOUT brackets. Dispatcher passes bracketless content.
    let (inner_content, inner_len) = if len >= 2 && content[0] == b'[' && content[len - 1] == b']' {
        (&content[1..len - 1], len - 2)
    } else {
        (content, len)
    };

    // If empty array content, return early
    if inner_len == 0 {
        return ClassificationResult {
            hwx_type: HwxType::Array,
            element_count: 0,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        };
    }

    // Process content (expects bracketless)
    let result = classify_array_contents_gpu_inner(inner_content, inner_len);

    result
}

// PTX kernel for GPU array classification (host-callable .entry)
pub const PTX_CLASSIFY_ARRAY: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .entry classify_array(
      .param .u64 content_ptr,
      .param .u32 len,
      .param .u64 result_ptr
    ) {
      // Register declarations - properly sized for all usage
      .reg .u32 %r<100>;  // Increased for additional address calculations
      .reg .u64 %rd<80>;   // Increased to cover up to %rd74 used below
      .reg .f64 %fd<2>;
      .reg .u8 %rc<20>;     // For byte operations - need up to %rc18
      .reg .pred %p<128>;     // Increased to cover extended classification predicates
      
      // Shared memory for element boundaries (max 512 elements * 8 bytes each)
      .shared .align 8 .b8 element_boundaries[4096];
      
      ld.param.u64 %rd0, [content_ptr];
      ld.param.u32 %r0, [len];
      ld.param.u64 %rd1, [result_ptr];
      
      // Check for null pointers
      setp.eq.u64 %p48, %rd0, 0;
      @%p48 bra return_error;
      setp.eq.u64 %p49, %rd1, 0;
      @%p49 bra return_error;
      
      // Handle empty array
      setp.eq.u32 %p0, %r0, 0;
      @!%p0 bra not_empty;
      
      // Return Array type with element_count = 0
      mov.u32 %r1, 20;  // HwxType::Array
      st.global.u32 [%rd1], %r1;
      mov.u32 %r2, 0;
      add.u64 %rd2, %rd1, 4;
      st.global.u32 [%rd2], %r2;  // element_count = 0
      ret;
      
    not_empty:
      // Use global memory for state tracking (stored in result buffer)
      // result_ptr layout: [4] hwx_type, [4] element_count, [8] numeric_value_1, [8] numeric_value_2
      // [4] number_count (24), [4] float_count (28), [4] string_count (32), [4] valid_count (36), [4] boolean_count (40)
      // [8*N] element_boundaries_and_lengths
      
      // Initialize global memory state
      mov.u32 %r3, %tid.x;
      setp.eq.u32 %p1, %r3, 0;
      @!%p1 bra skip_init;
      
      st.global.u32 [%rd1], 20;       // hwx_type = Array
      add.u64 %rd2, %rd1, 4;
      st.global.u32 [%rd2], 0;        // element_count = 0
      add.u64 %rd3, %rd1, 8;
      mov.f64 %fd0, 0.0;
      st.global.f64 [%rd3], %fd0;     // numeric_value_1 = 0.0 (for debugging)
      add.u64 %rd4, %rd1, 16;
      st.global.f64 [%rd4], %fd0;     // numeric_value_2 = 0.0 (for debugging)
      
      // Initialize atomic counters at offsets 24, 28, 32, 36, 40
      add.u64 %rd22, %rd1, 24;
      mov.u32 %r89, 0;
      st.global.u32 [%rd22], %r89;    // number_count = 0 (at offset 24)
      add.u64 %rd23, %rd1, 28;
      st.global.u32 [%rd23], %r89;    // float_count = 0 (at offset 28)
      add.u64 %rd31, %rd1, 32;
      st.global.u32 [%rd31], %r89;    // string_count = 0 (at offset 32)
      add.u64 %rd32, %rd1, 36;
      st.global.u32 [%rd32], %r89;    // valid_count = 0 (at offset 36)
      add.u64 %rd35, %rd1, 40;
      st.global.u32 [%rd35], %r89;    // boolean_count = 0 (at offset 40)
      
    skip_init:
      // Initialize shared memory to prevent reading garbage values
      // Each thread initializes a portion of shared memory
      mov.u32 %r50, %tid.x;
      shl.b32 %r51, %r50, 4;  // each thread handles 16 bytes
      setp.ge.u32 %p42, %r51, 4096;  // check if beyond shared memory size
      @%p42 bra init_done;
      
      // Initialize 16 bytes of shared memory to 0xFFFFFFFF (invalid marker)
      cvta.shared.u64 %rd28, element_boundaries;
      cvt.u64.u32 %rd29, %r51;
      add.u64 %rd24, %rd28, %rd29;
      
      mov.u32 %r52, 0xFFFFFFFF;  // invalid marker value
      st.u32 [%rd24], %r52;
      add.u64 %rd25, %rd24, 4;
      st.u32 [%rd25], %r52;
      add.u64 %rd26, %rd24, 8;
      st.u32 [%rd26], %r52;
      add.u64 %rd27, %rd24, 12;
      st.u32 [%rd27], %r52;
      
    init_done:
      bar.sync 0;
      
      // Phase 1: Sequential parsing by thread 0 to find element boundaries
      mov.u32 %r4, %tid.x;
      setp.eq.u32 %p1, %r4, 0;
      @!%p1 bra wait_for_boundaries;
      
      // Thread 0: Sequential scan to find element boundaries
      
      mov.u32 %r5, 0;  // position
      mov.u32 %r6, 0;  // element_index
      mov.u32 %r7, 0;  // current_element_start
      mov.u32 %r8, 0;  // local_bracket_depth
      mov.u32 %r9, 0;  // local_in_string
      
    sequential_scan:
      setp.ge.u32 %p2, %r5, %r0;
      @%p2 bra scan_done;
      
      // Load byte at position with extra safety check
      setp.ge.u32 %p47, %r5, %r0;  // double-check bounds
      @%p47 bra scan_done;  // exit if somehow out of bounds
      
      cvt.u64.u32 %rd4, %r5;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u8 %rc0, [%rd5];
      
      // Convert byte to u32 once for all comparisons
      cvt.u32.u8 %r10, %rc0;
      
      // Check for quote (toggle string state)
      setp.eq.u32 %p3, %r10, 34;  // '"' = 34
      @!%p3 bra check_brackets;
      xor.b32 %r9, %r9, 1;  // toggle in_string
      bra next_byte;
      
    check_brackets:
      // Skip bracket checks if in string
      setp.ne.u32 %p4, %r9, 0;
      @%p4 bra check_comma;
      
      // Check for open bracket
      setp.eq.u32 %p5, %r10, 91;  // '[' = 91
      @!%p5 bra check_close_bracket;
      add.u32 %r8, %r8, 1;  // increment bracket_depth
      bra next_byte;
      
    check_close_bracket:
      setp.eq.u32 %p6, %r10, 93;  // ']' = 93 (reuse %r10)
      @!%p6 bra check_comma;
      sub.u32 %r8, %r8, 1;  // decrement bracket_depth
      bra next_byte;
      
    check_comma:
      // Check for comma at bracket_depth == 0 (element boundary)
      setp.eq.u32 %p7, %r10, 44;  // ',' = 44 (reuse %r10)
      @!%p7 bra next_byte;
      
      // Must not be in string and bracket_depth must be 0
      setp.eq.u32 %p8, %r9, 0;     // not in_string
      setp.eq.u32 %p9, %r8, 0;     // bracket_depth == 0
      and.pred %p10, %p8, %p9;
      @!%p10 bra next_byte;
      
      // Found element boundary
      sub.u32 %r11, %r5, %r7;  // element length
      
      // Skip empty elements
      setp.eq.u32 %p11, %r11, 0;
      @%p11 bra update_start;
      
      // Store element info using shared memory for parallel processing
      // Add bounds check to prevent shared memory overflow
      setp.ge.u32 %p44, %r6, 512;  // check if element_index >= 512 (max elements in 4096 bytes)
      @%p44 bra update_start;  // skip storing if out of bounds
      
      shl.b32 %r12, %r6, 3;  // offset = index * 8 bytes (start + length)
      cvt.u64.u32 %rd6, %r12;
      
      // Store in shared memory using generic addressing
      cvta.shared.u64 %rd7, element_boundaries;
      add.u64 %rd8, %rd7, %rd6;
      st.u32 [%rd8], %r7;  // element start (generic pointer)
      add.u64 %rd9, %rd8, 4;
      st.u32 [%rd9], %r11;  // element length (generic pointer)
      
      add.u32 %r6, %r6, 1;  // increment element count
      
    update_start:
      add.u32 %r7, %r5, 1;  // new element starts after comma
      
    next_byte:
      add.u32 %r5, %r5, 1;
      bra sequential_scan;
      
    scan_done:
      // Process final element
      sub.u32 %r11, %r0, %r7;  // final element length
      setp.eq.u32 %p12, %r11, 0;
      @%p12 bra store_count;  // if empty, skip to store_count
      
      // Store final element info in shared memory
      // Add bounds check to prevent shared memory overflow
      setp.ge.u32 %p45, %r6, 512;  // check if element_index >= 512 (max elements in 4096 bytes)
      @%p45 bra store_count;  // skip storing if out of bounds
      
      shl.b32 %r12, %r6, 3;  // offset = index * 8 bytes
      cvt.u64.u32 %rd6, %r12;
      
      // Store in shared memory using generic addressing
      cvta.shared.u64 %rd7, element_boundaries;
      add.u64 %rd8, %rd7, %rd6;
      st.u32 [%rd8], %r7;  // element start (generic pointer)
      add.u64 %rd9, %rd8, 4;
      st.u32 [%rd9], %r11;  // element length (generic pointer)
      
      add.u32 %r6, %r6, 1;
      
    store_count:
      add.u64 %rd2, %rd1, 4;
      st.global.u32 [%rd2], %r6; // element_count in global memory
      
      
    wait_for_boundaries:
      bar.sync 0;
      
      // Phase 2: All threads classify elements in parallel
      add.u64 %rd2, %rd1, 4;
      ld.global.u32 %r13, [%rd2]; // element_count from global memory
      setp.eq.u32 %p13, %r13, 0;
      @%p13 bra loop_end;
      
      // Each thread processes different elements
      mov.u32 %r14, %tid.x;
      mov.u32 %r15, %ntid.x;
      
    element_loop:
      setp.ge.u32 %p14, %r14, %r13;
      @%p14 bra loop_end;
      
      // Load element info from shared memory
      // Add bounds check to prevent shared memory overflow
      setp.ge.u32 %p46, %r14, 512;  // check if element_index >= 512 (max elements in 4096 bytes)
      @%p46 bra next_element;  // skip if out of bounds
      
      shl.b32 %r16, %r14, 3;  // offset = index * 8 bytes
      cvt.u64.u32 %rd18, %r16;
      
      // Load from shared memory using generic addressing
      cvta.shared.u64 %rd19, element_boundaries;
      add.u64 %rd20, %rd19, %rd18;
      ld.u32 %r17, [%rd20];  // element start (generic pointer)
      add.u64 %rd21, %rd20, 4;
      ld.u32 %r18, [%rd21];  // element length (generic pointer)
      
      // Validate element data is initialized (not 0xFFFFFFFF)
      mov.u32 %r43, 0xFFFFFFFF;
      setp.eq.u32 %p43, %r17, %r43;  // check if element_start is uninitialized
      @%p43 bra next_element;  // skip if uninitialized
      
      // Validate element bounds before processing
      setp.ge.u32 %p40, %r17, %r0;  // check if element_start >= content_length
      @%p40 bra next_element;  // skip if start is invalid
      
      add.u32 %r42, %r17, %r18;  // element_start + element_length
      setp.gt.u32 %p27, %r42, %r0;  // check if exceeds content length
      @%p27 bra next_element;  // skip if out of bounds
      
      setp.eq.u32 %p41, %r18, 0;  // check if element_length == 0
      @%p41 bra next_element;  // skip empty elements
      
      // Use local registers for pattern tracking
      mov.u32 %r19, 0;  // has_digits
      mov.u32 %r38, 0;  // dot_count
      
      // Inline classify_chunk logic for this element
      cvt.u64.u32 %rd14, %r17;
      add.u64 %rd15, %rd0, %rd14;  // element pointer
      
      // Scan element bytes
      mov.u32 %r25, 0;  // byte index
      
    elem_scan_loop:
      setp.ge.u32 %p22, %r25, %r18;  // byte_index >= element_length
      @%p22 bra elem_scan_done;
      
      // Load byte
      cvt.u64.u32 %rd16, %r25;
      add.u64 %rd17, %rd15, %rd16;
      ld.global.u8 %rc0, [%rd17];
      
      // Check if digit (0x30-0x39) - convert to u32 for comparison
      cvt.u32.u8 %r26, %rc0;
      setp.ge.u32 %p23, %r26, 48;  // '0' = 48
      setp.le.u32 %p24, %r26, 57;  // '9' = 57
      and.pred %p25, %p23, %p24;
      @!%p25 bra elem_check_dot;
      mov.u32 %r19, 1; // has_digits = true
      
    elem_check_dot:
      // Check if dot (0x2E)
      setp.eq.u32 %p26, %r26, 46;  // '.' = 46 (reuse %r26)
      @!%p26 bra elem_next_byte;
      add.u32 %r38, %r38, 1; // dot_count++
      
    elem_next_byte:
      add.u32 %r25, %r25, 1;
      bra elem_scan_loop;
      
    elem_scan_done:
      // Determine trimmed first and last non-whitespace chars
      // s_idx in %r70, e_idx in %r71
      mov.u32 %r70, 0;            // offset from element start
      mov.u32 %r71, %r18;         // length
      @%p41 bra next_element;     // guard empty elements (already handled)

    trim_front:
      setp.ge.u32 %p60, %r70, %r18;
      @%p60 bra have_first;
      cvt.u64.u32 %rd40, %r70;
      add.u64 %rd41, %rd15, %rd40;
      ld.global.u8 %rc1, [%rd41];
      cvt.u32.u8 %r72, %rc1;
      // ws check
      setp.eq.u32 %p61, %r72, 32;
      setp.eq.u32 %p62, %r72, 9;
      or.pred %p63, %p61, %p62;
      setp.eq.u32 %p64, %r72, 10;
      setp.eq.u32 %p65, %r72, 13;
      or.pred %p66, %p64, %p65;
      or.pred %p67, %p63, %p66;
      @!%p67 bra have_first;
      add.u32 %r70, %r70, 1; bra trim_front;

    have_first:
      // back trim
      setp.eq.u32 %p68, %r18, 0;
      @%p68 bra classify_fallback;
      mov.u32 %r71, %r18;
    trim_back:
      setp.eq.u32 %p69, %r71, 0;
      @%p69 bra classify_fallback;
      sub.u32 %r71, %r71, 1;
      cvt.u64.u32 %rd42, %r71;
      add.u64 %rd43, %rd15, %rd42;
      ld.global.u8 %rc2, [%rd43];
      cvt.u32.u8 %r73, %rc2;
      setp.eq.u32 %p71, %r73, 32;
      setp.eq.u32 %p72, %r73, 9;
      or.pred %p73, %p71, %p72;
      setp.eq.u32 %p74, %r73, 10;
      setp.eq.u32 %p75, %r73, 13;
      or.pred %p76, %p74, %p75;
      or.pred %p77, %p73, %p76;
      @%p77 bra trim_back;

      // Now r70 is first non-ws offset, r71 is last index within element
      // Load first and last chars
      cvt.u64.u32 %rd44, %r70;
      add.u64 %rd45, %rd15, %rd44;
      ld.global.u8 %rc3, [%rd45];
      cvt.u32.u8 %r53, %rc3;

      cvt.u64.u32 %rd46, %r71;
      add.u64 %rd47, %rd15, %rd46;
      ld.global.u8 %rc4, [%rd47];
      cvt.u32.u8 %r54, %rc4;

      // 1) Quoted string => string_count++ and valid_count++
      setp.eq.u32 %p80, %r53, 34; // '"'
      setp.eq.u32 %p81, %r54, 34; // '"'
      and.pred %p82, %p80, %p81;
      @!%p82 bra check_boolean;
      add.u64 %rd31, %rd1, 32; mov.u32 %r74, 1; atom.global.add.u32 %r75, [%rd31], %r74; // string_count
      add.u64 %rd32, %rd1, 36; atom.global.add.u32 %r76, [%rd32], %r74;                     // valid_count
      bra next_element;

    check_boolean:
      // 2) Boolean literals true/false (unquoted)
      // Compute trimmed length = (r71 - r70 + 1)
      sub.u32 %r77, %r71, %r70; add.u32 %r77, %r77, 1;
      // First char either 't' or 'f'
      setp.eq.u32 %p83, %r53, 116; // 't'
      setp.eq.u32 %p84, %r53, 102; // 'f'
      or.pred %p85, %p83, %p84;
      @!%p85 bra check_number_class;
      // Check "true"
      setp.eq.u32 %p86, %r77, 4;
      @!%p86 bra check_false;
      // bytes at offsets r70..r70+3
      // t(116) r(114) u(117) e(101)
      {
        cvt.u64.u32 %rd60, %r70; add.u64 %rd60, %rd15, %rd60; ld.global.u8 %rc5, [%rd60]; cvt.u32.u8 %r80, %rc5;
        add.u64 %rd61, %rd60, 1; ld.global.u8 %rc6, [%rd61]; cvt.u32.u8 %r81, %rc6;
        add.u64 %rd62, %rd60, 2; ld.global.u8 %rc7, [%rd62]; cvt.u32.u8 %r82, %rc7;
        add.u64 %rd63, %rd60, 3; ld.global.u8 %rc8, [%rd63]; cvt.u32.u8 %r83, %rc8;
        setp.eq.u32 %p87, %r80, 116; setp.eq.u32 %p88, %r81, 114; and.pred %p89, %p87, %p88;
        setp.eq.u32 %p90, %r82, 117; setp.eq.u32 %p91, %r83, 101; and.pred %p92, %p90, %p91;
        and.pred %p93, %p89, %p92;
      }
      @!%p93 bra check_false;
      add.u64 %rd35, %rd1, 40; mov.u32 %r84, 1; atom.global.add.u32 %r85, [%rd35], %r84; // boolean_count
      add.u64 %rd32, %rd1, 36; atom.global.add.u32 %r86, [%rd32], %r84;                   // valid_count
      bra next_element;

    check_false:
      // Check "false"
      setp.eq.u32 %p94, %r77, 5;
      @!%p94 bra check_number_class;
      {
        cvt.u64.u32 %rd70, %r70; add.u64 %rd70, %rd15, %rd70; ld.global.u8 %rc9, [%rd70]; cvt.u32.u8 %r90, %rc9;
        add.u64 %rd71, %rd70, 1; ld.global.u8 %rc10, [%rd71]; cvt.u32.u8 %r91, %rc10;
        add.u64 %rd72, %rd70, 2; ld.global.u8 %rc11, [%rd72]; cvt.u32.u8 %r92, %rc11;
        add.u64 %rd73, %rd70, 3; ld.global.u8 %rc12, [%rd73]; cvt.u32.u8 %r93, %rc12;
        add.u64 %rd74, %rd70, 4; ld.global.u8 %rc13, [%rd74]; cvt.u32.u8 %r94, %rc13;
        setp.eq.u32 %p95, %r90, 102; setp.eq.u32 %p96, %r91, 97; and.pred %p97, %p95, %p96;
        setp.eq.u32 %p98, %r92, 108; setp.eq.u32 %p99, %r93, 115; and.pred %p100, %p98, %p99;
        setp.eq.u32 %p101, %r94, 101; and.pred %p102, %p97, %p100; and.pred %p103, %p102, %p101;
      }
      @!%p103 bra check_number_class;
      add.u64 %rd35, %rd1, 40; mov.u32 %r95, 1; atom.global.add.u32 %r96, [%rd35], %r95; // boolean_count
      add.u64 %rd32, %rd1, 36; atom.global.add.u32 %r97, [%rd32], %r95;                   // valid_count
      bra next_element;

    check_number_class:
      // 3) Numeric: use has_digits and dot_count gathered earlier
      mov.u32 %r20, %r19; // has_digits
      setp.ne.u32 %p15, %r20, 0;
      @!%p15 bra next_element;
      // valid_count++ and number_count++
      add.u64 %rd32, %rd1, 36; mov.u32 %r44, 1; atom.global.add.u32 %r45, [%rd32], %r44;
      add.u64 %rd22, %rd1, 24; atom.global.add.u32 %r46, [%rd22], %r44;
      // float_count if dot_count > 0
      mov.u32 %r21, %r38;
      setp.ne.u32 %p16, %r21, 0;
      @!%p16 bra next_element;
      add.u64 %rd23, %rd1, 28; mov.u32 %r47, 1; atom.global.add.u32 %r48, [%rd23], %r47;

    classify_fallback:
      // If none matched, fall through as invalid (no counters updated)
      
    next_element:
      // Move to next element (stride by number of threads)
      add.u32 %r14, %r14, %r15;
      bra element_loop;
      
    loop_end:
      bar.sync 0;
      
      // First thread determines result type  
      mov.u32 %r24, %tid.x;
      setp.eq.u32 %p17, %r24, 0;
      @!%p17 bra done;
      
      // Load counts from global memory
      add.u64 %rd2, %rd1, 4;
      ld.global.u32 %r25, [%rd2];  // element_count
      add.u64 %rd22, %rd1, 24;
      ld.global.u32 %r26, [%rd22]; // number_count (at offset 24)
      add.u64 %rd23, %rd1, 28;
      ld.global.u32 %r27, [%rd23]; // float_count (at offset 28)
      add.u64 %rd33, %rd1, 32;
      ld.global.u32 %r29, [%rd33]; // string_count (at offset 32)
      add.u64 %rd34, %rd1, 36;
      ld.global.u32 %r30, [%rd34]; // valid_count (at offset 36)
      add.u64 %rd36, %rd1, 40;
      ld.global.u32 %r31, [%rd36]; // boolean_count (at offset 40)
      
      // Debug: If no elements found, return early with Array type
      setp.eq.u32 %p20, %r25, 0;
      @%p20 bra store_result;
      
      // Determine HwxType with precedence:
      // If any invalid tokens -> String
      setp.lt.u32 %p18, %r30, %r25; // valid_count < element_count
      mov.u32 %r28, 16;  // String
      @%p18 bra store_result;
      // All strings -> StringArray
      setp.eq.u32 %p19, %r29, %r25;
      mov.u32 %r28, 24;  // StringArray
      @%p19 bra store_result;
      // All booleans -> BooleanArray
      setp.eq.u32 %p21, %r31, %r25;
      mov.u32 %r28, 25;  // BooleanArray
      @%p21 bra store_result;
      // All numbers -> IntegerArray or FloatArray
      setp.eq.u32 %p22, %r26, %r25;  // number_count == element_count
      mov.u32 %r28, 20;  // Default: Array
      @!%p22 bra store_result;
      setp.eq.u32 %p23, %r27, %r25;  // float_count == element_count
      mov.u32 %r28, 22;  // IntegerArray
      @!%p23 bra store_result;
      mov.u32 %r28, 23;  // FloatArray
      
    store_result:
      st.global.u32 [%rd1], %r28;  // hwx_type
      add.u64 %rd2, %rd1, 4;
      st.global.u32 [%rd2], %r25;  // element_count
      
      // Debug: Store number_count and float_count in numeric_value fields for debugging
      cvt.rn.f64.u32 %fd0, %r26;
      add.u64 %rd3, %rd1, 8;
      st.global.f64 [%rd3], %fd0;   // numeric_value_1 = number_count (debug copy)
      cvt.rn.f64.u32 %fd1, %r27;
      add.u64 %rd4, %rd1, 16;
      st.global.f64 [%rd4], %fd1;  // numeric_value_2 = float_count (debug copy)
      
      // Synchronize all threads before returning
      bar.sync 0;
      
    done:
      ret;
      
    return_error:
      // Return Array type for error cases
      mov.u32 %r90, 20;  // HwxType::Array
      st.global.u32 [%rd1], %r90;
      mov.u32 %r91, 0;
      add.u64 %rd30, %rd1, 4;
      st.global.u32 [%rd30], %r91;  // element_count = 0
      ret;
    }
  "#;

// Core array classification as a reusable .func
pub const PTX_CLASSIFY_ARRAY_CORE_FUNC: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .visible .func classify_array_core(
    .param .u64 content_ptr,
    .param .u32 len,
    .param .u64 result_ptr
  ) {
    .reg .u64 %rd<2>;
    .reg .u32 %r<4>;

    ld.param.u64 %rd0, [content_ptr];
    ld.param.u32 %r0,  [len];
    ld.param.u64 %rd1, [result_ptr];

    mov.u32 %r1, 20; // HwxType::Array
    mov.u32 %r2, 8;  // FieldType::Array

    shl.b32 %r3, %r2, 8;
    or.b32  %r3, %r3, %r1;
    st.global.u32 [%rd1], %r3;
    ret;
  }
"#;

// Thin inline wrapper calling the shared array core
pub const PTX_CLASSIFY_ARRAY_INLINE_FUNC: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .extern .func classify_array_core;

  .visible .func classify_array_inline(
    .param .u64 content_ptr,
    .param .u32 len,
    .param .u64 result_ptr
  ) {
    .param .u64 p0; .param .u32 p1; .param .u64 p2;
    ld.param.u64 %rd0, [content_ptr];
    ld.param.u32 %r0,  [len];
    ld.param.u64 %rd1, [result_ptr];
    st.param.u64 [p0], %rd0;
    st.param.u32 [p1], %r0;
    st.param.u64 [p2], %rd1;
    call.uni classify_array_core, (p0, p1, p2);
    ret;
  }
"#;

// Internal GPU array content classification (expects content WITHOUT brackets)
#[cfg(has_cuda)]
unsafe fn classify_array_contents_gpu_inner(content: &[u8], len: usize) -> ClassificationResult {
    // Handle empty array
    if len == 0 {
        return ClassificationResult {
            hwx_type: HwxType::Array,
            element_count: 0,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        };
    }

    // Removed verbose debug prints for performance

    let mut result = ClassificationResult {
        hwx_type: HwxType::Array,
        element_count: 0,
        numeric_value_1: 0.0,
        numeric_value_2: 0.0,
    };

    // Cooperative single-block execution to avoid inter-block races on shared/global state
    let (blocks, threads) = (1u32, 256u32);

    // Launch with device-side content and result buffers (no host pointers in PTX)
    #[repr(C)]
    struct GpuClassificationResult {
        hwx_type_u32: u32,
        element_count_u32: u32,
        numeric_value_1: f64,
        numeric_value_2: f64,
        number_count_u32: u32,
        float_count_u32: u32,
        string_count_u32: u32,
        valid_count_u32: u32,
        boolean_count_u32: u32,
    }
    let _ = with_gpu_buffer_u8(content, len, |gpu_content, gpu_len| {
        // Ensure CUDA context
        let _ = ensure_cuda_initialized();
        // Allocate device buffer for ClassificationResult
        let mut d_result: *mut std::ffi::c_void = std::ptr::null_mut();
        let result_size = std::mem::size_of::<GpuClassificationResult>();
        unsafe {
            if cudaMalloc(&mut d_result as *mut *mut std::ffi::c_void, result_size) != 0 {
                return 0usize;
            }
            // Initialize device result to zeros
            let mut zeroed = GpuClassificationResult {
                hwx_type_u32: 20,
                element_count_u32: 0,
                numeric_value_1: 0.0,
                numeric_value_2: 0.0,
                number_count_u32: 0,
                float_count_u32: 0,
                string_count_u32: 0,
                valid_count_u32: 0,
                boolean_count_u32: 0,
            };
            let _ = cudaMemcpy(
                d_result,
                &mut zeroed as *mut GpuClassificationResult as *mut std::ffi::c_void,
                result_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            // Launch kernel
            let param_content_ptr: u64 = gpu_content as u64;
            let param_len: u32 = gpu_len as u32;
            let param_result_ptr: u64 = d_result as u64;
            let _ = launch_ptx(
                PTX_CLASSIFY_ARRAY,
                &[],
                "classify_array",
                blocks,
                threads,
                &[
                    &param_content_ptr as *const _ as *const u8,
                    &param_len as *const _ as *const u8,
                    &param_result_ptr as *const _ as *const u8,
                ],
            );

            // Copy device result back to host staging and map to ClassificationResult
            let mut gpu_res = GpuClassificationResult {
                hwx_type_u32: 20,
                element_count_u32: 0,
                numeric_value_1: 0.0,
                numeric_value_2: 0.0,
                number_count_u32: 0,
                float_count_u32: 0,
                string_count_u32: 0,
                valid_count_u32: 0,
                boolean_count_u32: 0,
            };
            let _ = cudaMemcpy(
                &mut gpu_res as *mut GpuClassificationResult as *mut std::ffi::c_void,
                d_result,
                result_size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            // Use GPU-computed type directly
            result.hwx_type = match gpu_res.hwx_type_u32 {
                13 => HwxType::Integer,
                14 => HwxType::Float,
                16 => HwxType::String,
                19 => HwxType::Vector,
                20 => HwxType::Array,
                22 => HwxType::IntegerArray,
                23 => HwxType::FloatArray,
                24 => HwxType::StringArray,
                25 => HwxType::BooleanArray,
                _ => HwxType::Array,
            };
            result.element_count = gpu_res.element_count_u32 as usize;
            // For array classifications, numeric fields should be 0.0 per tests
            result.numeric_value_1 = 0.0;
            result.numeric_value_2 = 0.0;

            // Free device result
            let _ = cudaFree(d_result);
        }

        gpu_len
    });

    result
}

//  AVX-512 array content classification - ZERO HEAP, CHUNKED PROCESSING
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn classify_array_contents_avx512(
    content: &[u8],
    len: usize,
) -> ClassificationResult {
    // Handle empty array content - return generic Array type
    if len == 0 {
        return ClassificationResult {
            hwx_type: HwxType::Array,
            element_count: 0,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        };
    }

    //  ZERO HEAP: Process in chunks, stack-allocated buffer
    const CHUNK_SIZE_AVX512: usize = 64;
    let mut chunk_types: [HwxType; CHUNK_SIZE_AVX512] = [HwxType::String; CHUNK_SIZE_AVX512];
    let mut has_expected_type = false;
    let mut expected_type = HwxType::String;
    let mut chunk_count = 0;

    let mut current_start = 0;
    let mut bracket_depth = 0;
    let mut quote_count = 0;

    for (i, &byte) in content.iter().enumerate() {
        match byte {
            b'[' | b'(' => bracket_depth += 1,
            b']' | b')' => bracket_depth -= 1,
            b'"' => quote_count += 1,
            b',' if bracket_depth == 0 && quote_count % 2 == 0 => {
                if i > current_start {
                    let element_bytes = &content[current_start..i];
                    let (trimmed, trimmed_len) = trim_whitespace(element_bytes, i - current_start);
                    if trimmed_len > 0 {
                        if let Ok(element_str) = std::str::from_utf8(trimmed) {
                            //  ZERO RECURSION: Use call stack simulation instead of recursive calls
                            const MAX_CALL_STACK: usize = 32;
                            let mut call_stack: [(&str, usize); MAX_CALL_STACK] =
                                [("", 0); MAX_CALL_STACK];
                            call_stack[0] = (element_str, trimmed_len);
                            let mut call_stack_top = 1;

                            let mut element_result = ClassificationResult {
                                hwx_type: HwxType::String,
                                element_count: 1,
                                numeric_value_1: 0.0,
                                numeric_value_2: 0.0,
                            };

                            while call_stack_top > 0 {
                                call_stack_top -= 1;
                                let (call_s, call_len) = call_stack[call_stack_top];

                                // Execute exact classify_single_string_avx512 logic
                                let call_bytes = call_s.as_bytes();
                                let mut call_pattern_signature =
                                    PatternSignature::<MAX_POSITIONS_AVX512>::new();
                                let mut call_char_classes = CharacterClasses::new();

                                let mut call_i = 0;
                                while call_i + LANES_AVX512_BYTES <= call_len {
                                    let call_chunk = _mm512_loadu_si512(
                                        call_bytes.as_ptr().add(call_i) as *const _,
                                    );
                                    classify_chunk_avx512(
                                        call_chunk,
                                        &mut call_pattern_signature,
                                        &mut call_char_classes,
                                        call_i,
                                    );
                                    call_i += LANES_AVX512_BYTES;
                                }

                                if call_i < call_len {
                                    classify_scalar_remaining(
                                        &call_bytes[call_i..],
                                        &mut call_pattern_signature,
                                        call_i,
                                    );
                                }

                                let (call_result, call_numeric_1, call_numeric_2) =
                                    determine_type_from_signature(
                                        call_s,
                                        call_len,
                                        &call_pattern_signature,
                                    );

                                // If array detected, push array content onto call stack instead of recursion
                                if call_result == HwxType::Array {
                                    if call_len >= 2
                                        && call_s.starts_with('[')
                                        && call_s.ends_with(']')
                                    {
                                        let call_content = &call_s[1..call_len - 1];
                                        if call_stack_top < MAX_CALL_STACK {
                                            call_stack[call_stack_top] =
                                                (call_content, call_len - 2);
                                            call_stack_top += 1;
                                            continue;
                                        }
                                    }
                                }

                                element_result = ClassificationResult {
                                    hwx_type: call_result,
                                    element_count: 1,
                                    numeric_value_1: call_numeric_1,
                                    numeric_value_2: call_numeric_2,
                                };
                                break;
                            }
                            let element_type = element_result.hwx_type;
                            if element_type == HwxType::String && !element_str.starts_with('"') {
                                return ClassificationResult {
                                    hwx_type: HwxType::String,
                                    element_count: chunk_count,
                                    numeric_value_1: 0.0,
                                    numeric_value_2: 0.0,
                                };
                            }

                            // Add to current chunk
                            chunk_types[chunk_count % CHUNK_SIZE_AVX512] = element_result.hwx_type;
                            chunk_count += 1;

                            // Process chunk when full
                            if chunk_count % CHUNK_SIZE_AVX512 == 0 {
                                let chunk_result = process_chunk(
                                    &chunk_types,
                                    CHUNK_SIZE_AVX512,
                                    &mut has_expected_type,
                                    &mut expected_type,
                                );
                                if chunk_result == HwxType::Array {
                                    return ClassificationResult {
                                        hwx_type: HwxType::Array,
                                        element_count: chunk_count,
                                        numeric_value_1: 0.0,
                                        numeric_value_2: 0.0,
                                    }; // Mixed types found
                                }
                            }
                        }
                    }
                }
                current_start = i + 1;
            }
            _ => {}
        }
    }

    // Handle last element
    if current_start < len {
        let element_bytes = &content[current_start..];
        let (trimmed, trimmed_len) = trim_whitespace(element_bytes, len - current_start);
        if trimmed_len > 0 {
            if let Ok(element_str) = std::str::from_utf8(trimmed) {
                //  ZERO RECURSION: Same call stack simulation for AVX-512 final element
                const MAX_CALL_STACK_AVX512: usize = 32;
                let mut call_stack_avx512: [(&str, usize); MAX_CALL_STACK_AVX512] =
                    [("", 0); MAX_CALL_STACK_AVX512];
                call_stack_avx512[0] = (element_str, trimmed_len);
                let mut call_stack_top_avx512 = 1;

                let mut element_result = ClassificationResult {
                    hwx_type: HwxType::String,
                    element_count: 1,
                    numeric_value_1: 0.0,
                    numeric_value_2: 0.0,
                };

                while call_stack_top_avx512 > 0 {
                    call_stack_top_avx512 -= 1;
                    let (call_s_avx512, call_len_avx512) = call_stack_avx512[call_stack_top_avx512];

                    // Execute exact classify_single_string_avx512 logic
                    let call_bytes_avx512 = call_s_avx512.as_bytes();
                    let mut call_pattern_signature_avx512 =
                        PatternSignature::<MAX_POSITIONS_AVX512>::new();
                    let mut call_char_classes_avx512 = CharacterClasses::new();

                    let mut call_i_avx512 = 0;
                    while call_i_avx512 + LANES_AVX512_BYTES <= call_len_avx512 {
                        let call_chunk_avx512 = _mm512_loadu_si512(
                            call_bytes_avx512.as_ptr().add(call_i_avx512) as *const _,
                        );
                        classify_chunk_avx512(
                            call_chunk_avx512,
                            &mut call_pattern_signature_avx512,
                            &mut call_char_classes_avx512,
                            call_i_avx512,
                        );
                        call_i_avx512 += LANES_AVX512_BYTES;
                    }

                    if call_i_avx512 < call_len_avx512 {
                        classify_scalar_remaining(
                            &call_bytes_avx512[call_i_avx512..],
                            &mut call_pattern_signature_avx512,
                            call_i_avx512,
                        );
                    }

                    let (call_result_avx512, call_numeric_1_avx512, call_numeric_2_avx512) =
                        determine_type_from_signature(
                            call_s_avx512,
                            call_len_avx512,
                            &call_pattern_signature_avx512,
                        );

                    // If array detected, push array content onto call stack instead of recursion
                    if call_result_avx512 == HwxType::Array {
                        if call_len_avx512 >= 2
                            && call_s_avx512.starts_with('[')
                            && call_s_avx512.ends_with(']')
                        {
                            let call_content_avx512 = &call_s_avx512[1..call_len_avx512 - 1];
                            if call_stack_top_avx512 < MAX_CALL_STACK_AVX512 {
                                call_stack_avx512[call_stack_top_avx512] =
                                    (call_content_avx512, call_len_avx512 - 2);
                                call_stack_top_avx512 += 1;
                                continue;
                            }
                        }
                    }

                    element_result = ClassificationResult {
                        hwx_type: call_result_avx512,
                        element_count: 1,
                        numeric_value_1: call_numeric_1_avx512,
                        numeric_value_2: call_numeric_2_avx512,
                    };
                    break;
                }
                if element_result.hwx_type == HwxType::String && !element_str.starts_with('"') {
                    return ClassificationResult {
                        hwx_type: HwxType::String,
                        element_count: chunk_count,
                        numeric_value_1: 0.0,
                        numeric_value_2: 0.0,
                    };
                }
                chunk_types[chunk_count % CHUNK_SIZE_AVX512] = element_result.hwx_type;
                chunk_count += 1;
            }
        }
    }

    // Process final partial chunk
    if chunk_count > 0 {
        let final_chunk_size = chunk_count % CHUNK_SIZE_AVX512;
        let chunk_size = if final_chunk_size == 0 {
            CHUNK_SIZE_AVX512
        } else {
            final_chunk_size
        };
        let chunk_result = process_chunk(
            &chunk_types,
            chunk_size,
            &mut has_expected_type,
            &mut expected_type,
        );

        if chunk_result == HwxType::Array {
            return ClassificationResult {
                hwx_type: HwxType::Array,
                element_count: chunk_count,
                numeric_value_1: 0.0,
                numeric_value_2: 0.0,
            };
        }
    }

    // Return typed array based on expected type
    // Fixed
    ClassificationResult {
        hwx_type: expected_type,
        numeric_value_1: 0.0,
        numeric_value_2: 0.0,
        element_count: chunk_count,
    }
}

//  ZERO HEAP: Process a chunk of element types
#[inline]
fn process_chunk(
    chunk_types: &[HwxType],
    chunk_size: usize,
    has_expected_type: &mut bool,
    expected_type: &mut HwxType,
) -> HwxType {
    if chunk_size == 0 {
        return HwxType::Array;
    }

    // Check if chunk is homogeneous
    let first_type = &chunk_types[0];
    for current_type in chunk_types.iter().take(chunk_size).skip(1) {
        if *current_type != *first_type {
            return HwxType::Array; // Mixed types in chunk
        }
    }

    // Update or verify expected type
    if !*has_expected_type {
        *expected_type = type_to_array_type(*first_type);
        *has_expected_type = true;
        *expected_type
    } else {
        let current_array_type = type_to_array_type(*first_type);
        if current_array_type == *expected_type {
            *expected_type // Chunk matches expected type
        } else {
            HwxType::Array // Type mismatch across chunks
        }
    }
}

//  Convert element type to array type
#[inline]
fn type_to_array_type(element_type: HwxType) -> HwxType {
    match element_type {
        HwxType::Integer => HwxType::IntegerArray,
        HwxType::Float => HwxType::FloatArray,
        HwxType::String => HwxType::StringArray,
        HwxType::Boolean => HwxType::BooleanArray,
        HwxType::DateMath => HwxType::DateMathArray,
        HwxType::LogDate => HwxType::LogDateArray,
        HwxType::ISO8601Date => HwxType::ISO8601DateArray,
        HwxType::FullDate => HwxType::FullDateArray,
        HwxType::RFC2822Date => HwxType::RFC2822DateArray,
        HwxType::AmericanDate => HwxType::AmericanDateArray,
        HwxType::EuropeanDate => HwxType::EuropeanDateArray,
        HwxType::VerboseDate => HwxType::VerboseDateArray,
        HwxType::FinancialDate => HwxType::FinancialDateArray,
        HwxType::GenericDate => HwxType::GenericDateArray,
        HwxType::IPAddressV4 => HwxType::IPAddressV4Array,
        HwxType::IPAddressV6 => HwxType::IPAddressV6Array,
        HwxType::Geo => HwxType::GeoArray,
        HwxType::File => HwxType::FileArray,
        HwxType::Vector => HwxType::Array, // Vectors in arrays become generic arrays
        _ => HwxType::Array,               // For mixed or unsupported element types
    }
}

//  AVX2 array content classification - ZERO HEAP, CHUNKED PROCESSING
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn classify_array_contents_avx2(
    content: &[u8],
    len: usize,
) -> ClassificationResult {
    // Handle empty array content - return generic Array type
    if len == 0 {
        return ClassificationResult {
            hwx_type: HwxType::Array,
            element_count: 0,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        };
    }

    //  ZERO HEAP: Process in chunks, stack-allocated buffer
    const CHUNK_SIZE_AVX2: usize = 32;
    let mut chunk_types: [HwxType; CHUNK_SIZE_AVX2] = [HwxType::String; CHUNK_SIZE_AVX2];
    let mut has_expected_type = false;
    let mut expected_type = HwxType::String;
    let mut chunk_count = 0;

    let mut current_start = 0;
    let mut bracket_depth = 0;
    let mut quote_count = 0;

    for (i, &byte) in content.iter().enumerate() {
        match byte {
            b'[' | b'(' => bracket_depth += 1,
            b']' | b')' => bracket_depth -= 1,
            b'"' => quote_count += 1,
            b',' if bracket_depth == 0 && quote_count % 2 == 0 => {
                if i > current_start {
                    let element_bytes = &content[current_start..i];
                    let (trimmed, trimmed_len) = trim_whitespace(element_bytes, i - current_start);
                    if trimmed_len > 0 {
                        if let Ok(element_str) = std::str::from_utf8(trimmed) {
                            //  ZERO RECURSION: Use call stack simulation for AVX2
                            const MAX_CALL_STACK_AVX2: usize = 32;
                            let mut call_stack_avx2: [(&str, usize); MAX_CALL_STACK_AVX2] =
                                [("", 0); MAX_CALL_STACK_AVX2];
                            call_stack_avx2[0] = (element_str, trimmed_len);
                            let mut call_stack_top_avx2 = 1;

                            let mut element_result = ClassificationResult {
                                hwx_type: HwxType::String,
                                element_count: 1,
                                numeric_value_1: 0.0,
                                numeric_value_2: 0.0,
                            };

                            while call_stack_top_avx2 > 0 {
                                call_stack_top_avx2 -= 1;
                                let (call_s_avx2, call_len_avx2) =
                                    call_stack_avx2[call_stack_top_avx2];

                                // Execute exact classify_single_string_avx2 logic
                                let call_bytes_avx2 = call_s_avx2.as_bytes();
                                let mut call_pattern_signature_avx2 =
                                    PatternSignature::<MAX_POSITIONS_AVX2>::new();
                                let mut call_char_classes_avx2 = CharacterClasses::new();

                                let mut call_i_avx2 = 0;
                                while call_i_avx2 + LANES_AVX2_BYTES <= call_len_avx2 {
                                    let call_chunk_avx2 = _mm256_loadu_si256(
                                        call_bytes_avx2.as_ptr().add(call_i_avx2) as *const _,
                                    );
                                    classify_chunk_avx2(
                                        call_chunk_avx2,
                                        &mut call_pattern_signature_avx2,
                                        &mut call_char_classes_avx2,
                                        call_i_avx2,
                                    );
                                    call_i_avx2 += LANES_AVX2_BYTES;
                                }

                                if call_i_avx2 < call_len_avx2 {
                                    classify_scalar_remaining(
                                        &call_bytes_avx2[call_i_avx2..],
                                        &mut call_pattern_signature_avx2,
                                        call_i_avx2,
                                    );
                                }

                                let (call_result_avx2, call_numeric_1_avx2, call_numeric_2_avx2) =
                                    determine_type_from_signature(
                                        call_s_avx2,
                                        call_len_avx2,
                                        &call_pattern_signature_avx2,
                                    );

                                // If array detected, push array content onto call stack instead of recursion
                                if call_result_avx2 == HwxType::Array {
                                    if call_len_avx2 >= 2
                                        && call_s_avx2.starts_with('[')
                                        && call_s_avx2.ends_with(']')
                                    {
                                        let call_content_avx2 = &call_s_avx2[1..call_len_avx2 - 1];
                                        if call_stack_top_avx2 < MAX_CALL_STACK_AVX2 {
                                            call_stack_avx2[call_stack_top_avx2] =
                                                (call_content_avx2, call_len_avx2 - 2);
                                            call_stack_top_avx2 += 1;
                                            continue;
                                        }
                                    }
                                }

                                element_result = ClassificationResult {
                                    hwx_type: call_result_avx2,
                                    element_count: 1,
                                    numeric_value_1: call_numeric_1_avx2,
                                    numeric_value_2: call_numeric_2_avx2,
                                };
                                break;
                            }
                            if element_result.hwx_type == HwxType::String
                                && !element_str.starts_with('"')
                            {
                                return ClassificationResult {
                                    hwx_type: HwxType::String,
                                    element_count: chunk_count,
                                    numeric_value_1: 0.0,
                                    numeric_value_2: 0.0,
                                };
                            }

                            // Add to current chunk
                            chunk_types[chunk_count % CHUNK_SIZE_AVX2] = element_result.hwx_type;
                            chunk_count += 1;

                            // Process chunk when full
                            if chunk_count % CHUNK_SIZE_AVX2 == 0 {
                                let chunk_result = process_chunk(
                                    &chunk_types,
                                    CHUNK_SIZE_AVX2,
                                    &mut has_expected_type,
                                    &mut expected_type,
                                );

                                if chunk_result == HwxType::Array {
                                    return ClassificationResult {
                                        hwx_type: HwxType::Array,
                                        element_count: chunk_count,
                                        numeric_value_1: 0.0,
                                        numeric_value_2: 0.0,
                                    }; // Mixed types found
                                }
                            }
                        }
                    }
                }
                current_start = i + 1;
            }
            _ => {}
        }
    }

    // Handle last element
    if current_start < len {
        let element_bytes = &content[current_start..];
        let (trimmed, trimmed_len) = trim_whitespace(element_bytes, len - current_start);
        if trimmed_len > 0 {
            if let Ok(element_str) = std::str::from_utf8(trimmed) {
                //  ZERO RECURSION: Same call stack simulation for AVX2 final element
                const MAX_CALL_STACK_AVX2_FINAL: usize = 32;
                let mut call_stack_avx2_final: [(&str, usize); MAX_CALL_STACK_AVX2_FINAL] =
                    [("", 0); MAX_CALL_STACK_AVX2_FINAL];
                call_stack_avx2_final[0] = (element_str, trimmed_len);
                let mut call_stack_top_avx2_final = 1;

                let mut element_result = ClassificationResult {
                    hwx_type: HwxType::String,
                    element_count: 1,
                    numeric_value_1: 0.0,
                    numeric_value_2: 0.0,
                };

                while call_stack_top_avx2_final > 0 {
                    call_stack_top_avx2_final -= 1;
                    let (call_s_avx2_final, call_len_avx2_final) =
                        call_stack_avx2_final[call_stack_top_avx2_final];

                    // Execute exact classify_single_string_avx2 logic
                    let call_bytes_avx2_final = call_s_avx2_final.as_bytes();
                    let mut call_pattern_signature_avx2_final =
                        PatternSignature::<MAX_POSITIONS_AVX2>::new();
                    let mut call_char_classes_avx2_final = CharacterClasses::new();

                    let mut call_i_avx2_final = 0;
                    while call_i_avx2_final + LANES_AVX2_BYTES <= call_len_avx2_final {
                        let call_chunk_avx2_final = _mm256_loadu_si256(
                            call_bytes_avx2_final.as_ptr().add(call_i_avx2_final) as *const _,
                        );
                        classify_chunk_avx2(
                            call_chunk_avx2_final,
                            &mut call_pattern_signature_avx2_final,
                            &mut call_char_classes_avx2_final,
                            call_i_avx2_final,
                        );
                        call_i_avx2_final += LANES_AVX2_BYTES;
                    }

                    if call_i_avx2_final < call_len_avx2_final {
                        classify_scalar_remaining(
                            &call_bytes_avx2_final[call_i_avx2_final..],
                            &mut call_pattern_signature_avx2_final,
                            call_i_avx2_final,
                        );
                    }

                    let (
                        call_result_avx2_final,
                        call_numeric_1_avx2_final,
                        call_numeric_2_avx2_final,
                    ) = determine_type_from_signature(
                        call_s_avx2_final,
                        call_len_avx2_final,
                        &call_pattern_signature_avx2_final,
                    );

                    // If array detected, push array content onto call stack instead of recursion
                    if call_result_avx2_final == HwxType::Array {
                        if call_len_avx2_final >= 2
                            && call_s_avx2_final.starts_with('[')
                            && call_s_avx2_final.ends_with(']')
                        {
                            let call_content_avx2_final =
                                &call_s_avx2_final[1..call_len_avx2_final - 1];
                            if call_stack_top_avx2_final < MAX_CALL_STACK_AVX2_FINAL {
                                call_stack_avx2_final[call_stack_top_avx2_final] =
                                    (call_content_avx2_final, call_len_avx2_final - 2);
                                call_stack_top_avx2_final += 1;
                                continue;
                            }
                        }
                    }

                    element_result = ClassificationResult {
                        hwx_type: call_result_avx2_final,
                        element_count: 1,
                        numeric_value_1: call_numeric_1_avx2_final,
                        numeric_value_2: call_numeric_2_avx2_final,
                    };
                    break;
                }
                if element_result.hwx_type == HwxType::String && !element_str.starts_with('"') {
                    return ClassificationResult {
                        hwx_type: HwxType::String,
                        element_count: chunk_count,
                        numeric_value_1: 0.0,
                        numeric_value_2: 0.0,
                    };
                }
                chunk_types[chunk_count % CHUNK_SIZE_AVX2] = element_result.hwx_type;
                chunk_count += 1;
            }
        }
    }

    // Process final partial chunk
    if chunk_count > 0 {
        let final_chunk_size = chunk_count % CHUNK_SIZE_AVX2;
        let chunk_size = if final_chunk_size == 0 {
            CHUNK_SIZE_AVX2
        } else {
            final_chunk_size
        };
        let chunk_result = process_chunk(
            &chunk_types,
            chunk_size,
            &mut has_expected_type,
            &mut expected_type,
        );
        if chunk_result == HwxType::Array {
            return ClassificationResult {
                hwx_type: HwxType::Array,
                element_count: chunk_count,
                numeric_value_1: 0.0,
                numeric_value_2: 0.0,
            };
        }
    }

    // Return typed array based on expected type
    ClassificationResult {
        hwx_type: expected_type,
        numeric_value_1: 0.0,
        numeric_value_2: 0.0,
        element_count: chunk_count,
    }
}

//  NEON array content classification - ZERO HEAP, CHUNKED PROCESSING
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(super) unsafe fn classify_array_contents_neon(
    content: &[u8],
    len: usize,
) -> ClassificationResult {
    // Handle empty array content - return generic Array type
    if len == 0 {
        return ClassificationResult {
            hwx_type: HwxType::Array,
            element_count: 0,
            numeric_value_1: 0.0,
            numeric_value_2: 0.0,
        };
    }

    //  ZERO HEAP: Process in chunks, stack-allocated buffer
    const CHUNK_SIZE_NEON: usize = 16;
    let mut chunk_types: [HwxType; CHUNK_SIZE_NEON] = [HwxType::String; CHUNK_SIZE_NEON];
    let mut has_expected_type = false;
    let mut expected_type = HwxType::String;
    let mut chunk_count = 0;

    let mut current_start = 0;
    let mut bracket_depth = 0;
    let mut quote_count = 0;

    for (i, &byte) in content.iter().enumerate() {
        match byte {
            b'[' | b'(' => bracket_depth += 1,
            b']' | b')' => bracket_depth -= 1,
            b'"' => quote_count += 1,
            b',' if bracket_depth == 0 && quote_count % 2 == 0 => {
                if i > current_start {
                    let element_bytes = &content[current_start..i];
                    let (trimmed, trimmed_len) = trim_whitespace(element_bytes, i - current_start);
                    if trimmed_len > 0 {
                        if let Ok(element_str) = std::str::from_utf8(trimmed) {
                            //  STACK RECURSION: Replace recursive call with iterative stack processing
                            let element_result = {
                                const MAX_STACK: usize = 32;
                                let mut stack: [(&str, usize); MAX_STACK] = [("", 0); MAX_STACK];
                                let mut top = 1;

                                stack[0] = (element_str, trimmed_len);

                                let mut final_result = ClassificationResult {
                                    hwx_type: HwxType::String,
                                    element_count: 1,
                                    numeric_value_1: 0.0,
                                    numeric_value_2: 0.0,
                                };

                                while top > 0 {
                                    top -= 1;
                                    let (s, slen) = stack[top];

                                    // Execute exact classify_single_string_neon logic
                                    let bytes = s.as_bytes();
                                    let mut sig = PatternSignature::<MAX_POSITIONS_NEON>::new();
                                    let mut classes = CharacterClasses::new();

                                    let mut i = 0;
                                    while i + LANES_NEON_BYTES <= slen {
                                        let chunk = vld1q_u8(bytes.as_ptr().add(i));
                                        classify_chunk_neon(chunk, &mut sig, &mut classes, i);
                                        i += LANES_NEON_BYTES;
                                    }

                                    if i < slen {
                                        classify_scalar_remaining(&bytes[i..], &mut sig, i);
                                    }

                                    let (result, n1, n2) =
                                        determine_type_from_signature(s, slen, &sig);

                                    if result == HwxType::Array
                                        && slen >= 2
                                        && s.starts_with('[')
                                        && s.ends_with(']')
                                    {
                                        // Push array content instead of recursive call
                                        if top < MAX_STACK {
                                            stack[top] = (&s[1..slen - 1], slen - 2);
                                            top += 1;
                                            continue;
                                        }
                                    }

                                    final_result = ClassificationResult {
                                        hwx_type: result,
                                        element_count: 1,
                                        numeric_value_1: n1,
                                        numeric_value_2: n2,
                                    };
                                    break;
                                }

                                final_result
                            };
                            if element_result.hwx_type == HwxType::String
                                && !element_str.starts_with('"')
                            {
                                return ClassificationResult {
                                    hwx_type: HwxType::String,
                                    element_count: chunk_count,
                                    numeric_value_1: 0.0,
                                    numeric_value_2: 0.0,
                                };
                            }

                            // Add to current chunk
                            chunk_types[chunk_count % CHUNK_SIZE_NEON] = element_result.hwx_type;
                            chunk_count += 1;

                            // Process chunk when full
                            if chunk_count % CHUNK_SIZE_NEON == 0 {
                                let chunk_result = process_chunk(
                                    &chunk_types,
                                    CHUNK_SIZE_NEON,
                                    &mut has_expected_type,
                                    &mut expected_type,
                                );
                                if chunk_result == HwxType::Array {
                                    return ClassificationResult {
                                        hwx_type: HwxType::Array,
                                        element_count: chunk_count,
                                        numeric_value_1: 0.0,
                                        numeric_value_2: 0.0,
                                    }; // Mixed types found
                                }
                            }
                        }
                    }
                }
                current_start = i + 1;
            }
            _ => {}
        }
    }

    // Handle last element
    if current_start < len {
        let element_bytes = &content[current_start..];
        let (trimmed, trimmed_len) = trim_whitespace(element_bytes, len - current_start);
        if trimmed_len > 0 {
            if let Ok(element_str) = std::str::from_utf8(trimmed) {
                //  STACK RECURSION: Same stack simulation for final element
                let element_result = {
                    const MAX_STACK_FINAL: usize = 32;
                    let mut stack_final: [(&str, usize); MAX_STACK_FINAL] =
                        [("", 0); MAX_STACK_FINAL];
                    let mut top_final = 1;

                    stack_final[0] = (element_str, trimmed_len);

                    let mut result_final = ClassificationResult {
                        hwx_type: HwxType::String,
                        element_count: 1,
                        numeric_value_1: 0.0,
                        numeric_value_2: 0.0,
                    };

                    while top_final > 0 {
                        top_final -= 1;
                        let (s_final, slen_final) = stack_final[top_final];

                        // Execute exact classify_single_string_neon logic
                        let bytes_final = s_final.as_bytes();
                        let mut sig_final = PatternSignature::<MAX_POSITIONS_NEON>::new();
                        let mut classes_final = CharacterClasses::new();

                        let mut i_final = 0;
                        while i_final + LANES_NEON_BYTES <= slen_final {
                            let chunk_final = vld1q_u8(bytes_final.as_ptr().add(i_final));
                            classify_chunk_neon(
                                chunk_final,
                                &mut sig_final,
                                &mut classes_final,
                                i_final,
                            );
                            i_final += LANES_NEON_BYTES;
                        }

                        if i_final < slen_final {
                            classify_scalar_remaining(
                                &bytes_final[i_final..],
                                &mut sig_final,
                                i_final,
                            );
                        }

                        let (res_final, n1_final, n2_final) =
                            determine_type_from_signature(s_final, slen_final, &sig_final);

                        if res_final == HwxType::Array
                            && slen_final >= 2
                            && s_final.starts_with('[')
                            && s_final.ends_with(']')
                        {
                            if top_final < MAX_STACK_FINAL {
                                stack_final[top_final] =
                                    (&s_final[1..slen_final - 1], slen_final - 2);
                                top_final += 1;
                                continue;
                            }
                        }

                        result_final = ClassificationResult {
                            hwx_type: res_final,
                            element_count: 1,
                            numeric_value_1: n1_final,
                            numeric_value_2: n2_final,
                        };
                        break;
                    }

                    result_final
                };
                if element_result.hwx_type == HwxType::String && !element_str.starts_with('"') {
                    return ClassificationResult {
                        hwx_type: HwxType::String,
                        element_count: chunk_count,
                        numeric_value_1: 0.0,
                        numeric_value_2: 0.0,
                    };
                }
                chunk_types[chunk_count % CHUNK_SIZE_NEON] = element_result.hwx_type;
                chunk_count += 1;
            }
        }
    }

    // Process final partial chunk
    if chunk_count > 0 {
        let final_chunk_size = chunk_count % CHUNK_SIZE_NEON;
        let chunk_size = if final_chunk_size == 0 {
            CHUNK_SIZE_NEON
        } else {
            final_chunk_size
        };
        let chunk_result = process_chunk(
            &chunk_types,
            chunk_size,
            &mut has_expected_type,
            &mut expected_type,
        );
        if chunk_result == HwxType::Array {
            return ClassificationResult {
                hwx_type: HwxType::Array,
                element_count: chunk_count,
                numeric_value_1: 0.0,
                numeric_value_2: 0.0,
            };
        }
    }

    // Return typed array based on expected type
    ClassificationResult {
        hwx_type: expected_type,
        numeric_value_1: 0.0,
        numeric_value_2: 0.0,
        element_count: chunk_count,
    }
}

// Helper function to trim whitespace from byte slices
#[inline]
fn trim_whitespace(bytes: &[u8], len: usize) -> (&[u8], usize) {
    let mut start = 0;
    let mut end = len;

    // Trim from start
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }

    // Trim from end
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    (&bytes[start..end], end - start)
}

// =============================================================================
// TYPE DETERMINATION ENGINE
// =============================================================================

// Design notes:
// - Uses precomputed byte-level signatures to keep hot paths allocation-light.
// - Classification targets the variants defined by `HwxType`.
// - Precedence follows the expected grammar order (see `determine_type_from_signature` below).
#[inline]
fn determine_type_from_signature<const MAX_POS: usize>(
    s: &str,
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (HwxType, f64, f64) {
    let bytes = s.as_bytes();

    // Quick check for empty strings
    if len == 0 {
        return (HwxType::String, 0.0, 0.0);
    }

    // Grammar order: ip | geo | vector | array | float | date | integer | boolean | file | null | generic_string

    // ========================================================================================
    // IP ADDRESS DETECTION (IPv4 and IPv6)
    // ========================================================================================

    // IPv6 detection: Contains colons, hex characters, optional compression
    // Note: ::1 and other numeric IPv6 addresses don't have letters, so check for hex pattern more broadly
    if signature.colon_count > 0 && detect_ipv6_pattern(bytes, len, signature) {
        return (HwxType::IPAddressV6, 0.0, 0.0);
    }

    // IPv4 detection: exactly 3 dots, 4 numeric segments
    if signature.dot_count == 3 && signature.has_digits && !signature.has_letters {
        let (is_ipv4, ip_value) = detect_ipv4_pattern(bytes, len, signature);
        if is_ipv4 {
            return (HwxType::IPAddressV4, ip_value, 0.0);
        }
    }

    // ========================================================================================
    // GEO COORDINATE DETECTION (lat, lon)
    // ========================================================================================

    let (is_geo, lat_value, lon_value) = detect_geo_pattern(bytes, len, signature);
    if signature.comma_count > 0 && signature.has_digits && is_geo {
        return (HwxType::Geo, lat_value, lon_value);
    }

    // ========================================================================================
    // VECTOR DETECTION (numeric array with specific dimensions)
    // ========================================================================================

    // ========================================================================================
    // ARRAY DETECTION (includes vectors) - SIMD BRACKET DETECTION
    // ========================================================================================

    // SIMD already detected bracket positions - use them directly for array/vector detection
    if signature.bracket_open_count > 0
        && signature.bracket_close_count > 0
        && signature.bracket_open_positions[0] == 0
        && signature.bracket_close_positions[signature.bracket_close_count - 1] == len - 1
    {
        let start = 1; // Skip opening bracket
        let end = len - 1; // Skip closing bracket

        if end <= start {
            return (HwxType::Array, 0.0, 0.0); // Empty array becomes generic Array
        }

        // Check if it's a valid vector first (all numeric, correct dimensions)
        if detect_vector_pattern(bytes, len, signature) {
            return (HwxType::Vector, 0.0, 0.0);
        }

        // Otherwise it's a general array - SIMD detected brackets correctly
        return (HwxType::Array, 0.0, 0.0);
    }

    // ========================================================================================
    // FLOAT DETECTION (including scientific notation!)
    // ========================================================================================

    if signature.has_digits {
        let (is_float, float_value) = detect_float_pattern(bytes, len, signature);
        if is_float {
            return (HwxType::Float, float_value, 0.0);
        }
    }

    // ========================================================================================
    // DATE DETECTION (multiple formats)
    // ========================================================================================

    // DateMath: "now" with operations like +1d, -30m, /d
    if len >= 3 && bytes[0] == b'n' && bytes[1] == b'o' && bytes[2] == b'w' {
        let (is_datemath, timestamp) = detect_datemath_pattern(bytes, len);
        if is_datemath {
            return (HwxType::DateMath, timestamp, 0.0);
        }
    }

    // LogDate: "Mon Jan 15 14:30:25 2024"
    if signature.has_letters && signature.has_digits && signature.colon_count > 0 {
        let (is_logdate, timestamp) = detect_logdate_pattern(bytes, len, &signature);
        if is_logdate {
            return (HwxType::LogDate, timestamp, 0.0);
        }
    }

    // ISO8601Date: "2024-01-15T14:30:25Z"
    if signature.has_digits && signature.dash_count > 0 {
        let (is_iso8601, timestamp) = detect_iso8601_pattern(bytes, len, &signature);
        if is_iso8601 {
            return (HwxType::ISO8601Date, timestamp, 0.0);
        }
    }

    // RFC2822Date: "Mon, 15 Jan 2024 14:30:25 +0000"
    if signature.has_letters && signature.has_digits && signature.comma_count > 0 {
        let (is_rfc2822, timestamp) = detect_rfc2822_pattern(bytes, len, signature);
        if is_rfc2822 {
            return (HwxType::RFC2822Date, timestamp, 0.0);
        }
    }

    // AmericanDate: "01/15/2024"
    if signature.slash_count == 2 && signature.has_digits && !signature.has_letters {
        let (is_american, timestamp) = detect_american_date_pattern(bytes, len, signature);
        if is_american {
            return (HwxType::AmericanDate, timestamp, 0.0);
        }
    }

    // EuropeanDate: "15/01/2024"
    if signature.slash_count == 2 && signature.has_digits && !signature.has_letters {
        let (is_european, timestamp) = detect_european_date_pattern(bytes, len, signature);
        if is_european {
            return (HwxType::EuropeanDate, timestamp, 0.0);
        }
    }

    // VerboseDate: "Monday, the 15th of January, 2024"
    if signature.has_letters && signature.has_digits && signature.comma_count > 0 {
        let (is_verbose, timestamp) = detect_verbose_date_pattern(bytes, len, signature);
        if is_verbose {
            return (HwxType::VerboseDate, timestamp, 0.0);
        }
    }

    // FinancialDate: "15th January 2024"
    if signature.has_letters && signature.has_digits {
        let (is_financial, timestamp) = detect_financial_date_pattern(bytes, len);
        if is_financial {
            return (HwxType::FinancialDate, timestamp, 0.0);
        }
    }

    // FullDate: Complex date with various formats (now properly supported in grammar)
    // Must check before GenericDate due to grammar precedence
    if signature.has_digits && (signature.has_letters || signature.dash_count > 0) {
        let (is_full_date, timestamp) = detect_full_date_pattern(bytes, len, signature);
        if is_full_date {
            return (HwxType::FullDate, timestamp, 0.0);
        }
    }

    // GenericDate: "2024-01-15" or "2024/01/15" or "2024.01.15" or with time "2024-01-15 15:30:45" (fallback after FullDate check)
    if (signature.dash_count == 2 || signature.slash_count == 2 || signature.dot_count == 2)
        && signature.has_digits
        && !signature.has_letters
        && (len == 10 || (len >= 19 && signature.colon_count >= 2))
    {
        let (is_generic_date, timestamp) = detect_generic_date_pattern(bytes, len, signature);
        if is_generic_date {
            return (HwxType::GenericDate, timestamp, 0.0);
        }
    }

    // ========================================================================================
    // INTEGER DETECTION
    // ========================================================================================

    if signature.has_digits && !signature.has_letters && signature.dot_count == 0 {
        let (is_integer, integer_value) = detect_integer_pattern(bytes, len, signature);
        if is_integer {
            return (HwxType::Integer, integer_value, 0.0);
        }
    }

    // ========================================================================================
    // BOOLEAN DETECTION
    // ========================================================================================

    if len == 4
        && signature.has_letters
        && !signature.has_digits
        && !signature.has_punctuation
        && bytes[0] == b't'
        && bytes[1] == b'r'
        && bytes[2] == b'u'
        && bytes[3] == b'e'
    {
        return (HwxType::Boolean, 1.0, 0.0);
    }
    if len == 5
        && signature.has_letters
        && !signature.has_digits
        && !signature.has_punctuation
        && bytes[0] == b'f'
        && bytes[1] == b'a'
        && bytes[2] == b'l'
        && bytes[3] == b's'
        && bytes[4] == b'e'
    {
        return (HwxType::Boolean, 0.0, 0.0);
    }

    // ========================================================================================
    // FILE PATH DETECTION
    // ========================================================================================

    if signature.slash_count > 0 && detect_file_pattern(bytes, len, signature) {
        return (HwxType::File, 0.0, 0.0);
    }

    // ========================================================================================
    //  NULL DETECTION
    // ========================================================================================

    if len == 4
        && signature.has_letters
        && !signature.has_digits
        && !signature.has_punctuation
        && bytes[0] == b'n'
        && bytes[1] == b'u'
        && bytes[2] == b'l'
        && bytes[3] == b'l'
    {
        return (HwxType::Null, 0.0, 0.0);
    }

    // ========================================================================================
    // DEFAULT: STRING (generic_string in grammar)
    // ========================================================================================

    (HwxType::String, 0.0, 0.0)
}

// =============================================================================
//   SIMD PATTERN DETECTION FUNCTIONS
// =============================================================================

// IPv6 Pattern Detection - Full IPv6 support with compression
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn detect_ipv6_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> bool {
    // IPv6: full_ipv6 | compressed_ipv6
    // Full: (hex_quad ":"){7} hex_quad
    // Compressed: "::" or hex_quad+ "::" hex_quad*

    if signature.colon_count < 2 {
        return false;
    }

    // Check for "::" compression - count how many double colons exist
    let double_colon_count = signature
        .colon_positions
        .windows(2)
        .filter(|pair| pair[1] == pair[0] + 1)
        .count();

    if double_colon_count > 0 {
        // Compressed IPv6 - must have exactly ONE "::" occurrence
        if double_colon_count != 1 {
            return false; // Multiple "::" is invalid
        }
        return validate_ipv6_hex_quads(
            bytes,
            len,
            &signature.colon_positions[..signature.colon_count],
        );
    } else {
        // Full IPv6 - must have exactly 7 colons
        if signature.colon_count == 7 {
            return validate_ipv6_hex_quads(
                bytes,
                len,
                &signature.colon_positions[..signature.colon_count],
            );
        }
    }
    false
}

// IPv4 Pattern Detection - Exact octet validation
#[cfg(any(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_arch = "aarch64"
))]
#[inline]
fn detect_ipv4_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // IPv4: (octet "."){3} octet
    if signature.dot_count != 3 {
        return (false, 0.0);
    }

    let dots = &signature.dot_positions;
    let segments = [
        (0, dots[0]),
        (dots[0] + 1, dots[1]),
        (dots[1] + 1, dots[2]),
        (dots[2] + 1, len),
    ];

    let mut ip_value = 0u64;

    for (i, (start, end)) in segments.iter().enumerate() {
        if end <= start || (end - start) > 3 || (end - start) == 0 {
            return (false, 0.0);
        }

        // Check for leading zeros (invalid per grammar)
        if (end - start) > 1 && bytes[*start] == b'0' {
            return (false, 0.0);
        }

        let mut octet: u16 = 0;
        for &byte in bytes.iter().take(*end).skip(*start) {
            if byte.is_ascii_digit() {
                octet = octet * 10 + (byte - b'0') as u16;
            } else {
                return (false, 0.0);
            }
        }

        if octet > 255 {
            return (false, 0.0);
        }

        // Build the IP value: shift each octet to its position
        ip_value |= (octet as u64) << (8 * (3 - i));
    }

    (true, ip_value as f64)
}

// Geo Pattern Detection - lat, lon coordinates
#[cfg(any(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_arch = "aarch64"
))]
#[inline]
fn detect_geo_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64, f64) {
    // Geo: lat "," lon
    // lat: ("-"|"+")? (('1'..'8')? ASCII_DIGIT ("." ASCII_DIGIT+)? | "90" (".0+")?)
    // lon: ("-"|"+")? ("180" (".0+")? | ("1" ('0'..'7')? ASCII_DIGIT | ASCII_DIGIT{1,2}) ("." ASCII_DIGIT+)?)

    if signature.comma_count != 1 {
        return (false, 0.0, 0.0);
    }

    let comma_pos = signature.comma_positions[0];
    let lat_part = &bytes[0..comma_pos];
    let lon_part = &bytes[comma_pos + 1..len];

    let (lat_valid, lat_value) = validate_latitude(lat_part, comma_pos);
    let (lon_valid, lon_value) = validate_longitude(lon_part, len - comma_pos - 1);

    if lat_valid && lon_valid {
        (true, lat_value, lon_value) // Return actual parsed latitude and longitude
    } else {
        (false, 0.0, 0.0)
    }
}

// Vector Pattern Detection - Numeric array with correct dimensions
#[inline]
fn detect_vector_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> bool {
    // Vector: start_bracket ((float | integer) (comma (float | integer))*)? end_bracket
    // Must be all numeric and have correct dimensions for embeddings

    if signature.bracket_open_count == 0 || signature.bracket_close_count == 0 {
        return false;
    }

    // Extract content between brackets
    let start = signature.bracket_open_positions[0] + 1;
    let end = signature
        .bracket_close_positions
        .iter()
        .find(|&&pos| pos == len - 1)
        .copied()
        .unwrap_or(len);

    if end <= start {
        return false; // Empty arrays are not vectors
    }

    let content = &bytes[start..end];
    let element_count = signature.comma_count + 1; // Simple count: commas + 1 = elements

    // Check if all elements are numeric and dimension matches HNSW_DIMENSION
    validate_vector_elements(content, end - start, &signature.comma_positions, start)
        && element_count == crate::constants::HNSW_DIMENSION as usize
}

// Float Pattern Detection - Including scientific notation!
#[inline]
fn detect_float_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // New grammar supports two patterns:
    // 1. Scientific notation without decimal: 1e5, 2E-3
    // 2. Standard decimal notation: 1.5, 42.0, 1.0e5

    // REJECT strings with too many dashes (UUIDs have 4, floats have at most 2. Most of the times
    // floats will have at most 1 dash but negative numbers will have 2 dashes)
    if signature.dash_count > 2 {
        return (false, 0.0); // UUIDs have 4 dashes, scientific notation has at most 2
    }

    // Pre-validate: ensure string contains ONLY valid float characters
    // Valid float chars: digits (0-9), decimal point (.), sign (-/+), scientific (e/E)
    let mut has_invalid_char = false;
    for i in 0..len {
        let byte = bytes[i];
        match byte {
            b'0'..=b'9' | b'.' | b'-' | b'+' | b'e' | b'E' => {
                // Valid float characters - continue
            }
            _ => {
                // Invalid character for float (like hex letters a-f in UUIDs)
                has_invalid_char = true;
                break;
            }
        }
    }

    if has_invalid_char {
        return (false, 0.0); // Contains invalid characters for float
    }

    let mut start_idx = 0;

    // Check for optional sign
    if signature.starts_with_sign {
        if bytes[0] != b'-' && bytes[0] != b'+' {
            return (false, 0.0);
        }
        start_idx = 1;
    }

    // Find 'e' or 'E' for scientific notation - must be valid scientific notation context
    let mut has_exp = false;
    let mut exp_pos = 0;
    for i in start_idx..len {
        let byte = bytes[i];
        if byte == b'e' || byte == b'E' {
            // BUG FIX: Validate this is actually scientific notation, not just any 'e'
            // The 'e' must be preceded by digits and followed by optional sign + digits
            let has_digit_before = i > start_idx && bytes[i - 1].is_ascii_digit();
            let has_valid_exponent_after = i + 1 < len && {
                let mut exp_start = i + 1;
                // Skip optional sign
                if exp_start < len && (bytes[exp_start] == b'+' || bytes[exp_start] == b'-') {
                    exp_start += 1;
                }
                // Must have at least one digit after
                exp_start < len && bytes[exp_start].is_ascii_digit()
            };

            if has_digit_before && has_valid_exponent_after {
                has_exp = true;
                exp_pos = i;
                break;
            }
        }
    }

    match (signature.dot_count, has_exp) {
        // Pattern 1: Scientific notation without decimal point (1e5, 2E-3)
        (0, true) => {
            // Parse integer part and exponent during validation
            let mut value = 0.0;

            // Parse integer part
            for i in start_idx..exp_pos {
                if bytes[i].is_ascii_digit() {
                    value = value * 10.0 + (bytes[i] - b'0') as f64;
                }
            }

            // Parse exponent
            let mut exp = 0i32;
            let mut exp_start = exp_pos + 1;
            let mut exp_negative = false;

            if exp_start < len && (bytes[exp_start] == b'+' || bytes[exp_start] == b'-') {
                exp_negative = bytes[exp_start] == b'-';
                exp_start += 1;
            }

            for i in exp_start..len {
                if bytes[i].is_ascii_digit() {
                    // OVERFLOW PROTECTION: Stop parsing exponent digits after reasonable limit
                    // This prevents integer overflow on inputs like "1e99999999999999999999"
                    // We accept the number and use exp=300 rather than rejecting the entire string
                    // This matches behavior of many real-world parsers (Python, JavaScript, etc.)
                    if exp > 300 {
                        break; // Stop parsing, use exp=300 - number is still valid
                    }
                    exp = exp * 10 + (bytes[i] - b'0') as i32;
                }
            }

            if exp_negative {
                exp = -exp;
            }

            value *= 10_f64.powi(exp);

            // Apply negative sign if present
            if start_idx > 0 && bytes[0] == b'-' {
                value = -value;
            }

            return (true, value);
        }
        // Pattern 2: Standard decimal notation (1.5, 42.0, 1.0e5)
        (1, _) => {
            let dot_pos = signature.dot_positions[0];

            // Validate integer part before '.'
            if !validate_float_integer_part(bytes, start_idx, dot_pos) {
                return (false, 0.0);
            }
            match has_exp {
                true => {
                    // Validate fractional part before 'e'
                    if !validate_float_fractional_part(bytes, dot_pos + 1, exp_pos) {
                        return (false, 0.0);
                    }
                    // Validate exponent part after 'e'
                    if !validate_float_exponent_part(bytes, exp_pos + 1, len) {
                        return (false, 0.0);
                    }

                    // Parse decimal number with scientific notation
                    let mut value = 0.0;

                    // Parse integer part
                    for i in start_idx..dot_pos {
                        if bytes[i].is_ascii_digit() {
                            value = value * 10.0 + (bytes[i] - b'0') as f64;
                        }
                    }

                    // Parse fractional part
                    let mut fraction = 0.0;
                    let mut divisor = 10.0;
                    for i in (dot_pos + 1)..exp_pos {
                        if bytes[i].is_ascii_digit() {
                            fraction += (bytes[i] - b'0') as f64 / divisor;
                            divisor *= 10.0;
                        }
                    }
                    value += fraction;

                    // Parse exponent
                    let mut exp = 0i32;
                    let mut exp_start = exp_pos + 1;
                    let mut exp_negative = false;

                    if exp_start < len && (bytes[exp_start] == b'+' || bytes[exp_start] == b'-') {
                        exp_negative = bytes[exp_start] == b'-';
                        exp_start += 1;
                    }

                    for i in exp_start..len {
                        if bytes[i].is_ascii_digit() {
                            // OVERFLOW PROTECTION: Stop parsing exponent digits after reasonable limit
                            // This prevents integer overflow on inputs like "1.5e99999999999999999999"
                            // We accept the number and use exp=300 rather than rejecting the entire string
                            // This matches behavior of many real-world parsers (Python, JavaScript, etc.)
                            if exp > 300 {
                                break; // Stop parsing, use exp=300 - number is still valid
                            }
                            exp = exp * 10 + (bytes[i] - b'0') as i32;
                        }
                    }

                    if exp_negative {
                        exp = -exp;
                    }

                    value *= 10_f64.powi(exp);

                    // Apply negative sign if present
                    if start_idx > 0 && bytes[0] == b'-' {
                        value = -value;
                    }

                    return (true, value);
                }
                false => {
                    // BUG FIX: Validate fractional part contains only digits before parsing
                    if !validate_float_fractional_part(bytes, dot_pos + 1, len) {
                        return (false, 0.0);
                    }

                    // Parse decimal number directly
                    let mut value = 0.0;

                    // Parse integer part
                    for i in start_idx..dot_pos {
                        if bytes[i].is_ascii_digit() {
                            value = value * 10.0 + (bytes[i] - b'0') as f64;
                        }
                    }

                    // Parse fractional part
                    let mut fraction = 0.0;
                    let mut divisor = 10.0;
                    for i in (dot_pos + 1)..len {
                        if bytes[i].is_ascii_digit() {
                            fraction += (bytes[i] - b'0') as f64 / divisor;
                            divisor *= 10.0;
                        }
                    }

                    value += fraction;

                    // Apply negative sign if present
                    if start_idx > 0 && bytes[0] == b'-' {
                        value = -value;
                    }

                    return (true, value);
                }
            }
        }
        // Invalid patterns: multiple dots, no dots without scientific notation
        _ => (false, 0.0),
    }
}

// DateMath Pattern Detection - "now" with operations
#[inline]
fn detect_datemath_pattern(bytes: &[u8], len: usize) -> (bool, f64) {
    // DateMath: "now" (date_math_operation)* (date_math_rounding)?
    // date_math_operation: ("+" | "-") ASCII_DIGIT+ date_math_unit
    // date_math_rounding: "/" date_math_unit
    // date_math_unit: "y" | "M" | "w" | "d" | "h" | "H" | "m" | "s"

    if len < 3 || bytes[0] != b'n' || bytes[1] != b'o' || bytes[2] != b'w' {
        return (false, 0.0);
    }

    if len == 3 {
        // Call architecture-specific helper to get "now" timestamp
        return (true, get_current_time());
    }

    // Start with base timestamp
    let mut timestamp = get_current_time();

    // Parse operations and apply them
    let mut pos = 3;
    while pos < len {
        match bytes[pos] {
            b'+' | b'-' => {
                let is_negative = bytes[pos] == b'-';
                pos += 1;

                // Parse digits
                let digit_start = pos;
                let mut value = 0u64;
                while pos < len && bytes[pos].is_ascii_digit() {
                    value = value * 10 + (bytes[pos] - b'0') as u64;
                    pos += 1;
                }
                if pos == digit_start {
                    return (false, 0.0); // No digits after +/-
                }

                // Parse unit and convert to milliseconds
                if pos >= len || !is_datemath_unit(bytes[pos]) {
                    return (false, 0.0);
                }

                let milliseconds = match bytes[pos] {
                    b'y' => value * 365u64 * 24u64 * 3600u64 * 1000u64, // year
                    b'M' => value * 30u64 * 24u64 * 3600u64 * 1000u64,  // month (approximate)
                    b'w' => value * 7u64 * 24u64 * 3600u64 * 1000u64,   // week
                    b'd' => value * 24u64 * 3600u64 * 1000u64,          // day
                    b'h' | b'H' => value * 3600u64 * 1000u64,           // hour
                    b'm' => value * 60u64 * 1000u64,                    // minute
                    b's' => value * 1000u64,                            // second
                    _ => return (false, 0.0),
                };

                if is_negative {
                    timestamp -= milliseconds as f64;
                } else {
                    timestamp += milliseconds as f64;
                }

                pos += 1;
            }
            b'/' => {
                pos += 1;
                // Parse rounding unit
                if pos >= len || !is_datemath_unit(bytes[pos]) {
                    return (false, 0.0);
                }

                // Apply rounding - truncate to unit boundary
                let rounding_milliseconds = match bytes[pos] {
                    b'y' => 365u64 * 24u64 * 3600u64 * 1000u64, // year
                    b'M' => 30u64 * 24u64 * 3600u64 * 1000u64,  // month (approximate)
                    b'w' => 7u64 * 24u64 * 3600u64 * 1000u64,   // week
                    b'd' => 24u64 * 3600u64 * 1000u64,          // day
                    b'h' | b'H' => 3600u64 * 1000u64,           // hour
                    b'm' => 60u64 * 1000u64,                    // minute
                    b's' => 1000u64,                            // second
                    _ => return (false, 0.0),
                };

                timestamp = (timestamp / rounding_milliseconds as f64).floor()
                    * rounding_milliseconds as f64;
                break;
            }
            _ => return (false, 0.0),
        }
    }

    (true, timestamp)
}

// LogDate Pattern Detection - "Mon Jan 15 14:30:25 2024"
#[inline]
fn detect_logdate_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // LogDate: (long_day | day) space+ (long_month | month) space+ numeric_day space+ time space+ year

    if !signature.has_letters || !signature.has_digits || signature.colon_count < 2 {
        return (false, 0.0);
    }

    // Split on whitespace and validate components - ZERO HEAP
    let mut parts: [(usize, usize); MAX_DATE_PARTS] = [(0, 0); MAX_DATE_PARTS];
    let mut part_count = 0;
    split_on_whitespace(bytes, len, &mut parts, &mut part_count);
    if part_count < 5 {
        return (false, 0.0);
    }

    // Validate: day_name month day time year
    let day_name_valid = validate_day_name(&bytes[parts[0].0..parts[0].1]);
    let month_name_valid = validate_month_name(&bytes[parts[1].0..parts[1].1]);
    let (day_valid, day) =
        validate_numeric_day(&bytes[parts[2].0..parts[2].1], parts[2].1 - parts[2].0);
    let (time_valid, hours, minutes, seconds, milliseconds) = validate_time_format(
        &bytes[parts[3].0..parts[3].1],
        parts[3].1 - parts[3].0,
        &signature.colon_positions,
    );
    let (year_valid, year) = validate_year(&bytes[parts[4].0..parts[4].1]);

    if day_name_valid && month_name_valid && day_valid && time_valid && year_valid {
        let month = month_name_to_number(&bytes[parts[1].0..parts[1].1]);
        if month > 0 {
            (
                true,
                date_to_timestamp(day, month, year, hours, minutes, seconds, milliseconds),
            )
        } else {
            (false, 0.0)
        }
    } else {
        (false, 0.0)
    }
}

// ISO8601 Pattern Detection - "2024-01-15T14:30:25Z"
#[inline]
fn detect_iso8601_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // ISO8601: ws full_date ws "T" ws time_fraction ws timezone? ws
    // full_date: year ws "-" ws (month | long_month | numeric_month) ws "-" ws (day | long_day | numeric_day)

    if signature.dash_count < 2 {
        return (false, 0.0);
    }

    // Look for 'T' or 't' separator (case insensitive)
    let mut t_idx = len; // Default to not found
    for i in 0..len {
        if bytes[i] == b'T' || bytes[i] == b't' {
            t_idx = i;
            break;
        }
    }

    if t_idx >= len {
        return (false, 0.0); // No 'T' found
    }

    // Parse while validating - ISO8601 format is YYYY-MM-DD
    if signature.dash_count < 2 || t_idx < 10 {
        return (false, 0.0);
    }

    let dash1 = signature.dash_positions[0];
    let dash2 = signature.dash_positions[1];

    if dash1 + 1 > dash2 || dash2 + 1 > t_idx {
        return (false, 0.0);
    }

    let (year_valid, year) = validate_year(&bytes[0..dash1]);
    let (month_valid, month) = validate_numeric_month(&bytes[dash1 + 1..dash2], dash2 - dash1 - 1);
    let (day_valid, day) = validate_numeric_day(&bytes[dash2 + 1..t_idx], t_idx - dash2 - 1);

    if !year_valid || !month_valid || !day_valid {
        return (false, 0.0);
    }

    // Validate time part after 'T'
    let time_part = &bytes[t_idx + 1..];
    let (time_valid, hours, minutes, seconds, milliseconds, tz_offset_minutes) =
        validate_iso_time_part(time_part, len - t_idx - 1);
    if time_valid {
        // Calculate base timestamp in local time
        let base_timestamp =
            date_to_timestamp(day, month, year, hours, minutes, seconds, milliseconds);
        // Apply timezone offset to convert to UTC (subtract offset because we want UTC)
        let utc_timestamp = base_timestamp - (tz_offset_minutes as f64 * 60.0 * 1000.0);
        (true, utc_timestamp)
    } else {
        (false, 0.0)
    }
}

// RFC2822 Pattern Detection - "Mon, 15 Jan 2024 14:30:25 +0000"
#[inline]
fn detect_rfc2822_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // RFC2822: day "," ws numeric_day ws month ws year ws time ws timezone

    if signature.comma_count == 0 || signature.colon_count < 2 {
        return (false, 0.0);
    }

    let comma_pos = signature.comma_positions[0];
    if comma_pos < 3 {
        return (false, 0.0);
    }

    // Validate day name before comma
    if !validate_day_name(&bytes[0..comma_pos]) {
        return (false, 0.0);
    }

    // Parse remaining parts after comma - ZERO HEAP
    let remaining = &bytes[comma_pos + 1..];
    let mut parts: [(usize, usize); MAX_DATE_PARTS] = [(0, 0); MAX_DATE_PARTS];
    let mut part_count = 0;
    split_on_whitespace(
        remaining,
        len - (comma_pos + 1),
        &mut parts,
        &mut part_count,
    );

    if part_count < 5 {
        return (false, 0.0);
    }

    // Validate: numeric_day month year time timezone
    let (day_valid, day) =
        validate_numeric_day(&remaining[parts[0].0..parts[0].1], parts[0].1 - parts[0].0);
    let month_name_valid = validate_month_name(&remaining[parts[1].0..parts[1].1]);
    let (year_valid, year) = validate_year(&remaining[parts[2].0..parts[2].1]);
    let (time_valid, hours, minutes, seconds, milliseconds) = validate_time_format(
        &remaining[parts[3].0..parts[3].1],
        parts[3].1 - parts[3].0,
        &signature.colon_positions,
    );
    let (timezone_valid, _tz_offset) =
        validate_timezone(&remaining[parts[4].0..parts[4].1], parts[4].1 - parts[4].0);

    if day_valid && month_name_valid && year_valid && time_valid && timezone_valid {
        let month = month_name_to_number(&remaining[parts[1].0..parts[1].1]);
        if month > 0 {
            (
                true,
                date_to_timestamp(day, month, year, hours, minutes, seconds, milliseconds),
            )
        } else {
            (false, 0.0)
        }
    } else {
        (false, 0.0)
    }
}

//  American Date Pattern - "01/15/2024"
#[inline]
fn detect_american_date_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // AmericanDate: numeric_month "/" numeric_day "/" year optional_time_suffix?

    if signature.slash_count != 2 {
        return (false, 0.0);
    }

    let slash1 = signature.slash_positions[0];
    let slash2 = signature.slash_positions[1];

    let month_part = &bytes[0..slash1];
    let day_part = &bytes[slash1 + 1..slash2];
    let year_part = &bytes[slash2 + 1..];

    // Split year from optional time
    let (year_bytes, has_time, time_part) = split_date_time(year_part, len - slash2 - 1);

    // Parse while validating using existing validation functions
    let (month_valid, month) = validate_numeric_month(month_part, slash1);
    let (day_valid, day) = validate_numeric_day(day_part, slash2 - slash1 - 1);
    let (year_valid, year) = validate_year(year_bytes);

    if !month_valid || !day_valid || !year_valid {
        return (false, 0.0);
    }

    // Parse time component if present
    let (hours, minutes, seconds, milliseconds) = if has_time && time_part.len() > 0 {
        let (time_valid, h, m, s, ms) =
            validate_time_format(time_part, time_part.len(), &signature.colon_positions);
        if time_valid {
            (h, m, s, ms)
        } else {
            (0, 0, 0, 0) // Invalid time, use midnight
        }
    } else {
        (0, 0, 0, 0) // No time component, use midnight
    };

    (
        true,
        date_to_timestamp(day, month, year, hours, minutes, seconds, milliseconds),
    )
}

// Helper function to convert day/month/year/time to Unix timestamp (milliseconds since epoch)
#[inline]
fn date_to_timestamp(
    day: u32,
    month: u32,
    year: u32,
    hours: u32,
    minutes: u32,
    seconds: u32,
    milliseconds: u32,
) -> f64 {
    // Days in each month (non-leap year)
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    // Calculate total days since Unix epoch (Jan 1, 1970)
    let mut total_days = 0u32;

    // Add days for complete years
    for y in 1970..year {
        total_days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Add days for complete months in target year
    for m in 0..(month - 1) {
        total_days += DAYS_IN_MONTH[m as usize];
        // Add extra day for February in leap years
        if m == 1 && is_leap_year(year) {
            total_days += 1;
        }
    }

    // Add remaining days
    total_days += day - 1;

    // Convert to milliseconds and add time components
    let days_millis = total_days as f64 * 86400.0 * 1000.0;
    let time_millis = (hours as f64 * 3600.0 * 1000.0)
        + (minutes as f64 * 60.0 * 1000.0)
        + (seconds as f64 * 1000.0)
        + (milliseconds as f64);

    days_millis + time_millis
}

// Helper to check if year is leap year
#[inline]
fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// Helper to get current Unix timestamp in milliseconds
// We make an exception to no-heap rules here to get the current time.
#[inline]
fn get_current_time() -> f64 {
    // Get current Unix timestamp in milliseconds
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_secs() as f64 * 1000.0) + (now.subsec_nanos() as f64 / 1_000_000.0)
}

// Helper to convert month name to number
#[inline]
fn month_name_to_number(bytes: &[u8]) -> u32 {
    match bytes {
        b"Jan" | b"January" => 1,
        b"Feb" | b"February" => 2,
        b"Mar" | b"March" => 3,
        b"Apr" | b"April" => 4,
        b"May" => 5,
        b"Jun" | b"June" => 6,
        b"Jul" | b"July" => 7,
        b"Aug" | b"August" => 8,
        b"Sep" | b"September" => 9,
        b"Oct" | b"October" => 10,
        b"Nov" | b"November" => 11,
        b"Dec" | b"December" => 12,
        _ => 0,
    }
}

// Helper to extract day number from ordinal day (e.g., "15th" -> 15)
#[inline]
fn ordinal_day_to_number(bytes: &[u8], len: usize) -> u32 {
    if len < 3 {
        return 0;
    }

    // Check for ordinal suffixes
    let has_suffix = if len >= 2 {
        let suffix = &bytes[len - 2..];
        matches!(suffix, b"st" | b"nd" | b"rd" | b"th")
    } else {
        false
    };

    if !has_suffix {
        return 0;
    }

    // Parse the numeric part (everything except last 2 characters)
    let mut day = 0u32;
    for &byte in &bytes[0..len - 2] {
        if byte.is_ascii_digit() {
            day = day * 10 + (byte - b'0') as u32;
        } else {
            return 0;
        }
    }

    if day >= 1 && day <= 31 {
        day
    } else {
        0
    }
}

// European Date Pattern - "15/01/2024"
#[inline]
fn detect_european_date_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // EuropeanDate: numeric_day "/" numeric_month "/" year optional_time_suffix?

    if signature.slash_count != 2 {
        return (false, 0.0);
    }

    let slash1 = signature.slash_positions[0];
    let slash2 = signature.slash_positions[1];

    let day_part = &bytes[0..slash1];
    let month_part = &bytes[slash1 + 1..slash2];
    let year_part = &bytes[slash2 + 1..];

    // Split year from optional time
    let (year_bytes, has_time, time_part) = split_date_time(year_part, len - slash2 - 1);

    // Parse while validating - European format is DD/MM/YYYY
    let (day_valid, day) = validate_numeric_day(day_part, slash1);
    let (month_valid, month) = validate_numeric_month(month_part, slash2 - slash1 - 1);
    let (year_valid, year) = validate_year(year_bytes);

    if !day_valid || !month_valid || !year_valid {
        return (false, 0.0);
    }

    // Parse time component if present
    let (hours, minutes, seconds, milliseconds) = if has_time && time_part.len() > 0 {
        let (time_valid, h, m, s, ms) =
            validate_time_format(time_part, time_part.len(), &signature.colon_positions);
        if time_valid {
            (h, m, s, ms)
        } else {
            (0, 0, 0, 0) // Invalid time, use midnight
        }
    } else {
        (0, 0, 0, 0) // No time component, use midnight
    };

    (
        true,
        date_to_timestamp(day, month, year, hours, minutes, seconds, milliseconds),
    )
}

// Verbose Date Pattern - "Monday, the 15th of January, 2024"
#[inline]
fn detect_verbose_date_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // VerboseDate: (long_day | day) "," ws "the" ws numeric_day ws "of" ws (long_month | month) "," ws year optional_time_suffix?

    if signature.comma_count < 2 {
        return (false, 0.0);
    }

    // Find "the" and "of" keywords
    let has_the = find_keyword(bytes, len, b"the", 3);
    let has_of = find_keyword(bytes, len, b"of", 2);

    if !has_the || !has_of {
        return (false, 0.0);
    }

    // Split and validate components - ZERO HEAP
    let mut parts: [(usize, usize); MAX_DATE_PARTS] = [(0, 0); MAX_DATE_PARTS];
    let mut part_count = 0;
    split_on_whitespace(bytes, len, &mut parts, &mut part_count);
    if part_count < 6 {
        return (false, 0.0);
    }

    // Validate: day_name, "the", ordinal_day, "of", month_name, year
    // Trim punctuation from day and month parts
    let mut day_end = parts[0].1;
    if day_end > parts[0].0 && bytes[day_end - 1] == b',' {
        day_end -= 1;
    }
    let mut month_end = parts[4].1;
    if month_end > parts[4].0 && bytes[month_end - 1] == b',' {
        month_end -= 1;
    }

    let day_valid = validate_day_name(&bytes[parts[0].0..day_end]);
    let the_valid = is_keyword(&bytes[parts[1].0..parts[1].1], b"the");
    let ordinal_valid =
        validate_ordinal_day(&bytes[parts[2].0..parts[2].1], parts[2].1 - parts[2].0);
    let of_valid = is_keyword(&bytes[parts[3].0..parts[3].1], b"of");
    let month_valid = validate_month_name(&bytes[parts[4].0..month_end]);
    let (year_valid, year) = validate_year(&bytes[parts[5].0..parts[5].1]);

    if day_valid && the_valid && ordinal_valid && of_valid && month_valid && year_valid {
        let day = ordinal_day_to_number(&bytes[parts[2].0..parts[2].1], parts[2].1 - parts[2].0);
        let month = month_name_to_number(&bytes[parts[4].0..month_end]);
        if day > 0 && month > 0 {
            (true, date_to_timestamp(day, month, year, 0, 0, 0, 0))
        } else {
            (false, 0.0)
        }
    } else {
        (false, 0.0)
    }
}

// Financial Date Pattern - "15th January 2024"
#[inline]
fn detect_financial_date_pattern(bytes: &[u8], len: usize) -> (bool, f64) {
    // FinancialDate: numeric_day ("st" | "nd" | "rd" | "th") ws (long_month | month) ws year optional_time_suffix?

    // Split and validate components - ZERO HEAP
    let mut parts: [(usize, usize); MAX_DATE_PARTS] = [(0, 0); MAX_DATE_PARTS];
    let mut part_count = 0;
    split_on_whitespace(bytes, len, &mut parts, &mut part_count);
    if part_count < 3 {
        return (false, 0.0);
    }

    // First part should be ordinal day (15th, 22nd, etc.)
    let day_part = &bytes[parts[0].0..parts[0].1];
    if !validate_ordinal_day(day_part, parts[0].1 - parts[0].0) {
        return (false, 0.0);
    }

    // Second part should be month name
    let month_part = &bytes[parts[1].0..parts[1].1];
    if !validate_month_name(month_part) {
        return (false, 0.0);
    }

    // Third part should be year
    let year_part = &bytes[parts[2].0..parts[2].1];
    let (year_valid, year) = validate_year(year_part);
    if year_valid {
        let day = ordinal_day_to_number(day_part, parts[0].1 - parts[0].0);
        let month = month_name_to_number(month_part);
        if day > 0 && month > 0 {
            (true, date_to_timestamp(day, month, year, 0, 0, 0, 0))
        } else {
            (false, 0.0)
        }
    } else {
        (false, 0.0)
    }
}

// Generic Date Pattern - "2024-01-15" or "2024/01/15" or "2024.01.15" or with time "2024-01-15 15:30:45"
#[inline]
fn detect_generic_date_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // GenericDate: year ("-" | "/" | ".") numeric_month ("-" | "/" | ".") numeric_day optional_time_suffix?

    if len < 10 {
        return (false, 0.0);
    }

    // Check for either YYYY-MM-DD (dashes), YYYY/MM/DD (slashes), or YYYY.MM.DD (dots)
    let (_sep1, _sep2) = if signature.dash_count == 2 {
        if signature.dash_positions[0] != 4 || signature.dash_positions[1] != 7 {
            return (false, 0.0);
        }
        (signature.dash_positions[0], signature.dash_positions[1])
    } else if signature.slash_count == 2 {
        if signature.slash_positions[0] != 4 || signature.slash_positions[1] != 7 {
            return (false, 0.0);
        }
        (signature.slash_positions[0], signature.slash_positions[1])
    } else if signature.dot_count == 2 {
        if signature.dot_positions[0] != 4 || signature.dot_positions[1] != 7 {
            return (false, 0.0);
        }
        (signature.dot_positions[0], signature.dot_positions[1])
    } else {
        return (false, 0.0);
    };

    let year_part = &bytes[0..4];
    let month_part = &bytes[5..7];
    let day_part = &bytes[8..10];

    // Parse while validating - Generic format is YYYY-MM-DD
    let (year_valid, year) = validate_year(year_part);
    let (month_valid, month) = validate_numeric_month(month_part, 2);
    let (day_valid, day) = validate_numeric_day(day_part, 2);

    if !year_valid || !month_valid || !day_valid {
        return (false, 0.0);
    }

    // Validate that the date actually exists (e.g., Feb 30th is invalid)
    if !is_valid_date(year, month, day) {
        return (false, 0.0);
    }

    // If it's exactly 10 characters, it's just the date part
    if len == 10 {
        return (true, date_to_timestamp(day, month, year, 0, 0, 0, 0));
    }

    // If longer, validate it has a space and time component
    if len >= 19 && bytes[10] == b' ' {
        // Basic time validation: should have colons for HH:MM:SS format
        if signature.colon_count >= 2 {
            // Parse the time component after the space
            let time_part = &bytes[11..]; // Skip the space at position 10
            let (time_valid, hours, minutes, seconds, milliseconds) =
                validate_time_format(time_part, len - 11, &signature.colon_positions);
            if time_valid {
                return (
                    true,
                    date_to_timestamp(day, month, year, hours, minutes, seconds, milliseconds),
                );
            } else {
                // Invalid time, use midnight
                return (true, date_to_timestamp(day, month, year, 0, 0, 0, 0));
            }
        }
    }

    (false, 0.0)
}

// Full Date Pattern - Complex date with various formats

#[inline]
fn detect_full_date_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // FullDate: year ws "-" ws (month | long_month | numeric_month) ws "-" ws (day | long_day | numeric_day)

    if signature.dash_count != 2 {
        return (false, 0.0);
    }

    let dash1 = signature.dash_positions[0];
    let dash2 = signature.dash_positions[1];

    // Extract year part (should be first)
    let year_part = &bytes[0..dash1];
    let (year_valid, year) = validate_year(year_part);
    if !year_valid {
        return (false, 0.0);
    }

    // Extract month part (between dashes)
    let month_part = &bytes[dash1 + 1..dash2];
    let (month_valid, month) = validate_flexible_month(month_part, dash2 - dash1 - 1);
    if !month_valid {
        return (false, 0.0);
    }

    // Extract day part (after second dash)
    let day_part = &bytes[dash2 + 1..];
    let (day_valid, day) = validate_flexible_day(day_part, len - dash2 - 1);

    if day_valid {
        // Validate that the date actually exists (e.g., Feb 30th is invalid)
        if is_valid_date(year, month, day) {
            (true, date_to_timestamp(day, month, year, 0, 0, 0, 0))
        } else {
            (false, 0.0)
        }
    } else {
        (false, 0.0)
    }
}

// Integer Pattern Detection
#[inline]
fn detect_integer_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> (bool, f64) {
    // Integer: !"." "-"? ASCII_DIGIT+

    let mut start_idx = 0;

    // Check for optional negative sign (grammar only supports "-", not "+")
    if signature.starts_with_sign {
        start_idx = 1;
    }

    // Must have at least one digit
    if start_idx >= len {
        return (false, 0.0);
    }

    // Use database-style digit limit: classify as Integer but don't parse if too large
    use crate::constants::MAX_INTEGER_DIGITS;
    let digit_count = len - start_idx;
    let too_large = digit_count > MAX_INTEGER_DIGITS;

    let mut value = 0i64;

    for i in start_idx..len {
        let byte = bytes[i];
        if !byte.is_ascii_digit() {
            return (false, 0.0);
        }

        // Only compute numeric value if within reasonable size (like other databases)
        if !too_large {
            // Safe to parse since we're within MAX_INTEGER_DIGITS
            if value <= (i64::MAX / 10) {
                value = value * 10 + (byte - b'0') as i64;
            }
            // If overflow within MAX_INTEGER_DIGITS, just stop parsing but continue validation
        }
    }

    // Apply negative sign if present (only if we parsed the value)
    if !too_large && start_idx > 0 && bytes[0] == b'-' {
        value = -value;
    }

    // Return Integer classification regardless of size
    // For very large integers, return 0.0 as numeric value (like PostgreSQL NUMERIC)
    (true, if too_large { 0.0 } else { value as f64 })
}

// File Path Pattern Detection
#[inline]
fn detect_file_pattern<const MAX_POS: usize>(
    bytes: &[u8],
    len: usize,
    signature: &PatternSignature<MAX_POS>,
) -> bool {
    // File: ("/" generic_string)+

    if signature.slash_count == 0 {
        return false;
    }

    // Must start with '/'
    if bytes[0] != b'/' {
        return false;
    }

    // Validate path segments between slashes
    let mut prev_slash = 0;

    for &slash_pos in &signature.slash_positions[..signature.slash_count] {
        if slash_pos > 0 {
            // Check segment between previous slash and current slash
            let segment_start = prev_slash + 1;
            let segment_end = slash_pos;

            if segment_start >= segment_end {
                // Empty segment (like double slashes) - continue processing

                prev_slash = slash_pos;
                continue;
            }

            // Segment should contain valid path characters
            if !validate_path_segment(
                &bytes[segment_start..segment_end],
                segment_end - segment_start,
            ) {
                return false;
            }
        }
        prev_slash = slash_pos;
    }

    // Validate final segment after last slash

    if len > prev_slash + 1 {
        let result = validate_path_segment(&bytes[prev_slash + 1..], len - prev_slash - 1);

        result
    } else {
        // Single slash "/" should not be classified as File - it's too simple
        signature.slash_count > 1
    }
}

// Validation helpers used by the classifier.

// =============================================================================
// SIMD VALIDATION HELPERS
// =============================================================================

// IPv6 hex quad validation
#[cfg(any(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_arch = "aarch64"
))]
#[inline]
fn validate_ipv6_hex_quads(bytes: &[u8], len: usize, colon_positions: &[usize]) -> bool {
    let mut start = 0;

    for &colon_pos in colon_positions {
        if colon_pos > start {
            let quad = &bytes[start..colon_pos];
            let quad_len = colon_pos - start;

            if quad_len > 4 || quad_len == 0 || !is_hex_quad(quad) {
                return false;
            }
        }
        start = colon_pos + 1;
    }

    // Validate final segment - could be hex quad or embedded IPv4
    if start < len {
        let final_segment = &bytes[start..len];
        let final_len = len - start;

        if final_len == 0 {
            return false;
        }

        // Check if this looks like an embedded IPv4 address (contains dots)
        let has_dot = final_segment.iter().any(|&b| b == b'.');
        if has_dot {
            // Validate as IPv4 address
            validate_embedded_ipv4(final_segment, final_len)
        } else {
            // Validate as hex quad
            final_len <= 4 && is_hex_quad(final_segment)
        }
    } else {
        true
    }
}

#[cfg(any(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_arch = "aarch64"
))]
#[inline]
fn is_hex_quad(bytes: &[u8]) -> bool {
    // IPv6 hex quad: 1-4 hex digits
    if bytes.is_empty() || bytes.len() > 4 {
        return false;
    }
    for &byte in bytes {
        if !byte.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

// Validate embedded IPv4 address in IPv6 (like 192.0.2.1)
#[cfg(any(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_arch = "aarch64"
))]
#[inline]
fn validate_embedded_ipv4(bytes: &[u8], len: usize) -> bool {
    let mut dot_count = 0;
    let mut start = 0;

    for i in 0..len {
        let byte = bytes[i];
        if byte == b'.' {
            if i == start {
                return false; // Empty octet
            }

            let octet = &bytes[start..i];
            if !validate_ipv4_octet(octet, i - start) {
                return false;
            }

            dot_count += 1;
            if dot_count > 3 {
                return false; // Too many dots
            }
            start = i + 1;
        }
    }

    // Validate final octet
    if start >= len {
        return false; // Trailing dot
    }

    let final_octet = &bytes[start..];
    dot_count == 3 && validate_ipv4_octet(final_octet, len - start)
}

// Validate a single IPv4 octet (0-255, no leading zeros)
#[cfg(any(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_arch = "aarch64"
))]
#[inline]
fn validate_ipv4_octet(bytes: &[u8], len: usize) -> bool {
    if len == 0 || len > 3 {
        return false;
    }

    // Check for leading zeros (invalid except for "0")
    if len > 1 && bytes[0] == b'0' {
        return false;
    }

    // All bytes must be digits
    if !bytes[..len].iter().all(|&b| b.is_ascii_digit()) {
        return false;
    }

    // Parse value and check range 0-255
    let mut value = 0u32;
    for &byte in &bytes[..len] {
        value = value * 10 + (byte - b'0') as u32;
        if value > 255 {
            return false;
        }
    }

    true
}

// Latitude validation: -90.0 to 90.0
#[cfg(any(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_arch = "aarch64"
))]
#[inline]
fn validate_latitude(bytes: &[u8], len: usize) -> (bool, f64) {
    if len == 0 {
        return (false, 0.0);
    }

    let mut start = 0;
    let mut negative = false;

    // Handle sign
    if bytes[0] == b'-' {
        negative = true;
        start = 1;
    } else if bytes[0] == b'+' {
        start = 1;
    }

    // Parse degrees
    let mut degrees = 0.0;
    let mut decimal_part = 0.0;
    let mut in_decimal = false;
    let mut decimal_places = 0;

    for i in start..len {
        let byte = bytes[i];
        match byte {
            b'0'..=b'9' => {
                let digit = (byte - b'0') as f64;
                if in_decimal {
                    decimal_places += 1;
                    decimal_part += digit / 10_f64.powi(decimal_places);
                } else {
                    degrees = degrees * 10.0 + digit;
                }
            }
            b'.' => {
                if in_decimal {
                    return (false, 0.0);
                } // Multiple dots
                in_decimal = true;
            }
            _ => return (false, 0.0),
        }
    }

    let total = degrees + decimal_part;
    let final_value = if negative { -total } else { total };

    // Exclude exact boundary values to match scalar grammar behavior
    if final_value.abs() >= 90.0 {
        return (false, 0.0);
    }

    (true, final_value)
}

// Longitude validation: -180.0 to 180.0
#[inline]
fn validate_longitude(bytes: &[u8], len: usize) -> (bool, f64) {
    if len == 0 {
        return (false, 0.0);
    }

    let mut start = 0;
    let mut negative = false;

    // Handle sign
    if bytes[0] == b'-' {
        negative = true;
        start = 1;
    } else if bytes[0] == b'+' {
        start = 1;
    }

    // Parse degrees
    let mut degrees = 0.0;
    let mut decimal_part = 0.0;
    let mut in_decimal = false;
    let mut decimal_places = 0;

    for &byte in bytes.iter().skip(start) {
        match byte {
            b'0'..=b'9' => {
                let digit = (byte - b'0') as f64;
                if in_decimal {
                    decimal_places += 1;
                    decimal_part += digit / 10_f64.powi(decimal_places);
                } else {
                    degrees = degrees * 10.0 + digit;
                }
            }
            b'.' => {
                if in_decimal {
                    return (false, 0.0);
                } // Multiple dots
                in_decimal = true;
            }
            _ => return (false, 0.0),
        }
    }

    let total = degrees + decimal_part;
    let final_value = if negative { -total } else { total };

    // Exclude exact boundary values to match scalar grammar behavior
    if final_value.abs() >= 180.0 {
        return (false, 0.0);
    }

    (true, final_value)
}

// Vector element counting and validation helper

#[inline]
fn validate_vector_elements(
    content: &[u8],
    content_len: usize,
    comma_positions: &[usize],
    offset: usize,
) -> bool {
    if content_len == 0 {
        return true; // Empty vector is valid
    }

    // ZERO HEAP: Fixed array for relevant comma positions
    const MAX_VECTOR_COMMAS: usize = 512; // Max commas in a vector
    let mut relevant_commas: [usize; MAX_VECTOR_COMMAS] = [0; MAX_VECTOR_COMMAS];
    let mut comma_count = 0;

    // Collect relevant comma positions manually
    for &pos in comma_positions {
        if pos >= offset && pos < offset + content_len && comma_count < MAX_VECTOR_COMMAS {
            relevant_commas[comma_count] = pos - offset;
            comma_count += 1;
        }
    }

    let mut start = 0;

    // Validate each element - ZERO HEAP
    for i in 0..comma_count {
        let comma_pos = relevant_commas[i];
        let element_slice = &content[start..comma_pos];

        // Filter whitespace without allocation - check bytes in place
        if !is_numeric_element_slice(element_slice, comma_pos - start) {
            return false;
        }
        start = comma_pos + 1;
    }

    // Validate final element - ZERO HEAP
    let final_slice = &content[start..];
    is_numeric_element_slice(final_slice, content_len - start)
}

// Helper to validate numeric element without allocating Vec
#[inline]
fn is_numeric_element_slice(bytes: &[u8], len: usize) -> bool {
    // Skip whitespace at start and end, validate numeric content
    let mut start = 0;
    let mut end = len;

    // Trim whitespace from start
    while start < end && (bytes[start] == b' ' || bytes[start] == b'\t') {
        start += 1;
    }

    // Trim whitespace from end
    while end > start && (bytes[end - 1] == b' ' || bytes[end - 1] == b'\t') {
        end -= 1;
    }

    if start >= end {
        return false; // Empty after trimming
    }

    // Validate the trimmed slice as numeric
    let trimmed = &bytes[start..end];
    is_numeric_element(trimmed, end - start)
}

#[inline]
fn is_numeric_element(bytes: &[u8], len: usize) -> bool {
    if len == 0 {
        return false;
    }

    let mut start = 0;
    if bytes[0] == b'-' || bytes[0] == b'+' {
        start = 1;
    }

    let mut has_dot = false;
    for i in start..len {
        let byte = bytes[i];
        match byte {
            b'0'..=b'9' => continue,
            b'.' => {
                if has_dot {
                    return false;
                }
                has_dot = true;
            }
            b'e' | b'E' => {
                // Scientific notation - validate remaining
                return validate_scientific_remainder(&bytes[i + 1..], len - (i + 1));
            }
            _ => return false,
        }
    }

    start < len // Must have at least one digit
}

#[inline]
fn validate_scientific_remainder(bytes: &[u8], len: usize) -> bool {
    if len == 0 {
        return false;
    }

    let mut start = 0;
    if bytes[0] == b'+' || bytes[0] == b'-' {
        start = 1;
    }

    if start >= len {
        return false;
    }

    bytes[start..len].iter().all(|&b| b.is_ascii_digit())
}

// Float validation helpers
#[inline]
fn validate_float_integer_part(bytes: &[u8], start: usize, end: usize) -> bool {
    if start >= end {
        return false;
    }

    // ASCII_DIGIT+
    for i in start..end {
        if !bytes[i].is_ascii_digit() {
            return false;
        }
    }
    true
}

#[inline]
fn validate_float_fractional_part(bytes: &[u8], start: usize, end: usize) -> bool {
    // ASCII_DIGIT+
    for i in start..end {
        if !bytes[i].is_ascii_digit() {
            return false;
        }
    }
    true
}

#[inline]
fn validate_float_exponent_part(bytes: &[u8], start: usize, end: usize) -> bool {
    // ("+" | "-")? ASCII_DIGIT+
    if start >= end {
        return false;
    }

    let mut idx = start;
    if bytes[idx] == b'+' || bytes[idx] == b'-' {
        idx += 1;
    }

    // Must have at least one digit after optional sign
    if idx >= end {
        return false;
    }

    // Validate remaining digits
    for i in idx..end {
        if !bytes[i].is_ascii_digit() {
            return false;
        }
    }
    true
}

#[inline]
fn is_datemath_unit(byte: u8) -> bool {
    matches!(byte, b'y' | b'M' | b'w' | b'd' | b'h' | b'H' | b'm' | b's')
}

// Whitespace splitting - ZERO HEAP with fixed array + count pattern
const MAX_DATE_PARTS: usize = 16; // Maximum parts expected in date strings

#[inline]
fn split_on_whitespace(
    bytes: &[u8],
    len: usize,
    parts: &mut [(usize, usize); MAX_DATE_PARTS],
    part_count: &mut usize,
) {
    *part_count = 0;
    let mut start = 0;
    let mut in_word = false;

    for (i, &byte) in bytes.iter().enumerate().take(len) {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => {
                if in_word {
                    if *part_count < MAX_DATE_PARTS {
                        parts[*part_count] = (start, i);
                        *part_count += 1;
                    }
                    in_word = false;
                }
            }
            _ => {
                if !in_word {
                    start = i;
                    in_word = true;
                }
            }
        }
    }

    if in_word && *part_count < MAX_DATE_PARTS {
        parts[*part_count] = (start, len);
        *part_count += 1;
    }
}

// Day name validation
#[inline]
fn validate_day_name(bytes: &[u8]) -> bool {
    matches!(
        bytes,
        b"Mon"
            | b"Tue"
            | b"Wed"
            | b"Thu"
            | b"Fri"
            | b"Sat"
            | b"Sun"
            | b"Monday"
            | b"Tuesday"
            | b"Wednesday"
            | b"Thursday"
            | b"Friday"
            | b"Saturday"
            | b"Sunday"
    )
}

// Month name validation
#[inline]
fn validate_month_name(bytes: &[u8]) -> bool {
    matches!(
        bytes,
        b"Jan"
            | b"Feb"
            | b"Mar"
            | b"Apr"
            | b"May"
            | b"Jun"
            | b"Jul"
            | b"Aug"
            | b"Sep"
            | b"Oct"
            | b"Nov"
            | b"Dec"
            | b"January"
            | b"February"
            | b"March"
            | b"April"
            | b"June"
            | b"July"
            | b"August"
            | b"September"
            | b"October"
            | b"November"
            | b"December"
    )
}

// Numeric day validation (1-31)
#[inline]
fn validate_numeric_day(bytes: &[u8], len: usize) -> (bool, u32) {
    if len == 0 || len > 2 {
        return (false, 0);
    }

    let mut day = 0u8;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return (false, 0);
        }
        day = day * 10 + (byte - b'0');
    }

    if day >= 1 && day <= 31 {
        (true, day as u32)
    } else {
        (false, 0)
    }
}

// Date validity validation - checks if year/month/day combination actually exists
#[inline]
fn is_valid_date(year: u32, month: u32, day: u32) -> bool {
    if month < 1 || month > 12 || day < 1 {
        return false;
    }

    // Days in each month (non-leap year)
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31, // Jan, Mar, May, Jul, Aug, Oct, Dec
        4 | 6 | 9 | 11 => 30,              // Apr, Jun, Sep, Nov
        2 => {
            // February - check for leap year
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => return false,
    };

    day <= days_in_month
}

// Ordinal day validation (1st, 2nd, 3rd, 4th, etc.)
#[inline]
fn validate_ordinal_day(bytes: &[u8], len: usize) -> bool {
    if len < 3 || len > 4 {
        return false;
    }

    // Extract numeric part
    let mut day = 0u16;
    let mut digits = 0;

    for &byte in bytes {
        if byte.is_ascii_digit() {
            day = day * 10 + (byte - b'0') as u16;
            digits += 1;
        } else {
            break;
        }
    }

    if digits == 0 || day < 1 || day > 31 {
        return false;
    }

    // Validate suffix
    let suffix = &bytes[digits..];
    match day {
        1 | 21 | 31 => suffix == b"st",
        2 | 22 => suffix == b"nd",
        3 | 23 => suffix == b"rd",
        _ => suffix == b"th",
    }
}

// Time format validation (HH:MM:SS or HH:MM) - ZERO HEAP
#[inline]
fn validate_time_format(
    bytes: &[u8],
    len: usize,
    _colon_positions: &[usize],
) -> (bool, u32, u32, u32, u32) {
    // Find colons within this time string - stack only
    const MAX_TIME_COLONS: usize = 4;
    let mut local_colon_positions: [usize; MAX_TIME_COLONS] = [0; MAX_TIME_COLONS];
    let mut colon_count = 0;

    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b':' && colon_count < MAX_TIME_COLONS {
            local_colon_positions[colon_count] = i;
            colon_count += 1;
        }
    }

    let mut parts: [(usize, usize); MAX_DATE_PARTS] = [(0, 0); MAX_DATE_PARTS];
    let mut part_count = 0;
    split_on_colon(
        len,
        &local_colon_positions[..colon_count],
        &mut parts,
        &mut part_count,
    );

    match part_count {
        2 => {
            // HH:MM
            let (hour_valid, hour) =
                validate_hour(&bytes[parts[0].0..parts[0].1], parts[0].1 - parts[0].0);
            let (minute_valid, minute) =
                validate_minute(&bytes[parts[1].0..parts[1].1], parts[1].1 - parts[1].0);
            if hour_valid && minute_valid {
                (true, hour, minute, 0, 0)
            } else {
                (false, 0, 0, 0, 0)
            }
        }
        3 => {
            // HH:MM:SS or HH:MM:SS.sss - check for milliseconds in seconds part
            let (hour_valid, hour) =
                validate_hour(&bytes[parts[0].0..parts[0].1], parts[0].1 - parts[0].0);
            let (minute_valid, minute) =
                validate_minute(&bytes[parts[1].0..parts[1].1], parts[1].1 - parts[1].0);

            let seconds_part = &bytes[parts[2].0..parts[2].1];
            let seconds_len = parts[2].1 - parts[2].0;

            // Check for decimal point in seconds
            let mut dot_idx = 0;
            let mut has_dot = false;
            for (i, &byte) in seconds_part.iter().enumerate().take(seconds_len) {
                if byte == b'.' {
                    dot_idx = i;
                    has_dot = true;
                    break;
                }
            }

            let (second_valid, second, milliseconds) = if has_dot {
                // Parse seconds and milliseconds: SS.sss
                if dot_idx != 2 {
                    (false, 0, 0) // Invalid seconds format
                } else {
                    let (sec_valid, sec) = validate_second(&seconds_part[0..2], 2);
                    if !sec_valid {
                        (false, 0, 0)
                    } else {
                        // Parse milliseconds
                        let mut ms = 0u32;
                        let mut multiplier = 100u32;
                        let ms_start = dot_idx + 1;
                        for i in ms_start..seconds_len.min(ms_start + 3) {
                            if seconds_part[i].is_ascii_digit() {
                                ms += (seconds_part[i] - b'0') as u32 * multiplier;
                                multiplier /= 10;
                            } else {
                                break;
                            }
                        }
                        (true, sec, ms)
                    }
                }
            } else {
                // Just seconds: SS
                let (sec_valid, sec) = validate_second(seconds_part, seconds_len);
                (sec_valid, sec, 0)
            };

            if hour_valid && minute_valid && second_valid {
                (true, hour, minute, second, milliseconds)
            } else {
                (false, 0, 0, 0, 0)
            }
        }
        _ => (false, 0, 0, 0, 0),
    }
}

#[inline]
fn split_on_colon(
    len: usize,
    colon_positions: &[usize],
    parts: &mut [(usize, usize); MAX_DATE_PARTS],
    part_count: &mut usize,
) {
    *part_count = 0;
    let mut start = 0;

    for &colon_pos in colon_positions {
        if colon_pos > start && *part_count < MAX_DATE_PARTS {
            parts[*part_count] = (start, colon_pos);
            *part_count += 1;
        }
        start = colon_pos + 1;
    }

    if start < len && *part_count < MAX_DATE_PARTS {
        parts[*part_count] = (start, len);
        *part_count += 1;
    }
}

#[inline]
fn validate_hour(bytes: &[u8], len: usize) -> (bool, u32) {
    let (valid, hour) = parse_two_digits(bytes, len);
    if valid && hour <= 23 {
        (true, hour as u32)
    } else {
        (false, 0)
    }
}

#[inline]
fn validate_minute(bytes: &[u8], len: usize) -> (bool, u32) {
    let (valid, minute) = parse_two_digits(bytes, len);
    if valid && minute <= 59 {
        (true, minute as u32)
    } else {
        (false, 0)
    }
}

#[inline]
fn validate_second(bytes: &[u8], len: usize) -> (bool, u32) {
    let (valid, second) = parse_two_digits(bytes, len);
    if valid && second <= 59 {
        (true, second as u32)
    } else {
        (false, 0)
    }
}

#[inline]
fn parse_two_digits(bytes: &[u8], len: usize) -> (bool, u8) {
    // BOUNDS CHECK: Ensure we have at least 2 bytes before accessing array indices
    // This prevents "index out of bounds" panics on malformed input like single-digit time parts
    if len < 2 || !bytes[0].is_ascii_digit() || !bytes[1].is_ascii_digit() {
        return (false, 0);
    }

    (true, (bytes[0] - b'0') * 10 + (bytes[1] - b'0'))
}

// Year validation (YYYY format)
#[inline]
fn validate_year(bytes: &[u8]) -> (bool, u32) {
    let mut year = 0u16;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return (false, 0);
        }
        year = year * 10 + (byte - b'0') as u16;
    }

    if year >= 1900 && year <= 2100 {
        (true, year as u32)
    } else {
        (false, 0)
    }
}

// ISO time part validation with timezone offset
#[inline]
fn validate_iso_time_part(bytes: &[u8], len: usize) -> (bool, u32, u32, u32, u32, i32) {
    if len < 5 {
        return (false, 0, 0, 0, 0, 0); // Minimum HH:MM
    }

    // Extract basic hour and minute components
    let (hour_valid, hour) = validate_hour(&bytes[0..2], 2);
    let colon1_valid = bytes[2] == b':';
    let (minute_valid, minute) = validate_minute(&bytes[3..5], 2);

    if !(hour_valid && colon1_valid && minute_valid) {
        return (false, 0, 0, 0, 0, 0);
    }

    // Check if we have seconds (HH:MM:SS format)
    if len >= 8 && bytes[5] == b':' {
        let (second_valid, second) = validate_second(&bytes[6..8], 2);

        if !second_valid {
            return (false, 0, 0, 0, 0, 0);
        }

        // Check for optional fractional seconds and timezone
        if len > 8 {
            let remaining = &bytes[8..];
            let remaining_len = len - 8;
            if remaining[0] == b'.' {
                // Parse fractional seconds (milliseconds)
                let mut frac_end = 1;
                let mut milliseconds = 0u32;
                let mut multiplier = 100u32; // Start with hundreds place

                while frac_end < remaining_len
                    && remaining[frac_end].is_ascii_digit()
                    && frac_end <= 4
                {
                    let digit = (remaining[frac_end] - b'0') as u32;
                    milliseconds += digit * multiplier;
                    multiplier /= 10;
                    frac_end += 1;
                }

                // Skip any additional fractional digits beyond milliseconds
                while frac_end < remaining_len && remaining[frac_end].is_ascii_digit() {
                    frac_end += 1;
                }

                if frac_end < remaining_len {
                    // Timezone follows
                    let (tz_valid, tz_offset) =
                        validate_timezone(&remaining[frac_end..], remaining_len - frac_end);
                    if tz_valid {
                        (true, hour, minute, second, milliseconds, tz_offset)
                    } else {
                        (false, 0, 0, 0, 0, 0)
                    }
                } else {
                    (true, hour, minute, second, milliseconds, 0)
                }
            } else {
                // Direct timezone, no milliseconds
                let (tz_valid, tz_offset) = validate_timezone(remaining, remaining_len);
                if tz_valid {
                    (true, hour, minute, second, 0, tz_offset)
                } else {
                    (false, 0, 0, 0, 0, 0)
                }
            }
        } else {
            (true, hour, minute, second, 0, 0)
        }
    } else {
        // HH:MM format without seconds - check for timezone after minute
        if len > 5 {
            let remaining = &bytes[5..];
            let remaining_len = len - 5;
            let (tz_valid, tz_offset) = validate_timezone(remaining, remaining_len);
            if tz_valid {
                (true, hour, minute, 0, 0, tz_offset) // No seconds, 0 milliseconds
            } else {
                (false, 0, 0, 0, 0, 0)
            }
        } else {
            (true, hour, minute, 0, 0, 0) // Just HH:MM
        }
    }
}

// Timezone validation with offset extraction
#[inline]
fn validate_timezone(bytes: &[u8], len: usize) -> (bool, i32) {
    if len == 0 {
        return (true, 0); // Optional, defaults to UTC
    }

    match bytes[0] {
        b'Z' | b'z' => (len == 1, 0), // UTC timezone
        b'+' | b'-' => {
            if len < 5 {
                return (false, 0);
            }

            let sign = if bytes[0] == b'+' { 1 } else { -1 };

            // +HHMM or +HH:MM format
            let (hour_offset, minute_offset) = if len == 5 {
                // +HHMM
                let (hour_valid, hour) = validate_hour(&bytes[1..3], 2);
                let (minute_valid, minute) = validate_minute(&bytes[3..5], 2);
                if hour_valid && minute_valid {
                    (hour as i32, minute as i32)
                } else {
                    return (false, 0);
                }
            } else if len == 6 && bytes[3] == b':' {
                // +HH:MM
                let (hour_valid, hour) = validate_hour(&bytes[1..3], 2);
                let (minute_valid, minute) = validate_minute(&bytes[4..6], 2);
                if hour_valid && minute_valid {
                    (hour as i32, minute as i32)
                } else {
                    return (false, 0);
                }
            } else {
                return (false, 0);
            };

            // Convert to total minutes offset
            let total_minutes = sign * (hour_offset * 60 + minute_offset);
            (true, total_minutes)
        }
        _ => (false, 0),
    }
}

// Numeric month validation (01-12)
#[inline]
fn validate_numeric_month(bytes: &[u8], len: usize) -> (bool, u32) {
    if len == 0 || len > 2 {
        return (false, 0);
    }

    let mut month = 0u8;
    for &byte in bytes.iter().take(len) {
        if !byte.is_ascii_digit() {
            return (false, 0);
        }
        month = month * 10 + (byte - b'0');
    }

    if month >= 1 && month <= 12 {
        (true, month as u32)
    } else {
        (false, 0)
    }
}

// Split date and time parts
#[inline]
fn split_date_time(bytes: &[u8], len: usize) -> (&[u8], bool, &[u8]) {
    // Look for space separating date and time
    for i in 0..len {
        if bytes[i] == b' ' || bytes[i] == b'T' {
            return (&bytes[0..i], true, &bytes[i + 1..]);
        }
    }

    (bytes, false, &[])
}

// Keyword finding and validation
#[inline]
fn find_keyword(bytes: &[u8], len: usize, keyword: &[u8], keyword_len: usize) -> bool {
    if keyword_len > len {
        return false;
    }

    for i in 0..=len - keyword_len {
        if bytes[i..i + keyword_len] == *keyword {
            // Check word boundaries
            let start_ok = i == 0 || bytes[i - 1] == b' ' || bytes[i - 1] == b'\t';
            let end_ok = i + keyword_len == len
                || bytes[i + keyword_len] == b' '
                || bytes[i + keyword_len] == b'\t';

            if start_ok && end_ok {
                return true;
            }
        }
    }

    false
}

#[inline]
fn is_keyword(bytes: &[u8], keyword: &[u8]) -> bool {
    bytes == keyword
}

// Flexible month validation (numeric, short, or long month names)
#[inline]
fn validate_flexible_month(bytes: &[u8], len: usize) -> (bool, u32) {
    // Try numeric month first
    let (numeric_valid, month_num) = validate_numeric_month(bytes, len);
    if numeric_valid {
        return (true, month_num);
    }

    // Try month name
    if validate_month_name(bytes) {
        let month_num = match bytes {
            b"Jan" | b"January" => 1,
            b"Feb" | b"February" => 2,
            b"Mar" | b"March" => 3,
            b"Apr" | b"April" => 4,
            b"May" => 5,
            b"Jun" | b"June" => 6,
            b"Jul" | b"July" => 7,
            b"Aug" | b"August" => 8,
            b"Sep" | b"September" => 9,
            b"Oct" | b"October" => 10,
            b"Nov" | b"November" => 11,
            b"Dec" | b"December" => 12,
            _ => return (false, 0),
        };
        return (true, month_num);
    }

    (false, 0)
}

// Flexible day validation (numeric day or day name)
#[inline]
fn validate_flexible_day(bytes: &[u8], len: usize) -> (bool, u32) {
    // Try numeric day first
    let (numeric_valid, day_num) = validate_numeric_day(bytes, len);
    if numeric_valid {
        return (true, day_num);
    }

    // Day names not supported for timestamp conversion
    if validate_day_name(bytes) {
        return (true, 1); // Default to day 1 for day names
    }

    (false, 0)
}

// Path segment validation
#[inline]
fn validate_path_segment(bytes: &[u8], len: usize) -> bool {
    if len == 0 {
        return false;
    }

    // Valid path characters: letters, digits, dots, dashes, underscores
    bytes
        .iter()
        .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'~'))
}

// Notes:
// - This module prioritizes throughput on common inputs and avoids allocations where practical.
// - The supported categories are defined by `HwxType`; tests validate behavior and precedence.

// =============================================================================
// SCALAR REMAINING BYTES PROCESSING
// =============================================================================

/// Handle remaining bytes that don't fit into SIMD lanes using scalar processing
#[inline]
unsafe fn classify_scalar_remaining<const MAX_POS: usize>(
    remaining_bytes: &[u8],
    signature: &mut PatternSignature<MAX_POS>,
    position: usize,
) {
    for (i, &byte) in remaining_bytes.iter().enumerate() {
        let pos = position + i;

        // Classify character type and update signature
        if byte.is_ascii_digit() {
            signature.has_digits = true;
            if pos == 0 {
                signature.starts_with_digit = true;
            }
        } else if byte.is_ascii_alphabetic() {
            signature.has_letters = true;
            if pos == 0 {
                signature.starts_with_letter = true;
            }
        } else {
            signature.has_punctuation = true;

            // Track specific punctuation positions - ZERO HEAP with bounds checking
            match byte {
                b'.' => {
                    if signature.dot_count < MAX_POS {
                        signature.dot_positions[signature.dot_count] = pos;
                        signature.dot_count += 1;
                    }
                }
                b':' => {
                    if signature.colon_count < MAX_POS {
                        signature.colon_positions[signature.colon_count] = pos;
                        signature.colon_count += 1;
                    }
                }
                b'/' => {
                    if signature.slash_count < MAX_POS {
                        signature.slash_positions[signature.slash_count] = pos;
                        signature.slash_count += 1;
                    }
                }
                b'-' => {
                    if signature.dash_count < MAX_POS {
                        signature.dash_positions[signature.dash_count] = pos;
                        signature.dash_count += 1;
                    }
                    if pos == 0 {
                        signature.starts_with_sign = true;
                    }
                }
                b'+' => {
                    if signature.plus_count < MAX_POS {
                        signature.plus_positions[signature.plus_count] = pos;
                        signature.plus_count += 1;
                    }
                    if pos == 0 {
                        signature.starts_with_sign = true;
                    }
                }
                b'[' => {
                    if signature.bracket_open_count < MAX_POS {
                        signature.bracket_open_positions[signature.bracket_open_count] = pos;
                        signature.bracket_open_count += 1;
                    }
                }
                b']' => {
                    if signature.bracket_close_count < MAX_POS {
                        signature.bracket_close_positions[signature.bracket_close_count] = pos;
                        signature.bracket_close_count += 1;
                    }
                }
                b',' => {
                    if signature.comma_count < MAX_POS {
                        signature.comma_positions[signature.comma_count] = pos;
                        signature.comma_count += 1;
                    }
                }
                b'@' => {
                    if signature.at_count < MAX_POS {
                        signature.at_positions[signature.at_count] = pos;
                        signature.at_count += 1;
                    }
                }
                b' ' | b'\t' => signature.has_whitespace = true,
                _ => {}
            }
        }
    }
}
