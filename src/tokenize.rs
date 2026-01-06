// SPDX-License-Identifier: Apache-2.0

//! Unicode tokenization
//!
//! Tokenization utilities with Unicode-aware word boundary handling.
//! Implementations may use scalar code, SIMD, and (when enabled) CUDA kernels.
//!
//! ## Performance notes
//! Some kernels are written in a performance-oriented style. When modifying hot paths,
//! prefer changes that keep allocations out of inner loops.
//
// Some clippy lints are noisy for low-level SIMD/FFI code; we opt out at the module level.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::all)]

// =============================================================================
// UNICODE TOKENIZATION AND WORD BOUNDARY DETECTION OPERATIONS
// =============================================================================

// ARM NEON imports
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_u8, vandq_u8, vceqq_u8, vcgeq_u8, vcleq_u8, vdupq_n_u8, vld1q_u8, vorrq_u8, vst1q_u8,
};

// x86_64 SIMD intrinsics imports - AVX-512 (nightly feature)

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use std::arch::x86_64::{
    _mm512_cmpeq_epi8_mask, _mm512_cmpge_epi8_mask, _mm512_cmple_epi8_mask, _mm512_loadu_si512,
    _mm512_mask_add_epi8, _mm512_set1_epi8, _mm512_storeu_si512,
};

// x86_64 SIMD intrinsics imports - AVX2 only (not when nightly AVX-512 is enabled)
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use std::arch::x86_64::{
    // AVX2 intrinsics
    _mm256_add_epi8,
    _mm256_and_si256,
    _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8,
    _mm256_loadu_si256,
    _mm256_movemask_epi8,
    _mm256_or_si256,
    _mm256_set1_epi8,
    _mm256_setzero_si256,
    _mm256_storeu_si256,
};

// Note: CUDA paths use `log::error` for diagnostics. Keep logging out of tight loops.
#[cfg(has_cuda)]
use log::error;

// Conditional imports for constants based on target architecture and features
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::constants::{LANES_AVX512_BYTES, MAX_POSITIONS_AVX512};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
use super::constants::{LANES_AVX2_BYTES, MAX_POSITIONS_AVX2};

#[cfg(target_arch = "aarch64")]
use super::constants::{LANES_NEON_BYTES, MAX_POSITIONS_NEON};

#[cfg(has_cuda)]
use crate::gpu::{
    launch_ptx, with_gpu_buffer_u32_inplace, with_gpu_buffer_u8, with_gpu_buffer_u8_mut,
    LaunchConfig,
};

// =============================================================================
// TOKENIZATION RESULT AND WORD BOUNDARY STRUCTURES
// =============================================================================

// =============================================================================
// WORD BOUNDARY SIGNATURE AND TRACKER - MIRRORING PATTERNSIGNATURE
// =============================================================================

/// Word boundary signature for tracking tokenization patterns during SIMD processing
/// Mirrors PatternSignature structure from classify.rs
#[derive(Debug, Clone)]
pub struct WordBoundarySignature<const MAX_POS: usize> {
    // Character type flags
    has_letters: bool,
    has_digits: bool,
    has_punctuation: bool,
    has_whitespace: bool,

    // Word boundary positions (fixed-size stack arrays)
    word_start_positions: [usize; MAX_POS],
    word_start_count: usize,
    word_end_positions: [usize; MAX_POS],
    word_end_count: usize,

    // Current state tracking
    current_word_start: usize, // usize::MAX if not in word
    prev_was_word_char: bool,
}

impl<const MAX_POS: usize> WordBoundarySignature<MAX_POS> {
    #[inline]
    fn new() -> Self {
        Self {
            has_letters: false,
            has_digits: false,
            has_punctuation: false,
            has_whitespace: false,
            word_start_positions: [0; MAX_POS],
            word_start_count: 0,
            word_end_positions: [0; MAX_POS],
            word_end_count: 0,
            current_word_start: usize::MAX,
            prev_was_word_char: false,
        }
    }

    // Update signature from AVX-512 SIMD results
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[inline]
    unsafe fn update_avx512(
        &mut self,
        letter_mask: u64,
        digit_mask: u64,
        punct_mask: u64,
        space_mask: u64,
        position: usize,
        full_input: &[u8],
    ) {
        // Update character type flags
        if letter_mask != 0 {
            self.has_letters = true;
        }
        if digit_mask != 0 {
            self.has_digits = true;
        }
        if punct_mask != 0 {
            self.has_punctuation = true;
        }
        if space_mask != 0 {
            self.has_whitespace = true;
        }

        // Process word boundaries based on masks
        self.process_word_boundaries_avx512(letter_mask, digit_mask, position, full_input);
    }

    // Update signature from AVX2 SIMD results
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    #[inline]
    unsafe fn update_avx2(
        &mut self,
        letter_mask: i32,
        digit_mask: i32,
        punct_mask: i32,
        space_mask: i32,
        position: usize,
        full_input: &[u8],
    ) {
        // Update character type flags
        if letter_mask != 0 {
            self.has_letters = true;
        }
        if digit_mask != 0 {
            self.has_digits = true;
        }
        if punct_mask != 0 {
            self.has_punctuation = true;
        }
        if space_mask != 0 {
            self.has_whitespace = true;
        }

        // Process word boundaries based on masks
        self.process_word_boundaries_avx2(letter_mask, digit_mask, position, full_input);
    }

    // Update signature from NEON SIMD results
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn update_neon(
        &mut self,
        letter_mask: u64,
        digit_mask: u64,
        punct_mask: u64,
        space_mask: u64,
        position: usize,
        full_input: &[u8],
    ) {
        // Update character type flags
        if letter_mask != 0 {
            self.has_letters = true;
        }
        if digit_mask != 0 {
            self.has_digits = true;
        }
        if punct_mask != 0 {
            self.has_punctuation = true;
        }
        if space_mask != 0 {
            self.has_whitespace = true;
        }

        // Process word boundaries based on masks
        self.process_word_boundaries_neon(letter_mask, digit_mask, position, full_input);
    }

    // Process word boundaries from AVX-512 masks with proper Unicode contextual rules
    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[inline]
    unsafe fn process_word_boundaries_avx512(
        &mut self,
        letter_mask: u64,
        digit_mask: u64,
        position: usize,
        full_input: &[u8],
    ) {
        // Letters, digits, and UTF-8 sequences are core word characters
        let word_char_mask = letter_mask | digit_mask;

        // Process each byte position in the chunk
        let mut lane = 0;
        while lane < LANES_AVX512_BYTES {
            let current_pos = position + lane;

            // Skip if beyond input length
            if current_pos >= full_input.len() {
                break;
            }

            let byte = full_input[current_pos];

            // Check for 0xC2 sequences that should split words
            if byte == 0xC2 && current_pos + 1 < full_input.len() {
                let next_byte = full_input[current_pos + 1];
                // NBSP (0xC2 0xA0) and middle dot (0xC2 0xB7) split words
                if next_byte == 0xA0 || next_byte == 0xB7 {
                    if self.prev_was_word_char {
                        self.end_word(current_pos);
                    }
                    self.prev_was_word_char = false;
                    lane += 2; // Skip both bytes
                    continue;
                }
            }

            let is_core_word_char = (word_char_mask >> lane) & 1 != 0;

            // Apply contextual rules for MidNumLet (period, colon between letters/numbers)
            let mut is_word_char = is_core_word_char;

            // Handle UTF-8 sequences more intelligently
            if !is_core_word_char && byte >= 0x80 {
                // Check if this is part of a UTF-8 letter sequence
                is_word_char = is_utf8_letter_byte(full_input, current_pos);
            }

            // Apply MidNumLet contextual rules for periods and colons
            if !is_word_char {
                if is_mid_num_let(byte) {
                    // Check if between word characters
                    let prev_is_word = if current_pos > 0 {
                        is_unicode_word_char(full_input[current_pos - 1])
                    } else {
                        false
                    };
                    let next_is_word = if current_pos + 1 < full_input.len() {
                        is_unicode_word_char(full_input[current_pos + 1])
                    } else {
                        false
                    };

                    if prev_is_word && next_is_word {
                        is_word_char = true;
                    }
                }
            }

            // Update word boundary state - only at valid UTF-8 character boundaries
            if is_word_char && !self.prev_was_word_char {
                // Start word boundary - find the next valid UTF-8 character boundary
                let mut boundary_pos = current_pos;
                while boundary_pos < full_input.len()
                    && !self.is_char_boundary(full_input, boundary_pos)
                {
                    boundary_pos += 1;
                }
                if boundary_pos < full_input.len() {
                    self.start_word(boundary_pos);
                }
            } else if !is_word_char && self.prev_was_word_char {
                // End word boundary - use the current position if it's a valid boundary,
                // otherwise find the previous valid boundary
                let mut boundary_pos = current_pos;
                while boundary_pos > 0 && !self.is_char_boundary(full_input, boundary_pos) {
                    boundary_pos -= 1;
                }
                self.end_word(boundary_pos);
            }

            self.prev_was_word_char = is_word_char;
            lane += 1;
        }
    }

    // Process word boundaries from AVX2 masks with proper Unicode contextual rules
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    #[inline]
    unsafe fn process_word_boundaries_avx2(
        &mut self,
        letter_mask: i32,
        digit_mask: i32,
        position: usize,
        full_input: &[u8],
    ) {
        // Letters, digits, and UTF-8 sequences are core word characters
        let word_char_mask = letter_mask | digit_mask;

        // Process each byte position in the chunk
        let mut lane = 0;
        while lane < LANES_AVX2_BYTES {
            let current_pos = position + lane;

            // Skip if beyond input length
            if current_pos >= full_input.len() {
                break;
            }

            let byte = full_input[current_pos];

            // Check for 0xC2 sequences that should split words
            if byte == 0xC2 && current_pos + 1 < full_input.len() {
                let next_byte = full_input[current_pos + 1];
                // NBSP (0xC2 0xA0) and middle dot (0xC2 0xB7) split words
                if next_byte == 0xA0 || next_byte == 0xB7 {
                    if self.prev_was_word_char {
                        self.end_word(current_pos);
                    }
                    self.prev_was_word_char = false;
                    lane += 2; // Skip both bytes
                    continue;
                }
            }

            let is_core_word_char = ((word_char_mask >> lane) & 1) != 0;

            // Apply contextual rules for MidNumLet (period, colon between letters/numbers)
            let mut is_word_char = is_core_word_char;

            // Handle UTF-8 sequences more intelligently
            if !is_core_word_char && byte >= 0x80 {
                // Check if this is part of a UTF-8 letter sequence
                is_word_char = is_utf8_letter_byte(full_input, current_pos);
            }

            // Apply MidNumLet contextual rules for periods and colons
            if !is_word_char {
                if is_mid_num_let(byte) {
                    // Check if between word characters
                    let prev_is_word = if current_pos > 0 {
                        is_unicode_word_char(full_input[current_pos - 1])
                    } else {
                        false
                    };
                    let next_is_word = if current_pos + 1 < full_input.len() {
                        is_unicode_word_char(full_input[current_pos + 1])
                    } else {
                        false
                    };

                    if prev_is_word && next_is_word {
                        is_word_char = true;
                    }
                }
            }

            // Update word boundary state - only at valid UTF-8 character boundaries
            if is_word_char && !self.prev_was_word_char {
                // Start word boundary - find the next valid UTF-8 character boundary
                let mut boundary_pos = current_pos;
                while boundary_pos < full_input.len()
                    && !self.is_char_boundary(full_input, boundary_pos)
                {
                    boundary_pos += 1;
                }
                if boundary_pos < full_input.len() {
                    self.start_word(boundary_pos);
                }
            } else if !is_word_char && self.prev_was_word_char {
                // End word boundary - use the current position if it's a valid boundary,
                // otherwise find the previous valid boundary
                let mut boundary_pos = current_pos;
                while boundary_pos > 0 && !self.is_char_boundary(full_input, boundary_pos) {
                    boundary_pos -= 1;
                }
                self.end_word(boundary_pos);
            }

            self.prev_was_word_char = is_word_char;
            lane += 1;
        }
    }

    // Process word boundaries from NEON masks with proper Unicode contextual rules
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn process_word_boundaries_neon(
        &mut self,
        letter_mask: u64,
        digit_mask: u64,
        position: usize,
        full_input: &[u8],
    ) {
        // Letters, digits, and UTF-8 sequences are core word characters
        let word_char_mask = letter_mask | digit_mask;

        // Process each byte position in the chunk
        let mut lane = 0;
        while lane < LANES_NEON_BYTES {
            let current_pos = position + lane;

            // Skip if beyond input length
            if current_pos >= full_input.len() {
                break;
            }

            let byte = full_input[current_pos];

            // Check for 0xC2 sequences that should split words
            if byte == 0xC2 && current_pos + 1 < full_input.len() {
                let next_byte = full_input[current_pos + 1];
                // NBSP (0xC2 0xA0) and middle dot (0xC2 0xB7) split words
                if next_byte == 0xA0 || next_byte == 0xB7 {
                    if self.prev_was_word_char {
                        self.end_word(current_pos);
                    }
                    self.prev_was_word_char = false;
                    lane += 2; // Skip both bytes
                    continue;
                }
            }

            let is_core_word_char = ((word_char_mask >> lane) & 1) != 0;

            // Apply contextual rules for MidNumLet (period, colon between letters/numbers)
            let mut is_word_char = is_core_word_char;

            // Handle UTF-8 sequences more intelligently
            if !is_core_word_char && byte >= 0x80 {
                // Check if this is part of a UTF-8 letter sequence
                is_word_char = is_utf8_letter_byte(full_input, current_pos);
            }

            // Apply MidNumLet contextual rules for periods and colons
            if !is_word_char {
                if is_mid_num_let(byte) {
                    // Check if between word characters
                    let prev_is_word = if current_pos > 0 {
                        is_unicode_word_char(full_input[current_pos - 1])
                    } else {
                        false
                    };
                    let next_is_word = if current_pos + 1 < full_input.len() {
                        is_unicode_word_char(full_input[current_pos + 1])
                    } else {
                        false
                    };

                    if prev_is_word && next_is_word {
                        is_word_char = true;
                    }
                }
            }

            // Update word boundary state - only at valid UTF-8 character boundaries
            if is_word_char && !self.prev_was_word_char {
                // Start word boundary - find the next valid UTF-8 character boundary
                let mut boundary_pos = current_pos;
                while boundary_pos < full_input.len()
                    && !self.is_char_boundary(full_input, boundary_pos)
                {
                    boundary_pos += 1;
                }
                if boundary_pos < full_input.len() {
                    self.start_word(boundary_pos);
                }
            } else if !is_word_char && self.prev_was_word_char {
                // End word boundary - use the current position if it's a valid boundary,
                // otherwise find the previous valid boundary
                let mut boundary_pos = current_pos;
                while boundary_pos > 0 && !self.is_char_boundary(full_input, boundary_pos) {
                    boundary_pos -= 1;
                }
                self.end_word(boundary_pos);
            }

            self.prev_was_word_char = is_word_char;
            lane += 1;
        }
    }

    #[inline]
    fn start_word(&mut self, position: usize) {
        if self.current_word_start == usize::MAX {
            self.current_word_start = position;
            if self.word_start_count < MAX_POS {
                self.word_start_positions[self.word_start_count] = position;
            }
            self.word_start_count += 1;
        }
    }

    #[inline]
    fn end_word(&mut self, position: usize) {
        if self.current_word_start != usize::MAX {
            if self.word_end_count < MAX_POS {
                self.word_end_positions[self.word_end_count] = position;
            }
            self.word_end_count += 1;
            self.current_word_start = usize::MAX;
        }
    }

    // Validate that a position is a valid UTF-8 character boundary
    #[inline]
    fn is_char_boundary(&self, input: &[u8], pos: usize) -> bool {
        if pos == 0 || pos >= input.len() {
            return true;
        }

        let byte = input[pos];
        // UTF-8 character boundaries start with bytes that are either:
        // - ASCII (0xxxxxxx)
        // - Start of multi-byte sequence (11xxxxxx)
        // They do NOT start with continuation bytes (10xxxxxx)
        byte < 0x80 || byte >= 0xC0
    }

    #[inline]
    fn finalize_word(&mut self, final_position: usize) {
        if self.current_word_start != usize::MAX && self.word_end_count < MAX_POS {
            self.word_end_positions[self.word_end_count] = final_position;
            self.word_end_count += 1;
            self.current_word_start = usize::MAX;
        }
    }
}

// =============================================================================
// UNICODE CHARACTER CLASSES - MIRRORING CHARACTERCLASSES
// =============================================================================

/// Unicode character classification counters during SIMD processing
/// Mirrors CharacterClasses structure from classify.rs
#[derive(Debug, Clone)]
pub struct UnicodeCharacterClasses {
    letter_count: usize,
    digit_count: usize,
    punctuation_count: usize,
    whitespace_count: usize,
    total_count: usize,
}

impl UnicodeCharacterClasses {
    #[inline]
    fn new() -> Self {
        Self {
            letter_count: 0,
            digit_count: 0,
            punctuation_count: 0,
            whitespace_count: 0,
            total_count: 0,
        }
    }

    #[cfg(all(
        feature = "hwx-nightly",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[inline]
    unsafe fn update_avx512(
        &mut self,
        letter_mask: u64,
        digit_mask: u64,
        punct_mask: u64,
        space_mask: u64,
    ) {
        self.letter_count += letter_mask.count_ones() as usize;
        self.digit_count += digit_mask.count_ones() as usize;
        self.punctuation_count += punct_mask.count_ones() as usize;
        self.whitespace_count += space_mask.count_ones() as usize;
        self.total_count += LANES_AVX512_BYTES;
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "hwx-nightly")
    ))]
    #[inline]
    unsafe fn update_avx2(
        &mut self,
        letter_mask: i32,
        digit_mask: i32,
        punct_mask: i32,
        space_mask: i32,
    ) {
        self.letter_count += (letter_mask as u32).count_ones() as usize;
        self.digit_count += (digit_mask as u32).count_ones() as usize;
        self.punctuation_count += (punct_mask as u32).count_ones() as usize;
        self.whitespace_count += (space_mask as u32).count_ones() as usize;
        self.total_count += LANES_AVX2_BYTES;
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn update_neon(
        &mut self,
        letter_count: u32,
        digit_count: u32,
        punct_count: u32,
        space_count: u32,
    ) {
        self.letter_count += letter_count as usize;
        self.digit_count += digit_count as usize;
        self.punctuation_count += punct_count as usize;
        self.whitespace_count += space_count as usize;
        self.total_count += LANES_NEON_BYTES;
    }
}

// =============================================================================
// IMPLEMENTATION DETAILS - AVX-512 TOKENIZATION - MIRRORING CLASSIFY STRUCTURE
// =============================================================================

// Helper function for single string AVX-512 tokenization

// GPU Implementation - multi-block parallel tokenization with warp intrinsics and grid-stride loops
#[cfg(has_cuda)]
pub unsafe fn tokenize_single_string_gpu(
    string_to_tokenize: &mut [u8],
    len: usize,
    word_boundaries: &mut [u32],
) -> u32 {
    // Two-pass PTX: 1) mark flags per byte, 2) emit start/end pairs into boundaries, 3) compact
    const PTX_TOKENIZE_GPU: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .address_size 64

    // Pass 1: mark is_word flags (u32 0/1) into word_boundaries[0..len)
    .entry mark_word_flags(
      .param .u64 input_ptr,
      .param .u32 input_len,
      .param .u64 flags_ptr
    ) {
      .reg .u32 %r<64>;
      .reg .u64 %rd<32>;
      .reg .pred %p<32>;
      .reg .u8 %rc<4>;

      ld.param.u64 %rd0, [input_ptr];
      ld.param.u32 %r0, [input_len];
      ld.param.u64 %rd1, [flags_ptr];

      mov.u32 %r1, %tid.x;
      mov.u32 %r2, %ctaid.x;
      mov.u32 %r3, %ntid.x;
      mov.u32 %r4, %nctaid.x;
      mul.lo.u32 %r5, %r2, %r3;
      add.u32 %r6, %r5, %r1;     // global_tid
      mul.lo.u32 %r7, %r4, %r3;  // grid_stride

    f_loop:
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra f_done;

      // load byte
      cvt.u64.u32 %rd2, %r6;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.u8 %rc0, [%rd3];

      // is_word = [A-Z] | [a-z] | [0-9] | '_' | '.' | (byte>=128)
      cvt.u32.u8 %r8, %rc0;
      setp.ge.u32 %p1, %r8, 65; setp.le.u32 %p2, %r8, 90; and.pred %p3, %p1, %p2;
      setp.ge.u32 %p4, %r8, 97; setp.le.u32 %p5, %r8, 122; and.pred %p6, %p4, %p5;
      setp.ge.u32 %p7, %r8, 48; setp.le.u32 %p8, %r8, 57; and.pred %p9, %p7, %p8;
      setp.eq.u32 %p10, %r8, 95; // '_'
      setp.eq.u32 %p27, %r8, 46; // '.'
      setp.ge.u32 %p11, %r8, 128; // non-ASCII => treat as word char (safe for UTF-8)
      or.pred %p12, %p3, %p6; or.pred %p13, %p9, %p10; or.pred %p14, %p12, %p13; or.pred %p14, %p14, %p27; or.pred %p15, %p14, %p11;
      selp.u32 %r9, 1, 0, %p15;
      // Override for NBSP continuation byte (0xA0): treat as whitespace (non-word)
      // CPU path treats NBSP as whitespace; this aligns GPU behavior
      setp.eq.u32 %p16, %r8, 160; // 0xA0
      @%p16 mov.u32 %r9, 0;
      setp.eq.u32 %p16, %r8, 183;
      @%p16 mov.u32 %r9, 0;

      // Handle multi-byte whitespaces
      // If current is 0xC2 and next is 0xA0 (NBSP), force non-word
      setp.eq.u32 %p17, %r8, 194; // 0xC2
      @!%p17 bra chk_fullwidth_space;
      // bounds check i+1 < len
      add.u32 %r40, %r6, 1;
      setp.ge.u32 %p18, %r40, %r0;
      @%p18 bra chk_fullwidth_space;
      // load next byte
      cvt.u64.u32 %rd20, %r40;
      add.u64 %rd21, %rd0, %rd20;
      ld.global.u8 %rc1, [%rd21];
      cvt.u32.u8 %r41, %rc1;
      setp.eq.u32 %p19, %r41, 160; // 0xA0
      @%p19 mov.u32 %r9, 0;
      setp.eq.u32 %p19, %r41, 183;
      @%p19 mov.u32 %r9, 0;

    chk_fullwidth_space:
      // U+3000 IDEOGRAPHIC SPACE encoded as E3 80 80: zero flags for any of its bytes
      setp.eq.u32 %p20, %r8, 227; // 0xE3
      @!%p20 bra after_ws_overrides;
      // check next two bytes exist
      add.u32 %r42, %r6, 1;
      add.u32 %r43, %r6, 2;
      setp.ge.u32 %p21, %r43, %r0;
      @%p21 bra after_ws_overrides;
      // load next two
      cvt.u64.u32 %rd22, %r42; add.u64 %rd22, %rd0, %rd22; ld.global.u8 %rc2, [%rd22]; cvt.u32.u8 %r44, %rc2;
      cvt.u64.u32 %rd23, %r43; add.u64 %rd23, %rd0, %rd23; ld.global.u8 %rc3, [%rd23]; cvt.u32.u8 %r45, %rc3;
      setp.eq.u32 %p22, %r44, 128; // 0x80
      setp.eq.u32 %p23, %r45, 128; // 0x80
      and.pred %p24, %p22, %p23;
      @%p24 mov.u32 %r9, 0;

    after_ws_overrides:
      // store flag (u32) at flags_ptr + 4 + i*4 (index 0 reserved for count)
      add.u32 %r10, %r6, 1; // i + 1
      shl.b32 %r11, %r10, 2; // offset = (i+1) * 4
      cvt.u64.u32 %rd4, %r11;
      add.u64 %rd5, %rd1, %rd4;
      st.global.u32 [%rd5], %r9;

      add.u32 %r6, %r6, %r7;
      bra f_loop;

    f_done:
      ret;
    }

    // Pass 2: build (start,end) pairs from flags only (no byte re-scan)
    // Flags layout: flags_ptr[1..=len] holds u32 0/1 per byte (index 0 reserved)
    // Output layout: boundaries_ptr[0] = word_count (atomic),
    //                pairs written at byte offset base_bytes=(len+1)*4,
    //                as [start0,end0,start1,end1,...]
    .entry emit_boundaries_from_flags(
      .param .u64 input_ptr,        // unused
      .param .u32 input_len,
      .param .u64 boundaries_ptr,
      .param .u64 flags_ptr,
      .param .u32 max_boundaries
    ) {
      .reg .u32 %r<48>;
      .reg .u64 %rd<12>;
      .reg .pred %p<8>;

      // Load params
      ld.param.u32 %r0, [input_len];
      ld.param.u64 %rd1, [boundaries_ptr];
      ld.param.u64 %rd0, [flags_ptr];
      ld.param.u32 %r1, [max_boundaries];

      // base byte offset for pairs: (len+1)*4
      add.u32 %r2, %r0, 1;
      shl.b32 %r3, %r2, 2;
      cvt.u64.u32 %rd2, %r3; // rd2 = base bytes

      // grid-stride init
      mov.u32 %r4, %tid.x;
      mov.u32 %r5, %ctaid.x;
      mov.u32 %r6, %ntid.x;
      mov.u32 %r7, %nctaid.x;
      mad.lo.u32 %r8, %r5, %r6, %r4; // i = block*blockDim + thread
      mul.lo.u32 %r9, %r7, %r6;      // stride = gridDim * blockDim

    loop_i:
      setp.ge.u32 %p0, %r8, %r0;
      @%p0 bra end_kernel;

      // Load fi = flags[i]
      add.u32 %r10, %r8, 1;       // i+1 (flags start at index 1)
      shl.b32 %r11, %r10, 2;      // byte offset
      cvt.u64.u32 %rd3, %r11;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.u32 %r12, [%rd4];  // fi

      // Load prev flag (0 when i==0)
      setp.eq.u32 %p1, %r8, 0;
      @%p1 bra L_prev_zero;
      sub.u32 %r13, %r8, 1;          // i-1
      add.u32 %r14, %r13, 1;        // (i-1)+1
      shl.b32 %r15, %r14, 2;
      cvt.u64.u32 %rd5, %r15;
      add.u64 %rd6, %rd0, %rd5;
      ld.global.u32 %r16, [%rd6];   // fprev
      bra L_have_prev;
    L_prev_zero:
      mov.u32 %r16, 0;
    L_have_prev:

      // Start condition: fi != 0 && fprev == 0
      setp.ne.u32 %p2, %r12, 0;   // fi != 0
      setp.eq.u32 %p3, %r16, 0;   // prev==0
      and.pred %p4, %p2, %p3;
      @!%p4 bra cont_i;

      // Scan to end where next flag is 0 or at input end
      mov.u32 %r17, %r8;          // start = i
      mov.u32 %r18, %r8;          // e = i
    scan_j:
      add.u32 %r19, %r18, 1;      // j = e+1
      setp.ge.u32 %p5, %r19, %r0;
      @%p5 bra have_end;
      // fj = flags[j]
      add.u32 %r20, %r19, 1;
      shl.b32 %r21, %r20, 2;
      cvt.u64.u32 %rd7, %r21;
      add.u64 %rd8, %rd0, %rd7;
      ld.global.u32 %r22, [%rd8];  // fj
      setp.eq.u32 %p6, %r22, 0;
      @%p6 bra have_end;
      mov.u32 %r18, %r19;
      bra scan_j;

    have_end:
      // idx = atomicAdd(boundaries[0], 1)
      atom.global.add.u32 %r23, [%rd1], 1;
      // bounds check: base_index + 2*idx + 1 < max_boundaries
      shl.b32 %r24, %r23, 1;      // 2*idx
      add.u32 %r25, %r2, %r24;    // base_index + 2*idx
      add.u32 %r26, %r25, 1;      // +1 for end
      setp.ge.u32 %p7, %r26, %r1;
      @%p7 bra cont_i;

      // write pair
      shl.b32 %r27, %r24, 2;      // bytes offset
      cvt.u64.u32 %rd9, %r27;
      add.u64 %rd10, %rd1, %rd2;
      add.u64 %rd10, %rd10, %rd9;
      st.global.u32 [%rd10], %r17;  // start
      add.u32 %r28, %r18, 1;        // end = e+1
      add.u64 %rd11, %rd10, 4;
      st.global.u32 [%rd11], %r28;  // end

    cont_i:
      add.u32 %r8, %r8, %r9;
      bra loop_i;

    end_kernel:
      ret;
    }
  "#;

    // Use device buffers for correctness: copy input and boundaries to GPU, run kernels, copy back
    // Ensure device counter starts at 0
    if !word_boundaries.is_empty() {
        word_boundaries[0] = 0;
    }

    let _copy_back_u32s = unsafe {
        with_gpu_buffer_u32_inplace(
      word_boundaries,
      word_boundaries.len(),
      |gpu_boundaries, gpu_bounds_len| {
        // Run with a device copy of input bytes
        let _ = with_gpu_buffer_u8(string_to_tokenize, len, |gpu_input, input_len| {
          // Tune launch to input length to avoid massively oversubscribing for small inputs
          let threads: u32 = 256;
          let mut blocks: u32 = ((input_len as u32 + threads - 1) / threads)
            .max(1)
            .min(1024);
          if input_len as u32 <= 4096 {
            blocks = blocks.min(8);
          }
          // Prepare kernel params as pointers to values
          let param_input_ptr: u64 = gpu_input as u64;
          let param_input_len: u32 = input_len as u32;
          let param_flags_ptr: u64 = gpu_boundaries as u64;
          let res_mark = launch_ptx(
            PTX_TOKENIZE_GPU,
            &[],
            "mark_word_flags",
            blocks,
            threads,
            &[
              &param_input_ptr as *const _ as *const u8,
              &param_input_len as *const _ as *const u8,
              &param_flags_ptr as *const _ as *const u8,
            ],
          );
          if let Err(e) = res_mark {
            error!(
              "HWX GPU tokenization: mark_word_flags launch failed: {} (len={}, blocks={}, threads={})",
              e, input_len, blocks, threads
            );
          }

          // Prepare params for pass 2: (input_ptr, input_len, boundaries_ptr, flags_ptr, max_boundaries)
          let param_input_ptr2: u64 = gpu_input as u64;
          let param_input_len2: u32 = input_len as u32;
          let param_boundaries_ptr: u64 = gpu_boundaries as u64;
          let param_flags_ptr2: u64 = gpu_boundaries as u64;
          let param_max_boundaries: u32 = gpu_bounds_len as u32;
          let res_emit = launch_ptx(
            PTX_TOKENIZE_GPU,
            &[],
            "emit_boundaries_from_flags",
            blocks,
            threads,
            &[
              &param_input_ptr2 as *const _ as *const u8,
              &param_input_len2 as *const _ as *const u8,
              &param_boundaries_ptr as *const _ as *const u8,
              &param_flags_ptr2 as *const _ as *const u8,
              &param_max_boundaries as *const _ as *const u8,
            ],
          );
          if let Err(e) = res_emit {
            error!(
              "HWX GPU tokenization: emit_boundaries_from_flags launch failed: {} (len={}, blocks={}, threads={})",
              e, input_len, blocks, threads
            );
          }

          // Copy back the full buffer (contains count at [0] and pairs at base offset)
          gpu_bounds_len
        });

        // The helper copied back the full buffer
        gpu_bounds_len as u32
      },
    )
    .unwrap_or(0)
    } as u32;

    // Use the device-produced word count at index 0
    let word_count = if !word_boundaries.is_empty() {
        word_boundaries[0]
    } else {
        0
    };
    word_count
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn tokenize_single_string_avx512(
    string_to_tokenize: &mut [u8],
    len: usize,
    word_boundaries: &mut [u32],
) -> u32 {
    // Process input in chunks with deduplication
    const PROCESSING_CHUNK_SIZE: usize = 100;
    let mut input_pos = 0;
    let mut output_pos = 0;
    let mut total_words = 0u32;
    let mut last_word_end = 0u32;

    while input_pos < len && output_pos < word_boundaries.len() {
        // Determine chunk end - find next whitespace boundary
        let mut chunk_end = std::cmp::min(input_pos + PROCESSING_CHUNK_SIZE, len);
        if chunk_end < len {
            // Find whitespace to split at
            while chunk_end > input_pos && !string_to_tokenize[chunk_end].is_ascii_whitespace() {
                chunk_end -= 1;
            }
            if chunk_end == input_pos {
                chunk_end = std::cmp::min(input_pos + PROCESSING_CHUNK_SIZE, len);
            }
        }

        // Process this chunk with buffer flushes when hitting MAX_POSITIONS
        let mut boundary_signature = WordBoundarySignature::<MAX_POSITIONS_AVX512>::new();
        let mut char_classes = UnicodeCharacterClasses::new();
        let mut i = 0;

        loop {
            // Process SIMD chunks until we hit MAX_POSITIONS or run out of data
            while i + 64 <= (chunk_end - input_pos)
                && boundary_signature.word_start_count < MAX_POSITIONS_AVX512
            {
                let chunk =
                    _mm512_loadu_si512(string_to_tokenize.as_ptr().add(input_pos + i) as *const _);
                tokenize_chunk_avx512(
                    chunk,
                    &mut boundary_signature,
                    &mut char_classes,
                    input_pos + i,
                    string_to_tokenize,
                );
                i += 64;
            }

            // If we haven't hit MAX_POSITIONS, process remaining bytes and finalize
            let buffer_full = boundary_signature.word_start_count >= MAX_POSITIONS_AVX512;
            if !buffer_full {
                // Handle remaining bytes in this chunk
                if i < (chunk_end - input_pos) {
                    tokenize_scalar_remaining(
                        &mut string_to_tokenize[input_pos + i..chunk_end],
                        &mut boundary_signature,
                        input_pos + i,
                        chunk_end,
                    );
                }

                // Finalize if last chunk
                if chunk_end >= len {
                    boundary_signature.finalize_word(len);
                }
            }

            // Flush current buffer to output
            let chunk_words = std::cmp::min(
                boundary_signature.word_start_count,
                boundary_signature.word_end_count,
            );
            let space_left = (word_boundaries.len() - output_pos) / 2;
            let words_to_copy =
                std::cmp::min(std::cmp::min(chunk_words, MAX_POSITIONS_AVX512), space_left);

            let mut words_added = 0;
            for j in 0..words_to_copy {
                let start_pos = boundary_signature.word_start_positions[j] as u32;
                let end_pos = boundary_signature.word_end_positions[j] as u32;

                // Skip words that end before or at the last word we processed
                if end_pos > last_word_end {
                    if output_pos + words_added * 2 + 1 < word_boundaries.len() {
                        word_boundaries[output_pos + words_added * 2] = start_pos;
                        word_boundaries[output_pos + words_added * 2 + 1] = end_pos;
                        last_word_end = end_pos;
                        words_added += 1;
                    }
                }
            }

            total_words += words_added as u32;
            output_pos += words_added * 2;

            // If buffer was full and we haven't processed all bytes, reset and continue
            if buffer_full && i < (chunk_end - input_pos) {
                boundary_signature = WordBoundarySignature::<MAX_POSITIONS_AVX512>::new();
                char_classes = UnicodeCharacterClasses::new();
            } else {
                break;
            }
        }

        // Move to next chunk with minimal overlap to catch boundary words
        if chunk_end < len {
            // Skip whitespace and back up slightly to catch any word that might span the boundary
            while chunk_end < len && string_to_tokenize[chunk_end].is_ascii_whitespace() {
                chunk_end += 1;
            }
            input_pos = chunk_end.saturating_sub(10); // Small overlap
        } else {
            input_pos = chunk_end;
        }
    }

    total_words
}

// GPU version of tokenize_chunk - processes 32 bytes with warp intrinsics
#[cfg(has_cuda)]
pub unsafe fn tokenize_chunk_gpu<const MAX_POS: usize>(
    chunk: *const u8,
    signature: &mut WordBoundarySignature<MAX_POS>,
    char_classes: &mut UnicodeCharacterClasses,
    position: usize,
    full_input: &[u8],
) {
    const PTX_TOKENIZE_CHUNK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .func (.param .b32 result) create_unicode_letter_mask_ptx(.param .u64 chunk_ptr);
    .extern .func (.param .b32 result) create_unicode_digit_mask_ptx(.param .u64 chunk_ptr);
    .extern .func (.param .b32 result) create_unicode_punctuation_mask_ptx(.param .u64 chunk_ptr);
    .extern .func (.param .b32 result) create_unicode_whitespace_mask_ptx(.param .u64 chunk_ptr);
    
    .entry tokenize_chunk_kernel(
      .param .u64 chunk_ptr,
      .param .u64 signature_ptr,
      .param .u64 char_classes_ptr,
      .param .u32 position,
      .param .u64 full_input_ptr,
      .param .u32 full_input_len
    ) {
      .reg .u32 %r<30>;
      .reg .u64 %rd<15>;
      .reg .pred %p<10>;
      
      ld.param.u64 %rd0, [chunk_ptr];
      ld.param.u64 %rd1, [signature_ptr];
      ld.param.u64 %rd2, [char_classes_ptr];
      ld.param.u32 %r10, [position];
      ld.param.u64 %rd3, [full_input_ptr];
      ld.param.u32 %r11, [full_input_len];
      
      // Call PTX functions to get masks
      .param .b32 letter_mask_param;
      .param .b32 digit_mask_param;
      .param .b32 punct_mask_param;
      .param .b32 space_mask_param;
      
      call create_unicode_letter_mask_ptx, (letter_mask_param), (%rd0);
      ld.param.b32 %r0, [letter_mask_param];  // letter_mask
      
      call create_unicode_digit_mask_ptx, (digit_mask_param), (%rd0);
      ld.param.b32 %r1, [digit_mask_param];  // digit_mask
      
      call create_unicode_punctuation_mask_ptx, (punct_mask_param), (%rd0);
      ld.param.b32 %r2, [punct_mask_param];  // punct_mask
      
      call create_unicode_whitespace_mask_ptx, (space_mask_param), (%rd0);
      ld.param.b32 %r3, [space_mask_param];  // space_mask
      
      // Store masks to signature structure
      st.global.u32 [%rd1], %r0;        // letter_mask at offset 0
      st.global.u32 [%rd1 + 4], %r1;    // digit_mask at offset 4
      st.global.u32 [%rd1 + 8], %r2;    // punct_mask at offset 8
      st.global.u32 [%rd1 + 12], %r3;   // space_mask at offset 12
      st.global.u32 [%rd1 + 16], %r10;  // position at offset 16
      
      // Update character class counts
      popc.b32 %r4, %r0;  // letter count
      popc.b32 %r5, %r1;  // digit count
      popc.b32 %r6, %r2;  // punct count
      popc.b32 %r7, %r3;  // space count
      
      // Atomic add to character class counts
      atom.global.add.u32 %r8, [%rd2], %r4;        // letter_count
      atom.global.add.u32 %r9, [%rd2 + 4], %r5;    // digit_count
      atom.global.add.u32 %r12, [%rd2 + 8], %r6;   // punct_count
      atom.global.add.u32 %r13, [%rd2 + 12], %r7;  // space_count
      
      // Count word boundaries - simplified version
      // A word starts when we transition from non-word to word char
      // For now, just count letters+digits as word chars
      or.b32 %r14, %r0, %r1;  // word_char_mask = letter_mask | digit_mask
      popc.b32 %r15, %r14;     // Count word chars in this chunk
      
      // Simple heuristic: if we have word chars, we likely have at least one word
      // This is a placeholder - real implementation would track transitions
      setp.ne.u32 %p5, %r15, 0;
      selp.u32 %r16, 1, 0, %p5;  // word_count = (word_chars > 0) ? 1 : 0
      
      // Return the word count
      st.param.b32 [word_count_result], %r16;
      ret;
    }
  "#;

    let (blocks, threads) = (1, 32); // Single chunk = single warp

    let _ = launch_ptx(
        PTX_TOKENIZE_CHUNK,
        &[
            PTX_CREATE_UNICODE_LETTER_MASK,
            PTX_CREATE_UNICODE_DIGIT_MASK,
            PTX_CREATE_UNICODE_PUNCT_MASK,
            PTX_CREATE_UNICODE_SPACE_MASK,
        ],
        "tokenize_chunk_kernel",
        blocks,
        threads,
        &[
            chunk as *const u8,
            signature as *mut WordBoundarySignature<MAX_POS> as *const u8,
            char_classes as *mut UnicodeCharacterClasses as *const u8,
            &position as *const _ as *const u8,
            full_input.as_ptr() as *const u8,
            &(full_input.len() as u32) as *const _ as *const u8,
        ],
    );
}

// AVX-512 chunk tokenization using comprehensive Unicode character detection
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn tokenize_chunk_avx512(
    chunk: std::arch::x86_64::__m512i,
    signature: &mut WordBoundarySignature<MAX_POSITIONS_AVX512>,
    char_classes: &mut UnicodeCharacterClasses,
    position: usize,
    full_input: &[u8],
) {
    //  DEEP SIMD: Create Unicode character class masks in parallel
    let letter_mask = create_unicode_letter_mask_avx512(chunk);
    let digit_mask = create_unicode_digit_mask_avx512(chunk);
    let punct_mask = create_unicode_punctuation_mask_avx512(chunk);
    let space_mask = create_unicode_whitespace_mask_avx512(chunk);

    // Update boundary signature with SIMD results
    signature.update_avx512(
        letter_mask,
        digit_mask,
        punct_mask,
        space_mask,
        position,
        full_input,
    );

    // Update character class counts
    char_classes.update_avx512(letter_mask, digit_mask, punct_mask, space_mask);

    // Tokenization only handles word boundary detection, not lowercase conversion
}

// GPU implementation of create_unicode_letter_mask using PTX assembly

#[inline]
#[cfg(has_cuda)]
pub unsafe fn create_unicode_letter_mask_gpu(bytes: *const u8, len: usize) -> u32 {
    const PTX_CREATE_LETTER_MASK_KERNEL: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .entry create_unicode_letter_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
  ) {
      .reg .pred %p<30>;
      .reg .u32 %r<30>;
      .reg .u64 %rd<30>;
      .reg .u8 %b<10>;

      // Load parameters
      ld.param.u64 %rd0, [bytes_ptr];
      ld.param.u32 %r0, [len];
      ld.param.u64 %rd1, [mask_ptr];

      // Each thread in warp processes one byte (thread ID = byte position)
      mov.u32 %r1, %tid.x;       // Use thread ID as position (0-31)
      
      // Check if this thread should process a byte
      setp.ge.u32 %p0, %r1, %r0;  // position >= len
      
      // Default predicate to false (no letter)
      mov.pred %p26, 0;
      
      // Only process if within bounds
      @%p0 bra create_mask;

      // Load byte at position (coalesced access when threads load consecutive bytes)
      cvt.u64.u32 %rd3, %r1;
      add.u64 %rd4, %rd0, %rd3;
      ld.global.u8 %b0, [%rd4];

      // ASCII letters: A-Z
      setp.ge.u8 %p2, %b0, 65;  // byte >= 'A'
      setp.le.u8 %p3, %b0, 90;  // byte <= 'Z'
      and.pred %p4, %p2, %p3;
      
      // ASCII letters: a-z
      setp.ge.u8 %p5, %b0, 97;  // byte >= 'a'
      setp.le.u8 %p6, %b0, 122; // byte <= 'z'
      and.pred %p7, %p5, %p6;
      
      // Underscore mask - treat underscore as part of word characters
      setp.eq.u8 %p8, %b0, 95;  // byte == '_'
      
      // Degree symbol first byte (0xC2) - treat as part of word characters for 25°C
      setp.eq.s8 %p9, %b0, -62;  // byte == 0xC2 (as signed -62)
      
      // Degree symbol second byte (0xB0) - treat as part of word characters for 25°C
      setp.eq.s8 %p10, %b0, -80;  // byte == 0xB0 (as signed -80)
      
      // UTF-8 continuation bytes (0x80-0xBF) - part of multi-byte characters
      setp.ge.s8 %p11, %b0, -128;  // byte >= 0x80 (as signed -128)
      setp.le.s8 %p12, %b0, -65;   // byte <= 0xBF (as signed -65)
      and.pred %p13, %p11, %p12;
      
      // UTF-8 2-byte start (0xC0-0xDF) - Latin Extended, etc.
      setp.ge.s8 %p14, %b0, -64;  // byte >= 0xC0 (as signed -64)
      setp.le.s8 %p15, %b0, -33;  // byte <= 0xDF (as signed -33)
      and.pred %p16, %p14, %p15;
      
      // UTF-8 3-byte start (0xE0-0xEF) - covers most international scripts
      setp.ge.s8 %p17, %b0, -32;  // byte >= 0xE0 (as signed -32)
      setp.le.s8 %p18, %b0, -17;  // byte <= 0xEF (as signed -17)
      and.pred %p19, %p17, %p18;
      
      // Combine all checks (matching AVX-512 logic exactly)
      or.pred %p20, %p4, %p7;   // upper | lower
      or.pred %p21, %p20, %p8;  // | underscore
      or.pred %p22, %p21, %p9;  // | degree_first
      or.pred %p23, %p22, %p10; // | degree_second
      or.pred %p24, %p23, %p13; // | continuation
      or.pred %p25, %p24, %p16; // | utf8_2byte
      or.pred %p26, %p25, %p19; // | utf8_3byte

  create_mask:
      // Use warp ballot to create mask from all threads' predicates
      // Each thread contributes one bit based on whether its byte is a letter
      vote.ballot.b32 %r2, %p26;
      
      // Thread 0 stores the final mask (only one thread writes to avoid races)
      setp.eq.u32 %p27, %r1, 0;
      @!%p27 bra done;
      
      // Store result mask (u32 for 32 bytes)
      st.global.u32 [%rd1], %r2;

  done:
      ret;
  }
  "#;

    let mut mask: u32 = 0;

    // Single warp processes 32 bytes with Block^Warp^Vector optimization
    let (blocks, threads) = (1, 32);

    let _ = launch_ptx(
        PTX_CREATE_LETTER_MASK_KERNEL,
        &[],
        "create_unicode_letter_mask_ptx",
        blocks,
        threads,
        &[
            bytes as *const u8,
            &(len.min(32) as u32) as *const u32 as *const u8,
            &mut mask as *mut u32 as *const u8,
        ],
    );

    mask
}

// PTX constants for mask creation - used by tokenize_chunk_gpu
// PTX extern declaration that references the kernel defined in create_unicode_digit_mask_gpu function
pub const PTX_CREATE_UNICODE_DIGIT_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_unicode_digit_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
    );
"#;

// PTX extern declaration that references the kernel defined in create_unicode_punctuation_mask_gpu function
pub const PTX_CREATE_UNICODE_PUNCT_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_unicode_punctuation_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
    );
"#;

// PTX extern declaration that references the kernel defined in create_unicode_whitespace_mask_gpu function
pub const PTX_CREATE_UNICODE_SPACE_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_unicode_whitespace_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
    );
"#;

// PTX extern declaration that references the kernel defined in create_unicode_letter_mask_gpu function
pub const PTX_CREATE_UNICODE_LETTER_MASK: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    
    .extern .entry create_unicode_letter_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
    );
"#;

// Create Unicode letter detection mask (A-Z, a-z, UTF-8 sequences) using AVX-512
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_unicode_letter_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    // ASCII letters: A-Z, a-z
    let upper_mask = _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(b'A' as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(b'Z' as i8));
    let lower_mask = _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(b'a' as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(b'z' as i8));

    // Underscore mask - treat underscore as part of word characters
    let underscore_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'_' as i8));

    // Degree symbol first byte (0xC2) - treat as part of word characters for 25°C
    let degree_first_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(0xC2u8 as i8));

    // Degree symbol second byte (0xB0) - treat as part of word characters for 25°C
    let degree_second_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(0xB0u8 as i8));

    // UTF-8 continuation bytes (0x80-0xBF) - part of multi-byte characters
    let continuation_mask = _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(0x80u8 as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(0xBFu8 as i8));

    // UTF-8 2-byte start (0xC0-0xDF) - Latin Extended, etc.
    let utf8_2byte_mask = _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(0xC0u8 as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(0xDFu8 as i8));

    // UTF-8 3-byte start (0xE0-0xEF) - covers most international scripts
    let utf8_3byte_mask = _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(0xE0u8 as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(0xEFu8 as i8));

    upper_mask
        | lower_mask
        | underscore_mask
        | degree_first_mask
        | degree_second_mask
        | continuation_mask
        | utf8_2byte_mask
        | utf8_3byte_mask
}

// GPU implementation of create_unicode_digit_mask using PTX assembly

#[inline]
#[cfg(has_cuda)]
pub unsafe fn create_unicode_digit_mask_gpu(bytes: *const u8, len: usize) -> u32 {
    // This constant is embedded locally but we export a public version
    const PTX_CREATE_DIGIT_MASK_KERNEL: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .entry create_unicode_digit_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
  ) {
      .reg .pred %p<32>;
      .reg .u32 %r<32>;
      .reg .u64 %rd<32>;
      .reg .u8 %b<16>;

      // Load parameters
      ld.param.u64 %rd0, [bytes_ptr];
      ld.param.u32 %r0, [len];
      ld.param.u64 %rd1, [mask_ptr];

      // Block^Warp^Vector optimization: 8 threads load 4 bytes each = 32 bytes total
      mov.u32 %r1, %tid.x;       // Thread ID (0-7 for vector loads)
      and.u32 %r2, %r1, 7;       // Lane within 8-thread group (0-7)
      
      // Default to no matches
      mov.u32 %r20, 0;           // Local digit mask for this thread's 4 bytes
      
      // Check if we have at least 4 bytes to load
      setp.lt.u32 %p0, %r0, 4;
      @%p0 bra fallback_scalar;
      
      // Vector load: each of 8 threads loads 4 consecutive bytes
      mul.lo.u32 %r3, %r2, 4;    // Byte offset for this thread (0, 4, 8, 12, 16, 20, 24, 28)
      setp.ge.u32 %p1, %r3, %r0; // Check if offset >= len
      @%p1 bra warp_reduce;       // Skip if beyond bounds
      
      // Load 4 bytes using vector instruction
      cvt.u64.u32 %rd2, %r3;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.v4.u8 {%b0, %b1, %b2, %b3}, [%rd3];
      
      // Process 4 bytes in parallel: check if each is a digit (0x30-0x39)
      // Byte 0
      setp.ge.u8 %p4, %b0, 48;   // >= '0'
      setp.le.u8 %p5, %b0, 57;   // <= '9'
      and.pred %p6, %p4, %p5;
      selp.u32 %r4, 1, 0, %p6;   // Convert to bit
      
      // Byte 1  
      setp.ge.u8 %p7, %b1, 48;
      setp.le.u8 %p8, %b1, 57;
      and.pred %p9, %p7, %p8;
      selp.u32 %r5, 2, 0, %p9;   // Bit position 1
      
      // Byte 2
      setp.ge.u8 %p10, %b2, 48;
      setp.le.u8 %p11, %b2, 57;
      and.pred %p12, %p10, %p11;
      selp.u32 %r6, 4, 0, %p12;  // Bit position 2
      
      // Byte 3
      setp.ge.u8 %p13, %b3, 48;
      setp.le.u8 %p14, %b3, 57;
      and.pred %p15, %p13, %p14;
      selp.u32 %r7, 8, 0, %p15;  // Bit position 3
      
      // Combine the 4 bits for this thread's bytes
      or.b32 %r8, %r4, %r5;
      or.b32 %r9, %r6, %r7;
      or.b32 %r20, %r8, %r9;     // 4-bit mask for this thread
      
      // Shift to correct position in final 32-bit mask
      shl.b32 %r20, %r20, %r3;   // Shift by byte offset
      bra warp_reduce;

  fallback_scalar:
      // Handle case with < 4 bytes total - each thread checks 1 byte
      and.u32 %r10, %r1, 31;     // Thread within warp (0-31)
      setp.ge.u32 %p16, %r10, %r0;
      @%p16 bra warp_reduce;
      
      cvt.u64.u32 %rd4, %r10;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u8 %b4, [%rd5];
      
      setp.ge.u8 %p17, %b4, 48;
      setp.le.u8 %p18, %b4, 57;
      and.pred %p19, %p17, %p18;
      selp.u32 %r11, 1, 0, %p19;
      shl.b32 %r20, %r11, %r10;  // Position bit correctly

  warp_reduce:
      // Warp shuffle butterfly reduction to combine all thread results
      shfl.sync.bfly.b32 %r21, %r20, 16, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r21;
      shfl.sync.bfly.b32 %r22, %r20, 8, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r22;
      shfl.sync.bfly.b32 %r23, %r20, 4, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r23;
      shfl.sync.bfly.b32 %r24, %r20, 2, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r24;
      shfl.sync.bfly.b32 %r25, %r20, 1, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r25;
      
      // Thread 0 stores final result
      setp.eq.u32 %p20, %r1, 0;
      @!%p20 bra done;
      st.global.u32 [%rd1], %r20;

  done:
      ret;
  }
  "#;

    let mut mask: u32 = 0;
    let (blocks, threads) = (1, 32); // Single warp for 32-byte processing

    let _ = launch_ptx(
        PTX_CREATE_DIGIT_MASK_KERNEL,
        &[],
        "create_unicode_digit_mask_ptx",
        blocks,
        threads,
        &[
            bytes as *const u8,
            &(len.min(32) as u32) as *const u32 as *const u8,
            &mut mask as *mut u32 as *const u8,
        ],
    );

    mask
}

// Create Unicode digit detection mask (0-9) using AVX-512
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_unicode_digit_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    let digit_0 = _mm512_set1_epi8(b'0' as i8);
    let digit_9 = _mm512_set1_epi8(b'9' as i8);

    let ge_0 = _mm512_cmpge_epi8_mask(chunk, digit_0);
    let le_9 = _mm512_cmple_epi8_mask(chunk, digit_9);

    ge_0 & le_9
}

// GPU implementation of create_unicode_punctuation_mask using PTX assembly

#[inline]
#[cfg(has_cuda)]
pub unsafe fn create_unicode_punctuation_mask_gpu(bytes: *const u8, len: usize) -> u32 {
    const PTX_CREATE_PUNCT_MASK_KERNEL: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .entry create_unicode_punctuation_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
  ) {
      .reg .pred %p<32>;
      .reg .u32 %r<32>;
      .reg .u64 %rd<32>;
      .reg .u8 %b<16>;

      // Load parameters
      ld.param.u64 %rd0, [bytes_ptr];
      ld.param.u32 %r0, [len];
      ld.param.u64 %rd1, [mask_ptr];

      // Block^Warp^Vector optimization: 8 threads load 4 bytes each = 32 bytes total
      mov.u32 %r1, %tid.x;       // Thread ID (0-7 for vector loads)
      and.u32 %r2, %r1, 7;       // Lane within 8-thread group (0-7)
      
      // Default to no matches
      mov.u32 %r20, 0;           // Local punctuation mask for this thread's 4 bytes
      
      // Check if we have at least 4 bytes to load
      setp.lt.u32 %p0, %r0, 4;
      @%p0 bra fallback_scalar;
      
      // Vector load: each of 8 threads loads 4 consecutive bytes
      mul.lo.u32 %r3, %r2, 4;    // Byte offset for this thread (0, 4, 8, 12, 16, 20, 24, 28)
      setp.ge.u32 %p1, %r3, %r0; // Check if offset >= len
      @%p1 bra warp_reduce;       // Skip if beyond bounds
      
      // Load 4 bytes using vector instruction
      cvt.u64.u32 %rd2, %r3;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.v4.u8 {%b0, %b1, %b2, %b3}, [%rd3];
      
      // Process 4 bytes in parallel: check if each is punctuation (33-47: ! to /)
      // Byte 0
      setp.ge.u8 %p4, %b0, 33;   // >= '!'
      setp.le.u8 %p5, %b0, 47;   // <= '/'
      and.pred %p6, %p4, %p5;
      selp.u32 %r4, 1, 0, %p6;   // Convert to bit
      
      // Byte 1  
      setp.ge.u8 %p7, %b1, 33;
      setp.le.u8 %p8, %b1, 47;
      and.pred %p9, %p7, %p8;
      selp.u32 %r5, 2, 0, %p9;   // Bit position 1
      
      // Byte 2
      setp.ge.u8 %p10, %b2, 33;
      setp.le.u8 %p11, %b2, 47;
      and.pred %p12, %p10, %p11;
      selp.u32 %r6, 4, 0, %p12;  // Bit position 2
      
      // Byte 3
      setp.ge.u8 %p13, %b3, 33;
      setp.le.u8 %p14, %b3, 47;
      and.pred %p15, %p13, %p14;
      selp.u32 %r7, 8, 0, %p15;  // Bit position 3
      
      // Combine the 4 bits for this thread's bytes
      or.b32 %r8, %r4, %r5;
      or.b32 %r9, %r6, %r7;
      or.b32 %r20, %r8, %r9;     // 4-bit mask for this thread
      
      // Shift to correct position in final 32-bit mask
      shl.b32 %r20, %r20, %r3;   // Shift by byte offset
      bra warp_reduce;

  fallback_scalar:
      // Handle case with < 4 bytes total - each thread checks 1 byte
      and.u32 %r10, %r1, 31;     // Thread within warp (0-31)
      setp.ge.u32 %p16, %r10, %r0;
      @%p16 bra warp_reduce;
      
      cvt.u64.u32 %rd4, %r10;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u8 %b4, [%rd5];
      
      setp.ge.u8 %p17, %b4, 33;
      setp.le.u8 %p18, %b4, 47;
      and.pred %p19, %p17, %p18;
      selp.u32 %r11, 1, 0, %p19;
      shl.b32 %r20, %r11, %r10;  // Position bit correctly

  warp_reduce:
      // Warp shuffle butterfly reduction to combine all thread results
      shfl.sync.bfly.b32 %r21, %r20, 16, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r21;
      shfl.sync.bfly.b32 %r22, %r20, 8, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r22;
      shfl.sync.bfly.b32 %r23, %r20, 4, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r23;
      shfl.sync.bfly.b32 %r24, %r20, 2, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r24;
      shfl.sync.bfly.b32 %r25, %r20, 1, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r25;
      
      // Thread 0 stores final result
      setp.eq.u32 %p20, %r1, 0;
      @!%p20 bra done;
      st.global.u32 [%rd1], %r20;

  done:
      ret;
  }
  "#;

    let mut mask: u32 = 0;
    let (blocks, threads) = (1, 32); // Single warp for 32-byte processing

    let _ = launch_ptx(
        PTX_CREATE_PUNCT_MASK_KERNEL,
        &[],
        "create_unicode_punctuation_mask_ptx",
        blocks,
        threads,
        &[
            bytes as *const u8,
            &(len.min(32) as u32) as *const u32 as *const u8,
            &mut mask as *mut u32 as *const u8,
        ],
    );

    mask
}

// Create Unicode punctuation detection mask using AVX-512
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_unicode_punctuation_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    // Basic punctuation range check
    _mm512_cmpge_epi8_mask(chunk, _mm512_set1_epi8(b'!' as i8))
        & _mm512_cmple_epi8_mask(chunk, _mm512_set1_epi8(b'/' as i8))
}

// GPU implementation of create_unicode_whitespace_mask using PTX assembly

#[inline]
#[cfg(has_cuda)]
pub unsafe fn create_unicode_whitespace_mask_gpu(bytes: *const u8, len: usize) -> u32 {
    const PTX_CREATE_SPACE_MASK_KERNEL: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .entry create_unicode_whitespace_mask_ptx(
      .param .u64 bytes_ptr,
      .param .u32 len,
      .param .u64 mask_ptr
  ) {
      .reg .pred %p<48>;
      .reg .u32 %r<32>;
      .reg .u64 %rd<32>;
      .reg .u8 %b<16>;

      // Load parameters
      ld.param.u64 %rd0, [bytes_ptr];
      ld.param.u32 %r0, [len];
      ld.param.u64 %rd1, [mask_ptr];

      // Block^Warp^Vector optimization: 8 threads load 4 bytes each = 32 bytes total
      mov.u32 %r1, %tid.x;       // Thread ID (0-7 for vector loads)
      and.u32 %r2, %r1, 7;       // Lane within 8-thread group (0-7)
      
      // Default to no matches
      mov.u32 %r20, 0;           // Local whitespace mask for this thread's 4 bytes
      
      // Check if we have at least 4 bytes to load
      setp.lt.u32 %p0, %r0, 4;
      @%p0 bra fallback_scalar;
      
      // Vector load: each of 8 threads loads 4 consecutive bytes
      mul.lo.u32 %r3, %r2, 4;    // Byte offset for this thread (0, 4, 8, 12, 16, 20, 24, 28)
      setp.ge.u32 %p1, %r3, %r0; // Check if offset >= len
      @%p1 bra warp_reduce;       // Skip if beyond bounds
      
      // Load 4 bytes using vector instruction
      cvt.u64.u32 %rd2, %r3;
      add.u64 %rd3, %rd0, %rd2;
      ld.global.v4.u8 {%b0, %b1, %b2, %b3}, [%rd3];
      
      // Process 4 bytes in parallel: check if each is whitespace (space=32, tab=9, newline=10, cr=13)
      // Byte 0
      setp.eq.u8 %p4, %b0, 32;   // space
      setp.eq.u8 %p5, %b0, 9;    // tab
      setp.eq.u8 %p6, %b0, 10;   // newline
      setp.eq.u8 %p7, %b0, 13;   // carriage return
      or.pred %p8, %p4, %p5;     // space | tab
      or.pred %p9, %p6, %p7;     // newline | cr
      or.pred %p10, %p8, %p9;    // all whitespace
      selp.u32 %r4, 1, 0, %p10;  // Convert to bit
      
      // Byte 1  
      setp.eq.u8 %p11, %b1, 32;
      setp.eq.u8 %p12, %b1, 9;
      setp.eq.u8 %p13, %b1, 10;
      setp.eq.u8 %p14, %b1, 13;
      or.pred %p15, %p11, %p12;
      or.pred %p16, %p13, %p14;
      or.pred %p17, %p15, %p16;
      selp.u32 %r5, 2, 0, %p17;  // Bit position 1
      
      // Byte 2
      setp.eq.u8 %p18, %b2, 32;
      setp.eq.u8 %p19, %b2, 9;
      setp.eq.u8 %p20, %b2, 10;
      setp.eq.u8 %p21, %b2, 13;
      or.pred %p22, %p18, %p19;
      or.pred %p23, %p20, %p21;
      or.pred %p24, %p22, %p23;
      selp.u32 %r6, 4, 0, %p24;  // Bit position 2
      
      // Byte 3
      setp.eq.u8 %p25, %b3, 32;
      setp.eq.u8 %p26, %b3, 9;
      setp.eq.u8 %p27, %b3, 10;
      setp.eq.u8 %p28, %b3, 13;
      or.pred %p29, %p25, %p26;
      or.pred %p30, %p27, %p28;
      or.pred %p31, %p29, %p30;
      selp.u32 %r7, 8, 0, %p31;  // Bit position 3
      
      // Combine the 4 bits for this thread's bytes
      or.b32 %r8, %r4, %r5;
      or.b32 %r9, %r6, %r7;
      or.b32 %r20, %r8, %r9;     // 4-bit mask for this thread
      
      // Shift to correct position in final 32-bit mask
      shl.b32 %r20, %r20, %r3;   // Shift by byte offset
      bra warp_reduce;

  fallback_scalar:
      // Handle case with < 4 bytes total - each thread checks 1 byte
      and.u32 %r10, %r1, 31;     // Thread within warp (0-31)
      setp.ge.u32 %p32, %r10, %r0;
      @%p32 bra warp_reduce;
      
      cvt.u64.u32 %rd4, %r10;
      add.u64 %rd5, %rd0, %rd4;
      ld.global.u8 %b4, [%rd5];
      
      setp.eq.u8 %p33, %b4, 32;  // space
      setp.eq.u8 %p34, %b4, 9;   // tab  
      setp.eq.u8 %p35, %b4, 10;  // newline
      setp.eq.u8 %p36, %b4, 13;  // carriage return
      or.pred %p37, %p33, %p34;  // space | tab
      or.pred %p38, %p35, %p36;  // newline | cr
      or.pred %p39, %p37, %p38;  // all whitespace
      selp.u32 %r11, 1, 0, %p39;
      shl.b32 %r20, %r11, %r10;  // Position bit correctly

  warp_reduce:
      // Warp shuffle butterfly reduction to combine all thread results
      shfl.sync.bfly.b32 %r21, %r20, 16, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r21;
      shfl.sync.bfly.b32 %r22, %r20, 8, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r22;
      shfl.sync.bfly.b32 %r23, %r20, 4, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r23;
      shfl.sync.bfly.b32 %r24, %r20, 2, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r24;
      shfl.sync.bfly.b32 %r25, %r20, 1, 0x1f, 0xffffffff;
      or.b32 %r20, %r20, %r25;
      
      // Thread 0 stores final result
      setp.eq.u32 %p40, %r1, 0;
      @!%p40 bra done;
      st.global.u32 [%rd1], %r20;

  done:
      ret;
  }
  "#;

    let mut mask: u32 = 0;
    let (blocks, threads) = (1, 32); // Single warp for 32-byte processing

    let _ = launch_ptx(
        PTX_CREATE_SPACE_MASK_KERNEL,
        &[],
        "create_unicode_whitespace_mask_ptx",
        blocks,
        threads,
        &[
            bytes as *const u8,
            &(len.min(32) as u32) as *const u32 as *const u8,
            &mut mask as *mut u32 as *const u8,
        ],
    );

    mask
}

// Create Unicode whitespace detection mask using AVX-512
#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn create_unicode_whitespace_mask_avx512(chunk: std::arch::x86_64::__m512i) -> u64 {
    let space_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b' ' as i8));
    let tab_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'\t' as i8));
    let newline_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'\n' as i8));
    let cr_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'\r' as i8));

    space_mask | tab_mask | newline_mask | cr_mask
}

// =============================================================================
// IMPLEMENTATION DETAILS - AVX2 TOKENIZATION - MIRRORING CLASSIFY STRUCTURE
// =============================================================================

// Helper function for single string AVX2 tokenization
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn tokenize_single_string_avx2(
    string_to_tokenize: &mut [u8],
    len: usize,
    word_boundaries: &mut [u32],
) -> u32 {
    // Process input in chunks with deduplication
    const PROCESSING_CHUNK_SIZE: usize = 100;
    let mut input_pos = 0;
    let mut output_pos = 0;
    let mut total_words = 0u32;
    let mut last_word_end = 0u32;

    while input_pos < len && output_pos < word_boundaries.len() {
        // Determine chunk end - find next whitespace boundary
        let mut chunk_end = std::cmp::min(input_pos + PROCESSING_CHUNK_SIZE, len);
        if chunk_end < len {
            // Find whitespace to split at
            while chunk_end > input_pos && !string_to_tokenize[chunk_end].is_ascii_whitespace() {
                chunk_end -= 1;
            }
            if chunk_end == input_pos {
                chunk_end = std::cmp::min(input_pos + PROCESSING_CHUNK_SIZE, len);
            }
        }

        // Process this chunk with buffer flushes when hitting MAX_POSITIONS
        let mut boundary_signature = WordBoundarySignature::<MAX_POSITIONS_AVX2>::new();
        let mut char_classes = UnicodeCharacterClasses::new();
        let mut i = 0;

        loop {
            // Process SIMD chunks until we hit MAX_POSITIONS or run out of data
            while i + 32 <= (chunk_end - input_pos)
                && boundary_signature.word_start_count < MAX_POSITIONS_AVX2
            {
                let chunk =
                    _mm256_loadu_si256(string_to_tokenize.as_ptr().add(input_pos + i) as *const _);
                tokenize_chunk_avx2(
                    chunk,
                    &mut boundary_signature,
                    &mut char_classes,
                    input_pos + i,
                    string_to_tokenize,
                );
                i += 32;
            }

            // If we haven't hit MAX_POSITIONS, process remaining bytes and finalize
            let buffer_full = boundary_signature.word_start_count >= MAX_POSITIONS_AVX2;
            if !buffer_full {
                // Handle remaining bytes in this chunk
                if i < (chunk_end - input_pos) {
                    tokenize_scalar_remaining(
                        &mut string_to_tokenize[input_pos + i..chunk_end],
                        &mut boundary_signature,
                        input_pos + i,
                        chunk_end,
                    );
                }

                // Finalize if last chunk
                if chunk_end >= len {
                    boundary_signature.finalize_word(len);
                }
            }

            // Flush current buffer to output
            let chunk_words = std::cmp::min(
                boundary_signature.word_start_count,
                boundary_signature.word_end_count,
            );
            let space_left = (word_boundaries.len() - output_pos) / 2;
            let words_to_copy =
                std::cmp::min(std::cmp::min(chunk_words, MAX_POSITIONS_AVX2), space_left);

            let mut words_added = 0;
            for j in 0..words_to_copy {
                let start_pos = boundary_signature.word_start_positions[j] as u32;
                let end_pos = boundary_signature.word_end_positions[j] as u32;

                // Skip words that end before or at the last word we processed
                if end_pos > last_word_end {
                    if output_pos + words_added * 2 + 1 < word_boundaries.len() {
                        word_boundaries[output_pos + words_added * 2] = start_pos;
                        word_boundaries[output_pos + words_added * 2 + 1] = end_pos;
                        last_word_end = end_pos;
                        words_added += 1;
                    }
                }
            }

            total_words += words_added as u32;
            output_pos += words_added * 2;

            // If buffer was full and we haven't processed all bytes, reset and continue
            if buffer_full && i < (chunk_end - input_pos) {
                boundary_signature = WordBoundarySignature::<MAX_POSITIONS_AVX2>::new();
                char_classes = UnicodeCharacterClasses::new();
            } else {
                break;
            }
        }

        // Move to next chunk with minimal overlap to catch boundary words
        if chunk_end < len {
            // Skip whitespace and back up slightly to catch any word that might span the boundary
            while chunk_end < len && string_to_tokenize[chunk_end].is_ascii_whitespace() {
                chunk_end += 1;
            }
            input_pos = chunk_end.saturating_sub(10); // Small overlap
        } else {
            input_pos = chunk_end;
        }
    }

    total_words
}

// Helper function for single string NEON tokenization
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn tokenize_single_string_neon(
    string_to_tokenize: &mut [u8],
    len: usize,
    word_boundaries: &mut [u32],
) -> u32 {
    // Process input in chunks with deduplication
    const PROCESSING_CHUNK_SIZE: usize = 100;
    let mut input_pos = 0;
    let mut output_pos = 0;
    let mut total_words = 0u32;
    let mut last_word_end = 0u32;

    while input_pos < len && output_pos < word_boundaries.len() {
        // Determine chunk end - find next whitespace boundary
        let mut chunk_end = std::cmp::min(input_pos + PROCESSING_CHUNK_SIZE, len);
        if chunk_end < len {
            // Find whitespace to split at
            while chunk_end > input_pos && !string_to_tokenize[chunk_end].is_ascii_whitespace() {
                chunk_end -= 1;
            }
            if chunk_end == input_pos {
                chunk_end = std::cmp::min(input_pos + PROCESSING_CHUNK_SIZE, len);
            }
        }

        // Process this chunk with buffer flushes when hitting MAX_POSITIONS
        let mut boundary_signature = WordBoundarySignature::<MAX_POSITIONS_NEON>::new();
        let mut char_classes = UnicodeCharacterClasses::new();
        let mut i = 0;

        loop {
            // Process SIMD chunks until we hit MAX_POSITIONS or run out of data
            while i + 16 <= (chunk_end - input_pos)
                && boundary_signature.word_start_count < MAX_POSITIONS_NEON
            {
                let chunk = vld1q_u8(string_to_tokenize.as_ptr().add(input_pos + i));
                tokenize_chunk_neon(
                    chunk,
                    &mut boundary_signature,
                    &mut char_classes,
                    input_pos + i,
                    string_to_tokenize,
                );
                i += 16;
            }

            // If we haven't hit MAX_POSITIONS, process remaining bytes and finalize
            let buffer_full = boundary_signature.word_start_count >= MAX_POSITIONS_NEON;
            if !buffer_full {
                // Handle remaining bytes in this chunk
                if i < (chunk_end - input_pos) {
                    tokenize_scalar_remaining(
                        &mut string_to_tokenize[input_pos + i..chunk_end],
                        &mut boundary_signature,
                        input_pos + i,
                        chunk_end,
                    );
                }

                // Finalize if last chunk
                if chunk_end >= len {
                    boundary_signature.finalize_word(len);
                }
            }

            // Flush current buffer to output
            let chunk_words = std::cmp::min(
                boundary_signature.word_start_count,
                boundary_signature.word_end_count,
            );
            let space_left = (word_boundaries.len() - output_pos) / 2;
            let words_to_copy =
                std::cmp::min(std::cmp::min(chunk_words, MAX_POSITIONS_NEON), space_left);

            let mut words_added = 0;
            for j in 0..words_to_copy {
                let start_pos = boundary_signature.word_start_positions[j] as usize;
                let end_pos = boundary_signature.word_end_positions[j] as usize;

                // Skip words that end before or at the last word we processed
                if end_pos as u32 > last_word_end {
                    if output_pos + words_added * 2 + 1 < word_boundaries.len() {
                        word_boundaries[output_pos + words_added * 2] = start_pos as u32;
                        word_boundaries[output_pos + words_added * 2 + 1] = end_pos as u32;
                        last_word_end = end_pos as u32;
                        words_added += 1;
                    }
                }
            }

            total_words += words_added as u32;
            output_pos += words_added * 2;

            // If buffer was full and we haven't processed all bytes, reset and continue
            if buffer_full && i < (chunk_end - input_pos) {
                boundary_signature = WordBoundarySignature::<MAX_POSITIONS_NEON>::new();
                char_classes = UnicodeCharacterClasses::new();
            } else {
                break;
            }
        }

        // Move to next chunk with minimal overlap to catch boundary words
        if chunk_end < len {
            // Skip whitespace and back up slightly to catch any word that might span the boundary
            while chunk_end < len && string_to_tokenize[chunk_end].is_ascii_whitespace() {
                chunk_end += 1;
            }
            input_pos = chunk_end.saturating_sub(10); // Small overlap
        } else {
            input_pos = chunk_end;
        }
    }

    total_words
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn tokenize_chunk_avx2(
    chunk: std::arch::x86_64::__m256i,
    signature: &mut WordBoundarySignature<MAX_POSITIONS_AVX2>,
    char_classes: &mut UnicodeCharacterClasses,
    position: usize,
    full_input: &[u8],
) {
    //  DEEP SIMD: Create Unicode character class masks in parallel
    let letter_mask = create_unicode_letter_mask_avx2(chunk);
    let digit_mask = create_unicode_digit_mask_avx2(chunk);
    let punct_mask = create_unicode_punctuation_mask_avx2(chunk);
    let space_mask = create_unicode_whitespace_mask_avx2(chunk);

    // Update boundary signature with SIMD results
    signature.update_avx2(
        letter_mask,
        digit_mask,
        punct_mask,
        space_mask,
        position,
        full_input,
    );

    // Update character class counts
    char_classes.update_avx2(letter_mask, digit_mask, punct_mask, space_mask);

    // Tokenization only handles word boundary detection, not lowercase conversion
}

// Create Unicode digit detection mask using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_unicode_digit_mask_avx2(chunk: std::arch::x86_64::__m256i) -> i32 {
    let digit_mask = _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8((b'0' - 1) as i8)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8((b'9' + 1) as i8), chunk),
    );

    _mm256_movemask_epi8(digit_mask)
}

// Create Unicode letter detection mask (A-Z, a-z) using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_unicode_letter_mask_avx2(chunk: std::arch::x86_64::__m256i) -> i32 {
    // ASCII letters: A-Z, a-z
    let upper_mask = _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8((b'A' - 1) as i8)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8((b'Z' + 1) as i8), chunk),
    );
    let lower_mask = _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8((b'a' - 1) as i8)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8((b'z' + 1) as i8), chunk),
    );

    // Underscore mask - treat underscore as part of word characters
    let underscore_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'_' as i8));

    // Degree symbol first byte (0xC2) - treat as part of word characters for 25°C
    let degree_first_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(0xC2u8 as i8));

    // Degree symbol second byte (0xB0) - treat as part of word characters for 25°C
    let degree_second_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(0xB0u8 as i8));

    // UTF-8 continuation bytes (0x80-0xBF)
    let continuation_mask = _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8(0x7F as i8)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8(0xC0u8 as i8), chunk),
    );

    // UTF-8 2-byte start (0xC0-0xDF) - Latin Extended
    let utf8_2byte_mask = _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8(0xBFu8 as i8)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8(0xE0u8 as i8), chunk),
    );

    // UTF-8 3-byte start (0xE0-0xEF)
    let utf8_3byte_mask = _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, _mm256_set1_epi8(0xDFu8 as i8)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8(0xF0u8 as i8), chunk),
    );

    let combined_mask = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_or_si256(_mm256_or_si256(upper_mask, lower_mask), underscore_mask),
            _mm256_or_si256(degree_first_mask, degree_second_mask),
        ),
        _mm256_or_si256(
            _mm256_or_si256(continuation_mask, utf8_2byte_mask),
            utf8_3byte_mask,
        ),
    );

    _mm256_movemask_epi8(combined_mask)
}

// Create Unicode punctuation detection mask using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_unicode_punctuation_mask_avx2(chunk: std::arch::x86_64::__m256i) -> i32 {
    let punct_chars = [
        b'!', b'@', b'#', b'$', b'%', b'^', b'&', b'*', b'(', b')', b'-', b'_', b'=', b'+', b'[',
        b']', b'{', b'}', b'\\', b'|', b';', b':', b'"', b'\'', b'<', b'>', b',', b'.', b'?', b'/',
        b'`', b'~',
    ];

    let mut punct_mask = _mm256_setzero_si256();
    for &punct_char in &punct_chars {
        let char_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(punct_char as i8));
        punct_mask = _mm256_or_si256(punct_mask, char_mask);
    }

    _mm256_movemask_epi8(punct_mask)
}

// Create Unicode whitespace detection mask using AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn create_unicode_whitespace_mask_avx2(chunk: std::arch::x86_64::__m256i) -> i32 {
    let space_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8));
    let tab_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\t' as i8));
    let newline_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\n' as i8));
    let cr_mask = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\r' as i8));

    let combined_mask = _mm256_or_si256(
        _mm256_or_si256(space_mask, tab_mask),
        _mm256_or_si256(newline_mask, cr_mask),
    );

    _mm256_movemask_epi8(combined_mask)
}

// =============================================================================
// IMPLEMENTATION DETAILS - NEON TOKENIZATION - MIRRORING CLASSIFY STRUCTURE
// =============================================================================

// Create Unicode character class masks using NEON
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn create_unicode_masks_neon(chunk: std::arch::aarch64::uint8x16_t) -> (u64, u64, u64, u64) {
    // ASCII letters (A-Z, a-z)
    let upper_a = vdupq_n_u8(b'A');
    let upper_z = vdupq_n_u8(b'Z');
    let lower_a = vdupq_n_u8(b'a');
    let lower_z = vdupq_n_u8(b'z');

    let upper_mask = vandq_u8(vcgeq_u8(chunk, upper_a), vcleq_u8(chunk, upper_z));
    let lower_mask = vandq_u8(vcgeq_u8(chunk, lower_a), vcleq_u8(chunk, lower_z));

    // UTF-8 continuation bytes (0x80-0xBF)
    let continuation_start = vdupq_n_u8(0x80);
    let continuation_end = vdupq_n_u8(0xBF);
    let continuation_mask = vandq_u8(
        vcgeq_u8(chunk, continuation_start),
        vcleq_u8(chunk, continuation_end),
    );

    // UTF-8 2-byte start (0xC0-0xDF) - Latin Extended
    let utf8_2byte_start = vdupq_n_u8(0xC0);
    let utf8_2byte_end = vdupq_n_u8(0xDF);
    let utf8_2byte_mask = vandq_u8(
        vcgeq_u8(chunk, utf8_2byte_start),
        vcleq_u8(chunk, utf8_2byte_end),
    );

    // UTF-8 3-byte start (0xE0-0xEF)
    let utf8_3byte_start = vdupq_n_u8(0xE0);
    let utf8_3byte_end = vdupq_n_u8(0xEF);
    let utf8_3byte_mask = vandq_u8(
        vcgeq_u8(chunk, utf8_3byte_start),
        vcleq_u8(chunk, utf8_3byte_end),
    );

    // Underscore mask - treat underscore as part of word characters
    let underscore_mask = vceqq_u8(chunk, vdupq_n_u8(b'_'));

    // Degree symbol first byte (0xC2) - treat as part of word characters for 25°C
    let degree_first_mask = vceqq_u8(chunk, vdupq_n_u8(0xC2));

    // Degree symbol second byte (0xB0) - treat as part of word characters for 25°C
    let degree_second_mask = vceqq_u8(chunk, vdupq_n_u8(0xB0));

    let letter_mask = vorrq_u8(
        vorrq_u8(
            vorrq_u8(vorrq_u8(upper_mask, lower_mask), underscore_mask),
            vorrq_u8(degree_first_mask, degree_second_mask),
        ),
        vorrq_u8(
            vorrq_u8(continuation_mask, utf8_2byte_mask),
            utf8_3byte_mask,
        ),
    );

    // ASCII digits (0-9)
    let digit_0 = vdupq_n_u8(b'0');
    let digit_9 = vdupq_n_u8(b'9');
    let digit_mask = vandq_u8(vcgeq_u8(chunk, digit_0), vcleq_u8(chunk, digit_9));

    // Common ASCII whitespace
    let space_mask = vceqq_u8(chunk, vdupq_n_u8(b' '));
    let tab_mask = vceqq_u8(chunk, vdupq_n_u8(b'\t'));
    let newline_mask = vceqq_u8(chunk, vdupq_n_u8(b'\n'));
    let cr_mask = vceqq_u8(chunk, vdupq_n_u8(b'\r'));
    let whitespace_mask = vorrq_u8(
        vorrq_u8(space_mask, tab_mask),
        vorrq_u8(newline_mask, cr_mask),
    );

    // Common ASCII punctuation
    let punct_chars = [
        b'!', b'@', b'#', b'$', b'%', b'^', b'&', b'*', b'(', b')', b'-', b'_', b'=', b'+', b'[',
        b']', b'{', b'}', b'\\', b'|', b';', b':', b'"', b'\'', b'<', b'>', b',', b'.', b'?', b'/',
        b'`', b'~',
    ];

    let mut punct_mask = vdupq_n_u8(0);
    for &punct_char in &punct_chars {
        let char_mask = vceqq_u8(chunk, vdupq_n_u8(punct_char));
        punct_mask = vorrq_u8(punct_mask, char_mask);
    }

    // Convert masks to bit patterns for boundary detection
    let letter_bits = neon_mask_to_u64(letter_mask);
    let digit_bits = neon_mask_to_u64(digit_mask);
    let punct_bits = neon_mask_to_u64(punct_mask);
    let space_bits = neon_mask_to_u64(whitespace_mask);

    (letter_bits, digit_bits, punct_bits, space_bits)
}

// NEON chunk tokenization using comprehensive Unicode character detection
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn tokenize_chunk_neon(
    chunk: std::arch::aarch64::uint8x16_t,
    signature: &mut WordBoundarySignature<MAX_POSITIONS_NEON>,
    char_classes: &mut UnicodeCharacterClasses,
    position: usize,
    full_input: &[u8],
) {
    //  DEEP SIMD: Create Unicode character class masks in parallel
    let (letter_mask, digit_mask, punct_mask, space_mask) = create_unicode_masks_neon(chunk);

    // Update boundary signature with SIMD results
    signature.update_neon(
        letter_mask,
        digit_mask,
        punct_mask,
        space_mask,
        position,
        full_input,
    );

    // Update character class counts
    char_classes.update_neon(
        letter_mask as u32,
        digit_mask as u32,
        punct_mask as u32,
        space_mask as u32,
    );

    // Tokenization only handles word boundary detection, not lowercase conversion
}

// Convert NEON mask to u64 bit pattern
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_mask_to_u64(mask: std::arch::aarch64::uint8x16_t) -> u64 {
    let mask_bytes: [u8; 16] = std::mem::transmute(mask);
    let mut result = 0u64;
    for i in 0..16 {
        if mask_bytes[i] != 0 {
            result |= 1u64 << i;
        }
    }
    result
}

// =============================================================================
// SCALAR IMPLEMENTATIONS AND HELPER FUNCTIONS - MIRRORING CLASSIFY STRUCTURE
// =============================================================================

// Handle remaining bytes after SIMD processing
unsafe fn tokenize_scalar_remaining<const MAX_POS: usize>(
    remaining_bytes: &mut [u8],
    signature: &mut WordBoundarySignature<MAX_POS>,
    start_position: usize,
    _total_len: usize,
) {
    let mut i = 0;
    let remaining_len = remaining_bytes.len();

    while i < remaining_len {
        let byte = remaining_bytes[i];

        // Check for 0xC2 sequences that should split words (whitespace and certain punctuation)
        if byte == 0xC2 && i + 1 < remaining_len {
            let next_byte = remaining_bytes[i + 1];
            // NBSP (0xC2 0xA0) and middle dot (0xC2 0xB7) split words
            if next_byte == 0xA0 || next_byte == 0xB7 {
                if signature.prev_was_word_char {
                    signature.end_word(start_position + i);
                }
                signature.prev_was_word_char = false;
                i += 2; // Skip both bytes
                continue;
            }
        }

        let is_core_word_char = is_unicode_word_char(byte);

        // Apply contextual rules for MidNumLet (period, colon between letters/numbers)
        let mut is_word_char = is_core_word_char;

        // Handle UTF-8 sequences more intelligently
        if !is_core_word_char && byte >= 0x80 {
            // Check if this is part of a UTF-8 letter sequence
            // Special case for degree symbol (°) = [0xC2, 0xB0]
            if byte == 0xC2 && i + 1 < remaining_len && remaining_bytes[i + 1] == 0xB0 {
                is_word_char = true; // Degree symbol should be part of word (25°C)
            } else {
                // Use UTF-8-aware letter detection to better handle non-ASCII scripts
                is_word_char = is_utf8_letter_byte(remaining_bytes, i);
            }
        }

        // Check if this is a MidNumLet character between word characters
        if !is_word_char {
            if is_mid_num_let(byte) {
                // Look ahead and behind to see if we're between word characters
                let prev_is_word = if i > 0 {
                    is_unicode_word_char(remaining_bytes[i - 1])
                } else {
                    signature.prev_was_word_char
                };
                let next_is_word = if i + 1 < remaining_bytes.len() {
                    is_unicode_word_char(remaining_bytes[i + 1])
                } else {
                    false
                };

                // If between word characters, treat as part of word
                if prev_is_word && next_is_word {
                    is_word_char = true;
                }
            }
        }

        // Handle word boundary detection - only at valid UTF-8 character boundaries
        if is_word_char && !signature.prev_was_word_char {
            // Start word boundary - find the next valid UTF-8 character boundary
            let mut boundary_idx = i;
            while boundary_idx < remaining_bytes.len()
                && !signature.is_char_boundary(remaining_bytes, boundary_idx)
            {
                boundary_idx += 1;
            }
            if boundary_idx < remaining_bytes.len() {
                signature.start_word(start_position + boundary_idx);
            }
        } else if !is_word_char && signature.prev_was_word_char {
            // End word boundary - use current position if valid, otherwise find previous valid boundary
            let mut boundary_idx = i;
            while boundary_idx > 0 && !signature.is_char_boundary(remaining_bytes, boundary_idx) {
                boundary_idx -= 1;
            }
            signature.end_word(start_position + boundary_idx);
        }

        signature.prev_was_word_char = is_word_char;
        i += 1;
    }
}

/// Check if a byte is a core Unicode word character (ALetter, Numeric, ExtendNumLet)
/// according to UAX#29. Does NOT include contextual characters like MidLetter.
#[inline]
fn is_unicode_word_char(byte: u8) -> bool {
    // First check if it's whitespace - whitespace is never a word char
    if is_whitespace_byte(byte) {
        return false;
    }

    match byte {
        // ASCII letters (A-Z, a-z) - ALetter category
        b'A'..=b'Z' | b'a'..=b'z' => true,
        // ASCII digits (0-9) - Numeric category
        b'0'..=b'9' => true,
        // Underscore (connector punctuation Pc) - ExtendNumLet category
        b'_' => true,
        // UTF-8 multi-byte sequences (0x80-0xFF)
        // We need to distinguish between UTF-8 letters and UTF-8 punctuation
        0x80..=0xFF => {
            // Continuation bytes (0x80-0xBF) are part of multi-byte sequences
            // Treat them as word chars to keep sequences together, except 0xA0 (NBSP continuation)
            if byte >= 0x80 && byte <= 0xBF {
                return byte != 0xA0;
            }
            // Lead bytes: exclude known punctuation ranges
            // 0xE2: includes general punctuation (en dash, em dash, quotes) - not core word chars
            if byte == 0xE2 {
                return false;
            }
            // 0xC2 includes symbols (©, ^^, °) which should be tokens, so treat as word chars
            // NBSP (0xC2 0xA0) is handled separately in all paths
            // Other lead bytes (0xC3-0xDF, 0xE0-0xE1, 0xE3-0xEF, 0xF0-0xF4) - likely letters
            true
        }
        // All other ASCII punctuation are not core word chars
        _ => false,
    }
}

/// Check if a UTF-8 byte at a given position is part of a letter sequence
#[inline]
fn is_utf8_letter_byte(input: &[u8], pos: usize) -> bool {
    if pos >= input.len() {
        return false;
    }

    let byte = input[pos];

    // Handle UTF-8 sequences based on lead byte patterns
    if byte >= 0xC0 {
        // This is a UTF-8 lead byte - check the specific ranges
        match byte {
            // Latin-1 Supplement (0xC2-0xC3): includes accented letters
            0xC3 => {
                // 0xC3 sequences are mostly accented letters (^-ÿ)
                if pos + 1 < input.len() {
                    let second = input[pos + 1];
                    // Most 0xC3 sequences are letters, except some symbols
                    // This is a simplified heuristic
                    second >= 0x80 && second <= 0xBF
                } else {
                    false
                }
            }
            0xC2 => {
                // 0xC2 sequences include punctuation and symbols (including NBSP 0xC2 0xA0)
                // Treat as non-letters to avoid gluing through NBSP and similar symbols
                false
            }
            // 0xC4..0xDF: other 2-byte letters (Latin Extended, Greek, Cyrillic, etc.)
            0xC4..=0xDF => {
                // Ensure continuation exists
                pos + 1 < input.len()
            }
            // 0xE0..0xEF: three-byte sequences (covers most BMP scripts including CJK)
            0xE0..=0xEF => {
                if pos + 2 >= input.len() {
                    return false;
                }
                // Exclude U+3000 IDEOGRAPHIC SPACE (E3 80 80)
                if byte == 0xE3 && input[pos + 1] == 0x80 && input[pos + 2] == 0x80 {
                    return false;
                }
                // Exclude 0xE2 0x80 range (general punctuation: en dash, em dash, quotes, etc.)
                if byte == 0xE2 && input[pos + 1] == 0x80 {
                    return false;
                }
                true
            }
            // 0xF0..0xF4: four-byte sequences (supplementary plane letters)
            0xF0..=0xF4 => pos + 3 < input.len(),
            // Other UTF-8 lead bytes  default to non-letter
            _ => false,
        }
    } else if byte >= 0x80 && byte <= 0xBF {
        // This is a UTF-8 continuation byte
        // Check if it's part of a letter sequence by looking at the lead byte
        let mut lead_pos = pos;
        while lead_pos > 0 && input[lead_pos] >= 0x80 && input[lead_pos] < 0xC0 {
            lead_pos -= 1;
        }

        if lead_pos < pos && input[lead_pos] >= 0xC0 {
            // Found the lead byte, check if it indicates a letter
            is_utf8_letter_byte(input, lead_pos)
        } else {
            false
        }
    } else {
        // ASCII byte - not handled here
        false
    }
}

/// Check if a byte is MidNumLet (period, colon) - contextual between letters OR numbers
/// Note: Being more restrictive here - only periods for now
#[inline]
fn is_mid_num_let(byte: u8) -> bool {
    // Only periods should be treated as MidNumLet (for cases like hello.world, example.com)
    // Colons should create boundaries in most cases
    matches!(byte, b'.')
}

#[inline]
fn is_whitespace_byte(byte: u8) -> bool {
    match byte {
        // ASCII whitespace
        b' ' | b'\t' | b'\n' | b'\r' => true,
        // UTF-8 encoded whitespace - this is a simplified check
        // Non-breaking space (U+00A0) is encoded as [194, 160]
        // We'll check for common UTF-8 whitespace patterns
        160 => true, // Second byte of non-breaking space
        _ => false,
    }
}

// =============================================================================
// SIMD LOWERCASE CONVERSION FUNCTIONS
// =============================================================================

/// Convert bytes to lowercase using AVX-512 SIMD acceleration
// GPU implementation of lowercase conversion using PTX assembly

#[cfg(has_cuda)]
pub unsafe fn to_lowercase_gpu(bytes: *mut u8, len: usize) {
    const PTX_LOWERCASE: &str = r#"
    .version 7.5
    .target sm_70
    .address_size 64
    .entry to_lowercase (
      .param .u64 bytes,
      .param .u32 len
    ) {
      .reg .u32 %r<40>;
      .reg .u64 %rd<8>;
      .reg .u8  %b<2>;
      .reg .pred %p<20>;

      // Load parameters
      ld.param.u64 %rd0, [bytes];
      ld.param.u32 %r0, [len];

      // Grid-stride setup
      mov.u32 %r1, %tid.x;      // thread id
      mov.u32 %r2, %ntid.x;     // blockDim.x
      mov.u32 %r3, %ctaid.x;    // blockIdx.x
      mov.u32 %r4, %nctaid.x;   // gridDim.x
      mul.lo.u32 %r5, %r3, %r2; // block offset
      add.u32 %r6, %r1, %r5;    // i = tid + block*blockDim
      mul.lo.u32 %r7, %r2, %r4; // stride = blockDim * gridDim

    L_loop:
      // if i >= len: done
      setp.ge.u32 %p0, %r6, %r0;
      @%p0 bra L_done;

      // Load b = bytes[i]
      cvt.u64.u32 %rd1, %r6;
      add.u64 %rd1, %rd0, %rd1;
      ld.global.u8 %b0, [%rd1];
      cvt.u32.u8 %r8, %b0;

      // ASCII A-Z => lowercase
      setp.ge.u32 %p1, %r8, 65;   // 'A'
      setp.le.u32 %p2, %r8, 90;   // 'Z'
      and.pred %p3, %p1, %p2;
      @!%p3 bra L_check_cont;
      or.b32 %r9, %r8, 32;     // +0x20
      cvt.u8.u32 %b0, %r9;
      st.global.u8 [%rd1], %b0;
      add.u32 %r6, %r6, %r7;
      bra L_loop;

    L_check_cont:
      // Continuation byte 0x80..0xBF and i>0 and prev==0xC3
      setp.ge.u32 %p4, %r8, 128;   // >=0x80
      setp.le.u32 %p5, %r8, 191;   // <=0xBF
      and.pred %p6, %p4, %p5;      // continuation
      setp.ne.u32 %p7, %r6, 0;     // i != 0
      and.pred %p16, %p6, %p7;     // continuation && i>0
      @!%p16 bra L_step;

      // Load prev byte
      add.u32 %r10, %r6, 0xffffffff; // i-1
      cvt.u64.u32 %rd2, %r10;
      add.u64 %rd2, %rd0, %rd2;
      ld.global.u8 %b1, [%rd2];
      cvt.u32.u8 %r11, %b1;
      setp.eq.u32 %p8, %r11, 195; // prev == 0xC3
      @!%p8 bra L_step;

      // Latin-1 uppercase ranges on second byte: 0x80-0x96 or 0x98-0x9E
      setp.ge.u32 %p9,  %r8, 128; // 0x80
      setp.le.u32 %p10, %r8, 150; // 0x96
      and.pred %p11, %p9, %p10;
      setp.ge.u32 %p12, %r8, 152; // 0x98
      setp.le.u32 %p13, %r8, 158; // 0x9E
      and.pred %p14, %p12, %p13;
      or.pred  %p15, %p11, %p14;
      @!%p15 bra L_check_special;
      add.u32 %r12, %r8, 32;  // +0x20
      cvt.u8.u32 %b0, %r12;
      st.global.u8 [%rd1], %b0;
      add.u32 %r6, %r6, %r7;
      bra L_loop;

    L_check_special:
      // Special case: 0x8F -> 0xAF when prev==0xC3
      setp.eq.u32 %p16, %r8, 143;  // 0x8F
      @!%p16 bra L_step;
      mov.u32 %r13, 175;       // 0xAF
      cvt.u8.u32 %b0, %r13;
      st.global.u8 [%rd1], %b0;
      add.u32 %r6, %r6, %r7;
      bra L_loop;

    L_step:
      // No change; advance by stride
      add.u32 %r6, %r6, %r7;
      bra L_loop;

    L_done:
      ret;
    }
  "#;

    let _ = with_gpu_buffer_u8_mut(
        std::slice::from_raw_parts_mut(bytes, len),
        len,
        |gpu_bytes, gpu_len| {
            let (blocks, threads) = LaunchConfig::strings();
            let param_ptr: u64 = gpu_bytes as u64;
            let param_len: u32 = gpu_len as u32;
            let _ = launch_ptx(
                PTX_LOWERCASE,
                &[],
                "to_lowercase",
                blocks,
                threads,
                &[
                    &param_ptr as *const _ as *const u8,
                    &param_len as *const _ as *const u8,
                ],
            );
        },
    );
}

#[cfg(all(
    feature = "hwx-nightly",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn to_lowercase_avx512(bytes: &mut [u8], len: usize) {
    let mut i = 0;

    // Process 64-byte chunks with AVX-512
    while i + 64 <= len {
        let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i) as *const _);

        // Create masks for uppercase letters (A-Z)
        let upper_a = _mm512_set1_epi8(b'A' as i8);
        let upper_z = _mm512_set1_epi8(b'Z' as i8);
        let case_diff = _mm512_set1_epi8(32); // 'a' - 'A'

        // Check if bytes are in range A-Z
        let ge_a = _mm512_cmpge_epi8_mask(chunk, upper_a);
        let le_z = _mm512_cmple_epi8_mask(chunk, upper_z);
        let uppercase_mask = ge_a & le_z;

        // Add 32 to convert uppercase to lowercase
        let lowercase_chunk = _mm512_mask_add_epi8(chunk, uppercase_mask, chunk, case_diff);

        // Store result back
        _mm512_storeu_si512(bytes.as_mut_ptr().add(i) as *mut _, lowercase_chunk);
        i += 64;
    }

    // Handle remaining bytes with scalar processing
    while i < len {
        let byte = bytes[i];
        if byte >= b'A' && byte <= b'Z' {
            bytes[i] = byte + 32;
        }
        // UTF-8 2-byte sequences starting with 0xC2 (includes non-breaking space, etc.)
        else if byte == 0xC2 && i + 1 < len {
            // For 0xC2 sequences, preserve them as-is (no lowercase conversion)
            // This includes non-breaking space \u{A0} = [0xC2, 0xA0]
            i += 2;
            continue;
        }
        // UTF-8 2-byte sequences starting with 0xC3 (Latin-1 Supplement: ^-ÿ)
        else if byte == 0xC3 && i + 1 < len {
            let second_byte = bytes[i + 1];
            if second_byte >= 0x80 && second_byte <= 0x96 {
                bytes[i + 1] = second_byte + 0x20;
            } else if second_byte >= 0x98 && second_byte <= 0x9E {
                bytes[i + 1] = second_byte + 0x20;
            } else if second_byte == 0x8F {
                bytes[i + 1] = 0xAF;
            }
            i += 2;
            continue;
        }
        i += 1;
    }
}

/// Convert bytes to lowercase using AVX2 SIMD acceleration
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "hwx-nightly")
))]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn to_lowercase_avx2(bytes: &mut [u8], len: usize) {
    let mut i = 0;

    // Process 32-byte chunks with AVX2
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const _);

        // Create masks for uppercase letters (A-Z)
        let upper_a = _mm256_set1_epi8(b'A' as i8);
        let upper_z = _mm256_set1_epi8(b'Z' as i8);
        let case_diff = _mm256_set1_epi8(32); // 'a' - 'A'

        // Check if bytes are in range A-Z
        let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_add_epi8(upper_a, _mm256_set1_epi8(-1)));
        let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(upper_z, _mm256_set1_epi8(1)), chunk);
        let uppercase_mask = _mm256_and_si256(ge_a, le_z);

        // Add 32 to convert uppercase to lowercase where mask is true
        let case_adjustment = _mm256_and_si256(uppercase_mask, case_diff);
        let lowercase_chunk = _mm256_add_epi8(chunk, case_adjustment);

        // Store result back
        _mm256_storeu_si256(bytes.as_mut_ptr().add(i) as *mut _, lowercase_chunk);
        i += 32;
    }

    // Handle remaining bytes with scalar processing
    while i < len {
        let byte = bytes[i];
        if byte >= b'A' && byte <= b'Z' {
            bytes[i] = byte + 32;
        }
        // UTF-8 2-byte sequences starting with 0xC2 (includes non-breaking space, etc.)
        else if byte == 0xC2 && i + 1 < len {
            // For 0xC2 sequences, preserve them as-is (no lowercase conversion)
            // This includes non-breaking space \u{A0} = [0xC2, 0xA0]
            i += 2;
            continue;
        }
        // UTF-8 2-byte sequences starting with 0xC3 (Latin-1 Supplement: ^-ÿ)
        else if byte == 0xC3 && i + 1 < len {
            let second_byte = bytes[i + 1];
            if second_byte >= 0x80 && second_byte <= 0x96 {
                bytes[i + 1] = second_byte + 0x20;
            } else if second_byte >= 0x98 && second_byte <= 0x9E {
                bytes[i + 1] = second_byte + 0x20;
            } else if second_byte == 0x8F {
                bytes[i + 1] = 0xAF;
            }
            i += 2;
            continue;
        }
        i += 1;
    }
}

/// Convert bytes to lowercase using NEON SIMD acceleration
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn to_lowercase_neon(bytes: &mut [u8], len: usize) {
    let mut i = 0;

    // Process 16-byte chunks with NEON
    while i + 16 <= len {
        let chunk = vld1q_u8(bytes.as_ptr().add(i));

        // Create masks for uppercase letters (A-Z)
        let upper_a = vdupq_n_u8(b'A');
        let upper_z = vdupq_n_u8(b'Z');
        let case_diff = vdupq_n_u8(32); // 'a' - 'A'

        // Check if bytes are in range A-Z
        let ge_a = vcgeq_u8(chunk, upper_a);
        let le_z = vcleq_u8(chunk, upper_z);
        let uppercase_mask = vandq_u8(ge_a, le_z);

        // Add 32 to convert uppercase to lowercase where mask is true
        let case_adjustment = vandq_u8(uppercase_mask, case_diff);
        let lowercase_chunk = vaddq_u8(chunk, case_adjustment);

        // Store result back
        vst1q_u8(bytes.as_mut_ptr().add(i), lowercase_chunk);
        i += 16;
    }

    // Handle remaining bytes with scalar processing
    while i < len {
        let byte = bytes[i];
        if byte >= b'A' && byte <= b'Z' {
            bytes[i] = byte + 32;
        }
        // UTF-8 2-byte sequences starting with 0xC2 (includes non-breaking space, etc.)
        else if byte == 0xC2 && i + 1 < len {
            // For 0xC2 sequences, preserve them as-is (no lowercase conversion)
            // This includes non-breaking space \u{A0} = [0xC2, 0xA0]
            i += 2;
            continue;
        }
        // UTF-8 2-byte sequences starting with 0xC3 (Latin-1 Supplement: ^-ÿ)
        else if byte == 0xC3 && i + 1 < len {
            let second_byte = bytes[i + 1];
            if second_byte >= 0x80 && second_byte <= 0x96 {
                bytes[i + 1] = second_byte + 0x20;
            } else if second_byte >= 0x98 && second_byte <= 0x9E {
                bytes[i + 1] = second_byte + 0x20;
            } else if second_byte == 0x8F {
                bytes[i + 1] = 0xAF;
            }
            i += 2;
            continue;
        }
        i += 1;
    }
}

// GPU function for updating WordBoundarySignature - matches update_avx512 logic exactly
#[cfg(has_cuda)]
pub unsafe fn update_word_boundary_signature_gpu(
    signature_ptr: *mut u8,
    letter_mask: u64,
    digit_mask: u64,
    punct_mask: u64,
    space_mask: u64,
    position: u32,
    full_input_ptr: *const u8,
    full_input_len: usize,
) {
    const PTX_UPDATE_WORD_BOUNDARY_SIGNATURE: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .entry update_word_boundary_signature_ptx(
      .param .u64 signature_ptr,
      .param .u32 letter_mask,
      .param .u32 digit_mask,
      .param .u32 punct_mask,
      .param .u32 space_mask,
      .param .u32 position,
      .param .u64 full_input_ptr,
      .param .u32 full_input_len
  ) {
      .reg .pred %p<10>;
      .reg .u32 %r<20>;
      .reg .u64 %rd<10>;
      .reg .u8 %b<10>;

      // Load parameters
      ld.param.u64 %rd0, [signature_ptr];
      ld.param.u32 %r0, [letter_mask];
      ld.param.u32 %r1, [digit_mask];
      ld.param.u32 %r2, [punct_mask];
      ld.param.u32 %r3, [space_mask];
      ld.param.u32 %r4, [position];
      ld.param.u64 %rd1, [full_input_ptr];
      ld.param.u32 %r5, [full_input_len];

      // Update has_letters flag (offset 0)
      setp.ne.u32 %p0, %r0, 0;
      @!%p0 bra skip_letters;
      mov.u8 %b0, 1;
      st.global.u8 [%rd0], %b0;
  skip_letters:

      // Update has_digits flag (offset 1)
      setp.ne.u32 %p1, %r1, 0;
      @!%p1 bra skip_digits;
      mov.u8 %b1, 1;
      st.global.u8 [%rd0 + 1], %b1;
  skip_digits:

      // Update has_punctuation flag (offset 2)
      setp.ne.u32 %p2, %r2, 0;
      @!%p2 bra skip_punct;
      mov.u8 %b2, 1;
      st.global.u8 [%rd0 + 2], %b2;
  skip_punct:

      // Update has_whitespace flag (offset 3)
      setp.ne.u32 %p3, %r3, 0;
      @!%p3 bra skip_space;
      mov.u8 %b3, 1;
      st.global.u8 [%rd0 + 3], %b3;
  skip_space:

      // Now call process_word_boundaries logic inline
      // This should be the same as process_word_boundaries_gpu but inline here
      // For now, we'll need to call it separately since PTX can't call other PTX functions
      
      ret;
  }
  "#;

    let (blocks, threads) = (1, 32); // Single operation, single warp

    // First update the flags
    let _ = launch_ptx(
        PTX_UPDATE_WORD_BOUNDARY_SIGNATURE,
        &[],
        "update_word_boundary_signature_ptx",
        blocks,
        threads,
        &[
            signature_ptr as *const u8,
            &(letter_mask as u32) as *const u32 as *const u8,
            &(digit_mask as u32) as *const u32 as *const u8,
            &(punct_mask as u32) as *const u32 as *const u8,
            &(space_mask as u32) as *const u32 as *const u8,
            &position as *const u32 as *const u8,
            full_input_ptr as *const u8,
            &(full_input_len as u32) as *const u32 as *const u8,
        ],
    );

    // Then process word boundaries
    process_word_boundaries_gpu(
        signature_ptr,
        letter_mask,
        digit_mask,
        position,
        full_input_ptr,
        full_input_len,
    );
}

// GPU function for updating Unicode character class counts
#[cfg(has_cuda)]
pub unsafe fn update_unicode_character_classes_gpu(
    char_classes_ptr: *mut u8,
    letter_mask: u32,
    digit_mask: u32,
    punct_mask: u32,
    space_mask: u32,
) {
    const PTX_UNICODE_CHARACTER_CLASSES_UPDATE: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .entry unicode_character_classes_update_ptx(
      .param .u64 char_classes_ptr,
      .param .u32 letter_mask,
      .param .u32 digit_mask,
      .param .u32 punct_mask,
      .param .u32 space_mask
  ) {
      .reg .u32 %r<20>;
      .reg .u64 %rd<20>;

      // Load parameters
      ld.param.u64 %rd0, [char_classes_ptr];
      ld.param.u32 %r10, [letter_mask];
      ld.param.u32 %r11, [digit_mask];
      ld.param.u32 %r12, [punct_mask];
      ld.param.u32 %r13, [space_mask];

      // Count ones in each mask using population count (32-bit)
      popc.b32 %r14, %r10;  // letter count
      popc.b32 %r15, %r11;  // digit count
      popc.b32 %r16, %r12;  // punct count
      popc.b32 %r17, %r13;  // space count

      // Load current counts, add new counts, and store back

      // Update letter_count
      ld.global.u32 %r0, [%rd0];       // Load letter_count
      add.u32 %r0, %r0, %r14;          // Add new count
      st.global.u32 [%rd0], %r0;       // Store back

      // Update digit_count
      ld.global.u32 %r2, [%rd0 + 4];   // Load digit_count
      add.u32 %r2, %r2, %r15;
      st.global.u32 [%rd0 + 4], %r2;

      // Update punctuation_count
      ld.global.u32 %r4, [%rd0 + 8];   // Load punctuation_count
      add.u32 %r4, %r4, %r16;
      st.global.u32 [%rd0 + 8], %r4;

      // Update whitespace_count
      ld.global.u32 %r6, [%rd0 + 12];  // Load whitespace_count
      add.u32 %r6, %r6, %r17;
      st.global.u32 [%rd0 + 12], %r6;

      // Update total_count by 32 (PTX processes 32 bytes)
      ld.global.u32 %r8, [%rd0 + 16];  // Load total_count
      add.u32 %r8, %r8, 32;
      st.global.u32 [%rd0 + 16], %r8;

      ret;
  }
  "#;

    let (blocks, threads) = (1, 32); // Single operation, single warp

    let _ = launch_ptx(
        PTX_UNICODE_CHARACTER_CLASSES_UPDATE,
        &[],
        "unicode_character_classes_update_ptx",
        blocks,
        threads,
        &[
            char_classes_ptr as *const u8,
            &letter_mask as *const u32 as *const u8,
            &digit_mask as *const u32 as *const u8,
            &punct_mask as *const u32 as *const u8,
            &space_mask as *const u32 as *const u8,
        ],
    );
}

// GPU function for processing word boundaries - matches AVX-512 logic exactly
#[cfg(has_cuda)]
pub unsafe fn process_word_boundaries_gpu(
    signature_ptr: *mut u8,
    letter_mask: u64,
    digit_mask: u64,
    position: u32,
    full_input_ptr: *const u8,
    full_input_len: usize,
) {
    const PTX_PROCESS_WORD_BOUNDARIES: &str = r#"
  .version 7.5
  .target sm_70
  .address_size 64

  .entry process_word_boundaries_ptx(
      .param .u64 signature_ptr,
      .param .u32 letter_mask,
      .param .u32 digit_mask,
      .param .u32 position,
      .param .u64 full_input_ptr,
      .param .u32 full_input_len
  ) {
      .reg .pred %p<100>;
      .reg .u32 %r<200>;
      .reg .u64 %rd<100>;
      .reg .u8 %b<50>;

      // Load parameters
      ld.param.u64 %rd0, [signature_ptr];
      ld.param.u32 %r10, [letter_mask];
      ld.param.u32 %r11, [digit_mask];
      ld.param.u32 %r0, [position];
      ld.param.u64 %rd3, [full_input_ptr];
      ld.param.u32 %r1, [full_input_len];

      // Combine letter and digit masks for word character mask (32-bit)
      or.b32 %r12, %r10, %r11;  // word_char_mask = letter_mask | digit_mask

      // Load word_boundary_tracker pointer from signature (offset 8)
      add.u64 %rd5, %rd0, 8;
      ld.global.u64 %rd6, [%rd5];  // word_boundary_tracker pointer

      // Load tracker state
      ld.global.u8 %b10, [%rd6];  // prev_was_word_char
      ld.global.u32 %r20, [%rd6 + 4];  // word_start_count
      ld.global.u32 %r21, [%rd6 + 8];  // word_end_count

      // Process each byte position in the chunk (32 bytes for GPU)
      mov.u32 %r2, 0;  // lane = 0

  process_lane_loop:
      setp.ge.u32 %p0, %r2, 32;  // lane >= 32
      @%p0 bra done;

      // Calculate current position
      add.u32 %r3, %r0, %r2;  // current_pos = position + lane

      // Check if beyond input length
      setp.ge.u32 %p1, %r3, %r1;  // current_pos >= full_input_len
      @%p1 bra next_lane;

      // Load byte at current position
      cvt.u64.u32 %rd7, %r3;
      add.u64 %rd8, %rd3, %rd7;
      ld.global.u8 %b0, [%rd8];  // byte = full_input[current_pos]

      // Check if this is a core word character (letter or digit)
      mov.u32 %r9, 1;
      shl.b32 %r9, %r9, %r2;  // Create mask for current lane (32-bit)
      and.b32 %r13, %r12, %r9;  // Check word_char_mask at this position
      setp.ne.u32 %p2, %r13, 0;
      mov.u32 %r4, 0;  // is_word_char = false
      @%p2 mov.u32 %r4, 1;  // is_word_char = true if core word char

      // Check if UTF-8 continuation byte (>= 0x80) and not core word char
      setp.eq.u32 %p3, %r4, 0;  // !is_core_word_char
      setp.ge.u8 %p4, %b0, 128;  // byte >= 0x80
      and.pred %p5, %p3, %p4;
      @%p5 bra check_utf8_letter;
      bra check_mid_num_let;

  check_utf8_letter:
      // Check if this is part of a UTF-8 letter sequence
      // For GPU, we accept all UTF-8 continuation bytes as potential letters
      mov.u32 %r4, 1;  // is_word_char = true for UTF-8
      bra check_mid_num_let;

  check_mid_num_let:
      // Apply MidNumLet contextual rules for periods and colons
      setp.eq.u32 %p6, %r4, 0;  // !is_word_char
      @!%p6 bra update_boundary_state;

      // Check if period (46) or colon (58)
      setp.eq.u8 %p7, %b0, 46;  // period
      setp.eq.u8 %p8, %b0, 58;  // colon
      or.pred %p9, %p7, %p8;
      @!%p9 bra update_boundary_state;

      // Check previous character
      mov.u32 %r5, 0;  // prev_is_word = false
      setp.gt.u32 %p10, %r3, 0;  // current_pos > 0
      @!%p10 bra check_next;

      sub.u32 %r6, %r3, 1;  // prev_pos
      cvt.u64.u32 %rd11, %r6;
      add.u64 %rd12, %rd3, %rd11;
      ld.global.u8 %b1, [%rd12];  // prev_byte

      // Check if prev_byte is alphanumeric or UTF-8
      setp.ge.u8 %p11, %b1, 48;
      setp.le.u8 %p12, %b1, 57;
      and.pred %p13, %p11, %p12;  // is digit
      @%p13 mov.u32 %r5, 1;

      setp.ge.u8 %p14, %b1, 65;
      setp.le.u8 %p15, %b1, 90;
      and.pred %p16, %p14, %p15;  // is uppercase
      @%p16 mov.u32 %r5, 1;

      setp.ge.u8 %p17, %b1, 97;
      setp.le.u8 %p18, %b1, 122;
      and.pred %p19, %p17, %p18;  // is lowercase
      @%p19 mov.u32 %r5, 1;

      setp.ge.u8 %p20, %b1, 128;  // UTF-8
      @%p20 mov.u32 %r5, 1;

  check_next:
      // Check next character
      mov.u32 %r7, 0;  // next_is_word = false
      add.u32 %r8, %r3, 1;  // next_pos
      setp.ge.u32 %p21, %r8, %r1;  // next_pos >= full_input_len
      @%p21 bra check_context;

      cvt.u64.u32 %rd13, %r8;
      add.u64 %rd14, %rd3, %rd13;
      ld.global.u8 %b2, [%rd14];  // next_byte

      // Check if next_byte is alphanumeric or UTF-8
      setp.ge.u8 %p22, %b2, 48;
      setp.le.u8 %p23, %b2, 57;
      and.pred %p24, %p22, %p23;  // is digit
      @%p24 mov.u32 %r7, 1;

      setp.ge.u8 %p25, %b2, 65;
      setp.le.u8 %p26, %b2, 90;
      and.pred %p27, %p25, %p26;  // is uppercase
      @%p27 mov.u32 %r7, 1;

      setp.ge.u8 %p28, %b2, 97;
      setp.le.u8 %p29, %b2, 122;
      and.pred %p30, %p28, %p29;  // is lowercase
      @%p30 mov.u32 %r7, 1;

      setp.ge.u8 %p31, %b2, 128;  // UTF-8
      @%p31 mov.u32 %r7, 1;

  check_context:
      // If between word characters, mark as word char
      setp.eq.u32 %p33, %r5, 1;  // prev_is_word
      setp.eq.u32 %p34, %r7, 1;  // next_is_word
      and.pred %p35, %p33, %p34;
      @%p35 mov.u32 %r4, 1;  // is_word_char = true if between words

  update_boundary_state:
      // Check for word boundary transitions
      setp.ne.u8 %p37, %b10, 0;  // prev_was_word_char
      setp.eq.u32 %p38, %r4, 1;  // is_word_char
      
      // Starting new word: !prev_was_word_char && is_word_char
      setp.eq.u8 %p39, %b10, 0;  // !prev_was_word_char
      and.pred %p40, %p39, %p38;  // starting word
      @!%p40 bra check_word_end;

      // Store word start position
      setp.lt.u32 %p41, %r20, 256;  // word_start_count < MAX_WORDS
      @!%p41 bra update_counts;
      
      // Calculate offset for word_starts array (offset 12 + count * 4)
      mul.lo.u32 %r22, %r20, 4;
      add.u32 %r22, %r22, 12;
      cvt.u64.u32 %rd15, %r22;
      add.u64 %rd16, %rd6, %rd15;
      st.global.u32 [%rd16], %r3;  // word_starts[count] = current_pos
      
      add.u32 %r20, %r20, 1;  // word_start_count++
      bra update_counts;

  check_word_end:
      // Ending word: prev_was_word_char && !is_word_char
      setp.eq.u32 %p42, %r4, 0;  // !is_word_char
      and.pred %p43, %p37, %p42;  // ending word
      @!%p43 bra update_counts;

      // Store word end position
      setp.lt.u32 %p44, %r21, 256;  // word_end_count < MAX_WORDS
      @!%p44 bra update_counts;
      
      // Calculate offset for word_ends array (offset 1036 + count * 4)
      mul.lo.u32 %r23, %r21, 4;
      add.u32 %r23, %r23, 1036;  // 12 + 256*4 = 1036
      cvt.u64.u32 %rd17, %r23;
      add.u64 %rd18, %rd6, %rd17;
      st.global.u32 [%rd18], %r3;  // word_ends[count] = current_pos
      
      add.u32 %r21, %r21, 1;  // word_end_count++

  update_counts:
      // Update prev_was_word_char for this lane
      setp.eq.u32 %p45, %r4, 1;
      mov.u8 %b10, 0;
      @%p45 mov.u8 %b10, 1;

  next_lane:
      // Next lane
      add.u32 %r2, %r2, 1;
      bra process_lane_loop;

  done:
      // Store updated tracker state
      st.global.u8 [%rd6], %b10;  // prev_was_word_char
      st.global.u32 [%rd6 + 4], %r20;  // word_start_count
      st.global.u32 [%rd6 + 8], %r21;  // word_end_count
      
      ret;
  }
  "#;

    let (blocks, threads) = (1, 32); // Single operation, single warp

    let _ = launch_ptx(
        PTX_PROCESS_WORD_BOUNDARIES,
        &[],
        "process_word_boundaries_ptx",
        blocks,
        threads,
        &[
            signature_ptr as *const u8,
            &(letter_mask as u32) as *const u32 as *const u8,
            &(digit_mask as u32) as *const u32 as *const u8,
            &position as *const u32 as *const u8,
            full_input_ptr as *const u8,
            &(full_input_len as u32) as *const u32 as *const u8,
        ],
    );
}
