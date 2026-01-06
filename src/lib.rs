// SPDX-License-Identifier: Apache-2.0

//! HWX library
//!
//! A collection of utilities with hardware-accelerated implementations where it makes sense.
//! Many operations have scalar fallbacks and optional SIMD/CUDA backends.
//!
//! - Array utilities (set operations, dedup, filtering)
//! - Distance/similarity functions
//! - String operations and tokenization
//! - Simple traversal/filtering helpers
//!
//! ## Hardware support
//! - **AVX2 / NEON** are used on stable Rust where available
//! - **AVX-512** is available behind the `hwx-nightly` feature (nightly Rust)
//! - **CUDA** is enabled when detected by `build.rs` (requires `nvcc`)
//!
//! ## Usage
//!
//! ```rust
//! use hwx;
//!
//! // Distance computation (automatically selects best SIMD implementation)
//! let va = [1.0, 2.0, 3.0, 4.0];
//! let vb = [2.0, 3.0, 4.0, 5.0];
//! let distance = hwx::distance_l2_f32(&va, &vb);
//!
//! // Search operations
//! let mut a = vec![1, 3, 5, 7];
//! let b = [2, 3, 6, 7];  
//! hwx::intersect_sorted_u32(&mut a, &b, 100, false, true).unwrap();
//!
//! // Check available SIMD capabilities
//! let caps = hwx::get_hw_capabilities();
//! println!("Has AVX-512: {}", caps.has_avx512);
//! ```

#![allow(clippy::missing_safety_doc)]

pub mod arrays;
pub mod classify;
pub mod constants;
pub mod dispatch;
pub mod distance;
#[cfg(has_cuda)]
pub mod gpu;
pub mod strings;
pub mod tokenize;
pub mod traverse;
pub mod types;

pub use types::*;

#[cfg(test)]
pub mod test_utils;

#[cfg(test)]
#[path = "tests/arrays_tests.rs"]
mod arrays_tests;
#[cfg(test)]
#[path = "tests/classify_tests.rs"]
mod classify_tests;
#[cfg(test)]
#[path = "tests/distance_tests.rs"]
mod distance_tests;
#[cfg(test)]
#[path = "tests/strings_tests.rs"]
mod strings_tests;
#[cfg(test)]
#[path = "tests/tokenize_tests.rs"]
mod tokenize_tests;
#[cfg(test)]
#[path = "tests/traverse_tests.rs"]
mod traverse_tests;

// Re-export the main API from core
#[cfg(has_cuda)]
pub use classify::{
    classify_array_contents_gpu, classify_chunk_gpu, classify_single_string_gpu,
    create_digit_mask_gpu, create_letter_mask_gpu, create_punctuation_mask_gpu,
    create_space_mask_gpu, PTX_CLASSIFY_ARRAY_CORE_FUNC, PTX_CLASSIFY_ARRAY_INLINE_FUNC,
    PTX_CLASSIFY_SINGLE_CORE_FUNC, PTX_CLASSIFY_SINGLE_INLINE_FUNC, PTX_CREATE_DIGIT_MASK,
    PTX_CREATE_LETTER_MASK, PTX_CREATE_PUNCT_MASK, PTX_CREATE_SPACE_MASK,
};
pub use dispatch::*;
#[cfg(has_cuda)]
pub use gpu::{
    with_gpu_buffer_2f64_scalar, with_gpu_buffer_f64_minmax, with_gpu_buffer_f64_stdvar,
    with_gpu_buffer_f64_u64, with_gpu_buffer_u32_minmax, with_gpu_buffer_u32_mut_f64,
    with_gpu_buffer_u32_mut_i64, with_gpu_buffer_u32_mut_u64, with_gpu_buffer_u32_to_u64,
};
#[cfg(has_cuda)]
pub use tokenize::{
    create_unicode_letter_mask_gpu, process_word_boundaries_gpu, to_lowercase_gpu,
    tokenize_chunk_gpu, tokenize_single_string_gpu, update_unicode_character_classes_gpu,
    update_word_boundary_signature_gpu, PTX_CREATE_UNICODE_DIGIT_MASK,
    PTX_CREATE_UNICODE_LETTER_MASK, PTX_CREATE_UNICODE_PUNCT_MASK, PTX_CREATE_UNICODE_SPACE_MASK,
};
