# HWX Framework

A Rust library of data-processing primitives with scalar fallbacks and optional hardware acceleration.

## Overview

HWX includes a small dispatch layer that selects an implementation based on target capabilities and input sizes.

- **Dispatch**: scalar / CPU SIMD / CUDA (when available)
- **Targets**: x86_64 (AVX2; AVX-512 behind a nightly feature), aarch64 (NEON), and NVIDIA GPUs via CUDA (when available)

## Status

This project is early and evolving. Public APIs may change while the crate is being
stabilized and documented.

## Where it fits (and where it doesn’t)

HWX was built for database/search-engine style workloads (filtering, set operations, string/token processing, and traversal/search over in-memory data), but the primitives can be used anywhere you need predictable, allocation-light kernels with optional acceleration.

It is **not** a general GPU compute framework or a query engine. For large dense linear algebra, you’ll usually want purpose-built libraries (and different data layouts) rather than HWX’s kernels.

## Getting started

Add `hwx` to your `Cargo.toml`:

```toml
[dependencies]
hwx = { path = "../hwx" }
```

If you're using a workspace, you can also use a workspace dependency:

```toml
[workspace.dependencies]
hwx = { path = "crates/hwx" }
```

## Key features

HWX currently focuses on:

- **Arrays**: set operations, filtering, and dedup helpers
- **Distances**: vector distance/similarity functions
- **Strings**: pattern matching utilities
- **Tokenization**: Unicode-aware tokenization utilities
- **Classification**: classify strings into `HwxType`
- **Traverse/filter**: simple traversal-style helpers (including time-range filtering over `MetricPoint`)

## Documentation

- **Overview**: `docs/overview.md`
- **CUDA/PTX notes**: `docs/cuda.md`
- **Full primitives reference (all public functions)**: `docs/primitives.md`

## Architecture

HWX provides a single API surface; internally, many functions dispatch to the best
available implementation.

## Crate layout

```
hwx/
├── Cargo.toml
├── README.md
├── LICENSE
├── build.rs            # Optional CUDA build integration (auto-detected)
└── src/
    ├── lib.rs          # Public API exports
    ├── dispatch.rs     # Main HWX dispatch layer
    ├── constants.rs    # Lane counts and thresholds
    ├── gpu.rs          # CUDA helpers (compiled when CUDA is available)
    ├── arrays.rs
    ├── classify.rs
    ├── distance.rs
    ├── strings.rs
    ├── tokenize.rs
    ├── traverse.rs
    └── types.rs
```

## Usage examples

### Basic Array Operations

```rust
use hwx::{intersect_sorted_u32, set_difference_sorted_u32, union_sorted_u32, dedup_sorted_u32};

// Intersect two sorted arrays (dispatch selects an implementation)
let mut a = vec![1, 3, 5, 7, 9];
let b = vec![2, 3, 6, 7, 10];
intersect_sorted_u32(&mut a, &b, 100, false, true).unwrap();  // ascending=true, dedup=false
assert_eq!(a, vec![3, 7]);

// Remove duplicates from sorted array in-place
let mut data = vec![1, 1, 2, 2, 3, 4, 4, 5];
dedup_sorted_u32(&mut data).unwrap();
assert_eq!(data, vec![1, 2, 3, 4, 5]);
```

### Distance Calculations

```rust
use hwx::{distance_l2_f32, distance_cosine_f32};

// Vector similarity (dispatch selects an implementation)
let vector_a = [1.0f32, 2.0, 3.0, 4.0];
let vector_b = [2.0f32, 3.0, 4.0, 5.0];
let l2_distance = distance_l2_f32(&vector_a, &vector_b).unwrap();
let cosine_dist = distance_cosine_f32(&vector_a, &vector_b).unwrap();
```

## Error handling

The library uses a custom `HwxError` type for all operations.

```rust
use hwx::types::HwxError;
```

## Configuration

HWX uses Cargo features and automatic detection to enable hardware-specific optimizations.

### Feature flags

- **`hwx-nightly`**: Enables AVX-512 implementations (requires a nightly Rust toolchain).
- **`disable-hwx`**: Forces scalar implementations (useful for debugging and correctness checks).

### Enabling AVX-512 (`hwx-nightly`)
To use AVX-512 optimizations, you must use a **nightly Rust compiler** and enable the `hwx-nightly` feature.

**CLI:**
```bash
cargo build --features hwx-nightly
```

**Cargo.toml:**
```toml
[dependencies]
hwx = { version = "0.1", features = ["hwx-nightly"] }
```

### CUDA support (auto-detected)
CUDA support is enabled at build time if `nvcc` is found (either in `PATH`, or via `CUDA_HOME` / `CUDA_PATH`).

- **Automatic**: If `nvcc` is present, CUDA kernels are compiled and used where available.
- **Explicit Path**: If `nvcc` is in a non-standard location, set `CUDA_HOME` (or `CUDA_PATH`):
  ```bash
  export CUDA_HOME=/usr/local/cuda
  cargo build
  ```

Notes:
- Some GPU kernels are shipped as PTX and loaded via the CUDA driver at runtime. This keeps the crate self-contained for custom kernels (especially for operations that don't map cleanly to existing CUDA libraries), but it can introduce a first-use JIT cost and may be less predictable than shipping precompiled cubins for specific SM targets.
- If you need to avoid JIT overhead for hot kernels, a common alternative is to ship precompiled cubins/fatbins for a fixed set of SM architectures.
- HWX attempts to detect the GPU compute capability via `nvidia-smi`. If that fails, it falls back to `sm_70`.
- The build targets an NVIDIA “SM” architecture (compute capability) such as:
  - `sm_70` (Volta)
  - `sm_75` (Turing)
  - `sm_80` / `sm_86` (Ampere)
  - `sm_89` (Ada)
  - `sm_90` (Hopper)
- CUDA is expected to work primarily on Linux systems with a working CUDA toolkit installation.

## Safety notes

- Some optimized implementations use `unsafe` and `#[target_feature]` internally.
- Prefer calling the public dispatch functions (they handle feature detection and fallbacks) rather than calling architecture-specific kernels directly.

## Disabling hardware acceleration

To force scalar implementations (useful for debugging), use the `disable-hwx` feature:
```bash
cargo build --features disable-hwx
```

## Testing

Run tests with:

```bash
cargo test
```

## Contributing

Issues and pull requests are welcome. If you're making a change:

- Keep changes focused and add/adjust tests when behavior changes.
- Run `cargo fmt` and `cargo test` before opening a PR.

## License

Apache-2.0
