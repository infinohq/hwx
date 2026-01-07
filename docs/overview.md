# Overview

HWX is a Rust crate of database/search-style data-processing primitives with scalar fallbacks and optional acceleration.

APIs are intentionally small and "kernel-like": filtering, set operations, traversal/search on sorted arrays, string/token primitives, and a collection of numeric helpers.

## What HWX is (and is not)

- **HWX is**: a collection of reusable kernels with a thin dispatch layer.
- **HWX is not**: a query engine, a dataframe runtime, or a general GPU compute framework.

## Dispatch model

Most public functions are exposed from `hwx::dispatch` (re-exported at the crate root). Internally, functions may choose between:

- scalar implementations
- CPU SIMD implementations (AVX2 / NEON on stable; AVX-512 behind `hwx-nightly`)
- GPU implementations via CUDA when available (PTX kernels; see `docs/cuda.md`)

The dispatch decisions are typically based on the target architecture, runtime capability detection, and input size thresholds.

## Documentation

- Full list of primitives: see [`docs/primitives.md`](primitives.md)
- CUDA/PTX notes: see [`docs/cuda.md`](cuda.md)
