# CUDA / PTX notes

HWX enables CUDA support when `nvcc` is detected by `build.rs`. GPU kernels are shipped as **PTX** and loaded through the CUDA driver at runtime.

## Why PTX

PTX is a pragmatic way to ship custom kernels when they don't map cleanly to existing CUDA libraries. It keeps the crate self-contained for those kernels.

Tradeoffs:

- **First-use latency**: the driver may JIT compile PTX the first time a module is loaded.
- **Performance predictability**: PTX JIT can be less predictable than shipping precompiled cubins/fatbins for specific SM targets.

If you need to avoid JIT overhead for hot kernels, a common alternative is to ship precompiled cubins/fatbins for a fixed set of SM architectures.

## Module caching

HWX caches loaded modules to avoid recompiling on every call.

## Supported targets and toolchain

CUDA support primarily targets Linux systems with a working CUDA toolkit installation (driver + toolkit). HWX also uses `nvcc` for a small CUDA/CUB wrapper where needed.
