// SPDX-License-Identifier: Apache-2.0

//! CUDA support for HWX
//!
//! This module contains the CUDA-facing pieces used by HWX when `has_cuda` is enabled
//! (detected by `build.rs` when `nvcc` is available).
//!
//! It provides:
//! - Helpers for allocating/copying buffers via the CUDA runtime
//! - PTX/kernel launch helpers used by accelerated implementations
//!
//! ## Example (illustrative)
//! ```rust
//! use hwx::types::HwxError;
//!
//! fn main() -> Result<(), HwxError> {
//!     // Define PTX kernel as a string (example only).
//!     const MY_KERNEL: &str = r#"
//!       .version 7.0
//!       .target sm_70
//!       .entry my_kernel(...) { ... }
//!     "#;
//!
//!     // GPU APIs are only compiled when CUDA is detected at build time (`cfg(has_cuda)`).
//!     // Guard CUDA-only calls so the example compiles on all platforms.
//!     #[cfg(has_cuda)]
//!     {
//!         let _ = MY_KERNEL;
//!         // hwx::gpu::launch_ptx(MY_KERNEL, &[], "my_kernel", blocks, threads, &args)?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! All GPU functions in the HWX framework use this pattern for consistent
//! GPU acceleration across different operations (sorting, searching, distance
//! calculations, string processing, etc.)
use crate::types::HwxError;
use log::debug;
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::ptr;
use std::sync::Mutex;

// CUDA runtime API declarations
#[cfg(has_cuda)]
unsafe extern "C" {
    pub(crate) fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    pub(crate) fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        size: usize,
        kind: i32,
    ) -> i32;
    pub(crate) fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaMemset(ptr: *mut std::ffi::c_void, value: i32, size: usize) -> i32;
    pub(crate) fn cudaDeviceSynchronize() -> i32;

    // CUB DeviceRadixSort wrappers
    pub(crate) fn cub_device_radix_sort_u32(d_values: *mut u32, len: usize, ascending: bool)
        -> i32;

    pub(crate) fn cub_device_radix_sort_f64(d_values: *mut f64, len: usize, ascending: bool)
        -> i32;

    pub(crate) fn cub_device_radix_sort_u64(d_values: *mut u64, len: usize, ascending: bool)
        -> i32;

    pub(crate) fn cub_device_radix_sort_i64(d_values: *mut i64, len: usize, ascending: bool)
        -> i32;

    // CUB SortPairs for sorting indices by values
    pub(crate) fn cub_sort_pairs_u32_by_u64(
        d_indices: *mut u32,
        d_values: *mut u64,
        len: usize,
        ascending: bool,
    ) -> i32;

    pub(crate) fn cub_sort_pairs_u32_by_f64(
        d_indices: *mut u32,
        d_values: *mut f64,
        len: usize,
        ascending: bool,
    ) -> i32;

    pub(crate) fn cub_sort_pairs_u32_by_i64(
        d_indices: *mut u32,
        d_values: *mut i64,
        len: usize,
        ascending: bool,
    ) -> i32;

    // CUB DeviceSelect::Unique for deduplication
    pub(crate) fn cub_device_unique_u32(
        d_values: *mut u32,
        d_temp_out: *mut u32,
        len: usize,
        d_num_selected: *mut usize,
    ) -> i32;

    pub(crate) fn cub_device_unique_f64(
        d_values: *mut f64,
        d_temp_out: *mut f64,
        len: usize,
        d_num_selected: *mut usize,
    ) -> i32;

    pub(crate) fn cub_device_unique_u64(
        d_values: *mut u64,
        d_temp_out: *mut u64,
        len: usize,
        d_num_selected: *mut usize,
    ) -> i32;

    pub(crate) fn cub_device_unique_i64(
        d_values: *mut i64,
        d_temp_out: *mut i64,
        len: usize,
        d_num_selected: *mut usize,
    ) -> i32;

    // CUB variance calculation
    pub(crate) fn cub_device_variance_f64(
        d_values: *mut f64,
        len: usize,
        d_result: *mut f64,
    ) -> i32;
}

// CUDA driver API declarations for raw FFI
#[repr(C)]
struct CUmod_st {
    _opaque: u8,
}
type CUmodule = *mut CUmod_st;

#[repr(C)]
struct CUfunc_st {
    _opaque: u8,
}
type CUfunction = *mut CUfunc_st;

#[repr(C)]
struct CUctx_st {
    _opaque: u8,
}
type CUcontext = *mut CUctx_st;

#[repr(C)]
struct CUstream_st {
    _opaque: u8,
}
type CUstream = *mut CUstream_st;

// Wrapper to make CUDA pointers Send - we know CUDA is thread-safe
struct SendModule(CUmodule);
unsafe impl Send for SendModule {}
unsafe impl Sync for SendModule {}

struct SendContext(CUcontext);
unsafe impl Send for SendContext {}
unsafe impl Sync for SendContext {}

#[allow(non_camel_case_types)]
type CUresult = i32;

// CUDA driver API functions
// Linking support for device functions
#[repr(C)]
struct CUlinkState_st {
    _opaque: u8,
}
type CUlinkState = *mut CUlinkState_st;

// JIT option type and constants used
#[allow(non_camel_case_types)]
type CUjit_option = i32;

const CU_JIT_INFO_LOG_BUFFER: CUjit_option = 3;
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: CUjit_option = 4;
const CU_JIT_ERROR_LOG_BUFFER: CUjit_option = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: CUjit_option = 6;
const CU_JIT_LOG_VERBOSE: CUjit_option = 12;

#[repr(C)]
#[allow(non_camel_case_types)]
enum CUjitInputType {
    CU_JIT_INPUT_PTX = 1,
}

#[cfg(has_cuda)]
unsafe extern "C" {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> CUresult;
    fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: i32) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuModuleGetFunction(func: *mut CUfunction, module: CUmodule, name: *const i8) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult;
    fn cuStreamSynchronize(stream: CUstream) -> CUresult;

    // Linking functions for device code
    fn cuLinkCreate_v2(
        num_options: u32,
        options: *mut CUjit_option,
        option_values: *mut *mut c_void,
        state_out: *mut CUlinkState,
    ) -> CUresult;

    fn cuLinkAddData_v2(
        state: CUlinkState,
        type_: CUjitInputType,
        data: *const c_void,
        size: usize,
        name: *const i8,
        num_options: u32,
        options: *mut CUjit_option,
        option_values: *mut *mut c_void,
    ) -> CUresult;

    fn cuLinkComplete(
        state: CUlinkState,
        cubin_out: *mut *mut c_void,
        size_out: *mut usize,
    ) -> CUresult;

    fn cuLinkDestroy(state: CUlinkState) -> CUresult;

    // Device property functions
    fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> CUresult;
    fn cuDeviceGetName(name: *mut i8, len: i32, dev: i32) -> CUresult;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: i32) -> CUresult;

    fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const c_void,
        num_options: u32,
        options: *mut CUjit_option,
        option_values: *mut *mut c_void,
    ) -> CUresult;
}

// CUDA memory copy directions
pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

// CUDA device attributes for cuDeviceGetAttribute
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 10;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

// GPU device properties and configuration
#[derive(Debug, Clone)]
pub struct GpuDeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_multiprocessor: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
    pub shared_memory_per_block: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
}

#[derive(Debug, Clone)]
pub struct GpuLaunchConfig {
    pub blocks: u32,
    pub threads_per_block: u32,
}

impl GpuLaunchConfig {
    /// Create optimal configuration for reduction operations
    pub fn for_reduction(props: &GpuDeviceProperties, data_size: usize) -> Self {
        // Reductions benefit from fewer blocks to minimize atomic contention
        let threads_per_block = 256u32; // Good balance for most GPUs

        // Use 2-4 blocks per SM for reductions
        let blocks = (props.multiprocessor_count as u32 * 2)
            .min((data_size as u32).div_ceil(threads_per_block))
            .max(1);

        Self {
            blocks,
            threads_per_block,
        }
    }

    /// Create optimal configuration for embarrassingly parallel operations
    pub fn for_map_operation(props: &GpuDeviceProperties, data_size: usize) -> Self {
        let threads_per_block = 256u32;

        // Use more blocks for better occupancy
        let blocks = (props.multiprocessor_count as u32 * 8)
            .min((data_size as u32).div_ceil(threads_per_block))
            .max(1);

        Self {
            blocks,
            threads_per_block,
        }
    }

    /// Create optimal configuration for sorting operations
    pub fn for_sort(props: &GpuDeviceProperties, data_size: usize) -> Self {
        // Sorting needs balanced configuration
        let threads_per_block = 512u32.min(props.max_threads_per_block as u32);

        let blocks = (props.multiprocessor_count as u32 * 4)
            .min((data_size as u32).div_ceil(threads_per_block))
            .max(1);

        Self {
            blocks,
            threads_per_block,
        }
    }

    /// Create configuration for distance/similarity operations
    pub fn for_distance(props: &GpuDeviceProperties, vector_dim: usize) -> Self {
        // Distance calculations benefit from coalesced memory access
        let threads_per_block = if vector_dim <= 128 { 128u32 } else { 256u32 };

        let blocks = (props.multiprocessor_count as u32 * 4).max(1);

        Self {
            blocks,
            threads_per_block,
        }
    }
}

// Global module cache - can be accessed from any thread
lazy_static::lazy_static! {
  static ref MODULE_CACHE: Mutex<HashMap<String, SendModule>> = Mutex::new(HashMap::new());
  static ref CUDA_INITIALIZED: Mutex<bool> = Mutex::new(false);
  static ref GPU_PROPERTIES: Mutex<Option<GpuDeviceProperties>> = Mutex::new(None);
  static ref CUDA_CONTEXT: Mutex<Option<SendContext>> = Mutex::new(None);
  // Serialize GPU kernel launches to prevent race conditions
  static ref GPU_LAUNCH_MUTEX: Mutex<()> = Mutex::new(());
}

// Thread-local stream for each thread to have its own
thread_local! {
  static THREAD_STREAM: std::cell::RefCell<Option<CUstream>> = const { std::cell::RefCell::new(None) };
}

// Initialize CUDA if not already done
pub(crate) fn ensure_cuda_initialized() -> Result<(), crate::types::HwxError> {
    let mut initialized = CUDA_INITIALIZED.lock().unwrap();
    if !*initialized {
        unsafe {
            let result = cuInit(0);
            if result != 0 {
                debug!("HWX GPU: cuInit failed code={}", result);
                return Err(crate::types::HwxError::Internal(format!(
                    "cuInit failed: {}",
                    result
                )));
            }

            let mut device = 0;
            let result = cuDeviceGet(&mut device, 0);
            if result != 0 {
                debug!("HWX GPU: cuDeviceGet failed code={}", result);
                return Err(crate::types::HwxError::Internal(format!(
                    "cuDeviceGet failed: {}",
                    result
                )));
            }

            let mut ctx = ptr::null_mut();
            let result = cuCtxCreate_v2(&mut ctx, 0, device);
            if result != 0 {
                debug!("HWX GPU: cuCtxCreate_v2 failed code={}", result);
                return Err(crate::types::HwxError::Internal(format!(
                    "cuCtxCreate failed: {}",
                    result
                )));
            }

            // Store the context globally
            let mut ctx_cache = CUDA_CONTEXT.lock().unwrap();
            *ctx_cache = Some(SendContext(ctx));
        }
        *initialized = true;
    }
    // Always set context current for the calling thread
    unsafe {
        if let Some(ref ctx) = *CUDA_CONTEXT.lock().unwrap() {
            let result = cuCtxSetCurrent(ctx.0);
            if result != 0 {
                debug!(
                    "HWX GPU: cuCtxSetCurrent (post-init) failed code={}",
                    result
                );
                return Err(crate::types::HwxError::Internal(format!(
                    "cuCtxSetCurrent failed: {}",
                    result
                )));
            }
        }
    }

    Ok(())
}

#[inline]
fn ensure_context_current_for_thread() -> Result<(), crate::types::HwxError> {
    ensure_cuda_initialized()?;
    unsafe {
        if let Some(ref ctx) = *CUDA_CONTEXT.lock().unwrap() {
            let result = cuCtxSetCurrent(ctx.0);
            if result != 0 {
                debug!(
                    "HWX GPU: cuCtxSetCurrent (ensure_context_current_for_thread) failed code={}",
                    result
                );
                return Err(crate::types::HwxError::Internal(format!(
                    "cuCtxSetCurrent failed: {}",
                    result
                )));
            }
        }
    }
    Ok(())
}

/// Get GPU device properties (cached after first call)
#[cfg(has_cuda)]
pub fn get_gpu_properties() -> Result<GpuDeviceProperties, crate::types::HwxError> {
    ensure_cuda_initialized()?;

    let mut props_cache = GPU_PROPERTIES.lock().unwrap();
    if let Some(ref props) = *props_cache {
        return Ok(props.clone());
    }

    // Query device properties
    unsafe {
        let device = 0i32; // Use device 0

        // Get device name
        let mut name_bytes = vec![0i8; 256];
        let result = cuDeviceGetName(name_bytes.as_mut_ptr(), 256, device);
        if result != 0 {
            return Err(crate::types::HwxError::Internal(format!(
                "cuDeviceGetName failed: {}",
                result
            )));
        }

        // name_bytes is a fixed-size buffer returned by CUDA; interpret as C string without taking ownership
        let name = CStr::from_ptr(name_bytes.as_ptr())
            .to_string_lossy()
            .to_string();

        // Get total memory
        let mut total_memory = 0usize;
        let result = cuDeviceTotalMem_v2(&mut total_memory, device);
        if result != 0 {
            return Err(crate::types::HwxError::Internal(format!(
                "cuDeviceTotalMem failed: {}",
                result
            )));
        }

        // Get various attributes
        let get_attribute = |attr: i32| -> Result<i32, crate::types::HwxError> {
            let mut value = 0i32;
            let result = cuDeviceGetAttribute(&mut value, attr, device);
            if result != 0 {
                return Err(crate::types::HwxError::Internal(format!(
                    "cuDeviceGetAttribute failed: {}",
                    result
                )));
            }
            Ok(value)
        };

        let props = GpuDeviceProperties {
            name,
            total_memory,
            multiprocessor_count: get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            max_threads_per_multiprocessor: get_attribute(
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            )?,
            max_threads_per_block: get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            warp_size: get_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
            shared_memory_per_block: get_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?
                as usize,
            compute_capability_major: get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            compute_capability_minor: get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        };

        // Cache the properties
        *props_cache = Some(props.clone());
        Ok(props)
    }
}

// Get or create a stream for this thread
fn get_thread_stream() -> Result<CUstream, crate::types::HwxError> {
    THREAD_STREAM.with(|stream_cell| {
        let mut stream_opt = stream_cell.borrow_mut();
        if stream_opt.is_none() {
            let mut stream = ptr::null_mut();
            unsafe {
                let result = cuStreamCreate(&mut stream, 0);
                if result != 0 {
                    return Err(crate::types::HwxError::Internal(format!(
                        "cuStreamCreate failed: {}",
                        result
                    )));
                }
            }
            *stream_opt = Some(stream);
        }
        Ok(stream_opt.unwrap())
    })
}

/// Optimal launch configurations for different kernel types
pub struct LaunchConfig;

impl LaunchConfig {
    /// For reduction operations (sum, min, max, etc)
    /// Uses fewer blocks to minimize atomic contention
    pub fn reduction() -> (u32, u32) {
        if let Ok(props) = get_gpu_properties() {
            let blocks = (props.multiprocessor_count as u32 * 2).min(256);
            (blocks, 256)
        } else {
            (80, 256) // Fallback: 80 blocks × 256 threads
        }
    }

    /// For embarrassingly parallel operations (map, transform, etc)
    /// Maximizes parallelism
    pub fn parallel() -> (u32, u32) {
        if let Ok(props) = get_gpu_properties() {
            let blocks = (props.multiprocessor_count as u32 * 8).min(2048);
            (blocks, 256)
        } else {
            (320, 256) // Fallback: 320 blocks × 256 threads
        }
    }

    /// For sorting operations
    /// Balanced configuration
    pub fn sort() -> (u32, u32) {
        if let Ok(props) = get_gpu_properties() {
            let blocks = (props.multiprocessor_count as u32 * 4).min(1024);
            (blocks, 256)
        } else {
            (160, 256) // Fallback: 160 blocks × 256 threads
        }
    }

    /// For string operations with warp intrinsics and inter-warp cooperation
    /// Uses multiple warps per block for maximum parallelism
    pub fn strings() -> (u32, u32) {
        if let Ok(props) = get_gpu_properties() {
            // Use fewer blocks but more threads per block for string cooperation
            let blocks = (props.multiprocessor_count as u32 * 2).min(512);
            (blocks, 256) // 256 threads = 8 warps per block for cooperation
        } else {
            (128, 256) // Fallback: 128 blocks × 256 threads (8 warps per block)
        }
    }
}

/// Launch PTX kernel - THE ONLY LAUNCH FUNCTION
/// Pass dependencies as empty slice &[] if none needed
pub fn launch_ptx(
    ptx_code: &'static str,
    dependencies: &[&'static str],
    kernel_name: &str,
    blocks: u32,
    threads: u32,
    args: &[*const u8],
) -> Result<(), crate::types::HwxError> {
    debug!(
        "HWX GPU: launch_ptx kernel={} deps={}",
        kernel_name,
        dependencies.len()
    );
    debug!("HWX GPU: ensure_cuda_initialized begin");
    ensure_cuda_initialized()?;
    debug!("HWX GPU: ensure_cuda_initialized ok");
    // Get or compile the linked module (cache by PTX content and dependencies, not kernel name)
    let module = {
        // Acquire the GPU launch mutex only for context set and module cache/JIT
        let _gpu_lock = GPU_LAUNCH_MUTEX.lock().unwrap();

        // Set the context as current for this thread
        unsafe {
            debug!("HWX GPU: cuCtxSetCurrent begin");
            if let Some(ref ctx) = *CUDA_CONTEXT.lock().unwrap() {
                let result = cuCtxSetCurrent(ctx.0);
                if result != 0 {
                    return Err(crate::types::HwxError::Internal(format!(
                        "cuCtxSetCurrent failed: {}",
                        result
                    )));
                }
            }
            debug!("HWX GPU: cuCtxSetCurrent ok");
        }

        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a 64-bit offset basis
        let ptx_bytes = ptx_code.as_bytes();
        for &byte in &(ptx_bytes.len() as u64).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for &b in ptx_bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for dep in dependencies {
            let dep_bytes = dep.as_bytes();
            for &byte in &(dep_bytes.len() as u64).to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            for &b in dep_bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        let key = format!("ptx:{:016x}", hash);
        let mut cache = MODULE_CACHE.lock().unwrap();
        debug!(
            "HWX GPU: computed module key {} (cache_len={})",
            key,
            cache.len()
        );

        if !cache.contains_key(&key) {
            // Evict least-recently-inserted if cache grows too large to avoid device OOM
            const MAX_MODULES: usize = 64;
            if cache.len() >= MAX_MODULES {
                if let Some(first_key) = cache.keys().next().cloned() {
                    cache.remove(&first_key);
                }
            }

            // Keep logs alive for the whole JIT scope
            let mut error_log: Vec<i8> = vec![0; 8192];
            let mut info_log: Vec<i8> = vec![0; 8192];
            let error_log_size_u32: u32 = error_log.len() as u32;
            let info_log_size_u32: u32 = info_log.len() as u32;
            let verbose_flag_u32: u32 = 1;
            let mut options: [CUjit_option; 5] = [
                CU_JIT_ERROR_LOG_BUFFER,
                CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                CU_JIT_INFO_LOG_BUFFER,
                CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                CU_JIT_LOG_VERBOSE,
            ];
            let mut option_values: [*mut c_void; 5] = [
                error_log.as_mut_ptr() as *mut c_void,
                (error_log_size_u32 as usize) as *mut c_void,
                info_log.as_mut_ptr() as *mut c_void,
                (info_log_size_u32 as usize) as *mut c_void,
                (verbose_flag_u32 as usize) as *mut c_void,
            ];

            if dependencies.is_empty() {
                // Direct JIT compile PTX to module without linker
                let mut module = ptr::null_mut();
                let ptx_cstring = CString::new(ptx_code).map_err(|e| {
                    crate::types::HwxError::Internal(format!("Invalid PTX code: {}", e))
                })?;
                unsafe {
                    debug!("HWX GPU: cuModuleLoadDataEx (direct PTX)");
                    let result = cuModuleLoadDataEx(
                        &mut module,
                        ptx_cstring.as_ptr() as *const c_void,
                        options.len() as u32,
                        options.as_mut_ptr(),
                        option_values.as_mut_ptr(),
                    );
                    if result != 0 {
                        let len = error_log
                            .iter()
                            .position(|&c| c == 0)
                            .unwrap_or(error_log.len());
                        let err = {
                            let ptr = error_log.as_ptr() as *const u8;
                            let slice = std::slice::from_raw_parts(ptr, len);
                            String::from_utf8_lossy(slice).to_string()
                        };
                        debug!(
              "HWX GPU: cuModuleLoadDataEx (direct PTX) failed (result={}) | error_log=\"{}\"",
              result, err
            );
                        return Err(crate::types::HwxError::Internal(format!(
                            "cuModuleLoadDataEx (PTX) failed: {} | {}",
                            result, err
                        )));
                    }
                    // Log any JIT info output for diagnostics
                    let info_len = info_log
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(info_log.len());
                    if info_len > 0 {
                        let info_msg = {
                            let ptr = info_log.as_ptr() as *const u8;
                            let slice = std::slice::from_raw_parts(ptr, info_len);
                            String::from_utf8_lossy(slice).to_string()
                        };
                        debug!("HWX GPU: load info: {}", info_msg);
                    }
                }

                debug!(
                    "HWX GPU: module cache insert {} (before_len={})",
                    key,
                    cache.len()
                );
                cache.insert(key.clone(), SendModule(module));
                debug!(
                    "HWX GPU: module cache insert ok {} (after_len={})",
                    key,
                    cache.len()
                );
                module
            } else {
                // Linker path for when there are dependencies
                let mut link_state = ptr::null_mut();
                unsafe {
                    debug!("HWX GPU: cuLinkCreate_v2 begin (with logs)");
                    let result = cuLinkCreate_v2(
                        options.len() as u32,
                        options.as_mut_ptr(),
                        option_values.as_mut_ptr(),
                        &mut link_state,
                    );
                    if result != 0 {
                        let len = error_log
                            .iter()
                            .position(|&c| c == 0)
                            .unwrap_or(error_log.len());
                        let err = {
                            let ptr = error_log.as_ptr() as *const u8;
                            let slice = std::slice::from_raw_parts(ptr, len);
                            String::from_utf8_lossy(slice).to_string()
                        };
                        debug!(
                            "HWX GPU: cuLinkCreate_v2 failed (result={}) | error_log=\"{}\"",
                            result, err
                        );
                        return Err(crate::types::HwxError::Internal(format!(
                            "cuLinkCreate failed: {} | {}",
                            result, err
                        )));
                    }
                    debug!("HWX GPU: cuLinkCreate_v2 ok");
                }

                // Add main PTX code
                let ptx_cstring = CString::new(ptx_code).map_err(|e| {
                    crate::types::HwxError::Internal(format!("Invalid PTX code: {}", e))
                })?;
                let kernel_name_cstr = CString::new(kernel_name).map_err(|e| {
                    crate::types::HwxError::Internal(format!("Invalid kernel name: {}", e))
                })?;

                unsafe {
                    debug!("HWX GPU: cuLinkAddData_v2 main ({} bytes)", ptx_code.len());
                    let result = cuLinkAddData_v2(
                        link_state,
                        CUjitInputType::CU_JIT_INPUT_PTX,
                        ptx_cstring.as_ptr() as *const c_void,
                        ptx_cstring.as_bytes_with_nul().len(),
                        kernel_name_cstr.as_ptr(),
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    );
                    if result != 0 {
                        cuLinkDestroy(link_state);
                        let len = error_log
                            .iter()
                            .position(|&c| c == 0)
                            .unwrap_or(error_log.len());
                        let err = {
                            let ptr = error_log.as_ptr() as *const u8;
                            let slice = std::slice::from_raw_parts(ptr, len);
                            String::from_utf8_lossy(slice).to_string()
                        };
                        debug!(
              "HWX GPU: cuLinkAddData_v2 (main) failed (result={}) | error_log=\"{}\"",
              result, err
            );
                        return Err(crate::types::HwxError::Internal(format!(
                            "cuLinkAddData failed: {} | {}",
                            result, err
                        )));
                    }
                    debug!("HWX GPU: cuLinkAddData_v2 main ok");
                }

                // Add dependencies (keep CStrings alive until after link completes)
                let mut dep_cstrings: Vec<CString> = Vec::with_capacity(dependencies.len());
                let mut dep_names: Vec<CString> = Vec::with_capacity(dependencies.len());
                for (i, dep_ptx) in dependencies.iter().enumerate() {
                    let dep_cstring = CString::new(*dep_ptx).map_err(|e| {
                        crate::types::HwxError::Internal(format!("Invalid dependency PTX: {}", e))
                    })?;
                    let dep_name = CString::new(format!("dep_{}", i)).map_err(|e| {
                        crate::types::HwxError::Internal(format!("Invalid dependency name: {}", e))
                    })?;

                    unsafe {
                        debug!(
                            "HWX GPU: cuLinkAddData_v2 dep {} ({} bytes)",
                            i,
                            dep_ptx.len()
                        );
                        let result = cuLinkAddData_v2(
                            link_state,
                            CUjitInputType::CU_JIT_INPUT_PTX,
                            dep_cstring.as_ptr() as *const c_void,
                            dep_cstring.as_bytes_with_nul().len(),
                            dep_name.as_ptr(),
                            0,
                            ptr::null_mut(),
                            ptr::null_mut(),
                        );
                        if result != 0 {
                            cuLinkDestroy(link_state);
                            let len = error_log
                                .iter()
                                .position(|&c| c == 0)
                                .unwrap_or(error_log.len());
                            let err = {
                                let ptr = error_log.as_ptr() as *const u8;
                                let slice = std::slice::from_raw_parts(ptr, len);
                                String::from_utf8_lossy(slice).to_string()
                            };
                            debug!(
                "HWX GPU: cuLinkAddData_v2 (dep {}) failed (result={}) | error_log=\"{}\"",
                i, result, err
              );
                            return Err(crate::types::HwxError::Internal(format!(
                                "cuLinkAddData for dependency failed: {} | {}",
                                result, err
                            )));
                        }
                        debug!("HWX GPU: cuLinkAddData_v2 dep {} ok", i);
                    }

                    dep_cstrings.push(dep_cstring);
                    dep_names.push(dep_name);
                }

                // Complete linking
                let mut cubin: *mut c_void = ptr::null_mut();
                let mut cubin_size: usize = 0;
                unsafe {
                    debug!("HWX GPU: cuLinkComplete");
                    let result = cuLinkComplete(link_state, &mut cubin, &mut cubin_size);
                    if result != 0 {
                        cuLinkDestroy(link_state);
                        let len = error_log
                            .iter()
                            .position(|&c| c == 0)
                            .unwrap_or(error_log.len());
                        let err = {
                            let ptr = error_log.as_ptr() as *const u8;
                            let slice = std::slice::from_raw_parts(ptr, len);
                            String::from_utf8_lossy(slice).to_string()
                        };
                        debug!(
                            "HWX GPU: cuLinkComplete failed (result={}) | error_log=\"{}\"",
                            result, err
                        );
                        return Err(crate::types::HwxError::Internal(format!(
                            "cuLinkComplete failed: {} | {}",
                            result, err
                        )));
                    }
                    // Log any linker info output for diagnostics
                    let info_len = info_log
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(info_log.len());
                    if info_len > 0 {
                        let info_msg = {
                            let ptr = info_log.as_ptr() as *const u8;
                            let slice = std::slice::from_raw_parts(ptr, info_len);
                            String::from_utf8_lossy(slice).to_string()
                        };
                        debug!("HWX GPU: link info: {}", info_msg);
                    }
                    debug!("HWX GPU: cuLinkComplete ok (cubin_size={})", cubin_size);
                }

                // Load module from cubin
                let mut module = ptr::null_mut();
                unsafe {
                    debug!("HWX GPU: cuModuleLoadDataEx");
                    let result =
                        cuModuleLoadDataEx(&mut module, cubin, 0, ptr::null_mut(), ptr::null_mut());
                    if result != 0 {
                        cuLinkDestroy(link_state);
                        debug!("HWX GPU: cuModuleLoadDataEx failed (result={})", result);
                        return Err(crate::types::HwxError::Internal(format!(
                            "cuModuleLoadDataEx failed: {}",
                            result
                        )));
                    }

                    cuLinkDestroy(link_state);
                    debug!("HWX GPU: cuModuleLoadDataEx ok");
                }

                debug!(
                    "HWX GPU: module cache insert {} (before_len={})",
                    key,
                    cache.len()
                );
                cache.insert(key.clone(), SendModule(module));
                debug!(
                    "HWX GPU: module cache insert ok {} (after_len={})",
                    key,
                    cache.len()
                );
                module
            }
        } else {
            debug!("HWX GPU: module cache hit {}", key);
            cache[&key].0
        }
    };

    // Get the kernel function from module
    let kernel_cstring = CString::new(kernel_name)
        .map_err(|e| crate::types::HwxError::Internal(format!("Invalid kernel name: {}", e)))?;

    let mut function = ptr::null_mut();
    unsafe {
        debug!("HWX GPU: cuModuleGetFunction({})", kernel_name);
        let result = cuModuleGetFunction(&mut function, module, kernel_cstring.as_ptr());
        if result != 0 {
            return Err(crate::types::HwxError::Internal(format!(
                "cuModuleGetFunction failed: {}",
                result
            )));
        }
    }

    // Get this thread's stream
    let stream = get_thread_stream()?;

    // Launch the kernel
    unsafe {
        // CUDA expects an array of pointers to the actual parameters
        // args already contains pointers to the parameter values (e.g. pointers to u64 values)
        // We just need to convert them to *mut c_void for CUDA
        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(args.len());

        for arg in args {
            // Each arg is already a pointer to the parameter value
            // Just cast it to *mut c_void for CUDA
            kernel_params.push(*arg as *mut c_void);
        }

        // Radix sort kernels need shared memory
        let shared_mem = if kernel_name.contains("radix_sort") {
            512 * std::mem::size_of::<u32>() as u32 // 2KB for histogram and counters
        } else {
            0
        };

        debug!(
            "HWX GPU: cuLaunchKernel blocks={} threads={} args={}",
            blocks,
            threads,
            kernel_params.len()
        );
        let result = cuLaunchKernel(
            function,
            blocks,
            1,
            1, // grid dimensions
            threads,
            1,
            1,          // block dimensions
            shared_mem, // shared memory
            stream,     // stream for this thread
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        );

        if result != 0 {
            return Err(crate::types::HwxError::Internal(format!(
                "cuLaunchKernel failed: {}",
                result
            )));
        }

        debug!("HWX GPU: cuStreamSynchronize");
        // Synchronize the stream
        let result = cuStreamSynchronize(stream);
        if result != 0 {
            return Err(crate::types::HwxError::Internal(format!(
                "cuStreamSynchronize failed: {}",
                result
            )));
        }
    }

    Ok(())
}

/// Helper for byte array GPU operations (string processing)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u8<F>(
    input: &[u8],
    len: usize,
    compute_fn: F,
) -> Result<usize, crate::types::HwxError>
where
    F: FnOnce(*const u8, usize) -> usize,
{
    // Ensure CUDA context is current for this thread
    ensure_context_current_for_thread()?;
    // Allocate GPU memory
    let mut gpu_input: *mut u8 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u8>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u8 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    let result = compute_fn(gpu_input, len);

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };

    Ok(result)
}

/// Helper for mutable byte array GPU operations (in-place string processing)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u8_mut<F>(
    input: &mut [u8],
    len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u8, usize),
{
    // Ensure CUDA context is current for this thread
    ensure_context_current_for_thread()?;
    // Allocate GPU memory
    let mut gpu_input: *mut u8 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u8>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u8 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, len);

    // Copy back modified data
    if unsafe {
        cudaMemcpy(
            input.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_input as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };

    Ok(())
}

/// Helper for single-array GPU operations (reductions, math functions)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64<F>(
    input: &[f64],
    len: usize,
    init_value: f64,
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, usize, *mut f64),
{
    // Allocate GPU memory
    let mut gpu_input: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut f64 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut f64 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<f64>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Initialize result with init_value
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &init_value as *const f64 as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU init failed".to_string(),
        ));
    }

    // Compute on GPU - pass the GPU result pointer
    compute_fn(gpu_input, len, gpu_result);

    // Copy result back from GPU
    let mut result: f64 = 0.0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut f64 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_result as *mut std::ffi::c_void);
    }

    Ok(result)
}

/// Helper for string pattern matching operations on GPU
#[cfg(has_cuda)]
pub unsafe fn with_gpu_string_match<F>(
    strings: &mut [u8],
    pattern: &[u8],
    strings_len: usize,
    pattern_len: usize,
    max_size: usize,
    compute_fn: F,
) -> Result<usize, crate::types::HwxError>
where
    F: FnOnce(*mut u8, *const u8, usize, usize, usize) -> usize,
{
    // Allocate GPU memory
    let mut gpu_strings: *mut u8 = ptr::null_mut();
    let mut gpu_pattern: *mut u8 = ptr::null_mut();
    let strings_size = strings_len * std::mem::size_of::<u8>();
    let pattern_size = pattern_len * std::mem::size_of::<u8>();

    if unsafe {
        cudaMalloc(
            &mut gpu_strings as *mut *mut u8 as *mut *mut std::ffi::c_void,
            strings_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_pattern as *mut *mut u8 as *mut *mut std::ffi::c_void,
            pattern_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy inputs to GPU
    if unsafe {
        cudaMemcpy(
            gpu_strings as *mut std::ffi::c_void,
            strings.as_ptr() as *const std::ffi::c_void,
            strings_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_pattern as *mut std::ffi::c_void,
            pattern.as_ptr() as *const std::ffi::c_void,
            pattern_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    let count = compute_fn(gpu_strings, gpu_pattern, strings_len, pattern_len, max_size);

    // Copy result back to CPU (filtered strings)
    if unsafe {
        cudaMemcpy(
            strings.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_strings as *const std::ffi::c_void,
            count * std::mem::size_of::<u8>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };

    Ok(count)
}

/// Helper for two-array GPU operations (distance functions)  
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffers_2d_f32<F>(
    input_a: &[f32],
    input_b: &[f32],
    len_a: usize,
    len_b: usize,
    compute_fn: F,
) -> Result<f32, crate::types::HwxError>
where
    F: FnOnce(*const f32, *const f32, usize, usize, *mut f32),
{
    // Allocate GPU memory for inputs
    let mut gpu_a: *mut f32 = ptr::null_mut();
    let mut gpu_b: *mut f32 = ptr::null_mut();
    let size_a = len_a * std::mem::size_of::<f32>();
    let size_b = len_b * std::mem::size_of::<f32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_a as *mut *mut f32 as *mut *mut std::ffi::c_void,
            size_a,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_b as *mut *mut f32 as *mut *mut std::ffi::c_void,
            size_b,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut f32 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut f32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<f32>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Initialize result with 0.0
    let init_value: f32 = 0.0;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &init_value as *const f32 as *const std::ffi::c_void,
            std::mem::size_of::<f32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU init failed".to_string(),
        ));
    }

    // Copy inputs to GPU
    if unsafe {
        cudaMemcpy(
            gpu_a as *mut std::ffi::c_void,
            input_a.as_ptr() as *const std::ffi::c_void,
            size_a,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_b as *mut std::ffi::c_void,
            input_b.as_ptr() as *const std::ffi::c_void,
            size_b,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU - pass the GPU result pointer
    compute_fn(gpu_a, gpu_b, len_a, len_b, gpu_result);

    // Copy result back from GPU
    let mut result: f32 = 0.0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut f32 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<f32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_a as *mut std::ffi::c_void);
        cudaFree(gpu_b as *mut std::ffi::c_void);
        cudaFree(gpu_result as *mut std::ffi::c_void);
    }

    Ok(result)
}

/// Helper for two-array GPU operations with u32 arrays
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffers_2d_u32<F>(
    input_a: &[u32],
    input_b: &[u32],
    len_a: usize,
    len_b: usize,
    compute_fn: F,
) -> Result<f32, crate::types::HwxError>
where
    F: FnOnce(*const u32, *const u32, usize, usize, *mut f32),
{
    // Allocate GPU memory
    let mut gpu_a: *mut u32 = ptr::null_mut();
    let mut gpu_b: *mut u32 = ptr::null_mut();
    let size_a = len_a * std::mem::size_of::<u32>();
    let size_b = len_b * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_a as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_a,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_b as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_b,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut f32 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut f32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<f32>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Initialize result with 0
    let init_value: f32 = 0.0;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &init_value as *const f32 as *const std::ffi::c_void,
            std::mem::size_of::<f32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU init failed".to_string(),
        ));
    }

    // Copy inputs to GPU
    if unsafe {
        cudaMemcpy(
            gpu_a as *mut std::ffi::c_void,
            input_a.as_ptr() as *const std::ffi::c_void,
            size_a,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_b as *mut std::ffi::c_void,
            input_b.as_ptr() as *const std::ffi::c_void,
            size_b,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_a, gpu_b, len_a, len_b, gpu_result);

    // Copy result back from GPU
    let mut result: f32 = 0.0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut f32 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<f32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };

    Ok(result)
}

/// Helper for two-array GPU operations with u16 arrays
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffers_2d_u16<F>(
    input_a: &[u16],
    input_b: &[u16],
    len_a: usize,
    len_b: usize,
    compute_fn: F,
) -> Result<f32, crate::types::HwxError>
where
    F: FnOnce(*const u16, *const u16, usize, usize, *mut f32),
{
    // Allocate GPU memory
    let mut gpu_a: *mut u16 = ptr::null_mut();
    let mut gpu_b: *mut u16 = ptr::null_mut();
    let size_a = len_a * std::mem::size_of::<u16>();
    let size_b = len_b * std::mem::size_of::<u16>();

    if unsafe {
        cudaMalloc(
            &mut gpu_a as *mut *mut u16 as *mut *mut std::ffi::c_void,
            size_a,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_b as *mut *mut u16 as *mut *mut std::ffi::c_void,
            size_b,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut f32 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut f32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<f32>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Initialize result with 0
    let init_value: f32 = 0.0;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &init_value as *const f32 as *const std::ffi::c_void,
            std::mem::size_of::<f32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_a as *mut std::ffi::c_void);
            cudaFree(gpu_b as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU init failed".to_string(),
        ));
    }

    // Copy inputs to GPU
    if unsafe {
        cudaMemcpy(
            gpu_a as *mut std::ffi::c_void,
            input_a.as_ptr() as *const std::ffi::c_void,
            size_a,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_b as *mut std::ffi::c_void,
            input_b.as_ptr() as *const std::ffi::c_void,
            size_b,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_a, gpu_b, len_a, len_b, gpu_result);

    // Copy result back from GPU
    let mut result: f32 = 0.0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut f32 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<f32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };

    Ok(result)
}

/// Helper for single-array GPU operations with u32 arrays (returns u32)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32<F>(
    input: &[u32],
    len: usize,
    init_value: u32,
    compute_fn: F,
) -> Result<u32, crate::types::HwxError>
where
    F: FnOnce(*mut u32, usize, *mut u32),
{
    // Ensure CUDA context is current for this thread
    ensure_context_current_for_thread()?;
    // Allocate GPU memory for input
    let mut gpu_input: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut u32 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut u32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<u32>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &init_value as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU init failed".to_string(),
        ));
    }

    // Compute on GPU - pass the GPU result pointer
    compute_fn(gpu_input, len, gpu_result);

    // The kernel should have written to gpu_result, now copy it back
    let mut result: u32 = 0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut u32 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_result as *mut std::ffi::c_void);
    }

    Ok(result)
}

/// Helper for single-array GPU operations with u32 arrays that modify in-place
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_inplace<F>(
    values: &mut [u32],
    len: usize,
    compute_fn: F,
) -> Result<u32, crate::types::HwxError>
where
    F: FnOnce(*mut u32, usize) -> u32,
{
    // Ensure CUDA context is current for this thread
    ensure_context_current_for_thread()?;
    // Allocate GPU memory
    let mut gpu_values: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Process on GPU
    let count = compute_fn(gpu_values, len);

    // Copy back from GPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_values as *const std::ffi::c_void,
            count as usize * std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };

    Ok(count)
}

/// Helper for single-array GPU operations with u64 arrays (returns u64)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u64<F>(
    input: &[u64],
    len: usize,
    compute_fn: F,
) -> Result<u64, crate::types::HwxError>
where
    F: FnOnce(*const u64, usize) -> u64,
{
    // Allocate GPU memory
    let mut gpu_input: *mut u64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    let result = compute_fn(gpu_input, len);

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };

    Ok(result)
}

/// Helper for single-array GPU operations with i64 arrays (returns (i64, i64))
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_i64<F>(
    input: &[i64],
    len: usize,
    compute_fn: F,
) -> Result<(i64, i64), crate::types::HwxError>
where
    F: FnOnce(*const i64, usize) -> (i64, i64),
{
    // Allocate GPU memory
    let mut gpu_input: *mut i64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<i64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut i64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    let result = compute_fn(gpu_input, len);

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };

    Ok(result)
}

/// Helper for min/max operations with i64 arrays
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_i64_minmax<F>(
    input: &[i64],
    len: usize,
    compute_fn: F,
) -> Result<(i64, i64), crate::types::HwxError>
where
    F: FnOnce(*const i64, usize, *mut i64, *mut i64),
{
    // Allocate GPU memory for input
    let mut gpu_input: *mut i64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<i64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut i64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for min and max results
    let mut gpu_min: *mut i64 = ptr::null_mut();
    let mut gpu_max: *mut i64 = ptr::null_mut();

    if unsafe {
        cudaMalloc(
            &mut gpu_min as *mut *mut i64 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<i64>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for min failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_max as *mut *mut i64 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<i64>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for max failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Initialize min and max values on GPU
    let initial_min = i64::MAX;
    let initial_max = i64::MIN;

    if unsafe {
        cudaMemcpy(
            gpu_min as *mut std::ffi::c_void,
            &initial_min as *const i64 as *const std::ffi::c_void,
            std::mem::size_of::<i64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed for initial min".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_max as *mut std::ffi::c_void,
            &initial_max as *const i64 as *const std::ffi::c_void,
            std::mem::size_of::<i64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed for initial max".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, len, gpu_min, gpu_max);

    // Copy results back to CPU
    let mut cpu_min: i64 = 0;
    let mut cpu_max: i64 = 0;

    if unsafe {
        cudaMemcpy(
            &mut cpu_min as *mut i64 as *mut std::ffi::c_void,
            gpu_min as *const std::ffi::c_void,
            std::mem::size_of::<i64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed for min result".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            &mut cpu_max as *mut i64 as *mut std::ffi::c_void,
            gpu_max as *const std::ffi::c_void,
            std::mem::size_of::<i64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed for max result".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_min as *mut std::ffi::c_void);
        cudaFree(gpu_max as *mut std::ffi::c_void);
    };

    Ok((cpu_min, cpu_max))
}

/// Helper for single-array GPU search operations with u32 arrays + target (returns usize)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_search<F>(
    input: &[u32],
    target: u32,
    len: usize,
    compute_fn: F,
) -> Result<usize, crate::types::HwxError>
where
    F: FnOnce(*const u32, u32, usize, *mut u32),
{
    // Ensure CUDA context is current for this thread
    ensure_context_current_for_thread()?;
    // Allocate GPU memory for input
    let mut gpu_input: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut u32 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut u32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<u32>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Initialize result to provided value (len for GE, 0 for LE, etc.)
    let init_result = len as u32;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &init_result as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, target, len, gpu_result);

    // Copy result back from GPU
    let mut cpu_result: u32 = 0;
    if unsafe {
        cudaMemcpy(
            &mut cpu_result as *mut u32 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };

    // Handle sentinel value for LE searches (u32::MAX means not found)
    let final_result = if cpu_result == u32::MAX {
        len
    } else {
        cpu_result as usize
    };
    Ok(final_result)
}

/// Helper for single-array GPU search operations with u64 arrays + target (returns usize)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u64_search<F>(
    input: &[u64],
    target: u64,
    len: usize,
    compute_fn: F,
) -> Result<usize, crate::types::HwxError>
where
    F: FnOnce(*const u64, u64, usize, *mut u32),
{
    // Allocate GPU memory for input
    let mut gpu_input: *mut u64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut u32 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut u32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<u32>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Initialize result to provided value (len for GE, 0 for LE, etc.)
    let init_result = len as u32;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &init_result as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, target, len, gpu_result);

    // Copy result back from GPU
    let mut cpu_result: u32 = 0;
    if unsafe {
        cudaMemcpy(
            &mut cpu_result as *mut u32 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };

    // Handle sentinel value for LE searches (u32::MAX means not found)
    let final_result = if cpu_result == u32::MAX {
        len
    } else {
        cpu_result as usize
    };
    Ok(final_result)
}

/// Helper for tokenization operations on GPU
#[cfg(has_cuda)]
pub unsafe fn with_gpu_tokenization<F>(
    input: &mut [u8],
    word_boundaries: &mut [u32],
    input_len: usize,
    max_words: usize,
    compute_fn: F,
) -> Result<u32, crate::types::HwxError>
where
    F: FnOnce(*mut u8, *mut u32, usize, usize) -> u32,
{
    // Allocate GPU memory
    let mut gpu_input: *mut u8 = ptr::null_mut();
    let mut gpu_boundaries: *mut u32 = ptr::null_mut();
    let input_size = input_len * std::mem::size_of::<u8>();
    let boundaries_size = max_words * 2 * std::mem::size_of::<u32>(); // start & end for each word

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u8 as *mut *mut std::ffi::c_void,
            input_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_boundaries as *mut *mut u32 as *mut *mut std::ffi::c_void,
            boundaries_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            input_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_boundaries as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    let word_count = compute_fn(gpu_input, gpu_boundaries, input_len, max_words);

    // Copy back modified input (lowercased) and boundaries
    if unsafe {
        cudaMemcpy(
            input.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_input as *const std::ffi::c_void,
            input_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_boundaries as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    if word_count > 0 {
        let boundaries_to_copy = (word_count as usize * 2).min(max_words * 2);
        if unsafe {
            cudaMemcpy(
                word_boundaries.as_mut_ptr() as *mut std::ffi::c_void,
                gpu_boundaries as *const std::ffi::c_void,
                boundaries_to_copy * std::mem::size_of::<u32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        } != 0
        {
            unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
            unsafe { cudaFree(gpu_boundaries as *mut std::ffi::c_void) };
            return Err(crate::types::HwxError::Internal(
                "GPU boundaries copy back failed".to_string(),
            ));
        }
    }

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_boundaries as *mut std::ffi::c_void) };

    Ok(word_count)
}

/// Helper for Unicode mask generation on GPU
#[cfg(has_cuda)]
pub unsafe fn with_gpu_unicode_mask<F>(
    input: *const u8,
    len: usize,
    compute_fn: F,
) -> Result<u64, crate::types::HwxError>
where
    F: FnOnce(*const u8, usize) -> u64,
{
    // Allocate GPU memory
    let mut gpu_input: *mut u8 = ptr::null_mut();
    let size_bytes = len.min(64) * std::mem::size_of::<u8>(); // Masks are max 64 bits

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u8 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute mask on GPU
    let mask = compute_fn(gpu_input, len);

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };

    Ok(mask)
}

/// Helper for classification operations on GPU
#[cfg(has_cuda)]
pub unsafe fn with_gpu_classification<F>(
    input: *const u8,
    len: usize,
    compute_fn: F,
) -> Result<(bool, bool, bool, bool, usize, usize), crate::types::HwxError>
where
    F: FnOnce(*const u8, usize) -> (bool, bool, bool, bool, usize, usize),
{
    // Allocate GPU memory
    let mut gpu_input: *mut u8 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u8>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u8 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute classification on GPU
    let result = compute_fn(gpu_input, len);

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };

    Ok(result)
}

/// Helper for operations with two input arrays and one output array  
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffers_3way_f64<F>(
    input_a: &[f64],
    input_b: &[f64],
    output: &mut [f64],
    len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*const f64, *const f64, *mut f64, usize),
{
    // Allocate GPU memory
    let mut gpu_a: *mut f64 = ptr::null_mut();
    let mut gpu_b: *mut f64 = ptr::null_mut();
    let mut gpu_result: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_a as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_b as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy inputs to GPU
    if unsafe {
        cudaMemcpy(
            gpu_a as *mut std::ffi::c_void,
            input_a.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_b as *mut std::ffi::c_void,
            input_b.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_a, gpu_b, gpu_result, len);

    // Copy result back to CPU
    if unsafe {
        cudaMemcpy(
            output.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };

    Ok(())
}

/// Helper for single input array, single output array operations
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64_inplace<F>(
    data: &mut [f64],
    len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut f64, usize),
{
    // Allocate GPU memory
    let mut gpu_data: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    // Allocate GPU memory with error checking
    if unsafe {
        cudaMalloc(
            &mut gpu_data as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed for inplace operation".to_string(),
        ));
    }

    // Copy to GPU with error checking
    if unsafe {
        cudaMemcpy(
            gpu_data as *mut std::ffi::c_void,
            data.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_data as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed (host to device)".to_string(),
        ));
    }

    // Execute GPU computation
    compute_fn(gpu_data, len);

    // Copy result back to CPU with error checking
    if unsafe {
        cudaMemcpy(
            data.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_data as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_data as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed (device to host)".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_data as *mut std::ffi::c_void);
    }

    Ok(())
}

#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffers_1in_1out_f64<F>(
    input: &[f64],
    output: &mut [f64],
    len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*const f64, *mut f64, usize),
{
    // Allocate GPU memory
    let mut gpu_input: *mut f64 = ptr::null_mut();
    let mut gpu_result: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, gpu_result, len);

    // Copy result back to CPU
    if unsafe {
        cudaMemcpy(
            output.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_result as *mut std::ffi::c_void) };

    Ok(())
}

/// CUB-based GPU sorting for u32 arrays - calls CUB DeviceRadixSort
pub unsafe fn sort_u32_cub(
    values: &mut Vec<u32>,
    len: usize,
    ascending: bool,
    dedup: bool,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Call CUB DeviceRadixSort
    let result = unsafe { cub_device_radix_sort_u32(gpu_values, len, ascending) };

    if result != 0 {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB sort failed: {}",
            result
        )));
    }

    let final_len = if dedup {
        // Allocate temporary buffer for unique operation
        let mut gpu_temp: *mut u32 = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_temp as *mut *mut u32 as *mut *mut std::ffi::c_void,
                size_bytes,
            )
        } != 0
        {
            unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
            return Err(crate::types::HwxError::Internal(
                "GPU temp allocation failed".to_string(),
            ));
        }

        // Allocate GPU memory for result count
        let mut gpu_count: *mut usize = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU count allocation failed".to_string(),
            ));
        }

        // Call CUB DeviceSelect::Unique
        let dedup_result = unsafe { cub_device_unique_u32(gpu_values, gpu_temp, len, gpu_count) };

        if dedup_result != 0 {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(format!(
                "CUB unique failed: {}",
                dedup_result
            )));
        }

        // Get the final count
        let mut final_count: usize = 0;
        if unsafe {
            cudaMemcpy(
                &mut final_count as *mut usize as *mut std::ffi::c_void,
                gpu_count as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU count copy failed".to_string(),
            ));
        }

        // Copy unique result back to original buffer
        if unsafe {
            cudaMemcpy(
                gpu_values as *mut std::ffi::c_void,
                gpu_temp as *const std::ffi::c_void,
                final_count * std::mem::size_of::<u32>(),
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU unique copy failed".to_string(),
            ));
        }

        // Cleanup temp buffers
        unsafe {
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };

        final_count
    } else {
        len
    };

    // Copy result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_values as *const std::ffi::c_void,
            final_len * std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // DON'T truncate here - let dispatcher handle it

    // Cleanup
    unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };

    Ok(final_len)
}

/// CUB-based GPU sorting for f64 arrays
pub unsafe fn sort_f64_cub(
    values: &mut Vec<f64>,
    len: usize,
    ascending: bool,
    dedup: bool,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Call CUB DeviceRadixSort
    let result = unsafe { cub_device_radix_sort_f64(gpu_values, len, ascending) };

    if result != 0 {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB f64 sort failed: {}",
            result
        )));
    }

    let final_len = if dedup {
        // Allocate temporary buffer for unique operation
        let mut gpu_temp: *mut f64 = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_temp as *mut *mut f64 as *mut *mut std::ffi::c_void,
                size_bytes,
            )
        } != 0
        {
            unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
            return Err(crate::types::HwxError::Internal(
                "GPU temp allocation failed".to_string(),
            ));
        }

        // Allocate GPU memory for result count
        let mut gpu_count: *mut usize = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU count allocation failed".to_string(),
            ));
        }

        // Call CUB DeviceSelect::Unique
        let dedup_result = unsafe { cub_device_unique_f64(gpu_values, gpu_temp, len, gpu_count) };

        if dedup_result != 0 {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(format!(
                "CUB f64 unique failed: {}",
                dedup_result
            )));
        }

        // Get the final count
        let mut final_count: usize = 0;
        if unsafe {
            cudaMemcpy(
                &mut final_count as *mut usize as *mut std::ffi::c_void,
                gpu_count as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU f64 count copy failed".to_string(),
            ));
        }

        // Copy unique result back to original buffer
        if unsafe {
            cudaMemcpy(
                gpu_values as *mut std::ffi::c_void,
                gpu_temp as *const std::ffi::c_void,
                final_count * std::mem::size_of::<f64>(),
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU f64 unique copy failed".to_string(),
            ));
        }

        // Cleanup temp buffers
        unsafe {
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };

        final_count
    } else {
        len
    };

    // Copy result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_values as *const std::ffi::c_void,
            final_len * std::mem::size_of::<f64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // DON'T truncate here - let dispatcher handle it

    // Cleanup
    unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };

    Ok(final_len)
}

/// CUB-based GPU sorting for u64 arrays
pub unsafe fn sort_u64_cub(
    values: &mut Vec<u64>,
    len: usize,
    ascending: bool,
    dedup: bool,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut u64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Call CUB DeviceRadixSort
    let result = unsafe { cub_device_radix_sort_u64(gpu_values, len, ascending) };

    if result != 0 {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB u64 sort failed: {}",
            result
        )));
    }

    let final_len = if dedup {
        // Allocate temporary buffer for unique operation
        let mut gpu_temp: *mut u64 = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_temp as *mut *mut u64 as *mut *mut std::ffi::c_void,
                size_bytes,
            )
        } != 0
        {
            unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
            return Err(crate::types::HwxError::Internal(
                "GPU temp allocation failed".to_string(),
            ));
        }

        // Allocate GPU memory for result count
        let mut gpu_count: *mut usize = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU count allocation failed".to_string(),
            ));
        }

        // Call CUB DeviceSelect::Unique
        let dedup_result = unsafe { cub_device_unique_u64(gpu_values, gpu_temp, len, gpu_count) };

        if dedup_result != 0 {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(format!(
                "CUB u64 unique failed: {}",
                dedup_result
            )));
        }

        // Get the final count
        let mut final_count: usize = 0;
        if unsafe {
            cudaMemcpy(
                &mut final_count as *mut usize as *mut std::ffi::c_void,
                gpu_count as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU u64 count copy failed".to_string(),
            ));
        }

        // Copy unique result back to original buffer
        if unsafe {
            cudaMemcpy(
                gpu_values as *mut std::ffi::c_void,
                gpu_temp as *const std::ffi::c_void,
                final_count * std::mem::size_of::<u64>(),
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU u64 unique copy failed".to_string(),
            ));
        }

        // Cleanup temp buffers
        unsafe {
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };

        final_count
    } else {
        len
    };

    // Copy result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_values as *const std::ffi::c_void,
            final_len * std::mem::size_of::<u64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // DON'T truncate here - let dispatcher handle it

    // Cleanup
    unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };

    Ok(final_len)
}

/// CUB-based GPU sorting for i64 arrays
pub unsafe fn sort_i64_cub(
    values: &mut Vec<i64>,
    len: usize,
    ascending: bool,
    dedup: bool,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut i64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<i64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut i64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Call CUB DeviceRadixSort
    let result = unsafe { cub_device_radix_sort_i64(gpu_values, len, ascending) };

    if result != 0 {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB i64 sort failed: {}",
            result
        )));
    }

    let final_len = if dedup {
        // Allocate temporary buffer for unique operation
        let mut gpu_temp: *mut i64 = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_temp as *mut *mut i64 as *mut *mut std::ffi::c_void,
                size_bytes,
            )
        } != 0
        {
            unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
            return Err(crate::types::HwxError::Internal(
                "GPU temp allocation failed".to_string(),
            ));
        }

        // Allocate GPU memory for result count
        let mut gpu_count: *mut usize = ptr::null_mut();
        if unsafe {
            cudaMalloc(
                &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU count allocation failed".to_string(),
            ));
        }

        // Call CUB DeviceSelect::Unique
        let dedup_result = unsafe { cub_device_unique_i64(gpu_values, gpu_temp, len, gpu_count) };

        if dedup_result != 0 {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(format!(
                "CUB i64 unique failed: {}",
                dedup_result
            )));
        }

        // Get the final count
        let mut final_count: usize = 0;
        if unsafe {
            cudaMemcpy(
                &mut final_count as *mut usize as *mut std::ffi::c_void,
                gpu_count as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU i64 count copy failed".to_string(),
            ));
        }

        // Copy unique result back to original buffer
        if unsafe {
            cudaMemcpy(
                gpu_values as *mut std::ffi::c_void,
                gpu_temp as *const std::ffi::c_void,
                final_count * std::mem::size_of::<i64>(),
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        } != 0
        {
            unsafe {
                cudaFree(gpu_values as *mut std::ffi::c_void);
                cudaFree(gpu_temp as *mut std::ffi::c_void);
                cudaFree(gpu_count as *mut std::ffi::c_void);
            };
            return Err(crate::types::HwxError::Internal(
                "GPU i64 unique copy failed".to_string(),
            ));
        }

        // Cleanup temp buffers
        unsafe {
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };

        final_count
    } else {
        len
    };

    // Copy result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_values as *const std::ffi::c_void,
            final_len * std::mem::size_of::<i64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // DON'T truncate here - let dispatcher handle it

    // Cleanup
    unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };

    Ok(final_len)
}

/// CUB-based GPU deduplication for u32 arrays
pub unsafe fn dedup_sorted_u32_cub(
    values: &mut Vec<u32>,
    len: usize,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate temporary buffer for unique operation
    let mut gpu_temp: *mut u32 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_temp as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU temp allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result count
    let mut gpu_count: *mut usize = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU count allocation failed".to_string(),
        ));
    }

    // Call CUB DeviceSelect::Unique
    let result = unsafe { cub_device_unique_u32(gpu_values, gpu_temp, len, gpu_count) };

    if result != 0 {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB unique failed: {}",
            result
        )));
    }

    // Get the final count
    let mut final_count: usize = 0;
    if unsafe {
        cudaMemcpy(
            &mut final_count as *mut usize as *mut std::ffi::c_void,
            gpu_count as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU count copy failed".to_string(),
        ));
    }

    // Copy unique result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_temp as *const std::ffi::c_void,
            final_count * std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_values as *mut std::ffi::c_void);
        cudaFree(gpu_temp as *mut std::ffi::c_void);
        cudaFree(gpu_count as *mut std::ffi::c_void);
    };

    Ok(final_count)
}

/// CUB-based GPU deduplication for f64 arrays
pub unsafe fn dedup_sorted_f64_cub(
    values: &mut Vec<f64>,
    len: usize,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate temporary buffer for unique operation
    let mut gpu_temp: *mut f64 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_temp as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU temp allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result count
    let mut gpu_count: *mut usize = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU count allocation failed".to_string(),
        ));
    }

    // Call CUB DeviceSelect::Unique
    let result = unsafe { cub_device_unique_f64(gpu_values, gpu_temp, len, gpu_count) };

    if result != 0 {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB f64 unique failed: {}",
            result
        )));
    }

    // Get the final count
    let mut final_count: usize = 0;
    if unsafe {
        cudaMemcpy(
            &mut final_count as *mut usize as *mut std::ffi::c_void,
            gpu_count as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU f64 count copy failed".to_string(),
        ));
    }

    // Copy unique result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_temp as *const std::ffi::c_void,
            final_count * std::mem::size_of::<f64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // DON'T truncate here - let dispatcher handle it

    // Cleanup
    unsafe {
        cudaFree(gpu_values as *mut std::ffi::c_void);
        cudaFree(gpu_temp as *mut std::ffi::c_void);
        cudaFree(gpu_count as *mut std::ffi::c_void);
    };

    Ok(final_count)
}

/// CUB-based GPU deduplication for u64 arrays
pub unsafe fn dedup_sorted_u64_cub(
    values: &mut Vec<u64>,
    len: usize,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut u64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate temporary buffer for unique operation
    let mut gpu_temp: *mut u64 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_temp as *mut *mut u64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU temp allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result count
    let mut gpu_count: *mut usize = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU count allocation failed".to_string(),
        ));
    }

    // Call CUB DeviceSelect::Unique
    let result = unsafe { cub_device_unique_u64(gpu_values, gpu_temp, len, gpu_count) };

    if result != 0 {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB u64 unique failed: {}",
            result
        )));
    }

    // Get the final count
    let mut final_count: usize = 0;
    if unsafe {
        cudaMemcpy(
            &mut final_count as *mut usize as *mut std::ffi::c_void,
            gpu_count as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU u64 count copy failed".to_string(),
        ));
    }

    // Copy unique result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_temp as *const std::ffi::c_void,
            final_count * std::mem::size_of::<u64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // DON'T truncate here - let dispatcher handle it

    // Cleanup
    unsafe {
        cudaFree(gpu_values as *mut std::ffi::c_void);
        cudaFree(gpu_temp as *mut std::ffi::c_void);
        cudaFree(gpu_count as *mut std::ffi::c_void);
    };

    Ok(final_count)
}

/// CUB-based GPU deduplication for i64 arrays
pub unsafe fn dedup_sorted_i64_cub(
    values: &mut Vec<i64>,
    len: usize,
) -> Result<usize, crate::types::HwxError> {
    // Allocate GPU memory
    let mut gpu_values: *mut i64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<i64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut i64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate temporary buffer for unique operation
    let mut gpu_temp: *mut i64 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_temp as *mut *mut i64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU temp allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result count
    let mut gpu_count: *mut usize = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_count as *mut *mut usize as *mut *mut std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU count allocation failed".to_string(),
        ));
    }

    // Call CUB DeviceSelect::Unique
    let result = unsafe { cub_device_unique_i64(gpu_values, gpu_temp, len, gpu_count) };

    if result != 0 {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB i64 unique failed: {}",
            result
        )));
    }

    // Get the final count
    let mut final_count: usize = 0;
    if unsafe {
        cudaMemcpy(
            &mut final_count as *mut usize as *mut std::ffi::c_void,
            gpu_count as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU i64 count copy failed".to_string(),
        ));
    }

    // Copy unique result back to CPU
    if unsafe {
        cudaMemcpy(
            values.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_temp as *const std::ffi::c_void,
            final_count * std::mem::size_of::<i64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_temp as *mut std::ffi::c_void);
            cudaFree(gpu_count as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // DON'T truncate here - let dispatcher handle it

    // Cleanup
    unsafe {
        cudaFree(gpu_values as *mut std::ffi::c_void);
        cudaFree(gpu_temp as *mut std::ffi::c_void);
        cudaFree(gpu_count as *mut std::ffi::c_void);
    };

    Ok(final_count)
}

/// CUB-based GPU sorting for u32 indices by u64 values using SortPairs
pub unsafe fn sort_u32_by_u64_cub(
    indices: &mut [u32],
    values: &mut [u64],
    len: usize,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    // Allocate GPU memory for indices
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let indices_size = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut std::ffi::c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for indices failed".to_string(),
        ));
    }

    // Allocate GPU memory for values
    let mut gpu_values: *mut u64 = ptr::null_mut();
    let values_size = len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u64 as *mut *mut std::ffi::c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut std::ffi::c_void,
            indices.as_ptr() as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for indices failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Call CUB SortPairs
    let result = unsafe { cub_sort_pairs_u32_by_u64(gpu_indices, gpu_values, len, ascending) };

    if result != 0 {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB sort pairs failed: {}",
            result
        )));
    }

    // Copy results back to CPU
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_indices as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back for indices failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_indices as *mut std::ffi::c_void);
        cudaFree(gpu_values as *mut std::ffi::c_void);
    };

    Ok(())
}

/// CUB-based GPU sorting for u32 indices by f64 values using SortPairs
pub unsafe fn sort_u32_by_f64_cub(
    indices: &mut [u32],
    values: &mut [f64],
    len: usize,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    // Allocate GPU memory for indices
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let indices_size = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut std::ffi::c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for indices failed".to_string(),
        ));
    }

    // Allocate GPU memory for values
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let values_size = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut std::ffi::c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut std::ffi::c_void,
            indices.as_ptr() as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for indices failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Call CUB SortPairs
    let result = unsafe { cub_sort_pairs_u32_by_f64(gpu_indices, gpu_values, len, ascending) };

    if result != 0 {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB f64 sort pairs failed: {}",
            result
        )));
    }

    // Copy results back to CPU
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_indices as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back for indices failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_indices as *mut std::ffi::c_void);
        cudaFree(gpu_values as *mut std::ffi::c_void);
    };

    Ok(())
}

/// CUB-based GPU sorting for u32 indices by i64 values using SortPairs
pub unsafe fn sort_u32_by_i64_cub(
    indices: &mut [u32],
    values: &mut [i64],
    len: usize,
    ascending: bool,
) -> Result<(), crate::types::HwxError> {
    // Allocate GPU memory for indices
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let indices_size = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut std::ffi::c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for indices failed".to_string(),
        ));
    }

    // Allocate GPU memory for values
    let mut gpu_values: *mut i64 = ptr::null_mut();
    let values_size = len * std::mem::size_of::<i64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut i64 as *mut *mut std::ffi::c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut std::ffi::c_void,
            indices.as_ptr() as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for indices failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Call CUB SortPairs
    let result = unsafe { cub_sort_pairs_u32_by_i64(gpu_indices, gpu_values, len, ascending) };

    if result != 0 {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(format!(
            "CUB i64 sort pairs failed: {}",
            result
        )));
    }

    // Copy results back to CPU
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_indices as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_indices as *mut std::ffi::c_void);
            cudaFree(gpu_values as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back for indices failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_indices as *mut std::ffi::c_void);
        cudaFree(gpu_values as *mut std::ffi::c_void);
    };

    Ok(())
}

/// Helper for GPU operations that convert u64 to f64 (for time functions)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u64_to_f64<F>(
    timestamps: &[u64],
    result: &mut [f64],
    len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*const u64, *mut f64, usize),
{
    use std::ptr;

    // Allocate GPU memory for input and output
    let mut gpu_timestamps: *mut u64 = ptr::null_mut();
    let mut gpu_result: *mut f64 = ptr::null_mut();

    let timestamps_size = len * std::mem::size_of::<u64>();
    let result_size = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_timestamps as *mut _ as *mut *mut std::ffi::c_void,
            timestamps_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut _ as *mut *mut std::ffi::c_void,
            result_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_timestamps as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_timestamps as *mut std::ffi::c_void,
            timestamps.as_ptr() as *const std::ffi::c_void,
            timestamps_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_timestamps as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Run the computation
    compute_fn(gpu_timestamps, gpu_result, len);

    // Copy result back from GPU
    if unsafe {
        cudaMemcpy(
            result.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            result_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_timestamps as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_timestamps as *mut std::ffi::c_void);
        cudaFree(gpu_result as *mut std::ffi::c_void);
    };

    Ok(())
}

/// Helper for checking if array is sorted on GPU
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_check<F>(
    input: &[u32],
    len: usize,
    compute_fn: F,
) -> Result<bool, crate::types::HwxError>
where
    F: FnOnce(*const u32, usize, *mut u8),
{
    // Allocate GPU memory for input
    let mut gpu_input: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for result (single byte)
    let mut gpu_result: *mut u8 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut u8 as *mut *mut std::ffi::c_void,
            1,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Initialize result to 1 (true)
    let initial_value: u8 = 1;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut std::ffi::c_void,
            &initial_value as *const u8 as *const std::ffi::c_void,
            1,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result init failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Check on GPU
    compute_fn(gpu_input, len, gpu_result);

    // Copy result back
    let mut host_result: u8 = 0;
    if unsafe {
        cudaMemcpy(
            &mut host_result as *mut u8 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            1,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_result as *mut std::ffi::c_void);
    }

    Ok(host_result != 0)
}

/// Helper for intersect operations on GPU
#[allow(clippy::too_many_arguments)]
#[cfg(has_cuda)]
pub unsafe fn with_gpu_intersect_u32<F>(
    a: &mut Vec<u32>,
    b: &[u32],
    len_a: usize,
    len_b: usize,
    max_size: usize,
    dedup: bool,
    ascending: bool,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const u32, usize, usize, usize, bool, bool) -> usize,
{
    // Allocate GPU memory
    let mut gpu_a: *mut u32 = ptr::null_mut();
    let mut gpu_b: *mut u32 = ptr::null_mut();
    let size_a = len_a * std::mem::size_of::<u32>();
    let size_b = len_b * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_a as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_a,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_b as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_b,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy inputs to GPU
    if unsafe {
        cudaMemcpy(
            gpu_a as *mut std::ffi::c_void,
            a.as_ptr() as *const std::ffi::c_void,
            size_a,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_b as *mut std::ffi::c_void,
            b.as_ptr() as *const std::ffi::c_void,
            size_b,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute intersection on GPU
    let count = compute_fn(gpu_a, gpu_b, len_a, len_b, max_size, dedup, ascending);

    // Copy result back to CPU
    if unsafe {
        cudaMemcpy(
            a.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_a as *const std::ffi::c_void,
            count * std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_a as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_b as *mut std::ffi::c_void) };

    a.truncate(count);

    Ok(())
}

/// Helper for union operations on GPU
#[cfg(has_cuda)]
pub unsafe fn with_gpu_union_u32<F>(
    arrays: &[&[u32]],
    output: &mut Vec<u32>,
    max_size: usize,
    ascending: bool,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*const *const u32, *const usize, usize, *mut u32, usize, bool) -> usize,
{
    if arrays.is_empty() {
        // Acquire the GPU launch mutex to serialize ALL GPU operations
        let _gpu_lock = GPU_LAUNCH_MUTEX.lock().unwrap();

        return Ok(());
    }

    // Allocate GPU memory for array pointers and sizes
    let num_arrays = arrays.len();
    let mut gpu_arrays = Vec::with_capacity(num_arrays);
    let mut array_sizes = Vec::with_capacity(num_arrays);

    // Allocate GPU memory for each array
    for arr in arrays {
        let size = std::mem::size_of_val(*arr);
        array_sizes.push(arr.len());
        let mut gpu_ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        if unsafe { cudaMalloc(&mut gpu_ptr, size) } != 0 {
            // Cleanup already allocated
            for ptr in &gpu_arrays {
                unsafe { cudaFree(*ptr as *mut std::ffi::c_void) };
            }
            return Err(crate::types::HwxError::Internal(
                "GPU memory allocation failed".to_string(),
            ));
        }

        // Copy array to GPU
        if unsafe {
            cudaMemcpy(
                gpu_ptr,
                arr.as_ptr() as *const std::ffi::c_void,
                size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        } != 0
        {
            // Cleanup
            unsafe { cudaFree(gpu_ptr) };
            for ptr in &gpu_arrays {
                unsafe { cudaFree(*ptr as *mut std::ffi::c_void) };
            }
            return Err(crate::types::HwxError::Internal(
                "GPU memory copy failed".to_string(),
            ));
        }

        gpu_arrays.push(gpu_ptr as *const u32);
    }

    // Allocate GPU memory for output
    let output_size = max_size * std::mem::size_of::<u32>();
    let mut gpu_output: *mut std::ffi::c_void = std::ptr::null_mut();

    if unsafe { cudaMalloc(&mut gpu_output, output_size) } != 0 {
        for ptr in &gpu_arrays {
            unsafe { cudaFree(*ptr as *mut std::ffi::c_void) };
        }
        return Err(crate::types::HwxError::Internal(
            "GPU output allocation failed".to_string(),
        ));
    }

    // Call the GPU compute function
    let count = compute_fn(
        gpu_arrays.as_ptr(),
        array_sizes.as_ptr(),
        num_arrays,
        gpu_output as *mut u32,
        max_size,
        ascending,
    );

    // Copy result back
    output.resize(count, 0);
    if unsafe {
        cudaMemcpy(
            output.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_output,
            count * std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        // Cleanup
        for ptr in &gpu_arrays {
            unsafe { cudaFree(*ptr as *mut std::ffi::c_void) };
        }
        unsafe { cudaFree(gpu_output) };
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup GPU memory
    for ptr in &gpu_arrays {
        unsafe { cudaFree(*ptr as *mut std::ffi::c_void) };
    }
    unsafe { cudaFree(gpu_output) };

    Ok(())
}

/// Helper for string sorting on GPU
#[cfg(has_cuda)]
pub unsafe fn with_gpu_strings_sort<F>(
    strings: *const u8,       // Flattened string data
    string_offsets: *mut u32, // Offsets into string data
    indices: *mut u32,        // Sort indices
    total_bytes: usize,       // Total bytes in strings
    num_strings: usize,       // Number of strings
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*const u8, *mut u32, *mut u32, usize, usize),
{
    // Allocate GPU memory
    let mut gpu_strings: *mut u8 = ptr::null_mut();
    let mut gpu_offsets: *mut u32 = ptr::null_mut();
    let mut gpu_indices: *mut u32 = ptr::null_mut();

    let strings_size = total_bytes * std::mem::size_of::<u8>();
    let offsets_size = (num_strings + 1) * std::mem::size_of::<u32>();
    let indices_size = num_strings * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_strings as *mut *mut u8 as *mut *mut std::ffi::c_void,
            strings_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_offsets as *mut *mut u32 as *mut *mut std::ffi::c_void,
            offsets_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut std::ffi::c_void,
            indices_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_offsets as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy to GPU
    if unsafe {
        cudaMemcpy(
            gpu_strings as *mut std::ffi::c_void,
            strings as *const std::ffi::c_void,
            strings_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_offsets as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_offsets as *mut std::ffi::c_void,
            string_offsets as *const std::ffi::c_void,
            offsets_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_offsets as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut std::ffi::c_void,
            indices as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_offsets as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Sort on GPU
    compute_fn(
        gpu_strings,
        gpu_offsets,
        gpu_indices,
        total_bytes,
        num_strings,
    );

    // Copy back sorted indices
    if unsafe {
        cudaMemcpy(
            indices as *mut std::ffi::c_void,
            gpu_indices as *const std::ffi::c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_offsets as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_strings as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_offsets as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_indices as *mut std::ffi::c_void) };

    Ok(())
}

/// Helper for single-array GPU operations with u32 arrays (returns min/max tuple)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_minmax<F>(
    input: &[u32],
    len: usize,
    compute_fn: F,
) -> Result<(u32, u32), crate::types::HwxError>
where
    F: FnOnce(*const u32, usize, *mut u32, *mut u32),
{
    // Allocate GPU memory for input
    let mut gpu_input: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u32 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for min and max results
    let mut gpu_min: *mut u32 = ptr::null_mut();
    let mut gpu_max: *mut u32 = ptr::null_mut();

    if unsafe {
        cudaMalloc(
            &mut gpu_min as *mut *mut u32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<u32>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for min failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_max as *mut *mut u32 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<u32>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for max failed".to_string(),
        ));
    }

    // Initialize min/max values in GPU memory
    let init_min = u32::MAX;
    let init_max = 0u32;

    if unsafe {
        cudaMemcpy(
            gpu_min as *mut std::ffi::c_void,
            &init_min as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for min init failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_max as *mut std::ffi::c_void,
            &init_max as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for max init failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Ensure all memory transfers are complete before kernel launch
    if unsafe { cudaDeviceSynchronize() } != 0 {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU synchronization failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, len, gpu_min, gpu_max);

    // Copy results back to host
    let mut min_result = u32::MAX;
    let mut max_result = 0u32;

    if unsafe {
        cudaMemcpy(
            &mut min_result as *mut u32 as *mut std::ffi::c_void,
            gpu_min as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for min result failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            &mut max_result as *mut u32 as *mut std::ffi::c_void,
            gpu_max as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for max result failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_min as *mut std::ffi::c_void);
        cudaFree(gpu_max as *mut std::ffi::c_void);
    };

    Ok((min_result, max_result))
}

/// CUB-based variance calculation for f64 arrays
pub unsafe fn stdvar_f64_cub(values: &[f64], len: usize) -> Result<f64, crate::types::HwxError> {
    // Allocate GPU memory for input
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut std::ffi::c_void,
            values.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut f64 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut f64 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<f64>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Call CUB variance function
    let result_code = unsafe { cub_device_variance_f64(gpu_values, len, gpu_result) };

    if result_code != 0 {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(format!(
            "CUB variance failed: {}",
            result_code
        )));
    }

    // Copy result back to host
    let mut result: f64 = 0.0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut f64 as *mut std::ffi::c_void,
            gpu_result as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_values as *mut std::ffi::c_void);
            cudaFree(gpu_result as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_values as *mut std::ffi::c_void);
        cudaFree(gpu_result as *mut std::ffi::c_void);
    }

    Ok(result)
}

/// Helper for stdvar GPU operation which needs extra workspace
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64_stdvar<F>(
    input: &[f64],
    len: usize,
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, usize, *mut f64),
{
    // Allocate GPU memory for input
    let mut gpu_input: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate workspace for stdvar - needs 36 bytes minimum
    // The PTX kernel uses: offset 0 for result, offset 8 for sum, offset 16 for count, offset 24 for variance sum, offset 32 for block counter
    let mut gpu_workspace: *mut f64 = ptr::null_mut();
    let workspace_size = 48; // Allocate 48 bytes to cover all offsets (6 * 8 bytes)

    if unsafe {
        cudaMalloc(
            &mut gpu_workspace as *mut *mut f64 as *mut *mut std::ffi::c_void,
            workspace_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU workspace allocation failed".to_string(),
        ));
    }

    // Initialize workspace to zero
    if unsafe { cudaMemset(gpu_workspace as *mut std::ffi::c_void, 0, workspace_size) } != 0 {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_workspace as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU workspace init failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, len, gpu_workspace);

    // Synchronize to ensure kernel completes
    if unsafe { cudaDeviceSynchronize() } != 0 {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_workspace as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU synchronization failed".to_string(),
        ));
    }

    // Copy result back (first f64 in workspace)
    let mut result: f64 = 0.0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut f64 as *mut std::ffi::c_void,
            gpu_workspace as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_workspace as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_workspace as *mut std::ffi::c_void);
    }

    Ok(result)
}

/// Helper for single-array GPU operations with f64 arrays (returns min/max tuple)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64_minmax<F>(
    input: &[f64],
    len: usize,
    compute_fn: F,
) -> Result<(f64, f64), crate::types::HwxError>
where
    F: FnOnce(*const f64, usize, *mut f64, *mut f64),
{
    // Allocate GPU memory for input
    let mut gpu_input: *mut f64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut f64 as *mut *mut std::ffi::c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for min and max results
    let mut gpu_min: *mut f64 = ptr::null_mut();
    let mut gpu_max: *mut f64 = ptr::null_mut();

    if unsafe {
        cudaMalloc(
            &mut gpu_min as *mut *mut f64 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<f64>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for min failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_max as *mut *mut f64 as *mut *mut std::ffi::c_void,
            std::mem::size_of::<f64>(),
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for max failed".to_string(),
        ));
    }

    // Initialize min and max on GPU
    let init_min = f64::INFINITY;
    let init_max = f64::NEG_INFINITY;

    if unsafe {
        cudaMemcpy(
            gpu_min as *mut std::ffi::c_void,
            &init_min as *const f64 as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory init failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_max as *mut std::ffi::c_void,
            &init_max as *const f64 as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory init failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Compute on GPU
    compute_fn(gpu_input, len, gpu_min, gpu_max);

    // Copy results back from GPU
    let mut min_result = f64::INFINITY;
    let mut max_result = f64::NEG_INFINITY;

    if unsafe {
        cudaMemcpy(
            &mut min_result as *mut f64 as *mut std::ffi::c_void,
            gpu_min as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            &mut max_result as *mut f64 as *mut std::ffi::c_void,
            gpu_max as *const std::ffi::c_void,
            std::mem::size_of::<f64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
            cudaFree(gpu_min as *mut std::ffi::c_void);
            cudaFree(gpu_max as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_min as *mut std::ffi::c_void);
        cudaFree(gpu_max as *mut std::ffi::c_void);
    }

    Ok((min_result, max_result))
}

/// Helper for reduction operations that need GPU memory for results
/// This allocates GPU memory for partial results from each block
#[cfg(has_cuda)]
pub unsafe fn with_gpu_reduction_f64<F>(
    input: &[f64],
    len: usize,
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, usize, *mut f64),
{
    // Get launch configuration to know how many blocks we'll have
    let (blocks, _threads) = LaunchConfig::reduction();

    // Allocate GPU memory for input
    let mut gpu_input: *mut f64 = ptr::null_mut();
    let input_size = len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut f64 as *mut *mut std::ffi::c_void,
            input_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Allocate GPU memory for partial results (one per block)
    let mut gpu_results: *mut f64 = ptr::null_mut();
    let results_size = (blocks as usize) * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_results as *mut *mut f64 as *mut *mut std::ffi::c_void,
            results_size,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut std::ffi::c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Initialize all partial results with NEG_INFINITY for max
    let mut host_results = vec![f64::NEG_INFINITY; blocks as usize];
    unsafe {
        cudaMemcpy(
            gpu_results as *mut std::ffi::c_void,
            host_results.as_ptr() as *const std::ffi::c_void,
            results_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
    }

    // Copy input to GPU
    unsafe {
        cudaMemcpy(
            gpu_input as *mut std::ffi::c_void,
            input.as_ptr() as *const std::ffi::c_void,
            input_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
    }

    // Run the kernel - each block will write its partial result
    compute_fn(gpu_input, len, gpu_results);

    // Copy partial results back
    unsafe {
        cudaMemcpy(
            host_results.as_mut_ptr() as *mut std::ffi::c_void,
            gpu_results as *const std::ffi::c_void,
            results_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );
    }

    // Cleanup GPU memory
    unsafe {
        cudaFree(gpu_input as *mut std::ffi::c_void);
        cudaFree(gpu_results as *mut std::ffi::c_void);
    }

    // Final reduction on host - find max of all partial results
    let mut result = f64::NEG_INFINITY;
    for partial in host_results {
        if partial > result {
            result = partial;
        }
    }

    Ok(result)
}

/// Helper for field matching operations on GPU (doc_ids filtering)
#[allow(clippy::too_many_arguments)]
#[cfg(has_cuda)]
pub unsafe fn with_gpu_field_match<F>(
    doc_ids: &mut [u32],
    field_data: *const u8,
    pattern: *const u8,
    doc_ids_len: usize,
    field_len: usize,
    pattern_len: usize,
    max_size: usize,
    compute_fn: F,
) -> Result<usize, crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const u8, *const u8, usize, usize, usize, usize) -> usize,
{
    // Allocate GPU memory
    let mut gpu_doc_ids: *mut u32 = ptr::null_mut();
    let mut gpu_field: *mut u8 = ptr::null_mut();
    let mut gpu_pattern: *mut u8 = ptr::null_mut();

    let doc_ids_size = doc_ids_len * std::mem::size_of::<u32>();
    let field_size = field_len * std::mem::size_of::<u8>();
    let pattern_size = pattern_len * std::mem::size_of::<u8>();

    if unsafe {
        cudaMalloc(
            &mut gpu_doc_ids as *mut *mut u32 as *mut *mut std::ffi::c_void,
            doc_ids_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_field as *mut *mut u8 as *mut *mut std::ffi::c_void,
            field_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_doc_ids as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_pattern as *mut *mut u8 as *mut *mut std::ffi::c_void,
            pattern_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_doc_ids as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_field as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy to GPU
    if unsafe {
        cudaMemcpy(
            gpu_doc_ids as *mut std::ffi::c_void,
            doc_ids.as_ptr() as *const std::ffi::c_void,
            doc_ids_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_doc_ids as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_field as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_field as *mut std::ffi::c_void,
            field_data as *const std::ffi::c_void,
            field_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_doc_ids as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_field as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_pattern as *mut std::ffi::c_void,
            pattern as *const std::ffi::c_void,
            pattern_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_doc_ids as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_field as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Process on GPU
    let count = compute_fn(
        gpu_doc_ids,
        gpu_field,
        gpu_pattern,
        doc_ids_len,
        field_len,
        pattern_len,
        max_size,
    );

    // Copy back filtered doc_ids
    if count > 0
        && unsafe {
            cudaMemcpy(
                doc_ids.as_mut_ptr() as *mut std::ffi::c_void,
                gpu_doc_ids as *const std::ffi::c_void,
                count * std::mem::size_of::<u32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        } != 0
    {
        unsafe { cudaFree(gpu_doc_ids as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_field as *mut std::ffi::c_void) };
        unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_doc_ids as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_field as *mut std::ffi::c_void) };
    unsafe { cudaFree(gpu_pattern as *mut std::ffi::c_void) };

    Ok(count)
}

/// Helper for u32 array GPU operations that return u64 (for sum reductions)
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_to_u64<F>(
    input: &[u32],
    len: usize,
    compute_fn: F,
) -> Result<u64, crate::types::HwxError>
where
    F: FnOnce(*const u32, usize, *mut u64),
{
    // Ensure CUDA context is current for this thread
    ensure_context_current_for_thread()?;
    // Allocate GPU memory for input
    let mut gpu_input: *mut u32 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u32>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u32 as *mut *mut c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut c_void,
            input.as_ptr() as *const c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut u64 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut u64 as *mut *mut c_void,
            std::mem::size_of::<u64>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Initialize to 0 by copying from host
    let init_value: u64 = 0;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut c_void,
            &init_value as *const u64 as *const c_void,
            std::mem::size_of::<u64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut c_void);
            cudaFree(gpu_result as *mut c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU init failed".to_string(),
        ));
    }

    // Compute on GPU - pass GPU result pointer
    compute_fn(gpu_input as *const u32, len, gpu_result);

    // Copy result back from GPU
    let mut result: u64 = 0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut u64 as *mut c_void,
            gpu_result as *const c_void,
            std::mem::size_of::<u64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut c_void);
            cudaFree(gpu_result as *mut c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut c_void);
        cudaFree(gpu_result as *mut c_void);
    }

    Ok(result)
}

/// Helper for GPU operations that read u64 array and write single u64 result
/// Used for reduction operations like sum
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u64_to_u64<F>(
    input: &[u64],
    len: usize,
    compute_fn: F,
) -> Result<u64, crate::types::HwxError>
where
    F: FnOnce(*const u64, usize, *mut u64),
{
    // Allocate GPU memory for input
    let mut gpu_input: *mut u64 = ptr::null_mut();
    let size_bytes = len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_input as *mut *mut u64 as *mut *mut c_void,
            size_bytes,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy input to GPU
    if unsafe {
        cudaMemcpy(
            gpu_input as *mut c_void,
            input.as_ptr() as *const c_void,
            size_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Allocate GPU memory for result
    let mut gpu_result: *mut u64 = ptr::null_mut();
    if unsafe {
        cudaMalloc(
            &mut gpu_result as *mut *mut u64 as *mut *mut c_void,
            std::mem::size_of::<u64>(),
        )
    } != 0
    {
        unsafe { cudaFree(gpu_input as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU result allocation failed".to_string(),
        ));
    }

    // Initialize to 0 by copying from host
    let init_value: u64 = 0;
    if unsafe {
        cudaMemcpy(
            gpu_result as *mut c_void,
            &init_value as *const u64 as *const c_void,
            std::mem::size_of::<u64>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut c_void);
            cudaFree(gpu_result as *mut c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU init failed".to_string(),
        ));
    }

    // Compute on GPU - pass GPU result pointer
    compute_fn(gpu_input as *const u64, len, gpu_result);

    // Copy result back from GPU
    let mut result: u64 = 0;
    if unsafe {
        cudaMemcpy(
            &mut result as *mut u64 as *mut c_void,
            gpu_result as *const c_void,
            std::mem::size_of::<u64>(),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe {
            cudaFree(gpu_input as *mut c_void);
            cudaFree(gpu_result as *mut c_void);
        }
        return Err(crate::types::HwxError::Internal(
            "GPU result copy failed".to_string(),
        ));
    }

    // Cleanup
    unsafe {
        cudaFree(gpu_input as *mut c_void);
        cudaFree(gpu_result as *mut c_void);
    }

    Ok(result)
}

/// Helper for GPU operations with mutable u32 array (indices) and readonly f64 array (values)
/// Used for sorting u32 indices by f64 values
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_by_f64<F>(
    indices: &mut [u32],
    values: &[f64],
    ascending: bool,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const f64, usize, usize, bool),
{
    let indices_len = indices.len();
    let values_len = values.len();

    // Allocate GPU memory for indices
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let indices_size = std::mem::size_of_val(indices);
    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for indices failed".to_string(),
        ));
    }

    // Allocate GPU memory for values
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let values_size = std::mem::size_of_val(values);
    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Copy indices to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut c_void,
            indices.as_ptr() as *const c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for indices failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Execute GPU computation with ascending parameter
    compute_fn(
        gpu_indices,
        gpu_values as *const f64,
        indices_len,
        values_len,
        ascending,
    );

    // Synchronize
    // Removed sync

    // Copy sorted indices back
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut c_void,
            gpu_indices as *const c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Free GPU memory
    unsafe { cudaFree(gpu_indices as *mut c_void) };
    unsafe { cudaFree(gpu_values as *mut c_void) };

    Ok(())
}

/// Helper for GPU operations with mutable u32 array (indices) and readonly u64 array (values)
/// Used for sorting u32 indices by u64 values
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_by_u64<F>(
    indices: &mut [u32],
    values: &[u64],
    ascending: bool,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const u64, usize, usize, bool),
{
    let indices_len = indices.len();
    let values_len = values.len();

    // Allocate GPU memory for indices
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let indices_size = std::mem::size_of_val(indices);
    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for indices failed".to_string(),
        ));
    }

    // Allocate GPU memory for values
    let mut gpu_values: *mut u64 = ptr::null_mut();
    let values_size = std::mem::size_of_val(values);
    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Copy indices to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut c_void,
            indices.as_ptr() as *const c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for indices failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Execute GPU computation with ascending parameter
    compute_fn(
        gpu_indices,
        gpu_values as *const u64,
        indices_len,
        values_len,
        ascending,
    );

    // Synchronize
    // Removed sync

    // Copy sorted indices back
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut c_void,
            gpu_indices as *const c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Free GPU memory
    unsafe { cudaFree(gpu_indices as *mut c_void) };
    unsafe { cudaFree(gpu_values as *mut c_void) };

    Ok(())
}

/// Helper for GPU operations with mutable u32 array (indices) and readonly i64 array (values)
/// Used for sorting u32 indices by i64 values
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_by_i64<F>(
    indices: &mut [u32],
    values: &[i64],
    ascending: bool,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const i64, usize, usize, bool),
{
    let indices_len = indices.len();
    let values_len = values.len();

    // Allocate GPU memory for indices
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let indices_size = std::mem::size_of_val(indices);
    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for indices failed".to_string(),
        ));
    }

    // Allocate GPU memory for values
    let mut gpu_values: *mut i64 = ptr::null_mut();
    let values_size = std::mem::size_of_val(values);
    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut i64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Copy indices to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut c_void,
            indices.as_ptr() as *const c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for indices failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Execute GPU computation with ascending parameter
    compute_fn(
        gpu_indices,
        gpu_values as *const i64,
        indices_len,
        values_len,
        ascending,
    );

    // Synchronize
    // Removed sync

    // Copy sorted indices back
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut c_void,
            gpu_indices as *const c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Free GPU memory
    unsafe { cudaFree(gpu_indices as *mut c_void) };
    unsafe { cudaFree(gpu_values as *mut c_void) };

    Ok(())
}

/// Helper for GPU operations with two readonly arrays (f64 and u64) returning f64
/// Used for deriv_f64 computation
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64_u64_to_f64<F>(
    values: &[f64],
    timestamps: &[u64],
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, *const u64, usize) -> f64,
{
    let len = values.len().min(timestamps.len());

    // Allocate GPU memory for values
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let values_size = len * std::mem::size_of::<f64>();
    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Allocate GPU memory for timestamps
    let mut gpu_timestamps: *mut u64 = ptr::null_mut();
    let timestamps_size = len * std::mem::size_of::<u64>();
    if unsafe {
        cudaMalloc(
            &mut gpu_timestamps as *mut *mut u64 as *mut *mut c_void,
            timestamps_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for timestamps failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Copy timestamps to GPU
    if unsafe {
        cudaMemcpy(
            gpu_timestamps as *mut c_void,
            timestamps.as_ptr() as *const c_void,
            timestamps_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for timestamps failed".to_string(),
        ));
    }

    // Execute GPU computation
    let result = compute_fn(gpu_values as *const f64, gpu_timestamps as *const u64, len);

    // Free GPU memory
    unsafe { cudaFree(gpu_values as *mut c_void) };
    unsafe { cudaFree(gpu_timestamps as *mut c_void) };

    Ok(result)
}

/// Helper for GPU operations with two readonly f64 arrays returning f64
/// Used for predict_linear_f64 computation
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64_f64_to_f64<F>(
    values: &[f64],
    timestamps: &[f64],
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, *const f64, usize) -> f64,
{
    let len = values.len().min(timestamps.len());

    // Allocate GPU memory for values
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let values_size = len * std::mem::size_of::<f64>();
    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Allocate GPU memory for timestamps
    let mut gpu_timestamps: *mut f64 = ptr::null_mut();
    let timestamps_size = len * std::mem::size_of::<f64>();
    if unsafe {
        cudaMalloc(
            &mut gpu_timestamps as *mut *mut f64 as *mut *mut c_void,
            timestamps_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for timestamps failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Copy timestamps to GPU
    if unsafe {
        cudaMemcpy(
            gpu_timestamps as *mut c_void,
            timestamps.as_ptr() as *const c_void,
            timestamps_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for timestamps failed".to_string(),
        ));
    }

    // Execute GPU computation
    let result = compute_fn(gpu_values as *const f64, gpu_timestamps as *const f64, len);

    // Free GPU memory
    unsafe { cudaFree(gpu_values as *mut c_void) };
    unsafe { cudaFree(gpu_timestamps as *mut c_void) };

    Ok(result)
}

/// Helper for GPU operations with two readonly f64 arrays and a scalar, returning f64
/// Used for predict_linear_f64 computation with predict_time parameter
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64_f64_scalar_to_f64<F>(
    values: &[f64],
    timestamps: &[f64],
    scalar: f64,
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, *const f64, f64, usize) -> f64,
{
    let len = values.len().min(timestamps.len());

    // Allocate GPU memory for values
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let values_size = len * std::mem::size_of::<f64>();
    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for values failed".to_string(),
        ));
    }

    // Allocate GPU memory for timestamps
    let mut gpu_timestamps: *mut f64 = ptr::null_mut();
    let timestamps_size = len * std::mem::size_of::<f64>();
    if unsafe {
        cudaMalloc(
            &mut gpu_timestamps as *mut *mut f64 as *mut *mut c_void,
            timestamps_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation for timestamps failed".to_string(),
        ));
    }

    // Copy values to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for values failed".to_string(),
        ));
    }

    // Copy timestamps to GPU
    if unsafe {
        cudaMemcpy(
            gpu_timestamps as *mut c_void,
            timestamps.as_ptr() as *const c_void,
            timestamps_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy for timestamps failed".to_string(),
        ));
    }

    // Execute GPU computation with scalar parameter
    let result = compute_fn(
        gpu_values as *const f64,
        gpu_timestamps as *const f64,
        scalar,
        len,
    );

    // Free GPU memory
    unsafe { cudaFree(gpu_values as *mut c_void) };
    unsafe { cudaFree(gpu_timestamps as *mut c_void) };

    Ok(result)
}

/// Helper for GPU operations with mutable u32 array and readonly u64 array
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_mut_u64<F>(
    indices: &mut [u32],
    values: &[u64],
    indices_len: usize,
    values_len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const u64, usize, usize),
{
    // Allocate GPU memory
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let mut gpu_values: *mut u64 = ptr::null_mut();
    let indices_size = indices_len * std::mem::size_of::<u32>();
    let values_size = values_len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut u64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut c_void,
            indices.as_ptr() as *const c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Execute GPU computation
    compute_fn(gpu_indices, gpu_values, indices_len, values_len);

    // Copy sorted indices back
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut c_void,
            gpu_indices as *const c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_indices as *mut c_void) };
    unsafe { cudaFree(gpu_values as *mut c_void) };

    Ok(())
}

/// Helper for GPU operations with mutable u32 array and readonly f64 array
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_mut_f64<F>(
    indices: &mut [u32],
    values: &[f64],
    indices_len: usize,
    values_len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const f64, usize, usize),
{
    // Allocate GPU memory
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let indices_size = indices_len * std::mem::size_of::<u32>();
    let values_size = values_len * std::mem::size_of::<f64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut c_void,
            indices.as_ptr() as *const c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Execute GPU computation
    compute_fn(gpu_indices, gpu_values, indices_len, values_len);

    // Copy sorted indices back
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut c_void,
            gpu_indices as *const c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_indices as *mut c_void) };
    unsafe { cudaFree(gpu_values as *mut c_void) };

    Ok(())
}

/// Helper for GPU operations with mutable u32 array and readonly i64 array
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_u32_mut_i64<F>(
    indices: &mut [u32],
    values: &[i64],
    indices_len: usize,
    values_len: usize,
    compute_fn: F,
) -> Result<(), crate::types::HwxError>
where
    F: FnOnce(*mut u32, *const i64, usize, usize),
{
    // Allocate GPU memory
    let mut gpu_indices: *mut u32 = ptr::null_mut();
    let mut gpu_values: *mut i64 = ptr::null_mut();
    let indices_size = indices_len * std::mem::size_of::<u32>();
    let values_size = values_len * std::mem::size_of::<i64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_indices as *mut *mut u32 as *mut *mut c_void,
            indices_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut i64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_indices as *mut c_void,
            indices.as_ptr() as *const c_void,
            indices_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Execute GPU computation
    compute_fn(gpu_indices, gpu_values, indices_len, values_len);

    // Copy sorted indices back
    if unsafe {
        cudaMemcpy(
            indices.as_mut_ptr() as *mut c_void,
            gpu_indices as *const c_void,
            indices_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_indices as *mut c_void) };
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy back failed".to_string(),
        ));
    }

    // Cleanup
    unsafe { cudaFree(gpu_indices as *mut c_void) };
    unsafe { cudaFree(gpu_values as *mut c_void) };

    Ok(())
}

/// Helper for GPU operations with f64 and u64 arrays returning f64
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_f64_u64<F>(
    values: &[f64],
    timestamps: &[u64],
    len: usize,
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, *const u64, usize) -> f64,
{
    // Allocate GPU memory
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let mut gpu_timestamps: *mut u64 = ptr::null_mut();
    let values_size = len * std::mem::size_of::<f64>();
    let timestamps_size = len * std::mem::size_of::<u64>();

    if unsafe {
        cudaMalloc(
            &mut gpu_values as *mut *mut f64 as *mut *mut c_void,
            values_size,
        )
    } != 0
    {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_timestamps as *mut *mut u64 as *mut *mut c_void,
            timestamps_size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            values_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_timestamps as *mut c_void,
            timestamps.as_ptr() as *const c_void,
            timestamps_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Execute GPU computation
    let result = compute_fn(gpu_values, gpu_timestamps, len);

    // Cleanup
    unsafe { cudaFree(gpu_values as *mut c_void) };
    unsafe { cudaFree(gpu_timestamps as *mut c_void) };

    Ok(result)
}

/// Helper for GPU operations with two f64 arrays and scalar returning f64
#[cfg(has_cuda)]
pub unsafe fn with_gpu_buffer_2f64_scalar<F>(
    values: &[f64],
    timestamps: &[f64],
    predict_time: f64,
    len: usize,
    compute_fn: F,
) -> Result<f64, crate::types::HwxError>
where
    F: FnOnce(*const f64, *const f64, f64, usize) -> f64,
{
    // Allocate GPU memory
    let mut gpu_values: *mut f64 = ptr::null_mut();
    let mut gpu_timestamps: *mut f64 = ptr::null_mut();
    let size = len * std::mem::size_of::<f64>();

    if unsafe { cudaMalloc(&mut gpu_values as *mut *mut f64 as *mut *mut c_void, size) } != 0 {
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    if unsafe {
        cudaMalloc(
            &mut gpu_timestamps as *mut *mut f64 as *mut *mut c_void,
            size,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory allocation failed".to_string(),
        ));
    }

    // Copy data to GPU
    if unsafe {
        cudaMemcpy(
            gpu_values as *mut c_void,
            values.as_ptr() as *const c_void,
            size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    if unsafe {
        cudaMemcpy(
            gpu_timestamps as *mut c_void,
            timestamps.as_ptr() as *const c_void,
            size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    } != 0
    {
        unsafe { cudaFree(gpu_values as *mut c_void) };
        unsafe { cudaFree(gpu_timestamps as *mut c_void) };
        return Err(crate::types::HwxError::Internal(
            "GPU memory copy failed".to_string(),
        ));
    }

    // Execute GPU computation
    let result = compute_fn(gpu_values, gpu_timestamps, predict_time, len);

    // Cleanup
    unsafe { cudaFree(gpu_values as *mut c_void) };
    unsafe { cudaFree(gpu_timestamps as *mut c_void) };

    Ok(result)
}
