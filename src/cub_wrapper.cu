// SPDX-License-Identifier: Apache-2.0

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

// CUB DeviceRadixSort wrapper for FFI
int cub_device_radix_sort_u32(
    uint32_t* d_values,   // Device memory pointer
    size_t len,
    bool ascending
) {
    // Allocate output buffer
    uint32_t* d_values_out = nullptr;
    cudaError_t err = cudaMalloc(&d_values_out, len * sizeof(uint32_t));
    if (err != cudaSuccess) return (int)err;
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Run sorting operation
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Copy result back to original buffer
    err = cudaMemcpy(d_values, d_values_out, len * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_values_out);
    
    return (int)err;
}

// CUB DeviceRadixSort wrapper for f64 arrays
int cub_device_radix_sort_f64(
    double* d_values,
    size_t len,
    bool ascending
) {
    // Allocate output buffer
    double* d_values_out = nullptr;
    cudaError_t err = cudaMalloc(&d_values_out, len * sizeof(double));
    if (err != cudaSuccess) return (int)err;
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Run sorting operation
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Copy result back to original buffer
    err = cudaMemcpy(d_values, d_values_out, len * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_values_out);
    
    return (int)err;
}

// CUB DeviceRadixSort wrapper for u64 arrays
int cub_device_radix_sort_u64(
    uint64_t* d_values,
    size_t len,
    bool ascending
) {
    // Allocate output buffer
    uint64_t* d_values_out = nullptr;
    cudaError_t err = cudaMalloc(&d_values_out, len * sizeof(uint64_t));
    if (err != cudaSuccess) return (int)err;
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Run sorting operation
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Copy result back to original buffer
    err = cudaMemcpy(d_values, d_values_out, len * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    
    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_values_out);
    
    return (int)err;
}

// CUB DeviceRadixSort wrapper for i64 arrays
int cub_device_radix_sort_i64(
    int64_t* d_values,
    size_t len,
    bool ascending
) {
    // Allocate output buffer
    int64_t* d_values_out = nullptr;
    cudaError_t err = cudaMalloc(&d_values_out, len * sizeof(int64_t));
    if (err != cudaSuccess) return (int)err;
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Run sorting operation
    if (ascending) {
        err = cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Copy result back to original buffer
    err = cudaMemcpy(d_values, d_values_out, len * sizeof(int64_t), cudaMemcpyDeviceToDevice);
    
    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_values_out);
    
    return (int)err;
}

// CUB SortPairs wrapper for sorting u32 indices by u64 values
int cub_sort_pairs_u32_by_u64(
    uint32_t* d_indices,    // Keys to sort
    uint64_t* d_values,     // Values to sort by
    size_t len,
    bool ascending
) {
    // Allocate output buffers
    uint32_t* d_indices_out = nullptr;
    uint64_t* d_values_out = nullptr;
    
    cudaError_t err = cudaMalloc(&d_indices_out, len * sizeof(uint32_t));
    if (err != cudaSuccess) return (int)err;
    
    err = cudaMalloc(&d_values_out, len * sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        return (int)err;
    }
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    
    if (ascending) {
        err = cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,  // Sort by values
            d_indices, d_indices_out, // Carry indices along
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Run sorting operation
    if (ascending) {
        err = cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Copy results back to original buffers
    err = cudaMemcpy(d_indices, d_indices_out, len * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_indices_out);
    cudaFree(d_values_out);
    
    return (int)err;
}

// CUB SortPairs wrapper for sorting u32 indices by f64 values
int cub_sort_pairs_u32_by_f64(
    uint32_t* d_indices,    // Keys to sort
    double* d_values,       // Values to sort by
    size_t len,
    bool ascending
) {
    // Allocate output buffers
    uint32_t* d_indices_out = nullptr;
    double* d_values_out = nullptr;
    
    cudaError_t err = cudaMalloc(&d_indices_out, len * sizeof(uint32_t));
    if (err != cudaSuccess) return (int)err;
    
    err = cudaMalloc(&d_values_out, len * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        return (int)err;
    }
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    
    if (ascending) {
        err = cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,  // Sort by values
            d_indices, d_indices_out, // Carry indices along
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Run sorting operation
    if (ascending) {
        err = cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Copy results back to original buffers
    err = cudaMemcpy(d_indices, d_indices_out, len * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_indices_out);
    cudaFree(d_values_out);
    
    return (int)err;
}

// CUB SortPairs wrapper for sorting u32 indices by i64 values
int cub_sort_pairs_u32_by_i64(
    uint32_t* d_indices,    // Keys to sort
    int64_t* d_values,      // Values to sort by
    size_t len,
    bool ascending
) {
    // Allocate output buffers
    uint32_t* d_indices_out = nullptr;
    int64_t* d_values_out = nullptr;
    
    cudaError_t err = cudaMalloc(&d_indices_out, len * sizeof(uint32_t));
    if (err != cudaSuccess) return (int)err;
    
    err = cudaMalloc(&d_values_out, len * sizeof(int64_t));
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        return (int)err;
    }
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    
    if (ascending) {
        err = cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,  // Sort by values
            d_indices, d_indices_out, // Carry indices along
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Run sorting operation
    if (ascending) {
        err = cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    } else {
        err = cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_values, d_values_out,
            d_indices, d_indices_out,
            len
        );
    }
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Copy results back to original buffers
    err = cudaMemcpy(d_indices, d_indices_out, len * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_indices_out);
        cudaFree(d_values_out);
        return (int)err;
    }
    
    // Free temporary buffers
    cudaFree(d_temp_storage);
    cudaFree(d_indices_out);
    cudaFree(d_values_out);
    
    return (int)err;
}

// CUB DeviceSelect::Unique wrapper for deduplicating sorted arrays
int cub_device_unique_u32(
    uint32_t* d_values,     // Input/output array
    uint32_t* d_temp_out,   // Temporary output buffer
    size_t len,
    size_t* d_num_selected  // Output: number of unique elements
) {
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    cudaError_t err = cub::DeviceSelect::Unique(
        nullptr, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) return (int)err;
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) return (int)err;
    
    // Run unique operation
    err = cub::DeviceSelect::Unique(
        d_temp_storage, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    // Copy result back to original buffer
    size_t h_num_selected;
    err = cudaMemcpy(&h_num_selected, d_num_selected, sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    err = cudaMemcpy(d_values, d_temp_out, h_num_selected * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    // Free temporary storage
    cudaFree(d_temp_storage);
    
    return (int)err;
}

// CUB DeviceSelect::Unique wrapper for f64 arrays
int cub_device_unique_f64(
    double* d_values,       // Input/output array
    double* d_temp_out,     // Temporary output buffer
    size_t len,
    size_t* d_num_selected  // Output: number of unique elements
) {
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    cudaError_t err = cub::DeviceSelect::Unique(
        nullptr, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) return (int)err;
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) return (int)err;
    
    // Run unique operation
    err = cub::DeviceSelect::Unique(
        d_temp_storage, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    // Copy result back to original buffer
    size_t h_num_selected;
    err = cudaMemcpy(&h_num_selected, d_num_selected, sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    err = cudaMemcpy(d_values, d_temp_out, h_num_selected * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // Free temporary storage
    cudaFree(d_temp_storage);
    
    return (int)err;
}

// CUB DeviceSelect::Unique wrapper for u64 arrays
int cub_device_unique_u64(
    uint64_t* d_values,     // Input/output array
    uint64_t* d_temp_out,   // Temporary output buffer
    size_t len,
    size_t* d_num_selected  // Output: number of unique elements
) {
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    cudaError_t err = cub::DeviceSelect::Unique(
        nullptr, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) return (int)err;
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) return (int)err;
    
    // Run unique operation
    err = cub::DeviceSelect::Unique(
        d_temp_storage, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    // Copy result back to original buffer
    size_t h_num_selected;
    err = cudaMemcpy(&h_num_selected, d_num_selected, sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    err = cudaMemcpy(d_values, d_temp_out, h_num_selected * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    
    // Free temporary storage
    cudaFree(d_temp_storage);
    
    return (int)err;
}

// CUB DeviceSelect::Unique wrapper for i64 arrays
int cub_device_unique_i64(
    int64_t* d_values,      // Input/output array
    int64_t* d_temp_out,    // Temporary output buffer
    size_t len,
    size_t* d_num_selected  // Output: number of unique elements
) {
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    cudaError_t err = cub::DeviceSelect::Unique(
        nullptr, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) return (int)err;
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) return (int)err;
    
    // Run unique operation
    err = cub::DeviceSelect::Unique(
        d_temp_storage, temp_storage_bytes,
        d_values, d_temp_out,
        d_num_selected,
        len
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    // Copy result back to original buffer
    size_t h_num_selected;
    err = cudaMemcpy(&h_num_selected, d_num_selected, sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return (int)err;
    }
    
    err = cudaMemcpy(d_values, d_temp_out, h_num_selected * sizeof(int64_t), cudaMemcpyDeviceToDevice);
    
    // Free temporary storage
    cudaFree(d_temp_storage);
    
    return (int)err;
}

// Kernel to compute squared differences
__global__ void compute_squared_diff_kernel(const double* values, double* squared_diff, double mean, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double diff = values[idx] - mean;
        squared_diff[idx] = diff * diff;
    }
}

// CUB variance calculation for f64
int cub_device_variance_f64(
    double* d_values,
    size_t len,
    double* d_result  // Result pointer on device
) {
    if (len == 0) {
        // Set result to NaN for empty input
        double nan_val = std::numeric_limits<double>::quiet_NaN();
        cudaMemcpy(d_result, &nan_val, sizeof(double), cudaMemcpyHostToDevice);
        return 0;
    }
    
    // Step 1: Calculate sum using CUB DeviceReduce::Sum
    size_t temp_storage_bytes = 0;
    double* d_sum = nullptr;
    cudaError_t err = cudaMalloc(&d_sum, sizeof(double));
    if (err != cudaSuccess) return (int)err;
    
    // Get temp storage size for sum
    err = cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_values, d_sum, len);
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        return (int)err;
    }
    
    // Allocate temp storage
    void* d_temp_storage = nullptr;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        return (int)err;
    }
    
    // Calculate sum using CUB
    err = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_values, d_sum, len);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        return (int)err;
    }
    
    // Calculate mean on device
    double* d_mean = nullptr;
    err = cudaMalloc(&d_mean, sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        return (int)err;
    }
    
    // Copy sum to host to calculate mean (we need the mean value for the kernel)
    double h_sum;
    err = cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        cudaFree(d_mean);
        return (int)err;
    }
    double mean = h_sum / len;
    
    // Step 2: Compute squared differences using kernel
    double* d_squared_diff = nullptr;
    err = cudaMalloc(&d_squared_diff, len * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        cudaFree(d_mean);
        return (int)err;
    }
    
    // Launch kernel to compute squared differences
    dim3 blockSize(256);
    dim3 gridSize((len + blockSize.x - 1) / blockSize.x);
    compute_squared_diff_kernel<<<gridSize, blockSize>>>(d_values, d_squared_diff, mean, len);
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        cudaFree(d_mean);
        cudaFree(d_squared_diff);
        return (int)err;
    }
    
    // Step 3: Sum squared differences using CUB DeviceReduce::Sum
    cudaFree(d_temp_storage);
    temp_storage_bytes = 0;
    
    // Get temp storage size for variance sum
    err = cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_squared_diff, d_result, len);
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        cudaFree(d_mean);
        cudaFree(d_squared_diff);
        return (int)err;
    }
    
    // Allocate temp storage for variance sum
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        cudaFree(d_mean);
        cudaFree(d_squared_diff);
        return (int)err;
    }
    
    // Calculate variance sum using CUB
    err = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_squared_diff, d_result, len);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        cudaFree(d_mean);
        cudaFree(d_squared_diff);
        return (int)err;
    }
    
    // Divide by n to get variance (result is already in d_result)
    // Need to divide the result by len
    double h_var_sum;
    err = cudaMemcpy(&h_var_sum, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        cudaFree(d_mean);
        cudaFree(d_squared_diff);
        return (int)err;
    }
    
    double variance = h_var_sum / len;
    err = cudaMemcpy(d_result, &variance, sizeof(double), cudaMemcpyHostToDevice);
    
    // Cleanup
    cudaFree(d_temp_storage);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_squared_diff);
    
    return (int)err;
}

} // extern "C"