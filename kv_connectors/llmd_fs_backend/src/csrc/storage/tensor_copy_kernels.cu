#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include "tensor_copy.hpp"
#include "thread_pool.hpp"
#include "debug_utils.hpp"

//----------------------------------------------------------------------
// Helper Structures and Functions
//----------------------------------------------------------------------

// Helper to wrap CPU arrays as GPU tensors for kernel access
template <typename T>
torch::Tensor to_gpu_tensor(const std::vector<T>& data) {
    return torch::from_blob(const_cast<T*>(data.data()),
                            {static_cast<int64_t>(data.size())},
                            torch::dtype(torch::kInt64))
        .to(torch::kCUDA, /*non_blocking=*/true);
}

// Thread configuration constant
constexpr int COPY_THREADS =
    512;  // TODO: check optimal thread count (256 or 512)

//----------------------------------------------------------------------
// CUDA Kernel
//----------------------------------------------------------------------
// Kernel copies one K or V plane of one block.
// Each thread cooperates to copy bytes from src → dst.
__global__ void copy_blocks_kernel(
    const uint8_t* __restrict__ src_base,  // Source (CPU for GET, GPU for PUT)
    uint8_t* __restrict__ dst_base,  // Destination (GPU for GET, CPU for PUT)
    const int64_t* __restrict__ block_ids,          // Global block IDs to copy
    const int64_t* __restrict__ src_block_offsets,  // Per-block source offsets
                                                    // (within file or buffer)
    const int num_blocks,                           // Number of blocks to copy
    const int layer,                                // Layer index
    const int64_t
        num_blocks_tot,  // Total blocks per tensor (used for offset math)
    const size_t bytes_per_plane,  // Bytes per K or V plane
    const size_t bytes_per_block,  // Total bytes per block (K+V)
    const bool kv_before_blocks,  // use plane-based layout or full-block layout
    const bool layers_before_blocks,  // layers appear before block dimension
    const bool is_put)                // Add direction flag
{
    const int bi = blockIdx.x;  // block index
    const int k_or_v =
        blockIdx.y;  // 0=K, 1=V (only used if kv_before_blocks==true)
    const int tid = threadIdx.x;

    if (bi >= num_blocks) return;

    // Global block ID
    const int64_t gpu_block_idx = block_ids[bi];
    // CPU offset for this block
    const size_t src_block_base = src_block_offsets[bi];
    size_t src_plane_offset, dst_plane_offset, src_block_offset,
        dst_block_offset;

    // ----------------------------------------------------------------------
    // Case 1: Default layout + K/V planes appear before the block dimension
    // (kv_before_blocks == true) by default in this case layers_before_blocks
    // == true
    // ----------------------------------------------------------------------
    if (kv_before_blocks) {
        size_t plane_offset = (k_or_v == 1 ? bytes_per_plane : 0);

        if (is_put) {
            // PUT: GPU → CPU
            src_plane_offset =
                static_cast<size_t>(gpu_block_idx + k_or_v * num_blocks_tot) *
                bytes_per_plane;
            dst_plane_offset = src_block_base +
                               static_cast<size_t>(layer) * bytes_per_block +
                               plane_offset;
        } else {
            // GET: CPU → GPU
            src_plane_offset = src_block_base +
                               static_cast<size_t>(layer) * bytes_per_block +
                               plane_offset;
            dst_plane_offset =
                static_cast<size_t>(gpu_block_idx + k_or_v * num_blocks_tot) *
                bytes_per_plane;
        }

        const uint8_t* src = src_base + src_plane_offset;
        uint8_t* dst = dst_base + dst_plane_offset;

        // Copy cooperatively across threads
        for (size_t i = tid; i < bytes_per_plane; i += blockDim.x) {
            dst[i] = src[i];
        }
        return;
    }

    // ----------------------------------------------------------------------
    // Case 2: Default layout + block dimension appear before the K/V
    // Copy entire block for this layer
    // ----------------------------------------------------------------------
    if (layers_before_blocks) {
        if (is_put) {
            // PUT: GPU → CPU
            src_block_offset =
                static_cast<size_t>(gpu_block_idx) * bytes_per_block;
            dst_block_offset =
                src_block_base + static_cast<size_t>(layer) * bytes_per_block;
        } else {
            // GET: CPU → GPU
            src_block_offset =
                src_block_base + static_cast<size_t>(layer) * bytes_per_block;
            dst_block_offset =
                static_cast<size_t>(gpu_block_idx) * bytes_per_block;
        }

        const uint8_t* src = src_base + src_block_offset;
        uint8_t* dst = dst_base + dst_block_offset;

        // Copy cooperatively across threads
        for (size_t i = tid; i < bytes_per_block; i += blockDim.x) {
            dst[i] = src[i];
        }
        return;
    }

    // ----------------------------------------------------------------------
    // Case 3: Cross-layer layout (layers_before_blocks == false)
    // Entire block contains all layers - copy block once
    // ----------------------------------------------------------------------
    if (is_put) {
        // PUT: GPU → CPU
        src_block_offset = static_cast<size_t>(gpu_block_idx) * bytes_per_block;
        dst_block_offset = src_block_base;
    } else {
        // GET: CPU → GPU
        src_block_offset = src_block_base;
        dst_block_offset = static_cast<size_t>(gpu_block_idx) * bytes_per_block;
    }

    const uint8_t* src = src_base + src_block_offset;
    uint8_t* dst = dst_base + dst_block_offset;

    // Copy cooperatively across threads
    for (size_t i = tid; i < bytes_per_block; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// GPU kernel-based copy path (uses CUDA threads for copying)
void copy_via_kernel(uint8_t* cpu_base,
                     const std::vector<torch::Tensor>& gpu_tensors,
                     const std::vector<int64_t>& block_ids_list,
                     const c10::cuda::CUDAStream& stream,
                     bool is_put,
                     const ConnectorConfig& cfg) {
    const int num_layers = static_cast<int>(gpu_tensors.size());

    // Calculate CPU buffer offset for each block (maps global block ID to local
    // file offset)
    std::vector<int64_t> cpu_offsets(block_ids_list.size());
    for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
        cpu_offsets[bi] = (block_ids_list[bi] % cfg.gpu_blocks_per_file) *
                          gpu_tensors.size() * cfg.layout.bytes_per_block;
    }
    // Wrap block IDs in tensor and copy to GPU for kernel access
    torch::Tensor block_ids_tensor = to_gpu_tensor(block_ids_list);

    // Wrap CPU offsets in tensor and copy to GPU for kernel access
    torch::Tensor cpu_offsets_tensor = to_gpu_tensor(cpu_offsets);

    // Map CPU memory to device pointer (required for GPU kernel to write to
    // host memory - zero-copy)
    uint8_t* cpu_base_dev = cpu_base;
    if (is_put) {
        cudaError_t map_err =
            cudaHostGetDevicePointer(&cpu_base_dev, cpu_base, 0);
        TORCH_CHECK(map_err == cudaSuccess,
                    "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(map_err));
    }
    if (cfg.layout.layers_before_blocks) {  // standard layout
        dim3 grid(block_ids_list.size(),
                  cfg.layout.kv_before_blocks ? 2 : 1);  // (blocks, K/V)
        dim3 block(COPY_THREADS);

        // Launch copy kernel for all the blocks on each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            uint8_t* gpu_ptr =
                reinterpret_cast<uint8_t*>(gpu_tensors[layer].data_ptr());

            copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
                is_put ? gpu_ptr : cpu_base,      // Source
                is_put ? cpu_base_dev : gpu_ptr,  // Destination
                block_ids_tensor.data_ptr<int64_t>(),
                cpu_offsets_tensor.data_ptr<int64_t>(),
                block_ids_list.size(),
                layer,
                cfg.layout.num_blocks,
                cfg.layout.kv_bytes_per_plane,
                cfg.layout.bytes_per_block,
                cfg.layout.kv_before_blocks,
                cfg.layout.layers_before_blocks,
                is_put);
        }

        // Check for kernel launch errors
        cudaError_t launch_err = cudaGetLastError();
        TORCH_CHECK(launch_err == cudaSuccess,
                    "Kernel launch failed: ",
                    cudaGetErrorString(launch_err));

    } else {  // / Cross-layer layout (each block contains all layers)
        TORCH_CHECK(!cfg.layout.kv_before_blocks,
                    "Invalid layout: kv_before_blocks=true but "
                    "layers_before_blocks=false");

        dim3 grid(block_ids_list.size(), 1);
        dim3 block(COPY_THREADS);

        uint8_t* gpu_ptr =
            reinterpret_cast<uint8_t*>(gpu_tensors[0].data_ptr());
        // Launch the copy kernel once for all blocks across all layers
        copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
            is_put ? gpu_ptr : cpu_base,      // Source
            is_put ? cpu_base_dev : gpu_ptr,  // Destination
            block_ids_tensor.data_ptr<int64_t>(),
            cpu_offsets_tensor.data_ptr<int64_t>(),
            block_ids_list.size(),
            /*layer=*/0,  // Ignored in cross-layer layout
            cfg.layout.num_blocks,
            cfg.layout.kv_bytes_per_plane,
            cfg.layout.bytes_per_block,
            cfg.layout.kv_before_blocks,
            cfg.layout.layers_before_blocks,
            is_put);

        cudaError_t launch_err = cudaGetLastError();
        TORCH_CHECK(launch_err == cudaSuccess,
                    "Kernel launch failed: ",
                    cudaGetErrorString(launch_err));
    }
}
