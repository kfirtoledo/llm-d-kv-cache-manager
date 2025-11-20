#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include "tensor_copy.hpp"
#include "buffer.hpp"        // for get_thread_local_pinned
#include "debug_utils.hpp"   // for DEBUG_PRINT


//----------------------------------------------------------------------
// GPU → Storage (PUT)
// ----------------------------------------------------------------------
__global__ void copy_blocks_kernel(
    const uint8_t* __restrict__ src_base,           // Source (CPU for GET, GPU for PUT)
    uint8_t* __restrict__ dst_base,                 // Destination (GPU for GET, CPU for PUT)
    const int64_t* __restrict__ block_ids,          // Global block IDs to copy
    const int64_t* __restrict__ src_block_offsets,  // Per-block source offsets (within file or buffer)
    const int num_blocks,                           // Number of blocks to copy
    const int layer,                                // Layer index
    const int64_t num_blocks_tot,                   // Total blocks per tensor (used for offset math)
    const size_t bytes_per_plane,                   // Bytes per K or V plane
    const size_t bytes_per_block,                   // Total bytes per block (K+V)
    const bool is_put) {                            // Add direction flag

    const int bi = blockIdx.x;          // block index
    const int k_or_v = blockIdx.y;      // 0=K, 1=V
    const int tid = threadIdx.x;

    if (bi >= num_blocks) return;

    const int64_t gblock = block_ids[bi];
    const size_t src_block_base = src_block_offsets[bi];

    size_t src_offset, dst_offset;

    if (is_put) {
        // PUT: GPU→CPU Source is GPU, destination is CPU
        src_offset = static_cast<size_t>(gblock + k_or_v * num_blocks_tot) * bytes_per_plane;
        dst_offset = src_block_base +
                     static_cast<size_t>(layer) * bytes_per_block +
                     (k_or_v == 1 ? bytes_per_plane : 0);
    } else {
        // GET: CPU→GPU - Source is CPU, destination is GPU
        src_offset = src_block_base +
                     static_cast<size_t>(layer) * bytes_per_block +
                     (k_or_v == 1 ? bytes_per_plane : 0);
        dst_offset = static_cast<size_t>(gblock + k_or_v * num_blocks_tot) * bytes_per_plane;
    }

    const uint8_t* src = src_base + src_offset;
    uint8_t* dst = dst_base + dst_offset;

    for (size_t i = tid; i < bytes_per_plane; i += blockDim.x) {
        dst[i] = src[i];  // Copy cooperatively across threads
    }
}

// Copy selected GPU tensor blocks into pinned CPU buffer asynchronously
torch::Tensor copy_gpu_tensors_to_buffer(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    const c10::cuda::CUDAStream& stream) {

    TORCH_CHECK(!src_tensors.empty(), "Source tensors list is empty");
    const auto& ref = src_tensors[0];
    TORCH_CHECK(ref.is_contiguous(), "src_tensors must be contiguous");

    const auto shape = ref.sizes();  // [2, num_blocks_total, H, B, D]
    TORCH_CHECK(shape.size() == 5, "Expected shape [2, num_blocks, H, B, D]");

    const int64_t num_blocks_tot = shape[1];
    const int64_t H = shape[2];  //H: number of attention heads
    const int64_t B = shape[3]; // B: number of tokens per block (block size)
    const int64_t D = shape[4]; // D: head dimension
    const int num_layers = static_cast<int>(src_tensors.size());

    const size_t elem_size = ref.element_size();
    const size_t bytes_per_plane = static_cast<size_t>(H * B * D) * elem_size;
    const size_t bytes_per_block = 2 * bytes_per_plane;  // [K,V]

    // Calculate total size needed
    const size_t total_bytes = block_ids_list.size() * num_layers * bytes_per_block;

    auto [pinned_ptr, pinned_size] = get_thread_local_pinned(total_bytes);
    TORCH_CHECK(pinned_size >= total_bytes,
        "Pinned buffer too small: need ", total_bytes, " got ", pinned_size);

    auto dtype = ref.dtype();
    const int64_t total_elements = static_cast<int64_t>(total_bytes / elem_size);

    // Create output tensor view
    torch::Tensor result_cpu = torch::from_blob(
        pinned_ptr, {total_elements},
        torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true)
    );

    auto* cpu_base = static_cast<uint8_t*>(pinned_ptr);

    const char* env = std::getenv("USE_KERNEL_COPY_WRITE");
    bool use_kernel_copy_write = (env && std::string(env) == "1");
    if (use_kernel_copy_write) {
        // Prepare block IDs and CPU offsets
        std::vector<int64_t> cpu_offsets(block_ids_list.size());
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            cpu_offsets[bi] = bi * num_layers * bytes_per_block;
        }

        // Create tensor view of block IDs and transfer to GPU memory
        torch::Tensor block_ids_tensor = torch::from_blob(
            const_cast<int64_t*>(block_ids_list.data()),
            {static_cast<int64_t>(block_ids_list.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Create tensor view of CPU offsets and transfer to GPU memory
        torch::Tensor cpu_offsets_tensor = torch::from_blob(
            cpu_offsets.data(),
            {static_cast<int64_t>(cpu_offsets.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Map pinned CPU memory to device pointer (required for GPU kernel to write to host memory - zero-copy)
        uint8_t* cpu_base_dev = nullptr;
        cudaError_t map_err = cudaHostGetDevicePointer(&cpu_base_dev, cpu_base, 0);
        TORCH_CHECK(map_err == cudaSuccess,
            "cudaHostGetDevicePointer failed: ", cudaGetErrorString(map_err));

        // Launch one kernel per layer
        dim3 grid(block_ids_list.size(), 2);  // (blocks, K/V)
        dim3 block(512); // TODO check optimal thread count (256 or 512)

        // Launch copy kernel for each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src_tensors[layer].data_ptr());
            copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
                src_ptr,                                    // Source: GPU tensor
                cpu_base_dev,                               // Destination: CPU pinned memory
                block_ids_tensor.data_ptr<int64_t>(),
                cpu_offsets_tensor.data_ptr<int64_t>(),
                block_ids_list.size(),
                layer,
                num_blocks_tot,
                bytes_per_plane,
                bytes_per_block,
                true  // is_put = true
            );
        }

        // Check for kernel launch errors
        cudaError_t launch_err = cudaGetLastError();
        TORCH_CHECK(launch_err == cudaSuccess,
            "Kernel launch failed: ", cudaGetErrorString(launch_err));

    } else {  // Default behavior - memcpyAsync path
        // Direct pointer arithmetic - NO INDEXING operations
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            const int64_t gblock = block_ids_list[bi];
            const size_t cpu_block_base = bi * num_layers * bytes_per_block;

            for (int layer = 0; layer < num_layers; ++layer) {
                const auto& layer_tensor = src_tensors[layer];
                auto* src_base = reinterpret_cast<const uint8_t*>(layer_tensor.data_ptr());

                // Compute GPU source offsets for K and V
                const size_t gpu_K_off = static_cast<size_t>(gblock) * bytes_per_plane;
                const size_t gpu_V_off = (static_cast<size_t>(num_blocks_tot) + gblock) * bytes_per_plane;

                // Compute CPU destination offsets for K and V
                const size_t cpu_K_off = cpu_block_base + static_cast<size_t>(layer) * bytes_per_block;
                const size_t cpu_V_off = cpu_K_off + bytes_per_plane;

                const void* src_K = src_base + gpu_K_off;
                const void* src_V = src_base + gpu_V_off;
                void* dst_K = cpu_base + cpu_K_off;
                void* dst_V = cpu_base + cpu_V_off;

                cudaError_t err1 = cudaMemcpyAsync(
                    dst_K, src_K, bytes_per_plane,
                    cudaMemcpyDeviceToHost, stream.stream());

                cudaError_t err2 = cudaMemcpyAsync(
                    dst_V, src_V, bytes_per_plane,
                    cudaMemcpyDeviceToHost, stream.stream());

                if (err1 != cudaSuccess || err2 != cudaSuccess) {
                    std::cerr << "[ERROR] cudaMemcpyAsync failed for block=" << gblock
                              << " layer=" << layer
                              << " err1=" << cudaGetErrorString(err1)
                              << " err2=" << cudaGetErrorString(err2)
                              << std::endl;
                    TORCH_CHECK(false, "cudaMemcpyAsync failed");
                }
            }
        }
    }

    // Reinterpret bfloat16 tensor as uint16_t for safe raw byte access (I/O or memcpy)
    if (result_cpu.dtype() == torch::kBFloat16) {
        result_cpu = result_cpu.view(torch::kUInt16);
    }

    return result_cpu;
}

bool copy_buffer_to_gpu_tensors(
    torch::Tensor cpu_buf,
    const std::vector<int64_t>& block_ids_list,
    const std::vector<torch::Tensor>& dst_tensors,
    int num_blocks_in_file,
    const c10::cuda::CUDAStream& stream) {

    TORCH_CHECK(!dst_tensors.empty(), "Destination tensors list is empty");
    const auto& ref = dst_tensors[0];
    TORCH_CHECK(ref.is_contiguous(), "dst_tensors must be contiguous");
    TORCH_CHECK(cpu_buf.is_contiguous(), "cpu buffer must be contiguous");

    // CRITICAL: Verify cpu_buf is pinned memory
    TORCH_CHECK(cpu_buf.is_pinned(),
        "cpu_buf must be pinned memory for kernel-based copy");

    const auto shape = ref.sizes();  // [2, num_blocks_total, H, B, D]
    TORCH_CHECK(shape.size() == 5, "Expected shape [2, num_blocks, H, B, D]");

    const int64_t num_blocks_tot = shape[1];
    const int64_t H = shape[2];  // H: number of attention heads
    const int64_t B = shape[3];  // B: number of tokens per block (block size)
    const int64_t D = shape[4];  // D: head dimension
    const int num_layers = static_cast<int>(dst_tensors.size());

    const size_t elem_size = ref.element_size(); // Size (in bytes) of a single element

    const size_t bytes_per_plane = static_cast<size_t>(H * B * D) * elem_size;
    const size_t bytes_per_block = 2 * bytes_per_plane;  // [K,V] in CPU buf

    auto* cpu_base = cpu_buf.data_ptr<uint8_t>();

    const char* env = std::getenv("USE_KERNEL_COPY_READ");
    bool use_kernel_copy_read = (!env || std::string(env) != "0");

    if (use_kernel_copy_read) {  // Default behavior - batched kernel
        // Calculate CPU buffer offset for each block (maps global block ID to local file offset)
        std::vector<int64_t> cpu_offsets(block_ids_list.size());
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            const int64_t gblock = block_ids_list[bi];
            const int64_t lblock = (num_blocks_in_file > 0) ? (gblock % num_blocks_in_file) : 0;
            cpu_offsets[bi] = static_cast<int64_t>(lblock) * num_layers * bytes_per_block;
        }

        // Wrap block IDs in tensor and copy to GPU for kernel access
        torch::Tensor block_ids_tensor = torch::from_blob(
            const_cast<int64_t*>(block_ids_list.data()),
            {static_cast<int64_t>(block_ids_list.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Wrap CPU offsets in tensor and copy to GPU for kernel access
        torch::Tensor cpu_offsets_tensor = torch::from_blob(
            cpu_offsets.data(),
            {static_cast<int64_t>(cpu_offsets.size())},
            torch::dtype(torch::kInt64)
        ).to(torch::kCUDA, /*non_blocking=*/true);

        // Launch one kernel per layer
        dim3 grid(block_ids_list.size(), 2);  // (blocks, K/V)
        dim3 block(512); //TODO chechk optimal thread count (256 or 512)

        for (int layer = 0; layer < num_layers; ++layer) {
            uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_tensors[layer].data_ptr());

            copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
                cpu_base,
                dst_ptr,
                block_ids_tensor.data_ptr<int64_t>(),
                cpu_offsets_tensor.data_ptr<int64_t>(),
                block_ids_list.size(),
                layer,
                num_blocks_tot,
                bytes_per_plane,
                bytes_per_block,
                false // is_put = false
            );
        }

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            std::cerr << "[ERROR] Kernel launch failed: "
                    << cudaGetErrorString(launch_err) << std::endl;
            return false;
        }

    } else {  // Standard cudaMemcpyAsync path
        for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
            const int64_t gblock = block_ids_list[bi];
            const int64_t lblock = (num_blocks_in_file > 0) ? (gblock % num_blocks_in_file) : 0;
            const size_t cpu_block_base = static_cast<size_t>(lblock) * num_layers * bytes_per_block;

            for (int layer = 0; layer < num_layers; ++layer) {
                const auto& layer_tensor = dst_tensors[layer];
                auto* dst_base = reinterpret_cast<uint8_t*>(layer_tensor.data_ptr());

                // Compute CPU source offsets for K and V
                const size_t cpu_K_off = cpu_block_base + static_cast<size_t>(layer) * bytes_per_block;
                const size_t cpu_V_off = cpu_K_off + bytes_per_plane;

                // Compute GPU destination offsets for K and V
                const size_t gpu_K_off = static_cast<size_t>(gblock) * bytes_per_plane;
                const size_t gpu_V_off = (static_cast<size_t>(num_blocks_tot) + gblock) * bytes_per_plane;

                void* dst_K = dst_base + gpu_K_off;
                void* dst_V = dst_base + gpu_V_off;
                void* src_K = cpu_base + cpu_K_off;
                void* src_V = cpu_base + cpu_V_off;

                cudaError_t err1 = cudaMemcpyAsync(
                    dst_K, src_K, bytes_per_plane,
                    cudaMemcpyHostToDevice, stream.stream());

                cudaError_t err2 = cudaMemcpyAsync(
                    dst_V, src_V, bytes_per_plane,
                    cudaMemcpyHostToDevice, stream.stream());

                if (err1 != cudaSuccess || err2 != cudaSuccess) {
                    std::cerr << "[ERROR] cudaMemcpyAsync failed for block=" << gblock
                              << " layer=" << layer
                              << " err1=" << cudaGetErrorString(err1)
                              << " err2=" << cudaGetErrorString(err2)
                              << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}