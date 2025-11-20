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
// Helper Structures and Functions
//----------------------------------------------------------------------

// Encapsulates KV cache block layout and dimensions
struct CacheLayout {
    int64_t num_blocks_tot;  // Total blocks per tensor
    int64_t H;               // Number of attention heads
    int64_t B;               // Number of tokens per block (block size)
    int64_t D;               // Head dimension
    size_t elem_size;        // Size (in bytes) of a single element
    size_t bytes_per_plane;  // Bytes per K or V plane
    size_t bytes_per_block;  // Total bytes per block (K+V)

    // Factory method to extract layout from tensor shape
    static CacheLayout from_tensor(const torch::Tensor& ref) {
        const auto shape = ref.sizes();
        TORCH_CHECK(shape.size() == 5, "Expected shape [2, num_blocks, H, B, D]");

        CacheLayout layout;
        layout.num_blocks_tot = shape[1];
        layout.H = shape[2];
        layout.B = shape[3];
        layout.D = shape[4];
        layout.elem_size = ref.element_size();
        layout.bytes_per_plane = static_cast<size_t>(layout.H * layout.B * layout.D) * layout.elem_size;
        layout.bytes_per_block = 2 * layout.bytes_per_plane;  // [K,V]
        return layout;
    }
};

// Helper to wrap CPU arrays as GPU tensors for kernel access
template<typename T>
torch::Tensor to_gpu_tensor(const std::vector<T>& data) {
    return torch::from_blob(
        const_cast<T*>(data.data()),
        {static_cast<int64_t>(data.size())},
        torch::dtype(torch::kInt64)
    ).to(torch::kCUDA, /*non_blocking=*/true);
}

// Helper for reading environment variable flags
inline bool get_env_flag(const char* name, bool default_val) {
    const char* env = std::getenv(name);
    return env ? (std::string(env) == "1") : default_val;
}

inline int64_t compute_cpu_offset(
    size_t bi,
    bool is_put,
    int num_layers,
    size_t bytes_per_block,
    int64_t gblock = 0,
    int num_blocks_in_file = 0)
{
    if (is_put) {
        // PUT - all blocks are stored contiguously in buffer
        return bi * num_layers * bytes_per_block;
    } else {
        // GET - may have only partial blocks to load from the file
        int64_t lblock = gblock % num_blocks_in_file;
        return lblock * num_layers * bytes_per_block;
    }
}

// Thread configuration constant
constexpr int COPY_THREADS = 512;  // TODO: check optimal thread count (256 or 512)

//----------------------------------------------------------------------
// CUDA Kernel
//----------------------------------------------------------------------

// Kernel copies one K or V plane of one block.
// Each thread cooperates to copy bytes from src → dst.
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
    const bool is_put)                              // Add direction flag
{
    const int bi = blockIdx.x;          // block index
    const int k_or_v = blockIdx.y;      // 0=K, 1=V
    const int tid = threadIdx.x;

    if (bi >= num_blocks) return;

    // Global block ID
    const int64_t gblock = block_ids[bi];
    // CPU offset for this block
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

    // Copy cooperatively across threads
    for (size_t i = tid; i < bytes_per_plane; i += blockDim.x) {
        dst[i] = src[i];
    }
}

//----------------------------------------------------------------------
// Copy Implementation Functions
//----------------------------------------------------------------------
// Standard cudaMemcpyAsync path (DMA-based copying)
void copy_via_cuda_memcpy(
    uint8_t* cpu_base,
    const std::vector<torch::Tensor>& gpu_tensors,
    const std::vector<int64_t>& block_ids_list,
    const CacheLayout& layout,
    const c10::cuda::CUDAStream& stream,
    bool is_put,
    int num_blocks_in_file)
{

    const int num_layers = static_cast<int>(gpu_tensors.size());

    // Direct pointer arithmetic - NO INDEXING operations
    for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
        const int64_t gblock = block_ids_list[bi];
        const size_t cpu_block_base = compute_cpu_offset(bi, is_put, num_layers, layout.bytes_per_block, gblock, num_blocks_in_file);

        for (int layer = 0; layer < num_layers; ++layer) {
            const auto& layer_tensor = gpu_tensors[layer];
            auto* gpu_base = reinterpret_cast<uint8_t*>(layer_tensor.data_ptr());

            // Compute GPU offsets for K and V
            const size_t gpu_K_off = static_cast<size_t>(gblock) * layout.bytes_per_plane;
            const size_t gpu_V_off = (static_cast<size_t>(layout.num_blocks_tot) + gblock) * layout.bytes_per_plane;

            // Compute CPU offsets for K and V
            const size_t cpu_K_off = cpu_block_base + static_cast<size_t>(layer) * layout.bytes_per_block;
            const size_t cpu_V_off = cpu_K_off + layout.bytes_per_plane;

            // Set source and destination pointers based on direction
            const void* src_K = is_put ? (gpu_base + gpu_K_off) : (cpu_base + cpu_K_off);
            const void* src_V = is_put ? (gpu_base + gpu_V_off) : (cpu_base + cpu_V_off);
            void* dst_K = is_put ? (cpu_base + cpu_K_off) : (gpu_base + gpu_K_off);
            void* dst_V = is_put ? (cpu_base + cpu_V_off) : (gpu_base + gpu_V_off);

            cudaMemcpyKind kind = is_put ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

            cudaError_t err1 = cudaMemcpyAsync(dst_K, src_K, layout.bytes_per_plane, kind, stream.stream());
            cudaError_t err2 = cudaMemcpyAsync(dst_V, src_V, layout.bytes_per_plane, kind, stream.stream());

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

// GPU kernel-based copy path (uses CUDA threads for copying)
void copy_via_kernel(
    uint8_t* cpu_base,
    const std::vector<torch::Tensor>& gpu_tensors,
    const std::vector<int64_t>& block_ids_list,
    const CacheLayout& layout,
    const c10::cuda::CUDAStream& stream,
    bool is_put,
    int num_blocks_in_file
) {

    const int num_layers = static_cast<int>(gpu_tensors.size());

    // Calculate CPU buffer offset for each block (maps global block ID to local file offset)
    std::vector<int64_t> cpu_offsets(block_ids_list.size());
    for (size_t bi = 0; bi < block_ids_list.size(); ++bi) {
        cpu_offsets[bi] = compute_cpu_offset(bi, is_put, num_layers, layout.bytes_per_block, block_ids_list[bi], num_blocks_in_file);
    }

    // Wrap block IDs in tensor and copy to GPU for kernel access
    torch::Tensor block_ids_tensor = to_gpu_tensor(block_ids_list);

    // Wrap CPU offsets in tensor and copy to GPU for kernel access
    torch::Tensor cpu_offsets_tensor = to_gpu_tensor(cpu_offsets);

    // Map pinned CPU memory to device pointer (required for GPU kernel to write to host memory - zero-copy)
    uint8_t* cpu_base_dev = cpu_base;
    if (is_put) {
        cudaError_t map_err = cudaHostGetDevicePointer(&cpu_base_dev, cpu_base, 0);
        TORCH_CHECK(map_err == cudaSuccess,
            "cudaHostGetDevicePointer failed: ", cudaGetErrorString(map_err));
    }

    // Launch one kernel per layer
    dim3 grid(block_ids_list.size(), 2);  // (blocks, K/V)
    dim3 block(COPY_THREADS);

    // Launch copy kernel for each layer
    for (int layer = 0; layer < num_layers; ++layer) {
        uint8_t* gpu_ptr = reinterpret_cast<uint8_t*>(gpu_tensors[layer].data_ptr());

        copy_blocks_kernel<<<grid, block, 0, stream.stream()>>>(
            is_put ? gpu_ptr : cpu_base,                    // Source
            is_put ? cpu_base_dev : gpu_ptr,                // Destination
            block_ids_tensor.data_ptr<int64_t>(),
            cpu_offsets_tensor.data_ptr<int64_t>(),
            block_ids_list.size(),
            layer,
            layout.num_blocks_tot,
            layout.bytes_per_plane,
            layout.bytes_per_block,
            is_put
        );
    }

    // Check for kernel launch errors
    cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess,
        "Kernel launch failed: ", cudaGetErrorString(launch_err));
}



// Main transfer function - dispatches to kernel or memcpy path
void transfer_kv_blocks(
    uint8_t* cpu_base,
    const std::vector<torch::Tensor>& gpu_tensors,
    const std::vector<int64_t>& block_ids_list,
    const CacheLayout& layout,
    const c10::cuda::CUDAStream& stream,
    bool is_put,
    bool use_kernel,
    int num_blocks_in_file = 0 // default is 0, ignored for PUT
    ) {

    if (use_kernel) { //TODO- support new layout
        copy_via_kernel(cpu_base, gpu_tensors, block_ids_list, layout, stream, is_put, num_blocks_in_file);
    } else {
        copy_via_cuda_memcpy(cpu_base, gpu_tensors, block_ids_list, layout, stream, is_put, num_blocks_in_file);
    }
}

//----------------------------------------------------------------------
// GPU → Storage (PUT)
//----------------------------------------------------------------------

// Copy selected GPU K/V blocks into a single pinned CPU buffer.
// The pinned buffer is returned as a CPU tensor (no copy, just a view).
torch::Tensor copy_gpu_tensors_to_buffer(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    const c10::cuda::CUDAStream& stream) {

    TORCH_CHECK(!src_tensors.empty(), "Source tensors list is empty");
    const auto& ref = src_tensors[0];
    TORCH_CHECK(ref.is_contiguous(), "src_tensors must be contiguous");

    // Extract tensor geometry
    auto layout = CacheLayout::from_tensor(ref);
    const int num_layers = static_cast<int>(src_tensors.size());

    // Total required pinned memory
    const size_t total_bytes = block_ids_list.size() * num_layers * layout.bytes_per_block;

    // Fetch or allocate thread-local pinned buffer
    auto [pinned_ptr, pinned_size] = get_thread_local_pinned(total_bytes);
    TORCH_CHECK(pinned_size >= total_bytes,
        "Pinned buffer too small: need ", total_bytes, " got ", pinned_size);

    auto dtype = ref.dtype();
    const int64_t total_elements = static_cast<int64_t>(total_bytes / layout.elem_size);

    // Wrap pinned memory as tensor view (no copy)
    torch::Tensor result_cpu = torch::from_blob(
        pinned_ptr, {total_elements},
        torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true)
    );

    auto* cpu_base = static_cast<uint8_t*>(pinned_ptr);

    // Check environment variable to determine copy method
    bool use_kernel = get_env_flag("USE_KERNEL_COPY_WRITE", false);
    bool is_put = true;

    // Execute the copy operation
    transfer_kv_blocks(cpu_base, src_tensors, block_ids_list, layout, stream, is_put, use_kernel);

    // Reinterpret bfloat16 tensor as uint16_t for safe raw byte access (I/O or memcpy)
    if (result_cpu.dtype() == torch::kBFloat16) {
        result_cpu = result_cpu.view(torch::kUInt16);
    }

    return result_cpu;
}

//----------------------------------------------------------------------
// Storage → GPU (GET)
//----------------------------------------------------------------------

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

    // Extract tensor geometry
    auto layout = CacheLayout::from_tensor(ref);
    const int num_layers = static_cast<int>(dst_tensors.size());

    auto* cpu_base = cpu_buf.data_ptr<uint8_t>();

    // Check environment variable to determine copy method (default: kernel for READ)
    bool use_kernel = get_env_flag("USE_KERNEL_COPY_READ", true);
    bool is_put = false;

    // Execute the copy operation
    transfer_kv_blocks(cpu_base, dst_tensors, block_ids_list, layout, stream, is_put, use_kernel, num_blocks_in_file);

    return true;
}
