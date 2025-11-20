#pragma once

#include <torch/extension.h>
#include <vector>
#include <cstdint>

// Copy selected GPU blocks into a pinned CPU buffer.
// Returns a pinned CPU tensor containing raw K/V block bytes.
torch::Tensor copy_gpu_tensors_to_buffer(
    const std::vector<torch::Tensor>& src_tensors,
    const std::vector<int64_t>& block_ids_list,
    const c10::cuda::CUDAStream& stream
);

// Copy data from a pinned CPU buffer back into GPU tensors
bool copy_buffer_to_gpu_tensors(
    torch::Tensor cpu_buf,
    const std::vector<int64_t>& block_ids_list,
    const std::vector<torch::Tensor>& dst_tensors,
    int num_blocks_in_file,
    const c10::cuda::CUDAStream& stream
);