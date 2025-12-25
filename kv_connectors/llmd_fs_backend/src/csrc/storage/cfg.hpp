// cfg.h
#pragma once

#include <torch/extension.h>
#include <cstddef>

// cfg.hpp
#pragma once

#include <torch/extension.h>
#include <cstddef>
#include <cstdint>
// Encapsulates KV cache layout and dimensions
class CacheLayout {
   public:
    // Number of blocks, from num_blocks_dimension
    int64_t num_blocks;
    // Data type of the tensors
    torch::Dtype dtype;
    // Bytes per tensor element
    size_t elem_size;
    // Total Bytes for each block
    size_t bytes_per_block;
    // Bytes per K or V plane use only when kv_before_blocks is true
    size_t kv_bytes_per_plane;
    // Which axis is the num_block index
    int num_blocks_dimension;
    // Reference tensor for layout info
    int num_layers;

    // True if {k,v} dimension is before num_blocks_dimension
    bool kv_before_blocks;
    // True if layer dimension is before num_blocks_dimension
    bool layers_before_blocks;

    CacheLayout(std::vector<torch::Tensor>& tensors,
                int num_blocks_dimension,
                bool kv_before_blocks,
                bool layers_before_blocks);
};

// ------------------------------------------------------------
// ConnectorConfig (pure config + derived state)
// ------------------------------------------------------------
class ConnectorConfig {
   public:
    // Static configuration
    int gpu_blocks_per_file;

    // KV-cache layout
    CacheLayout layout;
    // Staging buffer size in bytes
    int64_t staging_buffer_size_bytes;
    // Staging buffer size in number of tensor elements
    int64_t staging_buffer_total_elements;

    // Env flags
    // Verbosity Debug flag
    bool debug;
    // Use kernel-based copy for put operations
    bool use_kernel_copy_write;
    // Use kernel-based copy for get operations
    bool use_kernel_copy_read;

    ConnectorConfig(int gpu_blocks_per_file,
                    size_t staging_buffer_size_mb,
                    CacheLayout layout);

    // Helper for reading environment variable flags
    static bool get_env_flag(const char* name, bool default_value = false);
};
