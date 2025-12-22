// cfg.h
#pragma once

#include <torch/extension.h>
#include <cstddef>

// cfg.hpp
#pragma once

#include <torch/extension.h>
#include <cstddef>
#include <cstdint>
// Encapsulates KV cache block layout and dimensions
class CacheLayout {
   public:
    // Number of blocks, from num_blocks_dimension
    int64_t num_blocks;
    // Bytes for each block, calculated from stride[num_blocks_dimension] *
    // element_size
    size_t bytes_per_block;
    // Bytes per tensor element
    size_t elem_size;
    // Which axis is the num_block index
    int num_blocks_dimension;
    // True if {k,v} dimension is before num_blocks_dimension
    bool kv_before_blocks;
    // True if layer dimension is before num_blocks_dimension
    bool layers_before_blocks;
    // Bytes per K or V plane use only when kv_before_blocks is true
    size_t kv_bytes_per_plane;

    CacheLayout(const torch::Tensor& ref,
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

    // Env flags
    bool debug;
    bool use_kernel_copy_read;
    bool use_kernel_copy_write;

    ConnectorConfig(int gpu_blocks_per_file, CacheLayout layout);

    // Helper for reading environment variable flags
    static bool get_env_flag(const char* name, bool default_value = false);
};
