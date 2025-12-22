// cfg.cpp
#include "cfg.hpp"

#include <cstdlib>
#include <string>

// ------------------------------------------------------------
// CacheLayout
// ------------------------------------------------------------

// Encapsulates KV cache block layout and dimensions
CacheLayout::CacheLayout(const torch::Tensor& ref,
                         int num_blocks_dimension,
                         bool kv_before_blocks,
                         bool layers_before_blocks)
    : num_blocks_dimension(num_blocks_dimension),
      kv_before_blocks(kv_before_blocks),
      layers_before_blocks(layers_before_blocks) {
    TORCH_CHECK(num_blocks_dimension >= 0 && num_blocks_dimension < ref.dim(),
                "num_blocks_dimension out of bounds!");

    // Number of blocks, from num_blocks_dimension
    num_blocks = ref.size(num_blocks_dimension);

    // Bytes per tensor element
    elem_size = ref.element_size();

    // Bytes per K or V plane use only when kv_before_blocks is true
    kv_bytes_per_plane =
        static_cast<size_t>(ref.stride(num_blocks_dimension)) * elem_size;

    // Bytes for each block, calculated from stride[num_blocks_dimension] *
    // element_size
    bytes_per_block =
        kv_before_blocks ? kv_bytes_per_plane * 2 : kv_bytes_per_plane;
    std::cout << "[DEBUG] CacheLayout: num_blocks=" << num_blocks
              << ", bytes_per_block=" << bytes_per_block
              << ", elem_size=" << elem_size
              << ", kv_bytes_per_plane=" << kv_bytes_per_plane
              << ", kv_before_blocks=" << kv_before_blocks
              << ", layers_before_blocks=" << layers_before_blocks << std::endl;
}

// ------------------------------------------------------------
// ConnectorConfig
// ------------------------------------------------------------

ConnectorConfig::ConnectorConfig(int _gpu_blocks_per_file, CacheLayout _layout)
    : gpu_blocks_per_file(_gpu_blocks_per_file), layout(_layout) {
    // Env flags
    debug = get_env_flag("STORAGE_CONNECTOR_DEBUG", false);
    use_kernel_copy_read = get_env_flag("USE_KERNEL_COPY_READ", false);
    use_kernel_copy_write = get_env_flag("USE_KERNEL_COPY_WRITE", false);
    std::cout << "[DEBUG] ConnectorConfig: debug=" << debug
              << ", use_kernel_copy_read=" << use_kernel_copy_read
              << ", use_kernel_copy_write=" << use_kernel_copy_write
              << std::endl;
}

// ------------------------------------------------------------
// Helper for reading environment variable flags
// ------------------------------------------------------------
bool ConnectorConfig::get_env_flag(const char* name, bool default_val) {
    const char* env = std::getenv(name);
    if (!env) return default_val;

    std::string v(env);
    if (v == "1" || v == "true" || v == "TRUE") return true;
    if (v == "0" || v == "false" || v == "FALSE") return false;

    return default_val;
}
