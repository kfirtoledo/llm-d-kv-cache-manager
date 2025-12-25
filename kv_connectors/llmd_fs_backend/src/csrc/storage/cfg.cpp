// cfg.cpp
#include "cfg.hpp"

#include <cstdlib>
#include <string>

// ------------------------------------------------------------
// CacheLayout
// ------------------------------------------------------------
// CacheLayout constructor: Encapsulates KV cache block layout and dimensions
CacheLayout::CacheLayout(std::vector<torch::Tensor>& tensors,
                         int num_blocks_dimension,
                         bool kv_before_blocks,
                         bool layers_before_blocks)
    : num_blocks_dimension(num_blocks_dimension),
      kv_before_blocks(kv_before_blocks),
      layers_before_blocks(layers_before_blocks) {
    const auto& ref = tensors[0];
    num_blocks = ref.size(num_blocks_dimension);
    elem_size = ref.element_size();
    kv_bytes_per_plane =
        static_cast<size_t>(ref.stride(num_blocks_dimension)) * elem_size;
    bytes_per_block =
        kv_before_blocks ? kv_bytes_per_plane * 2 : kv_bytes_per_plane;

    num_layers = static_cast<int>(tensors.size());
    dtype = torch::Dtype(ref.scalar_type());
    std::cout << "[INFO] CacheLayout: num_blocks=" << num_blocks
              << ", bytes_per_block=" << bytes_per_block
              << ", elem_size=" << elem_size
              << ", kv_bytes_per_plane=" << kv_bytes_per_plane
              << ", kv_before_blocks=" << kv_before_blocks
              << ", layers_before_blocks=" << layers_before_blocks << std::endl;
}

// ------------------------------------------------------------
// ConnectorConfig
// ------------------------------------------------------------
// ConnectorConfig constructor
ConnectorConfig::ConnectorConfig(int _gpu_blocks_per_file,
                                 size_t staging_buffer_size_mb,
                                 CacheLayout _layout)
    : gpu_blocks_per_file(_gpu_blocks_per_file), layout(_layout) {
    staging_buffer_size_bytes = staging_buffer_size_mb * 1024ULL * 1024ULL;
    staging_buffer_total_elements =
        staging_buffer_size_bytes / layout.elem_size;

    // Env flags
    debug = get_env_flag("STORAGE_CONNECTOR_DEBUG", false);
    use_kernel_copy_read = get_env_flag("USE_KERNEL_COPY_READ", false);
    use_kernel_copy_write = get_env_flag("USE_KERNEL_COPY_WRITE", false);
    std::cout << "[INFO] ConnectorConfig: use_kernel_copy_read="
              << use_kernel_copy_read
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
