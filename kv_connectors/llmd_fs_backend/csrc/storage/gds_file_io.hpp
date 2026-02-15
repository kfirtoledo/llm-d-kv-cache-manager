/*
 * Copyright 2025 The llm-d Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <torch/extension.h>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

// Forward declare cuFile types to avoid header dependency when GDS is not
// available
#ifdef USE_CUFILE
  #include <cufile.h>
#else
typedef void* CUfileHandle_t;
typedef struct CUfileDescr_st {
  int handle;
  int type;
} CUfileDescr_t;
#endif

#include "storage_types.hpp"

// GDS File I/O class - unified interface for both storage modes
class GdsFileIO {
 public:
  // Constructor - initializes and detects GDS support
  // gpu_buffers: optional list of GPU buffers to pre-register
  // block_size: size of each block for per-block registration (0 = register entire buffer)
  GdsFileIO(const std::vector<std::pair<void*, size_t>>& gpu_buffers = {},
            size_t block_size = 0);

  // Destructor - cleanup resources
  ~GdsFileIO();

  // Check if GDS is available and initialized
  bool is_gds_available() const { return m_gds_initialized; }

  // Static capability check - can be called before construction
  static bool is_gds_supported();

  // Register GPU buffer for GDS (optional, improves performance)
  // Only effective in GDS mode
  // Note: Buffers are automatically deregistered in destructor
  // block_size: if > 0, register buffer in blocks of this size
  bool register_gpu_buffer(void* gpu_ptr, size_t size, size_t block_size = 0);

  // Write multiple blocks to a file in a single file-open session
  // Opens file once, writes all blocks sequentially, then closes
  bool write_blocks_to_file(const std::string& file_path,
                            const std::vector<torch::Tensor>& tensors,
                            const std::vector<int64_t>& block_ids,
                            size_t block_size,
                            cudaStream_t stream);

  // Read multiple blocks from a file in a single file-open session
  // Opens file once, reads all blocks sequentially, then closes
  bool read_blocks_from_file(const std::string& file_path,
                             const std::vector<torch::Tensor>& tensors,
                             const std::vector<int64_t>& block_ids,
                             size_t block_size,
                             cudaStream_t stream);

 private:
  // GDS initialization state
  bool m_gds_initialized;

  // Registered GPU buffers (for GDS mode)
  std::unordered_map<void*, size_t> m_registered_buffers;

  // Initialize GDS driver
  bool initialize_gds();
};

