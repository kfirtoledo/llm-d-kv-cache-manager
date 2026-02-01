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
#include <mutex>
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
  // enable_gds: whether to attempt GDS initialization
  // gpu_buffers: optional list of GPU buffers to pre-register
  GdsFileIO(bool enable_gds = true,
            const std::vector<std::pair<void*, size_t>>& gpu_buffers = {});

  // Destructor - cleanup resources
  ~GdsFileIO();

  // Check if GDS is available and initialized
  bool is_gds_available() const { return m_gds_initialized; }

  // Static capability check - can be called before construction
  static bool is_gds_supported();

  // Register GPU buffer for GDS (optional, improves performance)
  // Only effective in GDS mode
  // Note: Buffers are automatically deregistered in destructor
  bool register_gpu_buffer(void* gpu_ptr, size_t size);

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

  // Mutex for thread-safe operations (only used during buffer registration)
  mutable std::mutex m_mutex;

  // Registered GPU buffers (for GDS mode)
  std::unordered_map<void*, size_t> m_registered_buffers;

  // Initialize GDS driver
  bool initialize_gds();
};

// Made with Bob
