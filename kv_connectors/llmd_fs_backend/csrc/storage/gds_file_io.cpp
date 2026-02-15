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

#include "gds_file_io.hpp"
#include "file_io.hpp"
#include "logger.hpp"

#include <torch/extension.h>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

// Helper function to check if GDS Bounce Buffer mode is enabled via environment variable
static bool is_gds_bb_enabled() {
  const char* env_val = std::getenv("USE_GDS_BB");
  return env_val != nullptr && (std::string(env_val) == "1" || std::string(env_val) == "true");
}

// Constructor
GdsFileIO::GdsFileIO(const std::vector<std::pair<void*, size_t>>& gpu_buffers,
                     size_t block_size)
    : m_gds_initialized(false) {
  if (!is_gds_supported()) {
    FS_LOG_INFO("GdsFileIO: GDS not supported, using CPU buffer staging");
    return;
  }
  
  if (initialize_gds()) {
    FS_LOG_INFO("GdsFileIO: GPUDirect Storage (GDS) enabled");
    
    // Register GPU buffers with optional per-block registration
    for (const auto& [ptr, size] : gpu_buffers) {
      if (!register_gpu_buffer(ptr, size, block_size)) {
        FS_LOG_WARN("GdsFileIO: Failed to register buffer " << ptr);
      }
    }
  } else {
    FS_LOG_WARN("GdsFileIO: GDS initialization failed, using CPU "
                "buffer staging");
  }
}

// Destructor - cleanup all GDS resources
GdsFileIO::~GdsFileIO() {
#ifdef USE_CUFILE
  if (!m_gds_initialized) {
    return;
  }

  // Deregister all buffers
  for (const auto& [ptr, size] : m_registered_buffers) {
    cuFileBufDeregister(ptr);
  }
  m_registered_buffers.clear();

  // Close driver
  CUfileError_t status = cuFileDriverClose();
  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_WARN("GdsFileIO: cuFileDriverClose failed with error code: "
                << status.err);
  }

  m_gds_initialized = false;
#endif
}

// Static capability check
bool GdsFileIO::is_gds_supported() {
#ifdef USE_CUFILE
  // Check if cuFile library is available
  CUfileError_t status;

  // Try to get cuFile version as a simple availability check
  int version = 0;
  status = cuFileGetVersion(&version);

  if (status.err != CU_FILE_SUCCESS) {
    return false;
  }

  // Check if GPU supports GDS
  int device_id = 0;
  cudaError_t cuda_err = cudaGetDevice(&device_id);
  if (cuda_err != cudaSuccess) {
    return false;
  }

  // Check GPU capability
  int gds_supported = 0;
  cuda_err = cudaDeviceGetAttribute(&gds_supported,
                                    cudaDevAttrGPUDirectRDMASupported,
                                    device_id);

  if (cuda_err != cudaSuccess || gds_supported == 0) {
    return false;
  }

  return true;
#else
  return false;
#endif
}

// Initialize GDS driver - opens cuFile driver and queries capabilities
bool GdsFileIO::initialize_gds() {
#ifdef USE_CUFILE
  CUfileError_t status = cuFileDriverOpen();  // Initialize cuFile driver (must
                                              // be called once per process)

  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_ERROR("GdsFileIO: cuFileDriverOpen failed with error code: "
                 << status.err);
    return false;
  }

  m_gds_initialized = true;  // Mark as initialized for subsequent operations

  // Query and log driver capabilities (optional, for debugging/monitoring)
  CUfileDrvProps_t props;
  status = cuFileDriverGetProperties(&props);
  if (status.err == CU_FILE_SUCCESS) {
    FS_LOG_INFO("GdsFileIO: cuFile driver properties:\n"
                << "  - max_device_cache_size: " << props.max_device_cache_size
                << "\n"  // Device cache limit
                << "  - max_device_pinned_mem_size: "
                << props.max_device_pinned_mem_size);
  }

  return true;
#else
  return false;
#endif
}

// Register GPU buffer with cuFile for optimized DMA transfers
bool GdsFileIO::register_gpu_buffer(void* gpu_ptr, size_t size, size_t block_size) {
#ifdef USE_CUFILE
  if (!m_gds_initialized) {
    return false;
  }

  if (m_registered_buffers.find(gpu_ptr) != m_registered_buffers.end()) {
    return true;  // Skip if already registered
  }
  
  FS_LOG_INFO("cuFileBufRegister: gpu_ptr " << gpu_ptr << " size " << size << " bytes (" << (size / (1024.0 * 1024.0)) << " MB)" << (block_size > 0 ? " block_size " + std::to_string(block_size) + " bytes (" + std::to_string(block_size / (1024.0 * 1024.0)) + " MB)" : ""));
   
  // Check if GDS Bounce Buffer mode is enabled via environment variable
  bool use_gds_bb = is_gds_bb_enabled();
  
  // If block_size is 0 or GDS_BB mode is enabled, register entire buffer at once
  if (block_size == 0 || use_gds_bb) {
    FS_LOG_INFO("GDS with BB mode: Registering entire buffer" << (use_gds_bb ? " (GDS_BB mode)" : "")
                << ": ptr " << gpu_ptr << " size " << size << " bytes ("
                << (size / (1024.0 * 1024.0)) << " MB)");
    
    CUfileError_t status = cuFileBufRegister(gpu_ptr, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
      FS_LOG_WARN("GdsFileIO: cuFileBufRegister failed with error code: " << status.err);
      return false;
    }

    m_registered_buffers[gpu_ptr] = size;
    return true;
  }
  
  // Register buffer in blocks
  size_t num_blocks = (size + block_size - 1) / block_size;
  for (size_t i = 0; i < num_blocks; i++) {
    void* block_ptr = static_cast<uint8_t*>(gpu_ptr) + (i * block_size);
    size_t current_block_size = std::min(block_size, size - (i * block_size));
    
    //FS_LOG_INFO("  Registering block " << i << "/" << num_blocks << ": ptr " << block_ptr << " size " << current_block_size << " bytes (" << (current_block_size / (1024.0 * 1024.0)) << " MB)");
    
    CUfileError_t status = cuFileBufRegister(block_ptr, current_block_size, 0);
    if (status.err != CU_FILE_SUCCESS) {
      FS_LOG_WARN("GdsFileIO: cuFileBufRegister failed for block " << i << " with error code: " << status.err);
      return false;
    }

    m_registered_buffers[block_ptr] = current_block_size;
  }
  FS_LOG_INFO("GDS mode: Registering "  << num_blocks  << " blocks, size " << block_size << " bytes (" << (block_size / (1024.0 * 1024.0)) << " MB)");
  return true;
#else
  return false;
#endif
}


// OPTIMIZED: Write multiple blocks to a file in a single file-open session
// Opens file once, writes all blocks sequentially, then closes
bool GdsFileIO::write_blocks_to_file(const std::string& file_path,
                                     const std::vector<torch::Tensor>& tensors,
                                     const std::vector<int64_t>& block_ids,
                                     size_t block_size,
                                     cudaStream_t stream) {
#ifdef USE_CUFILE
  // Create parent directory if needed
  fs::path path(file_path);
  fs::path parent_dir = path.parent_path();
  try {
    fs::create_directories(parent_dir);
  } catch (const fs::filesystem_error& e) {
    FS_LOG_ERROR("GdsFileIO: Failed to create directories: " << e.what());
    return false;
  }

  // Open file once with O_RDWR and O_DIRECT for GDS
  int fd = open(file_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
  if (fd < 0) {
    FS_LOG_ERROR("GdsFileIO: Failed to open file " << file_path << ": "
                 << std::strerror(errno) << " (errno=" << errno << ")");
    return false;
  }

  // Register file descriptor with cuFile driver for DMA setup
  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(CUfileDescr_t));
  /// Configure the descriptor with our file descriptor
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  /// register the file with cuFile driver
  CUfileHandle_t handle;
  CUfileError_t status = cuFileHandleRegister(&handle, &descr);
  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_ERROR("GdsFileIO: cuFileHandleRegister failed with error code: "
                 << status.err);
    close(fd);
    return false;
  }

  // Write all blocks sequentially
  bool success = true;
  off_t file_offset = 0;

  for (size_t bi = 0; bi < block_ids.size() && success; ++bi) {
    int64_t gpu_block_idx = block_ids[bi];

    for (const auto& tensor : tensors) {
      // Calculate GPU pointer (base + offset)
      void* gpu_base_ptr = tensor.data_ptr();
      void* actual_gpu_ptr = static_cast<uint8_t*>(gpu_base_ptr) + 
                             (gpu_block_idx * block_size);

      // Write this block's data for this layer
      ssize_t bytes_written = cuFileWrite(handle, actual_gpu_ptr, block_size, 
                                          file_offset, 0);

      if (bytes_written < 0) {
        FS_LOG_ERROR("GdsFileIO: cuFileWrite failed with error: "
                     << bytes_written << " at file_offset=" << file_offset);
        success = false;
        break;
      } else if (bytes_written != static_cast<ssize_t>(block_size)) {
        FS_LOG_ERROR("GdsFileIO: Incomplete write: " << bytes_written
                     << " / " << block_size << " bytes at file_offset="
                     << file_offset);
        success = false;
        break;
      }

      file_offset += block_size;
    }
  }

  cuFileHandleDeregister(handle);
  close(fd);

  return success;
#else
  return false;
#endif
}

// OPTIMIZED: Read multiple blocks from a file in a single file-open session
// Opens file once, reads all blocks sequentially, then closes
bool GdsFileIO::read_blocks_from_file(const std::string& file_path,
                                      const std::vector<torch::Tensor>& tensors,
                                      const std::vector<int64_t>& block_ids,
                                      size_t block_size,
                                      cudaStream_t stream) {
#ifdef USE_CUFILE
  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    FS_LOG_ERROR("GdsFileIO: Failed to open file " << file_path << ": "
                 << std::strerror(errno) << " (errno=" << errno << ")");
    return false;
  }

  // Register file descriptor with cuFile driver
  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(CUfileDescr_t));
  /// Configure the descriptor with our file descriptor
  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // register the file with cuFile driver
  CUfileHandle_t handle;
  CUfileError_t status = cuFileHandleRegister(&handle, &descr);
  if (status.err != CU_FILE_SUCCESS) {
    FS_LOG_ERROR("GdsFileIO: cuFileHandleRegister failed with error code: "
                 << status.err);
    close(fd);
    return false;
  }

  // Read all blocks sequentially
  bool success = true;
  off_t file_offset = 0;

  for (size_t bi = 0; bi < block_ids.size() && success; ++bi) {
    int64_t gpu_block_idx = block_ids[bi];

    for (const auto& tensor : tensors) {
      // Calculate GPU pointer (base + offset)
      void* gpu_base_ptr = tensor.data_ptr();
      void* actual_gpu_ptr = static_cast<uint8_t*>(gpu_base_ptr) + 
                             (gpu_block_idx * block_size);

      // Read this block's data for this layer
      ssize_t bytes_read = cuFileRead(handle, actual_gpu_ptr, block_size, 
                                      file_offset, 0);

      if (bytes_read < 0) {
        FS_LOG_ERROR("GdsFileIO: cuFileRead failed with error: "
                     << bytes_read << " at file_offset=" << file_offset);
        success = false;
        break;
      } else if (bytes_read != static_cast<ssize_t>(block_size)) {
        FS_LOG_ERROR("GdsFileIO: Incomplete read: " << bytes_read
                     << " / " << block_size << " bytes at file_offset="
                     << file_offset);
        success = false;
        break;
      }

      file_offset += block_size;
    }
  }

  cuFileHandleDeregister(handle);
  close(fd);

  return success;
#else
  return false;
#endif
}
