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
#include "debug_utils.hpp"

#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>

namespace fs = std::filesystem;

// Constructor
GdsFileIO::GdsFileIO(bool enable_gds,
                     const std::vector<std::pair<void*, size_t>>& gpu_buffers)
    : m_gds_initialized(false) {
  if (enable_gds && is_gds_supported()) {
    if (initialize_gds()) {
      std::cout << "[INFO] GdsFileIO: GPUDirect Storage (GDS) enabled\n";

      // Register GPU buffers if provided
      if (!gpu_buffers.empty()) {
        std::cout << "[INFO] GdsFileIO: Registering " << gpu_buffers.size()
                  << " GPU buffers for GDS\n";
        for (const auto& [ptr, size] : gpu_buffers) {
          if (register_gpu_buffer(ptr, size)) {
            std::cout << "[INFO] GdsFileIO: Registered buffer " << ptr << " ("
                      << size << " bytes)\n";
          } else {
            std::cerr << "[WARN] GdsFileIO: Failed to register buffer " << ptr
                      << "\n";
          }
        }
      }
    } else {
      std::cerr << "[WARN] GdsFileIO: GDS initialization failed, using CPU "
                   "buffer staging\n";
    }
  } else {
    if (!enable_gds) {
      std::cout << "[INFO] GdsFileIO: GDS disabled by configuration, using CPU "
                   "buffer staging\n";
    } else {
      std::cout
          << "[INFO] GdsFileIO: GDS not supported, using CPU buffer staging\n";
    }
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
    std::cerr << "[WARN] GdsFileIO: cuFileDriverClose failed with error code: "
              << status.err << "\n";
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
    std::cerr << "[ERROR] GdsFileIO: cuFileDriverOpen failed with error code: "
              << status.err << "\n";
    return false;
  }

  m_gds_initialized = true;  // Mark as initialized for subsequent operations

  // Query and log driver capabilities (optional, for debugging/monitoring)
  CUfileDrvProps_t props;
  status = cuFileDriverGetProperties(&props);
  if (status.err == CU_FILE_SUCCESS) {
    std::cout << "[INFO] GdsFileIO: cuFile driver properties:\n"
              << "  - max_device_cache_size: " << props.max_device_cache_size
              << "\n"  // Device cache limit
              << "  - max_device_pinned_mem_size: "
              << props.max_device_pinned_mem_size
              << "\n";  // Pinned memory limit
  }

  return true;
#else
  return false;
#endif
}

// Register GPU buffer with cuFile for optimized DMA transfers
bool GdsFileIO::register_gpu_buffer(void* gpu_ptr, size_t size) {
#ifdef USE_CUFILE
  if (!m_gds_initialized) {  // Ensure driver is initialized first
    return false;
  }

  std::lock_guard<std::mutex> lock(
      m_mutex);  // Thread-safe registration (only used during construction)

  if (m_registered_buffers.find(gpu_ptr) !=
      m_registered_buffers.end()) {  // Skip if already registered
    return true;
  }

  CUfileError_t status =
      cuFileBufRegister(gpu_ptr,
                        size,
                        0);  // Register GPU memory region with cuFile driver
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "[WARN] GdsFileIO: cuFileBufRegister failed with error code: "
              << status.err << "\n";
    return false;
  }

  m_registered_buffers[gpu_ptr] =
      size;  // Track registered buffer for cleanup in destructor
  return true;
#else
  return false;
#endif
}

// GDS write implementation
bool GdsFileIO::write_gds_direct(const std::string& file_path,
                                 void* gpu_ptr,
                                 size_t size,
                                 off_t file_offset,
                                 cudaStream_t stream) {
#ifdef USE_CUFILE
  // Create parent directory if needed
  fs::path path(file_path);
  fs::path parent_dir = path.parent_path();
  try {
    fs::create_directories(parent_dir);
  } catch (const fs::filesystem_error& e) {
    std::cerr << "[ERROR] GdsFileIO: Failed to create directories: " << e.what()
              << "\n";
    return false;
  }

  // Open file with O_DIRECT (bypasses kernel page cache, required for GDS)
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644);
  if (fd < 0) {
    std::cerr << "[ERROR] GdsFileIO: Failed to open file " << file_path << ": "
              << std::strerror(errno) << "\n";
    return false;
  }

  // Register file descriptor with cuFile driver for DMA setup
  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(CUfileDescr_t));
  descr.handle.fd = fd;                        // POSIX file descriptor
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;  // Opaque FD type

  CUfileHandle_t handle;  // cuFile handle for GDS operations
  CUfileError_t status = cuFileHandleRegister(&handle, &descr);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr
        << "[ERROR] GdsFileIO: cuFileHandleRegister failed with error code: "
        << status.err << "\n";
    close(fd);
    return false;
  }

  // Perform async GPU→File write via DMA (bypasses CPU entirely)
  ssize_t bytes_written = 0;
  status =
      cuFileWriteAsync(handle,        // cuFile handle
                       gpu_ptr,       // Source: GPU memory (must be registered)
                       &size,         // Bytes to write
                       &file_offset,  // File offset
                       &file_offset,  // GPU buffer offset (same as file offset)
                       &bytes_written,  // Output: bytes written
                       stream           // CUDA stream for async operation
      );

  bool write_success = true;
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "[ERROR] GdsFileIO: cuFileWriteAsync failed: " << status.err
              << "\n";
    write_success = false;
  } else {
    cudaError_t cuda_err =
        cudaStreamSynchronize(stream);  // Wait for write completion
    if (cuda_err != cudaSuccess) {
      std::cerr << "[ERROR] GdsFileIO: cudaStreamSynchronize failed: "
                << cudaGetErrorString(cuda_err) << "\n";
      write_success = false;
    } else if (bytes_written !=
               static_cast<ssize_t>(size)) {  // Verify complete write
      std::cerr << "[ERROR] GdsFileIO: Incomplete write: " << bytes_written
                << " / " << size << " bytes\n";
      write_success = false;
    }
  }

  cuFileHandleDeregister(handle);  // Deregister cuFile handle
  close(fd);                       // Close file descriptor

  return write_success;
#else
  return false;
#endif
}

// GDS read implementation
bool GdsFileIO::read_gds_direct(const std::string& file_path,
                                void* gpu_ptr,
                                size_t size,
                                off_t file_offset,
                                cudaStream_t stream) {
#ifdef USE_CUFILE
  // Open file with O_DIRECT (bypasses kernel page cache, required for GDS)
  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    std::cerr << "[ERROR] GdsFileIO: Failed to open file " << file_path << ": "
              << std::strerror(errno) << "\n";
    return false;
  }

  // Register file descriptor with cuFile driver for DMA setup
  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(CUfileDescr_t));
  descr.handle.fd = fd;                        // POSIX file descriptor
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;  // Opaque FD type

  CUfileHandle_t handle;  // cuFile handle for GDS operations
  CUfileError_t status = cuFileHandleRegister(&handle, &descr);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr
        << "[ERROR] GdsFileIO: cuFileHandleRegister failed with error code: "
        << status.err << "\n";
    close(fd);
    return false;
  }

  // Perform async File→GPU read via DMA (bypasses CPU entirely)
  ssize_t bytes_read = 0;
  status =
      cuFileReadAsync(handle,   // cuFile handle
                      gpu_ptr,  // Destination: GPU memory (must be registered)
                      &size,    // Bytes to read
                      &file_offset,  // File offset
                      &file_offset,  // GPU buffer offset (same as file offset)
                      &bytes_read,   // Output: bytes read
                      stream         // CUDA stream for async operation
      );

  bool read_success = true;
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "[ERROR] GdsFileIO: cuFileReadAsync failed: " << status.err
              << "\n";
    read_success = false;
  } else {
    cudaError_t cuda_err =
        cudaStreamSynchronize(stream);  // Wait for read completion
    if (cuda_err != cudaSuccess) {
      std::cerr << "[ERROR] GdsFileIO: cudaStreamSynchronize failed: "
                << cudaGetErrorString(cuda_err) << "\n";
      read_success = false;
    } else if (bytes_read !=
               static_cast<ssize_t>(size)) {  // Verify complete read
      std::cerr << "[ERROR] GdsFileIO: Incomplete read: " << bytes_read << " / "
                << size << " bytes\n";
      read_success = false;
    }
  }

  cuFileHandleDeregister(handle);  // Deregister cuFile handle
  close(fd);                       // Close file descriptor

  return read_success;
#else
  return false;
#endif
}
