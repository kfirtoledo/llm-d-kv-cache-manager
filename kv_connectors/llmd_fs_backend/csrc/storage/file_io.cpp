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

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <random>

#include "tensor_copier.hpp"
#include "debug_utils.hpp"
#include "file_io.hpp"
#include "thread_pool.hpp"
#include "logger.hpp"

namespace fs = std::filesystem;

// -------------------------------------------------------------------
// Constants and thread-local buffers
// -------------------------------------------------------------------
// Define a larger buffer (1MB) to reduce syscall overhead and speed up I/O
const size_t WRITE_BUFFER_SIZE = 1 * 1024 * 1024;  // 1MB buffer

// Allocate custom I/O buffer for this thread (replaces small default buffer)
thread_local std::vector<char> thread_write_buffer(WRITE_BUFFER_SIZE);

// Thread-local unique suffix for temporary files
thread_local std::string tmp_file_suffix =
    "_" + std::to_string(std::random_device{}()) + ".tmp";
// -------------------------------------------------------------------
// file-IO Functions
// -------------------------------------------------------------------
// Write a buffer to disk using a temporary file and atomic rename
bool write_buffer_to_file(const StagingBufferInfo& buf,
                          const std::string& target_path) {
  // Create parent directory if needed
  fs::path file_path(target_path);
  fs::path parent_dir = file_path.parent_path();
  try {
    fs::create_directories(parent_dir);
  } catch (const fs::filesystem_error& e) {
    FS_LOG_ERROR("Failed to create directories: " << e.what());
    return false;
  }

  // Write to a temporary file to ensure atomic replace on rename
  // Include tmp_file_suffix so each thread uses a unique temporary file
  std::string tmp_path = target_path + tmp_file_suffix;

  std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary);
  if (!ofs) {
    FS_LOG_ERROR("Failed to open temporary file for writing: "
                   << tmp_path << " - " << std::strerror(errno));
    return false;
  }

  // Apply the custom buffer to the file stream
  ofs.rdbuf()->pubsetbuf(thread_write_buffer.data(), WRITE_BUFFER_SIZE);

  // Write file contents
  ofs.write(reinterpret_cast<const char*>(buf.ptr), buf.size);
  if (!ofs) {
    FS_LOG_ERROR("Failed to write to temporary file: "
                   << tmp_path << " - " << std::strerror(errno));
    std::remove(tmp_path.c_str());  // Clean up temp file
    return false;
  }

  ofs.flush();
  if (!ofs) {
    FS_LOG_ERROR("Failed to flush data to temporary file: "
                 << tmp_path << " - " << std::strerror(errno));
    return false;
  }

  // Atomically rename temp file to final target name after a successful write
  if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
    FS_LOG_ERROR("Failed to rename " << tmp_path << " to "
                 << target_path << " - " << std::strerror(errno));
    std::remove(tmp_path.c_str());
    return false;
  }

  return true;
}

// Read a file into a thread-local staging buffer
bool read_buffer_from_file(const std::string& path, StagingBufferInfo& buf) {
  // Open file
  std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs) {
    FS_LOG_ERROR("Failed to open file: " << path);
    return false;
  }

  // Determine file size
  std::ifstream::pos_type end_pos = ifs.tellg();
  if (end_pos == std::streampos(-1)) {
    FS_LOG_ERROR("Failed to determine file size: " << path);
    return false;
  }
  size_t file_size = static_cast<size_t>(end_pos);
  ifs.seekg(0, std::ios::beg);  // Move read pointer to start for reading

  // Acquire staging buffer of the required size
  if (!buf.ptr || buf.size < file_size) {
    FS_LOG_ERROR("Staging buffer too small for file: " << path
                 << " (required=" << file_size
                 << " available=" << buf.size
                 << " ptr=" << buf.ptr << ")");
    return false;
  }

  // Read file into Staging buffer
  ifs.read(reinterpret_cast<char*>(buf.ptr),
           static_cast<std::streamsize>(file_size));
  std::streamsize bytes_read = ifs.gcount();
  if (bytes_read != static_cast<std::streamsize>(file_size) || !ifs.good()) {
    FS_LOG_ERROR("Failed to read full file: " << path
                 << " (read " << bytes_read << "/" << file_size << " bytes)");
    return false;
  }

  return true;
}

// update_atime update only the atime of a file without changing mtime
void update_atime(const std::string& path) {
  struct timespec times[2];
  times[1].tv_nsec = UTIME_NOW;   // update atime to now
  times[0].tv_nsec = UTIME_OMIT;  // keep mtime unchanged
  utimensat(AT_FDCWD, path.c_str(), times, 0);
}

// Write via CPU staging - wraps copy_blocks + write_buffer_to_file
// This is the original logic from storage_offload.cpp
bool write_via_cpu_staged(TensorCopier& tensor_copier,
                          const std::vector<int64_t>& block_ids,
                          StagingBufferInfo& buf,
                          cudaStream_t stream,
                          const std::string& dst_file) {
  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_store = true;

  // Stage 1: copy tensors from GPU to staging CPU tensor
  TIME_EXPR("write phase 1: copy_blocks ",
            tensor_copier.copy_blocks(cpu_base, block_ids, is_store),
            "file: ",
            dst_file);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] write_via_cpu_staged: cudaStreamSynchronize failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Stage 2: Write the cpu tensor to disk
  bool success = TIME_EXPR("write phase 2: write_buffer_to_file",
                           write_buffer_to_file(buf, dst_file),
                           "file:",
                           dst_file,
                           " size:",
                           buf.size);

  if (!success) {
    std::cerr
        << "[ERROR] write_via_cpu_staged: Store failed during file write: "
        << dst_file << "\n";
  }

  return success;
}

// Read via CPU staging - wraps read_buffer_from_file + copy_blocks
// This is the original logic from storage_offload.cpp
bool read_via_cpu_staged(TensorCopier& tensor_copier,
                         const std::vector<int64_t>& block_ids,
                         StagingBufferInfo& buf,
                         cudaStream_t stream,
                         const std::string& src_file) {
  // Stage 1: Read file to staging CPU tensor
  bool success = TIME_EXPR("read phase 1: read_buffer_from_file",
                           read_buffer_from_file(src_file, buf),
                           "file:",
                           src_file);
  if (!success) {
    std::cerr
        << "[ERROR] read_via_cpu_staged: read_buffer_from_file failed for "
        << src_file << std::endl;
    return false;
  }

  // Stage 2: copy tensors from staging CPU tensor to GPU
  auto* cpu_base = static_cast<uint8_t*>(buf.ptr);
  bool is_store = false;

  success = TIME_EXPR("read phase 2: copy_cpu_tensor_to_gpu_tensors",
                      tensor_copier.copy_blocks(cpu_base, block_ids, is_store),
                      "file: ",
                      src_file);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] read_via_cpu_staged: cudaStreamSynchronize failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  return success;
}
