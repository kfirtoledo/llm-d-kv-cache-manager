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

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "storage_types.hpp"
#include "tensor_copier.hpp"

// Write a buffer to disk using a temporary file and atomic rename
bool write_buffer_to_file(const StagingBufferInfo& buf,
                          const std::string& target_path);

// Read a file into a thread-local staging buffer
bool read_buffer_from_file(const std::string& path, StagingBufferInfo& buf);

// update_atime update only the atime of a file without changing mtime
void update_atime(const std::string& path);

// Write via CPU staging - wraps copy_blocks + write_buffer_to_file
// This is the original logic from storage_offload.cpp
bool write_via_cpu_staged(TensorCopier& tensor_copier,
                          const std::vector<int64_t>& block_ids,
                          StagingBufferInfo& buf,
                          cudaStream_t stream,
                          const std::string& dst_file);

// Read via CPU staging - wraps read_buffer_from_file + copy_blocks
// This is the original logic from storage_offload.cpp
bool read_via_cpu_staged(TensorCopier& tensor_copier,
                         const std::vector<int64_t>& block_ids,
                         StagingBufferInfo& buf,
                         cudaStream_t stream,
                         const std::string& src_file);
