#pragma once

#include <string>
#include <torch/extension.h>

// Write a tensor to disk using a temporary file and atomic rename
bool write_file_to_disk(const std::string& target_path,
                        const torch::Tensor& host_buf);

// Read a file into a pinned CPU tensor using the thread-local pinned buffer
torch::Tensor read_file_from_disk(const std::string& path);

// update_atime update only the atime of a file without changing mtime
void update_atime(const std::string& path);