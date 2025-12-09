#!/bin/bash
# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
POD_NAME="ubuntu"
REMOTE_BASE="/workspace/vllm/llm-d-kv-cache-manager/kv_connectors/llmd_fs_backend/"
#REMOTE_BASE="/vllm-workspace/llm-d-kv-cache-manager/kv_connectors/llmd_fs_backend/"
# List of files to copy (relative paths)
FILES=(
  # Python files
  "src/llmd_fs_backend/worker.py"
  # "src/llmd_fs_backend/factory.py"
  "src/llmd_fs_backend/spec.py"
  # "src/llmd_fs_backend/manager.py"
  # "src/llmd_fs_backend/mediums.py"
  # Tests
  # "tests/test_fs_backend.py"
  # CUDA extension files
  # "src/csrc/storage/storage_offload.cu"
  # "src/csrc/storage/buffer.cpp"
  # "src/csrc/storage/buffer.hpp"
  # "src/csrc/storage/debug_utils.hpp"
  # "src/csrc/storage/thread_pool.cpp"
  # "src/csrc/storage/thread_pool.hpp"
  # "src/csrc/storage/file_io.hpp"
  # "src/csrc/storage/file_io.cpp"
  # "src/csrc/storage/tensor_copy.cu"
  # "src/csrc/storage/tensor_copy.hpp"
  # Build script
  "setup.py"
)
echo "Copying files to pod: $POD_NAME"
echo "Destination base: $REMOTE_BASE"
echo "---------------------------------------------------------"
for file in "${FILES[@]}"; do
  # Compute remote directory
  REMOTE_DIR="$REMOTE_BASE/$(dirname "$file")"
  # Copy file
  echo "Copying: $file → $POD_NAME:$REMOTE_BASE/$file"
  kubectl cp "$file" "$POD_NAME":"$REMOTE_BASE/$file"
done
echo "---------------------------------------------------------"
echo "All files copied successfully."
echo "To enter the pod:"
echo "  kubectl exec -it $POD_NAME -- bash"
echo ""
echo "Connector is now inside: $REMOTE_BASE"
