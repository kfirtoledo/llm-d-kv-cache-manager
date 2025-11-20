#!/bin/bash

set -e

POD_NAME="ubuntu"
REMOTE_BASE="/vllm-workspace/kv-connectors/llmd_fs_backend"

# List of files to copy (relative paths)
FILES=(
  # Python files
  "src/llmd_fs_backend/worker.py"
  "src/llmd_fs_backend/factory.py"
  "src/llmd_fs_backend/spec.py"
  "src/llmd_fs_backend/manager.py"
  "src/llmd_fs_backend/mediums.py"

  # Tests
  "tests/test_shared_storage.py"

  # CUDA extension files
  "src/csrc/storage/storage_offload.cu"
  "src/csrc/storage/buffer.cpp"
  "src/csrc/storage/buffer.hpp"
  "src/csrc/storage/debug_utils.hpp"
  "src/csrc/storage/thread_pool.cpp"
  "src/csrc/storage/thread_pool.hpp"
  "src/csrc/storage/file_io.hpp"
  "src/csrc/storage/file_io.cpp"
  "src/csrc/storage/tensor_copy.cu"
  "src/csrc/storage/tensor_copy.hpp"

  # Build script
  "setup.py"
)

echo "Copying files to pod: $POD_NAME"
echo "Destination base: $REMOTE_BASE"
kubectl exec "$POD_NAME" -- mkdir -p "$REMOTE_BASE/src/csrc/storage/"
kubectl exec "$POD_NAME" -- mkdir -p "$REMOTE_BASE/src/llmd_fs_backend/"
kubectl exec "$POD_NAME" -- mkdir -p "$REMOTE_BASE/tests/"

echo "---------------------------------------------------------"

for file in "${FILES[@]}"; do
  # Compute remote directory
  REMOTE_DIR="$REMOTE_BASE/$(dirname "$file")"

  # echo "Creating directory in pod: $REMOTE_DIR"
  # kubectl exec "$POD_NAME" -- mkdir -p "$REMOTE_DIR"

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
