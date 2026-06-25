#!/usr/bin/env bash
# Reproduce llm-d/llm-d-kv-cache#656: fs_backend HMA store-path assertion on
# gemma-4 video inference. Single node, 4x H100, HMA ON, default block_size
# (so block_size_factor > 1 -> the interior-hole case can occur).
#
# Prereqs (already satisfied on this box):
#   - google/gemma-4-31B-it in the HF cache (gemma-3 has NO video support here)
#   - cv2 (OpenCV) for video decode  (or: pip install av)
#   - llmd_fs_backend connector importable (editable install)
#   - the patched vLLM at /home/kfirt/storage/project/vllm (editable)
set -euo pipefail

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
mkdir -p /tmp/prometheus_metrics
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_metrics

STORAGE=${STORAGE:-/home/kfirt/storage/llmd-kv-repro}
mkdir -p "$STORAGE"

# To compare against the WORKAROUND (no bug), add  "block_size":"16"  to
# kv_connector_extra_config below -> block_size_factor=1, no interior holes.

vllm serve google/gemma-4-31B-it \
  --kv-transfer-config '{
       "kv_connector": "OffloadingConnector",
       "kv_role": "kv_both",
       "kv_connector_extra_config": {
         "spec_name": "SharedStorageOffloadingSpec",
         "spec_module_path": "llmd_fs_backend.spec",
         "shared_storage_path": "'"$STORAGE"'",
         "threads_per_gpu": "32"
       }
     }' \
  --distributed_executor_backend mp \
  --port 8000 \
  --max_num_batched_tokens 16384 \
  --enable-chunked-prefill \
  --max-model-len 200000 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 4 \
  --enable_prefix_caching \
  --enforce-eager \
  --no-disable-hybrid-kv-cache-manager
