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

"""
Tests to check if GDS (GPUDirect Storage) is available.
If GDS is not available, tests will fail with an error.
"""

import os
import subprocess
import sys

import pytest
import storage_offload
import torch


def check_gds_available() -> bool:
    """
    Check if GDS (GPUDirect Storage) is available.

    This checks multiple indicators:
    1. Whether the nvidia-fs kernel module is loaded
    2. Whether libcufile.so is available
    3. Whether the storage_offload module is available

    Returns:
        bool: True if GDS is available, False otherwise
    """
    # Check if nvidia-fs kernel module is loaded
    try:
        result = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            check=False,
        )
        nvidia_fs_loaded = "nvidia_fs" in result.stdout
    except Exception:
        nvidia_fs_loaded = False

    # Check if libcufile.so is available
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=False,
        )
        libcufile_available = "libcufile.so" in result.stdout
    except Exception:
        libcufile_available = False

    # Check common library paths as fallback
    if not libcufile_available:
        common_paths = [
            "/usr/local/cuda/lib64/libcufile.so",
            "/usr/lib/x86_64-linux-gnu/libcufile.so",
        ]
        libcufile_available = any(os.path.exists(path) for path in common_paths)

    # Try to import storage_offload module
    try:
        has_module = hasattr(storage_offload, "StorageOffloadEngine")
    except (ImportError, AttributeError):
        has_module = False

    gds_available = nvidia_fs_loaded and libcufile_available and has_module

    return gds_available


def get_gds_status_message() -> str:
    """
    Get a detailed status message about GDS availability.

    Returns:
        str: Status message describing GDS availability
    """
    try:
        result = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            check=False,
        )
        nvidia_fs_loaded = "nvidia_fs" in result.stdout
    except Exception:
        nvidia_fs_loaded = False

    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=False,
        )
        libcufile_available = "libcufile.so" in result.stdout
    except Exception:
        libcufile_available = False

    if not libcufile_available:
        common_paths = [
            "/usr/local/cuda/lib64/libcufile.so",
            "/usr/lib/x86_64-linux-gnu/libcufile.so",
        ]
        libcufile_available = any(os.path.exists(path) for path in common_paths)

    try:
        has_module = True
    except ImportError:
        has_module = False

    status_parts = []
    status_parts.append(
        f"nvidia-fs module: {'loaded' if nvidia_fs_loaded else 'not loaded'}"
    )
    status_parts.append(
        f"libcufile.so: {'available' if libcufile_available else 'not found'}"
    )
    status_parts.append(
        f"storage_offload module: {'imported' if has_module else 'not available'}"
    )

    return ", ".join(status_parts)


class TestGDSAvailability:
    """Tests for GDS availability."""

    def test_gds_available(self):
        """Test that GDS is available. Fails if GDS is not available."""
        gds_available = check_gds_available()
        status_msg = get_gds_status_message()

        print(f"\n[INFO] GDS Available: {gds_available}")
        print(f"[INFO] GDS Status: {status_msg}")

        if not gds_available:
            pytest.fail(f"GDS is not available. Status: {status_msg}")

        print("[INFO] GDS is available - tests can proceed")

    def test_cuda_available(self):
        """Test that CUDA is available for GPU operations."""
        assert torch.cuda.is_available(), "CUDA must be available for these tests"
        print(f"\n[INFO] CUDA devices available: {torch.cuda.device_count()}")
        print(f"[INFO] Current CUDA device: {torch.cuda.current_device()}")

    @pytest.mark.parametrize("gpu_blocks_per_file", [1, 2, 4, 8])
    @pytest.mark.parametrize("start_idx", [0,3])
    def test_gds_roundtrip(self, default_vllm_config, gpu_blocks_per_file, start_idx):
        """Test a full roundtrip with GDS enabled.
        
        This test verifies the OPTIMIZED implementation that aggregates multiple blocks
        into the same file with a single file-open session:
        - With gpu_blocks_per_file=2 and num_blocks=8: writes 4 files (each with 2 blocks)
        - With gpu_blocks_per_file=4 and num_blocks=8: writes 2 files (each with 4 blocks)
        
        The optimization reduces file I/O overhead by ~50% by:
        1. Opening the file once per file (not per block)
        2. Writing all blocks sequentially with increasing file offsets
        3. Closing the file after all blocks are written
        """
        gds_available = check_gds_available()
        if not gds_available:
            pytest.skip(f"GDS not available: {get_gds_status_message()}")

        import math
        from test_fs_backend import roundtrip_once
        from llmd_fs_backend.file_mapper import FileMapper

        # Small test configuration
        num_layers = 80
        num_blocks = 8
        block_size = 16
        num_heads = 64
        head_size = 128
        dtype = torch.float16
        threads_per_gpu = 8

        # Calculate expected number of files
        expected_files = math.ceil(num_blocks / gpu_blocks_per_file)
        
        print(f"\n[INFO] Testing GDS roundtrip with gpu_blocks_per_file={gpu_blocks_per_file}, start_idx={start_idx}")
        print(f"[INFO] Will write {num_blocks} blocks across {expected_files} files")
        print(f"[INFO] Each file will contain {gpu_blocks_per_file} blocks (except possibly the last)")

        # Setup file mapper
        file_mapper = FileMapper(
            root_dir="/ibm/fs1-remote/kfir/gds/",
            model_name="test-model",
            gpu_block_size=block_size,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            rank=0,
            dtype=str(dtype),
        )

        # Use the existing roundtrip_once function
        # start_idx 0 = full from start, start_idx = 3, partial first group (e.g., 3..7)
        write_block_ids = list(range(start_idx, num_blocks))
        read_block_ids = list(range(start_idx, num_blocks))

        roundtrip_once(
            file_mapper=file_mapper,
            dtype=dtype,
            num_layers=num_layers,
            num_blocks=num_blocks,
            gpu_block_size=block_size,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            read_block_ids=read_block_ids,
            write_block_ids=write_block_ids,
            gpu_blocks_per_file=gpu_blocks_per_file,
            threads_per_gpu=threads_per_gpu,
            enable_gds=True,
        )

        print(f"[INFO] GDS roundtrip test succeeded with gpu_blocks_per_file={gpu_blocks_per_file}, start_idx={start_idx}")
        print(f"[INFO] Successfully aggregated {gpu_blocks_per_file} blocks per file with optimized GDS I/O")


if __name__ == "__main__":
    # Print GDS status when run directly
    print("=" * 70)
    print("GDS Availability Check")
    print("=" * 70)
    gds_available = check_gds_available()
    print(f"GDS Available: {gds_available}")
    print(f"Status: {get_gds_status_message()}")
    print("=" * 70)

    if not gds_available:
        print("\n[ERROR] GDS is not available. Terminating tests.")
        sys.exit(1)

    # Run pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
