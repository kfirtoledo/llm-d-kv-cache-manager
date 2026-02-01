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

import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Check if cuFile library is available
def check_cufile_available():
    """Check if cuFile library is available on the system"""
    try:
        # Try to find libcufile.so
        result = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, check=False
        )
        if "libcufile.so" in result.stdout:
            return True

        # Also check common CUDA library paths
        cuda_lib_paths = [
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
        ]
        for path in cuda_lib_paths:
            cufile_path = os.path.join(path, "libcufile.so")
            if os.path.exists(cufile_path):
                return True

        return False
    except Exception:
        return False


# Detect GDS support
use_cufile = check_cufile_available()
enable_gds = os.environ.get("ENABLE_GDS", "1") == "1"
use_gds = use_cufile and enable_gds

# Build source list
sources = [
    "csrc/storage/storage_offload.cpp",
    "csrc/storage/storage_offload_bindings.cpp",
    "csrc/storage/numa_utils.cpp",
    "csrc/storage/file_io.cpp",
    "csrc/storage/thread_pool.cpp",
    "csrc/storage/tensor_copier.cu",
    "csrc/storage/tensor_copier_kernels.cu",
    "csrc/storage/gds_file_io.cpp",  # Always include GDS file
]

# Build libraries list
libraries = ["numa", "cuda"]
if use_gds:
    libraries.append("cufile")

# Build compile args
cxx_args = ["-O3", "-std=c++17", "-fopenmp"]
nvcc_args = [
    "-O3",
    "-std=c++17",
    "-Xcompiler",
    "-std=c++17",
    "-Xcompiler",
    "-fopenmp",
]

# Add GDS flag if available
if use_gds:
    cxx_args.append("-DUSE_CUFILE")
    nvcc_args.append("-DUSE_CUFILE")
    print("=" * 60)
    print("Building with GPUDirect Storage (GDS) support")
    print("=" * 60)
else:
    print("=" * 60)
    if not use_cufile:
        print("cuFile library not found - building without GDS support")
    elif not enable_gds:
        print("GDS disabled via ENABLE_GDS=0 - building without GDS support")
    print("Will use CPU buffer staging mode")
    print("=" * 60)

setup(
    name="llmd_fs_connector",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "storage_offload",
            sources=sources,
            libraries=libraries,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
