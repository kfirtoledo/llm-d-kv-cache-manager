from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="storage_offload",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            "storage_offload",
             sources=[
                "src/csrc/storage/storage_offload.cu",
                "src/csrc/storage/buffer.cpp",
                "src/csrc/storage/file_io.cpp",
                "src/csrc/storage/thread_pool.cpp",
                "src/csrc/storage/tensor_copy.cu",
            ],
            libraries=['nvidia-ml', 'numa', 'cuda'],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fopenmp"],
                "nvcc": ["-O3", "-std=c++17", "-Xcompiler", "-std=c++17","-Xcompiler", "-fopenmp"]
            }
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
