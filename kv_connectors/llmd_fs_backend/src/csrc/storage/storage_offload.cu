#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <future>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <optional>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <filesystem>
#include <numa.h>

#include "file_io.hpp"
#include "numa_utils.hpp"
#include "thread_pool.hpp"
#include "debug_utils.hpp"
#include "tensor_copy.hpp"
#include "cfg.hpp"

namespace fs = std::filesystem;
namespace py = pybind11;

// Tracks progress and results for a multi-file async PUT/GET job
struct JobState {
    // Futures for each async task in the job
    std::vector<std::shared_future<bool>> futures;
    // Number of tasks completed so far
    std::atomic<int> completed_tasks{0};
    // Total number of tasks scheduled for this job
    std::atomic<int> total_tasks{0};
    // Flag indicating if all tasks succeeded
    std::atomic<bool> all_success{true};
};

// --------------------------------------
// Global resources
// --------------------------------------
// CUDA streams assigned to each worker thread
std::vector<at::cuda::CUDAStream> g_streams_pool;
// Thread-local CUDA stream bound to this worker thread
thread_local std::optional<c10::cuda::CUDAStream> thread_stream;
// Maps this worker thread to its preallocated CUDA stream
thread_local size_t thread_stream_idx = 0;
// Mutex protecting access to the jobs map
static std::mutex jobs_mutex;
// Global map of job_id → JobState, tracking async job progress
std::map<int, std::unique_ptr<JobState>> jobs;

// Global IO thread pool for scheduling async PUT/GET tasks
static std::unique_ptr<ThreadPool> g_io_pool;
// global connector config instance
std::unique_ptr<ConnectorConfig> g_connector_config;
// -------------------------------
// Initialize resources with pre-allocation
// -------------------------------

// Initialize IO threads, CUDA streams, and staging memory pool
void init_resources(int io_threads,
                    size_t staging_buffer_size_mb,
                    size_t max_staging_memory_gb,
                    int tp_rank,
                    int gpu_blocks_per_file,
                    std::vector<torch::Tensor>& tensors,
                    bool kv_before_blocks,
                    bool layers_before_blocks,
                    int num_blocks_dimension) {
    // Build connector configuration
    CacheLayout layout(tensors,
                       num_blocks_dimension,
                       kv_before_blocks,
                       layers_before_blocks);
    g_connector_config =
        std::make_unique<ConnectorConfig>(gpu_blocks_per_file,
                                          staging_buffer_size_mb,
                                          layout);

    // Initialize IO thread pool if not already done
    if (!g_io_pool) {
        if (io_threads == 0) {
            io_threads = std::max(4u, std::thread::hardware_concurrency() / 2);
        }

        // Get current device (should be set by vLLM before calling this)
        int device_id;
        cudaGetDevice(&device_id);

        std::cout << "[INFO] Initializing ThreadPool with " << io_threads
                  << " threads on device " << device_id << ", "
                  << staging_buffer_size_mb << " MB staging buffer per thread, "
                  << max_staging_memory_gb << " GB max staging memory\n";

        // Enable GPU access to mapped host memory (needed only for
        // cudaHostAllocMapped before any CUDA context)
        cudaSetDeviceFlags(cudaDeviceMapHost);
        int gpu_numa = get_gpu_numa_node(device_id);
        numa_set_preferred(gpu_numa);

        // Pass device_id to thread pool
        g_io_pool = std::make_unique<ThreadPool>(io_threads,
                                                 staging_buffer_size_mb,
                                                 tp_rank,
                                                 device_id);

        // Create dedicated streams for each thread on the current device
        g_streams_pool.clear();
        g_streams_pool.reserve(io_threads);
        for (size_t i = 0; i < io_threads; i++) {
            g_streams_pool.push_back(
                at::cuda::getStreamFromPool(/*isHighPriority=*/false,
                                            device_id));
        }

        // Warm up CUDA context and streams
        for (auto& stream : g_streams_pool) {
            cudaStreamSynchronize(stream.stream());
        }
    }
}

// -------------------------------
// Status and cleanup
// -------------------------------
// Return finished jobs and their success status
std::vector<std::pair<int, bool>> get_finished() {
    std::lock_guard<std::mutex> lock(jobs_mutex);

    std::vector<std::pair<int, bool>> results;
    std::vector<int> to_erase;

    // Iterate over all active jobs.
    for (auto& kv : jobs) {
        int job_id = kv.first;
        auto& job_state = kv.second;

        // Check if the job has completed all its tasks.
        if (job_state->completed_tasks.load() ==
            job_state->total_tasks.load()) {
            bool all_ok = job_state->all_success.load();
            results.emplace_back(job_id, all_ok);
            to_erase.push_back(job_id);
        }
    }

    // Remove all finished jobs from the map.
    for (int jid : to_erase) {
        jobs.erase(jid);
    }
    return results;
}

// Release IO threads, CUDA streams, and staging buffer
void cleanup_resources() {
    g_io_pool.reset();
    g_streams_pool.clear();

    if (t_staging_buffer.ptr) {
        cudaFreeHost(t_staging_buffer.ptr);
        t_staging_buffer.ptr = nullptr;
        t_staging_buffer.size = 0;
    }
}

// Wait for all tasks in the specified job to complete
void wait_job(int job_id) {
    std::vector<std::shared_future<bool>> futures;

    {
        std::lock_guard<std::mutex> lock(jobs_mutex);
        auto it = jobs.find(job_id);
        if (it == jobs.end()) return;
        futures = it->second->futures;
    }

    for (auto& fut : futures) {
        fut.wait();
    }
}
// -------------------------------
// Put and Get operations
// -------------------------------
// Async GPU → Storage transfer (PUT)
bool transfer_async_put(int job_id,
                        std::vector<std::string> dst_files,
                        std::vector<torch::Tensor> src_tensors,
                        std::vector<std::vector<int64_t>> all_block_ids) {
    // Create job state object that will track progress and futures for this
    // job.
    auto job_state = std::make_unique<JobState>();
    job_state->total_tasks = dst_files.size();

    // Store shared_ptr to tensors to avoid repeated refcount changes
    auto shared_src_tensors =
        std::make_shared<std::vector<torch::Tensor>>(std::move(src_tensors));

    // For each dst_file file, enqueue one async task in the I/O thread pool.
    for (size_t i = 0; i < dst_files.size(); i++) {
        std::string dst_file = dst_files[i];
        auto bids = all_block_ids[i];

        auto future = g_io_pool->enqueue([dst_file,
                                          bids,
                                          shared_src_tensors,
                                          job_state =
                                              job_state.get()]() -> bool {
            // Check if dst_file file already exists - skip write if it does
            if (std::ifstream(dst_file).good()) {
                update_atime(dst_file);
                job_state->completed_tasks.fetch_add(1);
                return true;  // File exists
            }
            // Ensure correct device is set (thread-local)
            int device_id;
            cudaGetDevice(&device_id);

            // Each thread gets a dedicated CUDA stream for async GPU ops.
            if (!thread_stream.has_value()) {
                // thread_stream = g_streams_pool[thread_stream_idx %
                // g_streams_pool.size()];
                thread_stream = at::cuda::getStreamFromPool(
                    /* isHighPriority = */ false);  // use best-effort
                                                    // stream for writes
            }

            // Save current CUDA stream so we can restore it later.
            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            bool success = false;
            try {
                // Use reference to avoid copy - dereference shared_ptr
                const auto& src = *shared_src_tensors;
                torch::Tensor cpu_tensor;
                // Stage 1: copy tensors from GPU to staging CPU tensor.
                success = TIME_EXPR(
                    "write phase 1: copy_gpu_tensors_to_cpu_tensor",
                    copy_gpu_tensors_to_cpu_tensor(src,
                                                   bids,
                                                   cpu_tensor,
                                                   *thread_stream,
                                                   *g_connector_config),
                    "file: " + dst_file);

                cudaError_t err =
                    cudaStreamSynchronize(thread_stream->stream());
                if (err != cudaSuccess) {
                    std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                              << cudaGetErrorString(err) << std::endl;
                }

                if (!success) {
                    std::cerr << "[ERROR] PUT failed during GPU->CPU staging: "
                              << dst_file << "\n";
                } else {
                    // Stage 2: Write the cpu tensor to disk.
                    success = TIME_EXPR(
                        "write phase 2: write_tensor_to_file",
                        write_tensor_to_file(cpu_tensor, dst_file),
                        ("file:" + dst_file +
                         " size:" + std::to_string(cpu_tensor.nbytes())));

                    if (!success)
                        std::cerr << "[ERROR] PUT failed during file write: "
                                  << dst_file << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] PUT failed for " << dst_file << ": "
                          << e.what() << std::endl;
                success = false;
            } catch (...) {
                std::cerr << "[ERROR] PUT failed for " << dst_file << "\n";
                success = false;
            }

            // Restore original CUDA stream for safety.
            at::cuda::setCurrentCUDAStream(current_stream);

            // Mark task completion.
            job_state->completed_tasks.fetch_add(1);
            // if (!success) job_state->all_success = false; // TODO- silent
            // ignore write failures for now offloading connector not
            // able to handle failures
            return success;
        });
        // Convert std::future → std::shared_future, which is copyable and can
        // be waited on by multiple threads.
        job_state->futures.push_back(future.share());
    }

    std::lock_guard<std::mutex> lock(jobs_mutex);  // protect jobs map
    jobs[job_id] = std::move(job_state);

    return true;
}

// Async Storage → GPU transfer (GET)
bool transfer_async_get(int job_id,
                        std::vector<std::string> src_files,
                        std::vector<std::vector<int64_t>> all_block_ids,
                        std::vector<torch::Tensor> dst_tensors) {
    // Create job state object to track progress and futures for this job.
    auto job_state = std::make_unique<JobState>();
    job_state->total_tasks = src_files.size();

    // For each source file, enqueue one async task in the I/O thread pool.
    for (size_t i = 0; i < src_files.size(); i++) {
        std::string src_file = src_files[i];
        auto block_ids = all_block_ids[i];
        auto future = g_io_pool->enqueue([=,
                                          job_state =
                                              job_state.get()]() -> bool {
            // Get dedicated stream for this thread
            if (!thread_stream.has_value()) {
                thread_stream =
                    g_streams_pool[thread_stream_idx % g_streams_pool.size()];
                // thread_stream = at::cuda::getStreamFromPool(/* isHighPriority
                // = */ true); // use high-priority stream for reads
            }

            // Save current CUDA stream so we can restore it later.
            auto current_stream = at::cuda::getCurrentCUDAStream();
            at::cuda::setCurrentCUDAStream(*thread_stream);

            bool success = false;
            try {
                // Stage 1: Read file to staging CPU tensor.
                torch::Tensor cpu_tensor;
                // Read data from disk into a tensor.
                success = TIME_EXPR("read phase 1: read_tensor_from_file",
                                    read_tensor_from_file(src_file, cpu_tensor),
                                    ("file:" + src_file));
                if (!success) {
                    std::cerr
                        << "[ERROR] Stage1 read_tensor_from_file failed for "
                        << src_file << std::endl;
                } else {
                    // Stage 2:  copy tensors from staging CPU tensor to GPU.
                    // Perform asynchronous GPU copy and tensor swap.
                    success = TIME_EXPR(
                        "read phase 2: copy_cpu_tensor_to_gpu_tensors",
                        copy_cpu_tensor_to_gpu_tensors(cpu_tensor,
                                                       block_ids,
                                                       dst_tensors,
                                                       *thread_stream,
                                                       *g_connector_config),
                        "file: " + src_file);
                    cudaError_t err =
                        cudaStreamSynchronize(thread_stream->stream());
                    if (err != cudaSuccess) {
                        std::cerr << "[ERROR] cudaStreamSynchronize failed: "
                                  << cudaGetErrorString(err) << std::endl;
                        success = false;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] GET failed for " << src_file << ": "
                          << e.what() << std::endl;
                success = false;
            } catch (...) {
                std::cerr << "[ERROR] GET unknown failure for " << src_file
                          << std::endl;
                success = false;
            }

            // Final cleanup & accounting
            // Synchronize only this thread's CUDA stream.
            at::cuda::setCurrentCUDAStream(current_stream);
            job_state->completed_tasks.fetch_add(1);
            if (!success) job_state->all_success = false;
            return success;
        });

        // Convert std::future → std::shared_future- is copyable and can be
        // waited on by multiple threads.
        job_state->futures.push_back(future.share());
    }

    std::lock_guard<std::mutex> lock(jobs_mutex);
    jobs[job_id] = std::move(job_state);
    return true;
}

// -------------------------------
// PYBIND11 module
// -------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_resources",
          &init_resources,
          py::arg("io_threads"),
          py::arg("staging_buffer_size_mb"),
          py::arg("max_staging_memory_gb"),
          py::arg("tp_rank"),
          py::arg("gpu_blocks_per_file"),
          py::arg("tensors"),
          py::arg("kv_before_blocks"),
          py::arg("layers_before_blocks"),
          py::arg("num_blocks_dimension"));

    m.def("cleanup_resources", &cleanup_resources);

    m.def("get_finished", &get_finished);

    m.def("transfer_async_put",
          &transfer_async_put,
          py::arg("job_id"),
          py::arg("dst_files"),
          py::arg("src_tensors"),
          py::arg("all_block_ids"));

    m.def("transfer_async_get",
          &transfer_async_get,
          py::arg("job_id"),
          py::arg("src_files"),
          py::arg("all_block_ids"),
          py::arg("dst_tensors"));

    m.def("wait_job", &wait_job, py::arg("job_id"));
}
