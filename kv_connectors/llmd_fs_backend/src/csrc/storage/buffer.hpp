#pragma once

#include <cstddef>
#include <vector>

struct PinnedBufferInfo {
    void* ptr = nullptr;
    size_t size = 0;
};

// Thread-local pinned pointers defined in buffer.cpp
extern thread_local PinnedBufferInfo t_pinned_buffer;

// Global preallocated pool per IO thread
extern std::vector<PinnedBufferInfo> g_pinned_buffers;

// Preallocate pinned buffers for IO threads
void preallocate_pinned_buffers(size_t io_threads, size_t pinned_buffer_size_mb);

// Return thread-local pinned buffer, allocating or reallocating if needed
std::pair<void*, size_t> get_thread_local_pinned(size_t required_bytes, int numa_node = -1);

// Return NUMA node that a GPU belongs to
int get_gpu_numa_node(int device_id);

// Return list of CPU cores in the given NUMA node
std::vector<int> get_cpus_in_numa_node(int node);