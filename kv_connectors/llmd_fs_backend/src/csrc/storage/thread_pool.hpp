#pragma once
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <sys/syscall.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "buffer.hpp"       // defines PinnedBuffer + get_gpu_numa_node + get_cpus_in_numa_node
#include "debug_utils.hpp"

// Thread-local storage used by each I/O thread
extern thread_local size_t thread_stream_idx;

// ThreadPool class
class ThreadPool {
public:
    ThreadPool(int threads,
               size_t pinned_buffer_mb,
               int tp_rank,
               int device_id);

    ~ThreadPool();

    template<class F> auto enqueue(F&& f) -> std::future<std::invoke_result_t<F>>;

private:
    std::vector<std::thread> workers;             // All worker threads
    std::queue<std::function<void()>> tasks;      // Queue of pending tasks

    std::mutex queue_mutex;                       // Protects access to the task queue
    std::condition_variable condition;            // Signals workers when tasks are available

    std::atomic<bool> stop{false};                // Tells workers to stop and exit
    int m_device_id;                              // CUDA device this thread pool is bound to
};

// enqueue: submit a task to the thread pool
template<class F> auto ThreadPool::enqueue(F&& f) -> std::future<std::invoke_result_t<F>>
{
    // Get the return type of the submitted task
    using return_type = std::invoke_result_t<F>;

    // Wrap the callable into a packaged_task so we can return a future
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::forward<F>(f)
    );

    // Future for the caller to wait on
    std::future<return_type> res = task->get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Reject new tasks if the pool is shutting down
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        // Push the task wrapper into the queue
        tasks.emplace([task]() { (*task)(); });
    }

    // Wake one worker thread to process the task
    condition.notify_one();

    return res;
}