#ifndef STAN_CALLBACKS_CONCURRENT_WRITER_HPP
#define STAN_CALLBACKS_CONCURRENT_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <tbb/concurrent_queue.h>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace stan::callbacks {
#ifdef STAN_THREADS
/**
 * Takes a writer and makes it thread safe via multiple queues.
 * On construction, a single busy thread is spawned to write to the writer.
 * This class uses an `std::thread` instead of a tbb task graph because
 * of deadlocking issues. A deadlock in two major cases.
 * 1. If TBB gives all threads to the
 * parallel for loop, and all threads hit an instance of max capacity. TBB can
 * choose to wait for a thread to finish instead of spinning up the write
 * thread. So to circumvent that issue, we use an std::thread.
 * 2. If the bounded queues are full but the queue reader thread is blocked.
 * The queue reader thread is blocked because the queues are full. The other threads
 * are blocked because the queue reader thread is blocked. The queue reader thread is blocked
 * because the other threads are blocked. This is a deadlock. To circumvent this
 * issue, we check in the queue reader thread if the queues are full and if they are,
 * we set `block_` to true which blocks all other threads from attempting to write
 * to the queues. 
 * @tparam Writer A type that inherits from `writer`
 */
template <typename Writer>
struct concurrent_writer {
  // A reference to the writer to write to
  std::reference_wrapper<Writer> writer;
  // Number of null writes queued
  std::atomic<int> null_writes_queued{0};
  // Queue for string messages
  tbb::concurrent_bounded_queue<std::string> str_messages_{};
  // Queue for vector of strings messages
  tbb::concurrent_bounded_queue<std::vector<std::string>> vec_str_messages_{};
  // Queue for Eigen vector messages
  tbb::concurrent_bounded_queue<Eigen::RowVectorXd> eigen_messages_{};
  // Flag to stop the writing thread once all queues are empty
  bool continue_writing_{true};
  // Flag to block threads from writing to queues if the queues are full
  std::atomic<bool> block_{false};
  // The writing thread
  std::thread thread_;
  /**
   * Constructs a concurrent writer from a writer and spins up a thread for
   * writing.
   * @param writer A writer to write to
   */
  explicit concurrent_writer(Writer& writer) : writer(writer) {
    str_messages_.set_capacity(1000);
    vec_str_messages_.set_capacity(1000);
    eigen_messages_.set_capacity(1000);
    thread_ = std::thread([&]() {
      std::string str;
      std::vector<std::string> vec_str;
      Eigen::RowVectorXd eigen;
      while (continue_writing_
             || !(str_messages_.empty() && vec_str_messages_.empty()
                  && eigen_messages_.empty() && null_writes_queued == 0)) {
        if (!(str_messages_.size() >= 999 || vec_str_messages_.size() >= 999
        || eigen_messages_.size() >= 999 || null_writes_queued >= 10)) {
          block_ = true;
        }
        bool processed = !(str_messages_.empty() && vec_str_messages_.empty()
        && eigen_messages_.empty() && null_writes_queued == 0);
        while (null_writes_queued > 0) {
          auto num_null_writes
              = null_writes_queued.load(std::memory_order_relaxed);
          for (int i = 0; i < num_null_writes; ++i) {
            writer();
          }
          null_writes_queued.fetch_sub(num_null_writes,
                                       std::memory_order_relaxed);
        }
        while (str_messages_.try_pop(str)) {
          writer(str);
        }
        while (vec_str_messages_.try_pop(vec_str)) {
          writer(vec_str);
        }
        while (eigen_messages_.try_pop(eigen)) {
          writer(eigen);
        }
        if (!processed) {
          std::this_thread::yield();
        }
        block_ = false;
      }
    });
  }
  /**
   * Place a value in a queue for writing.
   * @note This function will block if the queues are full
   * @tparam T Either an `std::vector<std::string|double>`, an Eigen vector, or
   * a string
   * @param t A value to put on a queue
   */
  template <typename T>
  void operator()(T&& t) {
    bool pushed = false;
    while(block_) {
      std::this_thread::yield();
    }
    while (!pushed) {
    if constexpr (stan::is_std_vector<T>::value) {
      if constexpr (std::is_arithmetic_v<stan::value_type_t<T>>) {
        pushed = eigen_messages_.try_push(Eigen::RowVectorXd::Map(t.data(), t.size()));
      } else {
        pushed = vec_str_messages_.try_push(t);
      }
    } else if constexpr (std::is_same_v<T, std::string>) {
      pushed = str_messages_.try_push(std::forward<T>(t));
    } else if constexpr (stan::is_eigen_vector<T>::value) {
      pushed = eigen_messages_.try_push(std::forward<T>(t));
    } else {
      throw std::domain_error(
          "Unsupported type passed to concurrent_writer. This is an "
          "internal error. Please file an issue on the stan github "
          "repository with the error log from the compiler.\n"
          "https://github.com/stan-dev/stan/issues/new?template=Blank+issue");
    }
    if (!pushed) {
      std::this_thread::yield();
    }
    }
  }
  /**
   * Writes a comment prefix to the writer.
   */
  void operator()() { null_writes_queued++; }
  /**
   * Waits till all writes are finished on the thread
   */
  void wait() {
    continue_writing_ = false;
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  ~concurrent_writer() { wait(); }
};
#else
/**
 * When STAN_THREADS is not defined, the concurrent writer is just a wrapper
 */
template <typename Writer>
struct concurrent_writer {
  std::reference_wrapper<Writer> writer;
  explicit concurrent_writer(Writer& writer) : writer(writer) {}
  template <typename T>
  void operator()(T&& t) {
    writer(std::forward<T>(t));
  }
  void operator()() { writer(); }
  inline static constexpr void wait() {}
};
#endif
}  // namespace stan::callbacks
#endif
