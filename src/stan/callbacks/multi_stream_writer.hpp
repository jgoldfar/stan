#ifndef STAN_CALLBACKS_MULTI_STREAM_WRITER_HPP
#define STAN_CALLBACKS_MULTI_STREAM_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
namespace callbacks {

/**
 * `multi_stream_writer` is an implementation
 * of `writer` that holds a unique pointer to the stream it is
 * writing to.
 * @tparam Stream A type with with a valid `operator<<(std::string)`
 * @tparam Deleter A class with a valid `operator()` method for deleting the
 * output stream
 */
template <typename... Streams>
class multi_stream_writer {
 public:
  /**
   * Constructs a multi stream writer with an output stream
   * and an optional prefix for comments.
   *
   * @param[in, out] output A unique pointer to a type inheriting from
   * `std::ostream`
   * @param[in] comment_prefix string to stream before each comment line.
   *  Default is "".
   */
  template <typename... Args>
  explicit multi_stream_writer(Args&&... args)
      : output_(std::forward<Args>(args)...) {}

  multi_stream_writer();
  /**
   * Writes a set of names on a single line in csv format followed
   * by a newline.
   *
   * Note: the names are not escaped.
   *
   * @param[in] names Names in a std::vector
   */
  template <typename T>
  void operator()(T&& x) {
    stan::math::for_each([&](auto&& output) {
      output(x);
    }, output_);
  }
  void operator()() {
    stan::math::for_each([](auto&& output) {
      output();
    }, output_);
  }

  /**
   * Get the underlying stream
   */
  inline auto& get_stream() noexcept { return output_; }


 private:
  /**
   * Output stream
   */
  std::tuple<std::reference_wrapper<Streams>...> output_;

};

}  // namespace callbacks
}  // namespace stan

#endif
