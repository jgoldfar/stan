#ifndef STAN_CALLBACKS_MULTI_WRITER_HPP
#define STAN_CALLBACKS_MULTI_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/prim/functor/for_each.hpp>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
namespace callbacks {

/**
 * `multi_writer` is an layer on top of a writer class that
 *  allows for multiple output streams to be written to.
 * @tparam Writers A parameter pack of types that inherit from `writer`
 */
template <typename... Writers>
class multi_writer {
 public:
  /**
   * Constructs a multi stream writer from a parameter pack of writers.
   *
   * @param[in, out] args A parameter pack of writers
   */
  template <typename... Args>
  explicit multi_writer(Args&&... args)
      : output_(std::forward<Args>(args)...) {}

  multi_writer();

  /**
   * @tparam T Any type accepted by a `writer` overload
   * @param[in] x A value to write to the output streams
   */
  template <typename T>
  void operator()(T&& x) {
    stan::math::for_each([&](auto&& output) { output(x); }, output_);
  }
  void operator()() {
    stan::math::for_each([](auto&& output) { output(); }, output_);
  }

  /**
   * Checks if all underlying writers are nonnull.
   */
  inline bool is_nonnull() const noexcept {
    return stan::math::apply([](auto&&... output) { return (output.is_nonnull() && ...); },
                         output_);
  }

  /**
   * Get the underlying stream
   */
  inline auto& get_stream() noexcept { return output_; }
  const char* comment_prefix() const noexcept { return std::get<0>(output_).comment_prefix(); }

 private:
  /**
   * Output stream
   */
  std::tuple<std::reference_wrapper<Writers>...> output_;
};

namespace internal {
template <typename T>
struct is_multi_writer : std::false_type {};

template <typename... Types>
struct is_multi_writer<multi_writer<Types...>> : std::true_type {};
}

template <typename T>
inline constexpr bool is_multi_writer_v = internal::is_multi_writer<std::decay_t<T>>::value;

}  // namespace callbacks
}  // namespace stan

#endif
