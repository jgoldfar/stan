#ifndef STAN_SERVICES_ARGUMENTS_ARG_EXHAUSTIVE_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_EXHAUSTIVE_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_max_depth.hpp>

namespace stan {
  namespace services {

    class arg_exhaustive: public categorical_argument {
    public:
      arg_exhaustive() {
        _name = "exhaustive";
        _description = "Exhaustive Hamiltonian Monte Carlo";

        _subarguments.push_back(new arg_max_depth());
      }
    };

  }  // services
}  // stan

#endif
