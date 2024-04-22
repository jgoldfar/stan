#include <stan/io/empty_var_context.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <test/unit/mcmc/hmc/mock_hmc.hpp>
#include <test/test-models/good/mcmc/hmc/hamiltonians/funnel.hpp>
#include <test/unit/util.hpp>

#include <gtest/gtest.h>

#include <string>

TEST(McmcSoftAbs, sample_p) {
  stan::rng_t base_rng = stan::services::util::create_rng(0, 0);

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;

  stan::mcmc::mock_model model(q.size());
  stan::mcmc::softabs_metric<stan::mcmc::mock_model, stan::rng_t> metric(model);
  stan::mcmc::softabs_point z(q.size());

  int n_samples = 1000;
  double m = 0;
  double m2 = 0;

  std::stringstream model_output;
  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  metric.update_metric(z, logger);

  for (int i = 0; i < n_samples; ++i) {
    metric.sample_p(z, base_rng);
    double tau = metric.tau(z);

    double delta = tau - m;
    m += delta / static_cast<double>(i + 1);
    m2 += delta * (tau - m);
  }

  double var = m2 / (n_samples + 1.0);

  // Mean within 5sigma of expected value (d / 2)
  EXPECT_TRUE(std::fabs(m - 0.5 * q.size()) < 5.0 * sqrt(var));

  // Variance within 10% of expected value (d / 2)
  EXPECT_TRUE(std::fabs(var - 0.5 * q.size()) < 0.1 * q.size());

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcSoftAbs, gradients) {
  Eigen::VectorXd q = Eigen::VectorXd::Ones(11);

  stan::mcmc::softabs_point z(q.size());
  z.q = q;
  z.p.setOnes();

  stan::io::empty_var_context data_var_context;

  std::stringstream model_output;
  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger(debug, info, warn, error, fatal);

  funnel_model_namespace::funnel_model model(data_var_context, 0,
                                             &model_output);

  stan::mcmc::softabs_metric<funnel_model_namespace::funnel_model, stan::rng_t>
      metric(model);

  double epsilon = 1e-6;

  metric.init(z, logger);
  Eigen::VectorXd g1 = metric.dtau_dq(z, logger);

  for (int i = 0; i < z.q.size(); ++i) {
    double delta = 0;

    z.q(i) += epsilon;
    metric.init(z, logger);
    delta += metric.tau(z);

    z.q(i) -= 2 * epsilon;
    metric.init(z, logger);
    delta -= metric.tau(z);

    z.q(i) += epsilon;

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g1(i), epsilon);
  }

  metric.init(z, logger);
  Eigen::VectorXd g2 = metric.dtau_dp(z);

  for (int i = 0; i < z.q.size(); ++i) {
    double delta = 0;

    z.p(i) += epsilon;
    delta += metric.tau(z);

    z.p(i) -= 2 * epsilon;
    delta -= metric.tau(z);

    z.p(i) += epsilon;

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g2(i), epsilon);
  }

  Eigen::VectorXd g3 = metric.dphi_dq(z, logger);

  for (int i = 0; i < z.q.size(); ++i) {
    double delta = 0;

    z.q(i) += epsilon;
    metric.init(z, logger);
    delta += metric.phi(z);

    z.q(i) -= 2 * epsilon;
    metric.init(z, logger);
    delta -= metric.phi(z);

    z.q(i) += epsilon;

    delta /= 2 * epsilon;

    EXPECT_NEAR(delta, g3(i), epsilon);
  }

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST(McmcSoftAbs, streams) {
  stan::test::capture_std_streams();

  Eigen::VectorXd q(2);
  q(0) = 5;
  q(1) = 1;
  stan::mcmc::mock_model model(q.size());

  // for use in Google Test macros below
  typedef stan::mcmc::softabs_metric<stan::mcmc::mock_model, stan::rng_t>
      softabs;

  EXPECT_NO_THROW(softabs metric(model));

  stan::test::reset_std_streams();
  EXPECT_EQ("", stan::test::cout_ss.str());
  EXPECT_EQ("", stan::test::cerr_ss.str());
}
