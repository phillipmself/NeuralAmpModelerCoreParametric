// Tests for Hypernetwork conditioning logic

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "NAM/hypernet.h"

namespace test_hypernet
{
namespace
{

nam::ParamSpec make_continuous_spec(const std::string& name, const float min, const float max,
                                    const float default_value)
{
  return nam::ParamSpec{name, min, max, default_value, "continuous", {}};
}

nam::ParamSpec make_switch_spec(const std::string& name, const std::vector<std::string>& enum_names,
                                const int default_value)
{
  return nam::ParamSpec{
    name, 0.0f, static_cast<float>(enum_names.size() - 1), static_cast<float>(default_value), "switch", enum_names};
}

nam::Hypernetwork::Target make_full_target(const std::string& name, const int numel, const int export_offset)
{
  return nam::Hypernetwork::Target{name, false, numel, export_offset, 0, 0, 0, numel};
}

nam::Hypernetwork::Target make_low_rank_target(const std::string& name, const int export_offset, const int rank,
                                               const int out_features, const int rest_features)
{
  return nam::Hypernetwork::Target{
    name,         true,          out_features * rest_features,         export_offset, rank,
    out_features, rest_features, rank * (out_features + rest_features)};
}

void assert_vector_near(const std::vector<float>& actual, const std::vector<float>& expected,
                        const float tolerance = 1.0e-6f)
{
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i)
    assert(std::abs(actual[i] - expected[i]) <= tolerance);
}

template <typename Fn>
void assert_throws_invalid_argument(Fn&& fn)
{
  auto threw = false;
  try
  {
    fn();
  }
  catch (const std::invalid_argument&)
  {
    threw = true;
  }
  assert(threw);
}

std::vector<float> flatten_row_major(const std::vector<std::vector<float>>& matrix)
{
  std::vector<float> flat;
  for (const auto& row : matrix)
    flat.insert(flat.end(), row.begin(), row.end());
  return flat;
}

std::vector<float> multiply_row_major(const std::vector<std::vector<float>>& a,
                                      const std::vector<std::vector<float>>& b)
{
  assert(!a.empty());
  assert(!b.empty());
  assert(a[0].size() == b.size());

  std::vector<std::vector<float>> result(a.size(), std::vector<float>(b[0].size(), 0.0f));
  for (size_t row = 0; row < a.size(); ++row)
    for (size_t inner = 0; inner < b.size(); ++inner)
      for (size_t col = 0; col < b[0].size(); ++col)
        result[row][col] += a[row][inner] * b[inner][col];
  return flatten_row_major(result);
}

} // namespace

void test_encode_continuous_and_switch()
{
  const std::vector<nam::ParamSpec> specs{
    make_continuous_spec("drive", -2.0f, 2.0f, 0.0f),
    make_switch_spec("mode", {"clean", "edge", "mean"}, 1),
  };

  nam::HypernetSpec hypernet_spec;
  hypernet_spec.hidden_sizes = {};
  hypernet_spec.activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  hypernet_spec.low_rank_mode = false;
  hypernet_spec.rank = 0;
  hypernet_spec.targets = {make_full_target("encoded", 4, 0)};

  const std::vector<float> weights{
    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };

  const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  assert(hypernetwork.EncodedDim() == 4); // 1 continuous + 3-way switch one-hot

  std::vector<float> out;

  hypernetwork.ValidateParams(std::vector<float>{-2.0f, 2.0f}); // well-formed: must not throw
  hypernetwork.ApplyConditioning(std::vector<float>(4, 0.0f), std::vector<float>{-2.0f, 2.0f}, out);
  assert_vector_near(out, {-1.0f, 0.0f, 0.0f, 1.0f});

  hypernetwork.ApplyConditioning(std::vector<float>(4, 0.0f), std::vector<float>{0.0f, 1.0f}, out);
  assert_vector_near(out, {0.0f, 0.0f, 1.0f, 0.0f});

  hypernetwork.ApplyConditioning(std::vector<float>(4, 0.0f), std::vector<float>{2.0f, 0.0f}, out);
  assert_vector_near(out, {1.0f, 1.0f, 0.0f, 0.0f});
}

void test_switch_validation()
{
  const std::vector<nam::ParamSpec> specs{
    make_switch_spec("mode", {"clean", "edge", "mean"}, 0),
  };

  nam::HypernetSpec hypernet_spec;
  hypernet_spec.hidden_sizes = {};
  hypernet_spec.activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  hypernet_spec.low_rank_mode = false;
  hypernet_spec.rank = 0;
  hypernet_spec.targets = {make_full_target("encoded", 3, 0)};

  const std::vector<float> weights{
    1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };

  const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  std::vector<float> out;

  // Validation is a control-thread concern: non-integer and out-of-range switch
  // indices must be rejected here, never on the audio path.
  assert_throws_invalid_argument([&]() { hypernetwork.ValidateParams(std::vector<float>{1.5f}); });
  assert_throws_invalid_argument([&]() { hypernetwork.ValidateParams(std::vector<float>{3.0f}); });
  assert_throws_invalid_argument([&]() { hypernetwork.ValidateParams(std::vector<float>{0.0f, 0.0f}); });

  // The audio path never throws; a value that slips past validation is clamped into
  // range instead of writing out of bounds. 3.0 clamps to the last index (2).
  hypernetwork.ApplyConditioning(std::vector<float>(3, 0.0f), std::vector<float>{3.0f}, out);
  assert_vector_near(out, {0.0f, 0.0f, 1.0f});
}

void test_full_mode_decode()
{
  const std::vector<nam::ParamSpec> specs{
    make_continuous_spec("tone", 0.0f, 1.0f, 0.5f),
  };

  nam::HypernetSpec hypernet_spec;
  hypernet_spec.hidden_sizes = {2};
  hypernet_spec.activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  hypernet_spec.low_rank_mode = false;
  hypernet_spec.rank = 0;
  hypernet_spec.targets = {make_full_target("target", 3, 0)};

  const std::vector<float> weights{
    1.0f, -2.0f, 0.5f, -0.25f, 2.0f, 10.0f, -1.0f, 1.0f, 0.5f, -0.5f, 0.1f, -0.2f, 0.3f, 0.0f, 0.0f, 0.0f,
  };

  const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  std::vector<float> out;
  hypernetwork.ApplyConditioning(std::vector<float>(3, 0.0f), std::vector<float>{0.75f}, out);

  assert_vector_near(out, {2.1f, -1.2f, 0.8f});
}

void test_low_rank_decode()
{
  const std::vector<nam::ParamSpec> specs{
    make_continuous_spec("gain", 0.0f, 1.0f, 0.5f),
  };

  nam::HypernetSpec hypernet_spec;
  hypernet_spec.hidden_sizes = {};
  hypernet_spec.activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  hypernet_spec.low_rank_mode = true;
  hypernet_spec.rank = 2;
  hypernet_spec.targets = {make_low_rank_target("kernel", 0, 2, 2, 3)};

  const std::vector<std::vector<float>> chunk_u{{1.0f, 2.0f}, {3.0f, 4.0f}};
  const std::vector<std::vector<float>> chunk_v{{5.0f, 6.0f, 7.0f}, {8.0f, 9.0f, 10.0f}};
  const std::vector<std::vector<float>> anchor_u{{0.5f, -0.5f}, {1.0f, 0.0f}};
  const std::vector<std::vector<float>> anchor_v{{-1.0f, 0.0f, 1.0f}, {2.0f, -2.0f, 0.5f}};

  auto final_bias = flatten_row_major(chunk_u);
  const auto chunk_v_flat = flatten_row_major(chunk_v);
  final_bias.insert(final_bias.end(), chunk_v_flat.begin(), chunk_v_flat.end());

  auto anchor = flatten_row_major(anchor_u);
  const auto anchor_v_flat = flatten_row_major(anchor_v);
  anchor.insert(anchor.end(), anchor_v_flat.begin(), anchor_v_flat.end());

  std::vector<float> weights(10, 0.0f);
  weights.insert(weights.end(), final_bias.begin(), final_bias.end());
  weights.insert(weights.end(), anchor.begin(), anchor.end());

  const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  std::vector<float> out;
  hypernetwork.ApplyConditioning(std::vector<float>(6, 0.0f), std::vector<float>{0.5f}, out);

  std::vector<std::vector<float>> total_u = chunk_u;
  std::vector<std::vector<float>> total_v = chunk_v;
  for (size_t row = 0; row < total_u.size(); ++row)
    for (size_t col = 0; col < total_u[row].size(); ++col)
      total_u[row][col] += anchor_u[row][col];
  for (size_t row = 0; row < total_v.size(); ++row)
    for (size_t col = 0; col < total_v[row].size(); ++col)
      total_v[row][col] += anchor_v[row][col];

  assert_vector_near(out, multiply_row_major(total_u, total_v));
}

void test_apply_conditioning_offsets()
{
  const std::vector<nam::ParamSpec> specs{
    make_continuous_spec("presence", 0.0f, 1.0f, 0.5f),
  };

  nam::HypernetSpec hypernet_spec;
  hypernet_spec.hidden_sizes = {};
  hypernet_spec.activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  hypernet_spec.low_rank_mode = false;
  hypernet_spec.rank = 0;
  hypernet_spec.targets = {
    make_full_target("first", 2, 1),
    make_full_target("second", 3, 5),
  };

  const std::vector<float> weights{
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, -1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };

  const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  const std::vector<float> base{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
  std::vector<float> out;
  hypernetwork.ApplyConditioning(base, std::vector<float>{0.5f}, out);

  assert_vector_near(out, {10.0f, 20.5f, 29.0f, 40.0f, 50.0f, 62.0f, 73.0f, 84.0f});
}

void test_constructor_validation()
{
  nam::HypernetSpec hypernet_spec;
  hypernet_spec.hidden_sizes = {};
  hypernet_spec.activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  hypernet_spec.low_rank_mode = false;
  hypernet_spec.rank = 0;
  hypernet_spec.targets = {make_full_target("encoded", 1, 0)};

  assert_throws_invalid_argument([&]() {
    const std::vector<nam::ParamSpec> specs{
      nam::ParamSpec{"", 0.0f, 1.0f, 0.5f, "continuous", {}},
    };
    const std::vector<float> weights{0.0f, 0.0f, 0.0f};
    const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  });

  assert_throws_invalid_argument([&]() {
    const std::vector<nam::ParamSpec> specs{
      nam::ParamSpec{"bad_type", 0.0f, 1.0f, 0.5f, "mystery", {}},
    };
    const std::vector<float> weights{0.0f, 0.0f, 0.0f};
    const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  });

  assert_throws_invalid_argument([&]() {
    const std::vector<nam::ParamSpec> specs{
      nam::ParamSpec{"bad_range", 1.0f, 1.0f, 1.0f, "continuous", {}},
    };
    const std::vector<float> weights{0.0f, 0.0f, 0.0f};
    const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  });

  assert_throws_invalid_argument([&]() {
    const std::vector<nam::ParamSpec> specs{
      nam::ParamSpec{"mode", 0.0f, 0.0f, 0.0f, "switch", {"clean"}},
    };
    const std::vector<float> weights{0.0f, 0.0f, 0.0f};
    const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  });

  assert_throws_invalid_argument([&]() {
    const std::vector<nam::ParamSpec> specs{
      make_continuous_spec("presence", 0.0f, 1.0f, 0.5f),
    };
    auto overlapping_spec = hypernet_spec;
    overlapping_spec.targets = {
      make_full_target("first", 3, 0),
      make_full_target("second", 2, 2),
    };
    const std::vector<float> weights(10, 0.0f);
    const nam::Hypernetwork hypernetwork(overlapping_spec, specs, weights);
  });
}

void test_mixed_targets_and_repeated_calls()
{
  const std::vector<nam::ParamSpec> specs{
    make_continuous_spec("drive", 0.0f, 1.0f, 0.5f),
    make_switch_spec("mode", {"clean", "edge"}, 0),
  };

  nam::HypernetSpec hypernet_spec;
  hypernet_spec.hidden_sizes = {};
  hypernet_spec.activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  hypernet_spec.low_rank_mode = true;
  hypernet_spec.rank = 1;
  hypernet_spec.targets = {
    make_full_target("bias", 3, 0),
    make_low_rank_target("kernel", 3, 1, 2, 2),
  };

  const std::vector<float> weights{
    // final weight: 7 outputs x 3 encoded inputs
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.5f,
    0.5f,
    0.0f,
    // final bias
    0.0f,
    0.0f,
    0.0f,
    0.25f,
    0.5f,
    -0.25f,
    1.0f,
    // anchor
    0.0f,
    0.0f,
    0.0f,
    0.5f,
    0.0f,
    0.0f,
    0.5f,
  };

  const nam::Hypernetwork hypernetwork(hypernet_spec, specs, weights);
  const std::vector<float> base{10.0f, 20.0f, 30.0f, 100.0f, 200.0f, 300.0f, 400.0f};
  std::vector<float> out{999.0f, 999.0f};

  hypernetwork.ApplyConditioning(base, std::vector<float>{0.25f, 1.0f}, out);
  assert_vector_near(out, {9.5f, 20.0f, 31.0f, 100.1875f, 200.3125f, 300.375f, 400.625f});

  hypernetwork.ApplyConditioning(base, std::vector<float>{1.0f, 0.0f}, out);
  assert_vector_near(out, {11.0f, 21.0f, 30.0f, 99.5625f, 204.375f, 299.625f, 403.75f});
}

} // namespace test_hypernet
