// End-to-end parity + allocation tests for HyperWaveNet against the training repo.
//
// Fixtures live in tools/test/fixtures/ and are produced by gen_hyperwavenet_fixtures.py,
// which drives the source-of-truth training repo (neural-amp-modeler-parametric,
// feature/parametric-hypernetwork). See that script for the contract. This test:
//   1. loads hyperwavenet.nam, applies each setting, and asserts the C++ streaming output
//      matches the Python-rendered golden (model(x, params, pad_start=True));
//   2. cross-checks each setting against the independently baked stock WaveNet `.nam`,
//      validating the whole encode -> generate -> decode -> apply pipeline against the
//      already-supported runtime path;
//   3. exercises a switch parameter across its full index range and asserts distinct
//      settings produce distinct output;
//   4. asserts a SetParams + process cycle on the real model performs no heap allocation.
//
// run_tests runs from the repo root, so fixtures are referenced by relative path.

#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/parametric_control.h"
#include "allocation_tracking.h"

namespace test_hyperwavenet_parity
{
namespace
{

constexpr auto kFixtureDir = "tools/test/fixtures/";
// Python renders in float32 and the C++ runtime accumulates in float32, but the two use
// different reduction orders, so allow a small absolute tolerance. Measured error is ~1e-7;
// this leaves a wide margin while mirroring other model tests' 1e-5 parity bar.
constexpr auto kGoldenTolerance = 1.0e-5f;
// HyperWaveNet at setting S vs the stock WaveNet baked at S: same arithmetic backend, but
// distinct weight blobs (conditioned-in-C++ vs baked-in-Python), so keep a modest tolerance.
constexpr auto kBakedTolerance = 1.0e-5f;

nlohmann::json load_json(const std::string& path)
{
  std::ifstream stream(path);
  assert(stream.is_open());
  nlohmann::json json;
  stream >> json;
  return json;
}

std::vector<NAM_SAMPLE> to_samples(const nlohmann::json& array)
{
  std::vector<NAM_SAMPLE> samples;
  samples.reserve(array.size());
  for (const auto& value : array)
    samples.push_back(static_cast<NAM_SAMPLE>(value.get<double>()));
  return samples;
}

// Prewarm-then-process so the streaming output is aligned with the Python forward's
// pad_start=True convention (receptive_field-1 zeros of history before the first sample).
std::vector<NAM_SAMPLE> render(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, const double sample_rate)
{
  const auto num_frames = static_cast<int>(input.size());
  std::vector<NAM_SAMPLE> input_copy(input);
  std::vector<NAM_SAMPLE> output(input.size(), 0.0);
  std::array<NAM_SAMPLE*, 1> input_ptrs{input_copy.data()};
  std::array<NAM_SAMPLE*, 1> output_ptrs{output.data()};
  dsp.Reset(sample_rate, num_frames); // Reset prewarms by default.
  dsp.process(input_ptrs.data(), output_ptrs.data(), num_frames);
  return output;
}

float max_abs_diff(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b)
{
  assert(a.size() == b.size());
  auto worst = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
  {
    const auto sample = static_cast<float>(a[i]);
    assert(std::isfinite(sample));
    worst = std::max(worst, std::abs(sample - static_cast<float>(b[i])));
  }
  return worst;
}

} // namespace

void test_matches_python_golden_and_baked_wavenet()
{
  const auto golden = load_json(std::string(kFixtureDir) + "hyperwavenet_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);

  auto dsp = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + "hyperwavenet.nam"));
  assert(dsp != nullptr);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  assert(control->ParamDim() == 2);

  std::vector<std::vector<NAM_SAMPLE>> outputs_by_setting;
  for (const auto& setting : golden["settings"])
  {
    std::vector<float> params;
    for (const auto& value : setting["params"])
      params.push_back(value.get<float>());

    // Apply the setting BEFORE Reset so the prewarm conditions on the chosen weights.
    control->SetParams(params);
    const auto output = render(*dsp, input, sample_rate);

    const auto expected = to_samples(setting["output"]);
    const auto golden_err = max_abs_diff(output, expected);
    assert(golden_err < kGoldenTolerance);

    // Cross-check against the stock WaveNet baked at this same setting.
    auto baked = nam::get_dsp(
      std::filesystem::path(std::string(kFixtureDir) + "baked_" + setting["name"].get<std::string>() + ".nam"));
    assert(baked != nullptr);
    const auto baked_output = render(*baked, input, sample_rate);
    const auto baked_err = max_abs_diff(output, baked_output);
    assert(baked_err < kBakedTolerance);

    outputs_by_setting.push_back(output);
  }

  // The fixture sweeps the switch param across all three indices (clean/crunch/lead) plus a
  // continuous gain change, so every setting must render distinctly.
  assert(outputs_by_setting.size() == 3);
  for (size_t i = 0; i < outputs_by_setting.size(); ++i)
    for (size_t j = i + 1; j < outputs_by_setting.size(); ++j)
      assert(max_abs_diff(outputs_by_setting[i], outputs_by_setting[j]) > kBakedTolerance);
}

void test_setparams_process_no_allocation_real_model()
{
  using namespace allocation_tracking;

  const auto golden = load_json(std::string(kFixtureDir) + "hyperwavenet_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);

  auto dsp = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + "hyperwavenet.nam"));
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);

  const auto num_frames = static_cast<int>(input.size());
  std::vector<NAM_SAMPLE> input_copy(input);
  std::vector<NAM_SAMPLE> output(input.size(), 0.0);
  std::array<NAM_SAMPLE*, 1> input_ptrs{input_copy.data()};
  std::array<NAM_SAMPLE*, 1> output_ptrs{output.data()};

  // Warm steady-state buffers (conditioning + set_weights_) before tracking.
  dsp->Reset(sample_rate, num_frames);
  const std::array<float, 2> first{9.0f, 2.0f};
  control->SetParams(first);
  dsp->process(input_ptrs.data(), output_ptrs.data(), num_frames);

  // A control-thread SetParams followed by an audio-thread process must not touch the heap.
  // This holds whether conditioning runs lazily inside process() (current synchronous model)
  // or eagerly inside SetParams() (Task 6's off-audio-thread handoff): the whole cycle is
  // covered, so the conditioning compute is allocation-free wherever it lands.
  const std::array<float, 2> second{1.0f, 0.0f};
  run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      control->SetParams(second);
      dsp->process(input_ptrs.data(), output_ptrs.data(), num_frames);
    },
    nullptr, "test_setparams_process_no_allocation_real_model");

  for (const auto sample : output)
    assert(std::isfinite(sample));
}

} // namespace test_hyperwavenet_parity
