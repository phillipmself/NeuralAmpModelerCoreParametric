// End-to-end parity + allocation tests for ConcatWaveNet against the training repo.
//
// Fixtures live in tools/test/fixtures/ and are produced by gen_concat_wavenet_fixtures.py,
// which drives the source-of-truth training repo (neural-amp-modeler-parametric,
// feature/parametric-main). See that script for the contract; unlike the HyperWaveNet
// generator it pins the torch seed, so regenerating reproduces the committed fixture
// byte-for-byte. This test:
//   1. loads concat_wavenet.nam, applies each setting, and asserts the C++ streaming output
//      matches the Python-rendered golden (model(x, params, pad_start=True));
//   2. cross-checks the same weights driven as a plain stock WaveNet with the concatenated
//      channels built by hand, which validates the encode-and-concatenate step against an
//      oracle that does not depend on Python;
//   3. asserts distinct settings render distinctly, so a runtime that ignored the controls
//      could not pass;
//   4. asserts a SetParams + process cycle on the real model performs no heap allocation.
//
// tools/test/test_concat_wavenet.cpp covers the same runtime with hand-written synthetic
// weights; this file is the real-export counterpart.
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

namespace test_concat_wavenet_parity
{
namespace
{

constexpr auto kFixtureDir = "tools/test/fixtures/";
// Python renders in float32 and the C++ runtime accumulates in float32, but the two use
// different reduction orders, so allow a small absolute tolerance. Measured error is ~1e-9;
// this mirrors the other model tests' 1e-5 parity bar.
constexpr auto kGoldenTolerance = 1.0e-5f;
// Same weights, same arithmetic backend, only the channel plumbing differs, so the manual
// concatenation oracle should agree far more tightly than the Python comparison.
constexpr auto kManualTolerance = 1.0e-6f;

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

std::vector<float> to_params(const nlohmann::json& array)
{
  std::vector<float> params;
  params.reserve(array.size());
  for (const auto& value : array)
    params.push_back(value.get<float>());
  return params;
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

// The control encoding documented on ConcatWaveNet: continuous params are signed min-max
// scaled to [-1, 1], switches become a one-hot block. Recomputed here from the raw params
// and the model's own specs so the oracle below does not borrow the runtime's encoder.
std::vector<float> encode_params(const std::vector<nam::ParamSpec>& specs, const std::vector<float>& raw)
{
  assert(specs.size() == raw.size());
  std::vector<float> encoded;
  for (size_t i = 0; i < specs.size(); ++i)
  {
    const auto& spec = specs[i];
    if (spec.type == "switch")
    {
      const auto num_inputs = spec.num_inputs();
      const auto index = static_cast<int>(std::lround(raw[i]));
      assert(index >= 0 && index < num_inputs);
      for (auto k = 0; k < num_inputs; ++k)
        encoded.push_back(k == index ? 1.0f : 0.0f);
      continue;
    }
    const auto fraction = (raw[i] - spec.min) / (spec.max - spec.min);
    encoded.push_back(-1.0f + 2.0f * fraction);
  }
  return encoded;
}

} // namespace

void test_matches_python_golden()
{
  const auto golden = load_json(std::string(kFixtureDir) + "concat_wavenet_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);

  auto dsp = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + "concat_wavenet.nam"));
  assert(dsp != nullptr);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  assert(control->ParamDim() == 2);

  std::vector<std::vector<NAM_SAMPLE>> outputs_by_setting;
  for (const auto& setting : golden["settings"])
  {
    // Apply the setting BEFORE Reset so the prewarm runs on the chosen control values.
    control->SetParams(to_params(setting["params"]));
    const auto output = render(*dsp, input, sample_rate);

    const auto expected = to_samples(setting["output"]);
    assert(max_abs_diff(output, expected) < kGoldenTolerance);

    outputs_by_setting.push_back(output);
  }

  // The fixture sweeps the switch param across all three indices (clean/crunch/lead) plus a
  // continuous gain change, so every setting must render distinctly. Without this a runtime
  // that dropped the conditioning entirely would still satisfy the comparisons above only if
  // Python dropped it too -- this pins that the controls actually reach the audio.
  assert(outputs_by_setting.size() == 3);
  for (size_t i = 0; i < outputs_by_setting.size(); ++i)
    for (size_t j = i + 1; j < outputs_by_setting.size(); ++j)
      assert(max_abs_diff(outputs_by_setting[i], outputs_by_setting[j]) > kGoldenTolerance);
}

void test_matches_manual_concat_wavenet_real_weights()
{
  // Second oracle, independent of Python: drive the SAME weights as a plain stock WaveNet
  // whose input channels are the concatenated [audio, encoded params] tensor, and require
  // it to agree with what ConcatWaveNet produces from the audio channel alone.
  // test_concat_wavenet::test_matches_manual_concat_wavenet does this with synthetic
  // weights on a one-layer model; this is the real-export, two-layer-array version.
  const auto golden = load_json(std::string(kFixtureDir) + "concat_wavenet_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);
  const auto model_json = load_json(std::string(kFixtureDir) + "concat_wavenet.nam");

  auto concat = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + "concat_wavenet.nam"));
  auto* control = dynamic_cast<nam::IParametricControl*>(concat.get());
  assert(control != nullptr);
  const auto& specs = control->GetParamSpecs();

  auto encoded_dim = 0;
  for (const auto& spec : specs)
    encoded_dim += spec.num_inputs();
  const auto in_channels = 1 + encoded_dim;

  // Same config and same weights, re-badged as the stock architecture. ConcatWaveNet derives
  // in_channels from the specs, so a plain WaveNet has to be told it explicitly.
  auto plain_json = model_json;
  plain_json["architecture"] = "WaveNet";
  plain_json["config"].erase("params");
  plain_json["config"]["in_channels"] = in_channels;
  auto plain = nam::get_dsp(plain_json);
  assert(plain->NumInputChannels() == in_channels);

  const auto num_frames = static_cast<int>(input.size());
  for (const auto& setting : golden["settings"])
  {
    const auto raw = to_params(setting["params"]);
    control->SetParams(raw);
    const auto expected = render(*concat, input, sample_rate);

    // Build the concatenated input the wrapper is supposed to be assembling internally.
    const auto encoded = encode_params(specs, raw);
    assert(static_cast<int>(encoded.size()) + 1 == in_channels);
    std::vector<std::vector<NAM_SAMPLE>> channels;
    channels.push_back(input);
    for (const auto value : encoded)
      channels.emplace_back(input.size(), static_cast<NAM_SAMPLE>(value));

    std::vector<NAM_SAMPLE*> channel_ptrs;
    for (auto& channel : channels)
      channel_ptrs.push_back(channel.data());
    std::vector<NAM_SAMPLE> actual(input.size(), 0.0);
    std::array<NAM_SAMPLE*, 1> output_ptrs{actual.data()};

    // Prewarm by hand rather than letting Reset do it. DSP::prewarm() zeros EVERY input
    // channel, so on this plain 5-channel model it would warm up with the control channels
    // silent -- whereas ConcatWaveNet, which the host sees as 1-channel, prewarms through
    // its own process() and therefore carries the encoded controls throughout. Replicate
    // the latter: zero audio, controls held at their encoded values, using the same
    // block-at-a-time loop DSP::prewarm() runs.
    plain->SetPrewarmOnReset(false);
    plain->Reset(sample_rate, num_frames);

    std::vector<std::vector<NAM_SAMPLE>> warmup_channels;
    warmup_channels.emplace_back(input.size(), 0.0); // silent audio
    for (const auto value : encoded)
      warmup_channels.emplace_back(input.size(), static_cast<NAM_SAMPLE>(value));
    std::vector<NAM_SAMPLE*> warmup_ptrs;
    for (auto& channel : warmup_channels)
      warmup_ptrs.push_back(channel.data());
    std::vector<NAM_SAMPLE> warmup_output(input.size(), 0.0);
    std::array<NAM_SAMPLE*, 1> warmup_output_ptrs{warmup_output.data()};
    for (auto warmed = 0; warmed < plain->GetPrewarmSamples(); warmed += num_frames)
      plain->process(warmup_ptrs.data(), warmup_output_ptrs.data(), num_frames);

    plain->process(channel_ptrs.data(), output_ptrs.data(), num_frames);

    assert(max_abs_diff(actual, expected) < kManualTolerance);
  }
}

void test_setparams_process_no_allocation_real_model()
{
  using namespace allocation_tracking;

  const auto golden = load_json(std::string(kFixtureDir) + "concat_wavenet_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);

  auto dsp = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + "concat_wavenet.nam"));
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);

  const auto num_frames = static_cast<int>(input.size());
  std::vector<NAM_SAMPLE> input_copy(input);
  std::vector<NAM_SAMPLE> output(input.size(), 0.0);
  std::array<NAM_SAMPLE*, 1> input_ptrs{input_copy.data()};
  std::array<NAM_SAMPLE*, 1> output_ptrs{output.data()};

  // Warm the steady-state buffers (encode + channel fan-out) before tracking.
  dsp->Reset(sample_rate, num_frames);
  const std::array<float, 2> first{9.0f, 2.0f};
  control->SetParams(first);
  dsp->process(input_ptrs.data(), output_ptrs.data(), num_frames);

  // A control-thread SetParams followed by an audio-thread process must not touch the heap.
  const std::array<float, 2> second{1.0f, 0.0f};
  run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      control->SetParams(second);
      dsp->process(input_ptrs.data(), output_ptrs.data(), num_frames);
    },
    nullptr, "test_setparams_process_no_allocation_real_model (ConcatWaveNet)");

  for (const auto sample : output)
    assert(std::isfinite(sample));
}

} // namespace test_concat_wavenet_parity
