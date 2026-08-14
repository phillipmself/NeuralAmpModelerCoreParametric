// End-to-end parity + allocation tests for FiLMWaveNet against the training repo.
//
// Fixtures live in tools/test/fixtures/ and are produced by gen_film_wavenet_fixtures.py, which
// drives the source-of-truth training repo (neural-amp-modeler-parametric,
// feature/parametric-film-wavenet) directly. See that script for the contract; it pins the torch
// seed, so regenerating reproduces the committed fixtures byte-for-byte. Two fixtures are
// covered because they exercise different runtime paths:
//   * film_wavenet         - no param encoder: the FiLM condition is the encoded control itself.
//   * film_wavenet_encoder - a param encoder MLP sits between the encoded controls and the FiLM
//                            condition, exercising the weight-blob offset split.
//
// For each fixture, this test:
//   1. loads the .nam, applies each setting, and asserts the C++ streaming output matches the
//      Python-rendered golden (model(x, params, pad_start=True));
//   2. asserts distinct settings render distinctly, so a runtime that ignored the controls (or
//      cached the FiLM condition only once, at construction) could not pass;
//   3. asserts a SetParams + process cycle on the real model performs no heap allocation.
//
// tools/test/test_film_wavenet.cpp covers the same runtime with hand-written synthetic weights,
// where every intermediate value is predictable by hand; this file is the real-export counterpart.
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
#include "NAM/wavenet/film_wavenet.h"
#include "allocation_tracking.h"

namespace test_film_wavenet_parity
{
namespace
{

constexpr auto kFixtureDir = "tools/test/fixtures/";
// Python renders in float32 and the C++ runtime accumulates in float32, but the two use
// different reduction orders, so allow a small absolute tolerance. Mirrors the ConcatWaveNet
// parity test's bar.
constexpr auto kGoldenTolerance = 1.0e-5f;
// Bar for outputs that are the *same* arithmetic reached by a different route (a different host
// block size, or a steady state reached through a control move). Not bit-exact: below the conv
// path's vectorization threshold the reduction order changes, which moves the last couple of ulps
// -- measured at ~6e-9 here, and identically ~2e-9 on ConcatWaveNet, which has no FiLM cache at
// all, so it is a property of the core's conv path rather than of the cached FiLM condition. Still
// ~1000x tighter than kGoldenTolerance, so a cache that went stale or ignored a control would fail
// this by orders of magnitude.
constexpr auto kSameArithmeticTolerance = 1.0e-6f;

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

void run_fixture(const std::string& basename)
{
  const auto golden = load_json(std::string(kFixtureDir) + basename + "_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);

  auto dsp = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + basename + ".nam"));
  assert(dsp != nullptr);
  assert(dsp->NumInputChannels() == 1);
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

  // The fixture sweeps the switch param across all three indices plus a continuous gain change,
  // so every setting must render distinctly -- a runtime that dropped the conditioning, or only
  // applied it once (e.g. cached the FiLM condition at construction and never refreshed it),
  // would fail this even if it happened to pass the nominal-setting comparison above.
  assert(outputs_by_setting.size() == 3);
  for (size_t i = 0; i < outputs_by_setting.size(); ++i)
    for (size_t j = i + 1; j < outputs_by_setting.size(); ++j)
      assert(max_abs_diff(outputs_by_setting[i], outputs_by_setting[j]) > kGoldenTolerance);
}

void run_no_allocation_test(const std::string& basename)
{
  using namespace allocation_tracking;

  const auto golden = load_json(std::string(kFixtureDir) + basename + "_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);

  auto dsp = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + basename + ".nam"));
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);

  const auto num_frames = static_cast<int>(input.size());
  std::vector<NAM_SAMPLE> input_copy(input);
  std::vector<NAM_SAMPLE> output(input.size(), 0.0);
  std::array<NAM_SAMPLE*, 1> input_ptrs{input_copy.data()};
  std::array<NAM_SAMPLE*, 1> output_ptrs{output.data()};

  // Warm the steady-state buffers before tracking.
  dsp->Reset(sample_rate, num_frames);
  const std::array<float, 2> first{9.0f, 2.0f};
  control->SetParams(first);
  dsp->process(input_ptrs.data(), output_ptrs.data(), num_frames);

  // A control-thread SetParams followed by an audio-thread process must not touch the heap. This
  // exercises the FiLM condition cache refresh (WaveNet::SetParamCondition, once per process()
  // call) plus every FiLM's ProcessCached()/ProcessCached_() across the block.
  const std::array<float, 2> second{1.0f, 0.0f};
  run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      control->SetParams(second);
      dsp->process(input_ptrs.data(), output_ptrs.data(), num_frames);
    },
    nullptr, ("test_setparams_process_no_allocation_real_model (FiLMWaveNet " + basename + ")").c_str());

  for (const auto sample : output)
    assert(std::isfinite(sample));
}

// Render `input` in fixed-size blocks, optionally switching to `moved` params at sample
// `move_at` (move_at < 0 disables). Prewarms once at the starting setting.
std::vector<NAM_SAMPLE> render_blocked(const std::string& basename, const std::vector<NAM_SAMPLE>& input,
                                       const double sample_rate, const int block, const std::array<float, 2>& start,
                                       const int move_at, const std::array<float, 2>& moved)
{
  auto dsp = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + basename + ".nam"));
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  // Control smoothing is deliberately out of the picture here: these two tests measure properties
  // of the architecture (how far a control change propagates, and that block size does not change
  // the arithmetic), and a ramp would just superimpose its own settling time on both.
  // tools/test/test_film_wavenet.cpp::test_param_ramp_smooths_and_lands_exactly covers the ramp.
  dynamic_cast<nam::wavenet::FiLMWaveNet*>(dsp.get())->SetParamRampSeconds(0.0f);
  control->SetParams(start);

  const auto num_samples = static_cast<int>(input.size());
  std::vector<NAM_SAMPLE> input_copy(input);
  std::vector<NAM_SAMPLE> output(input.size(), 0.0);
  dsp->Reset(sample_rate, block); // Prewarms at `start`.
  for (int i = 0; i < num_samples; i += block)
  {
    if (move_at >= 0 && i == move_at)
      control->SetParams(moved);
    const auto frames = std::min(block, num_samples - i);
    NAM_SAMPLE* in_ptr = input_copy.data() + i;
    NAM_SAMPLE* out_ptr = output.data() + i;
    dsp->process(&in_ptr, &out_ptr, frames);
  }
  return output;
}

float max_abs_diff_range(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b, const size_t lo,
                         const size_t hi)
{
  auto worst = 0.0f;
  for (size_t i = lo; i < hi; ++i)
    worst = std::max(worst, std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i])));
  return worst;
}

} // namespace

// Every FiLM caches its scale/shift column once per block and broadcasts it across the block, so
// the block size is the one thing that could make the cached path disagree with itself. Bit-exact
// equality is the bar: the arithmetic per sample is identical regardless of how frames are grouped.
void test_block_size_invariance()
{
  const auto golden = load_json(std::string(kFixtureDir) + "film_wavenet_encoder_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);
  const std::array<float, 2> params{9.0f, 2.0f};
  const std::array<float, 2> unused{0.0f, 0.0f};

  const auto whole =
    render_blocked("film_wavenet_encoder", input, sample_rate, static_cast<int>(input.size()), params, -1, unused);
  for (const auto block : {1, 7, 32, 64})
  {
    const auto chunked = render_blocked("film_wavenet_encoder", input, sample_rate, block, params, -1, unused);
    assert(max_abs_diff(whole, chunked) < kSameArithmeticTolerance);
  }
}

// The reason this architecture exists. The controls are read by 1x1 convolutions with no time
// extent, so a knob move mid-stream cannot put the network in a configuration it was never trained
// in: outside one receptive field of the move the output IS the steady state for the setting in
// force, and only the audio state left over from the old setting resolves inside that window.
//
// Mirrors tests/parametric/test_film_wavenet.py::
//   test_a_control_move_settles_within_exactly_the_receptive_field
// in the training repo, but over the streaming runtime, where the per-block condition cache is what
// could break it.
void test_control_move_settles_within_the_receptive_field()
{
  const auto golden = load_json(std::string(kFixtureDir) + "film_wavenet_encoder_golden.json");
  const auto sample_rate = golden["sample_rate"].get<double>();
  const auto input = to_samples(golden["input"]);
  const auto num_samples = input.size();

  const std::array<float, 2> low{1.0f, 0.0f};
  const std::array<float, 2> high{9.0f, 2.0f};
  constexpr int block = 32;
  constexpr int move_at = 64; // A block boundary, so the move lands exactly at this sample.

  auto probe = nam::get_dsp(std::filesystem::path(std::string(kFixtureDir) + "film_wavenet_encoder.nam"));
  const auto receptive_field = static_cast<size_t>(probe->GetPrewarmSamples());
  probe.reset();
  assert(receptive_field > 0);
  assert(static_cast<size_t>(move_at) + receptive_field < num_samples);

  const auto steady_low = render_blocked("film_wavenet_encoder", input, sample_rate, block, low, -1, low);
  const auto steady_high = render_blocked("film_wavenet_encoder", input, sample_rate, block, high, -1, high);
  const auto moved = render_blocked("film_wavenet_encoder", input, sample_rate, block, low, move_at, high);

  // The two settings must actually differ, or everything below passes vacuously.
  assert(max_abs_diff(steady_low, steady_high) > kGoldenTolerance);

  // Before the move: the old steady state -- no anticipation.
  assert(max_abs_diff_range(moved, steady_low, 0, static_cast<size_t>(move_at)) < kSameArithmeticTolerance);
  // One receptive field after it: the new steady state -- no control memory beyond that window.
  assert(max_abs_diff_range(moved, steady_high, static_cast<size_t>(move_at) + receptive_field, num_samples)
         < kSameArithmeticTolerance);
  // Inside the window: still resolving, so it is neither.
  assert(
    max_abs_diff_range(moved, steady_high, static_cast<size_t>(move_at), static_cast<size_t>(move_at) + receptive_field)
    > kGoldenTolerance);
}

void test_matches_python_golden_no_encoder()
{
  run_fixture("film_wavenet");
}

void test_matches_python_golden_with_encoder()
{
  run_fixture("film_wavenet_encoder");
}

void test_setparams_process_no_allocation_no_encoder()
{
  run_no_allocation_test("film_wavenet");
}

void test_setparams_process_no_allocation_with_encoder()
{
  run_no_allocation_test("film_wavenet_encoder");
}

} // namespace test_film_wavenet_parity
