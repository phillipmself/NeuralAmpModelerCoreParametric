// Tests that moving a runtime control does not step the conditioning at block boundaries.
//
// The fixtures here are deliberately degenerate: each one is wired so the model's output
// *is* the quantity under test (a conditioning channel for ConcatWaveNet, the hypernet's
// conditioned head_scale for HyperWaveNet), so the assertions measure the control
// trajectory directly rather than inferring it from an opaque nonlinearity.
//
// Only ConcatWaveNet is smoothed, and that asymmetry is deliberate -- see
// test_hyperwavenet_applies_immediately() for the measurements behind it.

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/param_ramp.h"
#include "NAM/parametric_control.h"
#include "allocation_tracking.h"

namespace test_param_smoothing
{
namespace
{

constexpr auto kSampleRate = 48000.0;
constexpr auto kBlockSize = 8;
constexpr auto kMaxBufferSize = 16;
// The default 200 ms ramp is 9600 samples at 48 kHz.
const auto kRampSamples = static_cast<int>(std::lround(kSampleRate * nam::kDefaultParamRampSeconds));

// -- ConcatWaveNet fixture: output is ReLU of one concatenated input channel. ------------

nlohmann::json concat_params()
{
  return nlohmann::json::array(
    {{{"name", "gain"}, {"min", 0.0}, {"max", 10.0}, {"default", 5.0}, {"type", "continuous"}},
     {{"name", "mode"},
      {"min", 0},
      {"max", 2},
      {"default", 1},
      {"type", "switch"},
      {"enum_names", {"clean", "crunch", "lead"}}}});
}

// selected_input indexes [audio, continuous, switch-0, switch-1, switch-2].
nlohmann::json concat_config(const int selected_input)
{
  std::vector<float> weights{
    0, 0, 0, 0, 0, // rechannel 5 -> 1
    0, 0, // layer conv weight, bias
    0, 0, 0, 0, 0, // input mixin 5 -> 1
    0, 0, // layer 1x1 weight, bias
    1, // head rechannel
    1, // head scale
  };
  weights[static_cast<size_t>(7 + selected_input)] = 1.0f;

  return nlohmann::json{
    {"version", "1.0.0"},
    {"metadata", nlohmann::json::object()},
    {"architecture", "ConcatWaveNet"},
    {"config",
     {{"head_scale", 1.0},
      {"layers",
       {{
         {"channels", 1},
         {"head", {{"out_channels", 1}, {"kernel_size", 1}, {"bias", false}}},
         {"kernel_size", 1},
         {"dilations", {1}},
         {"activation", "ReLU"},
         {"layer1x1", {{"active", true}, {"groups", 1}}},
       }}},
      {"params", concat_params()}}},
    {"weights", weights},
    {"sample_rate", 48000},
  };
}

// -- HyperWaveNet fixture: the hypernet conditions head_scale, so the output scales with it.

nlohmann::json hyper_config()
{
  return nlohmann::json{
    {"version", "1.0.0"},
    {"metadata", nlohmann::json::object()},
    {"architecture", "HyperWaveNet"},
    {"config",
     {
       {"head_scale", 1.0},
       {"layers",
        {{
          {"input_size", 1},
          {"condition_size", 1},
          {"channels", 1},
          {"head", {{"out_channels", 1}, {"kernel_size", 1}, {"bias", false}}},
          {"kernel_size", 1},
          {"dilations", {1}},
          {"activation", "ReLU"},
          {"layer1x1", {{"active", true}, {"groups", 1}}},
        }}},
       {"params",
        {{
          {"name", "drive"},
          {"min", 0.0},
          {"max", 1.0},
          {"default", 0.5},
          {"type", "continuous"},
        }}},
       {"hypernet",
        {{"mode", "full"},
         {"hidden_sizes", nlohmann::json::array()},
         {"activation", "ReLU"},
         {"delta_map", {{{"name", "head_scale"}, {"mode", "full"}, {"numel", 1}, {"export_offset", 7}}}}}},
     }},
    {"weights", {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f}},
    {"sample_rate", 48000},
  };
}

// -- Helpers -----------------------------------------------------------------------------

std::unique_ptr<nam::DSP> make_dsp(const nlohmann::json& config)
{
  auto dsp = nam::get_dsp(config);
  assert(dsp != nullptr);
  dsp->SetPrewarmOnReset(false);
  dsp->Reset(kSampleRate, kMaxBufferSize);
  return dsp;
}

nam::IParametricControl& control_of(nam::DSP& dsp)
{
  auto* control = dynamic_cast<nam::IParametricControl*>(&dsp);
  assert(control != nullptr);
  return *control;
}

// Render `num_frames` in kBlockSize blocks, returning every output sample.
std::vector<float> render(nam::DSP& dsp, const int num_frames, const float input_value)
{
  std::vector<float> collected;
  collected.reserve(static_cast<size_t>(num_frames));
  std::vector<NAM_SAMPLE> input(kBlockSize, input_value);
  std::vector<NAM_SAMPLE> output(kBlockSize);
  std::array<NAM_SAMPLE*, 1> inputs{input.data()};
  std::array<NAM_SAMPLE*, 1> outputs{output.data()};
  for (auto rendered = 0; rendered < num_frames; rendered += kBlockSize)
  {
    dsp.process(inputs.data(), outputs.data(), kBlockSize);
    for (auto i = 0; i < kBlockSize; ++i)
      collected.push_back(static_cast<float>(output[i]));
  }
  return collected;
}

float peak_sample_step(const std::vector<float>& samples)
{
  auto peak = 0.0f;
  for (size_t i = 1; i < samples.size(); ++i)
    peak = std::max(peak, std::abs(samples[i] - samples[i - 1]));
  return peak;
}

} // namespace

// The conditioning walks to its new value one sample at a time. The largest single-sample
// jump anywhere in the move is a ramp step, not the whole control change -- which is the
// difference between a click and no click.
void test_concat_wavenet_conditioning_is_continuous()
{
  auto dsp = make_dsp(concat_config(/*continuous channel=*/1));
  auto& control = control_of(*dsp);

  // gain 5 -> encoded 0. Settled, so the block is flat.
  const auto before = render(*dsp, kBlockSize, 0.0f);
  for (const auto sample : before)
    assert(std::abs(sample) < 1.0e-7f);

  // gain 10 -> encoded +1: the full positive half of the control range.
  control.SetParams(std::array<float, 2>{10.0f, 1.0f});
  auto moving = render(*dsp, kRampSamples + 4 * kBlockSize, 0.0f);
  // Carry the last pre-change sample in, so the very first transition is measured too --
  // that boundary is exactly where an unsmoothed model puts its whole step.
  moving.insert(moving.begin(), before.back());

  constexpr auto total_change = 1.0f;
  const auto expected_step = total_change / static_cast<float>(kRampSamples);
  const auto peak = peak_sample_step(moving);
  assert(peak < 2.0f * expected_step);
  // Guard against the assertion passing because nothing moved at all.
  assert(peak > 0.5f * expected_step);

  // Monotonic: the ramp approaches from one side and never overshoots.
  for (size_t i = 1; i < moving.size(); ++i)
    assert(moving[i] >= moving[i - 1] - 1.0e-7f);

  // And it arrives exactly, so the settled state is what an unsmoothed model would give.
  assert(std::abs(moving.back() - 1.0f) < 1.0e-7f);
}

// Switch controls are one-hot encoded; there is no meaningful value between two indices, so
// they step on commit rather than crossfading through conditioning vectors no model was
// trained on.
void test_concat_wavenet_switch_steps()
{
  // Output is the "lead" one-hot channel, which is 0 at the default index (crunch).
  auto dsp = make_dsp(concat_config(/*switch-2 channel=*/4));
  auto& control = control_of(*dsp);

  const auto before = render(*dsp, kBlockSize, 0.0f);
  for (const auto sample : before)
    assert(sample == 0.0f);

  control.SetParams(std::array<float, 2>{5.0f, 2.0f});
  const auto after = render(*dsp, kBlockSize, 0.0f);
  // Fully applied on the very next block, with no ramp in between.
  for (const auto sample : after)
    assert(sample == 1.0f);
}

// HyperWaveNet is deliberately NOT smoothed. Its controls reach the audio only by
// regenerating weights, and weights can only change between process() calls, so the
// transfer function is quantised to the block grid by construction. Ramping it therefore
// does not remove the discontinuity, it only relocates it: measured on a 37,892-weight
// model at a 32-sample buffer, smoothing moved the artifact from a maskable ~100 Hz tone at
// the host's commit rate to a 1378 Hz tone at the block rate -- lower in total energy but
// far more audible, sitting in the ear's sensitive band -- while costing 12x more CPU
// because every update regenerates the whole weight set.
//
// Ways out that were measured and rejected: sub-block updates cost 1.4x/2.1x/3.3x the
// model's own runtime at 16/8/4-sample chunks; jittering the update grid changed the tone's
// magnitude by 0.07%; crossfading needs two full model instances because every Conv1D
// carries a ring buffer over its receptive field.
//
// So this test pins the *absence* of smoothing: someone adding it back should have to
// delete this test, and re-measure before they do.
void test_hyperwavenet_applies_immediately()
{
  auto dsp = make_dsp(hyper_config());
  auto& control = control_of(*dsp);

  const auto before = render(*dsp, kBlockSize, 1.0f);
  control.SetParams(std::vector<float>{1.0f});
  const auto after = render(*dsp, kBlockSize, 1.0f);

  // Fully in effect on the very next block...
  assert(std::abs(after.back() - before.back()) > 1.0e-6f);
  const auto settled = render(*dsp, 4 * kBlockSize, 1.0f);
  assert(std::abs(settled.back() - after.back()) < 1.0e-6f);

  // ...and constant within a block, since the weights cannot change mid-call.
  for (auto i = 1; i < kBlockSize; ++i)
    assert(after[static_cast<size_t>(i)] == after[0]);
}

// Ramping must converge on exactly the state an unsmoothed model reaches, or the settled
// output would depend on how the control got there.
void test_settles_on_the_committed_value()
{
  auto ramped = make_dsp(concat_config(1));
  control_of(*ramped).SetParams(std::array<float, 2>{8.0f, 1.0f});
  const auto moved = render(*ramped, 2 * kRampSamples, 0.0f);

  auto direct = make_dsp(concat_config(1));
  control_of(*direct).SetParams(std::array<float, 2>{8.0f, 1.0f});
  direct->Reset(kSampleRate, kMaxBufferSize);
  const auto settled = render(*direct, kBlockSize, 0.0f);

  assert(std::abs(moved.back() - settled.back()) < 1.0e-7f);
}

// A reset is a stream restart, so controls committed before it apply immediately. This is
// what stops a freshly loaded model gliding from its defaults to the host's restored values
// -- two models need not even share a parameter set.
void test_reset_settles_rather_than_ramping()
{
  auto dsp = make_dsp(concat_config(1));
  auto& control = control_of(*dsp);

  control.SetParams(std::array<float, 2>{10.0f, 1.0f});
  dsp->Reset(kSampleRate, kMaxBufferSize);

  const auto after_reset = render(*dsp, kBlockSize, 0.0f);
  for (const auto sample : after_reset)
    assert(std::abs(sample - 1.0f) < 1.0e-7f);
}

// The ramp is pre-sized at Reset(), so running one costs no allocation on the audio thread.
void test_ramping_is_realtime_safe()
{
  auto concat = make_dsp(concat_config(1));
  auto& concat_control = control_of(*concat);

  std::array<NAM_SAMPLE, kBlockSize> input{};
  std::array<NAM_SAMPLE, kBlockSize> output{};
  std::array<NAM_SAMPLE*, 1> inputs{input.data()};
  std::array<NAM_SAMPLE*, 1> outputs{output.data()};

  // Prime it so the first tracked block is already mid-ramp.
  concat_control.SetParams(std::array<float, 2>{9.0f, 2.0f});
  concat->process(inputs.data(), outputs.data(), kBlockSize);

  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      // Run well past the end of the ramp so the landing and the settled fast path are
      // both inside the tracked region.
      for (auto rendered = 0; rendered < 2 * kRampSamples; rendered += kBlockSize)
        concat->process(inputs.data(), outputs.data(), kBlockSize);
    },
    nullptr, "Parametric control ramping realtime safety");
}

} // namespace test_param_smoothing
