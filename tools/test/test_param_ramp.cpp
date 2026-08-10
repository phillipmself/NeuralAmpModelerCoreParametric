// Tests for ParamRamp, the linear ramp parametric models travel to committed values on.

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include "NAM/param_ramp.h"

namespace test_param_ramp
{
namespace
{

constexpr auto kSampleRate = 48000.0;
constexpr auto kSeconds = 0.010f; // 480 samples
constexpr auto kLength = 480;

nam::ParamRamp make_ramp(const std::vector<float>& initial, const std::span<const char> smoothed = {})
{
  nam::ParamRamp ramp;
  ramp.Configure(initial.size(), kSampleRate, kSeconds, smoothed);
  ramp.Snap(initial);
  return ramp;
}

} // namespace

// A freshly configured and snapped ramp is settled, and reports the snapped values.
void test_snap_is_immediate()
{
  auto ramp = make_ramp({0.25f, -0.5f});
  assert(!ramp.IsRamping());
  assert(ramp.RemainingSamples() == 0);
  assert(ramp.Current()[0] == 0.25f);
  assert(ramp.Current()[1] == -0.5f);
  assert(ramp.Target()[0] == 0.25f);
  // Advancing a settled ramp is a no-op and reports no change, which is what lets callers
  // skip work when nothing is moving.
  assert(!ramp.Advance(64));
  assert(ramp.Current()[0] == 0.25f);

  // A snap mid-flight cancels the ramp outright.
  const std::array<float, 2> target{1.0f, 1.0f};
  assert(!ramp.SetTarget(target));
  assert(ramp.IsRamping());
  const std::array<float, 2> snapped{-1.0f, 2.0f};
  ramp.Snap(snapped);
  assert(!ramp.IsRamping());
  assert(ramp.Current()[0] == -1.0f);
  assert(ramp.Current()[1] == 2.0f);
}

// The ramp reaches the committed value exactly -- not merely close to it -- so the settled
// state is bit-identical to what an unsmoothed model would have used.
void test_lands_exactly_on_target()
{
  auto ramp = make_ramp({0.0f, 0.0f});
  const std::array<float, 2> target{1.0f, -3.0f};
  assert(!ramp.SetTarget(target)); // took the ramp, so the current values did not jump
  assert(ramp.IsRamping());
  assert(ramp.RemainingSamples() == kLength);

  // Partway through, the values sit between the endpoints and move monotonically.
  auto previous = ramp.Current()[0];
  for (auto consumed = 0; consumed < kLength - 8; consumed += 8)
  {
    assert(ramp.Advance(8));
    assert(ramp.Current()[0] > previous);
    assert(ramp.Current()[0] < 1.0f);
    assert(ramp.Current()[1] > -3.0f);
    previous = ramp.Current()[0];
  }

  assert(ramp.Advance(8));
  assert(!ramp.IsRamping());
  assert(ramp.Current()[0] == 1.0f);
  assert(ramp.Current()[1] == -3.0f);
  // Overshoot is impossible: the landing assigns the target rather than accumulating.
  assert(!ramp.Advance(1024));
  assert(ramp.Current()[0] == 1.0f);
}

// A block longer than the remaining ramp lands it rather than running past the target.
void test_advance_past_end_clamps()
{
  auto ramp = make_ramp({0.0f});
  const std::array<float, 1> target{1.0f};
  ramp.SetTarget(target);
  assert(ramp.Advance(kLength * 4));
  assert(!ramp.IsRamping());
  assert(ramp.Current()[0] == 1.0f);
}

// Re-targeting mid-ramp restarts from wherever the ramp currently sits, so a host
// committing a new value every block still produces a continuous trajectory.
void test_retarget_restarts_from_current()
{
  auto ramp = make_ramp({0.0f});
  const std::array<float, 1> first{1.0f};
  ramp.SetTarget(first);
  ramp.Advance(kLength / 2);
  const auto midpoint = ramp.Current()[0];
  assert(midpoint > 0.0f && midpoint < 1.0f);

  const std::array<float, 1> second{-1.0f};
  assert(!ramp.SetTarget(second));
  // No discontinuity at the hand-off: the new ramp begins at the value already in effect.
  assert(ramp.Current()[0] == midpoint);
  assert(ramp.RemainingSamples() == kLength);
  ramp.Advance(kLength);
  assert(ramp.Current()[0] == -1.0f);
}

// Re-committing the value already targeted must not restart the ramp. A host that pushes
// the same vector every block would otherwise keep a model permanently "moving", which for
// HyperWaveNet means regenerating weights forever.
void test_redundant_target_is_ignored()
{
  auto ramp = make_ramp({0.5f});
  const std::array<float, 1> same{0.5f};
  assert(!ramp.SetTarget(same));
  assert(!ramp.IsRamping());

  const std::array<float, 1> moved{0.75f};
  ramp.SetTarget(moved);
  ramp.Advance(kLength / 4);
  const auto partway = ramp.Current()[0];
  const auto remaining = ramp.RemainingSamples();
  assert(!ramp.SetTarget(moved));
  assert(ramp.Current()[0] == partway);
  assert(ramp.RemainingSamples() == remaining);
}

// Values excluded from smoothing jump on commit. When every changed value is excluded the
// ramp does not engage at all, so a switch-only change costs nothing.
void test_excluded_values_jump()
{
  const std::array<char, 2> smoothed{1, 0};
  auto ramp = make_ramp({0.0f, 0.0f}, smoothed);

  const std::array<float, 2> switch_only{0.0f, 1.0f};
  assert(ramp.SetTarget(switch_only)); // reports the jump
  assert(!ramp.IsRamping());
  assert(ramp.Current()[1] == 1.0f);

  // A mixed commit ramps the smoothed value while the excluded one is already there.
  const std::array<float, 2> mixed{1.0f, 0.0f};
  assert(ramp.SetTarget(mixed));
  assert(ramp.IsRamping());
  assert(ramp.Current()[1] == 0.0f);
  assert(ramp.Current()[0] == 0.0f);
  assert(ramp.Step()[1] == 0.0f);
  ramp.Advance(kLength);
  assert(ramp.Current()[0] == 1.0f);
}

// A zero ramp time (or an unknown sample rate) degenerates to the unsmoothed behaviour,
// which is how a consumer keeps the old semantics exactly.
void test_zero_length_ramp_steps()
{
  nam::ParamRamp ramp;
  ramp.Configure(1, kSampleRate, 0.0f);
  const std::array<float, 1> initial{0.0f};
  ramp.Snap(initial);
  const std::array<float, 1> target{1.0f};
  assert(ramp.SetTarget(target));
  assert(!ramp.IsRamping());
  assert(ramp.Current()[0] == 1.0f);

  nam::ParamRamp unknown_rate;
  unknown_rate.Configure(1, -1.0, nam::kDefaultParamRampSeconds);
  unknown_rate.Snap(initial);
  assert(unknown_rate.SetTarget(target));
  assert(!unknown_rate.IsRamping());
  assert(unknown_rate.Current()[0] == 1.0f);
}

// Step() is the per-sample slope callers walk across a block; it must agree with where
// Advance() lands, otherwise a block-filling loop and the ramp state would disagree.
void test_step_matches_advance()
{
  auto ramp = make_ramp({0.0f});
  const std::array<float, 1> target{1.0f};
  ramp.SetTarget(target);

  constexpr auto frames = 64;
  auto walked = ramp.Current()[0];
  const auto slope = ramp.Step()[0];
  for (auto i = 0; i < frames; ++i)
    walked += slope;

  ramp.Advance(frames);
  assert(std::abs(walked - ramp.Current()[0]) < 1.0e-6f);
}

} // namespace test_param_ramp
