// Tests for HyperWaveNet registration, runtime control, and RT safety

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/parametric_control.h"
#include "allocation_tracking.h"

namespace test_hyperwavenet
{
namespace
{

nlohmann::json make_config()
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
    {"weights",
     {
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f,
       1.0f,
       0.0f,
       0.0f,
     }},
    {"sample_rate", 48000},
  };
}

// A low-rank HyperWaveNet targeting the head_scale weight (export_offset 7).
//
// Sizes: 1 continuous param -> encoded_dim = 1; hidden_sizes = [] -> in_final = 1.
// One low-rank target with out_features = rest_features = rank = 1 -> numel = 1,
// output_width = rank * (out + rest) = 2, final_out = 2.
// final_param_count = final_out * in_final + final_out = 4; anchor = final_out = 2.
// hypernet_state_count = 0 (trunk) + 4 (final) + 2 (anchor) = 6; base_len = 8; total = 14.
nlohmann::json make_low_rank_config()
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
        {{"mode", "low_rank"},
         {"rank", 1},
         {"hidden_sizes", nlohmann::json::array()},
         {"activation", "ReLU"},
         {"delta_map",
          {{{"name", "head_scale"},
            {"mode", "low_rank"},
            {"numel", 1},
            {"export_offset", 7},
            {"rank", 1},
            {"out_features", 1},
            {"rest_features", 1}}}}}},
     }},
    {"weights",
     {
       // base WaveNet (8) | final weight (2) | final bias (2) | anchor (2)
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f, // base
       1.0f,
       1.0f, // final weight rows
       0.0f,
       0.0f, // final bias
       0.0f,
       0.0f, // anchor
     }},
    {"sample_rate", 48000},
  };
}

nlohmann::json make_condition_dsp_json()
{
  return nlohmann::json{
    {"version", "0.7.0"},
    {"metadata", nlohmann::json::object()},
    {"architecture", "WaveNet"},
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
     }},
    {"weights",
     {
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f,
     }},
    {"sample_rate", 48000},
  };
}

void reset_for_test(nam::DSP& dsp)
{
  constexpr auto sample_rate = 48000.0;
  constexpr auto buffer_size = 16;
  dsp.Reset(sample_rate, buffer_size);
}

std::vector<float> process_constant_block(nam::DSP& dsp, const float input_value)
{
  constexpr auto num_frames = 8;
  std::vector<NAM_SAMPLE> input(num_frames, input_value);
  std::vector<NAM_SAMPLE> output(num_frames, 0.0f);
  std::array<NAM_SAMPLE*, 1> input_ptrs{input.data()};
  std::array<NAM_SAMPLE*, 1> output_ptrs{output.data()};
  dsp.process(input_ptrs.data(), output_ptrs.data(), num_frames);

  return std::vector<float>(output.begin(), output.end());
}

void process_single_channel(nam::DSP& dsp, std::vector<NAM_SAMPLE>& input, std::vector<NAM_SAMPLE>& output)
{
  assert(input.size() == output.size());
  std::array<NAM_SAMPLE*, 1> input_ptrs{input.data()};
  std::array<NAM_SAMPLE*, 1> output_ptrs{output.data()};
  dsp.process(input_ptrs.data(), output_ptrs.data(), static_cast<int>(input.size()));
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

template <typename Fn>
void assert_throws_runtime_error(Fn&& fn, const std::string& expected_substring)
{
  auto threw = false;
  try
  {
    fn();
  }
  catch (const std::runtime_error& e)
  {
    threw = true;
    assert(std::string(e.what()).find(expected_substring) != std::string::npos);
  }
  assert(threw);
}

} // namespace

void test_load_and_control_hyperwavenet()
{
  auto dsp = nam::get_dsp(make_config());
  assert(dsp != nullptr);

  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  assert(control->ParamDim() == 1);
  assert(control->GetParamSpecs().size() == 1);
  assert(control->GetParamSpecs()[0].name == "drive");
  assert(control->GetParams().size() == 1);
  assert(std::abs(control->GetParams()[0] - 0.5f) < 1.0e-6f);

  auto nominal_reference = nam::get_dsp(make_config());
  auto updated_reference = nam::get_dsp(make_config());
  auto* updated_control = dynamic_cast<nam::IParametricControl*>(updated_reference.get());
  assert(updated_control != nullptr);

  reset_for_test(*dsp);
  reset_for_test(*nominal_reference);
  reset_for_test(*updated_reference);

  const auto first_output = process_constant_block(*dsp, 1.0f);
  const auto nominal_first_output = process_constant_block(*nominal_reference, 1.0f);
  updated_control->SetParams(std::vector<float>{1.0f});
  const auto updated_first_output = process_constant_block(*updated_reference, 1.0f);

  for (size_t i = 0; i < first_output.size(); ++i)
  {
    assert(std::isfinite(first_output[i]));
    assert(std::isfinite(nominal_first_output[i]));
    assert(std::isfinite(updated_first_output[i]));
    assert(std::abs(first_output[i] - nominal_first_output[i]) < 1.0e-6f);
  }

  control->SetParams(std::vector<float>{1.0f});
  assert(std::abs(control->GetParams()[0] - 1.0f) < 1.0e-6f);
  const auto second_output = process_constant_block(*dsp, 1.0f);
  const auto nominal_second_output = process_constant_block(*nominal_reference, 1.0f);
  const auto updated_second_output = process_constant_block(*updated_reference, 1.0f);

  auto any_changed = false;
  for (size_t i = 0; i < second_output.size(); ++i)
  {
    assert(std::isfinite(second_output[i]));
    assert(std::isfinite(nominal_second_output[i]));
    assert(std::isfinite(updated_second_output[i]));
    if (std::abs(second_output[i] - nominal_second_output[i]) > 1.0e-6f)
      any_changed = true;
    assert(std::abs(second_output[i] - updated_second_output[i]) < 1.0e-6f);
  }
  assert(any_changed);
  assert_throws_invalid_argument([&]() { control->SetParams(std::vector<float>{}); });
}

void test_reject_condition_dsp_for_now()
{
  auto config = make_config();
  config["config"]["condition_dsp"] = make_condition_dsp_json();

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "HyperWaveNet does not yet support condition_dsp");
}

void test_reject_duplicate_param_names()
{
  auto config = make_config();
  config["config"]["params"] = nlohmann::json::array({
    {{"name", "drive"}, {"min", 0.0}, {"max", 1.0}, {"default", 0.5}, {"type", "continuous"}},
    {{"name", "drive"}, {"min", 0.0}, {"max", 1.0}, {"default", 0.25}, {"type", "continuous"}},
  });

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "duplicates an earlier parameter name");
}

void test_load_and_control_low_rank_hyperwavenet()
{
  auto dsp = nam::get_dsp(make_low_rank_config());
  assert(dsp != nullptr);

  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  assert(control->ParamDim() == 1);

  reset_for_test(*dsp);
  const auto first_output = process_constant_block(*dsp, 1.0f);

  control->SetParams(std::vector<float>{1.0f});
  const auto second_output = process_constant_block(*dsp, 1.0f);

  auto any_changed = false;
  for (size_t i = 0; i < first_output.size(); ++i)
  {
    assert(std::isfinite(first_output[i]));
    assert(std::isfinite(second_output[i]));
    if (std::abs(first_output[i] - second_output[i]) > 1.0e-6f)
      any_changed = true;
  }
  assert(any_changed);
}

void test_reject_switch_param_out_of_range()
{
  auto config = make_config();
  // Switch with three options must use index bounds [0, 2]; max = 5 is invalid.
  config["config"]["params"] = nlohmann::json::array({
    {{"name", "mode"},
     {"min", 0.0},
     {"max", 5.0},
     {"default", 0.0},
     {"type", "switch"},
     {"enum_names", {"a", "b", "c"}}},
  });

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "switch range must be [0, 2]");
}

void test_reject_non_integer_switch_index()
{
  auto config = make_config();
  config["config"]["params"] = nlohmann::json::array({
    {{"name", "mode"},
     {"min", 0.0},
     {"max", 1.0},
     {"default", 0.5}, // non-integer index
     {"type", "switch"},
     {"enum_names", {"a", "b"}}},
  });

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "must be integer indices");
}

void test_reject_low_rank_target_under_full_mode()
{
  auto config = make_config();
  // hypernet.mode stays "full" but the target claims low_rank: rejected at parse time.
  config["config"]["hypernet"]["delta_map"][0]["mode"] = "low_rank";

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "low_rank but hypernet.mode is 'full'");
}

void test_reject_short_weight_blob()
{
  auto config = make_config();
  // Fewer weights than the hypernet state requires (state_count = 3 for this config).
  config["weights"] = nlohmann::json::array({1.0f, 1.0f});

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "weight blob too short");
}

void test_reject_export_offset_out_of_range()
{
  auto config = make_config();
  // export_offset past the end of the base-weight region (base_len = 8).
  config["config"]["hypernet"]["delta_map"][0]["export_offset"] = 100;

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "exceeds base weight length");
}

void test_setparams_process_realtime_safe()
{
  using namespace allocation_tracking;

  auto dsp = nam::get_dsp(make_config());
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);

  reset_for_test(*dsp);

  std::vector<NAM_SAMPLE> input(8, 1.0f);
  std::vector<NAM_SAMPLE> output(input.size(), 0.0f);
  process_single_channel(*dsp, input, output); // Warm steady-state buffers before tracking.

  const std::array<float, 1> params{1.0f};
  run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      control->SetParams(params);
      process_single_channel(*dsp, input, output);
    },
    nullptr, "test_setparams_process_realtime_safe");

  for (const auto sample : output)
    assert(std::isfinite(sample));
}

void test_reject_packed_layers()
{
  auto config = make_config();
  config["config"]["layers"][0]["packing"] = {{"num_models", 2}};

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "packed");
}

void test_reject_duplicate_enum_names()
{
  // The enum_names non-empty/uniqueness check previously only fired for ConcatWaveNet;
  // it now runs through the shared parser for HyperWaveNet too.
  auto config = make_config();
  config["config"]["params"] = nlohmann::json::array({
    {{"name", "mode"},
     {"min", 0.0},
     {"max", 2.0},
     {"default", 0.0},
     {"type", "switch"},
     {"enum_names", {"same", "same", "other"}}},
  });

  assert_throws_runtime_error([&]() { (void)nam::get_dsp(config); }, "unique");
}

void test_ignores_step_and_avoid_zero()
{
  // "step" and "avoid_zero" are capture-planning metadata the trainer writes into every
  // params[] entry (nam/models/parametric/_spec.py). They are not runtime concerns: the
  // parser must silently ignore them rather than rejecting the config or surfacing them.
  // This includes the switch shape the trainer emits ("step": null, "avoid_zero": false),
  // which would be rejected outright if the parser still validated these fields.
  auto config = make_config();
  config["config"]["params"] = nlohmann::json::array({
    {{"name", "drive"},
     {"min", 0.0},
     {"max", 1.0},
     {"default", 0.5},
     {"type", "continuous"},
     {"step", 0.5},
     {"avoid_zero", true}},
    {{"name", "mode"},
     {"min", 0.0},
     {"max", 1.0},
     {"default", 0.0},
     {"type", "switch"},
     {"enum_names", {"a", "b"}},
     {"step", nullptr},
     {"avoid_zero", false}},
  });
  // Adding the "mode" switch grows the encoded param dim from 1 (drive only) to 3
  // (drive + a 2-way one-hot), which grows the hypernet's final-layer weight count from
  // 1 to 3. Re-size the weight blob to match: base (8, unchanged) + final weight (3) +
  // final bias (1) + anchor (1) = 13.
  config["weights"] = nlohmann::json::array({
    1.0f,
    1.0f,
    0.0f,
    1.0f,
    1.0f,
    0.0f,
    1.0f,
    1.0f, // base WaveNet (8)
    1.0f,
    0.0f,
    0.0f, // hypernet final weight (final_out=1 x encoded_dim=3)
    0.0f, // hypernet final bias
    0.0f, // hypernet anchor
  });

  auto dsp = nam::get_dsp(config);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  const auto& specs = control->GetParamSpecs();
  assert(specs.size() == 2);
  assert(specs[0].name == "drive");
  assert(specs[0].min == 0.0f);
  assert(specs[0].max == 1.0f);
  assert(specs[0].defaultValue == 0.5f);
  assert(specs[0].type == "continuous");
  assert(specs[1].name == "mode");
  assert(specs[1].enum_names.size() == 2);
}

} // namespace test_hyperwavenet
