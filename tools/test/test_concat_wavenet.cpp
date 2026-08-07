// ConcatWaveNet loader, encoding-parity, validation, and realtime-safety tests.

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/model_config.h"
#include "NAM/parametric_control.h"
#include "allocation_tracking.h"

namespace test_concat_wavenet
{
namespace
{

nlohmann::json params()
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

nlohmann::json inner_config(const bool explicit_derived_sizes = false)
{
  nlohmann::json layer{
    {"channels", 1},        {"head", {{"out_channels", 1}, {"kernel_size", 1}, {"bias", false}}},
    {"kernel_size", 1},     {"dilations", {1}},
    {"activation", "ReLU"}, {"layer1x1", {{"active", true}, {"groups", 1}}},
  };
  if (explicit_derived_sizes)
  {
    layer["input_size"] = 5; // audio + continuous + three-way one-hot switch
    layer["condition_size"] = 5;
  }
  return nlohmann::json{{"head_scale", 1.0}, {"layers", {std::move(layer)}}, {"params", params()}};
}

// The tiny WaveNet has 16 weights. selected_input is a channel in the concatenated
// [audio, continuous, switch-0, switch-1, switch-2] tensor. Only the input mixin
// consumes it, making the expected output ReLU(selected channel).
std::vector<float> weights_selecting(const int selected_input)
{
  std::vector<float> weights{
    0, 0, 0, 0, 0, // rechannel 5 -> 1
    0, 0, // layer conv weight, bias
    0, 0, 0, 0, 0, // input mixin 5 -> 1
    0, 0, // layer 1x1 weight, bias
    1, // head rechannel
    1, // head scale
  };
  weights[7 + selected_input] = 1.0f;
  return weights;
}

nlohmann::json model_json(const int selected_input = 1)
{
  return nlohmann::json{
    {"version", "0.7.0"},       {"metadata", nlohmann::json::object()},         {"architecture", "ConcatWaveNet"},
    {"config", inner_config()}, {"weights", weights_selecting(selected_input)}, {"sample_rate", 48000},
  };
}

std::vector<float> process(nam::DSP& dsp, const float input_value = 0.25f)
{
  constexpr int frames = 8;
  std::vector<NAM_SAMPLE> input(frames, input_value);
  std::vector<NAM_SAMPLE> output(frames);
  std::array<NAM_SAMPLE*, 1> inputs{input.data()};
  std::array<NAM_SAMPLE*, 1> outputs{output.data()};
  dsp.process(inputs.data(), outputs.data(), frames);
  return std::vector<float>(output.begin(), output.end());
}

template <typename Fn>
void assert_runtime_error(Fn&& fn, const std::string& text)
{
  bool threw = false;
  try
  {
    fn();
  }
  catch (const std::runtime_error& e)
  {
    threw = true;
    assert(std::string(e.what()).find(text) != std::string::npos);
  }
  assert(threw);
}

} // namespace

void test_load_control_and_continuous_encoding()
{
  assert(nam::ConfigParserRegistry::instance().has("ConcatWaveNet"));
  auto dsp = nam::get_dsp(model_json());
  assert(dsp->NumInputChannels() == 1);
  assert(dsp->NumOutputChannels() == 1);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  assert(control->ParamDim() == 2);
  assert(control->GetParamSpecs()[0].name == "gain");
  assert(control->GetParamSpecs()[1].enum_names.size() == 3);
  assert(control->GetParams()[0] == 5.0f);
  assert(control->GetParams()[1] == 1.0f);

  dsp->SetPrewarmOnReset(false);
  dsp->Reset(48000.0, 16);
  const auto nominal = process(*dsp);
  control->SetParams(std::array<float, 2>{10.0f, 1.0f});
  const auto maximum = process(*dsp);
  for (size_t i = 0; i < nominal.size(); ++i)
  {
    assert(std::abs(nominal[i]) < 1.0e-7f); // gain=5 -> signed min-max encoding 0
    assert(std::abs(maximum[i] - 1.0f) < 1.0e-7f); // gain=10 -> +1
  }
}

void test_switch_one_hot_encoding()
{
  // Select the "crunch" one-hot input. It is 1 at the default mode and 0 at lead.
  auto dsp = nam::get_dsp(model_json(3));
  dsp->SetPrewarmOnReset(false);
  dsp->Reset(48000.0, 16);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  const auto crunch = process(*dsp);
  control->SetParams(std::array<float, 2>{5.0f, 2.0f});
  const auto lead = process(*dsp);
  for (size_t i = 0; i < crunch.size(); ++i)
  {
    assert(crunch[i] == 1.0f);
    assert(lead[i] == 0.0f);
  }

  bool threw = false;
  try
  {
    control->SetParams(std::array<float, 2>{5.0f, 1.5f});
  }
  catch (const std::invalid_argument&)
  {
    threw = true;
  }
  assert(threw);
}

void test_matches_manual_concat_wavenet()
{
  auto concat = nam::get_dsp(model_json(4)); // select switch "lead"

  auto plain_json = model_json(4);
  plain_json["architecture"] = "WaveNet";
  plain_json["config"] = inner_config(true);
  plain_json["config"].erase("params");
  plain_json["config"]["in_channels"] = 5;
  auto plain = nam::get_dsp(plain_json);

  concat->SetPrewarmOnReset(false);
  plain->SetPrewarmOnReset(false);
  concat->Reset(48000.0, 16);
  plain->Reset(48000.0, 16);
  auto* control = dynamic_cast<nam::IParametricControl*>(concat.get());
  control->SetParams(std::array<float, 2>{2.5f, 2.0f});

  constexpr int frames = 8;
  std::vector<NAM_SAMPLE> audio(frames, 0.25f);
  std::vector<NAM_SAMPLE> continuous(frames, -0.5f);
  std::vector<NAM_SAMPLE> switch0(frames, 0.0f);
  std::vector<NAM_SAMPLE> switch1(frames, 0.0f);
  std::vector<NAM_SAMPLE> switch2(frames, 1.0f);
  std::vector<NAM_SAMPLE> expected(frames), actual(frames);
  std::array<NAM_SAMPLE*, 5> plain_inputs{
    audio.data(), continuous.data(), switch0.data(), switch1.data(), switch2.data()};
  std::array<NAM_SAMPLE*, 1> expected_ptr{expected.data()};
  std::array<NAM_SAMPLE*, 1> concat_input{audio.data()};
  std::array<NAM_SAMPLE*, 1> actual_ptr{actual.data()};
  plain->process(plain_inputs.data(), expected_ptr.data(), frames);
  concat->process(concat_input.data(), actual_ptr.data(), frames);
  assert(actual == expected);
}

void test_parser_validation()
{
  auto config = model_json();
  config["config"].erase("params");
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "params");

  config = model_json();
  config["config"]["layers"][0]["input_size"] = 1;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "input_size");

  config = model_json();
  config["config"]["layers"][0]["condition_size"] = 1;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "condition_size");

  config = model_json();
  config["config"]["layers"][0]["packing"] = {{"num_models", 2}};
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "packed");

  config = model_json();
  config["config"]["params"][1]["enum_names"] = {"same", "same", "other"};
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "unique");
}

void test_setparams_and_process_realtime_safe()
{
  auto dsp = nam::get_dsp(model_json());
  dsp->SetPrewarmOnReset(false);
  dsp->Reset(48000.0, 16);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  std::array<NAM_SAMPLE, 8> input{};
  std::array<NAM_SAMPLE, 8> output{};
  std::array<NAM_SAMPLE*, 1> inputs{input.data()};
  std::array<NAM_SAMPLE*, 1> outputs{output.data()};
  const std::array<float, 2> first{2.0f, 0.0f};
  const std::array<float, 2> second{8.0f, 2.0f};

  control->SetParams(first);
  dsp->process(inputs.data(), outputs.data(), 8);
  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      control->SetParams(first);
      dsp->process(inputs.data(), outputs.data(), 8);
      control->SetParams(second);
      dsp->process(inputs.data(), outputs.data(), 8);
    },
    nullptr, "ConcatWaveNet SetParams + process realtime safety");
}

} // namespace test_concat_wavenet
