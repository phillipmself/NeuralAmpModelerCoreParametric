// FiLMWaveNet loader, encoding-parity, validation, and realtime-safety tests.
//
// The synthetic model below is built so that, after warmup, the audio input has zero effect
// on the output: every path that would otherwise carry it (rechannel -> conv, layer condition
// -> input_mixin) is zeroed by weight, and the surviving path is
//   activation_post_film shift == encoded_params[selected] -> layer1x1 (unused by the head
//   branch) / head rechannel (identity) -> head_scale (identity).
// That isolates the FiLM control path: output[n] == encoded_params[selected] for every frame,
// letting these tests assert exact values instead of just "changed".
//
// tools/test/test_film_wavenet_parity.cpp covers the same runtime against a real Python export.

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

namespace test_film_wavenet
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

// gain (1) + mode one-hot (3) = 4-wide encoded control.
constexpr int kEncodedDim = 4;

nlohmann::json layer_json()
{
  return nlohmann::json{
    {"channels", 1},
    {"head", {{"out_channels", 1}, {"kernel_size", 1}, {"bias", true}}},
    {"kernel_size", 1},
    {"dilations", {1}},
    {"activation", "ReLU"},
    {"layer1x1", {{"active", true}, {"groups", 1}}},
    {"activation_post_film", {{"active", true}, {"shift", true}, {"groups", 1}}},
  };
}

nlohmann::json inner_config(const nlohmann::json& param_encoder = nullptr)
{
  nlohmann::json config{{"head_scale", 1.0}, {"layers", {layer_json()}}, {"params", params()}};
  if (!param_encoder.is_null())
    config["param_encoder"] = param_encoder;
  return config;
}

// 19 weights: rechannel(1), conv(2), input_mixin(1), layer1x1(2),
// activation_post_film scale/shift matrix (2x4 row-major) + bias(2) (10), head_rechannel(2), head_scale(1).
// Only layer1x1's weight, the FiLM shift row's `selected_encoded_index` column, head_rechannel's weight, and
// head_scale are nonzero -- see the file comment for why that isolates encoded_params[selected_encoded_index].
std::vector<float> weights_selecting(const int selected_encoded_index)
{
  std::vector<float> weights(19, 0.0f);
  weights[4] = 1.0f; // layer1x1 weight
  weights[10 + selected_encoded_index] = 1.0f; // activation_post_film shift row, selected column
  weights[16] = 1.0f; // head_rechannel weight
  weights[18] = 1.0f; // head_scale
  return weights;
}

// Identity param_encoder (no hidden layer): weight = I(4), bias = 0. Prepended to weights_selecting()
// so the FiLM condition is unchanged by the encoder, letting the same expected values apply.
std::vector<float> identity_encoder_weights()
{
  std::vector<float> weights(kEncodedDim * kEncodedDim + kEncodedDim, 0.0f);
  for (int i = 0; i < kEncodedDim; ++i)
    weights[i * kEncodedDim + i] = 1.0f;
  return weights;
}

nlohmann::json model_json(const int selected_encoded_index = 0, const nlohmann::json& param_encoder = nullptr)
{
  auto weights = weights_selecting(selected_encoded_index);
  if (!param_encoder.is_null())
  {
    auto encoder_weights = identity_encoder_weights();
    encoder_weights.insert(encoder_weights.end(), weights.begin(), weights.end());
    weights = std::move(encoder_weights);
  }
  return nlohmann::json{
    {"version", "1.0.0"},
    {"metadata", nlohmann::json::object()},
    {"architecture", "FiLMWaveNet"},
    {"config", inner_config(param_encoder)},
    {"weights", weights},
    {"sample_rate", 48000},
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

// The encoding documented on FiLMWaveNet (shared with ConcatWaveNet): continuous params are
// signed min-max scaled to [-1, 1], switches become a one-hot block.
float expected_gain_encoding(const float gain)
{
  const auto fraction = (gain - 0.0) / (10.0 - 0.0);
  return static_cast<float>(-1.0 + 2.0 * fraction);
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

void test_load_and_select_continuous()
{
  assert(nam::ConfigParserRegistry::instance().has("FiLMWaveNet"));
  auto dsp = nam::get_dsp(model_json(0)); // select encoded_params[0] == encoded gain
  assert(dsp->NumInputChannels() == 1);
  assert(dsp->NumOutputChannels() == 1);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(control != nullptr);
  assert(control->ParamDim() == 2);
  assert(control->GetParamSpecs()[0].name == "gain");
  assert(control->GetParams()[0] == 5.0f);

  dsp->SetPrewarmOnReset(false);
  dsp->Reset(48000.0, 16);
  const auto nominal = process(*dsp);
  const auto expected_nominal = expected_gain_encoding(5.0f);
  for (const auto sample : nominal)
    assert(std::abs(sample - expected_nominal) < 1.0e-6f);

  control->SetParams(std::array<float, 2>{10.0f, 1.0f});
  const auto maximum = process(*dsp);
  for (const auto sample : maximum)
    assert(std::abs(sample - 1.0f) < 1.0e-6f); // gain=10 -> +1
}

void test_switch_one_hot_encoding()
{
  // Global encoded index 2 == the "crunch" one-hot slot (index 0 is gain).
  auto dsp = nam::get_dsp(model_json(2));
  dsp->SetPrewarmOnReset(false);
  dsp->Reset(48000.0, 16);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  const auto crunch = process(*dsp); // default mode == 1 == crunch
  for (const auto sample : crunch)
    assert(sample == 1.0f);

  control->SetParams(std::array<float, 2>{5.0f, 2.0f}); // lead
  const auto lead = process(*dsp);
  for (const auto sample : lead)
    assert(sample == 0.0f);

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

void test_param_encoder_weight_offset()
{
  // An identity param_encoder must not change the FiLM condition, so this must match
  // test_load_and_select_continuous's plain (no-encoder) expectations -- which only holds if
  // the encoder's weights were consumed first and the wavenet's own weights start after them.
  const nlohmann::json encoder{
    {"hidden_sizes", nlohmann::json::array()}, {"out_features", kEncodedDim}, {"activation", "ReLU"}};
  auto dsp = nam::get_dsp(model_json(0, encoder));
  dsp->SetPrewarmOnReset(false);
  dsp->Reset(48000.0, 16);
  auto* control = dynamic_cast<nam::IParametricControl*>(dsp.get());
  control->SetParams(std::array<float, 2>{10.0f, 1.0f});
  const auto maximum = process(*dsp);
  for (const auto sample : maximum)
    assert(std::abs(sample - 1.0f) < 1.0e-6f);
}

void test_unsupported_param_encoder_activation_throws()
{
  const nlohmann::json encoder{{"hidden_sizes", {5}}, {"out_features", kEncodedDim}, {"activation", "NotAnActivation"}};
  assert_runtime_error([&]() { (void)nam::get_dsp(model_json(0, encoder)); }, "activation");
}

void test_parser_validation()
{
  auto config = model_json();
  config["config"].erase("params");
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "params");

  config = model_json();
  config["config"]["condition_dsp"] = nlohmann::json::object();
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "condition_dsp");

  config = model_json();
  config["config"]["layers"][0]["packing"] = {{"num_models", 2}};
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "packed");

  config = model_json();
  config["config"]["layers"][0]["input_size"] = 2;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "input_size");

  config = model_json();
  config["config"]["layers"][0]["condition_size"] = 2;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "condition_size");

  config = model_json();
  config["config"]["layers"][0]["film_condition_size"] = kEncodedDim + 1;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "film_condition_size");

  config = model_json();
  config["config"]["params"][1]["enum_names"] = {"same", "same", "other"};
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "unique");

  // With no active FiLM site the controls have no path into the network at all: the model would
  // load and render identically at every setting, silently. Python rejects this at init too.
  config = model_json();
  config["config"]["layers"][0]["activation_post_film"]["active"] = false;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "at least one active FiLM");
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
    nullptr, "FiLMWaveNet SetParams + process realtime safety");
}

} // namespace test_film_wavenet
