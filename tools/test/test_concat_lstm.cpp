// ConcatLSTM loader, encoding-parity, validation, and realtime-safety tests.

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/concat_lstm.h"
#include "NAM/get_dsp.h"
#include "NAM/model_config.h"
#include "NAM/param_ramp.h"
#include "NAM/parametric_control.h"
#include "allocation_tracking.h"

namespace test_concat_lstm
{
namespace
{

constexpr int kHiddenSize = 3;
constexpr int kNumLayers = 2;
// audio + one continuous control + a three-way one-hot switch.
constexpr int kInnerChannels = 5;

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

// Deterministic and platform-independent (integer LCG, no floating-point RNG), so the
// ConcatLSTM and the hand-fed stock LSTM it is compared against load identical weights.
std::vector<float> make_weights(const size_t count)
{
  std::vector<float> weights(count);
  uint32_t state = 12345u;
  for (size_t i = 0; i < count; ++i)
  {
    state = state * 1664525u + 1013904223u;
    weights[i] = 0.25f * (static_cast<float>((state >> 16) % 2001u) / 1000.0f - 1.0f);
  }
  return weights;
}

size_t weight_count()
{
  nam::lstm::LSTMConfig shape;
  shape.num_layers = kNumLayers;
  shape.hidden_size = kHiddenSize;
  shape.input_size = kInnerChannels;
  shape.in_channels = kInnerChannels;
  shape.out_channels = 1;
  return nam::lstm::concat_lstm_weight_count(shape);
}

nlohmann::json model_json()
{
  return nlohmann::json{
    {"version", "1.0.0"},
    {"metadata", nlohmann::json::object()},
    {"architecture", "ConcatLSTM"},
    // Mirrors the trainer's exported config: input_size is deliberately absent, and the
    // truncated-BPTT hyperparameters ride along unused.
    {"config",
     {{"hidden_size", kHiddenSize},
      {"num_layers", kNumLayers},
      {"train_burn_in", 4096},
      {"train_truncate", 512},
      {"params", params()}}},
    {"weights", make_weights(weight_count())},
    {"sample_rate", 48000},
  };
}

// The same recurrent core, loaded as a stock multi-channel LSTM so the conditioning
// channels can be supplied by hand.
nlohmann::json plain_lstm_json()
{
  return nlohmann::json{
    {"version", "0.7.0"},
    {"metadata", nlohmann::json::object()},
    {"architecture", "LSTM"},
    {"config",
     {{"hidden_size", kHiddenSize},
      {"num_layers", kNumLayers},
      {"input_size", kInnerChannels},
      {"in_channels", kInnerChannels},
      {"out_channels", 1}}},
    {"weights", make_weights(weight_count())},
    {"sample_rate", 48000},
  };
}

std::vector<NAM_SAMPLE> audio_block(const int frames)
{
  std::vector<NAM_SAMPLE> audio(static_cast<size_t>(frames));
  for (int i = 0; i < frames; ++i)
    audio[static_cast<size_t>(i)] = static_cast<NAM_SAMPLE>(0.5 * std::sin(0.37 * i));
  return audio;
}

std::vector<float> process(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& audio)
{
  std::vector<NAM_SAMPLE> input = audio;
  std::vector<NAM_SAMPLE> output(audio.size());
  std::array<NAM_SAMPLE*, 1> inputs{input.data()};
  std::array<NAM_SAMPLE*, 1> outputs{output.data()};
  dsp.process(inputs.data(), outputs.data(), static_cast<int>(audio.size()));
  return std::vector<float>(output.begin(), output.end());
}

// Continuous controls are ramped to rather than applied instantly, so run the ramp out
// before asserting on the encoding itself. Derived from the ramp length rather than
// hard-coded, so retuning it does not silently gut these tests.
void settle_params(nam::DSP& dsp)
{
  const auto audio = audio_block(8);
  const auto ramp_frames = 48000.0f * nam::kDefaultParamRampSeconds;
  const int settle_blocks = static_cast<int>(ramp_frames / 8.0f) + 8;
  for (int i = 0; i < settle_blocks; ++i)
    (void)process(dsp, audio);
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

void test_load_and_control()
{
  assert(nam::ConfigParserRegistry::instance().has("ConcatLSTM"));
  auto dsp = nam::get_dsp(model_json());
  // The conditioning channels are an implementation detail: the host still sees mono in.
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
  const auto audio = audio_block(8);
  const auto nominal = process(*dsp, audio);

  control->SetParams(std::array<float, 2>{10.0f, 2.0f});
  assert(control->GetParams()[0] == 10.0f);
  settle_params(*dsp);
  const auto moved = process(*dsp, audio);

  // The control has to actually reach the recurrence, not merely be stored.
  bool differs = false;
  for (size_t i = 0; i < nominal.size(); ++i)
    differs |= std::abs(moved[i] - nominal[i]) > 1.0e-6f;
  assert(differs);
}

// The conditioning contract in full: a ConcatLSTM at a given setting must be the stock LSTM
// fed [audio, encoded controls] by hand. This pins the encoding (signed min-max for the
// continuous control, one-hot for the switch), the channel order, and the weight layout in
// one comparison, without needing the trainer to produce a fixture.
void test_matches_manual_lstm()
{
  auto concat = nam::get_dsp(model_json());
  auto plain = nam::get_dsp(plain_lstm_json());
  auto* control = dynamic_cast<nam::IParametricControl*>(concat.get());
  assert(control != nullptr);

  // Committed before Reset(), which settles the ramp on it: a reset is a stream restart, so
  // both models start from the loaded initial hidden/cell states with no ramp in flight.
  control->SetParams(std::array<float, 2>{2.5f, 2.0f});
  concat->SetPrewarmOnReset(false);
  plain->SetPrewarmOnReset(false);
  concat->Reset(48000.0, 64);
  plain->Reset(48000.0, 64);

  constexpr int frames = 32;
  auto audio = audio_block(frames);
  // gain 2.5 over [0, 10] -> fraction 0.25 -> signed min-max encoding -0.5.
  std::vector<NAM_SAMPLE> gain(frames, -0.5);
  std::vector<NAM_SAMPLE> clean(frames, 0.0);
  std::vector<NAM_SAMPLE> crunch(frames, 0.0);
  std::vector<NAM_SAMPLE> lead(frames, 1.0);
  std::vector<NAM_SAMPLE> expected(frames);
  std::array<NAM_SAMPLE*, kInnerChannels> plain_inputs{
    audio.data(), gain.data(), clean.data(), crunch.data(), lead.data()};
  std::array<NAM_SAMPLE*, 1> expected_ptr{expected.data()};
  plain->process(plain_inputs.data(), expected_ptr.data(), frames);

  const auto actual = process(*concat, audio);
  for (int i = 0; i < frames; ++i)
    assert(actual[static_cast<size_t>(i)] == static_cast<float>(expected[static_cast<size_t>(i)]));
}

void test_parser_validation()
{
  auto config = model_json();
  config["config"].erase("params");
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "params");

  config = model_json();
  config["config"].erase("hidden_size");
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "hidden_size");

  config = model_json();
  config["config"]["num_layers"] = 0;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "num_layers");

  // input_size and in_channels are derived, not read; a config that disagrees is rejected
  // rather than silently overridden.
  config = model_json();
  config["config"]["input_size"] = 1;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "input_size");

  config = model_json();
  config["config"]["in_channels"] = 4;
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "in_channels");

  // Agreeing with the derived value is fine.
  config = model_json();
  config["config"]["input_size"] = kInnerChannels;
  config["config"]["in_channels"] = kInnerChannels;
  (void)nam::get_dsp(config);

  config = model_json();
  config["config"]["params"][1]["enum_names"] = {"same", "same", "other"};
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "unique");

  // Stock-era file versions belong to the 0.x namespace and must not load as parametric.
  config = model_json();
  config["version"] = "0.7.0";
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "unsupported version");
}

// nam::lstm::LSTM only assert()s that it consumed the blob exactly, so the size has to be
// checked before construction or a short file walks its iterator off the end in release.
void test_weight_count_validation()
{
  auto config = model_json();
  config["weights"].erase(config["weights"].end() - 1);
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "expected " + std::to_string(weight_count()));

  config = model_json();
  config["weights"].push_back(0.0);
  assert_runtime_error([&]() { (void)nam::get_dsp(config); }, "weights");
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
    nullptr, "ConcatLSTM SetParams + process realtime safety");
}

} // namespace test_concat_lstm
