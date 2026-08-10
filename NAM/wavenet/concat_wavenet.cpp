#include "concat_wavenet.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>

#include "../registry.h"
#include "../parametric_version.h"

namespace nam
{
namespace wavenet
{

ConcatWaveNet::ConcatWaveNet(std::unique_ptr<WaveNet> wavenet, std::vector<ParamSpec> param_specs,
                             const double sample_rate)
: DSP(1, wavenet == nullptr ? 1 : wavenet->NumOutputChannels(), sample_rate)
, _wavenet(std::move(wavenet))
, _conditioner(std::move(param_specs))
{
  if (_wavenet == nullptr)
    throw std::invalid_argument("ConcatWaveNet: inner WaveNet must not be null");
  if (_wavenet->NumInputChannels() != _conditioner.NumInputChannels())
    throw std::invalid_argument("ConcatWaveNet: inner WaveNet input channel count does not match encoded params");
  _conditioner.Configure(sample_rate);
}

void ConcatWaveNet::SetParams(const std::span<const float> params)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    _conditioner.SetParams(params);
#ifndef NDEBUG
  }
  catch (...)
  {
    _debug_leave_param_api_();
    throw;
  }
  _debug_leave_param_api_();
#endif
}

std::span<const float> ConcatWaveNet::GetParams() const
{
#ifndef NDEBUG
  const_cast<ConcatWaveNet*>(this)->_debug_enter_param_api_();
#endif
  const auto result = _conditioner.Params();
#ifndef NDEBUG
  const_cast<ConcatWaveNet*>(this)->_debug_leave_param_api_();
#endif
  return result;
}

int ConcatWaveNet::ParamDim() const
{
  return _conditioner.ParamDim();
}

const std::vector<ParamSpec>& ConcatWaveNet::GetParamSpecs() const
{
  return _conditioner.ParamSpecs();
}

void ConcatWaveNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    assert(num_frames <= mMaxBufferSize);
    _wavenet->process(_conditioner.PrepareBlock(input[0], num_frames), output, num_frames);
#ifndef NDEBUG
  }
  catch (...)
  {
    _debug_leave_param_api_();
    throw;
  }
  _debug_leave_param_api_();
#endif
}

void ConcatWaveNet::Reset(const double sampleRate, const int maxBufferSize)
{
  const auto prewarm_on_reset = GetPrewarmOnReset();
  _wavenet->SetPrewarmOnReset(false);
  try
  {
    _wavenet->Reset(sampleRate, maxBufferSize);
  }
  catch (...)
  {
    _wavenet->SetPrewarmOnReset(prewarm_on_reset);
    throw;
  }
  _wavenet->SetPrewarmOnReset(prewarm_on_reset);
  // Settle before DSP::Reset(), which prewarms through process(): a reset is a stream
  // restart, so the committed controls apply immediately rather than being ramped into.
  // This is also what keeps a model load from gliding -- the host resets the newly loaded
  // model, and two models need not share a parameter set at all.
  _conditioner.Configure(sampleRate);
  DSP::Reset(sampleRate, maxBufferSize);
}

void ConcatWaveNet::SetPrewarmOnReset(const bool prewarmOnReset)
{
  DSP::SetPrewarmOnReset(prewarmOnReset);
  _wavenet->SetPrewarmOnReset(prewarmOnReset);
}

int ConcatWaveNet::GetPrewarmSamples()
{
  return _wavenet->GetPrewarmSamples();
}

void ConcatWaveNet::SetMaxBufferSize(const int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);
  _conditioner.SetMaxBufferSize(maxBufferSize);
}

#ifndef NDEBUG
void ConcatWaveNet::_debug_enter_param_api_()
{
  assert(!_debug_param_api_active.test_and_set(std::memory_order_acquire));
}

void ConcatWaveNet::_debug_leave_param_api_()
{
  _debug_param_api_active.clear(std::memory_order_release);
}
#endif

std::unique_ptr<DSP> ConcatWaveNetConfig::create(std::vector<float> weights, const double sampleRate)
{
  if (inner.condition_dsp != nullptr)
    throw std::runtime_error("ConcatWaveNet does not support condition_dsp");
  auto wavenet = std::make_unique<WaveNet>(inner.in_channels, inner.layer_array_params, inner.head_scale,
                                           inner.with_head, std::move(inner.head_params), std::move(weights),
                                           std::move(inner.condition_dsp), sampleRate);
  return std::make_unique<ConcatWaveNet>(std::move(wavenet), std::move(params), sampleRate);
}

std::unique_ptr<ModelConfig> create_concat_wavenet_config(const nlohmann::json& config, const double sampleRate)
{
  if (config.contains("condition_dsp"))
    throw std::runtime_error("ConcatWaveNet does not support condition_dsp");
  if (!config.contains("layers") || !config.at("layers").is_array() || config.at("layers").empty())
    throw std::runtime_error("ConcatWaveNet config must define at least one layer array");

  auto result = std::make_unique<ConcatWaveNetConfig>();
  result->params = nam::parse_param_specs(config, "ConcatWaveNet");
  const auto inner_channels = 1 + encoded_param_dim(result->params);
  auto derived_config = config;
  derived_config["in_channels"] = inner_channels;
  for (size_t i = 0; i < derived_config.at("layers").size(); ++i)
  {
    auto& layer = derived_config.at("layers").at(i);
    if (layer.contains("packing") && !layer.at("packing").is_null())
      throw std::runtime_error("ConcatWaveNet does not support packed layer arrays");
    if (layer.contains("slimmable") && !layer.at("slimmable").is_null())
      throw std::runtime_error("ConcatWaveNet does not support slimmable inner WaveNets");
    if (i == 0)
    {
      if (layer.contains("input_size") && layer.at("input_size").get<int>() != inner_channels)
        throw std::runtime_error("ConcatWaveNet first layer input_size must equal 1 + encoded param channels");
      layer["input_size"] = inner_channels;
    }
    if (layer.contains("condition_size") && layer.at("condition_size").get<int>() != inner_channels)
      throw std::runtime_error("ConcatWaveNet layer condition_size must equal 1 + encoded param channels");
    layer["condition_size"] = inner_channels;
  }
  result->inner = parse_config_json(derived_config, sampleRate);
  return result;
}

} // namespace wavenet
} // namespace nam

namespace
{
static nam::ConfigParserHelper _register_ConcatWaveNet("ConcatWaveNet", nam::wavenet::create_concat_wavenet_config);
static nam::ParametricArchitectureHelper _register_parametric_ConcatWaveNet("ConcatWaveNet");
}
