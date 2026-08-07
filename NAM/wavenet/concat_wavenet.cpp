#include "concat_wavenet.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "../registry.h"
#include "../parametric_version.h"

namespace
{

bool is_int_like(const float value)
{
  return std::isfinite(value) && std::trunc(value) == value;
}

int encoded_param_dim(const std::vector<nam::ParamSpec>& params)
{
  int dim = 0;
  for (const auto& spec : params)
    dim += spec.num_inputs();
  return dim;
}

} // namespace

namespace nam
{
namespace wavenet
{

ConcatWaveNet::ConcatWaveNet(std::unique_ptr<WaveNet> wavenet, std::vector<ParamSpec> param_specs,
                             const double sample_rate)
: DSP(1, wavenet == nullptr ? 1 : wavenet->NumOutputChannels(), sample_rate)
, _wavenet(std::move(wavenet))
, _param_specs(std::move(param_specs))
, _params(_param_specs.size())
, _encoded_params(static_cast<size_t>(encoded_param_dim(_param_specs)))
, _input_buffers(_encoded_params.size() + 1)
, _input_ptrs(_encoded_params.size() + 1)
{
  if (_wavenet == nullptr)
    throw std::invalid_argument("ConcatWaveNet: inner WaveNet must not be null");
  if (_param_specs.empty())
    throw std::invalid_argument("ConcatWaveNet: param_specs must contain at least one parameter");
  if (_wavenet->NumInputChannels() != static_cast<int>(_input_buffers.size()))
    throw std::invalid_argument("ConcatWaveNet: inner WaveNet input channel count does not match encoded params");
  for (size_t i = 0; i < _param_specs.size(); ++i)
    _params[i] = _param_specs[i].defaultValue;
  _encode_params();
}

void ConcatWaveNet::SetParams(const std::span<const float> params)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    if (params.size() != _params.size())
      throw std::invalid_argument("ConcatWaveNet::SetParams: expected " + std::to_string(_params.size())
                                  + " params, got " + std::to_string(params.size()));
    for (size_t i = 0; i < _param_specs.size(); ++i)
    {
      const auto& spec = _param_specs[i];
      if (spec.type != "switch")
        continue;
      if (!is_int_like(params[i]) || params[i] < 0.0f || params[i] >= static_cast<float>(spec.num_inputs()))
        throw std::invalid_argument("ConcatWaveNet switch parameter '" + spec.name
                                    + "' must be an integer index within [0, " + std::to_string(spec.num_inputs() - 1)
                                    + "]");
    }
    std::copy(params.begin(), params.end(), _params.begin());
    _encode_params();
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
  const auto result = std::span<const float>(_params);
#ifndef NDEBUG
  const_cast<ConcatWaveNet*>(this)->_debug_leave_param_api_();
#endif
  return result;
}

int ConcatWaveNet::ParamDim() const
{
  return static_cast<int>(_params.size());
}

const std::vector<ParamSpec>& ConcatWaveNet::GetParamSpecs() const
{
  return _param_specs;
}

void ConcatWaveNet::_encode_params()
{
  size_t encoded_index = 0;
  for (size_t i = 0; i < _param_specs.size(); ++i)
  {
    const auto& spec = _param_specs[i];
    if (spec.type == "switch")
    {
      std::fill_n(_encoded_params.begin() + static_cast<std::ptrdiff_t>(encoded_index), spec.num_inputs(), 0.0f);
      _encoded_params[encoded_index + static_cast<size_t>(_params[i])] = 1.0f;
      encoded_index += static_cast<size_t>(spec.num_inputs());
    }
    else
    {
      const auto fraction = (_params[i] - spec.min) / (spec.max - spec.min);
      _encoded_params[encoded_index++] = -1.0f + 2.0f * fraction;
    }
  }
}

void ConcatWaveNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    assert(num_frames <= mMaxBufferSize);
    std::copy_n(input[0], num_frames, _input_buffers[0].begin());
    for (size_t ch = 0; ch < _encoded_params.size(); ++ch)
      std::fill_n(_input_buffers[ch + 1].begin(), num_frames, static_cast<NAM_SAMPLE>(_encoded_params[ch]));
    _wavenet->process(_input_ptrs.data(), output, num_frames);
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
  for (size_t ch = 0; ch < _input_buffers.size(); ++ch)
  {
    _input_buffers[ch].resize(maxBufferSize);
    _input_ptrs[ch] = _input_buffers[ch].data();
  }
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
