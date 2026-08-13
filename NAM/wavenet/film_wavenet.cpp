#include "film_wavenet.h"

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

// ParamEncoder ================================================================

ParamEncoder::ParamEncoder(const int in_features, const std::vector<int>& hidden_sizes, const int out_features,
                           const std::string& activation_name)
: _weight_count(0)
{
  // Resolved (and validated) unconditionally, even though it's only applied between layers --
  // a config that names an unsupported activation should fail regardless of hidden_sizes.
  _activation = activations::Activation::get_activation(activation_name);
  if (_activation == nullptr)
    throw std::runtime_error("FiLMWaveNet param_encoder: unsupported activation '" + activation_name + "'");

  std::vector<int> sizes{in_features};
  sizes.insert(sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
  sizes.push_back(out_features);
  for (size_t i = 0; i + 1 < sizes.size(); ++i)
  {
    _layers.emplace_back(sizes[i], sizes[i + 1], /*bias=*/true, /*groups=*/1);
    _layers.back().SetMaxBufferSize(1);
    _weight_count += sizes[i + 1] * sizes[i] + sizes[i + 1];
  }
}

void ParamEncoder::set_weights_(std::vector<float>::iterator& weights)
{
  for (auto& layer : _layers)
    layer.set_weights_(weights);
}

const Eigen::MatrixXf& ParamEncoder::Process(const Eigen::Ref<const Eigen::MatrixXf>& input)
{
  _layers[0].process_(input, 1);
  for (size_t i = 0; i + 1 < _layers.size(); ++i)
  {
    _activation->apply(_layers[i].GetOutput().data(), _layers[i].GetOutput().rows());
    _layers[i + 1].process_(_layers[i].GetOutput(), 1);
  }
  return _layers.back().GetOutput();
}

// FiLMWaveNet ==================================================================

FiLMWaveNet::FiLMWaveNet(std::unique_ptr<WaveNet> wavenet, std::vector<ParamSpec> param_specs,
                         std::unique_ptr<ParamEncoder> param_encoder, const double sample_rate)
: DSP(1, wavenet == nullptr ? 1 : wavenet->NumOutputChannels(), sample_rate)
, _wavenet(std::move(wavenet))
, _param_specs(std::move(param_specs))
, _param_encoder(std::move(param_encoder))
, _params(_param_specs.size())
{
  if (_wavenet == nullptr)
    throw std::invalid_argument("FiLMWaveNet: inner WaveNet must not be null");
  if (_param_specs.empty())
    throw std::invalid_argument("FiLMWaveNet: param_specs must contain at least one parameter");
  if (_wavenet->NumInputChannels() != 1)
    throw std::invalid_argument("FiLMWaveNet: inner WaveNet must have exactly 1 input channel");

  const auto encoded_dim = encoded_param_dim(_param_specs);
  _encoded_params.resize(encoded_dim, 1);
  const auto condition_dim = _param_encoder != nullptr ? _param_encoder->out_features() : encoded_dim;
  _condition.resize(condition_dim, 1);

  for (size_t i = 0; i < _param_specs.size(); ++i)
    _params[i] = _param_specs[i].defaultValue;
  _encode_params();
  _update_condition();
}

void FiLMWaveNet::SetParams(const std::span<const float> params)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    if (params.size() != _params.size())
      throw std::invalid_argument("FiLMWaveNet::SetParams: expected " + std::to_string(_params.size()) + " params, got "
                                  + std::to_string(params.size()));
    for (size_t i = 0; i < _param_specs.size(); ++i)
    {
      const auto& spec = _param_specs[i];
      if (spec.type != "switch")
        continue;
      if (!is_int_like(params[i]) || params[i] < 0.0f || params[i] >= static_cast<float>(spec.num_inputs()))
        throw std::invalid_argument("FiLMWaveNet switch parameter '" + spec.name
                                    + "' must be an integer index within [0, " + std::to_string(spec.num_inputs() - 1)
                                    + "]");
    }
    std::copy(params.begin(), params.end(), _params.begin());
    _encode_params();
    _update_condition();
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

std::span<const float> FiLMWaveNet::GetParams() const
{
#ifndef NDEBUG
  const_cast<FiLMWaveNet*>(this)->_debug_enter_param_api_();
#endif
  const auto result = std::span<const float>(_params);
#ifndef NDEBUG
  const_cast<FiLMWaveNet*>(this)->_debug_leave_param_api_();
#endif
  return result;
}

int FiLMWaveNet::ParamDim() const
{
  return static_cast<int>(_params.size());
}

const std::vector<ParamSpec>& FiLMWaveNet::GetParamSpecs() const
{
  return _param_specs;
}

void FiLMWaveNet::_encode_params()
{
  size_t encoded_index = 0;
  for (size_t i = 0; i < _param_specs.size(); ++i)
  {
    const auto& spec = _param_specs[i];
    if (spec.type == "switch")
    {
      for (int k = 0; k < spec.num_inputs(); ++k)
        _encoded_params(static_cast<Eigen::Index>(encoded_index + static_cast<size_t>(k)), 0) = 0.0f;
      _encoded_params(static_cast<Eigen::Index>(encoded_index + static_cast<size_t>(_params[i])), 0) = 1.0f;
      encoded_index += static_cast<size_t>(spec.num_inputs());
    }
    else
    {
      const auto fraction = (_params[i] - spec.min) / (spec.max - spec.min);
      _encoded_params(static_cast<Eigen::Index>(encoded_index++), 0) = -1.0f + 2.0f * fraction;
    }
  }
}

void FiLMWaveNet::_update_condition()
{
  if (_param_encoder != nullptr)
    _condition = _param_encoder->Process(_encoded_params);
  else
    _condition = _encoded_params;
}

void FiLMWaveNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    assert(num_frames <= mMaxBufferSize);
    // Refreshed once per block, not once per frame: every FiLM in the inner WaveNet caches this
    // column and reuses it for every frame in the call (see FiLM::SetControlCondition /
    // ProcessCached). Do NOT move this into SetParams() -- SetMaxBufferSize() resizes the buffer
    // holding the cached column, so a host block-size change would leave every FiLM reading stale
    // memory until the next SetParams(). Refreshing per block is what makes that unobservable, and
    // it costs one 1x1 over a single column per FiLM.
    _wavenet->SetParamCondition(_condition);
    _wavenet->process(input, output, num_frames);
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

void FiLMWaveNet::Reset(const double sampleRate, const int maxBufferSize)
{
  // The inner net must not prewarm itself: its prewarm would drive WaveNet::process() directly,
  // with no SetParamCondition() call ahead of it, so every FiLM would read an unset cache. Suppress
  // it here and let DSP::Reset() below prewarm through FiLMWaveNet::process(), which sets the
  // condition first.
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

void FiLMWaveNet::SetPrewarmOnReset(const bool prewarmOnReset)
{
  DSP::SetPrewarmOnReset(prewarmOnReset);
  _wavenet->SetPrewarmOnReset(prewarmOnReset);
}

int FiLMWaveNet::GetPrewarmSamples()
{
  return _wavenet->GetPrewarmSamples();
}

void FiLMWaveNet::SetMaxBufferSize(const int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);
  // _wavenet's own buffers are sized by its Reset() (called from FiLMWaveNet::Reset() above); the
  // FiLM condition is always a single column, independent of the host's block size.
}

#ifndef NDEBUG
void FiLMWaveNet::_debug_enter_param_api_()
{
  assert(!_debug_param_api_active.test_and_set(std::memory_order_acquire));
}

void FiLMWaveNet::_debug_leave_param_api_()
{
  _debug_param_api_active.clear(std::memory_order_release);
}
#endif

// FiLMWaveNetConfig =============================================================

std::unique_ptr<DSP> FiLMWaveNetConfig::create(std::vector<float> weights, const double sampleRate)
{
  if (inner.condition_dsp != nullptr)
    throw std::runtime_error("FiLMWaveNet does not support condition_dsp");

  std::unique_ptr<ParamEncoder> encoder;
  if (param_encoder.has_value())
  {
    const auto& spec = *param_encoder;
    encoder = std::make_unique<ParamEncoder>(spec.in_features, spec.hidden_sizes, spec.out_features, spec.activation);
    if (static_cast<size_t>(encoder->weight_count()) > weights.size())
      throw std::runtime_error("FiLMWaveNet: weight blob too short for param_encoder");
    auto it = weights.begin();
    encoder->set_weights_(it);
    weights.erase(weights.begin(), it); // Remaining weights are the inner WaveNet's blob
  }

  auto wavenet = std::make_unique<WaveNet>(inner.in_channels, inner.layer_array_params, inner.head_scale,
                                           inner.with_head, std::move(inner.head_params), std::move(weights),
                                           std::move(inner.condition_dsp), sampleRate);
  return std::make_unique<FiLMWaveNet>(std::move(wavenet), std::move(params), std::move(encoder), sampleRate);
}

std::unique_ptr<ModelConfig> create_film_wavenet_config(const nlohmann::json& config, const double sampleRate)
{
  if (config.contains("condition_dsp"))
    throw std::runtime_error("FiLMWaveNet does not support condition_dsp");
  if (!config.contains("layers") || !config.at("layers").is_array() || config.at("layers").empty())
    throw std::runtime_error("FiLMWaveNet config must define at least one layer array");

  auto result = std::make_unique<FiLMWaveNetConfig>();
  result->params = nam::parse_param_specs(config, "FiLMWaveNet");
  const auto encoded_dim = encoded_param_dim(result->params);

  int film_condition_size = encoded_dim;
  if (config.contains("param_encoder") && !config.at("param_encoder").is_null())
  {
    const auto& encoder_json = config.at("param_encoder");
    ParamEncoderSpec spec;
    spec.in_features = encoded_dim;
    spec.hidden_sizes = encoder_json.contains("hidden_sizes") ? encoder_json.at("hidden_sizes").get<std::vector<int>>()
                                                              : std::vector<int>{};
    spec.out_features =
      encoder_json.contains("out_features") ? encoder_json.at("out_features").get<int>() : encoded_dim;
    spec.activation =
      encoder_json.contains("activation") ? encoder_json.at("activation").get<std::string>() : std::string("ReLU");
    film_condition_size = spec.out_features;
    result->param_encoder = std::move(spec);
  }

  auto derived_config = config;
  derived_config["in_channels"] = 1;
  for (size_t i = 0; i < derived_config.at("layers").size(); ++i)
  {
    auto& layer = derived_config.at("layers").at(i);
    if (layer.contains("packing") && !layer.at("packing").is_null())
      throw std::runtime_error("FiLMWaveNet does not support packed layer arrays");
    if (layer.contains("slimmable") && !layer.at("slimmable").is_null())
      throw std::runtime_error("FiLMWaveNet does not support slimmable inner WaveNets");
    if (i == 0)
    {
      if (layer.contains("input_size") && layer.at("input_size").get<int>() != 1)
        throw std::runtime_error("FiLMWaveNet first layer input_size must be 1 (audio only)");
      layer["input_size"] = 1;
    }
    if (layer.contains("condition_size") && layer.at("condition_size").get<int>() != 1)
      throw std::runtime_error("FiLMWaveNet layer condition_size must be 1 (audio only)");
    layer["condition_size"] = 1;
    if (layer.contains("film_condition_size") && !layer.at("film_condition_size").is_null()
        && layer.at("film_condition_size").get<int>() != film_condition_size)
      throw std::runtime_error("FiLMWaveNet layer film_condition_size must equal the control condition width");
    layer["film_condition_size"] = film_condition_size;
  }
  result->inner = parse_config_json(derived_config, sampleRate);

  // The controls reach the network only through FiLM, so a config with no active site loads a
  // model whose knobs do nothing at all -- silently. Python rejects this at init; match it here
  // rather than shipping a plausible-sounding model with dead controls.
  auto any_film_active = false;
  for (const auto& layer : result->inner.layer_array_params)
  {
    any_film_active |= layer.conv_pre_film_params.active;
    any_film_active |= layer.conv_post_film_params.active;
    any_film_active |= layer.input_mixin_pre_film_params.active;
    any_film_active |= layer.input_mixin_post_film_params.active;
    any_film_active |= layer.activation_pre_film_params.active;
    any_film_active |= layer.activation_post_film_params.active;
    any_film_active |= (layer._layer1x1_post_film_params.active && layer.layer1x1_params.active);
    any_film_active |= (layer.head1x1_post_film_params.active && layer.head1x1_params.active);
  }
  if (!any_film_active)
    throw std::runtime_error(
      "FiLMWaveNet requires at least one active FiLM site; the control vector "
      "has no other way into the network");

  return result;
}

} // namespace wavenet
} // namespace nam

namespace
{
static nam::ConfigParserHelper _register_FiLMWaveNet("FiLMWaveNet", nam::wavenet::create_film_wavenet_config);
static nam::ParametricArchitectureHelper _register_parametric_FiLMWaveNet("FiLMWaveNet");
} // namespace
