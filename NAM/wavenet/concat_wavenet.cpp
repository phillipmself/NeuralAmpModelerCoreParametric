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
} // namespace
