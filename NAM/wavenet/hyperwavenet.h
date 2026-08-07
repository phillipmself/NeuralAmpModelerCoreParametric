#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "../hypernet.h"
#include "../model_config.h"
#include "../parametric_control.h"
#include "model.h"

namespace nam
{
namespace wavenet
{

class HyperWaveNet : public WaveNet, public IParametricControl
{
public:
  // `condition_dsp` remains in the constructor signature to mirror the WaveNet base
  // and keep the eventual nested-DSP extension localized to the config parser plus
  // weight-splitting path. The current loader rejects condition_dsp exports because
  // delta_map export_offset values are defined only against the top-level WaveNet
  // weight blob today.
  HyperWaveNet(int in_channels, const std::vector<LayerArrayParams>& layer_array_params, float head_scale,
               bool with_head, std::optional<HeadParams> head_params, std::unique_ptr<DSP> condition_dsp,
               std::vector<ParamSpec> param_specs, const nam::HypernetSpec& hypernet_spec,
               std::vector<float> base_weights, std::span<const float> hypernet_state, double sample_rate);

  void SetParams(std::span<const float> params) override;
  std::span<const float> GetParams() const override;
  int ParamDim() const override;
  const std::vector<ParamSpec>& GetParamSpecs() const override;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;

private:
#ifndef NDEBUG
  void _debug_enter_param_api_();
  void _debug_leave_param_api_();
#endif

  std::vector<ParamSpec> _param_specs;
  std::vector<float> _params;
  std::vector<float> _base_weights;
  std::vector<float> _conditioned;
  nam::Hypernetwork _hypernet;
  bool _dirty = true;
  int _param_dim = 0;
#ifndef NDEBUG
  std::atomic_flag _debug_param_api_active = ATOMIC_FLAG_INIT;
#endif
};

struct HyperWaveNetConfig : public ModelConfig
{
  // This keeps the full parsed WaveNet config so adding nested condition_dsp support
  // later is a loader concern rather than a DSP redesign. For now
  // create_hyperwavenet_config() rejects condition_dsp explicitly.
  WaveNetConfig inner;
  std::vector<ParamSpec> params;
  nam::HypernetSpec hypernet;

  HyperWaveNetConfig() = default;
  HyperWaveNetConfig(HyperWaveNetConfig&&) = default;
  HyperWaveNetConfig& operator=(HyperWaveNetConfig&&) = default;
  HyperWaveNetConfig(const HyperWaveNetConfig&) = delete;
  HyperWaveNetConfig& operator=(const HyperWaveNetConfig&) = delete;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_hyperwavenet_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
