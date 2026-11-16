#pragma once

#include <atomic>
#include <memory>
#include <span>
#include <vector>

#include "../debug_param_api_guard.h"
#include "../dsp.h"
#include "../hypernet.h"
#include "../model_config.h"
#include "../parametric_control.h"
#include "../slimmable.h"
#include "model.h"
#include "slimmable.h"

namespace nam
{
namespace wavenet
{

/// WaveNet whose weights are regenerated from the controls by a hypernetwork.
///
/// The audio backend is a SlimmableWavenet: conditioning always runs against the full
/// weight blob (delta_map export offsets), then the result is pushed into the active
/// (possibly channel-slimmed) WaveNet via UpdateFullWeights.
///
/// This model does NOT smooth (see IParametricControl). Its controls reach the audio only by
/// regenerating weights, and weights can only change between process() calls, so its transfer
/// function is quantised to the block grid no matter what. Ramping does not remove that
/// discontinuity, only relocate it -- from the host's control-commit rate up to the block
/// rate, where it is a clearly audible tone -- while regenerating the whole weight set every
/// block. It was implemented, measured and removed; do not add it back without measuring both
/// the spectrum and the cost. test_hyperwavenet_applies_immediately() carries the numbers.
class HyperWaveNet : public DSP, public IParametricControl, public SlimmableModel
{
public:
  // `condition_dsp` remains rejected by the loader: delta_map export_offset values are
  // defined only against the top-level WaveNet weight blob today. SlimmableWavenet can
  // rebuild a condition_dsp from JSON, but HyperWaveNet does not yet own that path.
  HyperWaveNet(std::unique_ptr<slimmable_wavenet::SlimmableWavenet> backend, std::vector<ParamSpec> param_specs,
               const nam::HypernetSpec& hypernet_spec, std::vector<float> base_weights,
               std::span<const float> hypernet_state, double sample_rate);

  void SetParams(std::span<const float> params) override;
  std::span<const float> GetParams() const override;
  int ParamDim() const override;
  const std::vector<ParamSpec>& GetParamSpecs() const override;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;
  void prewarm() override;
  void Reset(const double sampleRate, const int maxBufferSize) override;
  void SetPrewarmOnReset(const bool prewarmOnReset) override;

  void SetSlimmableSize(const double val) override;
  std::vector<double> GetSlimmableSizeBreakpoints() const override;

protected:
  int GetPrewarmSamples() override { return 0; }

private:
  std::unique_ptr<slimmable_wavenet::SlimmableWavenet> _backend;
  std::vector<ParamSpec> _param_specs;
  std::vector<float> _params;
  std::vector<float> _base_weights;
  std::vector<float> _conditioned;
  nam::Hypernetwork _hypernet;
  bool _dirty = true;
  int _param_dim = 0;
  mutable std::atomic_flag _debug_param_api_active = ATOMIC_FLAG_INIT;
};

struct HyperWaveNetConfig : public ModelConfig
{
  // This keeps the full parsed WaveNet config so adding nested condition_dsp support
  // later is a loader concern rather than a DSP redesign. For now
  // create_hyperwavenet_config() rejects condition_dsp explicitly.
  WaveNetConfig inner;
  std::vector<ParamSpec> params;
  nam::HypernetSpec hypernet;
  /// Per layer-array allowed channel counts for the SlimmableWavenet backend.
  /// Missing/null slimmable config becomes {channels} (full-size only).
  std::vector<std::vector<int>> per_array_allowed_channels;

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
