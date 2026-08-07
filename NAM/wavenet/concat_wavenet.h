#pragma once

#include <atomic>
#include <memory>
#include <span>
#include <vector>

#include "../model_config.h"
#include "../parametric_control.h"
#include "model.h"

namespace nam
{
namespace wavenet
{

/// WaveNet conditioned by concatenating encoded controls with the audio input.
///
/// The wrapped WaveNet has 1 + encoded_param_dim input channels, while this DSP
/// exposes the single audio channel expected by a NAM host. Continuous controls
/// are min-max encoded to [-1, 1]; switches are one-hot encoded.
class ConcatWaveNet : public DSP, public IParametricControl
{
public:
  ConcatWaveNet(std::unique_ptr<WaveNet> wavenet, std::vector<ParamSpec> param_specs, double sample_rate);

  void SetParams(std::span<const float> params) override;
  std::span<const float> GetParams() const override;
  int ParamDim() const override;
  const std::vector<ParamSpec>& GetParamSpecs() const override;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;
  void Reset(double sampleRate, int maxBufferSize) override;
  void SetPrewarmOnReset(bool prewarmOnReset) override;
  int GetPrewarmSamples() override;

protected:
  void SetMaxBufferSize(int maxBufferSize) override;

private:
  void _encode_params();
#ifndef NDEBUG
  void _debug_enter_param_api_();
  void _debug_leave_param_api_();
#endif

  std::unique_ptr<WaveNet> _wavenet;
  std::vector<ParamSpec> _param_specs;
  std::vector<float> _params;
  std::vector<float> _encoded_params;
  std::vector<std::vector<NAM_SAMPLE>> _input_buffers;
  std::vector<NAM_SAMPLE*> _input_ptrs;
#ifndef NDEBUG
  std::atomic_flag _debug_param_api_active = ATOMIC_FLAG_INIT;
#endif
};

struct ConcatWaveNetConfig : public ModelConfig
{
  WaveNetConfig inner;
  std::vector<ParamSpec> params;

  ConcatWaveNetConfig() = default;
  ConcatWaveNetConfig(ConcatWaveNetConfig&&) = default;
  ConcatWaveNetConfig& operator=(ConcatWaveNetConfig&&) = default;
  ConcatWaveNetConfig(const ConcatWaveNetConfig&) = delete;
  ConcatWaveNetConfig& operator=(const ConcatWaveNetConfig&) = delete;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_concat_wavenet_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
