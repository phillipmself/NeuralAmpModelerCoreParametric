#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "../dsp.h"
#include "../model_config.h"
#include "../parametric_control.h"
#include "model.h"

namespace nam
{
namespace wavenet
{

/// Small MLP sitting between the encoded control vector and the FiLM condition it drives.
///
/// Mirrors the Python-side ``ParamEncoder``: a stack of ``nn.Linear`` layers with an activation
/// between (not after) each pair, sizes ``[in_features, *hidden_sizes, out_features]``. Reuses
/// Conv1x1 for each Linear, since a 1x1 convolution on a single-column input *is* a linear layer,
/// and its weight layout (out_channels x in_channels, row-major, then bias) already matches
/// ``nn.Linear``'s export order.
class ParamEncoder
{
public:
  ParamEncoder(int in_features, const std::vector<int>& hidden_sizes, int out_features,
               const std::string& activation_name);

  void set_weights_(std::vector<float>::iterator& weights);

  /// \param input Encoded controls (in_features x 1)
  /// \return Reference to the internal output buffer (out_features x 1)
  const Eigen::MatrixXf& Process(const Eigen::Ref<const Eigen::MatrixXf>& input);

  int out_features() const { return static_cast<int>(_layers.back().get_out_channels()); }
  int weight_count() const { return _weight_count; }

private:
  std::vector<Conv1x1> _layers;
  activations::Activation::Ptr _activation; // Applied between layers; unused (and unresolved) if none exist
  int _weight_count;
};

/// WaveNet conditioned by FiLM on the control vector alone -- see the module docstring in
/// nam/models/parametric/_film_wavenet.py (training repo) for the rationale.
///
/// The wrapped WaveNet is shape-identical to a stock one (1 input channel, condition_size 1);
/// every layer array with a FiLM module driven by controls carries film_condition_size, and its
/// FiLM sites read a control vector cached via WaveNet::SetParamCondition() instead of the
/// per-frame layer condition. Continuous controls are min-max encoded to [-1, 1]; switches are
/// one-hot encoded -- identical to ConcatWaveNet's control encoding.
class FiLMWaveNet : public DSP, public IParametricControl
{
public:
  FiLMWaveNet(std::unique_ptr<WaveNet> wavenet, std::vector<ParamSpec> param_specs,
              std::unique_ptr<ParamEncoder> param_encoder, double sample_rate);

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
  void _update_condition();
#ifndef NDEBUG
  void _debug_enter_param_api_();
  void _debug_leave_param_api_();
#endif

  std::unique_ptr<WaveNet> _wavenet;
  std::vector<ParamSpec> _param_specs;
  std::unique_ptr<ParamEncoder> _param_encoder; // May be null
  std::vector<float> _params;
  Eigen::MatrixXf _encoded_params; // encoded_param_dim x 1
  Eigen::MatrixXf _condition; // film_condition_size x 1, constant across a block until SetParams() changes it
#ifndef NDEBUG
  std::atomic_flag _debug_param_api_active = ATOMIC_FLAG_INIT;
#endif
};

/// Spec for the optional param_encoder block of a FiLMWaveNet config.
struct ParamEncoderSpec
{
  int in_features;
  std::vector<int> hidden_sizes;
  int out_features;
  std::string activation;
};

struct FiLMWaveNetConfig : public ModelConfig
{
  WaveNetConfig inner;
  std::vector<ParamSpec> params;
  std::optional<ParamEncoderSpec> param_encoder;

  FiLMWaveNetConfig() = default;
  FiLMWaveNetConfig(FiLMWaveNetConfig&&) = default;
  FiLMWaveNetConfig& operator=(FiLMWaveNetConfig&&) = default;
  FiLMWaveNetConfig(const FiLMWaveNetConfig&) = delete;
  FiLMWaveNetConfig& operator=(const FiLMWaveNetConfig&) = delete;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_film_wavenet_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
