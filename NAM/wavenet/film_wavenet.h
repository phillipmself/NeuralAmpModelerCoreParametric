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
/// \brief Default time a committed control takes to arrive at the FiLM sites.
///
/// FiLM applies the control as a per-channel gain, so a control that only updates at block
/// boundaries steps the gain once per block -- zipper noise, scaled by the signal and by the
/// block size. Ramping in scale/shift space makes that gain piecewise-linear instead, which
/// removes the discontinuity outright; the length then only bounds the slew rate.
///
/// Much shorter than ConcatWaveNet's kDefaultParamRampSeconds (200 ms) on purpose. Concat needs a
/// long ramp to cross its receptive-field-long window of control patterns it was never trained on
/// slowly; FiLM has no such window -- the controls are read by 1x1 convolutions with no time
/// extent, so every instant of the ramp is a configuration the model was trained at. The ramp here
/// only has to outrun the step.
///
/// 50 ms, measured on a trained SD1 capture (TONE 0->10 at DRIVE=10, -10 dBFS input) against a
/// quasi-static reference. Splitting the residual by band separates two things: content below
/// 200 Hz is tracking lag, which barely moves with ramp length and is not what a knob move sounds
/// like; content above 1 kHz is the click. The click band has a sharp knee here --
///
///     ramp     10 ms    25 ms    50 ms   100 ms   200 ms   500 ms
///     click   -18.4    -25.5    -30.2    -29.6    -31.2    -33.0   dB rel. signal
///
/// -- so on that measure 10 -> 50 ms is worth ~12 dB and everything past it looks like latency for
/// ~1 dB. An earlier 10 ms was reasoned from zipper-step arithmetic before there was a model to
/// measure at all.
///
/// The shipped value is nonetheless 1 s, chosen by ear over that analysis. The click band is not
/// the whole artifact: near silence the model emits a knob-dependent DC pedestal it generates from
/// its conv biases, and *that* term keeps falling with ramp length well past the click knee
/// (0.018 at 10 ms, 0.0051 at 50 ms, 0.0044 at 200 ms, 0.0022 at 500 ms, silent input at
/// DRIVE=10). It is also the term with no program material to mask it, which is why a ramp far
/// longer than the click analysis alone would justify audibly wins.
///
/// Known remaining, at this length: a pop when the ramp lands and when the knob reverses
/// mid-flight. Suspected cause is the exact-landing snap in FiLM::ProcessCached -- over ~48k
/// accumulation steps the ramped value can drift from the committed target, and snapping to it is
/// then a step. Not yet diagnosed.
///
/// None of this reaches the DC pedestal itself, which is a training-side fix (see the silence
/// anchors in the training repo). Tune with SetParamRampSeconds().
constexpr float kDefaultFiLMRampSeconds = 1.0f;

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

  /// \brief Time a committed control takes to arrive, in seconds (0 disables ramping).
  ///
  /// See kDefaultFiLMRampSeconds for what this trades off. Takes effect on the next control
  /// change; a ramp already in flight keeps its original length.
  void SetParamRampSeconds(float seconds);
  float GetParamRampSeconds() const { return _ramp_seconds; }
  void SetPrewarmOnReset(bool prewarmOnReset) override;
  int GetPrewarmSamples() override;

protected:
  void SetMaxBufferSize(int maxBufferSize) override;

private:
  void _encode_params();
  void _update_condition();
  void _update_ramp_samples();
#ifndef NDEBUG
  void _debug_enter_param_api_();
  void _debug_leave_param_api_();
#endif

  std::unique_ptr<WaveNet> _wavenet;
  std::vector<ParamSpec> _param_specs;
  std::unique_ptr<ParamEncoder> _param_encoder; // May be null
  std::vector<float> _params;
  float _ramp_seconds;
  int _ramp_samples = 0;
  bool _snap_next_condition = true; // A stream restart lands the control immediately.
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
