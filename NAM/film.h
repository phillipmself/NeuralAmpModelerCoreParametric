#pragma once

#include <algorithm>

#include <Eigen/Dense>
#include <cassert>
#include <vector>

#include "compiler.h"
#include "dsp.h"

namespace nam
{
/// \brief Feature-wise Linear Modulation (FiLM)
///
/// Given an input (input_dim x num_frames) and a condition (condition_dim x num_frames), compute:
///   scale, shift = Conv1x1(condition) split across channels (top/bottom half, respectively)
///   output = input * scale + shift  (elementwise)
///
/// FiLM applies per-channel scaling and optional shifting based on conditioning input,
/// allowing the model to adapt its behavior based on external signals.
class FiLM
{
public:
  /// \brief Constructor
  /// \param condition_dim Size of the conditioning input
  /// \param input_dim Size of the input to be modulated
  /// \param shift Whether to apply both scale and shift (true) or only scale (false)
  /// \param groups Number of groups for grouped convolution in the condition-to-scale-shift submodule (default: 1)
  FiLM(const int condition_dim, const int input_dim, const bool shift, const int groups = 1)
  : _cond_to_scale_shift(condition_dim, (shift ? 2 : 1) * input_dim, /*bias=*/true, groups)
  , _do_shift(shift)
  {
    // Sized here, never on the audio thread. These hold the cached-control path's state and are
    // independent of the block size, so unlike _output they survive SetMaxBufferSize().
    const int scale_shift_dim = (shift ? 2 : 1) * input_dim;
    _control_cached = Eigen::VectorXf::Zero(condition_dim);
    _scale_shift_current = Eigen::VectorXf::Zero(scale_shift_dim);
    _scale_shift_target = Eigen::VectorXf::Zero(scale_shift_dim);
    _scale_shift_delta = Eigen::VectorXf::Zero(scale_shift_dim);
  }

  /// \brief Get the entire internal output buffer
  ///
  /// This is intended for internal wiring between layers; callers should treat
  /// the buffer as pre-allocated storage and only consider the first num_frames columns
  /// valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the output buffer
  Eigen::MatrixXf& GetOutput() { return _output; }

  /// \brief Get the entire internal output buffer (const version)
  /// \return Const reference to the output buffer
  const Eigen::MatrixXf& GetOutput() const { return _output; }

  /// \brief Resize buffers to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize)
  {
    _cond_to_scale_shift.SetMaxBufferSize(maxBufferSize);
    _output.resize(get_input_dim(), maxBufferSize);
  }

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights) { _cond_to_scale_shift.set_weights_(weights); }

  /// \brief Get the condition dimension
  /// \return Size of the conditioning input
  long get_condition_dim() const { return _cond_to_scale_shift.get_in_channels(); }

  /// \brief Get the input dimension
  /// \return Size of the input to be modulated
  long get_input_dim() const
  {
    return _do_shift ? (_cond_to_scale_shift.get_out_channels() / 2) : _cond_to_scale_shift.get_out_channels();
  }

  /// \brief Process input with conditioning
  ///
  /// Writes (input_dim x num_frames) into internal output buffer; access via GetOutput().
  /// Uses Eigen::Ref to accept matrices and block expressions without creating temporaries (real-time safe).
  /// \param input Input matrix (input_dim x num_frames)
  /// \param condition Conditioning matrix (condition_dim x num_frames)
  /// \param num_frames Number of frames to process
  void Process(const Eigen::Ref<const Eigen::MatrixXf>& input, const Eigen::Ref<const Eigen::MatrixXf>& condition,
               const int num_frames)
  {
    assert(get_input_dim() == input.rows());
    assert(get_condition_dim() == condition.rows());
    assert(num_frames <= input.cols());
    assert(num_frames <= condition.cols());
    assert(num_frames <= _output.cols());

    _cond_to_scale_shift.process_(condition, num_frames);
    const auto& scale_shift = _cond_to_scale_shift.GetOutput();

#ifdef NAM_USE_INLINE_GEMM
    // Optimized inline FiLM operation
    const int input_dim = (int)get_input_dim();
    const float* NAM_RESTRICT input_ptr = input.data();
    const float* NAM_RESTRICT scale_shift_ptr = scale_shift.data();
    float* NAM_RESTRICT output_ptr = _output.data();
    const int scale_shift_rows = (int)scale_shift.rows();
    // Use outerStride() instead of rows() to correctly handle non-contiguous
    // block expressions (e.g. topRows()) where outerStride > rows
    const int input_stride = (int)input.outerStride();

    if (_do_shift)
    {
      // scale = top input_dim rows, shift = bottom input_dim rows
      if (input_dim == 3)
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          const float* NAM_RESTRICT shift_col = scale_col + 3;
          float* NAM_RESTRICT out_col = output_ptr + f * 3;
          out_col[0] = in_col[0] * scale_col[0] + shift_col[0];
          out_col[1] = in_col[1] * scale_col[1] + shift_col[1];
          out_col[2] = in_col[2] * scale_col[2] + shift_col[2];
        }
      }
      else
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          const float* NAM_RESTRICT shift_col = scale_col + input_dim;
          float* NAM_RESTRICT out_col = output_ptr + f * input_dim;

          int i = 0;
          for (; i + 3 < input_dim; i += 4)
          {
            out_col[i] = in_col[i] * scale_col[i] + shift_col[i];
            out_col[i + 1] = in_col[i + 1] * scale_col[i + 1] + shift_col[i + 1];
            out_col[i + 2] = in_col[i + 2] * scale_col[i + 2] + shift_col[i + 2];
            out_col[i + 3] = in_col[i + 3] * scale_col[i + 3] + shift_col[i + 3];
          }
          for (; i < input_dim; i++)
          {
            out_col[i] = in_col[i] * scale_col[i] + shift_col[i];
          }
        }
      }
    }
    else
    {
      // scale only
      if (input_dim == 3)
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          float* NAM_RESTRICT out_col = output_ptr + f * 3;
          out_col[0] = in_col[0] * scale_col[0];
          out_col[1] = in_col[1] * scale_col[1];
          out_col[2] = in_col[2] * scale_col[2];
        }
      }
      else
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          float* NAM_RESTRICT out_col = output_ptr + f * input_dim;

          int i = 0;
          for (; i + 3 < input_dim; i += 4)
          {
            out_col[i] = in_col[i] * scale_col[i];
            out_col[i + 1] = in_col[i + 1] * scale_col[i + 1];
            out_col[i + 2] = in_col[i + 2] * scale_col[i + 2];
            out_col[i + 3] = in_col[i + 3] * scale_col[i + 3];
          }
          for (; i < input_dim; i++)
          {
            out_col[i] = in_col[i] * scale_col[i];
          }
        }
      }
    }
#else
    const auto scale = scale_shift.topRows(get_input_dim()).leftCols(num_frames);
    if (_do_shift)
    {
      // scale = top input_dim, shift = bottom input_dim
      const auto shift = scale_shift.bottomRows(get_input_dim()).leftCols(num_frames);
      _output.leftCols(num_frames).array() = input.leftCols(num_frames).array() * scale.array() + shift.array();
    }
    else
    {
      _output.leftCols(num_frames).array() = input.leftCols(num_frames).array() * scale.array();
    }
#endif
  }

  /// \brief Process input with conditioning (in-place)
  ///
  /// Uses Eigen::Ref to accept matrices and block expressions without creating temporaries (real-time safe).
  /// Modifies the input matrix directly.
  /// \param input Input matrix (input_dim x num_frames), will be modified in-place
  /// \param condition Conditioning matrix (condition_dim x num_frames)
  /// \param num_frames Number of frames to process
  void Process_(Eigen::Ref<Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf>& condition,
                const int num_frames)
  {
    Process(input, condition, num_frames);
    input.leftCols(num_frames).noalias() = _output.leftCols(num_frames);
  }

  /// \brief Cache scale/shift from a condition that is constant across the block
  ///
  /// Runs the Conv1x1(condition) computation once, for a single-column control vector,
  /// instead of the once-per-``Process()``-call recomputation that a time-varying condition
  /// requires. ProcessCached()/ProcessCached_() then reuse the cached column for every frame.
  ///
  /// Idempotent: pushing an unchanged control is a no-op, so the natural place to call it is once
  /// per processing block, before the ProcessCached() calls that consume it. The cached scale/shift
  /// lives in members of this class rather than in a block-sized buffer, so it survives
  /// SetMaxBufferSize(); what must NOT happen is re-arming the ramp on an unchanged target every
  /// block, which would turn the linear ramp into a one-pole that never lands -- hence the
  /// early-out below rather than a caller-side "did it change?" check.
  /// \param control Control vector (condition_dim x 1)
  /// \param ramp_samples Samples to slew over in scale/shift space; 0 lands immediately
  void SetControlCondition(const Eigen::Ref<const Eigen::MatrixXf>& control, const int ramp_samples = 0)
  {
    assert(get_condition_dim() == control.rows());
    assert(control.cols() >= 1);
    // Callers push every block so the cache cannot go stale, but re-arming an unchanged target
    // every block would turn the linear ramp into a one-pole that never lands. Only an actual
    // change re-aims it; otherwise the ramp in flight keeps running.
    if (_have_control && (_control_cached.array() == control.col(0).array()).all())
      return;
    _control_cached = control.col(0);

    _cond_to_scale_shift.process_(control, 1);
    const Eigen::MatrixXf& computed = _cond_to_scale_shift.GetOutput();
    const int scale_shift_dim = (int)_scale_shift_target.size();
    for (int i = 0; i < scale_shift_dim; i++)
      _scale_shift_target[i] = computed(i, 0);

    // Ramping from an unset state would slew up from zero (silence, then a swell) on the first
    // block, so the first control of a stream always lands immediately -- as does an explicit
    // ramp_samples of 0, which is how a reset settles.
    if (!_have_control || ramp_samples <= 0)
    {
      _scale_shift_current = _scale_shift_target;
      _scale_shift_delta.setZero();
      _ramp_remaining = 0;
    }
    else
    {
      // Held as an offset from the destination rather than a per-sample increment, so the
      // ramp can be evaluated from its integer position instead of accumulated and cannot
      // drift away from the target over a long gesture. Re-arming mid-flight starts from
      // wherever the ramp currently is, which keeps the applied value continuous.
      _scale_shift_delta = _scale_shift_current - _scale_shift_target;
      _ramp_inv_total = 1.0f / (float)ramp_samples;
      _ramp_remaining = ramp_samples;
    }
    _have_control = true;
  }

  /// \brief Whether a control ramp is still in flight (test/diagnostic aid)
  bool IsRamping() const { return _ramp_remaining > 0; }

  /// \brief Apply the scale/shift cached by SetControlCondition(), broadcasting it across num_frames
  ///
  /// Writes (input_dim x num_frames) into internal output buffer; access via GetOutput().
  /// \param input Input matrix (input_dim x num_frames)
  /// \param num_frames Number of frames to process
  void ProcessCached(const Eigen::Ref<const Eigen::MatrixXf>& input, const int num_frames)
  {
    assert(get_input_dim() == input.rows());
    assert(num_frames <= input.cols());
    assert(num_frames <= _output.cols());

    const int input_dim = (int)get_input_dim();
    const float* NAM_RESTRICT input_ptr = input.data();
    const int input_stride = (int)input.outerStride();
    float* NAM_RESTRICT output_ptr = _output.data();

    // Settled: the committed scale/shift applies to the whole block, which is the common case and
    // is bit-identical to what this did before ramping existed.
    const int ramped = std::min(num_frames, _ramp_remaining);
    if (ramped > 0)
    {
      float* NAM_RESTRICT cur_ptr = _scale_shift_current.data();
      const float* NAM_RESTRICT tgt_ptr = _scale_shift_target.data();
      const float* NAM_RESTRICT delta_ptr = _scale_shift_delta.data();
      const int scale_shift_dim = (int)_scale_shift_current.size();
      for (int f = 0; f < ramped; f++)
      {
        // Position comes from the integer sample counter rather than a running sum. An
        // accumulated ramp lands wherever its rounding error has taken it, and the
        // correcting step to the committed target is then a jump: over a 1 s ramp that was
        // ~1.9e-4, larger than any step the ramp itself takes, and audible as a click at
        // the instant the knob's move completes.
        const float frac = (float)(_ramp_remaining - f) * _ramp_inv_total;
        for (int i = 0; i < scale_shift_dim; i++)
          cur_ptr[i] = tgt_ptr[i] + delta_ptr[i] * frac;
        const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
        float* NAM_RESTRICT out_col = output_ptr + f * input_dim;
        if (_do_shift)
        {
          const float* NAM_RESTRICT shift_ptr = cur_ptr + input_dim;
          for (int i = 0; i < input_dim; i++)
            out_col[i] = in_col[i] * cur_ptr[i] + shift_ptr[i];
        }
        else
        {
          for (int i = 0; i < input_dim; i++)
            out_col[i] = in_col[i] * cur_ptr[i];
        }
      }
      _ramp_remaining -= ramped;
      // Leave _scale_shift_current holding what the *next* sample will apply, so a re-arm
      // mid-flight is continuous. At _ramp_remaining == 0 the fraction is exactly zero, so
      // this is the committed target -- the ramp lands by construction, with nothing to
      // correct.
      const float frac = (float)_ramp_remaining * _ramp_inv_total;
      for (int i = 0; i < scale_shift_dim; i++)
        cur_ptr[i] = tgt_ptr[i] + delta_ptr[i] * frac;
    }

    const float* NAM_RESTRICT scale_ptr = _scale_shift_target.data();
    if (_do_shift)
    {
      const float* NAM_RESTRICT shift_ptr = scale_ptr + input_dim;
      for (int f = ramped; f < num_frames; f++)
      {
        const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
        float* NAM_RESTRICT out_col = output_ptr + f * input_dim;
        for (int i = 0; i < input_dim; i++)
          out_col[i] = in_col[i] * scale_ptr[i] + shift_ptr[i];
      }
    }
    else
    {
      for (int f = ramped; f < num_frames; f++)
      {
        const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
        float* NAM_RESTRICT out_col = output_ptr + f * input_dim;
        for (int i = 0; i < input_dim; i++)
          out_col[i] = in_col[i] * scale_ptr[i];
      }
    }
  }

  /// \brief Apply the cached scale/shift in-place (see ProcessCached())
  /// \param input Input matrix (input_dim x num_frames), will be modified in-place
  /// \param num_frames Number of frames to process
  void ProcessCached_(Eigen::Ref<Eigen::MatrixXf> input, const int num_frames)
  {
    ProcessCached(input, num_frames);
    input.leftCols(num_frames).noalias() = _output.leftCols(num_frames);
  }

private:
  Conv1x1 _cond_to_scale_shift; // condition_dim -> (shift ? 2 : 1) * input_dim
  // Cached-control state. Ramping happens here, in scale/shift space, rather than on the control
  // vector: the 1x1 still runs once per control change instead of once per frame, and the encoder
  // upstream is never fed an interpolated control -- which is what keeps a switch's one-hot exact
  // while its effect still arrives smoothly.
  Eigen::VectorXf _control_cached; // condition_dim; the control that produced _scale_shift_target
  Eigen::VectorXf _scale_shift_current; // (shift ? 2 : 1) * input_dim; what the block is applying
  Eigen::VectorXf _scale_shift_target; // committed destination
  Eigen::VectorXf _scale_shift_delta; // (value when armed) - target, scaled by the ramp position
  int _ramp_remaining = 0;
  float _ramp_inv_total = 0.0f;
  bool _have_control = false;
  Eigen::MatrixXf _output; // input_dim x maxBufferSize
  bool _do_shift;
};
} // namespace nam
