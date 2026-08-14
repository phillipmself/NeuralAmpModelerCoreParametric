// Tests for FiLM

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "NAM/film.h"

namespace test_film
{
void test_set_max_buffer_size()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 128;
  film.SetMaxBufferSize(maxBufferSize);

  const auto out = film.GetOutput();
  assert(out.rows() == input_dim);
  assert(out.cols() == maxBufferSize);
}

void test_process_bias_only()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Configure the internal Conv1x1 (condition_dim -> 2*input_dim) to have:
  // - all-zero weights
  // - fixed biases so that scale/shift are constants
  //
  // Layout for Conv1x1 weights when groups=1:
  // - matrix weights: (2*input_dim * condition_dim)
  // - bias: (2*input_dim)
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  // biases: [scale(0..input_dim-1), shift(0..input_dim-1)]
  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;
  const float shift0 = 10.0f;
  const float shift1 = -20.0f;
  const float shift2 = 3.0f;

  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;
  weights[bias_offset + 3] = shift0;
  weights[bias_offset + 4] = shift1;
  weights[bias_offset + 5] = shift2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  // Make each channel distinct, and vary over frames.
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  film.Process(input, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  // Expected: output = input * scale + shift (elementwise)
  const float scales[3] = {scale0, scale1, scale2};
  const float shifts[3] = {shift0, shift1, shift2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input(c, t) * scales[c] + shifts[c];
      assert(std::abs(out(c, t) - expected) < 1e-6f);
    }
  }
}

void test_process_scale_only()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/false);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Internal Conv1x1 is (condition_dim -> input_dim) in scale-only mode.
  // We'll use all-zero weights and biases that define the scale.
  std::vector<float> weights;
  weights.resize(input_dim * condition_dim + input_dim, 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;

  const int bias_offset = input_dim * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  film.Process(input, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  const float scales[3] = {scale0, scale1, scale2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input(c, t) * scales[c];
      assert(std::abs(out(c, t) - expected) < 1e-6f);
    }
  }
}

void test_process_inplace_with_shift()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Configure the internal Conv1x1 with zero weights and fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;
  const float shift0 = 10.0f;
  const float shift1 = -20.0f;
  const float shift2 = 3.0f;

  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;
  weights[bias_offset + 3] = shift0;
  weights[bias_offset + 4] = shift1;
  weights[bias_offset + 5] = shift2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  // Keep a copy of the original input for comparison
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  // Test in-place processing
  film.Process_(input, condition, num_frames);

  // Verify that input was modified in-place
  // Expected: output = input * scale + shift (elementwise)
  const float scales[3] = {scale0, scale1, scale2};
  const float shifts[3] = {shift0, shift1, shift2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input_original(c, t) * scales[c] + shifts[c];
      assert(std::abs(input(c, t) - expected) < 1e-6f);
    }
  }

  // Verify that Process_ produces the same result as Process
  Eigen::MatrixXf input_for_process = input_original;
  film.Process(input_for_process, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      assert(std::abs(input(c, t) - out(c, t)) < 1e-6f);
    }
  }
}

void test_process_inplace_scale_only()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/false);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Internal Conv1x1 is (condition_dim -> input_dim) in scale-only mode.
  std::vector<float> weights;
  weights.resize(input_dim * condition_dim + input_dim, 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;

  const int bias_offset = input_dim * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  // Keep a copy of the original input for comparison
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  // Test in-place processing
  film.Process_(input, condition, num_frames);

  // Verify that input was modified in-place
  const float scales[3] = {scale0, scale1, scale2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input_original(c, t) * scales[c];
      assert(std::abs(input(c, t) - expected) < 1e-6f);
    }
  }

  // Verify that Process_ produces the same result as Process
  Eigen::MatrixXf input_for_process = input_original;
  film.Process(input_for_process, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      assert(std::abs(input(c, t) - out(c, t)) < 1e-6f);
    }
  }
}

void test_process_inplace_partial_frames()
{
  // Test that only the first num_frames columns are modified when input has more columns
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Configure the internal Conv1x1 with zero weights and fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;
  const float shift0 = 10.0f;
  const float shift1 = -20.0f;
  const float shift2 = 3.0f;

  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;
  weights[bias_offset + 3] = shift0;
  weights[bias_offset + 4] = shift1;
  weights[bias_offset + 5] = shift2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  // Create input with more columns than num_frames
  const int total_cols = 8;
  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, total_cols);
  input << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, //
    0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f;

  // Keep a copy of the original input
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  // Test in-place processing with only num_frames frames
  film.Process_(input, condition, num_frames);

  // Verify that only the first num_frames columns were modified
  const float scales[3] = {scale0, scale1, scale2};
  const float shifts[3] = {shift0, shift1, shift2};
  for (int c = 0; c < input_dim; c++)
  {
    // First num_frames should be modified
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input_original(c, t) * scales[c] + shifts[c];
      assert(std::abs(input(c, t) - expected) < 1e-6f);
    }
    // Remaining columns should be unchanged
    for (int t = num_frames; t < total_cols; t++)
    {
      assert(std::abs(input(c, t) - input_original(c, t)) < 1e-6f);
    }
  }
}

void test_process_with_groups()
{
  // Test FiLM with groups parameter
  // Using groups=2, condition_dim=4, input_dim=4
  // Internal Conv1x1: condition_dim=4 -> 2*input_dim=8 (with shift)
  // With groups=2:
  // - Group 0: processes condition[0:1] -> output[0:3] (out_per_group = 8/2 = 4)
  // - Group 1: processes condition[2:3] -> output[4:7]
  // Output layout: [scale[0:3], shift[0:3]]
  // So Group 0 produces: scale[0:1] and shift[0:1] (first 2 of each)
  //    Group 1 produces: scale[2:3] and shift[2:3] (last 2 of each)
  const int condition_dim = 4;
  const int input_dim = 4;
  const int groups = 2;
  const bool shift = true;
  nam::FiLM film(condition_dim, input_dim, shift, groups);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Weight layout for grouped Conv1x1:
  // For groups=2, out_channels=8, in_channels=4:
  // - out_per_group = 4, in_per_group = 2
  // - Group 0: weight[0:3, 0:1] = 4*2 = 8 weights
  // - Group 1: weight[4:7, 2:3] = 4*2 = 8 weights
  // - Bias: 8 values
  // Weight vector layout: [group0 weights (row-major), group1 weights (row-major), biases]
  std::vector<float> weights;
  weights.resize(8 * 2 + 8, 0.0f); // 2 groups * 8 weights + 8 biases

  // Set biases to create distinct scale/shift values
  // Output indices: [0:3] = scale[0:3], [4:7] = shift[0:3]
  const int bias_offset = 8 * 2;
  weights[bias_offset + 0] = 2.0f; // scale[0] (from group 0)
  weights[bias_offset + 1] = -1.0f; // scale[1] (from group 0)
  weights[bias_offset + 2] = 0.5f; // scale[2] (from group 1)
  weights[bias_offset + 3] = 3.0f; // scale[3] (from group 1)
  weights[bias_offset + 4] = 10.0f; // shift[0] (from group 0)
  weights[bias_offset + 5] = -20.0f; // shift[1] (from group 0)
  weights[bias_offset + 6] = 5.0f; // shift[2] (from group 1)
  weights[bias_offset + 7] = -15.0f; // shift[3] (from group 1)

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 3;
  Eigen::MatrixXf input(input_dim, num_frames);
  // Make each channel distinct
  input << 1.0f, 2.0f, 3.0f, //
    2.0f, 4.0f, 6.0f, //
    3.0f, 6.0f, 9.0f, //
    4.0f, 8.0f, 12.0f;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  film.Process(input, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  // Expected: output = input * scale + shift (elementwise)
  const float scales[4] = {2.0f, -1.0f, 0.5f, 3.0f};
  const float shifts[4] = {10.0f, -20.0f, 5.0f, -15.0f};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input(c, t) * scales[c] + shifts[c];
      assert(std::abs(out(c, t) - expected) < 1e-6f);
    }
  }
}

void test_process_with_groups_scale_only()
{
  // Test FiLM with groups parameter in scale-only mode
  const int condition_dim = 4;
  const int input_dim = 4;
  const int groups = 2;
  const bool shift = false;
  nam::FiLM film(condition_dim, input_dim, shift, groups);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Internal Conv1x1: condition_dim=4 -> input_dim=4 (scale-only)
  // With groups=2:
  // - Group 0: condition[0:1] -> scale[0:1]
  // - Group 1: condition[2:3] -> scale[2:3]
  // Each group processes 2 input channels -> 2 output channels
  // Weight layout: [group0 weights (2x2), group1 weights (2x2), biases (4)]

  std::vector<float> weights;
  weights.resize((2 * 2) * 2 + 4, 0.0f); // 2 groups * (2 outputs * 2 inputs) + 4 biases

  // Set biases to create distinct scale values per group
  const int bias_offset = (2 * 2) * 2;
  weights[bias_offset + 0] = 2.0f; // scale[0]
  weights[bias_offset + 1] = -1.0f; // scale[1]
  weights[bias_offset + 2] = 0.5f; // scale[2]
  weights[bias_offset + 3] = 3.0f; // scale[3]

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 2;
  Eigen::MatrixXf input(input_dim, num_frames);
  input << 1.0f, 2.0f, //
    2.0f, 4.0f, //
    3.0f, 6.0f, //
    4.0f, 8.0f;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  film.Process(input, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  // Expected: output = input * scale (elementwise)
  const float scales[4] = {2.0f, -1.0f, 0.5f, 3.0f};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input(c, t) * scales[c];
      assert(std::abs(out(c, t) - expected) < 1e-6f);
    }
  }
}
// A control ramp must arrive at its destination the same way it travelled: the step it
// takes on the sample it lands must be no bigger than the steps it has been taking all
// along. Deriving the position from a running sum breaks that -- the sum drifts, and
// correcting to the committed target on the final sample is a jump. Over a 1 s ramp that
// jump measured ~1.9e-4 against a per-sample increment of ~3.9e-5, i.e. five times the
// motion either side of it, which is a click exactly when a knob move finishes.
void test_long_ramp_lands_without_a_step()
{
  constexpr int kCondDim = 8;
  constexpr int kInputDim = 8;
  constexpr int kBlock = 64;
  constexpr int kRamp = 48000; // 1 s at 48 kHz, the shipped FiLMWaveNet default

  nam::FiLM film(kCondDim, kInputDim, /*shift=*/false, /*groups=*/1);
  film.SetMaxBufferSize(kBlock);
  std::vector<float> weights;
  weights.reserve(kCondDim * kInputDim + kInputDim);
  for (int i = 0; i < kCondDim * kInputDim; i++)
    weights.push_back(0.37f * std::sin(0.7f * (float)i) + 0.11f);
  for (int i = 0; i < kInputDim; i++)
    weights.push_back(0.5f + 0.01f * (float)i);
  auto it = weights.begin();
  film.set_weights_(it);

  const Eigen::MatrixXf from = Eigen::MatrixXf::Constant(kCondDim, 1, -0.9f);
  const Eigen::MatrixXf to = Eigen::MatrixXf::Constant(kCondDim, 1, 0.8f);
  const Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(kInputDim, kBlock);

  film.SetControlCondition(from, 0); // settle, so the ramp starts from a known place
  film.SetControlCondition(to, kRamp);

  std::vector<float> steps;
  steps.reserve(kRamp + kBlock);
  float previous = 0.0f;
  bool have_previous = false;
  float settled = 0.0f;
  for (int n = 0; n < kRamp + 4 * kBlock; n += kBlock)
  {
    film.ProcessCached(ones, kBlock);
    const Eigen::MatrixXf& out = film.GetOutput();
    for (int f = 0; f < kBlock; f++)
    {
      const float value = out(0, f);
      if (have_previous)
        steps.push_back(std::abs(value - previous));
      previous = value;
      have_previous = true;
    }
    settled = previous;
  }

  // The ramp must have finished, and finished exactly on the committed value.
  assert(!film.IsRamping());
  film.SetControlCondition(from, 0);
  film.SetControlCondition(to, 0); // same destination, landed immediately
  film.ProcessCached(ones, kBlock);
  const float exact = film.GetOutput()(0, 0);
  assert(std::abs(settled - exact) < 1.0e-7f);

  // No sample is a jump: the largest step over the whole gesture is within a hair of the
  // typical one. This is what fails if the ramp is accumulated and then snapped.
  std::vector<float> sorted = steps;
  std::sort(sorted.begin(), sorted.end());
  const float largest = sorted.back();
  // Median over the ramping portion; the settled tail contributes zeros at the front.
  const float typical = sorted[sorted.size() - steps.size() / 4];
  assert(typical > 0.0f);
  assert(largest < 1.5f * typical);
}

} // namespace test_film
