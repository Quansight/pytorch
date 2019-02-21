// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSampling.h>

namespace at {
namespace native {

template <typename scalar_t>
void upsampling_linear1d_update_output(
    Tensor& input_,
    Tensor& output,
    int64_t output_width,
    bool align_corners){
  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_width = input_.size(2);

  upsampling_1d_shape_check(
      input_,
      static_cast<int64_t>(0),
      nbatch,
      channels,
      input_width,
      output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_width});
  output.zero_();

  scalar_t* idata = input.data<scalar_t>();
  scalar_t* odata = output.data<scalar_t>();

  channels = channels * nbatch;

  AT_ASSERT(input_width > 0 && output_width > 0);

  // special case: just copy
  if (input_width == output_width) {
    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const int64_t w1 = w2;
      const scalar_t* pos1 = &idata[w1];
      scalar_t* pos2 = &odata[w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += input_width;
        pos2 += output_width;
      }
    }
    return;
  }
  const scalar_t rwidth = linear_upsampling_compute_scale<scalar_t>(
      input_width, output_width, align_corners);

  for (int64_t w2 = 0; w2 < output_width; ++w2) {
    const scalar_t w1r = linear_upsampling_compute_source_index<scalar_t>(
        rwidth, w2, align_corners);

    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;
    const scalar_t w1lambda = w1r - w1;
    const scalar_t w0lambda = static_cast<scalar_t>(1.) - w1lambda;
    const scalar_t* pos1 = &idata[w1];
    // index w2 is interpolated by idata[w1] and (itself or idata[w1 + 1])
    scalar_t* pos2 = &odata[w2];

    for (int64_t c = 0; c < channels; ++c) {
      pos2[0] = w0lambda * pos1[0] + w1lambda * pos1[w1p];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
}

template <typename scalar_t>
void upsampling_linear1d_update_grad_input(
    Tensor& grad_output_,
    Tensor& grad_input,
    int64_t nbatch,
    int64_t channels,
    int64_t input_width,
    int64_t output_width,
    bool align_corners) {
  upsampling_1d_shape_check(
      grad_output_,
      static_cast<int64_t>(1),
      nbatch,
      channels,
      input_width,
      output_width);

  auto grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_width});
  grad_input.zero_();

  scalar_t* data1 = grad_input.data<scalar_t>();
  scalar_t* data2 = grad_output.data<scalar_t>();

  channels = nbatch * channels;

  // special case: same-size matching grids
  if (input_width == output_width) {
    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const int64_t w1 = w2;
      scalar_t* pos1 = &data1[w1];
      const scalar_t* pos2 = &data2[w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += input_width;
        pos2 += output_width;
      }
    }
    return;
  }
  const scalar_t rwidth = linear_upsampling_compute_scale<scalar_t>(
      input_width, output_width, align_corners);

  for (int64_t w2 = 0; w2 < output_width; ++w2) {
    const scalar_t w1r = linear_upsampling_compute_source_index<scalar_t>(
        rwidth, w2, align_corners);

    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;
    const scalar_t w1lambda = w1r - w1;
    const scalar_t w0lambda = static_cast<scalar_t>(1.) - w1lambda;
    scalar_t* pos1 = &data1[w1];
    const scalar_t* pos2 = &data2[w2];

    for (int64_t c = 0; c < channels; ++c) {
      pos1[0] += w0lambda * pos2[0];
      pos1[w1p] += w1lambda * pos2[0];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
}

} // namespace native
} // namespace at
