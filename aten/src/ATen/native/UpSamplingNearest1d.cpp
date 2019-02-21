#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSampling.h>

namespace at {
namespace native {

template <typename scalar_t>
void upsampling_nearest1d_update_output(
    Tensor* input_,
    Tensor* output,
    int output_width) {
  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_width = input_.size(2);
  const float scale = (float)input_width / (float)output_width;

  upsampling_1d_shape_check(
      input_, 0, nbatch, channels, input_width, output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_width});
  output.zero_();

  scalar_t* idata = input.data<scalar_t>();
  scalar_t* odata = output.data<scalar_t>();

  channels = channels * nbatch;

  AT_ASSERT(input_width > 0 && output_width > 0);

  // special case: just copy
  if (input_width == output_width) {
    for (int w2 = 0; w2 < output_width; ++w2) {
      const int w1 = w2;
      const scalar_t* pos1 = &idata[w1];
      scalar_t* pos2 = &odata[w2];

      for (int c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += input_width;
        pos2 += output_width;
      }
    }
    // c10::raw::intrusive_ptr::decref(input);
    return;
  }

  for (int w2 = 0; w2 < output_width; ++w2) {
    const scalar_t src_x =
        nearest_neighbor_compute_source_index(scale, w2, input_width);
    const int w1 = src_x;
    const scalar_t* pos1 = &idata[w1];
    scalar_t* pos2 = &odata[w2];

    for (int c = 0; c < channels; ++c) {
      pos2[0] = pos1[0];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
  // c10::raw::intrusive_ptr::decref(input);
}

template <typename scalar_t>
void upsampling_nearest1d_update_grad_input(
    Tensor* grad_output_,
    Tensor* grad_input,
    int nbatch,
    int channels,
    int input_width,
    int output_width) {
  upsampling_1d_shape_check(
      grad_output_, 1, nbatch, channels, input_width, output_width);

  auto grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_width});
  grad_input.zero_();

  scalar_t* data1 = grad_input.data<scalar_t>();
  scalar_t* data2 = grad_output.data<scalar_t>();

  channels = channels * nbatch;

  const float scale = (float)input_width / (float)output_width;

  // special case: same-size matching grids
  if (input_width == output_width) {
    for (int w2 = 0; w2 < output_width; ++w2) {
      const int w1 = w2;
      scalar_t* pos1 = &data1[w1];
      const scalar_t* pos2 = &data2[w2];

      for (int c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += input_width;
        pos2 += output_width;
      }
    }
    // c10::raw::intrusive_ptr::decref(grad_output);
    return;
  }

  for (int w2 = 0; w2 < output_width; ++w2) {
    const int w1 =
        nearest_neighbor_compute_source_index(scale, w2, input_width);

    scalar_t* pos1 = &data1[w1];
    const scalar_t* pos2 = &data2[w2];

    for (int c = 0; c < channels; ++c) {
      pos1[0] += pos2[0];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
  // c10::raw::intrusive_ptr::decref(grad_output);
}

} // namespace native
} // namespace at
