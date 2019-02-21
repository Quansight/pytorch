#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSampling.h>

namespace at {
namespace native {

template <typename scalar_t>
void upsampling_nearest2d_update_output(
    Tensor* input_,
    Tensor* output,
    int output_height,
    int output_width) {
  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);

  const float height_scale = (float)input_height / (float)output_height;
  const float width_scale = (float)input_width / (float)output_width;

  upsampling_2d_shape_check(
      input,
      0,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_height, output_width});
  output.zero_();

  scalar_t* idata = input.data<scalar_t>();
  scalar_t* odata = output.data<scalar_t>();

  channels = channels * nbatch;

  AT_ASSERT(input_width > 0 && output_width > 0);

  // special case: just copy
  if (input_height == output_height && input_width == output_width) {
    for (int h2 = 0; h2 < output_height; ++h2) {
      const int h1 = h2;

      for (int w2 = 0; w2 < output_width; ++w2) {
        const int w1 = w2;
        const scalar_t* pos1 = &idata[h1 * input_width + w1];
        scalar_t* pos2 = &odata[h2 * output_width + w2];

        for (int c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += input_height * input_width;
          pos2 += output_height * output_width;
        }
      }
    }
    // c10::raw::intrusive_ptr::decref(input);
    return;
  }

  for (int h2 = 0; h2 < output_height; ++h2) {
    const int h1 =
        nearest_neighbor_compute_source_index(height_scale, h2, input_height);

    for (int w2 = 0; w2 < output_width; ++w2) {
      const int w1 =
          nearest_neighbor_compute_source_index(width_scale, w2, input_width);

      const scalar_t* pos1 = &idata[h1 * input_width + w1];
      scalar_t* pos2 = &odata[h2 * output_width + w2];

      for (int c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += input_height * input_width;
        pos2 += output_height * output_width;
      }
    }
  }
  // c10::raw::intrusive_ptr::decref(input);
}

template <typename scalar_t>
void upsampling_nearest2d_update_grad_input)(
    Tensor *grad_output_,
    Tensor *grad_input,
    int nbatch,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width)
{
  upsampling_2d_shape_check(
      grad_output,
      1,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_input.zero_();

  auto grad_output = grad_output_.contiguous();

  scalar_t* idata = grad_input.data<scalar_t>();
  scalar_t* odata = grad_output.data<scalar_t>();

  channels = channels * nbatch;

  const float height_scale = (float)input_height / (float)output_height;
  const float width_scale = (float)input_width / (float)output_width;

  // special case: just copy
  if (input_height == output_height && input_width == output_width) {
    for (int h2 = 0; h2 < output_height; ++h2) {
      const int h1 = h2;

      for (int w2 = 0; w2 < output_width; ++w2) {
        const int w1 = w2;
        scalar_t* pos1 = &idata[h1 * input_width + w1];
        const scalar_t* pos2 = &odata[h2 * output_width + w2];

        for (int c = 0; c < channels; ++c) {
          pos1[0] = pos2[0];
          pos1 += input_height * input_width;
          pos2 += output_height * output_width;
        }
      }
    }
    // c10::raw::intrusive_ptr::decref(grad_output);
    return;
  }

  for (int h2 = 0; h2 < output_height; ++h2) {
    const int h1 =
        nearest_neighbor_compute_source_index(height_scale, h2, input_height);

    for (int w2 = 0; w2 < output_width; ++w2) {
      const int w1 =
          nearest_neighbor_compute_source_index(width_scale, w2, input_width);
      scalar_t* pos1 = &idata[h1 * input_width + w1];
      const scalar_t* pos2 = &odata[h2 * output_width + w2];

      for (int c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += input_height * input_width;
        pos2 += output_height * output_width;
      }
    }
  }
  // c10::raw::intrusive_ptr::decref(grad_output);
}

} // namespace native
} // namespace at
