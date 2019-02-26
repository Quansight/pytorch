#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSampling.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
Tensor upsampling_nearest2d_cpu_template(
    const Tensor& input_,
    Tensor& output,
    int64_t output_height,
    int64_t output_width) {
  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);

  const float height_scale = (float)input_height / (float)output_height;
  const float width_scale = (float)input_width / (float)output_width;

  upsampling_2d_shape_check(
      input_,
      static_cast<int64_t>(0),
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
    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const int64_t h1 = h2;

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const int64_t w1 = w2;
        const scalar_t* pos1 = &idata[h1 * input_width + w1];
        scalar_t* pos2 = &odata[h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += input_height * input_width;
          pos2 += output_height * output_width;
        }
      }
    }
    return output;
  }

  for (int64_t h2 = 0; h2 < output_height; ++h2) {
    const int64_t h1 =
        nearest_neighbor_compute_source_index(height_scale, h2, input_height);

    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const int64_t w1 =
          nearest_neighbor_compute_source_index(width_scale, w2, input_width);

      const scalar_t* pos1 = &idata[h1 * input_width + w1];
      scalar_t* pos2 = &odata[h2 * output_width + w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += input_height * input_width;
        pos2 += output_height * output_width;
      }
    }
  }
  return output;
}

template <typename scalar_t>
Tensor upsampling_nearest2d_backward_cpu_template(
    const Tensor& grad_output_,
    Tensor& grad_input,
    int64_t nbatch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  upsampling_2d_shape_check(
      grad_output_,
      static_cast<int64_t>(1),
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
    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const int64_t h1 = h2;

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const int64_t w1 = w2;
        scalar_t* pos1 = &idata[h1 * input_width + w1];
        const scalar_t* pos2 = &odata[h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos1[0] = pos2[0];
          pos1 += input_height * input_width;
          pos2 += output_height * output_width;
        }
      }
    }
    return grad_input;
  }

  for (int64_t h2 = 0; h2 < output_height; ++h2) {
    const int64_t h1 =
        nearest_neighbor_compute_source_index(height_scale, h2, input_height);

    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const int64_t w1 =
          nearest_neighbor_compute_source_index(width_scale, w2, input_width);
      scalar_t* pos1 = &idata[h1 * input_width + w1];
      const scalar_t* pos2 = &odata[h2 * output_width + w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos1[0] += pos2[0];
        pos1 += input_height * input_width;
        pos2 += output_height * output_width;
      }
    }
  }
  return grad_input;
}
} // namespace

Tensor upsampling_nearest2d_cpu(
    const Tensor& input,
    Tensor& output,
    int64_t output_height,
    int64_t output_width) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "upsampling_nearest2d_cpu", [&] {
        return upsampling_nearest2d_cpu_template<scalar_t>(
            input, output, output_height, output_width);
      });
}

Tensor upsampling_nearest2d_backward_cpu(
    const Tensor& grad_output,
    Tensor& grad_input,
    int64_t nbatch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "upsampling_nearest2d_backward_cpu", [&] {
        return upsampling_nearest2d_backward_cpu_template<scalar_t>(
            grad_output,
            grad_input,
            nbatch,
            channels,
            input_height,
            input_width,
            output_height,
            output_width);
      });
}

} // namespace native
} // namespace at
