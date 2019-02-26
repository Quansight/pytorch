#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
Tensor upsample_nearest3d_out_cpu_template(
    const Tensor& input_,
    Tensor& output,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_depth = input_.size(2);
  int64_t input_height = input_.size(3);
  int64_t input_width = input_.size(4);

  const float depth_scale = (float)input_depth / (float)output_depth;
  const float height_scale = (float)input_height / (float)output_height;
  const float width_scale = (float)input_width / (float)output_width;

  upsample_3d_shape_check(
      input_,
      static_cast<int64_t>(0),
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_depth, output_height, output_width});
  output.zero_();

  scalar_t* idata = input.data<scalar_t>();
  scalar_t* odata = output.data<scalar_t>();

  channels = channels * nbatch;

  AT_ASSERT(
      input_depth > 0 && input_height > 0 && input_width > 0 &&
      output_depth > 0 && output_height > 0 && output_width > 0);

  // special case: just copy
  if (input_depth == output_depth && input_height == output_height &&
      input_width == output_width) {
    for (int64_t d2 = 0; d2 < output_depth; ++d2) {
      const int64_t d1 = d2;

      for (int64_t h2 = 0; h2 < output_height; ++h2) {
        const int64_t h1 = h2;

        for (int64_t w2 = 0; w2 < output_width; ++w2) {
          const int64_t w1 = w2;
          const scalar_t* pos1 =
              &idata[d1 * input_height * input_width + h1 * input_width + w1];
          scalar_t* pos2 =
              &odata
                  [d2 * output_height * output_width + h2 * output_width + w2];

          for (int64_t c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += input_depth * input_height * input_width;
            pos2 += output_depth * output_height * output_width;
          }
        }
      }
    }
    return output;
  }

  for (int64_t d2 = 0; d2 < output_depth; ++d2) {
    const int64_t d1 =
        nearest_neighbor_compute_source_index(depth_scale, d2, input_depth);

    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const int64_t h1 =
          nearest_neighbor_compute_source_index(height_scale, h2, input_height);

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const int64_t w1 =
            nearest_neighbor_compute_source_index(width_scale, w2, input_width);
        const scalar_t* pos1 =
            &idata[d1 * input_height * input_width + h1 * input_width + w1];
        scalar_t* pos2 =
            &odata[d2 * output_height * output_width + h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += input_depth * input_height * input_width;
          pos2 += output_depth * output_height * output_width;
        }
      }
    }
  }
  return output;
}

template <typename scalar_t>
Tensor upsample_nearest3d_backward_out_cpu_template(
    const Tensor& grad_output_,
    Tensor& grad_input,
    int64_t nbatch,
    int64_t channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  upsample_3d_shape_check(
      grad_output_,
      static_cast<int64_t>(1),
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);

  grad_input.resize_(
      {nbatch, channels, input_depth, input_height, input_width});
  grad_input.zero_();

  auto grad_output = grad_output_.contiguous();

  scalar_t* idata = grad_input.data<scalar_t>();
  scalar_t* odata = grad_output.data<scalar_t>();

  channels = channels * nbatch;

  const float depth_scale = (float)input_depth / (float)output_depth;
  const float height_scale = (float)input_height / (float)output_height;
  const float width_scale = (float)input_width / (float)output_width;

  // special case: just copy
  if (input_depth == output_depth && input_height == output_height &&
      input_width == output_width) {
    for (int64_t d2 = 0; d2 < output_depth; ++d2) {
      const int64_t d1 = d2;

      for (int64_t h2 = 0; h2 < output_height; ++h2) {
        const int64_t h1 = h2;

        for (int64_t w2 = 0; w2 < output_width; ++w2) {
          const int64_t w1 = w2;
          scalar_t* pos1 =
              &idata[d1 * input_height * input_width + h1 * input_width + w1];
          const scalar_t* pos2 =
              &odata
                  [d2 * output_height * output_width + h2 * output_width + w2];

          for (int64_t c = 0; c < channels; ++c) {
            pos1[0] += pos2[0];
            pos1 += input_depth * input_height * input_width;
            pos2 += output_depth * output_height * output_width;
          }
        }
      }
    }
    return grad_input;
  }

  for (int64_t d2 = 0; d2 < output_depth; ++d2) {
    const int64_t d1 =
        nearest_neighbor_compute_source_index(depth_scale, d2, input_depth);

    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const int64_t h1 =
          nearest_neighbor_compute_source_index(height_scale, h2, input_height);

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const int64_t w1 =
            nearest_neighbor_compute_source_index(width_scale, w2, input_width);
        scalar_t* pos1 =
            &idata[d1 * input_height * input_width + h1 * input_width + w1];
        const scalar_t* pos2 =
            &odata[d2 * output_height * output_width + h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos1[0] += pos2[0];
          pos1 += input_depth * input_height * input_width;
          pos2 += output_depth * output_height * output_width;
        }
      }
    }
  }
  return grad_input;
}
} // namespace

Tensor upsample_nearest3d_out_cpu(
    const Tensor& input,
    Tensor& output,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "upsample_nearest3d_out_cpu", [&] {
        return upsample_nearest3d_out_cpu_template<scalar_t>(
            input,
            output,
            output_depth,
            output_height,
            output_width);
      });
}

Tensor upsample_nearest3d_cpu(
    const Tensor& input,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  auto output = at::empty({0}, input.options());
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "upsample_nearest3d_cpu", [&] {
        return upsample_nearest3d_out_cpu_template<scalar_t>(
            input,
            output,
            output_depth,
            output_height,
            output_width);
      });
}

Tensor upsample_nearest3d_backward_out_cpu(
    const Tensor& grad_output,
    Tensor& grad_input,
    int64_t nbatch,
    int64_t channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "upsample_nearest3d_backward_out_cpu", [&] {
        return upsample_nearest3d_backward_out_cpu_template<scalar_t>(
            grad_output,
            grad_input,
            nbatch,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width);
      });
}

Tensor upsample_nearest3d_backward_cpu(
    const Tensor& grad_output,
    int64_t nbatch,
    int64_t channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  auto grad_input = at::zeros_like(grad_output);
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "upsample_nearest3d_backward_cpu", [&] {
        return upsample_nearest3d_backward_out_cpu_template<scalar_t>(
            grad_output,
            grad_input,
            nbatch,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width);
      });
}

} // namespace native
} // namespace at
