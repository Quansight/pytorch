#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
Tensor upsample_nearest1d_out_cpu_template(
    const Tensor& input_,
    IntArrayRef output_size,
    Tensor& output) {

  int64_t output_width = output_size[0];

  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_width = input_.size(2);
  const float scale = (float)input_width / (float)output_width;

  upsample_1d_shape_check(
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
    return output;
  }

  for (int64_t w2 = 0; w2 < output_width; ++w2) {
    const scalar_t src_x =
        nearest_neighbor_compute_source_index(scale, w2, input_width);
    const int64_t w1 = src_x;
    const scalar_t* pos1 = &idata[w1];
    scalar_t* pos2 = &odata[w2];

    for (int64_t c = 0; c < channels; ++c) {
      pos2[0] = pos1[0];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
  return output;
}

template <typename scalar_t>
Tensor upsample_nearest1d_backward_out_cpu_template(
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    Tensor& grad_input) {

  int64_t output_width = output_size[0];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  upsample_1d_shape_check(
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

  channels = channels * nbatch;

  const float scale = (float)input_width / (float)output_width;

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
    return grad_input;
  }

  for (int64_t w2 = 0; w2 < output_width; ++w2) {
    const int64_t w1 =
        nearest_neighbor_compute_source_index(scale, w2, input_width);

    scalar_t* pos1 = &data1[w1];
    const scalar_t* pos2 = &data2[w2];

    for (int64_t c = 0; c < channels; ++c) {
      pos1[0] += pos2[0];
      pos1 += input_width;
      pos2 += output_width;
    }
  }
  return grad_input;
}
} // namespace

Tensor upsample_nearest1d_out_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "upsample_nearest1d_out_cpu", [&] {
        return upsample_nearest1d_out_cpu_template<scalar_t>(
            input, output_size, output);
      });
}

Tensor upsample_nearest1d_cpu(
    const Tensor& input,
    IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "upsample_nearest1d_cpu", [&] {
        return upsample_nearest1d_out_cpu_template<scalar_t>(
            input, output_size, output);
      });
}

Tensor upsample_nearest1d_backward_out_cpu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    Tensor& grad_input) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "upsample_nearest1d_backward_out_cpu", [&] {
        return upsample_nearest1d_backward_out_cpu_template<scalar_t>(
            grad_output,
            output_size,
            input_size,
            grad_input);
      });
}

Tensor upsample_nearest1d_backward_cpu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  auto grad_input = at::zeros_like(grad_output);
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "upsample_nearest1d_backward_cpu", [&] {
        return upsample_nearest1d_backward_out_cpu_template<scalar_t>(
            grad_output,
            output_size,
            input_size,
            grad_input);
      });
}

} // namespace native
} // namespace at
