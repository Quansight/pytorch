// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
Tensor upsample_trilinear3d_out_cpu_template(
    const Tensor& input_,
    Tensor& output,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    bool align_corners) {
  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_depth = input_.size(2);
  int64_t input_height = input_.size(3);
  int64_t input_width = input_.size(4);

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
    for (int64_t t2 = 0; t2 < output_depth; ++t2) {
      const int64_t t1 = t2;

      for (int64_t h2 = 0; h2 < output_height; ++h2) {
        const int64_t h1 = h2;

        for (int64_t w2 = 0; w2 < output_width; ++w2) {
          const int64_t w1 = w2;
          const scalar_t* pos1 =
              &idata[t1 * input_height * input_width + h1 * input_width + w1];
          scalar_t* pos2 =
              &odata
                  [t2 * output_height * output_width + h2 * output_width + w2];

          for (int64_t c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += input_width * input_height * input_depth;
            pos2 += output_width * output_height * output_depth;
          }
        }
      }
    }
    return output;
  }
  const scalar_t rdepth = linear_upsample_compute_scale<scalar_t>(
      input_depth, output_depth, align_corners);
  const scalar_t rheight = linear_upsample_compute_scale<scalar_t>(
      input_height, output_height, align_corners);
  const scalar_t rwidth = linear_upsample_compute_scale<scalar_t>(
      input_width, output_width, align_corners);
  for (int64_t t2 = 0; t2 < output_depth; ++t2) {
    const scalar_t t1r = linear_upsample_compute_source_index<scalar_t>(
        rdepth, t2, align_corners);

    const int64_t t1 = t1r;
    const int64_t t1p = (t1 < input_depth - 1) ? 1 : 0;
    const scalar_t t1lambda = t1r - t1;
    const scalar_t t0lambda = static_cast<scalar_t>(1.) - t1lambda;

    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const scalar_t h1r = linear_upsample_compute_source_index<scalar_t>(
          rheight, h2, align_corners);

      const int64_t h1 = h1r;
      const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;
      const scalar_t h1lambda = h1r - h1;
      const scalar_t h0lambda = static_cast<scalar_t>(1.) - h1lambda;

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const scalar_t w1r = linear_upsample_compute_source_index<scalar_t>(
            rwidth, w2, align_corners);

        const int64_t w1 = w1r;
        const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;
        const scalar_t w1lambda = w1r - w1;
        const scalar_t w0lambda = static_cast<scalar_t>(1.) - w1lambda;
        const scalar_t* pos1 =
            &idata[t1 * input_height * input_width + h1 * input_width + w1];
        scalar_t* pos2 =
            &odata[t2 * output_height * output_width + h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos2[0] = t0lambda *
                  (h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                   h1lambda *
                       (w0lambda * pos1[h1p * input_width] +
                        w1lambda * pos1[h1p * input_width + w1p])) +
              t1lambda *
                  (h0lambda *
                       (w0lambda * pos1[t1p * input_height * input_width] +
                        w1lambda *
                            pos1[t1p * input_height * input_width + w1p]) +
                   h1lambda *
                       (w0lambda *
                            pos1
                                [t1p * input_height * input_width +
                                 h1p * input_width] +
                        w1lambda *
                            pos1
                                [t1p * input_height * input_width +
                                 h1p * input_width + w1p]));
          pos1 += input_width * input_height * input_depth;
          pos2 += output_width * output_height * output_depth;
        }
      }
    }
  }
  return output;
}

template <typename scalar_t>
Tensor upsample_trilinear3d_backward_out_cpu_template(
    const Tensor& grad_output_,
    Tensor& grad_input,
    int64_t nbatch,
    int64_t channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    bool align_corners) {
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

  auto grad_output = grad_output_.contiguous();

  grad_input.resize_(
      {nbatch, channels, input_depth, input_height, input_width});
  grad_input.zero_();

  scalar_t* data1 = grad_input.data<scalar_t>();
  scalar_t* data2 = grad_output.data<scalar_t>();

  channels = channels * nbatch;

  // special case: same-size matching grids
  if (input_depth == output_depth && input_height == output_height &&
      input_width == output_width) {
    for (int64_t t2 = 0; t2 < output_depth; ++t2) {
      const int64_t t1 = t2;

      for (int64_t h2 = 0; h2 < output_height; ++h2) {
        const int64_t h1 = h2;

        for (int64_t w2 = 0; w2 < output_width; ++w2) {
          const int64_t w1 = w2;
          scalar_t* pos1 =
              &data1[t1 * input_height * input_width + h1 * input_width + w1];
          const scalar_t* pos2 =
              &data2
                  [t2 * output_height * output_width + h2 * output_width + w2];

          for (int64_t c = 0; c < channels; ++c) {
            pos1[0] += pos2[0];
            pos1 += input_width * input_height * input_depth;
            pos2 += output_width * output_height * output_depth;
          }
        }
      }
    }
    return grad_input;
  }
  const scalar_t rdepth = linear_upsample_compute_scale<scalar_t>(
      input_depth, output_depth, align_corners);

  const scalar_t rheight = linear_upsample_compute_scale<scalar_t>(
      input_height, output_height, align_corners);

  const scalar_t rwidth = linear_upsample_compute_scale<scalar_t>(
      input_width, output_width, align_corners);

  for (int64_t t2 = 0; t2 < output_depth; ++t2) {
    const scalar_t t1r = linear_upsample_compute_source_index<scalar_t>(
        rdepth, t2, align_corners);
    const int64_t t1 = t1r;
    const int64_t t1p = (t1 < input_depth - 1) ? 1 : 0;
    const scalar_t t1lambda = t1r - t1;
    const scalar_t t0lambda = static_cast<scalar_t>(1.) - t1lambda;

    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const scalar_t h1r = linear_upsample_compute_source_index<scalar_t>(
          rheight, h2, align_corners);
      const int64_t h1 = h1r;
      const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;
      const scalar_t h1lambda = h1r - h1;
      const scalar_t h0lambda = static_cast<scalar_t>(1.) - h1lambda;

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const scalar_t w1r = linear_upsample_compute_source_index<scalar_t>(
            rwidth, w2, align_corners);
        const int64_t w1 = w1r;
        const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;
        const scalar_t w1lambda = w1r - w1;
        const scalar_t w0lambda = static_cast<scalar_t>(1.) - w1lambda;
        scalar_t* pos1 =
            &data1[t1 * input_height * input_width + h1 * input_width + w1];
        const scalar_t* pos2 =
            &data2[t2 * output_height * output_width + h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos1[0] += t0lambda * h0lambda * w0lambda * pos2[0];
          pos1[w1p] += t0lambda * h0lambda * w1lambda * pos2[0];
          pos1[h1p * input_width] += t0lambda * h1lambda * w0lambda * pos2[0];
          pos1[h1p * input_width + w1p] +=
              t0lambda * h1lambda * w1lambda * pos2[0];
          pos1[t1p * input_height * input_width] +=
              t1lambda * h0lambda * w0lambda * pos2[0];
          pos1[t1p * input_height * input_width + w1p] +=
              t1lambda * h0lambda * w1lambda * pos2[0];
          pos1[t1p * input_height * input_width + h1p * input_width] +=
              t1lambda * h1lambda * w0lambda * pos2[0];
          pos1[t1p * input_height * input_width + h1p * input_width + w1p] +=
              t1lambda * h1lambda * w1lambda * pos2[0];
          pos1 += input_width * input_height * input_depth;
          pos2 += output_width * output_height * output_depth;
        }
      }
    }
  }
  return grad_input;
}
} // namespace

Tensor upsample_trilinear3d_out_cpu(
    const Tensor& input,
    Tensor& output,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    bool align_corners) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "upsample_trilinear3d_out_cpu", [&] {
        return upsample_trilinear3d_out_cpu_template<scalar_t>(
            input,
            output,
            output_depth,
            output_height,
            output_width,
            align_corners);
      });
}

Tensor upsample_trilinear3d_cpu(
    const Tensor& input,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    bool align_corners) {
  auto output = at::empty({0}, input.options());
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "upsample_trilinear3d_cpu", [&] {
        return upsample_trilinear3d_out_cpu_template<scalar_t>(
            input,
            output,
            output_depth,
            output_height,
            output_width,
            align_corners);
      });
}

Tensor upsample_trilinear3d_backward_out_cpu(
    const Tensor& grad_output,
    Tensor& grad_input,
    int64_t nbatch,
    int64_t channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    bool align_corners) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "upsample_trilinear3d_backward_out_cpu", [&] {
        return upsample_trilinear3d_backward_out_cpu_template<scalar_t>(
            grad_output,
            grad_input,
            nbatch,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            align_corners);
      });
}

Tensor upsample_trilinear3d_backward_cpu(
    const Tensor& grad_output,
    int64_t nbatch,
    int64_t channels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    bool align_corners) {
  auto grad_input = at::zeros_like(grad_output);
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "upsample_trilinear3d_backward_cpu", [&] {
        return upsample_trilinear3d_backward_out_cpu_template<scalar_t>(
            grad_output,
            grad_input,
            nbatch,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            align_corners);
      });
}

} // namespace native
} // namespace at
