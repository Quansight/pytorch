#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSampling.h>

namespace at {
namespace native {

template <typename scalar_t>
static void upsampling_bicubic2d_out_frame_template(
    const Tensor& input_,
    Tensor& output,
    int output_height,
    int output_width,
    bool align_corners) {
  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);

  upsampling_2d_shape_check<scalar_t>(
      input_,
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

  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    for (int output_y = 0; output_y < output_height; output_y++) {
      for (int output_x = 0; output_x < output_width; output_x++) {
        const scalar_t* in = &idata[output_y * input_width + output_x];
        scalar_t* out = &odata[output_y * output_width + output_x];

        for (int c = 0; c < channels; ++c) {
          out[0] = in[0];
          in += input_width * input_height;
          out += output_width * output_height;
        }
      }
    }

    // c10::raw::intrusive_ptr::decref(input);
    return;
  }

  // Bicubic interpolation
  const scalar_t height_scale = linear_upsampling_compute_scale<scalar_t>(
      input_height, output_height, align_corners);
  const scalar_t width_scale = linear_upsampling_compute_scale<scalar_t>(
      input_width, output_width, align_corners);

  for (int output_y = 0; output_y < output_height; output_y++) {
    for (int output_x = 0; output_x < output_width; output_x++) {
      scalar_t* in = idata;
      scalar_t* out = odata;

      const scalar_t real_x = width_scale * output_x;
      int input_x = real_x;
      const scalar_t t_x = real_x - input_x;

      const scalar_t real_y = height_scale * output_y;
      int input_y = real_y;
      const scalar_t t_y = real_y - input_y;

      for (int c = 0; c < channels * nbatch; c++) {
        scalar_t coefficients[4];

        // Interpolate 4 times in the x direction
        for (int i = 0; i < 4; i++) {
          coefficients[i] = cubic_interp1d<scalar_t>(
              upsampling_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x - 1, input_y - 1 + i),
              upsampling_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x + 0, input_y - 1 + i),
              upsampling_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x + 1, input_y - 1 + i),
              upsampling_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x + 2, input_y - 1 + i),
              t_x);
        }

        // Interpolate in the y direction using x interpolations
        out[output_y * output_width + output_x] = cubic_interp1d<scalar_t>(
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            t_y);

        // Move to next channel
        in += input_width * input_height;
        out += output_width * output_height;
      }
    }
  }

  // c10::raw::intrusive_ptr::decref(input);
}

template <typename scalar_t>
static void upsampling_bicubic2d_update_grad_input_template(
    const Tensor& grad_output_,
    Tensor& grad_input,
    int nbatch,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    bool align_corners) {
  upsampling_2d_shape_check<scalar_t>(
      grad_output_,
      1,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_input.zero_();

  scalar_t* idata = grad_input.data<scalar_t>();
  scalar_t* odata = grad_output.data<scalar_t>();

  channels = channels * nbatch;

  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    for (int output_y = 0; output_y < output_height; output_y++) {
      for (int output_x = 0; output_x < output_width; output_x++) {
        scalar_t* in = &idata[output_y * input_width + output_x];
        scalar_t* out = &odata[output_y * output_width + output_x];
        for (int c = 0; c < channels; ++c) {
          in[0] = out[0];
          in += input_width * input_height;
          out += output_width * output_height;
        }
      }
    }
    // c10::raw::intrusive_ptr::decref(grad_output);
    return;
  }

  const scalar_t height_scale = linear_upsampling_compute_scale<scalar_t>(
      input_height, output_height, align_corners);
  const scalar_t width_scale = linear_upsampling_compute_scale<scalar_t>(
      input_width, output_width, align_corners);

  for (int output_y = 0; output_y < output_height; output_y++) {
    for (int output_x = 0; output_x < output_width; output_x++) {
      scalar_t* in = idata;
      scalar_t* out = odata;

      scalar_t real_x = width_scale * output_x;
      int input_x = real_x;
      scalar_t t_x = real_x - input_x;

      scalar_t real_y = height_scale * output_y;
      int input_y = real_y;
      scalar_t t_y = real_y - input_y;

      scalar_t x_coeffs[4];
      scalar_t y_coeffs[4];

      get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, t_x);
      get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, t_y);

      for (int c = 0; c < channels; c++) {
        scalar_t out_value = out[output_y * output_width + output_x];

        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            upsampling_increment_value_bounded<scalar_t>(
                in,
                input_width,
                input_height,
                input_x - 1 + i,
                input_y - 1 + j,
                out_value * y_coeffs[j] * x_coeffs[i]);
          }
        }

        in += input_width * input_height;
        out += output_width * output_height;
      }
    }
  }

  // c10::raw::intrusive_ptr::decref(grad_output);
}

} // namespace native
} // namespace at
