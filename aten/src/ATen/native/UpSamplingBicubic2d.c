#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSampling.h>

namespace at {
namespace native {

Bool check_dim_size(const Tensor& output, inst64_t dim, int64_t ndim) {
  AT_CHECK(
    (input.dim() == 4 || input.dim() == 5) && input.dim() == grid.dim(),
    "grid_sampler(): expected 4D or 5D input and grid with same number "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  AT_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  AT_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());
  // cudnn does not support inputs larger than 1024
  if (at::native::cudnn_is_acceptable(input) &&
      static_cast<GridSamplerPadding>(padding_mode) == GridSamplerPadding::Zeros &&
      input.dim() == 4 &&
      input.size(1) <= 1024) {
    return cudnn_grid_sampler(input, grid);
  }
  if (input.dim() == 4) {
    return at::grid_sampler_2d(input, grid, 0, padding_mode);
  } else {
    return at::grid_sampler_3d(input, grid, 0, padding_mode);
  }
}

static inline void upsampling_bicubic2d_shape_check(
    Tensor* input,
    Tensor* grad_output,
    int nbatch,
    int nchannels,
    int input_height,
    int input_width,
    int output_height,
    int output_width) {
  AT_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "input and output sizes should be greater than 0,"
      " but got input (H: %d, W: %d) output (H: %d, W: %d)",
      input_height,
      input_width,
      output_height,
      output_width);

  if (input != NULL) {
    AT_CHECK(
        !input.numel() == 0 && input.dim() == 4,
        "non-empty 4D input tensor expected but got: %s");
  }

  if (grad_output != NULL) {
    THNN_CHECK_DIM_SIZE(grad_output, 4, 0, nbatch);
    THNN_CHECK_DIM_SIZE(grad_output, 4, 1, nchannels);
    THNN_CHECK_DIM_SIZE(grad_output, 4, 2, output_height);
    THNN_CHECK_DIM_SIZE(grad_output, 4, 3, output_width);
  }
}

void upsampling_bicubic2d_out(
    Tensor* input,
    Tensor* output,
    int output_height,
    int output_width,
    bool align_corners) {
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  upsampling_bicubic2d_shape_check(
      input,
      NULL,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  input = Tensor_(newContiguous)(input);
  THTensor_(resize4d)(
      output,
      THTensor_(size)(input, 0),
      THTensor_(size)(input, 1),
      output_height,
      output_width);
  THTensor_(zero)(output);
  scalar_t* idata = input->data<scalar_t>();
  scalar_t* odata = output->data<scalar_t>();

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
    c10::raw::intrusive_ptr::decref(input);
    return;
  }

  // Bicubic interpolation
  const accreal height_scale = linear_upsampling_compute_scale<accreal>(
      input_height, output_height, align_corners);
  const accreal width_scale = linear_upsampling_compute_scale<accreal>(
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

  c10::raw::intrusive_ptr::decref(input);
}

void upsampling_bicubic2d_update_grad_input(
    Tensor* grad_output,
    Tensor* gradInput,
    int nbatch,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    bool align_corners) {
  THNN_(spatial_upsampling_bicubic_shape_check)
  (NULL,
   grad_output,
   nbatch,
   channels,
   input_height,
   input_width,
   output_height,
   output_width);

  THTensor_(resize4d)(gradInput, nbatch, channels, input_height, input_width);
  THTensor_(zero)(gradInput);

  grad_output = THTensor_(newContiguous)(grad_output);
  scalar_t* idata = gradInput->data<scalar_t>();
  scalar_t* odata = grad_output->data<scalar_t>();
  channels = nbatch * channels;

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
    c10::raw::intrusive_ptr::decref(grad_output);
    return;
  }

  const accreal height_scale = linear_upsampling_compute_scale<accreal>(
      input_height, output_height, align_corners);
  const accreal width_scale = linear_upsampling_compute_scale<accreal>(
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

  c10::raw::intrusive_ptr::decref(grad_output);
}

#endif
