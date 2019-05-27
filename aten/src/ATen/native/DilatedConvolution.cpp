
#include <ATen/div_rtn.h>
#include <algorithm>
#include <tuple>
#include "ATen/ATen.h"
#include "TH/THBlasUtils.h"

#define TORCH_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE) \
  TORCH_CHECK(                                       \
      T.dim() == DIM && T.size(DIM_SIZE) == SIZE,    \
      "Need " #T " of dimension ",                   \
      DIM,                                           \
      " and " #T ".size[",                           \
      DIM_SIZE,                                      \
      "] == ",                                       \
      SIZE,                                          \
      " but got input to be of shape ",              \
      T.sizes())

/*
  The following CALL... macros are for convenience of this source file
  only. These should be replaced with ATen native functions as soon as
  col2im, im2col, gemm, and gemv are ported.
 */

#define CALLIM2COL(IM, COL)                 \
  {                                         \
    auto* COL##_ptr = COL.data<scalar_t>(); \
    auto* IM##_ptr = IM.data<scalar_t>();   \
    THBlas_im2col<scalar_t>(                \
        IM##_ptr,                           \
        nInputPlane,                        \
        inputHeight,                        \
        inputWidth,                         \
        outputHeight,                       \
        outputWidth,                        \
        kH,                                 \
        kW,                                 \
        padH,                               \
        padW,                               \
        dH,                                 \
        dW,                                 \
        dilationH,                          \
        dilationW,                          \
        COL##_ptr);                         \
  }

#define CALLCOL2IM(COL, IM)                 \
  {                                         \
    auto* COL##_ptr = COL.data<scalar_t>(); \
    auto* IM##_ptr = IM.data<scalar_t>();   \
    THBlas_col2im<scalar_t>(                \
        COL##_ptr,                          \
        nInputPlane,                        \
        inputHeight,                        \
        inputWidth,                         \
        outputHeight,                       \
        outputWidth,                        \
        kH,                                 \
        kW,                                 \
        padH,                               \
        padW,                               \
        dH,                                 \
        dW,                                 \
        dilationH,                          \
        dilationW,                          \
        IM##_ptr);                          \
  }

#define CALLGEMM(TA, TB, ALPHA, A, N, B, M, BETA, C)                       \
  {                                                                        \
    auto* A##_ptr = A.data<scalar_t>();                                    \
    auto* B##_ptr = B.data<scalar_t>();                                    \
    auto* C##_ptr = C.data<scalar_t>();                                    \
    THBlas_gemm<scalar_t>(                                                 \
        TA, TB, n, m, k, ALPHA, A##_ptr, N, B##_ptr, M, BETA, C##_ptr, n); \
  }

#define CALLGEMV(TA, ALPHA, A, K, X, BETA, Y)                       \
  {                                                                 \
    auto* A##_ptr = A.data<scalar_t>();                             \
    auto* X##_ptr = X.data<scalar_t>();                             \
    auto* Y##_ptr = Y.data<scalar_t>();                             \
    THBlas_gemv<scalar_t>(                                          \
        TA, m, n, ALPHA, A##_ptr, K, X##_ptr, 1, BETA, Y##_ptr, 1); \
  }

/*
  Some convenience macros
 */

#define CALL_TEMPLATE              \
  conv_dilated2d_all_cpu_template( \
      output,                      \
      input,                       \
      weight,                      \
      bias,                        \
      grad_output,                 \
      grad_input,                  \
      grad_weight,                 \
      grad_bias,                   \
      kernel_size,                 \
      stride_size,                 \
      pad_size,                    \
      dilation_size,               \
      columns,                     \
      ones);

#define CALL_OUT          \
  conv_dilated2d_out_cpu( \
      output,             \
      columns,            \
      ones,               \
      input,              \
      weight,             \
      kernel_size,        \
      bias,               \
      stride_size,        \
      pad_size,           \
      dilation_size)

#define CALL_FORWARD_OUT          \
  conv_dilated2d_forward_out_cpu( \
      output,                     \
      columns,                    \
      ones,                       \
      input,                      \
      weight,                     \
      kernel_size,                \
      bias,                       \
      stride_size,                \
      pad_size,                   \
      dilation_size)

#define CALL_BACKWARD_OUT          \
  conv_dilated2d_backward_out_cpu( \
      grad_input,                  \
      grad_weight,                 \
      grad_bias,                   \
      grad_output,                 \
      input,                       \
      weight,                      \
      kernel_size,                 \
      stride_size,                 \
      pad_size,                    \
      dilation_size,               \
      columns,                     \
      ones)

#define INSERT_BATCH_DIMENSION(A)                      \
  {                                                    \
    if (A.numel() > 0) {                               \
      A.resize_({1, A.size(0), A.size(1), A.size(2)}); \
    }                                                  \
  }

#define DROP_BATCH_DIMENSION(A)                     \
  {                                                 \
    if (A.numel() > 0) {                            \
      A.resize_({A.size(1), A.size(2), A.size(3)}); \
    }                                               \
  }

namespace at {
namespace native {

namespace {

inline bool all_positive(IntArrayRef& arr) {
  return std::all_of(
      arr.begin(), arr.end(), [](int64_t item) { return item > 0; });
}

#define OUTPUTSIZE(INDEX)                                      \
  (input.size(INDEX + input.dim() - 2) + 2 * pad_size[INDEX] - \
   (dilation_size[INDEX] * (kernel_size[INDEX] - 1) + 1)) /    \
          stride_size[INDEX] +                                 \
      1

void conv_dilated2d_shape_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    const Tensor& grad_weight,
    const Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const Tensor& columns,
    const Tensor& ones) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "kernel sizes length should be two, but got ",
      kernel_size);
  TORCH_CHECK(
      stride_size.size() == 2,
      "strides length should be two, but got ",
      stride_size);
  TORCH_CHECK(
      dilation_size.size() == 2,
      "dilations length should be two, but got ",
      dilation_size);
  TORCH_CHECK(
      pad_size.size() == 2, "pads length should be two, but got ", pad_size);

  TORCH_CHECK(
      all_positive(kernel_size),
      "kernel size should be greater than zero, but got ",
      kernel_size);
  TORCH_CHECK(
      all_positive(stride_size),
      "stride should be greater than zero, but got ",
      stride_size);
  TORCH_CHECK(
      all_positive(dilation_size),
      "dilation should be greater than zero, but got ",
      dilation_size);

  if (weight.numel() > 0) {
    TORCH_CHECK(
        weight.dim() == 4,
        "non-empty 4D weight tensor (nOutputPlane, nInputPlane, kH, kW) expected, "
        "but got ",
        weight.sizes());
    if (bias.numel() > 0) {
      TORCH_CHECK_DIM_SIZE(bias, 1, 0, weight.size(0));
      TORCH_CHECK(bias.is_contiguous(), "bias needs to be contiguous");
    }
  }

  if (grad_weight.numel() > 0) {
    TORCH_CHECK(
        grad_weight.dim() == 4,
        "non-empty 4D weight gradient tensor (nOutputPlane, nInputPlane, kH, kW) expected, "
        "but got ",
        grad_weight.sizes());
    TORCH_CHECK(
        grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
    if (grad_bias.numel() > 0) {
      TORCH_CHECK_DIM_SIZE(grad_bias, 1, 0, grad_weight.size(0));
      TORCH_CHECK(
          grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
      TORCH_CHECK(ones.is_contiguous(), "ones needs to be contiguous");
    }
  }

  TORCH_CHECK(input.numel() > 0 && input.dim() == 4,
              "non-empty 3D or 4D input tensor expected but got: ",
              input.sizes());
  //TODO: since check shape is called after adding batch dimension, we
  //always have input.dim()==4, and what follows can be simplified.
  
  int ndim = input.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }
  TORCH_CHECK(
      input.numel() > 0 && (ndim == 4 || ndim == 3),
      "non-empty 3D or 4D input tensor expected but got: ",
      input.sizes());

  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  auto dilationH = dilation_size[0];
  auto dilationW = dilation_size[1];

  int64_t inputHeight = input.size(dimh);
  int64_t inputWidth = input.size(dimw);
  int64_t outputHeight = OUTPUTSIZE(0);
  int64_t outputWidth = OUTPUTSIZE(1);
  TORCH_CHECK(
      outputWidth >= 0 && outputHeight >= 0,
      "Given input size per channel: (",
      inputHeight,
      " x ",
      inputWidth,
      "). "
      "Calculated output size per channel: (",
      outputHeight,
      " x ",
      outputWidth,
      "). Output size is too small");

  if (weight.numel() > 1) {
    int64_t nInputPlane = weight.size(1);
    TORCH_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  }

  if (grad_output.numel() > 0) {
    if (weight.numel() > 0) {
      int64_t nOutputPlane = weight.size(0);
      TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimf, nOutputPlane);
    } else if (bias.numel() > 0) {
      int64_t nOutputPlane = bias.size(0);
      TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimf, nOutputPlane);
    }
    TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimh, outputHeight);
    TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimw, outputWidth);
  }

  if (columns.numel() > 0) {
    TORCH_CHECK(columns.is_contiguous(), "columns needs to be contiguous");
  }

  if (ones.numel() > 0) {
    TORCH_CHECK(
        ones.numel() >= outputHeight * outputWidth,
        "expected at least ",
        outputHeight * outputWidth,
        " ones but got ",
        ones.sizes());
  }
} // conv_dilated2d_shape_check


  /*
    conv_dilated2d_all_cpu_template

    Main worker. Computes tensors output, grad_input, grad_weight,
    and/or grad_bias if non-empty.
   */
void conv_dilated2d_all_cpu_template(
    Tensor& output,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& grad_output_,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const Tensor& columns_,
    const Tensor& ones_) {
  auto input = input_.contiguous();
  auto weight = weight_.contiguous();
  auto bias = bias_.contiguous();
  auto grad_output = grad_output_.contiguous();
  Tensor columns = columns_;
  Tensor ones = ones_;

  bool is_batch = input.dim() == 3;
  if (is_batch) {
    INSERT_BATCH_DIMENSION(input);
    INSERT_BATCH_DIMENSION(output);
    INSERT_BATCH_DIMENSION(grad_input);
    INSERT_BATCH_DIMENSION(grad_output);
  }

  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  auto dilationH = dilation_size[0];
  auto dilationW = dilation_size[1];

  int64_t batchSize = input.size(0);
  int64_t nInputPlane = weight.size(1);
  int64_t nOutputPlane = weight.size(0);
  int64_t inputHeight = input.size(2);
  int64_t inputWidth = input.size(3);
  int64_t outputHeight = OUTPUTSIZE(0);
  int64_t outputWidth = OUTPUTSIZE(1);

  // Resize temporary columns
  if (output.numel() > 0 || grad_weight.numel() > 0 || grad_input.numel() > 0) {
    columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});
  }
  if (grad_weight.numel() > 0) {
    grad_weight.zero_();
  }
  if (grad_bias.numel() > 0) {
    grad_bias.zero_();
  }
  // Resize temporary ones
  if (bias.numel() > 0 || grad_bias.numel() > 0) {
    // Define a buffer of ones, for bias accumulation
    ones.resize_({outputHeight, outputWidth});
    ones.fill_(1);
  }

  // checking shapes after all shapes have been settled:
  conv_dilated2d_shape_check(
      input,
      weight,
      bias,
      grad_output,
      grad_weight,
      grad_bias,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size,
      columns,
      ones);

  // Helpers
  auto options = input.options();
  Tensor input_n = at::empty({0}, options);
  Tensor output_n = at::empty({0}, options);
  Tensor grad_input_n = at::empty({0}, options);
  Tensor grad_output_n = at::empty({0}, options);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_dilated2d", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt++) {
      // Matrix multiply per output:
      input_n = input.select(0, elt);

      // Output
      if (output.numel() > 0) {
        output_n = output.select(0, elt);
        if (bias.numel() > 0) {
          int64_t n = outputHeight * outputWidth;
          int64_t m = nOutputPlane;
          int64_t k = 1;
          CALLGEMM('t', 'n', 1, ones, k, bias, k, 0, output_n);
        } else {
          output_n.zero_();
        }
        // Extract columns:
        CALLIM2COL(input_n, columns);
        {
          int64_t n = outputHeight * outputWidth;
          int64_t m = nOutputPlane;
          int64_t k = nInputPlane * kH * kW;
          CALLGEMM('n', 'n', 1, columns, n, weight, k, 1, output_n);
        }
      } else {
        // All gradients
        grad_output_n = grad_output.select(0, elt);
      }

      // Gradient of input:
      if (grad_input.numel() > 0) {
        int64_t n = columns.size(1);
        int64_t m = nInputPlane * kW * kH;
        int64_t k = nOutputPlane;
        CALLGEMM('n', 't', 1, grad_output_n, n, weight, m, 0, columns);
        // Unpack columns back into input:
        grad_input_n = grad_input.select(0, elt);
        CALLCOL2IM(columns, grad_input_n);
      }

      // Gradient of weight:
      if (grad_weight.numel() > 0) {
        int64_t n = nInputPlane * kW * kH;
        int64_t m = nOutputPlane;
        int64_t k = columns.size(1);
        scalar_t scale = 1; // TODO: expose as argument
        // Extract columns:
        CALLIM2COL(input_n, columns);
        CALLGEMM('t', 'n', scale, columns, k, grad_output_n, k, 1, grad_weight);
      }

      // Gradient of bias:
      if (grad_bias.numel() > 0) {
        int64_t m = outputHeight * outputWidth;
        int64_t n = nOutputPlane;
        scalar_t scale = 1; // TODO: expose as argument
        CALLGEMV('t', scale, grad_output_n, m, ones, 1, grad_bias);
      }
    }
  });
  if (is_batch) {
    DROP_BATCH_DIMENSION(input);
    DROP_BATCH_DIMENSION(output);
    DROP_BATCH_DIMENSION(grad_input);
    DROP_BATCH_DIMENSION(grad_output);
  }
} // conv_dilated2d_all_cpu_template

} // namespace


std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated2d_out_cpu(
    Tensor& output,
    Tensor& columns,
    Tensor& ones,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  auto options = input.options();
  Tensor grad_output = at::empty({0}, options);  // not used
  Tensor grad_input = at::empty({0}, options);   // not used
  Tensor grad_weight = at::empty({0}, options);  // not used
  Tensor grad_bias = at::empty({0}, options);    // not used
  int64_t nOutputPlane = weight.size(0);
  int64_t outputHeight = OUTPUTSIZE(0);
  int64_t outputWidth = OUTPUTSIZE(1);
  if (input.dim() == 3) {
    output.resize_({nOutputPlane, outputHeight, outputWidth});
  } else {
    output.resize_({input.size(0), nOutputPlane, outputHeight, outputWidth});
  }
  CALL_TEMPLATE;
  return std::tie(output, columns, ones);
}

Tensor& conv_dilated2d_out_cpu(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  auto options = input.options();
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  CALL_OUT;
  return output;
}

Tensor conv_dilated2d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  auto options = input.options();
  Tensor output = at::empty({0}, options);
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  CALL_OUT;
  return output;
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated2d_forward_out_cpu(
    Tensor& output,
    Tensor& columns,
    Tensor& ones,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  std::cout << "NOT IMPLEMENTED: conv_dilated2d_forward_out_cpu3" << std::endl;
  CALL_OUT; // Is this correct??
  return std::tie(output, columns, ones);
}

Tensor& conv_dilated2d_forward_out_cpu(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  auto options = output.options();
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  std::cout << "NOT IMPLEMENTED: conv_dilated2d_forward_out_cpu1" << std::endl;
  CALL_FORWARD_OUT;
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated2d_forward_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  auto options = input.options();
  Tensor output = at::empty({0}, options);
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  std::cout << "NOT IMPLEMENTED: conv_dilated2d_forward_cpu" << std::endl;
  CALL_FORWARD_OUT;
  return std::tie(output, columns, ones);
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated2d_backward_out_cpu(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const Tensor& columns,
    const Tensor& ones) {
  auto options = grad_input.options();
  Tensor output = at::empty({0}, options);
  Tensor columns_buf = columns;
  Tensor ones_buf = ones;
  Tensor bias = at::empty({0}, options);

  grad_input.resize_(input.sizes());
  grad_weight.resize_(weight.sizes());
  grad_bias.resize_(weight.size(0)); // TODO: is this correct?
  CALL_TEMPLATE;
  return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const Tensor& columns,
    const Tensor& ones,
    const std::array<bool, 3ul> output_mask) {
  auto options = grad_output.options();
  Tensor grad_input = at::empty({0}, options);
  Tensor grad_weight = at::empty({0}, options);
  Tensor grad_bias = at::empty({0}, options);
  CALL_BACKWARD_OUT;
  return std::tie(grad_input, grad_weight, grad_bias);
}

} // namespace native
} // namespace at
