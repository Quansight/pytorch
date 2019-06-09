
#include <ATen/div_rtn.h>
#include <algorithm>
#include <tuple>
#include "ATen/ATen.h"
#include "ATen/native/im2col.h"
#include "ATen/native/vol2col.h"
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
  Some convenience macros
 */

#define CALL_TEMPLATE(DIM)               \
  conv_dilated##DIM##d_all_cpu_template( \
      output,                            \
      input,                             \
      weight,                            \
      bias,                              \
      grad_output,                       \
      grad_input,                        \
      grad_weight,                       \
      grad_bias,                         \
      kernel_size,                       \
      stride_size,                       \
      pad_size,                          \
      dilation_size,                     \
      columns,                           \
      ones)

#define CALL_OUT(DIM)           \
  conv_dilated##DIM##d_out_cpu( \
      output,                   \
      columns,                  \
      ones,                     \
      input,                    \
      weight,                   \
      kernel_size,              \
      bias,                     \
      stride_size,              \
      pad_size,                 \
      dilation_size)

#define CALL_FORWARD_OUT(DIM)           \
  conv_dilated##DIM##d_forward_out_cpu( \
      output,                           \
      columns,                          \
      ones,                             \
      input,                            \
      weight,                           \
      kernel_size,                      \
      bias,                             \
      stride_size,                      \
      pad_size,                         \
      dilation_size)

#define CALL_BACKWARD_OUT(DIM)           \
  conv_dilated##DIM##d_backward_out_cpu( \
      grad_input,                        \
      grad_weight,                       \
      grad_bias,                         \
      grad_output,                       \
      input,                             \
      weight,                            \
      kernel_size,                       \
      stride_size,                       \
      pad_size,                          \
      dilation_size,                     \
      columns,                           \
      ones)

#define INSERT_BATCH_DIMENSION(A, SIZE)        \
  if (A.numel() > 0) {                         \
    auto new_sizes = A.sizes().vec();          \
    new_sizes.insert(new_sizes.begin(), SIZE); \
    A.resize_(new_sizes);                      \
  }

#define DROP_BATCH_DIMENSION(A)    \
  if (A.numel() > 0) {             \
    A.resize_(A.sizes().slice(1)); \
  }

namespace at {
namespace native {

namespace {

inline bool all_positive(IntArrayRef& arr) {
  return std::all_of(
      arr.begin(), arr.end(), [](int64_t item) { return item > 0; });
}

#define OUTPUTSIZE(INDEX, DIM)                                       \
  (input.size((INDEX) + input.dim() - (DIM)) + 2 * pad_size[INDEX] - \
   (dilation_size[INDEX] * (kernel_size[INDEX] - 1) + 1)) /          \
          stride_size[INDEX] +                                       \
      1

void conv_dilated_shape_check(
    int64_t dim,
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
      kernel_size.size() == dim,
      "kernel sizes length should be ",
      dim,
      ", but got ",
      kernel_size.size());
  TORCH_CHECK(
      stride_size.size() == dim,
      "strides length should be ",
      dim,
      ", but got ",
      stride_size.size());
  TORCH_CHECK(
      dilation_size.size() == dim,
      "dilations length should be ",
      dim,
      ", but got ",
      dilation_size.size());
  TORCH_CHECK(
      pad_size.size() == dim,
      "pads length should be ",
      dim,
      ", but got ",
      pad_size.size());

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
        weight.dim() == dim + 2,
        "non-empty ",
        dim + 2,
        "D weight tensor (nOutputPlane, nInputPlane, ..., kH, kW) expected, "
        "but got ",
        weight.sizes());
    if (bias.numel() > 0) {
      TORCH_CHECK_DIM_SIZE(bias, 1, 0, weight.size(0));
      TORCH_CHECK(bias.is_contiguous(), "bias needs to be contiguous");
    }
  }

  if (grad_weight.numel() > 0) {
    TORCH_CHECK(
        grad_weight.dim() == dim + 2,
        "non-empty ",
        dim + 2,
        "D weight gradient tensor (nOutputPlane, nInputPlane, ..., kH, kW) expected, "
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

  // Since check shape is called after adding batch dimension, we
  // always have input.dim()==dim+2 and what follows could be
  // simplified..
  int ndim = input.dim();
  int dimf = 0;
  int dimd = dim - 2;
  int dimh = dim - 1;
  int dimw = dim;

  if (ndim == dim + 2) {
    dimd++;
    dimf++;
    dimh++;
    dimw++;
  }
  TORCH_CHECK(
      input.numel() > 0 && (ndim == dim + 2 || ndim == dim + 1),
      "non-empty ",
      dim + 1,
      "D or ",
      dim + 2,
      "D input tensor expected but got: ",
      input.sizes());

  switch (dim) {
    case 2: {
      int64_t inputHeight = input.size(dimh);
      int64_t inputWidth = input.size(dimw);
      int64_t outputHeight = OUTPUTSIZE(0, dim);
      int64_t outputWidth = OUTPUTSIZE(1, dim);
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
      if (grad_output.numel() > 0) {
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimh, outputHeight);
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimw, outputWidth);
      }
      if (ones.numel() > 0) {
        TORCH_CHECK(
            ones.numel() >= outputHeight * outputWidth,
            "expected at least ",
            outputHeight * outputWidth,
            " ones but got ",
            ones.sizes());
      }
    } break;
    case 3: {
      int64_t inputDepth = input.size(dimd);
      int64_t inputHeight = input.size(dimh);
      int64_t inputWidth = input.size(dimw);
      int64_t outputDepth = OUTPUTSIZE(0, dim);
      int64_t outputHeight = OUTPUTSIZE(1, dim);
      int64_t outputWidth = OUTPUTSIZE(2, dim);
      TORCH_CHECK(
          outputWidth >= 0 && outputHeight >= 0 && outputDepth >= 0,
          "Given input size per channel: (",
          inputDepth,
          " x ",
          inputHeight,
          " x ",
          inputWidth,
          "). "
          "Calculated output size per channel: (",
          outputDepth,
          " x ",
          outputHeight,
          " x ",
          outputWidth,
          "). Output size is too small");
      if (grad_output.numel() > 0) {
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimd, outputDepth);
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimh, outputHeight);
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimw, outputWidth);
      }

      if (ones.numel() > 0) {
        TORCH_CHECK(
            ones.numel() >= outputDepth * outputHeight * outputWidth,
            "expected at least ",
            outputDepth * outputHeight * outputWidth,
            " ones but got ",
            ones.sizes());
      }

    } break;
    default:
      TORCH_CHECK(false, "unexpected dim in conv_dilate shape check", dim);
  } // switch
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
  }

  if (columns.numel() > 0) {
    TORCH_CHECK(columns.is_contiguous(), "columns needs to be contiguous");
  }

} // conv_dilated_shape_check

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
    INSERT_BATCH_DIMENSION(input, 1);
    INSERT_BATCH_DIMENSION(output, 1);
    INSERT_BATCH_DIMENSION(grad_input, 1);
    INSERT_BATCH_DIMENSION(grad_output, 1);
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
  int64_t outputHeight = OUTPUTSIZE(0, 2);
  int64_t outputWidth = OUTPUTSIZE(1, 2);

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
  conv_dilated_shape_check(
      2,
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
          // For gemm argument derivation, see
          // conv_dilated2d_all_cuda_template in
          // ATen/native/cuda/DilatedConvolution.cu
          THBlas_gemm<scalar_t>(
              /*transa=*/'t',
              /*transb=*/'n',
              /*     m=*/outputHeight * outputWidth,
              /*     n=*/nOutputPlane,
              /*     k=*/1,
              /* alpha=*/1,
              /*     A=*/ones.data<scalar_t>(),
              /*   lda=*/1,
              /*     B=*/bias.data<scalar_t>(),
              /*   ldb=*/1,
              /*  beta=*/0,
              /*     C=*/output_n.data<scalar_t>(),
              /*   ldc=*/outputHeight * outputWidth);
        } else {
          output_n.zero_();
        }
        // Extract columns:
        im2col<scalar_t>(
            input_n.data<scalar_t>(),
            nInputPlane,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            kH,
            kW,
            padH,
            padW,
            dH,
            dW,
            dilationH,
            dilationW,
            columns.data<scalar_t>());
        
        // For gemm argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemm<scalar_t>(
            /*transa=*/'n',
            /*transb=*/'n',
            /*     m=*/columns.size(1),
            /*     n=*/nOutputPlane,
            /*     k=*/columns.size(0),
            /* alpha=*/1,
            /*     A=*/columns.data<scalar_t>(),
            /*   lda=*/columns.size(1),
            /*     B=*/weight.data<scalar_t>(),
            /*   ldb=*/columns.size(0),
            /*  beta=*/1,
            /*     C=*/output_n.data<scalar_t>(),
            /*   ldc=*/columns.size(1));
      } else {
        // All gradients
        grad_output_n = grad_output.select(0, elt);
      }

      // Gradient of input:
      if (grad_input.numel() > 0) {
        // For gemm argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemm<scalar_t>(
            /*transa=*/'n',
            /*transb=*/'t',
            /*     m=*/columns.size(1),
            /*     n=*/columns.size(0),
            /*     k=*/nOutputPlane,
            /* alpha=*/1,
            /*     A=*/grad_output_n.data<scalar_t>(), // op(A) is m x k
            /*   lda=*/columns.size(1),
            /*     B=*/weight.data<scalar_t>(), // op(B) is k x n
            /*   ldb=*/columns.size(0),
            /*  beta=*/0,
            /*     C=*/columns.data<scalar_t>(), // C is m x n
            /*   ldc=*/columns.size(1));
        // Unpack columns back into input:
        grad_input_n = grad_input.select(0, elt);
        col2im<scalar_t>(
            columns.data<scalar_t>(),
            nInputPlane,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            kH,
            kW,
            padH,
            padW,
            dH,
            dW,
            dilationH,
            dilationW,
            grad_input_n.data<scalar_t>());
      }

      // Gradient of weight:
      if (grad_weight.numel() > 0) {
        scalar_t scale = 1; // TODO: expose as argument?
        // Extract columns:
        im2col<scalar_t>(
            input_n.data<scalar_t>(),
            nInputPlane,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            kH,
            kW,
            padH,
            padW,
            dH,
            dW,
            dilationH,
            dilationW,
            columns.data<scalar_t>());
        // For gemm argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemm<scalar_t>(
            /*transa=*/'t',
            /*transb=*/'n',
            /*     m=*/columns.size(0),
            /*     n=*/nOutputPlane,
            /*     k=*/columns.size(1),
            /* alpha=*/scale,
            /*     A=*/columns.data<scalar_t>(),
            /*   lda=*/columns.size(1),
            /*     B=*/grad_output_n.data<scalar_t>(),
            /*   ldb=*/columns.size(1),
            /*  beta=*/1,
            /*     C=*/grad_weight.data<scalar_t>(),
            /*   ldc=*/columns.size(0));
      }

      // Gradient of bias:
      if (grad_bias.numel() > 0) {
        scalar_t scale = 1; // TODO: expose as argument?
        // For gemm argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemv<scalar_t>(
            /* trans=*/'t',
            /*     m=*/outputHeight * outputWidth,
            /*     n=*/nOutputPlane,
            /* alpha=*/scale,
            /*     A=*/grad_output_n.data<scalar_t>(),
            /*   lda=*/outputHeight * outputWidth,
            /*     x=*/ones.data<scalar_t>(),
            /*  incx=*/1,
            /*  beta=*/1,
            /*     y=*/grad_bias.data<scalar_t>(),
            /*  incy=*/1);
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

void conv_dilated3d_all_cpu_template(
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

  bool is_batch = input.dim() == 4;
  if (is_batch) {
    INSERT_BATCH_DIMENSION(input, 1);
    INSERT_BATCH_DIMENSION(output, 1);
    INSERT_BATCH_DIMENSION(grad_input, 1);
    INSERT_BATCH_DIMENSION(grad_output, 1);
  }

  auto kD = kernel_size[0];
  auto kH = kernel_size[1];
  auto kW = kernel_size[2];
  auto dD = stride_size[0];
  auto dH = stride_size[1];
  auto dW = stride_size[2];
  auto padD = pad_size[0];
  auto padH = pad_size[1];
  auto padW = pad_size[2];
  auto dilationD = dilation_size[0];
  auto dilationH = dilation_size[1];
  auto dilationW = dilation_size[2];

  int64_t batchSize = input.size(0);
  int64_t nInputPlane = weight.size(1);
  int64_t nOutputPlane = weight.size(0);
  int64_t inputDepth = input.size(2);
  int64_t inputHeight = input.size(3);
  int64_t inputWidth = input.size(4);
  int64_t outputDepth = OUTPUTSIZE(0, 3);
  int64_t outputHeight = OUTPUTSIZE(1, 3);
  int64_t outputWidth = OUTPUTSIZE(2, 3);

  // Resize temporary columns
  if (output.numel() > 0 || grad_weight.numel() > 0 || grad_input.numel() > 0) {
    columns.resize_(
        {nInputPlane * kW * kH * kD, outputDepth * outputHeight * outputWidth});
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
    ones.resize_({outputDepth, outputHeight, outputWidth});
    ones.fill_(1);
  }

  // checking shapes after all shapes have been settled:
  conv_dilated_shape_check(
      3,
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

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_dilated3d", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt++) {
      // Matrix multiply per output:
      input_n = input.select(0, elt);

      // Output
      if (output.numel() > 0) {
        output_n = output.select(0, elt);
        if (bias.numel() > 0) {
          // For gemm argument derivation, see
          // conv_dilated2d_all_cuda_template in
          // ATen/native/cuda/DilatedConvolution.cu
          THBlas_gemm<scalar_t>(
              /*transa=*/'t',
              /*transb=*/'n',
              /*     m=*/outputDepth * outputHeight * outputWidth,
              /*     n=*/nOutputPlane,
              /*     k=*/1,
              /* alpha=*/1,
              /*     A=*/ones.data<scalar_t>(),
              /*   lda=*/1,
              /*     B=*/bias.data<scalar_t>(),
              /*   ldb=*/1,
              /*  beta=*/0,
              /*     C=*/output_n.data<scalar_t>(),
              /*   ldc=*/outputDepth * outputHeight * outputWidth);
        } else {
          output_n.zero_();
        }
        // Extract columns:
        vol2col<scalar_t>(
            input_n.data<scalar_t>(),
            nInputPlane,
            inputDepth,
            inputHeight,
            inputWidth,
            outputDepth,
            outputHeight,
            outputWidth,
            kD,
            kH,
            kW,
            padD,
            padH,
            padW,
            dD,
            dH,
            dW,
            dilationD,
            dilationH,
            dilationW,
            columns.data<scalar_t>());
        // For gemm argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemm<scalar_t>(
            /*transa=*/'n',
            /*transb=*/'n',
            /*     m=*/columns.size(1),
            /*     n=*/nOutputPlane,
            /*     k=*/columns.size(0),
            /* alpha=*/1,
            /*     A=*/columns.data<scalar_t>(),
            /*   lda=*/columns.size(1),
            /*     B=*/weight.data<scalar_t>(),
            /*   ldb=*/columns.size(0),
            /*  beta=*/1,
            /*     C=*/output_n.data<scalar_t>(),
            /*   ldc=*/columns.size(1));
      } else {
        // All gradients
        grad_output_n = grad_output.select(0, elt);
      }

      // Gradient of input:
      if (grad_input.numel() > 0) {
        // For gemm argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemm<scalar_t>(
            /*transa=*/'n',
            /*transb=*/'t',
            /*     m=*/columns.size(1),
            /*     n=*/columns.size(0),
            /*     k=*/nOutputPlane,
            /* alpha=*/1,
            /*     A=*/grad_output_n.data<scalar_t>(),
            /*   lda=*/columns.size(1),
            /*     B=*/weight.data<scalar_t>(),
            /*   ldb=*/columns.size(0),
            /*  beta=*/0,
            /*     C=*/columns.data<scalar_t>(),
            /*   ldc=*/columns.size(1));

        // Unpack columns back into input:
        grad_input_n = grad_input.select(0, elt);
        col2vol<scalar_t>(
            columns.data<scalar_t>(),
            nInputPlane,
            inputDepth,
            inputHeight,
            inputWidth,
            outputDepth,
            outputHeight,
            outputWidth,
            kD,
            kH,
            kW,
            padD,
            padH,
            padW,
            dD,
            dH,
            dW,
            dilationD,
            dilationH,
            dilationW,
            grad_input_n.data<scalar_t>());
      }

      // Gradient of weight:
      if (grad_weight.numel() > 0) {
        scalar_t scale = 1; // TODO: expose as argument?
        // Extract columns:
        vol2col<scalar_t>(
            input_n.data<scalar_t>(),
            nInputPlane,
            inputDepth,
            inputHeight,
            inputWidth,
            outputDepth,
            outputHeight,
            outputWidth,
            kD,
            kH,
            kW,
            padD,
            padH,
            padW,
            dD,
            dH,
            dW,
            dilationD,
            dilationH,
            dilationW,
            columns.data<scalar_t>());
        // For gemm argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemm<scalar_t>(
            /*transa=*/'t',
            /*transb=*/'n',
            /*     m=*/columns.size(0),
            /*     n=*/nOutputPlane,
            /*     k=*/columns.size(1),
            /* alpha=*/scale,
            /*     A=*/columns.data<scalar_t>(),
            /*   lda=*/columns.size(1),
            /*     B=*/grad_output_n.data<scalar_t>(),
            /*   ldb=*/columns.size(1),
            /*  beta=*/1,
            /*     C=*/grad_weight.data<scalar_t>(),
            /*   ldc=*/columns.size(0));
      }

      // Gradient of bias:
      if (grad_bias.numel() > 0) {
        scalar_t scale = 1; // TODO: expose as argument?
        // For gemv argument derivation, see
        // conv_dilated2d_all_cuda_template in
        // ATen/native/cuda/DilatedConvolution.cu
        THBlas_gemv<scalar_t>(
            /* trans=*/'t',
            /*     m=*/outputDepth * outputHeight * outputWidth,
            /*     n=*/nOutputPlane,
            /* alpha=*/scale,
            /*     A=*/grad_output_n.data<scalar_t>(),
            /*   lda=*/outputDepth * outputHeight * outputWidth,
            /*     x=*/ones.data<scalar_t>(),
            /*  incx=*/1,
            /*  beta=*/1,
            /*     y=*/grad_bias.data<scalar_t>(),
            /*  incy=*/1);
      }
    }
  });
  if (is_batch) {
    DROP_BATCH_DIMENSION(input);
    DROP_BATCH_DIMENSION(output);
    DROP_BATCH_DIMENSION(grad_input);
    DROP_BATCH_DIMENSION(grad_output);
  }

} // conv_dilated3d_all_cpu_template

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
  Tensor grad_output = at::empty({0}, options); // not used
  Tensor grad_input = at::empty({0}, options); // not used
  Tensor grad_weight = at::empty({0}, options); // not used
  Tensor grad_bias = at::empty({0}, options); // not used
  int64_t nOutputPlane = weight.size(0);
  int64_t outputHeight = OUTPUTSIZE(0, 2);
  int64_t outputWidth = OUTPUTSIZE(1, 2);
  output.resize_({nOutputPlane, outputHeight, outputWidth});
  if (input.dim() == 4) {
    INSERT_BATCH_DIMENSION(output, input.size(0));
  }
  CALL_TEMPLATE(2);
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
  CALL_OUT(2);
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
  CALL_OUT(2);
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
  CALL_OUT(2); // Is this correct??
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
  CALL_FORWARD_OUT(2);
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
  CALL_FORWARD_OUT(2);
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
  CALL_TEMPLATE(2);
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
  CALL_BACKWARD_OUT(2);
  return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated3d_out_cpu(
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
  Tensor grad_output = at::empty({0}, options); // not used
  Tensor grad_input = at::empty({0}, options); // not used
  Tensor grad_weight = at::empty({0}, options); // not used
  Tensor grad_bias = at::empty({0}, options); // not used
  int64_t nOutputPlane = weight.size(0);
  int64_t outputDepth = OUTPUTSIZE(0, 3);
  int64_t outputHeight = OUTPUTSIZE(1, 3);
  int64_t outputWidth = OUTPUTSIZE(2, 3);
  output.resize_({nOutputPlane, outputDepth, outputHeight, outputWidth});
  if (input.dim() == 5) {
    INSERT_BATCH_DIMENSION(output, input.size(0));
  }
  CALL_TEMPLATE(3);
  return std::tie(output, columns, ones);
}

Tensor& conv_dilated3d_out_cpu(
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
  CALL_OUT(3);
  return output;
}

Tensor conv_dilated3d_cpu(
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
  CALL_OUT(3);
  return output;
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated3d_forward_out_cpu(
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
  std::cout << "NOT IMPLEMENTED: conv_dilated3d_forward_out_cpu3" << std::endl;
  CALL_OUT(3); // Is this correct??
  return std::tie(output, columns, ones);
}

Tensor& conv_dilated3d_forward_out_cpu(
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
  std::cout << "NOT IMPLEMENTED: conv_dilated3d_forward_out_cpu1" << std::endl;
  CALL_FORWARD_OUT(3);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated3d_forward_cpu(
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
  std::cout << "NOT IMPLEMENTED: conv_dilated3d_forward_cpu" << std::endl;
  CALL_FORWARD_OUT(3);
  return std::tie(output, columns, ones);
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated3d_backward_out_cpu(
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
  CALL_TEMPLATE(3);
  return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated3d_backward_cpu(
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
  CALL_BACKWARD_OUT(3);
  return std::tie(grad_input, grad_weight, grad_bias);
}

} // namespace native
} // namespace at
