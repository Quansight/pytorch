
#include <ATen/div_rtn.h>
#include <algorithm>
#include <tuple>
#include "ATen/ATen.h"

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCNumerics.cuh>
#include "THCUNN/im2col.h"
#include "THCUNN/vol2col.h"

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

#define CALL_TEMPLATE(DIM)                \
  conv_dilated##DIM##d_all_cuda_template( \
      output,                             \
      input_,                              \
      weight_,                             \
      bias_,                               \
      grad_output_,                        \
      grad_input,                         \
      grad_weight,                        \
      grad_bias,                          \
      kernel_size,                        \
      stride_size,                        \
      pad_size,                           \
      dilation_size,                      \
      columns,                            \
      ones)

#define CALL_OUT(DIM)            \
  conv_dilated##DIM##d_out_cuda( \
      output,                    \
      columns,                   \
      ones,                      \
      input,                     \
      weight,                    \
      kernel_size,               \
      bias,                      \
      stride_size,               \
      pad_size,                  \
      dilation_size)

#define CALL_FORWARD_OUT(DIM)            \
  conv_dilated##DIM##d_forward_out_cuda( \
      output,                            \
      columns,                           \
      ones,                              \
      input,                             \
      weight,                            \
      kernel_size,                       \
      bias,                              \
      stride_size,                       \
      pad_size,                          \
      dilation_size)

#define CALL_BACKWARD_OUT(DIM)            \
  conv_dilated##DIM##d_backward_out_cuda( \
      grad_input,                         \
      grad_weight,                        \
      grad_bias,                          \
      grad_output,                        \
      input,                              \
      weight,                             \
      kernel_size,                        \
      stride_size,                        \
      pad_size,                           \
      dilation_size,                      \
      columns,                            \
      ones)

#define INSERT_BATCH_DIMENSION(A, SIZE)        \
  if (A.defined()) {                           \
    auto new_sizes = A.sizes().vec();          \
    new_sizes.insert(new_sizes.begin(), SIZE); \
    A.resize_(new_sizes);                      \
  }

#define DROP_BATCH_DIMENSION(A)    \
  if (A.defined()) {               \
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
    const Tensor& grad_input,
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

  if (bias.defined()) {
    TORCH_CHECK_DIM_SIZE(bias, 1, 0, weight.size(0));
    TORCH_CHECK(bias.is_contiguous(), "bias ought to be contiguous");
  }

  if (grad_weight.defined()) {
    TORCH_CHECK(
        grad_weight.dim() == dim + 2,
        "non-empty ",
        dim + 2,
        "D weight gradient tensor (nOutputPlane, nInputPlane, ..., kH, kW) expected, "
        "but got ",
        grad_weight.sizes());
    TORCH_CHECK(
        grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
    if (grad_bias.defined()) {
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
      (ndim == dim + 2 || ndim == dim + 1),
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
      if (grad_output.defined()) {
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimh, outputHeight);
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimw, outputWidth);
      }
      if (bias.defined() || grad_bias.defined()) {
        TORCH_CHECK(
            ones.defined() && ones.numel() >= outputHeight * outputWidth,
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
      if (grad_output.defined()) {
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimd, outputDepth);
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimh, outputHeight);
        TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimw, outputWidth);
      }

      if (bias.defined() || grad_bias.defined()) {
        TORCH_CHECK(
            ones.defined() &&
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

  TORCH_CHECK_DIM_SIZE(input, ndim, dimf, weight.size(1));

  if (grad_output.defined()) {
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output ought to be contiguous");
    TORCH_CHECK_DIM_SIZE(grad_output, ndim, dimf, weight.size(0));
  }

  if (grad_input.defined() || grad_weight.defined() || grad_bias.defined()) {
    TORCH_CHECK(grad_output.defined(), "grad_output must be defined for gradients");
  }

  if (columns.defined()) {
    TORCH_CHECK(columns.is_contiguous(), "columns needs to be contiguous");
  }

} // conv_dilated_shape_check

/*
  conv_dilated2d_all_cuda_template

  Main worker. Computes tensors output, grad_input, grad_weight,
  and/or grad_bias if non-empty.
 */
void conv_dilated2d_all_cuda_template(
    Tensor& output,
    const Tensor& input_,
    const Tensor& weight,
    const Tensor& bias,
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
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Tensor input = input_;
  Tensor grad_output = grad_output_;
  Tensor columns = columns_;
  Tensor ones = ones_;

  // Preliminary shape checks
  TORCH_CHECK(input.defined(), "input must be defined");
  TORCH_CHECK(weight.defined(), "weight must be defined");
  TORCH_CHECK(weight.dim() == 4, "weight must be 4D tensor but got one with shape ", weight.sizes());

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
  if (output.defined() || grad_weight.defined() || grad_input.defined()) {
    columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});
  }
  if (grad_weight.defined()) {
    grad_weight.zero_();
  }
  if (grad_bias.defined()) {
    grad_bias.zero_();
  }
  // Resize temporary ones
  if (bias.defined() || grad_bias.defined()) {
    // Define a buffer of ones, for bias accumulation
    ones.resize_({outputHeight, outputWidth});
    ones.fill_(1);
  }
  // checking data locations
  TensorArg output_arg{output, "output", 1}, input_arg{input, "input", 2},
      weight_arg{weight, "weight", 3}, bias_arg{bias, "bias", 4},
      grad_output_arg{grad_output, "grad_output", 5},
      grad_input_arg{grad_input, "grad_input", 6},
      grad_weight_arg{grad_weight, "grad_weight", 7},
      grad_bias_arg{grad_bias, "grad_bias", 8},
      columns_arg{columns, "columns", 13}, ones_arg{ones, "ones", 14};
  if (output.defined()) {
    checkAllSameGPU(
        "conv_dilated2d_all_cuda_template",
        {input_arg, output_arg, weight_arg, columns_arg, ones_arg});
    if (bias.defined()) {
      checkAllSameGPU(
          "conv_dilated2d_all_cuda_template", {weight_arg, bias_arg});
    }
  }
  if (grad_input.defined()) {
    checkAllSameGPU(
        "conv_dilated2d_all_cuda_template",
        {grad_input_arg, grad_output_arg, weight_arg, columns_arg});
  }
  if (grad_weight.defined()) {
    checkAllSameGPU(
        "conv_dilated2d_all_cuda_template",
        {input_arg,
         grad_output_arg,
         grad_weight_arg,
         columns_arg,
         grad_input_arg});
  }
  if (grad_bias.defined()) {
    checkAllSameGPU(
        "conv_dilated2d_all_cuda_template",
        {grad_bias_arg, grad_output_arg, weight_arg, ones_arg});
  }
  // checking shapes after all shapes have been settled:
  conv_dilated_shape_check(
      2,
      input,
      weight,
      bias,
      grad_output,
      grad_input,
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "conv_dilated2d", [&] {
        // For each elt in batch, do:
        for (int elt = 0; elt < batchSize; elt++) {
          // Matrix multiply per output:
          input_n = input.select(0, elt);

          // Output
          if (output.defined()) {
            output_n = output.select(0, elt);
            if (bias.defined()) {
              /*
                Compute:

                  output_n = bias * ones^T

                where

                  bias is viewed as bias.view(nOutputPlane, 1)

                  ones is viewed as ones.view(outputHeight * outputWidth, 1)

                  output_n is viewed as output_n.view(nOutputPlane, outputHeight
              * outputWidth)

              gemm assumes column-major matrices:

                output_n^T = ones * bias^T
                C = alpha * op(A) * op(B)
                op(A) = 't', op(B) = 'n', alpha=1, beta=0
              */
              at::cuda::blas::gemm<scalar_t>(
                  stream,
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
                stream,
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

            /*
              Compute:

                output_n = weight * columns + output_n

              where

                weight is viewed as weight.view(nOutputPlane, nInputPlane * kD *
              kH * kW)

                columns size is (nInputPlane * kH * kW) x (outputHeight *
              outputWidth)

                output_n is viewed as output_n.view(nOutputPlane, outputHeight *
              outputWidth)

              gemm assumes column-major matrices:

                output_n^T = columns^T * weight^T + output_n^T
                C = alpha * op(A) * op(B) + beta * C
                op(A) = 'n', op(B) = 'n', alpha=1, beta=1
            */
            at::cuda::blas::gemm<scalar_t>(
                stream,
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
          if (grad_input.defined()) {
            /*
              Compute:

                columns = weight^T * grad_output_n

              where

                weight is viewed as weight.view(nOutputPlane, nInputPlane * kH *
              kW)

                grad_output_n is viewed as grad_output_n.view(nOutputPlane,
              outputHeight * outputWidth)

                columns size is (nInputPlane * kH * kW) x (outputHeight *
              outputWidth)

              gemm assumes column-major matrices:

                columns^T = grad_output_n^T * weight
                C = alpha * op(A) * op(B) + beta * C
                op(A) = 'n', op(B) = 't', alpha=1, beta=0
             */
            at::cuda::blas::gemm<scalar_t>(
                stream,
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
            col2im<scalar_t, scalar_t>(
                stream,
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
          if (grad_weight.defined()) {
            scalar_t scale = 1; // TODO: expose as argument?
            // Extract columns:
            im2col<scalar_t>(
                stream,
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

            /*
              Compute:

                grad_weight = scale * grad_output_n * columns^T + grad_weight

              where

                grad_output_n is viewed as grad_output_n.view(nOutputPlane,
              outputHeight * outputWidth)

                columns size is (nInputPlane * kD * kH * kW) x (outputHeight *
              outputWidth)

                grad_weight is viewed as grad_weight.view(nOutputPlane,
              nInputPlane * kH * kW)

              gemm assumes column-major matrices:

                grad_weight^T = scale * columns * grad_output_n^T +
              grad_weight^T C = alpha * op(A) * op(B) + beta * C op(A) = 't',
              op(B) = 'n', alpha=scale, beta=1
            */
            at::cuda::blas::gemm<scalar_t>(
                stream,
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
          if (grad_bias.defined()) {
            scalar_t scale = 1; // TODO: expose as argument
            /*
              Compute:
                grad_bias = scale * grad_output_n * ones + grad_bias

              where

                grad_bias is viewed as grad_bias.view(nOutputPlane, 1)

                ones is viewed as ones.view(outputHeight * outputWidth, 1)

                grad_output_n is viewed as grad_output_n.view(nOutputPlane,
              outputHeight * outputWidth)

              gemm assumes column-major matrices:

                grad_bias^T = scale * grad_output_n * ones + grad_bias^T
                y = alpha * op(A) * x + beta * y
                op(A) = 't', alpha=scale, beta=1
             */
            at::cuda::blas::gemv<scalar_t>(
                stream,
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

} // conv_dilated2d_all_cuda_template

void conv_dilated3d_all_cuda_template(
    Tensor& output,
    const Tensor& input_,
    const Tensor& weight,
    const Tensor& bias,
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
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Tensor input = input_;
  Tensor grad_output = grad_output_;
  Tensor columns = columns_;
  Tensor ones = ones_;

  // Preliminary shape checks
  TORCH_CHECK(input.defined(), "input must be defined");
  TORCH_CHECK(weight.defined(), "weight must be defined");
  TORCH_CHECK(weight.dim() == 5, "weight must be 5D tensor but got one with shape ", weight.sizes());

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
  if (output.defined() || grad_weight.defined() || grad_input.defined()) {
    columns.resize_(
        {nInputPlane * kW * kH * kD, outputDepth * outputHeight * outputWidth});
  }
  if (grad_weight.defined()) {
    grad_weight.zero_();
  }
  if (grad_bias.defined()) {
    grad_bias.zero_();
  }
  // Resize temporary ones
  if (bias.defined() || grad_bias.defined()) {
    // Define a buffer of ones, for bias accumulation
    ones.resize_({outputDepth, outputHeight, outputWidth});
    ones.fill_(1);
  }
  // checking data locations
  TensorArg output_arg{output, "output", 1}, input_arg{input, "input", 2},
      weight_arg{weight, "weight", 3}, bias_arg{bias, "bias", 4},
      grad_output_arg{grad_output, "grad_output", 5},
      grad_input_arg{grad_input, "grad_input", 6},
      grad_weight_arg{grad_weight, "grad_weight", 7},
      grad_bias_arg{grad_bias, "grad_bias", 8},
      columns_arg{columns, "columns", 13}, ones_arg{ones, "ones", 14};
  if (output.defined()) {
    checkAllSameGPU(
        "conv_dilated3d_all_cuda_template",
        {input_arg, output_arg, weight_arg, columns_arg, ones_arg});
    if (bias.defined()) {
      checkAllSameGPU(
          "conv_dilated3d_all_cuda_template", {weight_arg, bias_arg});
    }
  }
  if (grad_input.defined()) {
    checkAllSameGPU(
        "conv_dilated3d_all_cuda_template",
        {grad_input_arg, grad_output_arg, weight_arg, columns_arg});
  }
  if (grad_weight.defined()) {
    checkAllSameGPU(
        "conv_dilated3d_all_cuda_template",
        {input_arg,
         grad_output_arg,
         grad_weight_arg,
         columns_arg,
         grad_input_arg});
  }
  if (grad_bias.defined()) {
    checkAllSameGPU(
        "conv_dilated3d_all_cuda_template",
        {grad_bias_arg, grad_output_arg, weight_arg, ones_arg});
  }
  // checking shapes after all shapes have been settled:
  conv_dilated_shape_check(
      3,
      input,
      weight,
      bias,
      grad_output,
      grad_input,
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "conv_dilated3d", [&] {
        // For each elt in batch, do:
        for (int elt = 0; elt < batchSize; elt++) {
          // Matrix multiply per output:
          input_n = input.select(0, elt);

          // Output
          if (output.defined()) {
            output_n = output.select(0, elt);
            if (bias.defined()) {
              // Tensor is based on row-major ordering, but gemm
              // assumes column-major matrices, hence the choise of
              // trans operations and swapped sizes:
              at::cuda::blas::gemm<scalar_t>(
                  stream,
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
                stream,
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

            // Tensor is based on row-major ordering, but gemm
            // assumes column-major matrices, hence the choise of
            // trans operations and swapped sizes:
            at::cuda::blas::gemm<scalar_t>(
                stream,
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
          if (grad_input.defined()) {
            // Tensor is based on row-major ordering, but gemm
            // assumes column-major matrices, hence the choise of
            // trans operations and swapped sizes:
            at::cuda::blas::gemm<scalar_t>(
                stream,
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
            col2vol<scalar_t, scalar_t>(
                stream,
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
          if (grad_weight.defined()) {
            // Extract columns:
            vol2col<scalar_t>(
                stream,
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

            scalar_t scale = 1; // TODO: expose as argument?
            // Tensor is based on row-major ordering, but gemm
            // assumes column-major matrices, hence the choise of
            // trans operations and swapped sizes:
            at::cuda::blas::gemm<scalar_t>(
                stream,
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
          if (grad_bias.defined()) {
            scalar_t scale = 1; // TODO: expose as argument?
            // Tensor is based on row-major ordering, but gemv
            // assumes column-major matrices, hence the choise of
            // trans operations and swapped sizes:
            at::cuda::blas::gemv<scalar_t>(
                stream,
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

} // conv_dilated3d_all_cuda_template

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated2d_out_cuda(
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
  const Tensor weight_ = weight.contiguous();
  const Tensor input_ = input.contiguous();
  const Tensor bias_ = bias.contiguous();
  Tensor grad_output_;
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
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

Tensor& conv_dilated2d_out_cuda(
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

Tensor conv_dilated2d_cuda(
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

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated2d_forward_out_cuda(
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
  // Is this function dead?
  TORCH_CHECK(false, "Error conv_dilated2d_forward_out_cuda 3");
  CALL_OUT(2);
  return std::tie(output, columns, ones);
}

Tensor& conv_dilated2d_forward_out_cuda(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // Is this function dead?
  TORCH_CHECK(false, "Error conv_dilated2d_forward_out_cuda 1");
  auto options = output.options();
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  CALL_FORWARD_OUT(2);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated2d_forward_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // Is this function dead?
  TORCH_CHECK(false, "Error conv_dilated2d_forward_cuda");
  auto options = input.options();
  Tensor output = at::empty({0}, options);
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  CALL_FORWARD_OUT(2);
  return std::tie(output, columns, ones);
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated2d_backward_out_cuda(
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
  Tensor output;
  Tensor bias_;
  const Tensor grad_output_ = grad_output.contiguous();
  const Tensor input_ = input.contiguous();
  const Tensor weight_ = weight.contiguous();
  grad_input.resize_(input.sizes());
  grad_weight.resize_(weight.sizes());
  grad_bias.resize_(weight.size(0));
  CALL_TEMPLATE(2);
  return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated2d_backward_cuda(
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

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated3d_out_cuda(
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
  const Tensor input_ = input.contiguous();
  const Tensor bias_ = bias.contiguous();
  const Tensor weight_ = weight.contiguous();
  Tensor grad_output_;
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
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

Tensor& conv_dilated3d_out_cuda(
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

Tensor conv_dilated3d_cuda(
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

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated3d_forward_out_cuda(
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
  // Is this function dead?
  TORCH_CHECK(false, "Error conv_dilated3d_forward_out_cuda 3");
  CALL_OUT(3);
  return std::tie(output, columns, ones);
}

Tensor& conv_dilated3d_forward_out_cuda(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // Is this function dead?
  TORCH_CHECK(false, "Error conv_dilated3d_forward_out_cuda 1");
  auto options = output.options();
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  CALL_FORWARD_OUT(3);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated3d_forward_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // Is this function dead?
  TORCH_CHECK(false, "Error conv_dilated3d_forward_cuda 3");
  auto options = input.options();
  Tensor output = at::empty({0}, options);
  Tensor columns = at::empty({0}, options);
  Tensor ones = at::empty({0}, options);
  CALL_FORWARD_OUT(3);
  return std::tie(output, columns, ones);
}

std::tuple<Tensor&, Tensor&, Tensor&> conv_dilated3d_backward_out_cuda(
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
  Tensor output;
  Tensor bias_;
  const Tensor grad_output_ = grad_output.contiguous();
  const Tensor input_ = input.contiguous();
  const Tensor weight_ = weight.contiguous();
  grad_input.resize_(input.sizes());
  grad_weight.resize_(weight.sizes());
  grad_bias.resize_(weight.size(0));
  CALL_TEMPLATE(3);
  return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> conv_dilated3d_backward_cuda(
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
