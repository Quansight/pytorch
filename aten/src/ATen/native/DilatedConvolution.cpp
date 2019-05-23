
#include <tuple>
#include <algorithm>
#include "ATen/ATen.h"
//#include "ATen/NativeFunctions.h"
//#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/div_rtn.h>
#include "TH/THBlasUtils.h"

#define AT_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE)                       \
  AT_CHECK(T.dim()==DIM && T.size(DIM_SIZE)==SIZE,                      \
           "Need " #T " of dimension ",DIM," and " #T ".size[",DIM_SIZE,"] == ",SIZE, \
           " but got input to be of shape ", T.sizes());

namespace at {
namespace native {

namespace {

  inline bool all_positive(IntArrayRef& arr) {
    return std::all_of(arr.begin(), arr.end(), [](int64_t item) { return item>0; });
  }
  
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
                                const Tensor& ones,
                                const Tensor& grad_columns
                                ) {
  AT_CHECK(kernel_size.size() == 2,
    "kernel sizes length should be two, but got ", kernel_size);
  AT_CHECK(stride_size.size() == 2,
    "strides length should be two, but got ", stride_size);
  AT_CHECK(dilation_size.size() == 2,
    "dilations length should be two, but got ", dilation_size);
  AT_CHECK(pad_size.size() == 2,
    "pads length should be two, but got ", pad_size);

  AT_CHECK(all_positive(kernel_size),
           "kernel size should be greater than zero, but got ", kernel_size);
  AT_CHECK(all_positive(stride_size),
           "stride should be greater than zero, but got ", stride_size);
  AT_CHECK(all_positive(dilation_size),
           "dilation should be greater than zero, but got ", dilation_size);
  
  if (weight.numel() > 0) {
    AT_CHECK(weight.dim()==4,
             "non-empty 4D weight tensor (nOutputPlane, nInputPlane, kH, kW) expected, "
             "but got ", weight.sizes());
    if (bias.numel() > 0) {
      AT_CHECK_DIM_SIZE(bias, 1, 0, weight.size(0));
    }
  }

  if (grad_weight.numel() > 0) {
    AT_CHECK(grad_weight.dim()==4,
             "non-empty 4D weight gradient tensor (nOutputPlane, nInputPlane, kH, kW) expected, "
             "but got ", grad_weight.sizes());
    AT_CHECK(grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
    if (grad_bias.numel() > 0) {
      AT_CHECK_DIM_SIZE(grad_bias, 1, 0, grad_weight.size(0));
      AT_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
      AT_CHECK(ones.is_contiguous(), "ones needs to be contiguous");
    }
  }

  if (grad_columns.numel() > 0) {
    AT_CHECK(grad_columns.is_contiguous(), "grad_columns needs to be contiguous");
  }

  int ndim = input.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }
  AT_CHECK(input.numel() > 0 && (ndim==4 || ndim==3),
           "non-empty 3D or 4D input tensor expected but got: ",
           input.sizes()
           );

  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  auto dilationH = dilation_size[0];
  auto dilationW = dilation_size[1];
  
  int64_t inputHeight  = input.size(dimh);
  int64_t inputWidth   = input.size(dimw);
  int64_t outputHeight = div_rtn<int64_t>(inputHeight + 2*padH - (dilationH * (kH - 1) + 1), dH) + 1;
  int64_t outputWidth  = div_rtn<int64_t>(inputWidth + 2*padW - (dilationW * (kW - 1) + 1), dW) + 1;
  AT_CHECK(outputWidth>=0 && outputHeight>=0,
           "Given input size per channel: (",inputHeight," x ",inputWidth,"). "
           "Calculated output size per channel: (",outputHeight," x ",outputWidth,
           "). Output size is too small");

  if (weight.numel() > 1) {
    int64_t nInputPlane = weight.size(1);
    AT_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  }

  if (grad_output.numel() > 0) {
    if (weight.numel() > 0) {
      int64_t nOutputPlane = weight.size(0);
      AT_CHECK_DIM_SIZE(grad_output, ndim, dimf, nOutputPlane);
    } else if (bias.numel() > 0) {
      int64_t nOutputPlane = bias.size(0);
      AT_CHECK_DIM_SIZE(grad_output, ndim, dimf, nOutputPlane);
    }
    AT_CHECK_DIM_SIZE(grad_output, ndim, dimh, outputHeight);
    AT_CHECK_DIM_SIZE(grad_output, ndim, dimw, outputWidth);
  }

  AT_CHECK(columns.is_contiguous(), "columns needs to be contiguous");

  if (ones.numel() > 0) {
    AT_CHECK(ones.numel() >= outputHeight * outputWidth,
             "expected at least ", outputHeight * outputWidth,
             " ones but got ", ones.sizes());
  }
}

void conv_dilated_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& columns,
    Tensor& ones,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  std::cout << "conv_dilated_cpu_template" << std::endl;

  conv_dilated2d_shape_check(input, weight, bias,
                             at::empty({0}), at::empty({0}), at::empty({0}),  /* grad_output, grad_weight, grad_bias */
                             kernel_size, stride_size, pad_size, dilation_size, columns, ones,
                             at::empty({0})  /* grad_columns */);

  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  auto dilationH = dilation_size[0];
  auto dilationW = dilation_size[1];

  // Params:
  int nInputPlane = weight.size(1);
  int nOutputPlane = weight.size(0);
  auto input_data = input.contiguous();
  auto weight_data = weight.contiguous();

  auto bias_data = bias.contiguous();
  if (bias.numel() > 0) {
    AT_CHECK(bias_data.is_contiguous(), "ones needs to be contiguous");
  }
  bool is_batch = input.dim() == 3;
  if (is_batch) {
    input_data.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t inputHeight = input.size(2);
  int64_t inputWidth = input.size(3);
  int64_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input.size(0);

  // Resize output
  output.resize_({batchSize, nOutputPlane, outputHeight, outputWidth});
  output.zero_();

  // Resize temporary columns
  columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets
  // increased, and always contains ones.
  if (!ones.is_contiguous() || ones.dim() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    // Resize plane and fill with ones...
    ones.resize_({outputHeight, outputWidth});
    ones.fill_(1);
  }

  // Helpers
  auto options = input.options();
  Tensor input_n = at::empty({0}, options);
  Tensor output_n = at::empty({0}, options);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt++) {
    // Matrix multiply per output:
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    int64_t m_ = nOutputPlane;
    int64_t n_ = outputHeight * outputWidth;
    int64_t k_ = 1;

    if (bias.numel() > 0) {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_dilated2d", [&] {
        auto* ones_ptr = ones.data<scalar_t>();
        auto* bias_ptr = bias.data<scalar_t>();
        auto* output_n_ptr = output_n.data<scalar_t>();
        THBlas_gemm<scalar_t>(
            't',
            'n',
            n_,
            m_,
            k_,
            1,
            ones_ptr,
            k_,
            bias_ptr,
            k_,
            0,
            output_n_ptr,
            n_);
      });
    } else {
      output_n.zero_();
    }

    // Extract columns:
    columns = at::thnn_im2col(input_n, kernel_size, dilation_size, pad_size, stride_size);

    // M,N,K are dims of matrix A and B
    int64_t m = nOutputPlane;
    int64_t n = columns.size(1);
    int64_t k = nInputPlane * kH * kW;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_dilated2d", [&] {
      auto* columns_ptr = columns.data<scalar_t>();
      auto* weight_ptr = weight.data<scalar_t>();
      auto* output_n_ptr = output_n.data<scalar_t>();
      THBlas_gemm<scalar_t>(
          'n',
          'n',
          n,
          m,
          k,
          1,
          columns_ptr,
          n,
          weight_ptr,
          k,
          1,
          output_n_ptr,
          n);
    });
  }
  // Resize output
  if (is_batch) {
    output.resize_({nOutputPlane, outputHeight, outputWidth});
  }
}

void thnn_conv_dilated2d_backward_out_cpu_template(
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
    const Tensor& ones,
    Tensor& grad_columns
                                                   ) {
  conv_dilated2d_shape_check(input, weight, at::empty({0}) /* bias */,
                             grad_output, grad_weight, grad_bias,
                             kernel_size, stride_size, pad_size, dilation_size, columns, ones, at::empty({0}) /* grad_columns */ );

  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  auto dilationH = dilation_size[0];
  auto dilationW = dilation_size[1];
  
  int64_t nInputPlane = weight.size(1);
  int64_t nOutputPlane = weight.size(0);
  auto input_data = input.contiguous();
  auto weight_data = weight.contiguous();
  auto grad_output_data = grad_output.contiguous();
  bool is_batch = input.dim() == 3;
  if (is_batch) {
    input_data.resize_({1, input.size(0), input.size(1), input.size(2)});
    grad_output_data.resize_({1, grad_output_data.size(0), grad_output_data.size(1), grad_output_data.size(2)});
  }
  int64_t inputHeight = input.size(2);
  int64_t inputWidth = input.size(3);
  int64_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  // Batch size + input planes
  int64_t batchSize = input.size(0);

  // Resize output
  grad_input.resize_({batchSize, nInputPlane, inputHeight, inputWidth});
  grad_input.zero_();

  // Resize temporary columns
  grad_columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});
  grad_columns.zero_();

  // Helpers
  auto options = input.options();
  Tensor input_n = at::empty({0}, options);
  Tensor grad_input_n = at::empty({0}, options);
  Tensor grad_output_n = at::empty({0}, options);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt++) {
    // Matrix multiply per output:
    input_n = input.select(0, elt);
    grad_input_n = grad_input.select(0, elt);
    grad_output_n = grad_output_data.select(0, elt);

    // M,N,K are dims of matrix A and B
    int64_t m = nInputPlane * kW * kH;
    int64_t n = grad_columns.size(1);
    int64_t k = nOutputPlane;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_dilated2d_backward", [&] {
        auto* grad_columns_ptr = grad_columns.data<scalar_t>();
        auto* weight_ptr = weight_data.data<scalar_t>();
        auto* grad_output_n_ptr = grad_output_n.data<scalar_t>();
        // Gradient of input:
        if (grad_input.numel() > 0) {
          THBlas_gemm<scalar_t>(
                                'n',
                                't',
                                n,
                                m,
                                k,
                                1,
                                grad_output_n_ptr,
                                n,
                                weight_ptr,
                                m,
                                0,
                                grad_columns_ptr,
                                n
                                );
        }

        // Gradient of weight:
        if (grad_weight.numel() > 0) {
          Tensor columns_ = at::thnn_im2col(input_n, kernel_size, dilation_size, pad_size, stride_size);
          auto* columns_ptr = columns_.data<scalar_t>();
          auto* grad_weight_ptr = grad_weight.data<scalar_t>();
          int64_t m = nOutputPlane;
          int64_t n = nInputPlane * kW * kH;
          int64_t k = columns.size(1);
          scalar_t scale = 1; // TODO: expose as argument
          THBlas_gemm<scalar_t>(
                                          't',
                                          'n',
                                          n,
                                          m,
                                          k,
                                          scale,
                                          columns_ptr,
                                          k,
                                          grad_output_n_ptr,
                                          k,
                                          1,
                                          grad_weight_ptr,n
                                );
        }

        // Gradient of bias:
        if (grad_bias.numel() > 0) {
          int64_t m_ = nOutputPlane;
          int64_t k_ = outputHeight * outputWidth;
          // Do GEMV (note: this is a bit confusing because gemv assumes
          // column-major matrices) Define a buffer of ones, for bias accumulation
          /*
          if (ones.dim() != 2 ||
              ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
            // Resize plane and fill with ones...
            ones.resize_({outputHeight, outputWidth});
            ones.fill_(1);
          }
          */
          auto* ones_ptr = ones.data<scalar_t>();
          auto* grad_output_n_ptr = grad_output_n.data<scalar_t>();
          auto* grad_bias_ptr = grad_bias.data<scalar_t>();
          scalar_t scale = 1; // TODO: expose as argument
          THBlas_gemv<scalar_t>(
          't',
          k_,
          m_,
          scale,
          grad_output_n_ptr,
          k_,
          ones_ptr,
          1,
          1,
          grad_bias_ptr,
          1);
        }
      });

    if (grad_input.numel() > 0) {
      // Unpack columns back into input:
      grad_input_n = at::thnn_col2im(grad_columns,
                                     grad_input_n.sizes(),
                                     kernel_size, dilation_size, pad_size, stride_size);
    }
  }
  if (is_batch) {
    //grad_output.resize_({nOutputPlane, outputHeight, outputWidth});
    grad_input.resize_({nInputPlane, inputHeight, inputWidth});
    //input.resize_({nInputPlane, inputHeight, inputWidth});
  }
}
 
} // namespace

// Tensor &

std::tuple<Tensor&, Tensor&, Tensor&> thnn_conv_dilated2d_out_cpu(
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
  conv_dilated_cpu_template(
      output,
      input,
      weight,
      bias,
      columns,
      ones,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return std::tie(output, columns, ones);
}

Tensor& thnn_conv_dilated2d_out_cpu(
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
  conv_dilated_cpu_template(
      output,
      input,
      weight,
      bias,
      columns,
      ones,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return output;
}

Tensor thnn_conv_dilated2d_cpu(
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
  conv_dilated_cpu_template(
      output,
      input,
      weight,
      bias,
      columns,
      ones,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return output;
}

std::tuple<Tensor&, Tensor&, Tensor&> thnn_conv_dilated2d_forward_out_cpu(
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
  std::cout << "NOT IMPLEMENTED: thnn_conv_dilated2d_forward_out_cpu3" << std::endl;
  conv_dilated_cpu_template(  // TODO: is this correct?
      output,
      input,
      weight,
      bias,
      columns,
      ones,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return std::tie(output, columns, ones);
}

Tensor& thnn_conv_dilated2d_forward_out_cpu(
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
  std::cout << "NOT IMPLEMENTED: thnn_conv_dilated2d_forward_out_cpu1" << std::endl;
  conv_dilated_cpu_template(  // TODO: is this correct?
      output,
      input,
      weight,
      bias,
      columns,
      ones,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> thnn_conv_dilated2d_forward_cpu(
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
  std::cout << "NOT IMPLEMENTED: thnn_conv_dilated2d_forward_cpu" << std::endl;
  conv_dilated_cpu_template(  // TODO: is this correct?
      output,
      input,
      weight,
      bias,
      columns,
      ones,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return std::tie(output, columns, ones);
}

std::tuple<Tensor&, Tensor&, Tensor&> thnn_conv_dilated2d_backward_out_cpu(
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
    const Tensor& ones
                                                                           ) {
  // TODO
  std::cout << "thnn_conv_dilated2d_backward_out_cpu3" << std::endl;
  auto options = grad_input.options();
  Tensor grad_columns = at::empty({0}, options);
  thnn_conv_dilated2d_backward_out_cpu_template(grad_input, grad_weight, grad_bias, grad_output, input, weight, kernel_size, stride_size, pad_size, dilation_size, columns, ones, grad_columns);
  return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> thnn_conv_dilated2d_backward_cpu(
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
  Tensor grad_columns = at::empty({0}, options);
  std::cout << "thnn_conv_dilated2d_backward_cpu1" << std::endl;
  thnn_conv_dilated2d_backward_out_cpu_template(grad_input, grad_weight, grad_bias, grad_output, input, weight, kernel_size, stride_size, pad_size, dilation_size, columns, ones, grad_columns);
  return std::tie(grad_input, grad_weight, grad_bias);
}


} // namespace native
} // namespace at

#ifdef SKIPTHISBLOCK


// pearu: backward:
void THNN_(SpatialDilatedConvolution_accGradParameters)(
    THNNState* state,
    THTensor* input,
    THTensor* gradOutput,
    THTensor* gradWeight,
    THTensor* gradBias,
    THTensor* columns,
    THTensor* ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    accreal scale_) {
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THNN_(SpatialDilatedConvolution_shapeCheck)
  (input,
   gradOutput,
   gradWeight,
   gradBias,
   kH,
   kW,
   dH,
   dW,
   padH,
   padW,
   dilationH,
   dilationW,
   1);

  // Params
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  int is_batch = 1;
  if (input->dim() == 3) {
    // Force batch
    is_batch = 0;
    THTensor_(resize4d)(
        input, 1, input->size(0), input->size(1), input->size(2));
    THTensor_(resize4d)(
        gradOutput,
        1,
        gradOutput->size(0),
        gradOutput->size(1),
        gradOutput->size(2));
  }

  int64_t nInputPlane = input->size(1);
  int64_t nOutputPlane = gradOutput->size(1);
  int64_t inputWidth = input->size(3);
  int64_t inputHeight = input->size(2);
  int64_t outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input->size(0);

  // Resize temporary columns
  THTensor_(resize2d)(
      columns, nInputPlane * kW * kH, outputHeight * outputWidth);

  // Helpers
  THTensor* input_n = THTensor_(new)();
  THTensor* gradOutput_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt++) {
    // Matrix mulitply per output:
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Do Weight:
    if (gradWeight) {
      // Matrix mulitply per output:
      THTensor_(select)(input_n, input, 0, elt);

      // Extract columns:
      THNN_(im2col)
      (input_n->data<scalar_t>(),
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
       columns->data<scalar_t>());

      // M,N,K are dims of matrix A and B
      int64_t m = nOutputPlane;
      int64_t n = nInputPlane * kW * kH;
      int64_t k = columns->size(1);

      // Do GEMM (note: this is a bit confusing because gemm assumes
      // column-major matrices)
      THBlas_(gemm)(
          't',
          'n',
          n,
          m,
          k,
          scale,
          columns->data<scalar_t>(),
          k,
          gradOutput_n->data<scalar_t>(),
          k,
          1,
          gradWeight->data<scalar_t>(),
          n);
    }

    // Do Bias:
    if (gradBias) {
      // M,N,K are dims of matrix A and B
      int64_t m_ = nOutputPlane;
      int64_t k_ = outputHeight * outputWidth;

      // Do GEMV (note: this is a bit confusing because gemv assumes
      // column-major matrices) Define a buffer of ones, for bias accumulation
      if (ones->dim() != 2 ||
          ones->size(0) * ones->size(1) < outputHeight * outputWidth) {
        // Resize plane and fill with ones...
        THTensor_(resize2d)(ones, outputHeight, outputWidth);
        THTensor_(fill)(ones, 1);
      }
      THBlas_(gemv)(
          't',
          k_,
          m_,
          scale,
          gradOutput_n->data<scalar_t>(),
          k_,
          ones->data<scalar_t>(),
          1,
          1,
          gradBias->data<scalar_t>(),
          1);
    }
  }

  // Free
  c10::raw::intrusive_ptr::decref(input_n);
  c10::raw::intrusive_ptr::decref(gradOutput_n);

  // Resize
  if (is_batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
