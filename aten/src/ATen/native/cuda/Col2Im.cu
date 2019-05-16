// #ifndef THC_GENERIC_FILE
// #define THC_GENERIC_FILE "THCUNN/generic/Col2Im.cu"
// #else

#include <ATen/div_rtn.h>

static inline void Col2Im_shape_check(
    THCState* state,
    Tensor* input,
    Tensor* grad_output,
    int64_t output_height,
    int64_t output_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t padilation_height,
    int64_t padilation_width,
    int64_t stride_height,
    int64_t stride_width) {
  /* TODO: AT_CHECK just have 2 args: condition and message */
   AT_CHECK(
      kernel_width > 0 && kernel_height > 0,
      6,
      "kernel size should be greater than zero, but got kernel_height: %d kernel_width: %d",
      kernel_height,
      kernel_width);
  /* TODO: AT_CHECK just have 2 args: condition and message */
   AT_CHECK(
      stride_width > 0 && stride_height > 0,
      12,
      "stride should be greater than zero, but got stride_height: %d stride_width: %d",
      stride_height,
      stride_width);
  /* TODO: AT_CHECK just have 2 args: condition and message */
   AT_CHECK(
      dilation_width > 0 && dilation_height > 0,
      8,
      "dilation should be greater than zero, but got dilation_height: %d dilation_width: %d",
      dilation_height,
      dilation_width);

  int64_t ndim = input.ndimension();
  /* TODO: AT_CHECK just have 2 args*/ AT_CHECK(
      state,
      input.numel() != 0 && (ndim == 2 || ndim == 3),
      2,
      input,
      "Expected non-empty 2D or 3D input tensor, but got input of shape %s");

  int batch_dim = (ndim == 3) ? 0 : -1;
  int64_t n_input_plane = input.size(batch_dim + 1);

  if (n_input_plane % (kernel_width * kernel_height) != 0) {
    AT_ERROR(
        "Expected size of input's dimension 1 to be divisible by the "
        "product of kernel_size, but got input.size(1)=%lld and "
        "kernel_size=(%d, %d).",
        (long long)n_input_plane,
        kernel_height,
        kernel_width);
  }

  int64_t input_length = input.size(batch_dim + 2);
  int64_t nBlockstride_height =
      div_rtn<int64_t>(output_height + 2 * padilation_height - dilation_height * (kernel_height - 1) - 1, stride_height) + 1;
  int64_t nBlockstride_width =
      div_rtn<int64_t>(output_width + 2 * padilation_width - dilation_width * (kernel_width - 1) - 1, stride_width) + 1;

  if (input_length != (nBlockstride_height * nBlockstride_width)) {
    AT_ERROR(
        "Given output_size=(%d, %d), kernel_size=(%d, %d), "
        "dilation=(%d, %d), padding=(%d, %d), stride=(%d, %d), expected "
        "size of input's dimension 2 to match the calculated number of "
        "sliding blocks %lld * %lld = %lld, but got input.size(2)=%lld.",
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        dilation_height,
        dilation_width,
        padilation_height,
        padilation_width,
        stride_height,
        stride_width,
        (long long)nBlockstride_height,
        (long long)nBlockstride_width,
        (long long)(nBlockstride_height * nBlockstride_width),
        (long long)input_length);
  }

  if (output_width < 1 || output_height < 1) {
    AT_ERROR(
        "Expected output spatial size to be positive, but got: output_size=(%d, %d).",
        output_height,
        output_width);
  }
}

void Col2Im_out_cpu(
    THCState* state,
    Tensor* input,
    Tensor* output,
    int64_t output_height,
    int64_t output_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t padilation_height,
    int64_t padilation_width,
    int64_t stride_height,
    int64_t stride_width) {
  /* TODO: TensorArg tensorname_arg{tensorname, "tensorname", 1}; */
/* TODO: checkAllSameGPU should use TensorArg */
checkAllSameGPU(
  "/* TODO: use the name of the function as description here */",  { input, output });

  Col2Im_shape_check
  (state,
   input,
   Tensor(),
   output_height,
   output_width,
   kernel_height,
   kernel_width,
   dilation_height,
   dilation_width,
   padilation_height,
   padilation_width,
   stride_height,
   stride_width);

  bool batched_input = true;
  if (input.dim() == 2) {
    // Force batch
    batched_input = false;
    input.resize_({ 1, input.size(0), input.size(1) });
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

Tensor   input  = input_.contiguous(); /* TODO: add _ to the arg definition above */

  output.resize_({ batch_size, n_output_plane, output_height, output_width });
  output.zero_();

  Tensor* input_n = Tensor();
  Tensor* output_n = Tensor();

  int64_t height_col = (output_height + 2 * padilation_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
  int64_t width_col = (output_width + 2 * padilation_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;

  for (int64_t elt = 0; elt < batch_size; elt++) {
    THCTensor_(select)(input_n, input, 0, elt);
    THCTensor_(select)(output_n, output, 0, elt);

    col2im<scalar_t, accreal>(
        at::cuda::getCurrentCUDAStream(),
        input_n.data(),
        n_output_plane,
        output_height,
        output_width,
        height_col,
        width_col,
        kernel_height,
        kernel_width,
        padilation_height,
        padilation_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        output_n).data();
  }

  THCTensor_(free)(input_n);
  THCTensor_(free)(output_n);

  if (!batched_input) {
    output.resize_({ n_output_plane, output_height, output_width });
  }
  THCTensor_(free)(input);
}

void Col2Im_backward_out_cpu(
    THCState* state,
    Tensor* grad_output,
    Tensor* grad_input,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t padilation_height,
    int64_t padilation_width,
    int64_t stride_height,
    int64_t stride_width) {
  IntArrayRef kernel_size = IntArrayRef({kernel_height, kernel_width});
  IntArrayRef dilation = IntArrayRef({dilation_height, dilation_width});
  IntArrayRef padding = IntArrayRef({padilation_height, padilation_width});
  IntArrayRef stride = IntArrayRef({stride_height, stride_width});
  at::native::cuda::im2col_out_cuda(
      grad_output, grad_input, kernel_size, dilation, padding, stride);
}

// #endif


// THCUNN/generic
// #ifndef THC_GENERIC_FILE
// #define THC_GENERIC_FILE "THCUNN/generic/Col2Im.cu"
// #else

#include <ATen/div_rtn.h>

static inline void Col2Im_shape_check(
    THCState* state,
    Tensor* input,
    Tensor* grad_output,
    int64_t output_height,
    int64_t output_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t padilation_height,
    int64_t padilation_width,
    int64_t stride_height,
    int64_t stride_width) {
  /* TODO: AT_CHECK just have 2 args: condition and message */
   AT_CHECK(
      kernel_width > 0 && kernel_height > 0,
      6,
      "kernel size should be greater than zero, but got kernel_height: %d kernel_width: %d",
      kernel_height,
      kernel_width);
  /* TODO: AT_CHECK just have 2 args: condition and message */
   AT_CHECK(
      stride_width > 0 && stride_height > 0,
      12,
      "stride should be greater than zero, but got stride_height: %d stride_width: %d",
      stride_height,
      stride_width);
  /* TODO: AT_CHECK just have 2 args: condition and message */
   AT_CHECK(
      dilation_width > 0 && dilation_height > 0,
      8,
      "dilation should be greater than zero, but got dilation_height: %d dilation_width: %d",
      dilation_height,
      dilation_width);

  int64_t ndim = input.ndimension();
  /* TODO: AT_CHECK just have 2 args*/ AT_CHECK(
      state,
      input.numel() != 0 && (ndim == 2 || ndim == 3),
      2,
      input,
      "Expected non-empty 2D or 3D input tensor, but got input of shape %s");

  int batch_dim = (ndim == 3) ? 0 : -1;
  int64_t n_input_plane = input.size(batch_dim + 1);

  if (n_input_plane % (kernel_width * kernel_height) != 0) {
    AT_ERROR(
        "Expected size of input's dimension 1 to be divisible by the "
        "product of kernel_size, but got input.size(1)=%lld and "
        "kernel_size=(%d, %d).",
        (long long)n_input_plane,
        kernel_height,
        kernel_width);
  }

  int64_t input_length = input.size(batch_dim + 2);
  int64_t nBlockstride_height =
      div_rtn<int64_t>(output_height + 2 * padilation_height - dilation_height * (kernel_height - 1) - 1, stride_height) + 1;
  int64_t nBlockstride_width =
      div_rtn<int64_t>(output_width + 2 * padilation_width - dilation_width * (kernel_width - 1) - 1, stride_width) + 1;

  if (input_length != (nBlockstride_height * nBlockstride_width)) {
    AT_ERROR(
        "Given output_size=(%d, %d), kernel_size=(%d, %d), "
        "dilation=(%d, %d), padding=(%d, %d), stride=(%d, %d), expected "
        "size of input's dimension 2 to match the calculated number of "
        "sliding blocks %lld * %lld = %lld, but got input.size(2)=%lld.",
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        dilation_height,
        dilation_width,
        padilation_height,
        padilation_width,
        stride_height,
        stride_width,
        (long long)nBlockstride_height,
        (long long)nBlockstride_width,
        (long long)(nBlockstride_height * nBlockstride_width),
        (long long)input_length);
  }

  if (output_width < 1 || output_height < 1) {
    AT_ERROR(
        "Expected output spatial size to be positive, but got: output_size=(%d, %d).",
        output_height,
        output_width);
  }
}

void Col2Im_out_cpu(
    THCState* state,
    Tensor* input,
    Tensor* output,
    int64_t output_height,
    int64_t output_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t padilation_height,
    int64_t padilation_width,
    int64_t stride_height,
    int64_t stride_width) {
  /* TODO: TensorArg tensorname_arg{tensorname, "tensorname", 1}; */
/* TODO: checkAllSameGPU should use TensorArg */
checkAllSameGPU(
  "/* TODO: use the name of the function as description here */",  { input, output });

  Col2Im_shape_check
  (state,
   input,
   Tensor(),
   output_height,
   output_width,
   kernel_height,
   kernel_width,
   dilation_height,
   dilation_width,
   padilation_height,
   padilation_width,
   stride_height,
   stride_width);

  bool batched_input = true;
  if (input.dim() == 2) {
    // Force batch
    batched_input = false;
    input.resize_({ 1, input.size(0), input.size(1) });
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

Tensor   input  = input_.contiguous(); /* TODO: add _ to the arg definition above */

  output.resize_({ batch_size, n_output_plane, output_height, output_width });
  output.zero_();

  Tensor* input_n = Tensor();
  Tensor* output_n = Tensor();

  int64_t height_col = (output_height + 2 * padilation_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
  int64_t width_col = (output_width + 2 * padilation_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;

  for (int64_t elt = 0; elt < batch_size; elt++) {
    THCTensor_(select)(input_n, input, 0, elt);
    THCTensor_(select)(output_n, output, 0, elt);

    col2im<scalar_t, accreal>(
        at::cuda::getCurrentCUDAStream(),
        input_n.data(),
        n_output_plane,
        output_height,
        output_width,
        height_col,
        width_col,
        kernel_height,
        kernel_width,
        padilation_height,
        padilation_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        output_n).data();
  }

  THCTensor_(free)(input_n);
  THCTensor_(free)(output_n);

  if (!batched_input) {
    output.resize_({ n_output_plane, output_height, output_width });
  }
  THCTensor_(free)(input);
}

void Col2Im_backward_out_cpu(
    THCState* state,
    Tensor* grad_output,
    Tensor* grad_input,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t padilation_height,
    int64_t padilation_width,
    int64_t stride_height,
    int64_t stride_width) {
  IntArrayRef kernel_size = IntArrayRef({kernel_height, kernel_width});
  IntArrayRef dilation = IntArrayRef({dilation_height, dilation_width});
  IntArrayRef padding = IntArrayRef({padilation_height, padilation_width});
  IntArrayRef stride = IntArrayRef({stride_height, stride_width});
  at::native::cuda::im2col_out_cuda(
      grad_output, grad_input, kernel_size, dilation, padding, stride);
}

// #endif
