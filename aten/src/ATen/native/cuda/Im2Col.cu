#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAApplyUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/div_rtn.h>
#include <ATen/native/cuda/im2col.cuh>

namespace at {
namespace native {
namespace {

static inline void im2col_shape_check(
    Tensor* input,
    Tensor* grad_output,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t stride_height,
    int64_t stride_width) {
  AT_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);
  AT_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);
  AT_CHECK(
      pad_width >= 0 && pad_height >= 0,
      "padding should be non-negative, but got pad_height: ",
      pad_height,
      " pad_width: ",
      pad_width);
  AT_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  int64_t ndim = input.ndimension();

  AT_CHECK(
      input.numel() != 0 && (ndim == 3 || ndim == 4),
      "Expected non-empty 3D or 4D input tensor, but got input of size ",
      input.sizes()); // TODO: check it

  int dim_batch = 0;

  if (ndim == 3) {
    dim_batch = -1;
  }

  int64_t n_input_plane = input.size(dim_batch + 1);
  int64_t input_height = input.size(dim_batch + 2);
  int64_t input_width = input.size(dim_batch + 3);
  int64_t output_height = div_rtn<int64_t>(
                              input_height + 2 * pad_height -
                                  (dilation_height * (kernel_height - 1) + 1),
                              stride_height) +
      1;
  int64_t output_width = div_rtn<int64_t>(
                             input_width + 2 * pad_width -
                                 (dilation_width * (kernel_width - 1) + 1),
                             stride_width) +
      1;

  if (output_height < 1 || output_width < 1) {
    AT_ERROR(
        "Given input with spatial size (%d, %d), kernel_size=(%d, %d), "
        "dilation=(%d, %d), padding=(%d, %d), calculated "
        "shape of the array of sliding blocks as (%d, %d), which is "
        "too small (non-positive).",
        input_height,
        input_height,
        kernel_height,
        kernel_width,
        dilation_height,
        dilation_width,
        pad_height,
        pad_width,
        output_height,
        output_width);
  }
}

void im2col_out_cuda_template(
    Tensor* input_,
    Tensor* output,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  AT_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  AT_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  AT_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  AT_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  TensorArg input_arg{input, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU("im2col_cuda", {input_arg, output_arg});

  im2col_shape_check(
      input,
      Tensor(),
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  output.zero_();

  Tensor* input_n = Tensor();
  Tensor* output_n = Tensor();

  for (int64_t elt = 0; elt < batch_size; elt++) {
    /* TODO: check if it is working properly */
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);

    im2col(
        at::cuda::getCurrentCUDAStream(),
        input_n.data(),
        n_input_plane,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        output_n.data());
  }

  if (!batched_input) {
    output.resize_({n_output_plane, output_length});
  }
}

void im2col_backward_out_cuda_template(
    Tensor* grad_output,
    Tensor* grad_input,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  AT_CHECK(
      input_size.size() == 2,
      "It is expected input_size equals to 2, but got size ",
      input_size.size());

  AT_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  AT_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  AT_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  AT_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  int64_t input_height = input_size[0];
  int64_t input_width = input_size[1];
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  col2im(
      grad_output,
      grad_input,
      input_height,
      input_width,
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);
}

} // namespace

Tensor& im2col_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  im2col_out_cuda_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

Tensor im2col_cuda(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(input);
  im2col_out_cuda_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

Tensor& im2col_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  im2col_backward_out_cuda_template(
      grad_input,
      grad_output,
      input_size,
      kernel_size,
      dilation,
      padding,
      stride);
  return grad_input;
}

Tensor im2col_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor grad_input = at::empty_like(grad_output);
  im2col_backward_out_cuda_template(
      grad_input,
      grad_output,
      input_size,
      kernel_size,
      dilation,
      padding,
      stride);
  return grad_input;
}

} // namespace native
} // namespace at
