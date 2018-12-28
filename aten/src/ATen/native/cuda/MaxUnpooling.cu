#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"

#include "ATen/cuda/CUDAContext.h"
#include "c10/util/Exception.h"

namespace at {
namespace native {

template <typename T>
__host__ __device__ __forceinline__ T ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__global__ void MaxUnpooling2d_forward_kernel(
    const int64_t numInputElements,
    const T* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    T* output) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < numInputElements;
       linearIndex += blockDim.x * gridDim.x) {
    int c = (linearIndex / inputWidth / inputHeight) % numChannels;
    int n = linearIndex / inputWidth / inputHeight / numChannels;
    output += (n * numChannels + c) * outputHeight * outputWidth;
    int maxind = indices[linearIndex];
    output[maxind] = input[linearIndex];
  }
}

template <typename T>
__global__ void MaxUnpooling3d_forward_kernel(
    PackedTensorAccessor<T, 4> input,
    PackedTensorAccessor<int64_t, 4> indices,
    T* output,
    const int64_t oT,
    const int64_t oW,
    const int64_t oH,
    const int64_t offsetZ) {
  int64_t iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t iRow = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t iFrame = (blockIdx.z + offsetZ) % input.size(1); // input frame/time
  int64_t slice = (blockIdx.z + offsetZ) / input.size(1); // input slice/feature
  if (iRow < input.size(2) && iColumn < input.size(3)) {
    T val = input[slice][iFrame][iRow][iColumn];
    int64_t index = indices[slice][iFrame][iRow][iColumn];
    output[slice * oT * oH * oW + index] = val;
  }
}

template <typename T>
__global__ void MaxUnpooling2d_backward_kernel(
    const int64_t numInputElements,
    const T* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    T* output) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < numInputElements;
       linearIndex += blockDim.x * gridDim.x) {
    int c = (linearIndex / inputWidth / inputHeight) % numChannels;
    int n = linearIndex / inputWidth / inputHeight / numChannels;
    input += (n * numChannels + c) * outputHeight * outputWidth;
    int maxind = indices[linearIndex];
    output[linearIndex] = input[maxind];
  }
}

template <typename T>
__global__ void MaxUnpooling3d_backward_kernel(
    T* gradOutputData,
    int64_t oT,
    int64_t oH,
    int64_t oW,
    PackedTensorAccessor<int64_t, 4> indices,
    PackedTensorAccessor<T, 4> gradInput,
    int offsetZ) {
  int iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int iRow = blockIdx.y * blockDim.y + threadIdx.y;
  int iFrame = (blockIdx.z + offsetZ) % gradInput.size(1); // output frame/time
  int slice =
      (blockIdx.z + offsetZ) / gradInput.size(1); // output slice/feature

  if (iRow < gradInput.size(2) && iColumn < gradInput.size(3)) {
    int64_t index = indices[slice][iFrame][iRow][iColumn];
    T grad_val = gradOutputData[slice * oT * oH * oW + index];
    gradInput[slice][iFrame][iRow][iColumn] = grad_val;
  }
}

Tensor& MaxUnpooling2d_forward_out_cuda(
    Tensor& output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  auto outputHeight = output_size[0];
  auto outputWidth = output_size[1];

  TensorArg output_arg{output, "output", 1}, self_arg{self, "self", 2},
      indices_arg{indices, "indices", 3};
  checkAllSameGPU(
      "MaxUnpooling2d_forward_cuda_out", {output_arg, self_arg, indices_arg});

  AT_CHECK(self.numel() > 0, "Input must be non-empty tensor");

  AT_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "Input to MaxUnpooling2d should be a 3d or 4d Tensor",
      self.sizes());
  AT_CHECK(
      self.sizes() == indices.sizes(),
      "Shape of input must match shape of indices");
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");

  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t numBatch = 1;

  int64_t numChannels;
  int64_t inputHeight;
  int64_t inputWidth;

  if (self.ndimension() == 4) {
    numBatch = self.size(0);
    dimw++;
    dimh++;
  }
  numChannels = self.size(dimh - 1);
  inputHeight = self.size(dimh);
  inputWidth = self.size(dimw);

  auto input_contiguous = self.contiguous();
  auto indices_contiguous = indices.contiguous();

  output.resize_({numBatch, numChannels, outputHeight, outputWidth});

  output.zero_();

  dim3 block(512);
  dim3 grid((output.numel() + 512 - 1) / 512);

  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "MaxUnpooling2d_forward_kernel", ([&] {
        MaxUnpooling2d_forward_kernel<<<
            grid,
            block,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            self.numel(),
            input_contiguous.data<scalar_t>(),
            indices_contiguous.data<int64_t>(),
            numChannels,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            output.data<scalar_t>());
      }));
  AT_CHECK(
      cudaGetLastError() == cudaSuccess,
      "RoiPooling2d_forward_kernel failed with error code ",
      cudaGetLastError());
  if (self.ndimension() == 3) {
    output.resize_({numChannels, outputHeight, outputWidth});
  }
  return output;
}

Tensor MaxUnpooling2d_forward_cuda(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  AT_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "Input to MaxUnpooling2d should be a 3d or 4d Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");

  auto output = at::zeros(
      {self.size(0), self.size(1), output_size[0], output_size[1]},
      self.options());
  MaxUnpooling2d_forward_out_cuda(output, self, indices, output_size);
  return output;
}

void MaxUnpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding,
    bool check_grad) {
  // is_empty check
  AT_CHECK(input.numel() > 0, "Input must be non-empty");
  AT_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input must be 4d or 5d tensor");
  AT_CHECK(input.sizes() == indices.sizes());
  AT_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "stride should be never greater than zero, but got stride: ",
      stride);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input.ndimension() == 5) {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  int nslices = input.size(dimn);
  if (check_grad) {
    if (output_size[0] != gradOutput.size(dimt) ||
        output_size[1] != gradOutput.size(dimh) ||
        output_size[2] != gradOutput.size(dimw)) {
      AT_ERROR(
          "Inconsistent gradOutput size. output_size: ,",
          output_size,
          ". gradOutput: ",
          gradOutput);
    }
    AT_CHECK(
        gradOutput.ndimension() == input.ndimension() &&
        gradOutput.size(dimn) == nslices);
  }
}

Tensor& MaxUnpooling3d_forward_out_cuda(
    Tensor& output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
  MaxUnpooling3d_shape_check(
      self, at::empty({}), indices, output_size, stride, padding, false);
  AT_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size");
  AT_CHECK(
      stride.size() == 3, "There should be exactly three elements in stride");
  AT_CHECK(
      padding.size() == 3, "There should be exactly three elements in padding");

  auto outputTime = output_size[0];
  auto outputHeight = output_size[1];
  auto outputWidth = output_size[2];

  auto dT = stride[0];
  auto dH = stride[1];
  auto dW = stride[2];

  auto padT = padding[0];
  auto padH = padding[1];
  auto padW = padding[2];

  TensorArg output_arg{output, "output", 1}, self_arg{self, "self", 2},
      indices_arg{indices, "indices", 3};
  checkAllSameGPU(
      "MaxUnpooling3d_forward_out_cuda", {output_arg, self_arg, indices_arg});

  int64_t batchSize;
  int64_t inputSlices;
  int64_t inputTime;
  int64_t inputHeight;
  int64_t inputWidth;

  if (self.ndimension() == 4) {
    /* 5D */
    batchSize = 1;
    inputSlices = self.size(0);
    inputTime = self.size(1);
    inputHeight = self.size(2);
    inputWidth = self.size(3);
    output.resize_({inputSlices, outputTime, outputHeight, outputWidth});
  } else {
    /* resize output */
    batchSize = self.size(0);
    inputSlices = self.size(1);
    inputTime = self.size(2);
    inputHeight = self.size(3);
    inputWidth = self.size(4);
    output.resize_(
        {batchSize, inputSlices, outputTime, outputHeight, outputWidth});
  }
  auto input_contiguous = self.contiguous();
  auto indices_contiguous = indices.contiguous();

  auto output_contiguous = output.contiguous();
  output_contiguous.zero_();

  // Collapse batch and feature dimensions if needed
  auto input_contiguous_reshaped = input_contiguous;
  auto indices_contiguous_reshaped = indices_contiguous;

  if (input_contiguous.ndimension() == 5) {
    input_contiguous_reshaped = input_contiguous.reshape(
        {input_contiguous.size(0) * input_contiguous.size(1),
         input_contiguous.size(2),
         input_contiguous.size(3),
         input_contiguous.size(4)});
    indices_contiguous_reshaped = indices_contiguous.reshape(
        {indices_contiguous.size(0) * indices_contiguous.size(1),
         indices_contiguous.size(2),
         indices_contiguous.size(3),
         indices_contiguous.size(4)});
  }

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(
        ceilDiv(inputWidth, static_cast<int64_t>(block.x)),
        ceilDiv(inputHeight, static_cast<int64_t>(block.y)),
        totalZ > 65535 ? 65535 : totalZ);
    AT_DISPATCH_ALL_TYPES_AND_HALF(
        self.type(), "MaxUnpooling3d_forward_kernel", ([&] {
          MaxUnpooling3d_forward_kernel<<<
              grid,
              block,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              input_contiguous_reshaped.packed_accessor<scalar_t, 4>(),
              indices_contiguous_reshaped.packed_accessor<int64_t, 4>(),
              output_contiguous.data<scalar_t>(),
              outputTime,
              outputHeight,
              outputWidth,
              offsetZ);
        }));
    AT_CHECK(
        cudaGetLastError() == cudaSuccess,
        "RoiPooling3d_forward_kernel failed with error code ",
        cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }
  return output;
}

Tensor MaxUnpooling3d_forward_cuda(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
  AT_CHECK(
      (self.ndimension() == 4 || self.ndimension() == 5),
      "Input to MaxUnpooling3d should be a 4d or 5d Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size");

  auto output = at::zeros(
      {self.size(1),
       self.size(2),
       output_size[0],
       output_size[1],
       output_size[2]},
      self.options());
  MaxUnpooling3d_forward_out_cuda(
      output, self, indices, output_size, stride, padding);
  return output;
}

at::Tensor& MaxUnpooling2d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output, "grad_output", 2}, self_arg{self, "self", 3},
      indices_arg{indices, "indices", 4};
  checkAllSameGPU(
      "MaxUnpooling2d_backward_out_cuda",
      {grad_input_arg, grad_output_arg, self_arg, indices_arg});

  AT_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "Input to MaxUnpooling2d should be a 3d or 4d Tensor, instead got: ",
      self);

  AT_CHECK(
      self.sizes() == indices.sizes(),
      "Input should have same shape as indices");

  AT_CHECK(output_size.size() == 2, "output_size must have two elements");

  int64_t oheight = output_size[0];
  int64_t owidth = output_size[1];

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;

  int dimw = 2;
  int dimh = 1;

  if (self.ndimension() == 3) {
    nInputPlane = self.size(0);
    batchSize = 1;
  } else {
    ++dimw;
    ++dimh;
    nInputPlane = self.size(1);
    batchSize = self.size(0);
  }

  nInputCols = self.size(dimw);
  nInputRows = self.size(dimh);

  if (oheight != grad_output.size(dimh) || owidth != grad_output.size(dimw)) {
    AT_ERROR(
        "Inconsistent gradOutput size",
        oheight,
        ", output width= ",
        owidth,
        ", gradOutput: ",
        grad_output.size(dimh),
        "x",
        grad_output.size(dimw));
  }

  auto input_contiguous = self.contiguous();
  auto indices_contiguous = indices.contiguous();
  auto grad_output_contiguous = grad_output.contiguous();
  grad_input.resize_as_(input_contiguous);
  grad_input.zero_();

  int count = input_contiguous.numel();

  dim3 block(512);
  dim3 grid((count + 512 - 1) / 512);
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      input_contiguous.type(), "MaxUnpooling2d_backward_kernel", ([&] {
        MaxUnpooling2d_backward_kernel<<<
            grid,
            block,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            count,
            grad_output_contiguous.data<scalar_t>(),
            indices_contiguous.data<int64_t>(),
            nInputPlane,
            nInputRows,
            nInputCols,
            oheight,
            owidth,
            grad_input.data<scalar_t>());
      }));
  AT_CHECK(
      cudaGetLastError() == cudaSuccess,
      "MaxUnpooling2d_backward_kernel failed with error code ",
      cudaGetLastError());
  return grad_input;
}
at::Tensor MaxUnpooling2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  auto grad_input = at::zeros_like(self);
  MaxUnpooling2d_backward_out_cuda(
      grad_input, grad_output, self, indices, output_size);
  return grad_input;
}

at::Tensor& MaxUnpooling3d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
  int batchSize = 0;
  int inputSlices = 0;
  int inputTime = 0;
  int64_t inputHeight = 0;
  int64_t inputWidth = 0;
  AT_CHECK(output_size.size() == 3, "output_size must have three elements");
  AT_CHECK(stride.size() == 3, "stride must have three elements");
  AT_CHECK(padding.size() == 3, "padding must have three elements");

  auto outputTime = output_size[0];
  auto outputHeight = output_size[1];
  auto outputWidth = output_size[2];

  AT_CHECK(
      (self.ndimension() == 4 || self.ndimension() == 5),
      "self must be a 4d or 5d tensor");

  MaxUnpooling3d_shape_check(
      self, grad_output, indices, output_size, stride, padding, true);
  TensorArg self_arg{self, "self", 1}, indices_arg{indices, "indices", 2},
      grad_output_arg{grad_output, "grad_output", 3},
      grad_input_arg{grad_input, "grad_input", 4};
  checkAllSameGPU(
      "MaxUnpooling3d_backward_out_cuda",
      {self_arg, indices_arg, grad_output_arg, grad_input_arg});

  if (self.ndimension() == 4) {
    batchSize = 1;
    inputSlices = self.size(0);
    inputTime = self.size(1);
    inputHeight = self.size(2);
    inputWidth = self.size(3);
  } else {
    batchSize = self.size(0);
    inputSlices = self.size(1);
    inputTime = self.size(2);
    inputHeight = self.size(3);
    inputWidth = self.size(4);
  }

  auto input_contiguous = self.contiguous();
  grad_input.resize_as_(input_contiguous);
  grad_input.zero_();
  auto indices_contiguous = indices.contiguous();
  auto grad_output_contiguous = grad_output.contiguous();

  // Collapse batch and feature dimensions if needed
  auto grad_input_reshaped = grad_input;
  auto indices_contiguous_reshaped = indices_contiguous;

  if (grad_input.ndimension() == 5) {
    grad_input_reshaped =
        grad_input.reshape({grad_input.size(0) * grad_input.size(1),
                            grad_input.size(2),
                            grad_input.size(3),
                            grad_input.size(4)});
    indices_contiguous_reshaped = indices_contiguous.reshape(
        {indices_contiguous.size(0) * indices_contiguous.size(1),
         indices_contiguous.size(2),
         indices_contiguous.size(3),
         indices_contiguous.size(4)});
  }

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;

  dim3 block(32, 8);
  while (totalZ > 0) {
    dim3 grid(
        ceilDiv(inputWidth, static_cast<int64_t>(block.x)),
        ceilDiv(inputHeight, static_cast<int64_t>(block.y)),
        totalZ > 65535 ? 65535 : totalZ);
    AT_DISPATCH_ALL_TYPES_AND_HALF(
        input_contiguous.type(), "MaxUnpooling3d_backward_kernel", ([&] {
          MaxUnpooling3d_backward_kernel<<<
              grid,
              block,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              grad_output.data<scalar_t>(),
              outputTime,
              outputHeight,
              outputWidth,
              indices_contiguous_reshaped.packed_accessor<int64_t, 4>(),
              grad_input_reshaped.packed_accessor<scalar_t, 4>(),
              offsetZ);
        }));
    AT_CHECK(
        cudaGetLastError() == cudaSuccess,
        "MaxUnpool3d_backward_kernel failed with error code ",
        cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }
  return grad_input;
}

at::Tensor MaxUnpooling3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
  auto grad_input = at::zeros_like(self);
  MaxUnpooling3d_backward_out_cuda(
      grad_input, grad_output, self, indices, output_size, stride, padding);
  return grad_input;
}

} // namespace native
} // namespace at
