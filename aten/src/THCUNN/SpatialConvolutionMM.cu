#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <THCUNN/generic/SpatialConvolutionMM.cu>
#include <THC/THCGenerateFloatTypes.h>

#include <ATen/native/cuda/im2col.cuh>
