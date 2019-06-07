#pragma once
/*
  Provides a subset of CUDA BLAS functions as templates:

    gemm<Dtype>(stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
  ldc)

    gemv<Dtype>(stream, transa, m, n, alpha, a, lda, x, incx, beta, y, incy)

  where Dtype is double, float, or at::Half. The functions are
  available in at::cuda::blas namespace.
 */

#include <ATen/cuda/CUDAContext.h>

// In CUDA 8.0, definition of data types for sgemmex changed
#if CUDA_VERSION < 8000
#define CUDA_R_16F CUBLAS_DATA_HALF
#endif

#define TORCH_CUDABLAS_CHECK(EXPR)        \
  do {                                    \
    cublasStatus_t __err = EXPR;          \
    if (__err != CUBLAS_STATUS_SUCCESS) { \
      AT_ERROR(                           \
          "CUDA error: ",                 \
          _cublasGetErrorEnum(__err),     \
          " when calling `" #EXPR "`");   \
    }                                     \
  } while (0)

namespace {
static const char* _cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "<unknown>";
}

static cublasOperation_t _cublasOpFromChar(char op) {
  switch (op) {
    case 'n':
      return CUBLAS_OP_N;
    case 't':
      return CUBLAS_OP_T;
    case 'c':
      return CUBLAS_OP_C;
  }
  AT_ERROR(
      "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}
} // anonymous namespace

namespace at {
namespace cuda {
namespace blas {

#define GEMM_ARGTYPES(Dtype)                                               \
  cudaStream_t stream, char transa, char transb, int64_t m, int64_t n,     \
      int64_t k, Dtype alpha, const Dtype *a, int64_t lda, const Dtype *b, \
      int64_t ldb, Dtype beta, Dtype *c, int64_t ldc

template <typename Dtype>
void gemm(GEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::gemm: not implemented for ", typeid(Dtype).name());
}

template <>
void gemm<double>(GEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  TORCH_CUDABLAS_CHECK(cublasDgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<float>(GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<at::Half>(GEMM_ARGTYPES(at::Half)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
#ifdef __HIP_PLATFORM_HCC__
  TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &falpha,
      a,
      rocblas_datatype_f16_r,
      lda,
      b,
      rocblas_datatype_f16_r,
      ldb,
      &fbeta,
      c,
      rocblas_datatype_f16_r,
      ldc,
      c,
      rocblas_datatype_f16_r,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      0,
      NULL,
      NULL));
#else

#if CUDA_VERSION >= 9000
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5) {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DFALT_TENSOR_OP));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
#endif
    TORCH_CUDABLAS_CHECK(cublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc));
#if CUDA_VERSION >= 9000
  }
#endif
#endif
}

#define GEMV_ARGTYPES(Dtype)                                                 \
  cudaStream_t stream, char trans, int64_t m, int64_t n, Dtype alpha,        \
      const Dtype *a, int64_t lda, const Dtype *x, int64_t incx, Dtype beta, \
      Dtype *y, int64_t incy

template <typename Dtype>
void gemv(GEMV_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::gemv: not implemented for ", typeid(Dtype).name());
}

template <>
void gemv<double>(GEMV_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  TORCH_CUDABLAS_CHECK(
      cublasDgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<float>(GEMV_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = _cublasOpFromChar(trans);
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  TORCH_CUDABLAS_CHECK(
      cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<at::Half>(GEMV_ARGTYPES(at::Half)) {
  TORCH_CHECK(
      incx == 1, "at::cuda::gemv<Half>: support for incx != 1 not implemented");
  TORCH_CHECK(
      incy == 1, "at::cuda::gemv<Half>: support for incy != 1 not implemented");
  gemm<at::Half>(
      stream, trans, CUBLAS_OP_N, m, 1, n, alpha, a, n, x, n, beta, y, m);
}

} // namespace blas
} // namespace cuda
} // namespace at

#undef GEMM_ARGTYPES
#undef GEMV_ARGTYPES
