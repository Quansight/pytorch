#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

#include <THC/THCTensorMathPointwise.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#if defined(__CUDACC__) && (CUSPARSE_VERSION >= 11000 || (!defined(_MSC_VER) && CUSPARSE_VERSION >= 10301))
#define IS_SPMM_AVAILABLE() 1
#else
#define IS_SPMM_AVAILABLE() 0
#endif

#if IS_SPMM_AVAILABLE()
#include <library_types.h>
#endif


namespace at {
namespace native {

namespace {

using namespace at::sparse;

IntTensor _to_csr_int(const LongTensor& rowIndices, int64_t dim, int64_t nnz) {
  IntTensor csr = at::empty({dim + 1}, CUDA(kInt));
  IntTensor rowIndicesInt = at::empty({rowIndices.size(0)}, CUDA(kInt));
  rowIndicesInt.copy_(rowIndices);
  sparse::cuda::Xcoo2csr(
      rowIndicesInt.data_ptr<int32_t>(), nnz, dim, csr.data_ptr<int32_t>());
  return csr;
}

int confirm_mult_size(const std::vector<int>& mat1_size, const std::vector<int>& mat2_size) {
  TORCH_CHECK(
      mat1_size[1] == mat2_size[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1_size[0],
      "x",
      mat1_size[1],
      " and ",
      mat2_size[0],
      "x",
      mat2_size[1],
      ")");
  return mat1_size[1];
}

void create_general_description_(cusparseMatDescr_t& description_) {
  TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&description_));
  TORCH_CUDASPARSE_CHECK(cusparseSetMatType(description_, CUSPARSE_MATRIX_TYPE_GENERAL));
  TORCH_CUDASPARSE_CHECK(cusparseSetMatIndexBase(description_, CUSPARSE_INDEX_BASE_ZERO));
}

struct csrOutput {
  IntTensor csr_indices_{};
  IntTensor csr_pointers_{};
  at::Tensor csr_values_{};
  int nnz_{0};
  std::vector<int> size_;

  cusparseMatDescr_t description_{0};

  csrOutput(const std::vector<int> &size) : size_{size} {
    create_general_description_(description_);
  }

  int size(int index) const {
    return size_.at(index);
  }
};

template<class scalar_t> 
struct csrMatrixRef {
  int* csr_indices_{nullptr};
  int* csr_pointers_{nullptr};
  scalar_t* csr_values_{nullptr};
  int nnz_{0};
  std::vector<int> size_{};

  cusparseMatDescr_t description_{0};

  csrMatrixRef() {
    create_general_description_(description_);
  }

  csrMatrixRef(
      int* csr_indices,
      int* csr_pointers,
      scalar_t* csr_values,
      int nnz,
      const std::vector<int>& size)
      : csr_indices_{csr_indices},
        csr_pointers_{csr_pointers},
        csr_values_{csr_values},
        size_{size} {
    nnz_ = nnz;
    create_general_description_(description_);
  } 
 
  int size(int index) const {
    return size_.at(index);
  } 
};

using DcsrMatrixRef = csrMatrixRef<double>;
using ScsrMatrixRef = csrMatrixRef<float>; 

#if IS_SPMM_AVAILABLE()

template<class scalar_t>
csrOutput cuSparse_matrix_multiply(
    const csrMatrixRef<scalar_t>& lhs,
    const csrMatrixRef<scalar_t>& rhs,
    Tensor &output_values, 
    IntTensor &output_indices)
{
  TORCH_INTERNAL_ASSERT(false, "cusparse csr cuda 11 support is WIP.");
}

template<> csrOutput cuSparse_matrix_multiply<float>(
    const csrMatrixRef<float>& lhs,
    const csrMatrixRef<float>& rhs,
    Tensor &output_values, 
    IntTensor &output_indices);

template<> csrOutput cuSparse_matrix_multiply<double>(
    const csrMatrixRef<double>& lhs,
    const csrMatrixRef<double>& rhs,
    Tensor &output_values, 
    IntTensor &output_indices);

#else

csrOutput Sgemm2(
    const ScsrMatrixRef& A,
    const ScsrMatrixRef& B,
    const ScsrMatrixRef& C,
    const float* alpha,
    const float* beta,
    Tensor &output_values, 
    IntTensor &output_indices) {
  cusparseHandle_t cusparseHandle_;
  csrgemm2Info_t gemm2Info_;
  static void* buffer_{nullptr};
  static size_t currentBufferSize_{0};

  TORCH_CUDASPARSE_CHECK(cusparseCreate(&cusparseHandle_));
  TORCH_CUDASPARSE_CHECK(cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST));
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsrgemm2Info(&gemm2Info_));

  csrOutput out({A.size(0), B.size(1)});

  int innerSize = confirm_mult_size(A.size_, B.size_);

  out.csr_pointers_ = at::empty({out.size(0) + 1}, output_indices.options().dtype(kInt));

  // Compute needed buffer size
  size_t new_bubber_sz;
  TORCH_CUDASPARSE_CHECK(cusparseScsrgemm2_bufferSizeExt(
      cusparseHandle_,
      out.size(0),
      out.size(1),
      innerSize,
      alpha,
      A.description_,
      A.nnz_,
      A.csr_pointers_,
      A.csr_indices_,
      B.description_,
      B.nnz_,
      B.csr_pointers_,
      B.csr_indices_,
      beta,
      C.description_,
      C.nnz_,
      C.csr_pointers_,
      C.csr_indices_,
      gemm2Info_,
      &new_bubber_sz));

  // (Re)allocate buffer if needed
  if (new_bubber_sz > currentBufferSize_) {
    if (buffer_ != NULL) {
      cudaFree(buffer_);
    }
    cudaMalloc(&buffer_, new_bubber_sz);
    currentBufferSize_ = new_bubber_sz;
  }

  // Find the resulting non-zero pattern.
  TORCH_CUDASPARSE_CHECK(cusparseXcsrgemm2Nnz(
      cusparseHandle_,
      out.size(0),
      out.size(1),
      innerSize,
      A.description_,
      A.nnz_,
      A.csr_pointers_,
      A.csr_indices_,
      B.description_,
      B.nnz_,
      B.csr_pointers_,
      B.csr_indices_,
      C.description_,
      C.nnz_,
      C.csr_pointers_,
      C.csr_indices_,
      out.description_,
      out.csr_pointers_.data_ptr<int>(),
      &out.nnz_,
      gemm2Info_,
      buffer_));

  out.csr_indices_ = at::empty({out.nnz_}, output_indices.options().dtype(kInt));
  out.csr_values_ = at::empty({out.nnz_}, output_values.options());

  // Perform the gemm2 operation for doubles
  // out = alpha ∗ A ∗ B + beta ∗ C
  TORCH_CUDASPARSE_CHECK(cusparseScsrgemm2(
      cusparseHandle_,
      out.size(0),
      out.size(1),
      innerSize,
      alpha,
      A.description_,
      A.nnz_,
      A.csr_values_,
      A.csr_pointers_,
      A.csr_indices_,
      B.description_,
      B.nnz_,
      B.csr_values_,
      B.csr_pointers_,
      B.csr_indices_,
      beta,
      C.description_,
      C.nnz_,
      C.csr_values_,
      C.csr_pointers_,
      C.csr_indices_,
      out.description_,
      out.csr_values_.data_ptr<float>(),
      out.csr_pointers_.data_ptr<int>(),
      out.csr_indices_.data_ptr<int>(),
      gemm2Info_,
      buffer_));

  TORCH_CUDASPARSE_CHECK(cusparseDestroy(cusparseHandle_));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyCsrgemm2Info(gemm2Info_));
  return out;
}


csrOutput Dgemm2(
    const DcsrMatrixRef& A,
    const DcsrMatrixRef& B,
    const DcsrMatrixRef& C,
    const double* alpha,
    const double* beta,
    Tensor &output_values, 
    IntTensor &output_indices) {
  cusparseHandle_t cusparseHandle_;
  csrgemm2Info_t gemm2Info_;
  static void* buffer_{nullptr};
  static size_t currentBufferSize_{0};

  TORCH_CUDASPARSE_CHECK(cusparseCreate(&cusparseHandle_));
  TORCH_CUDASPARSE_CHECK(cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST));
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsrgemm2Info(&gemm2Info_));

  csrOutput out({A.size(0), B.size(1)});
  int innerSize = confirm_mult_size(A.size_, B.size_);
  out.csr_pointers_ = at::empty({out.size(0) + 1}, output_indices.options().dtype(kInt));

  // Compute needed buffer size
  size_t new_bubber_sz;
  TORCH_CUDASPARSE_CHECK(cusparseDcsrgemm2_bufferSizeExt(
      cusparseHandle_,
      out.size(0),
      out.size(1),
      innerSize,
      alpha,
      A.description_,
      A.nnz_,
      A.csr_pointers_,
      A.csr_indices_,
      B.description_,
      B.nnz_,
      B.csr_pointers_,
      B.csr_indices_,
      beta,
      C.description_,
      C.nnz_,
      C.csr_pointers_,
      C.csr_indices_,
      gemm2Info_,
      &new_bubber_sz));

  // (Re)allocate buffer if needed
  if (new_bubber_sz > currentBufferSize_) {
    if (buffer_ != NULL) {
      cudaFree(buffer_);
    }
    cudaMalloc(&buffer_, new_bubber_sz);
    currentBufferSize_ = new_bubber_sz;
  }

  // Find the resulting non-zero pattern.
  TORCH_CUDASPARSE_CHECK(cusparseXcsrgemm2Nnz(
      cusparseHandle_,
      out.size(0),
      out.size(1),
      innerSize,
      A.description_,
      A.nnz_,
      A.csr_pointers_,
      A.csr_indices_,
      B.description_,
      B.nnz_,
      B.csr_pointers_,
      B.csr_indices_,
      C.description_,
      C.nnz_,
      C.csr_pointers_,
      C.csr_indices_,
      out.description_,
      out.csr_pointers_.data_ptr<int>(),
      &out.nnz_,
      gemm2Info_,
      buffer_));

  out.csr_indices_ = at::empty({out.nnz_}, output_indices.options().dtype(kInt));
  out.csr_values_ = at::empty({out.nnz_}, output_values.options());

  // Perform the gemm2 operation for doubles
  // out = alpha ∗ A ∗ B + beta ∗ C
  TORCH_CUDASPARSE_CHECK(cusparseDcsrgemm2(
      cusparseHandle_,
      out.size(0),
      out.size(1),
      innerSize,
      alpha,
      A.description_,
      A.nnz_,
      A.csr_values_,
      A.csr_pointers_,
      A.csr_indices_,
      B.description_,
      B.nnz_,
      B.csr_values_,
      B.csr_pointers_,
      B.csr_indices_,
      beta,
      C.description_,
      C.nnz_,
      C.csr_values_,
      C.csr_pointers_,
      C.csr_indices_,
      out.description_,
      out.csr_values_.data_ptr<double>(),
      out.csr_pointers_.data_ptr<int>(),
      out.csr_indices_.data_ptr<int>(),
      gemm2Info_,
      buffer_));

  TORCH_CUDASPARSE_CHECK(cusparseDestroy(cusparseHandle_));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyCsrgemm2Info(gemm2Info_));
  return out;
}

template<class scalar_t>
csrOutput cuSparse_matrix_multiply(
    const csrMatrixRef<scalar_t>& lhs,
    const csrMatrixRef<scalar_t>& rhs,
    Tensor &output_values, 
    IntTensor &output_indices)
{
  TORCH_INTERNAL_ASSERT(false, "cusparse csr sparse-sparse MM only supports data type of float and double.");
}

template<>
csrOutput cuSparse_matrix_multiply<double>(
    const DcsrMatrixRef& lhs,
    const DcsrMatrixRef& rhs,
    Tensor &output_values, 
    IntTensor &output_indices) {
  double alpha = 1.0;
  DcsrMatrixRef empty;
  return Dgemm2(lhs, rhs, empty, &alpha, nullptr, output_values, output_indices);
}

template<>
csrOutput cuSparse_matrix_multiply<float>(
    const ScsrMatrixRef& lhs,
    const ScsrMatrixRef& rhs,
    Tensor &output_values, 
    IntTensor &output_indices) {
  float alpha = 1.0;
  ScsrMatrixRef empty;
  return Sgemm2(lhs, rhs, empty, &alpha, nullptr, output_values, output_indices);
}
 
#endif

template <typename scalar_t>
void sparse_sparse_matmul_cuda_kernel(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2) {

  static_assert(std::is_same<float, scalar_t>::value || std::is_same<double, scalar_t>::value, 
    "sparse_sparse_matmul_cuda_kernel only supports float and double value types");
  
  LongTensor mat1_indices_ = mat1._indices().contiguous();
  Tensor mat1_values = mat1._values().contiguous();

  LongTensor mat1_row_indices = mat1_indices_.select(0, 0);
  LongTensor mat1_col_indices = mat1_indices_.select(0, 1);

  IntTensor mat1_indptr = _to_csr_int(mat1_row_indices, mat1.size(0), mat1._nnz());
  
  IntTensor mat1_indices = at::empty(
      {mat1_col_indices.size(0)}, mat1_col_indices.options().dtype(kInt));
  
  mat1_indices.copy_(mat1_col_indices);

  LongTensor mat2_indices_ = mat2._indices().contiguous();
  Tensor mat2_values = mat2._values().contiguous();
  LongTensor mat2_row_indices = mat2_indices_.select(0, 0);
  LongTensor mat2_col_indices = mat2_indices_.select(0, 1);

  IntTensor mat2_indptr = _to_csr_int(mat2_row_indices, mat2.size(0), mat2._nnz());
  IntTensor mat2_indices = at::empty({mat2_col_indices.size(0)}, mat2_col_indices.options().dtype(kInt));
  mat2_indices.copy_(mat2_col_indices);

  auto m = mat1.size(0);
  auto k1 = mat1.size(1);

  auto k2 = mat2.size(0);
  auto n = mat2.size(1);

  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k1 <= INT_MAX),
    "At the moment, cusparseDcsrgemm2 only supports m, n, k, nnz with the bound [val] <= ", INT_MAX, ".",
    "If you need this, please file an issue on GitHub."
  );

  csrMatrixRef<scalar_t> csr_mat1(
      mat1_indices.data_ptr<int>(),
      mat1_indptr.data_ptr<int>(),
      mat1_values.data_ptr<scalar_t>(),
      (int)mat1._nnz(),
      {(int)mat1.size(0), (int)mat1.size(1)});

  csrMatrixRef<scalar_t> csr_mat2(
      mat2_indices.data_ptr<int>(),
      mat2_indptr.data_ptr<int>(),
      mat2_values.data_ptr<scalar_t>(),
      (int)mat2._nnz(),
      {(int)mat2.size(0), (int)mat2.size(1)});

  auto output_indices = result._indices();
  auto output_values = result._values();

  // Sparse matrix multiplication
  csrOutput csr_output = cuSparse_matrix_multiply<scalar_t>(csr_mat1, csr_mat2, output_values, output_indices); 
  auto nnz = csr_output.nnz_;

  output_values.set_(csr_output.csr_values_);
  output_indices.resize_({2, nnz});
  auto output_indices_accessor = output_indices.packed_accessor<int64_t, 2>();

  auto csr_output_pointers_accessor =
      csr_output.csr_pointers_.packed_accessor<int, 1>();

  auto csr_output_ind_accessor =
      csr_output.csr_indices_.packed_accessor<int, 1>();

  auto major_dim = result.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  thrust::for_each(
      policy,
      thrust::make_counting_iterator(int64_t(0)),
      thrust::make_counting_iterator(int64_t(major_dim)),
      [output_indices_accessor,
       csr_output_pointers_accessor,
       major_dim,
       nnz] __device__(int64_t i) {
        auto Ap = csr_output_pointers_accessor.data();
        int64_t* indices_row = output_indices_accessor[0].data();

        for (int jj = Ap[i];  jj < Ap[i + 1]; jj++) {
          indices_row[jj] = i;
        }
      });

  thrust::for_each(
    policy,
    thrust::make_counting_iterator(int64_t(0)),
    thrust::make_counting_iterator(int64_t(csr_output.nnz_)),
    [output_indices_accessor,
      csr_output_pointers_accessor,
      csr_output_ind_accessor,
      major_dim,
      nnz] __device__(int64_t i) {
      int64_t* indices_col = output_indices_accessor[1].data();
      indices_col[i] = csr_output_ind_accessor[i];
    });
}

template <typename scalar_t, bool grad_by_row>
void sparse_matmul_kernel_grad(Tensor& output, const Tensor& grad, const Tensor& x) {
  /* 
    Computes  the backward output  for matrix C = A*B.

    C = A@B 
      then 
    A_grad = C_grad @ B^T
    B_grad = A^T @ C_grad

    if grad_by_row == true:
      output = x^T @ C_grad 
    else:
      output = C_grad @ x^T 
  */
  Tensor grad_filled = at::ones(grad.sizes(), grad.options().layout(kStrided));
  if (grad_by_row) {
    sparse_sparse_matmul_cuda_kernel<scalar_t>(output, x.transpose(0, 1).coalesce(), grad_filled.to_sparse());
  } else {
    sparse_sparse_matmul_cuda_kernel<scalar_t>(output, grad_filled.to_sparse(), x.transpose(0, 1).coalesce());
  }
}

} // end anonymous namespace

Tensor sparse_sparse_matmul_cuda(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);

  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

  Tensor output =
      at::native::empty_sparse({mat1_.size(0), mat2_.size(1)}, mat1_.options());

  AT_DISPATCH_FLOATING_TYPES(mat1_.scalar_type(), "sparse_matmul", [&] {
    sparse_sparse_matmul_cuda_kernel<scalar_t>(output, mat1_.coalesce(), mat2_.coalesce());
  });
  return output;
}

Tensor sparse_sparse_matmul_backward_cuda(
    const Tensor& grad,
    const Tensor& var,
    int64_t mult_order) {
  TORCH_CHECK(
      mult_order == 0 || mult_order == 1,
      ": mult_order not in [0, 1] at sparse_matmul_backward_cpu function");
  Tensor output = at::native::empty_like(grad);
  if (mult_order == 0) {
    std::vector<int64_t> size = {var.size(1), grad.size(1)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(size.size(), 0, size);

    AT_DISPATCH_FLOATING_TYPES(
      output.scalar_type(), "sparse_matmul_kernel_grad_by_row", [&] {
        sparse_matmul_kernel_grad<scalar_t, true>(output,  grad, var);
      });
  } else if (mult_order == 1) {
    std::vector<int64_t> size = {grad.size(0), var.size(0)};
    at::sparse::get_sparse_impl(output)->resize_and_clear_(size.size(), 0, size);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "sparse_matmul_kernel_grad_by_col", [&] {
          sparse_matmul_kernel_grad<scalar_t, false>(output, grad, var);
        });
  }
  return output;
}

} // namespace native
} // namespace at
