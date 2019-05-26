#include <TH/THBlas.h>
#include <c10/core/ScalarType.h>

// This header file shouldn't be anything permanent; it's just a temporary
// dumping ground to help you get access to utilities in THBlas.h via templates,
// rather than by name directly.  Someone should figure out a reasonable way to
// rewrite these in more idiomatic ATen and move it into ATen proper.

template<typename T>
inline void THBlas_axpy(int64_t n, T a, T *x, int64_t incx, T *y, int64_t incy);

#define AXPY_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void THBlas_axpy<ctype>(int64_t n, ctype a, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_axpy(n, a, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(AXPY_SPECIALIZATION)


template<typename T>
inline void THBlas_copy(int64_t n, T *x, int64_t incx, T *y, int64_t incy);

#define COPY_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void THBlas_copy<ctype>(int64_t n, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_copy(n, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(COPY_SPECIALIZATION)

template<typename T>
inline T THBlas_dot(int64_t n, T *x, int64_t incx, T *y, int64_t incy);

#define DOT_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline ctype THBlas_dot<ctype>(int64_t n, ctype *x, int64_t incx, ctype *y, int64_t incy) { \
    return TH ## name ## Blas_dot(n, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(DOT_SPECIALIZATION)

template<typename T>
inline void THBlas_gemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    T alpha,
    T *a,
    int64_t lda,
    T *b,
    int64_t ldb,
    T beta,
    T *c,
    int64_t ldc);

#define GEMM_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void THBlas_gemm<ctype>( \
      char transa, \
      char transb, \
      int64_t m, \
      int64_t n, \
      int64_t k, \
      ctype alpha, \
      ctype *a, \
      int64_t lda, \
      ctype *b, \
      int64_t ldb, \
      ctype beta, \
      ctype *c, \
      int64_t ldc) { \
    TH ## name ## Blas_gemm(\
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(GEMM_SPECIALIZATION)

template<typename T>
inline void THBlas_gemv(
    char transa,
    int64_t m,
    int64_t n,
    T alpha,
    T *a,
    int64_t lda,
    T *x,
    int64_t incx,
    T beta,
    T *y,
    int64_t incy);

#define GEMV_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void THBlas_gemv<ctype>( \
      char transa, \
      int64_t m, \
      int64_t n, \
      ctype alpha, \
      ctype *a, \
      int64_t lda, \
      ctype *x, \
      int64_t incx, \
      ctype beta, \
      ctype *y, \
      int64_t incy) { \
    TH ## name ## Blas_gemv(\
      transa, m, n, alpha, a, lda, x, incx, beta, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(GEMV_SPECIALIZATION)

/*
  THBlas_im2col and THBlas_col2im are provided here temporarily and
  should be removed once im2col and col2im are ported to ATen native.
 */

template<typename T>
inline void THBlas_im2col(
                     const T* data_im, const int64_t channels,
                     const int64_t height, const int64_t width,
                     const int64_t output_height, const int64_t output_width,
                     const int64_t kernel_h, const int64_t kernel_w,
                     const int64_t pad_h, const int64_t pad_w,
                     const int64_t stride_h, const int64_t stride_w,
                     const int64_t dilation_h, const int64_t dilation_w,
                     T* data_col
                        );

#define IM2COL_SPECIALIZATION(ctype,name,_1)                            \
  template<>                                                            \
  inline void THBlas_im2col<ctype>(                                     \
    const ctype* data_im, const int64_t channels,                       \
    const int64_t height, const int64_t width,                          \
    const int64_t output_height, const int64_t output_width,            \
    const int64_t kernel_h, const int64_t kernel_w,                     \
    const int64_t pad_h, const int64_t pad_w,                           \
    const int64_t stride_h, const int64_t stride_w,                     \
    const int64_t dilation_h, const int64_t dilation_w,                 \
    ctype* data_col                                                     \
                                                                        ) { \
    TH ## name ## Blas_im2col(data_im, channels, height, width,        \
                              output_height, output_width, kernel_h, kernel_w, \
                              pad_h, pad_w, stride_h, stride_w, \
                              dilation_h, dilation_w, data_col); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(IM2COL_SPECIALIZATION)

template<typename T>
inline void THBlas_col2im(
                        const T* data_col, const int64_t channels,
                        const int64_t height, const int64_t width,
                        const int64_t output_height, const int64_t output_width,
                        const int64_t kernel_h, const int64_t kernel_w,
                        const int64_t pad_h, const int64_t pad_w,
                        const int64_t stride_h, const int64_t stride_w,
                        const int64_t dilation_h, const int64_t dilation_w,
                        T* data_im
                        );

#define COL2IM_SPECIALIZATION(ctype,name,_1)                            \
  template<>                                                            \
  inline void THBlas_col2im<ctype>(                                       \
                                 const ctype* data_col, const int64_t channels, \
                                 const int64_t height, const int64_t width, \
                                 const int64_t output_height, const int64_t output_width, \
                                 const int64_t kernel_h, const int64_t kernel_w, \
                                 const int64_t pad_h, const int64_t pad_w, \
                                 const int64_t stride_h, const int64_t stride_w, \
                                 const int64_t dilation_h, const int64_t dilation_w, \
                                 ctype* data_im                         \
                                                                        ) { \
    TH ## name ## Blas_col2im(data_col, channels, height, width,        \
                              output_height, output_width, kernel_h, kernel_w, \
                              pad_h, pad_w, stride_h, stride_w, \
                              dilation_h, dilation_w, data_im); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(COL2IM_SPECIALIZATION)
