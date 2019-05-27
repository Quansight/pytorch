#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THBlas.h"
#else

/* Level 1 */
TH_API void THBlas_(
    swap)(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy);
TH_API void THBlas_(scal)(int64_t n, scalar_t a, scalar_t* x, int64_t incx);
TH_API void THBlas_(
    copy)(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy);
TH_API void THBlas_(axpy)(
    int64_t n,
    scalar_t a,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy);
TH_API scalar_t THBlas_(
    dot)(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy);

/* Level 2 */
TH_API void THBlas_(gemv)(
    char trans,
    int64_t m,
    int64_t n,
    scalar_t alpha,
    scalar_t* a,
    int64_t lda,
    scalar_t* x,
    int64_t incx,
    scalar_t beta,
    scalar_t* y,
    int64_t incy);
TH_API void THBlas_(ger)(
    int64_t m,
    int64_t n,
    scalar_t alpha,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy,
    scalar_t* a,
    int64_t lda);

/* Level 3 */
TH_API void THBlas_(gemm)(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    scalar_t alpha,
    scalar_t* a,
    int64_t lda,
    scalar_t* b,
    int64_t ldb,
    scalar_t beta,
    scalar_t* c,
    int64_t ldc);

/*
  THBlas_(im2col) and THBlas_(col2im) are provided here temporarily
  and should be removed once im2col and col2im are ported to ATen
  native.
 */
TH_API void THBlas_(col2im)(
    const scalar_t* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    scalar_t* data_im);

TH_API void THBlas_(im2col)(
    const scalar_t* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    scalar_t* data_col);

TH_API void THBlas_(col2vol)(
    const scalar_t* data_col,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t output_depth,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_d,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_d,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_d,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_d,
    const int64_t dilation_h,
    const int64_t dilation_w,
    scalar_t* data_im);

TH_API void THBlas_(vol2col)(
    const scalar_t* data_im,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t output_depth,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_d,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_d,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_d,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_d,
    const int64_t dilation_h,
    const int64_t dilation_w,
    scalar_t* data_col);

#endif
