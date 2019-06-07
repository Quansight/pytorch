#pragma once

#include <cstring>

template <typename Dtype>
void im2col(
            const Dtype* data_im, const int64_t channels,
            const int64_t height, const int64_t width,
            const int64_t output_height, const int64_t output_width,
            const int64_t kernel_h, const int64_t kernel_w,
            const int64_t pad_h, const int64_t pad_w,
            const int64_t stride_h, const int64_t stride_w,
            const int64_t dilation_h, const int64_t dilation_w,
            Dtype* data_col
            ) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;
    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
      for (int64_t w_col = 0; w_col < width_col; ++w_col) {
        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        data_col[(c_col * height_col + h_col) * width_col + w_col] =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
          data_im[(c_im * height + h_im) * width + w_im] : 0;
      }
    }
  }
}

template <typename Dtype>
void col2im(
            const Dtype* data_col, const int64_t channels,
            const int64_t height, const int64_t width,
            const int64_t output_height, const int64_t output_width,
            const int64_t kernel_h, const int64_t kernel_w,
            const int64_t pad_h, const int64_t pad_w,
            const int64_t stride_h, const int64_t stride_w,
            const int64_t dilation_h, const int64_t dilation_w,
            Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * height * width * channels);
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;
    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
      for (int64_t w_col = 0; w_col < width_col; ++w_col) {
        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
            data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
  }
}
