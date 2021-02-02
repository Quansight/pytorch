#include <math.h>
#include <vector>
#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/IndexingUtils.h>

namespace at {
namespace native {
namespace {

template<typename scalar_t>
static inline void compute_source_index_and_lambda(
    int64_t& input_index0,
    int64_t& input_index1,
    scalar_t& lambda0,
    scalar_t& lambda1,
    scalar_t ratio,
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1, simply copy
    input_index0 = output_index;
    input_index1 = output_index;
    lambda0 = static_cast<scalar_t>(1);
    lambda1 = static_cast<scalar_t>(0);
  } else {
    const scalar_t real_input_index = area_pixel_compute_source_index<scalar_t>(
        ratio, output_index, align_corners, /*cubic=*/false);
    input_index0 = static_cast<int64_t>(real_input_index);
    int64_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;
    lambda1 = real_input_index - input_index0;
    lambda0 = static_cast<scalar_t>(1.) - lambda1;
  }
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear(
    Tensor& output_,
    const Tensor& input_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto input_sizes = input.sizes().vec();
  auto output_sizes = output.sizes().vec();
  auto ndim = input_sizes.size();
  auto numel = output.numel();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  int64_t output_slice_size = output_depth * output_height * output_width;

  auto loop1d = [&](int64_t begin, int64_t end) {
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[0]);

    auto input_indexr = [=](int64_t c, int64_t w) {
      return input_data[c * input_width + w];
    };

    int64_t iw0, iw1;
    scalar_t w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t ow = 0; ow < output_width; ow++) {
        compute_source_index_and_lambda(
            iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
        int64_t output_offset = c * output_slice_size + ow;
        output_data[output_offset] =
            w0lambda * input_indexr(c, iw0) + /* w0 * i0 */
            w1lambda * input_indexr(c, iw1);  /* w1 * i1 */
      }
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t c, int64_t h, int64_t w) {
      return input_data[c * input_height * input_width + h * input_width + w];
    };

    int64_t ih0, ih1, iw0, iw1;
    scalar_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (int64_t ow = 0; ow < output_width; ow++) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
          int64_t output_offset = c * output_slice_size + oh * output_width + ow;
          output_data[output_offset] =
              h0lambda * w0lambda * input_indexr(c, ih0, iw0) + /* h0 * w0 * i00 */
              h0lambda * w1lambda * input_indexr(c, ih0, iw1) + /* h0 * w1 * i01 */
              h1lambda * w0lambda * input_indexr(c, ih1, iw0) + /* h1 * w0 * i10 */
              h1lambda * w1lambda * input_indexr(c, ih1, iw1);  /* h1 * w1 * i11 */
        }
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    const scalar_t depth_scale = area_pixel_compute_scale<scalar_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[1]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[2]);

    auto input_indexr = [=](int64_t c, int64_t d, int64_t h, int64_t w) {
      return input_data[c * input_depth * input_height * input_width +
          d * input_height * input_width + h * input_width + w];
    };

    int64_t id0, id1, ih0, ih1, iw0, iw1;
    scalar_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t od = 0; od < output_depth; od++) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (int64_t oh = 0; oh < output_height; oh++) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (int64_t ow = 0; ow < output_width; ow++) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
            int64_t output_offset = c * output_slice_size +
                od * output_height * output_width + oh * output_width + ow;
            output_data[output_offset] =
                d0lambda * h0lambda * w0lambda * input_indexr(c, id0, ih0, iw0) + /* d0 * h0 * w0 * i000 */
                d0lambda * h0lambda * w1lambda * input_indexr(c, id0, ih0, iw1) + /* d0 * h0 * w1 * i001 */
                d0lambda * h1lambda * w0lambda * input_indexr(c, id0, ih1, iw0) + /* d0 * h1 * w0 * i010 */
                d0lambda * h1lambda * w1lambda * input_indexr(c, id0, ih1, iw1) + /* d0 * h1 * w1 * i011 */
                d1lambda * h0lambda * w0lambda * input_indexr(c, id1, ih0, iw0) + /* d1 * h0 * w0 * i100 */
                d1lambda * h0lambda * w1lambda * input_indexr(c, id1, ih0, iw1) + /* d1 * h0 * w1 * i101 */
                d1lambda * h1lambda * w0lambda * input_indexr(c, id1, ih1, iw0) + /* d1 * h1 * w0 * i110 */
                d1lambda * h1lambda * w1lambda * input_indexr(c, id1, ih1, iw1);  /* d1 * h1 * w1 * i111 */
          }
        }
      }
    }
  };

  // compared to "nearest" mode, lower the grain size:
  // "linear", "bilinear", "trilinear" mode are more computational expensive
  if (ndim == 3) {
    // upsample linear 1d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 2, loop1d);
  } else if (ndim == 4){
    // upsample bilinear 2d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // upsample trilinear 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_channels_last(
    Tensor& output_,
    const Tensor& input_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(input_.dtype() == output_.dtype(), "expected dtype ", input_.dtype(),
              " for `output` but got dtype ", output_.dtype());

  auto input_sizes = input_.sizes().vec();
  auto output_sizes = output_.sizes().vec();
  auto ndim = input_sizes.size();
  TORCH_CHECK(ndim >=4 && ndim <= 5, "Upsample with NHWC format supports tensors with 4 or 5 dims.")

  auto channels_last_memory_format = ndim == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::ChannelsLast3d;
  auto input = input_.contiguous(channels_last_memory_format);
  auto output = output_.contiguous(channels_last_memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t num_batches =  input_sizes[0];
  int64_t channels =  input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  TORCH_CHECK(channels > 0, "expected input and output channels greater than 0 but got ", channels);
  int64_t output_slice_size = output_depth * output_height * output_width * channels;

  using Vec = vec256::Vec256<scalar_t>;
  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t n, int64_t h, int64_t w) {
      return input_data + n * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    int64_t ih0, ih1, iw0, iw1;
    scalar_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t n = begin; n < end; n++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (int64_t ow = 0; ow < output_width; ow++) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

          scalar_t* out = output_data + n * output_slice_size +
              oh * output_width * channels + ow * channels;
          scalar_t* i00 = input_indexr(n, ih0, iw0);
          scalar_t* i01 = input_indexr(n, ih0, iw1);
          scalar_t* i10 = input_indexr(n, ih1, iw0);
          scalar_t* i11 = input_indexr(n, ih1, iw1);

          int64_t size = channels;
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec out_vec =
                Vec(h0lambda * w0lambda) * Vec::loadu(i00 + d) + /* h0 * w0 * i00 */
                Vec(h0lambda * w1lambda) * Vec::loadu(i01 + d) + /* h0 * w1 * i01 */
                Vec(h1lambda * w0lambda) * Vec::loadu(i10 + d) + /* h1 * w0 * i10 */
                Vec(h1lambda * w1lambda) * Vec::loadu(i11 + d);  /* h1 * w1 * i11 */
            out_vec.store(out + d);
          }
          for (; d < size; d++) {
            out[d] =
                h0lambda * w0lambda * i00[d] + /* h0 * w0 * i00 */
                h0lambda * w1lambda * i01[d] + /* h0 * w1 * i01 */
                h1lambda * w0lambda * i10[d] + /* h1 * w0 * i10 */
                h1lambda * w1lambda * i11[d];  /* h1 * w1 * i11 */
          }
        }
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    const scalar_t depth_scale = area_pixel_compute_scale<scalar_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[1]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[2]);

    auto input_indexr = [=](int64_t n, int64_t d, int64_t h, int64_t w) {
      return input_data + n * input_depth * input_height * input_width * channels +
          d * input_height * input_width * channels +
          h * input_width * channels + w * channels;
    };

    int64_t id0, id1, ih0, ih1, iw0, iw1;
    scalar_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t n = begin; n < end; n++) {
      for (int64_t od = 0; od < output_depth; od++) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (int64_t oh = 0; oh < output_height; oh++) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (int64_t ow = 0; ow < output_width; ow++) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);

            scalar_t* out = output_data + n * output_slice_size +
                od * output_height * output_width * channels +
                oh * output_width * channels + ow * channels;
            scalar_t* i000 = input_indexr(n, id0, ih0, iw0);
            scalar_t* i001 = input_indexr(n, id0, ih0, iw1);
            scalar_t* i010 = input_indexr(n, id0, ih1, iw0);
            scalar_t* i011 = input_indexr(n, id0, ih1, iw1);
            scalar_t* i100 = input_indexr(n, id1, ih0, iw0);
            scalar_t* i101 = input_indexr(n, id1, ih0, iw1);
            scalar_t* i110 = input_indexr(n, id1, ih1, iw0);
            scalar_t* i111 = input_indexr(n, id1, ih1, iw1);

            int64_t size = channels;
            int64_t d = 0;
            for (; d < size - (size % Vec::size()); d += Vec::size()) {
              Vec out_vec =
                  Vec(d0lambda * h0lambda * w0lambda) * Vec::loadu(i000 + d) + /* d0 * h0 * w0 * i000 */
                  Vec(d0lambda * h0lambda * w1lambda) * Vec::loadu(i001 + d) + /* d0 * h0 * w1 * i001 */
                  Vec(d0lambda * h1lambda * w0lambda) * Vec::loadu(i010 + d) + /* d0 * h1 * w0 * i010 */
                  Vec(d0lambda * h1lambda * w1lambda) * Vec::loadu(i011 + d) + /* d0 * h1 * w1 * i011 */
                  Vec(d1lambda * h0lambda * w0lambda) * Vec::loadu(i100 + d) + /* d1 * h0 * w0 * i100 */
                  Vec(d1lambda * h0lambda * w1lambda) * Vec::loadu(i101 + d) + /* d1 * h0 * w1 * i101 */
                  Vec(d1lambda * h1lambda * w0lambda) * Vec::loadu(i110 + d) + /* d1 * h1 * w0 * i110 */
                  Vec(d1lambda * h1lambda * w1lambda) * Vec::loadu(i111 + d);  /* d1 * h1 * w1 * i111 */
              out_vec.store(out + d);
            }
            for (; d < size; d++) {
              out[d] =
                  d0lambda * h0lambda * w0lambda * i000[d] + /* d0 * h0 * w0 * i000 */
                  d0lambda * h0lambda * w1lambda * i001[d] + /* d0 * h0 * w1 * i001 */
                  d0lambda * h1lambda * w0lambda * i010[d] + /* d0 * h1 * w0 * i010 */
                  d0lambda * h1lambda * w1lambda * i011[d] + /* d0 * h1 * w1 * i011 */
                  d1lambda * h0lambda * w0lambda * i100[d] + /* d1 * h0 * w0 * i100 */
                  d1lambda * h0lambda * w1lambda * i101[d] + /* d1 * h0 * w1 * i101 */
                  d1lambda * h1lambda * w0lambda * i110[d] + /* d1 * h1 * w0 * i110 */
                  d1lambda * h1lambda * w1lambda * i111[d];  /* d1 * h1 * w1 * i111 */
            }
          }
        }
      }
    }
  };

  if (ndim == 4) {
    // upsample nearest 2d
    at::parallel_for(0, num_batches, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // upsample nearest 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, num_batches, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!output_.is_contiguous(channels_last_memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename scale_type>
void cpu_upsample_linear_backward(
    Tensor& grad_input_,
    const Tensor& grad_output_,
    bool align_corners,
    const scale_type& scales) {
  TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
              " for `grad_input` but got dtype ", grad_input_.dtype());

  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto input_sizes = grad_input.sizes().vec();
  auto output_sizes = grad_output.sizes().vec();
  auto ndim = input_sizes.size();

  // treat nbatch and channels as one dimension
  int64_t channels = input_sizes[0] * input_sizes[1];
  int64_t input_depth = (ndim == 5) ? input_sizes[2] : 1;
  int64_t output_depth = (ndim == 5) ? output_sizes[2] : 1;
  int64_t input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
  int64_t output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
  int64_t input_width = input_sizes[ndim - 1];
  int64_t output_width = output_sizes[ndim - 1];

  int64_t output_slice_size = output_depth * output_height * output_width;

  auto loop1d = [&](int64_t begin, int64_t end) {
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[0]);

    auto input_indexr = [=](int64_t c, int64_t w) {
      return grad_input_data + c * input_width + w;
    };

    int64_t iw0, iw1;
    scalar_t w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++){
      for (int64_t ow = 0; ow < output_width; ow++) {
        compute_source_index_and_lambda(
            iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
        scalar_t grad_output_value = grad_output_data[c * output_slice_size + ow];
        *input_indexr(c, iw0) += w0lambda * grad_output_value; /* i0 */
        *input_indexr(c, iw1) += w1lambda * grad_output_value; /* i1*/
      }
    }
  };

  auto loop2d = [&](int64_t begin, int64_t end) {
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[0]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[1]);

    auto input_indexr = [=](int64_t c, int64_t h, int64_t w){
      return grad_input_data + c * input_height * input_width + h * input_width + w;
    };

    int64_t ih0, ih1, iw0, iw1;
    scalar_t h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t oh = 0; oh < output_height; oh++) {
        compute_source_index_and_lambda(
            ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
        for (int64_t ow = 0; ow < output_width; ow++) {
          compute_source_index_and_lambda(
              iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
          scalar_t grad_output_value = grad_output_data[c * output_slice_size + oh * output_width + ow];
          *input_indexr(c, ih0, iw0) += h0lambda * w0lambda * grad_output_value; /* i00 */
          *input_indexr(c, ih0, iw1) += h0lambda * w1lambda * grad_output_value; /* i01 */
          *input_indexr(c, ih1, iw0) += h1lambda * w0lambda * grad_output_value; /* i10 */
          *input_indexr(c, ih1, iw1) += h1lambda * w1lambda * grad_output_value; /* i11 */
        }
      }
    }
  };

  auto loop3d = [&](int64_t begin, int64_t end) {
    const scalar_t depth_scale = area_pixel_compute_scale<scalar_t>(
        input_depth, output_depth, align_corners, scales[0]);
    const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
        input_height, output_height, align_corners, scales[1]);
    const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
        input_width, output_width, align_corners, scales[2]);

    auto input_indexr = [=](int64_t c, int64_t d, int64_t h, int64_t w) {
      return grad_input_data + c * input_depth * input_height * input_width +
          d * input_height * input_width + h * input_width + w;
    };

    int64_t id0, id1, ih0, ih1, iw0, iw1;
    scalar_t d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
    for (int64_t c = begin; c < end; c++) {
      for (int64_t od = 0; od < output_depth; od++) {
        compute_source_index_and_lambda(
            id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
        for (int64_t oh = 0; oh < output_height; oh++) {
          compute_source_index_and_lambda(
              ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
          for (int64_t ow = 0; ow < output_width; ow++) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
            scalar_t grad_output_value = grad_output_data[c * output_slice_size +
                od *  output_height * output_width + oh * output_width + ow];
            *input_indexr(c, id0, ih0, iw0) += d0lambda * h0lambda * w0lambda * grad_output_value; /* i000 */
            *input_indexr(c, id0, ih0, iw1) += d0lambda * h0lambda * w1lambda * grad_output_value; /* i001 */
            *input_indexr(c, id0, ih1, iw0) += d0lambda * h1lambda * w0lambda * grad_output_value; /* i010 */
            *input_indexr(c, id0, ih1, iw1) += d0lambda * h1lambda * w1lambda * grad_output_value; /* i011 */
            *input_indexr(c, id1, ih0, iw0) += d1lambda * h0lambda * w0lambda * grad_output_value; /* i100 */
            *input_indexr(c, id1, ih0, iw1) += d1lambda * h0lambda * w1lambda * grad_output_value; /* i101 */
            *input_indexr(c, id1, ih1, iw0) += d1lambda * h1lambda * w0lambda * grad_output_value; /* i110 */
            *input_indexr(c, id1, ih1, iw1) += d1lambda * h1lambda * w1lambda * grad_output_value; /* i111 */
          }
        }
      }
    }
  };

  if (ndim == 3) {
    // upsample linear 1d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 2, loop1d);
  } else if (ndim == 4) {
    // upsample bilinear 2d
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
  } else {
    // upsample trilinear 3d
    TORCH_INTERNAL_ASSERT(ndim == 5);
    at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
  }

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

using scale_t = std::vector<c10::optional<double>>;
void upsample_linear1d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_linear1d", [&] {
    cpu_upsample_linear<scalar_t, scale_t>(output, input, align_corners, {scales_w});
  });
}

void upsample_bilinear2d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_bilinear2d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_h, scales_w});
    });
  } else {

    // Previous dispatch
    // AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_bilinear2d", [&] {
    //   cpu_upsample_linear<scalar_t, scale_t>(output, input, align_corners, {scales_h, scales_w});
    // });
    
    ti_upsample_bilinear2d_kernel_impl();

  }
}

void upsample_trilinear3d_kernel_impl(
    Tensor& output,
    const Tensor& input,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  if (input.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_trilinear3d_channels_last", [&] {
      cpu_upsample_linear_channels_last<scalar_t, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_trilinear3d", [&] {
      cpu_upsample_linear<scalar_t, scale_t>(output, input, align_corners, {scales_d, scales_h, scales_w});
    });
  }
}

void upsample_linear1d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_linear1d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_w});
  });
}

void upsample_bilinear2d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_bilinear2d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
  });
}

void upsample_trilinear3d_backward_kernel_impl(
    Tensor& grad_input,
    const Tensor& grad_output,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_trilinear3d_backward", [&] {
    cpu_upsample_linear_backward<scalar_t, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
  });
}

// Helper method for ti_cpu_upsample_linear
// Load source data into a buffer
template <typename scalar_t, typename index_t, int step>
inline void load(scalar_t* dst, scalar_t *src, index_t * index) {
  for (int k = 0; k < step; k++) {
    *dst = *(src + *index);
    dst++;
    index++;
  }
}

// Helper method for ti_cpu_upsample_linear
// Compute output value as linear interpolation of source data
// with changing weights
template <typename scalar_t, int step>
inline void compute_linear(scalar_t* dst, scalar_t * src1, scalar_t* src2, float* w0, float* w1) {
  for (int k = 0; k < step; k++) {
    *dst = *src1 * *w0 + *src2 * *w1;
    dst++;
    src1++;
    src2++;
    w0++;
    w1++;
  }
}

// Helper method for ti_cpu_upsample_linear
// Overriden moethod to compute output value as linear interpolation of source data
// with fixed weights
template <typename scalar_t, int step>
inline void compute_linear(scalar_t* dst, scalar_t * src1, scalar_t* src2, float w0, float w1) {
  for (int k = 0; k < step; k++) {
    *dst = *src1 * w0 + *src2 * w1;
    dst++;
    src1++;
    src2++;
  }
}


// Interpolation type structure to compute output value in n-dimensional case.
// - use buffers (buf) to prefetch source data.
// - recursively compute interpolated output for each dimension
//
// for example for 2d:
// 
// source[0, 0] -> buffer[0]
// source[0, 1] -> buffer[1]
// interpolate(buffer[0], weight00, buffer[1], weight01) -> buffer[2]
//
// source[1, 0] -> buffer[0]
// source[1, 1] -> buffer[1]
// interpolate(buffer[0], weight00, buffer[1], weight01) -> buffer[3]
// 
// interpolate(buffer[2], weight10, buffer[3], weight11) -> output
//
template <int n, typename scalar_t, typename index_t, int step>
struct Interp {
    static inline void eval(scalar_t* out, scalar_t* buf, scalar_t* src[], index_t* idx[], float* w[]) {
        constexpr int i = 2 * (n - 1);
        constexpr int j = 2 * (n - 1) + 1;
        constexpr int is = i * step;
        constexpr int js = j * step;
        constexpr int half = 1 << (n - 2);
        Interp<n - 1, scalar_t, index_t, step>::eval(&buf[is], buf, &src[0], idx, &w[2]);        
        Interp<n - 1, scalar_t, index_t, step>::eval(&buf[js], buf, &src[half], idx, &w[2]); 
        compute_linear<scalar_t, step>(out, &buf[is], &buf[js], w[0][0], w[1][0]);
    }
};


template <typename scalar_t, typename index_t, int step>
struct Interp<1, scalar_t, index_t, step> {
    static inline void eval(scalar_t* out, scalar_t* buf, scalar_t* src[], index_t* idx[], float* w[]) {
      load<scalar_t, index_t, step>(&buf[0], src[0], idx[0]);
      load<scalar_t, index_t, step>(&buf[step], src[0], idx[1]);
      compute_linear<scalar_t, step>(out, &buf[0], &buf[step], w[0], w[1]);
    }
};


template <int n, typename scalar_t, typename index_t, int step>
static inline void interp(scalar_t* out, scalar_t* buf, scalar_t* src[], index_t* idx[], float* w[]) {
  Interp<n, scalar_t, index_t, step>::eval(out, buf, src, idx, w);
}


template <typename scalar_t, typename index_t, int out_ndims>
inline void assert_strides_linear(const int64_t* strides) {
  for (int i=0; i<out_ndims; i++) {
    // Assert strides for indices 0, 1 and weights 0, 1
    TORCH_INTERNAL_ASSERT(
      strides[4 * i + 0 + 2] == strides[4 * i + 2 + 2], strides[4 * i + 0 + 2], strides[4 * i + 2 + 2]        
    );
    TORCH_INTERNAL_ASSERT(
      strides[4 * i + 1 + 2] == strides[4 * i + 3 + 2], strides[4 * i + 1 + 2], strides[4 * i + 3 + 2]
    );
  }

  // Assert zero stride for indices and weights on dims -2, -3, ...
  for (int i=0; i<out_ndims - 1; i++) {
    TORCH_INTERNAL_ASSERT(strides[4 * i + 0 + 2] == 0, strides[4 * i + 0 + 2]);
    TORCH_INTERNAL_ASSERT(strides[4 * i + 1 + 2] == 0, strides[4 * i + 1 + 2]);
  }

  // Assert zero stride for the source
  TORCH_INTERNAL_ASSERT(strides[1] == 0, strides[1]);

  // Assert stride for the output
  TORCH_INTERNAL_ASSERT(strides[0] == sizeof(scalar_t), strides[0], sizeof(scalar_t));

  // Assert non-zero stride for indices and weights on dim -1
  int i = out_ndims - 1;
  TORCH_INTERNAL_ASSERT(strides[4 * i + 0 + 2] == sizeof(index_t), strides[4 * i + 0 + 2], sizeof(index_t));
  TORCH_INTERNAL_ASSERT(strides[4 * i + 1 + 2] == sizeof(float), strides[4 * i + 1 + 2], sizeof(float));  
}


// Linear upsampling computation method using TensorIterator for Nd case.
// 
// Assumptions:
// - input and output are of shape (B, C, D1, D2, D3, ..., DN) and
// - upsampling is computed on D_i axes.
// - zero strides for construced indices and weights on dims D1, D2, ..., DN-1
// - zero stride for input source (as it is restrided)
// - non-zero stride for indices and weights on DN dimension
// 
// Using these assumptions we iterate over DN dimension and compute the output 
// using the following tricks for optimizations:
// - indices are already containing strides
// - src pointer is advanced once by the constant offset for D1, D2, ..., DN-1
// - using buffers to prefetch src data before the computations
// 
// Single loop function for 1d, 2d and 3d cases.
// For N dimensions, output value up to Di dimension can be computed as
///
// output_i[a] = linear_interp(output_{i+1}[a], w_{i+1}[a], output_{i+1}[a+1], w_{i+1}[a+1])
// with
// output_DN[a] = linear_interp(input_DN[a], w_DN[a], input_DN[a+1], w_DN[a+1])
//
// This recursive call is implemented with Interp struct using template for 
// the loop unrolling on compile time.
// 
template <typename scalar_t, typename index_t, int out_ndims>
void ti_cpu_upsample_linear(at::TensorIterator& iter) {

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {

    scalar_t * dst = (scalar_t *) data[0];
    scalar_t * src = (scalar_t *) data[1];

    assert_strides_linear<scalar_t, index_t, out_ndims>(strides);

    constexpr int step = 4;
    constexpr int p2_size = 1 << (out_ndims - 1);

    // temporary buffer for src values
    scalar_t buffer[p2_size * step];

    // placeholder for pointers on indices for iterated dimension (e.g. -1)
    index_t * idx_ptrs[2];
    // placeholder for pointers on all weights: w0 and w1 for each dimension
    float * weights_ptrs[2 * out_ndims];
    // placeholder src pointer with all possible constant offsets added
    scalar_t * src_offset[p2_size];
    {
      index_t * constval_idx_ptrs[2 * (out_ndims - 1)];
      int i = 0;
      for (; i<out_ndims - 1; i++) {
        constval_idx_ptrs[2 * i + 0] = (index_t *) data[4 * i + 0 + 2];
        weights_ptrs[2 * i + 0] = (float *) data[4 * i + 1 + 2];
        constval_idx_ptrs[2 * i + 1] = (index_t *) data[4 * i + 2 + 2];
        weights_ptrs[2 * i + 1] = (float *) data[4 * i + 3 + 2];
      }
      idx_ptrs[0] = (index_t *) data[4 * i + 0 + 2];
      weights_ptrs[2 * i + 0] = (float *) data[4 * i + 1 + 2];
      idx_ptrs[1] = (index_t *) data[4 * i + 2 + 2];
      weights_ptrs[2 * i + 1] = (float *) data[4 * i + 3 + 2];

      // Add all constant offsets to src
      int dim_idx = 0;
      for (int j=0; j<p2_size; j++) {
        src_offset[j] = src;
        for (int i=0; i<out_ndims - 1; i++) {
          dim_idx = (j >> (out_ndims - 2 - i)) % 2;
          src_offset[j] += *constval_idx_ptrs[2 * i + dim_idx];
        }
      }
    }
    
    index_t i = 0;
    for (; i < n - (n % step); i += step) {      
      interp<out_ndims, scalar_t, index_t, step>(dst + i, buffer, src_offset, idx_ptrs, weights_ptrs);
      // Here we advance only on the last dimension (i.e. dim -1)
      idx_ptrs[0] += step;
      idx_ptrs[1] += step;
      weights_ptrs[2 * (out_ndims - 1) + 0] += step;
      weights_ptrs[2 * (out_ndims - 1) + 1] += step;
    }    
    for (; i < n; i++) {            
      interp<out_ndims, scalar_t, index_t, 1>(dst + i, buffer, src_offset, idx_ptrs, weights_ptrs);
      idx_ptrs[0] += 1;
      idx_ptrs[1] += 1;
      weights_ptrs[2 * (out_ndims - 1) + 0] += 1;
      weights_ptrs[2 * (out_ndims - 1) + 1] += 1;
    }
    
  };

  iter.for_each(loop);
}

template<typename index_t, typename scalar_t>
std::vector<at::Tensor> ti_compute_indices_weights_linear(
  int64_t input_size, int64_t output_size, int64_t stride, int64_t ndims, int64_t reshape_dim, 
  bool align_corners, const c10::optional<double> opt_scale
) {

  scalar_t scale = at::native::area_pixel_compute_scale<scalar_t>(input_size, output_size, align_corners, opt_scale);

  std::vector<at::Tensor> output;
  auto new_shape = std::vector<int64_t>(ndims, 1);
  new_shape[reshape_dim] = output_size;

  output.emplace_back(at::empty(new_shape, at::CPU(c10::CppTypeToScalarType<index_t>())));
  output.emplace_back(at::empty(new_shape, at::CPU(c10::CppTypeToScalarType<scalar_t>())));  
  output.emplace_back(at::empty(new_shape, at::CPU(c10::CppTypeToScalarType<index_t>())));
  output.emplace_back(at::empty(new_shape, at::CPU(c10::CppTypeToScalarType<scalar_t>())));

  auto input_index0_ptr = output[0].data_ptr<index_t>();
  auto lambda0_ptr = output[1].data_ptr<scalar_t>();
  auto input_index1_ptr = output[2].data_ptr<index_t>();
  auto lambda1_ptr = output[3].data_ptr<scalar_t>();

  double xd;
  int64_t xl;

  for (index_t i=0; i<output_size; i++) {

    compute_source_index_and_lambda<scalar_t, index_t>(
      input_index0_ptr[i], input_index1_ptr[i],
      lambda0_ptr[i], lambda1_ptr[i],
      scale, i, input_size, output_size, align_corners
    );
    // put stride into indices
    input_index0_ptr[i] *= stride;
    input_index1_ptr[i] *= stride;
  }
  return output;
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;


// Upsampling linear interpolation kernel for N-d case.
// Internally, it uses TensorIterator to optimize the computations.
// Output is constructed inside the function and is a contiguous tensor.
template <typename index_t, int out_ndims>
at::Tensor ti_upsample_linearNd_kernel_impl(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<c10::ArrayRef<double>> scale_factors = c10::nullopt) {

  if (output_size) {
    TORCH_CHECK(out_ndims == output_size->size());
  }  
  if (scale_factors) {
    TORCH_CHECK(out_ndims == scale_factors->size());
  }  
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);

  // input can be NCHW, NCL or NCKHW
  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();

  for (int i=0; i<out_ndims; i++) {
    shape[i + 2] = osize[i];
    strides[i + 2] = 0;
  }
  auto restrided_input = input.as_strided(shape, strides);

  // Compute indices and weights for each interpolated dimension
  // indices_weights = {
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -n
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -(n-1)
  //      ...
  //      {indices_0, weights_0, indices_1, weights_1},  // dim -1
  // }
  // Indices and weights are reshaped as (1, 1, ..., N, ..., 1, 1) to
  // fit input/output tensors.
  // Indices are already containing the strides to optimize the computations
  //
  // Indices dtype can be int32_t or int64_t depending on canUse32BitIndexMath(input)
  // which should not overflow because maximum possible value that it could take is the 
  // product of interpolated input strides: input_size[dim-1] * input_size[dim-2] * ...
  // which is always smaller then the number of input elements checked by canUse32BitIndexMath
  std::vector<std::vector<at::Tensor>> indices_weights;
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "compute_indices_weights_linear", [&] {
      for (int i=0; i<out_ndims; i++) {
        indices_weights.emplace_back(
          ti_compute_indices_weights_linear<index_t, scalar_t>(
            input.size(i + 2), osize[i], input.stride(i + 2), input.dim(), i + 2, align_corners, get_scale_value(scale_factors, i))
        );
      }
    }
  );

  at::TensorIteratorConfig config;
  config.check_all_same_dtype(false)
    .declare_static_dtype_and_device(input.scalar_type(), input.device())
    .add_output(at::Tensor())
    .add_input(restrided_input);
  
  for (auto iter=indices_weights.begin(); iter!=indices_weights.end(); iter++) { 
    for (auto& tensor : *iter) {
      config.add_input(tensor);
    }
  }

  auto iter = config.build();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "upsample_linearNd", [&] {
      ti_cpu_upsample_linear<scalar_t, index_t, out_ndims>(iter);
  });

  return iter.output();
}


at::Tensor ti_upsample_bilinear2d_kernel_impl(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<c10::ArrayRef<double>> scale_factors = c10::nullopt) {

  if (at::native::canUse32BitIndexMath(input))
    return ti_upsample_linearNd_kernel_impl<int32_t, 2>(
        input, output_size, align_corners, scale_factors);
  
  return ti_upsample_linearNd_kernel_impl<int64_t, 2>(
      input, output_size, align_corners, scale_factors);
}


at::Tensor ti_upsample_linear1d_kernel_impl(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<c10::ArrayRef<double>> scale_factors = c10::nullopt) {

  if (at::native::canUse32BitIndexMath(input))
    return ti_upsample_linearNd_kernel_impl<int32_t, 1>(
        input, output_size, align_corners, scale_factors);

  return ti_upsample_linearNd_kernel_impl<int64_t, 1>(
      input, output_size, align_corners, scale_factors);
}


at::Tensor ti_upsample_trilinear3d_kernel_impl(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<c10::ArrayRef<double>> scale_factors = c10::nullopt) {

  if (at::native::canUse32BitIndexMath(input))
    return ti_upsample_linearNd_kernel_impl<int32_t, 3>(
        input, output_size, align_corners, scale_factors);

  return ti_upsample_linearNd_kernel_impl<int64_t, 3>(
      input, output_size, align_corners, scale_factors);
}

} // anonymous namespace

REGISTER_DISPATCH(upsample_linear1d_kernel, &upsample_linear1d_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_kernel, &upsample_bilinear2d_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_kernel, &upsample_trilinear3d_kernel_impl);
REGISTER_DISPATCH(upsample_linear1d_backward_kernel, &upsample_linear1d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_bilinear2d_backward_kernel, &upsample_bilinear2d_backward_kernel_impl);
REGISTER_DISPATCH(upsample_trilinear3d_backward_kernel, &upsample_trilinear3d_backward_kernel_impl);

} // namespace native
} // namespace at
