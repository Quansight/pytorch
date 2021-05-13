#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/native/TensorIterator.h>
#include <THC/THCIntegerDivider.cuh>

// If element_sizes is nullptr, then the strides will be in bytes, otherwise
// the strides will be in # of elements.
// Operands that share the same shape, but may have different strides.
// OffsetCalculator iterates the tensor in a column-major order

#ifdef __HIP_PLATFORM_HCC__
constexpr int MAX_DIMS = 16;
#else
constexpr int MAX_DIMS = 25;
#endif

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.
  using offset_type = at::detail::Array<index_t, std::max<int>(NARGS, 1)>;

  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(const int dims,
                   const int64_t* const sizes,
                   const int64_t* const* const strides,
                   const int64_t* const element_sizes=nullptr)
    : dims(dims),
      sizes_([sizes, dims] {
              TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
              std::remove_const_t<decltype(sizes_)> ret;
              for (int i=0; i < dims; i++){
                ret[i] = IntDivider<index_t>(sizes[i]);
               }
               return ret;
            }()),
      strides_([strides, dims, element_sizes] {
              std::remove_const_t<decltype(strides_)> ret;
              for (int i=0; i < dims; i++){
                for (int arg = 0; arg < NARGS; arg++) {
                  const int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
                  ret[i][arg] = strides[arg][i] / element_size;
                }
              }
              return ret;
            }()) {}

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }

    }
    return offsets;
  }

  const int dims;
  const at::detail::Array<IntDivider<index_t>, MAX_DIMS> sizes_;
  const at::detail::Array<at::detail::Array<index_t, std::max<int>(NARGS, 1)>, MAX_DIMS> strides_;
};

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // The offsets are in # of elements, not in bytes.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.
  using offset_type = at::detail::Array<index_t, std::max<int>(NARGS, 1)>;

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};

template<int N>
static OffsetCalculator<N> make_offset_calculator(const at::TensorIteratorBase& iter) {
  AT_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data());
}
