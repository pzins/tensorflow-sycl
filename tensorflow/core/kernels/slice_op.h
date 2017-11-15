/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_SLICE_OP_H_
#define TENSORFLOW_KERNELS_SLICE_OP_H_

// Functor definition for SliceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {

namespace internal {

template <typename Device, typename T>
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices);
template <typename Device, typename T>
void SliceSimpleGpu(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices);

#ifdef TENSORFLOW_USE_SYCL
namespace functor {
template <typename Scalar>
struct Slice {
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Slice(const int32 n_elems,
                                          const read_accessor input,
                                          const read_accessor indices,
                                          const int32 ndims,
                                          write_accessor output)
      : n_elems_(n_elems),
        input_accessor_(input),
        indices_accessor_(indices),
        ndims_(ndims),
        output_accessor_(output) {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const int32 index = item.get(0);
    if (index < n_elems_) {
      Scalar* output_data = ConvertToActualTypeSycl(Scalar, output_accessor_);
      Scalar* input_data = ConvertToActualTypeSycl(Scalar, input_accessor_);

      const int32* in_strides =
          ConvertToActualTypeSycl(int32, indices_accessor_);
      const int32* slice_indices =
          ConvertToActualTypeSycl(int32, indices_accessor_) + ndims_ * 2;
      int32* out_strides =
          ConvertToActualTypeSycl(int32, indices_accessor_) + ndims_;

      int32 t = index;
      int32 i_idx = 0;
      for (int i = 0; i < ndims_; ++i) {
        i_idx += (t / out_strides[i] + slice_indices[i]) * in_strides[i];
        t %= out_strides[i];
      }

      output_data[index] = *(input_data + i_idx);
    }
  }

 private:
  int32 n_elems_;
  const read_accessor input_accessor_;
  const read_accessor indices_accessor_;
  const int32 ndims_;
  write_accessor output_accessor_;
};
}

template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundUpToNearestMultiple(const IntegerType val, const IntegerType multiplier) {
  const IntegerType diff = val % multiplier;
  return val + (multiplier - diff);
}

template <typename Device, typename T>
void SliceSimpleSycl(const Device& d, Tensor* out, const Tensor& in,
                     const gtl::ArraySlice<int64>& slice_indices) {
  // Ensures we can use 32-bit index.
  const int64 in_nelem = in.NumElements();
  CHECK_LT(in_nelem, kint32max) << "Tensor too large to transpose on SYCL";
  const int64 out_nelem = out->NumElements();
  CHECK_LT(out_nelem, kint32max) << "Tensor too large to transpose on SYCL";
  // Pack strides and slice indices sizes into one buffer.
  const int32 ndims = in.dims();
  gtl::InlinedVector<int32, 24> host_buf(ndims * 3);
  gtl::InlinedVector<int32, 8> in_strides = ComputeStride<int32>(in.shape());
  gtl::InlinedVector<int32, 8> out_strides = ComputeStride<int32>(out->shape());
  for (int i = 0; i < ndims; ++i) {
    host_buf[i] = in_strides[i];
    host_buf[ndims + i] = out_strides[i];
    host_buf[ndims * 2 + i] = slice_indices[i];
  }
  auto num_bytes = sizeof(int64) * host_buf.size();

  auto dev_indices = static_cast<int32*>(d.allocate(num_bytes));
  d.memcpyHostToDevice(dev_indices, static_cast<int32*>(host_buf.data()),
                       num_bytes);
  // Launch kernel to q[...] = p[...].
  const T* p = in.flat<T>().data();
  T* q = out->flat<T>().data();

  auto input_buffer = d.get_sycl_buffer(p);
  auto indices_buffer = d.get_sycl_buffer(dev_indices);

  auto output_buffer = d.get_sycl_buffer(q);

  const int64 workgroup_size = d.maxSyclThreadsPerBlock();
  const int64 n_threads = RoundUpToNearestMultiple(out_nelem, workgroup_size);

  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;

  d.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    auto input_access = input_buffer.template get_access<read_mode>(cgh);
    auto index_access = indices_buffer.template get_access<read_mode>(cgh);
    auto output_access = output_buffer.template get_access<write_mode>(cgh);

    functor::Slice<T> slice(out_nelem, input_access, index_access, ndims,
                            output_access);
    cgh.parallel_for(cl::sycl::range<1>(n_threads), slice);
  });
  // Safe to deallocate immediately after the kernel launch.
  d.deallocate(dev_indices);
}
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices) {
  const int ndims = in.dims();
  const int64 nelem = out->NumElements();
  const gtl::InlinedVector<int64, 8> in_strides = ComputeStride<int64>(in.shape());
  const gtl::InlinedVector<int64, 8> out_strides = ComputeStride<int64>(out->shape());
  const T* p = in.flat<T>().data();
  T* q = out->flat<T>().data();

  std::vector<int64> i_idx(nelem, 0);
  std::vector<int64> t(nelem, 0);

  for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
    t[o_idx] = o_idx;
  }
  for (int i = 0; i < ndims; ++i) {
    int64 n = (nelem + 7) / 8;
    int64 o_idx = 0;
    switch (nelem % 8) {
#define CALC_INPUT_IDX                                                            \
  i_idx[o_idx] += (t[o_idx] / out_strides[i] + slice_indices[i]) * in_strides[i]; \
  t[o_idx] %= out_strides[i];                                                     \
  ++o_idx;
      case 0: do { CALC_INPUT_IDX;
      case 7:      CALC_INPUT_IDX;
      case 6:      CALC_INPUT_IDX;
      case 5:      CALC_INPUT_IDX;
      case 4:      CALC_INPUT_IDX;
      case 3:      CALC_INPUT_IDX;
      case 2:      CALC_INPUT_IDX;
      case 1:      CALC_INPUT_IDX;
#undef CALC_INPUT_IDX
              } while (--n > 0);
    }
  }
  for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
    q[o_idx] = p[i_idx[o_idx]];
  }
}

template <typename Device, typename T, int NDIMS>
void SliceUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices,
                 const gtl::ArraySlice<int64>& slice_sizes) {
  auto input = in.tensor<T, NDIMS>();
  auto output = out->tensor<T, NDIMS>();
  Eigen::DSizes<int, NDIMS> indices;
  for (int i = 0; i < NDIMS; ++i) {
    indices[i] = slice_indices[i];
  }
  Eigen::DSizes<int, NDIMS> sizes;
  for (int i = 0; i < NDIMS; ++i) {
    sizes[i] = slice_sizes[i];
  }
  const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
  if (!use_64bit &&
      (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value ||
       Eigen::internal::is_same<Device, Eigen::SyclDevice>::value)) {
    To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
  } else {
    output.device(d) = input.slice(indices, sizes);
  }
}

} // namespace internal

namespace functor {

// Template parameter NDIM is not neccesary here. The aim of keeping it
// is to compile struct slice seperately which minimizes the compiling time.
template <typename Device, typename T, int NDIM>
struct Slice {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<int64>& slice_indices,
                  const gtl::ArraySlice<int64>& slice_sizes) {
    if (in.dims() == NDIM) {
        internal::SliceUsingEigen<Device, T, NDIM>(d, out, in, slice_indices, slice_sizes);
    } else {
        if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
          internal::SliceSimpleGpu<Device, T>(d, out, in, slice_indices);
        } else {
          internal::SliceSimple<Device, T>(d, out, in, slice_indices);
        }
    }
  }
};

#ifdef TENSORFLOW_USE_SYCL
template <typename T, int NDIM>
struct Slice<Eigen::SyclDevice, T, NDIM> {
  void operator()(const Eigen::SyclDevice& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<int64>& slice_indices,
                  const gtl::ArraySlice<int64>& slice_sizes) {
    if (in.dims() == NDIM) {
      internal::SliceUsingEigen<Eigen::SyclDevice, T, NDIM>(
          d, out, in, slice_indices, slice_sizes);
    } else {
      internal::SliceSimpleSycl<Eigen::SyclDevice, T>(d, out, in,
                                                      slice_indices);
    }
  }
};
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
