#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_DIRECT_SYCL_NCHW_KERNELS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_DIRECT_SYCL_NCHW_KERNELS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"

#include "tensorflow/core/kernels/conv_ops_direct_sycl_kernels.h"
#include "tensorflow/core/kernels/conv_ops_direct_sycl_nchw_kernels.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace direct {
template <typename T, ConvType CType>
struct Conv2DNCHW;
template <typename T>
struct Conv2DNCHW<T, ConvType::Forward> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DNCHW(Index n_elems,
                                               const SYCLConv2DParams& params,
                                               const read_accessor input,
                                               const read_accessor kernel,
                                               write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);

    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      Index batch = index;

      const Index cstart =
          (batch % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
      const Index cend = cl::sycl::min(
          cstart + (p_.window_cols_ * p_.dilation_cols_), p_.in_cols_);
      batch /= p_.out_cols_;

      const Index rstart =
          (batch % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
      const Index rend = cl::sycl::min(
          rstart + (p_.window_rows_ * p_.dilation_rows_), p_.in_rows_);
      batch /= p_.out_rows_;

      const Index feature = batch % p_.features_;
      batch /= p_.features_;

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + batch * p_.in_cols_ * p_.in_rows_ * p_.channels_;
      for (Index r = rstart, i = 0; r < rend; ++r, ++i) {
        if (r >= 0) {
          for (Index c = cstart, j = 0; c < cend; ++c, ++j) {
            if (c >= 0) {
              for (Index channel = 0; channel < p_.channels_; ++channel) {
                const Index idx = (channel * p_.in_rows_ + r) * p_.in_cols_ + c;
                const Index k_idx = p_.kernel_index(channel, feature, i, j);
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct Conv2DNCHW<T, ConvType::InputBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DNCHW(Index n_elems,
                                               const SYCLConv2DParams& params,
                                               const read_accessor input,
                                               const read_accessor kernel,
                                               write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);
    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      Index n = index;
      // c is the index in the padded output tensor (ie with lots of extra
      // zeros),
      // but without the first padding. first_padded_c adds this extra padding.
      const Index c = (n % p_.in_cols_) + p_.pad_cols_;
      const Index first_padded_c = c - p_.window_cols_ + 1;
      // The first and last output indices affected by this input.
      const Index last_used_c = c / p_.stride_cols_;
      const Index first_used_c =
          RoundRatioUpAboveZero(first_padded_c, p_.stride_cols_);

      const Index firstc = first_used_c * p_.stride_cols_ - first_padded_c;
      const Index cstart = cl::sycl::max(first_used_c, static_cast<Index>(0));
      const Index cend = cl::sycl::min(last_used_c + 1, p_.out_cols_);
      n /= p_.in_cols_;

      const Index r = (n % p_.in_rows_) + p_.pad_rows_;
      const Index last_used_r = r / p_.stride_rows_;
      const Index first_padded_r = r - p_.window_rows_ + 1;
      const Index first_used_r =
          RoundRatioUpAboveZero(first_padded_r, p_.stride_rows_);

      const Index firstr = first_used_r * p_.stride_rows_ - first_padded_r;
      const Index rstart = cl::sycl::max(first_used_r, static_cast<Index>(0));
      const Index rend = cl::sycl::min(last_used_r + 1, p_.out_rows_);
      n /= p_.in_rows_;

      const Index feature = n % p_.features_;
      const Index batch = n / p_.features_;

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + batch * p_.out_cols_ * p_.out_rows_ * p_.channels_;
      for (Index channel = 0; channel < p_.channels_; ++channel) {
        for (Index r = rstart, i = firstr; r < rend;
             ++r, i += p_.stride_rows_) {
          for (Index c = cstart, j = firstc; c < cend;
               ++c, j += p_.stride_cols_) {
            const Index idx = (channel * p_.out_rows_ + r) * p_.out_cols_ + c;
            const Index k_idx = p_.backprop_index(feature, channel, i, j);
            out_val += input_data_n[idx] * kernel_data[k_idx];
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
/*
 * The main difference between the two backprop kernels is the way strides are
 * handled. In the filter backprop the input is strided and the kernel is not
 * whereas in the input backprop this is the other way around.
 */
template <typename T>
struct Conv2DNCHW<T, ConvType::FilterBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DNCHW(Index n_elems,
                                               const SYCLConv2DParams& params,
                                               const read_accessor input,
                                               const read_accessor kernel,
                                               write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);
    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
      const SYCL2DKernelWindow w = p_.kernel_window_from_output(index);
      const Index filter_rows =
          RoundRatioUpAboveZero(p_.window_rows_, p_.stride_rows_);
      const Index filter_cols =
          RoundRatioUpAboveZero(p_.window_cols_, p_.stride_cols_);

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data;
      for (Index b = 0; b < p_.batch_; b++) {
        for (Index r = w.rstart, i = 0; r < w.rend; ++i, r += p_.stride_rows_) {
          if (r >= 0) {
            for (Index c = w.cstart, j = 0; c < w.cend;
                 ++j, c += p_.stride_cols_) {
              if (c >= 0) {
                const Index idx =
                    (w.channel * p_.in_rows_ + r) * p_.in_cols_ + c;

                const Index k_idx =
                    ((b * p_.features_ + w.feature) * filter_rows + i) *
                        filter_cols +
                    j;
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
        input_data_n += p_.in_cols_ * p_.in_rows_ * p_.channels_;
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
}  // namespace direct
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_DIRECT_SYCL_NCHW_KERNELS_H_
