#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_im2col_sycl_kernels.h"
#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace im2col {
struct TileInfo {
  int rows;
  int cols;
  int size;
};
template <ConvType CType>
inline TileInfo get_tile_info(SYCLConv2DParams const& params);
template <>
inline TileInfo get_tile_info<ConvType::Forward>(
    SYCLConv2DParams const& params) {
  const int n_tile_rows = params.out_rows_;
  const int n_tile_cols = params.out_cols_;
  const int tile_size =
      params.window_rows_ * params.window_cols_ * params.channels_;
  const TileInfo result{n_tile_rows, n_tile_cols, tile_size};
  return result;
}
template <>
inline TileInfo get_tile_info<ConvType::InputBackprop>(
    SYCLConv2DParams const& params) {
  const int n_tile_rows = params.in_rows_;
  const int n_tile_cols = params.in_cols_;
  const int tile_size =
      params.window_rows_ * params.window_cols_ * params.features_;
  const TileInfo result{n_tile_rows, n_tile_cols, tile_size};
  return result;
}
struct AllocInfo {
  size_t alloc_limit;
  size_t images_per_alloc;
  size_t n_input_transforms;
  size_t last_batch_size;
};
template <ConvType CType>
inline AllocInfo get_alloc_info(cl::sycl::device const& device,
                                SYCLConv2DParams const& params,
                                size_t const alloc_size_per_image) {
  size_t alloc_limit =
      device.get_info<cl::sycl::info::device::max_mem_alloc_size>();
  if (TF_PREDICT_FALSE(alloc_size_per_image > alloc_limit)) {
    LOG(WARNING) << "The temporary buffer required by im2col for a single "
                    "image is too large to be allocated on the device. This "
                    "is likely to cause a CL_MEM_OBJECT_ALLOCATION_FAILURE "
                    "OpenCL error.";
    VLOG(2) << "buffer size per image: " << alloc_size_per_image
            << ", device allocation limit: " << alloc_limit;
    alloc_limit = alloc_size_per_image + 1;
  }
  // The number of images per alloc is bounded above by the total number of
  // images in a batch
  size_t images_per_alloc =
      std::min<size_t>(params.batch_, alloc_limit / alloc_size_per_image);

  const size_t n_input_transforms =
      RoundRatioUpAboveZero<size_t>(params.batch_, images_per_alloc);
  images_per_alloc =
      RoundRatioUpAboveZero<size_t>(params.batch_, n_input_transforms);
  assert(images_per_alloc * alloc_size_per_image < alloc_limit);
  const size_t last_batch_size =
      params.batch_ - images_per_alloc * (n_input_transforms - 1);
  assert(last_batch_size > 0);
  assert(last_batch_size <= params.batch_);
  const AllocInfo result{alloc_limit, images_per_alloc, n_input_transforms,
                         last_batch_size};
  return result;
}
}
template <typename T, ConvType CType>
struct LaunchIm2Col;
template <typename T>
struct LaunchIm2Col<T, ConvType::Forward> {
  using Index = int;
  static constexpr auto CType = ConvType::Forward;
  using InputTransform = im2col::ExtractInputTiles<T, CType>;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     const SYCLConv2DParams& params) {
    cl::sycl::queue& sycl_queue = device.sycl_queue();
    cl::sycl::device const& sycl_device = sycl_queue.get_device();

    const im2col::TileInfo tile_info = im2col::get_tile_info<CType>(params);

    const size_t alloc_size_per_image =
        tile_info.rows * tile_info.cols * tile_info.size * sizeof(T);
    const im2col::AllocInfo alloc_info = im2col::get_alloc_info<CType>(
        sycl_device, params, alloc_size_per_image);
    size_t images_per_alloc = alloc_info.images_per_alloc;

    // When the number of input transforms required is greater than one, it is
    // because each transform buffer is right on the limit of what can be
    // allocated on the device. This means that we need to reuse the same
    // buffer for each iteration of the algorithm, as enqueuing more than one
    // of these huge buffers can cause CL_MEM_OBJECT_ALLOCATION_FAILURE errors.
    // Reusing the same buffer will add a little overhead, as the iterations
    // cannot be run in parallel however it is required to avoid the allocation
    // failures.
    // TODO(jwlawson): Remove when sycl queue can handle this for us.
    const size_t transform_size = alloc_size_per_image * images_per_alloc;
    T* const transform = static_cast<T*>(device.allocate(transform_size));

    for (int i = 0; i < alloc_info.n_input_transforms; ++i) {
      const Index in_offset = i * images_per_alloc * params.in_rows_ *
                              params.in_cols_ * params.channels_;
      const Index out_offset = i * images_per_alloc * params.out_rows_ *
                               params.out_cols_ * params.features_;
      assert(i == 0 || in_offset > 0);
      assert(i == 0 || out_offset > 0);
      if (i == alloc_info.n_input_transforms - 1) {
        images_per_alloc = alloc_info.last_batch_size;
      }
      SYCLConv2DParams kernel_params{params};
      kernel_params.batch_ = images_per_alloc;
      const Index n_tiles = images_per_alloc * tile_info.rows * tile_info.cols;
      const size_t actual_transform_size =
          alloc_size_per_image * images_per_alloc;

      device.memset(transform, 0, actual_transform_size);
      const Index in_transform_items = images_per_alloc * params.in_rows_ *
                                       params.in_cols_ * params.channels_;
      sycl_conv::launch_transform<InputTransform>(
          device, input, transform, in_transform_items, kernel_params,
          in_offset, tile_info.size);
      sycl_conv::launch_matmul<false, false>(
          device, transform, filter, output + out_offset, static_cast<T>(0),
          n_tiles, tile_info.size, params.features_);
    }
    // At the moment we have to explicitly wait here to ensure that the device
    // queue is cleared before enqueuing the kernels which use huge buffers so
    // that we do not hit any memory allocation failures.
    // TODO(jwlawson): Remove wait when SYCL queue handles allocation waiting.
    device.synchronize();
    device.deallocate(transform);
    return true;
  }
};
template <typename T>
struct LaunchIm2Col<T, ConvType::InputBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::InputBackprop;
  using FilterTransform = im2col::ExtractKernelTiles<T, CType>;
  using InputTransform = im2col::ExtractInputTiles<T, CType>;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     const SYCLConv2DParams& params) {
    cl::sycl::queue& sycl_queue = device.sycl_queue();
    cl::sycl::device const& sycl_device = sycl_queue.get_device();

    const im2col::TileInfo tile_info = im2col::get_tile_info<CType>(params);

    const size_t alloc_size_per_image =
        tile_info.rows * tile_info.cols * tile_info.size * sizeof(T);
    const im2col::AllocInfo alloc_info = im2col::get_alloc_info<CType>(
        sycl_device, params, alloc_size_per_image);
    size_t images_per_alloc = alloc_info.images_per_alloc;

    const size_t filter_size = params.window_rows_ * params.window_cols_ *
                               params.channels_ * params.features_;
    const size_t filter_size_bytes = filter_size * sizeof(T);
    T* const filter_transform =
        static_cast<T*>(device.allocate(filter_size_bytes));
    const Index fil_transform_items = params.window_rows_ *
                                      params.window_cols_ * params.channels_ *
                                      params.features_;
    sycl_conv::launch_transform<FilterTransform>(
        device, filter, filter_transform, fil_transform_items, params, 0);

    // When the number of input transforms required is greater than one, it is
    // because each transform buffer is right on the limit of what can be
    // allocated on the device. This means that we need to reuse the same
    // buffer for each iteration of the algorithm, as enqueuing more than one
    // of these huge buffers can cause CL_MEM_OBJECT_ALLOCATION_FAILURE errors.
    // Reusing the same buffer will add a little overhead, as the iterations
    // cannot be run in parallel however it is required to avoid the allocation
    // failures.
    // TODO(jwlawson): Remove when sycl queue can handle this for us.
    const size_t transform_size = alloc_size_per_image * images_per_alloc;
    T* const transform = static_cast<T*>(device.allocate(transform_size));

    for (int i = 0; i < alloc_info.n_input_transforms; ++i) {
      const Index in_offset = i * images_per_alloc * params.out_rows_ *
                              params.out_cols_ * params.features_;
      const Index out_offset = i * images_per_alloc * params.in_rows_ *
                               params.in_cols_ * params.channels_;
      if (i == alloc_info.n_input_transforms - 1) {
        images_per_alloc = alloc_info.last_batch_size;
      }
      SYCLConv2DParams kernel_params{params};
      kernel_params.batch_ = images_per_alloc;
      const Index n_tiles = images_per_alloc * tile_info.rows * tile_info.cols;
      const size_t actual_transform_size =
          alloc_size_per_image * images_per_alloc;

      device.memset(transform, 0, actual_transform_size);
      const Index in_transform_items = images_per_alloc * params.out_rows_ *
                                       params.out_cols_ * params.features_;
      sycl_conv::launch_transform<InputTransform>(
          device, input, transform, in_transform_items, kernel_params,
          in_offset, tile_info.size);
      sycl_conv::launch_matmul<false, false>(
          device, transform, filter_transform, output + out_offset,
          static_cast<T>(0), n_tiles, tile_info.size, params.channels_);
    }
    // At the moment we have to explicitly wait here to ensure that the device
    // queue is cleared before enqueuing the kernels which use huge buffers so
    // that we do not hit any memory allocation failures.
    // TODO(jwlawson): Remove wait when SYCL queue handles allocation waiting.
    device.synchronize();
    device.deallocate(transform);
    device.deallocate(filter_transform);
    return true;
  }
};
template <typename T>
struct LaunchIm2Col<T, ConvType::FilterBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::FilterBackprop;
  using InputTransform = im2col::ExtractInputTiles<T, CType>;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     const SYCLConv2DParams& params) {
    cl::sycl::queue& sycl_queue = device.sycl_queue();
    cl::sycl::device const& sycl_device = sycl_queue.get_device();
    const Index n_tiles =
        params.window_rows_ * params.window_cols_ * params.channels_;

    const size_t alloc_size_per_image =
        params.out_rows_ * params.out_cols_ * n_tiles * sizeof(T);
    const im2col::AllocInfo alloc_info = im2col::get_alloc_info<CType>(
        sycl_device, params, alloc_size_per_image);
    size_t images_per_alloc = alloc_info.images_per_alloc;

    SYCLConv2DParams kernel_params{params};
    kernel_params.out_rows_ = params.window_rows_;
    kernel_params.out_cols_ = params.window_cols_;
    kernel_params.window_rows_ = params.out_rows_;
    kernel_params.window_cols_ = params.out_cols_;
    kernel_params.dilation_rows_ = params.stride_rows_;
    kernel_params.dilation_cols_ = params.stride_cols_;
    kernel_params.stride_rows_ = 1;
    kernel_params.stride_cols_ = 1;

    // When the number of input transforms required is greater than one, it is
    // because each transform buffer is right on the limit of what can be
    // allocated on the device. This means that we need to reuse the same
    // buffer for each iteration of the algorithm, as enqueuing more than one
    // of these huge buffers can cause CL_MEM_OBJECT_ALLOCATION_FAILURE errors.
    // Reusing the same buffer will add a little overhead, as the iterations
    // cannot be run in parallel however it is required to avoid the allocation
    // failures.
    // TODO(jwlawson): Remove buffer reuse when sycl queue can handle this.
    const size_t transform_size = alloc_size_per_image * images_per_alloc;
    T* const transform = static_cast<T*>(device.allocate(transform_size));

    for (int i = 0; i < alloc_info.n_input_transforms; ++i) {
      const Index in_offset = i * images_per_alloc * params.in_rows_ *
                              params.in_cols_ * params.channels_;
      const Index out_offset = i * images_per_alloc * params.out_rows_ *
                               params.out_cols_ * params.features_;
      if (i == alloc_info.n_input_transforms - 1) {
        images_per_alloc = alloc_info.last_batch_size;
      }
      const Index tile_size =
          images_per_alloc * params.out_rows_ * params.out_cols_;
      const size_t actual_transform_size =
          alloc_size_per_image * images_per_alloc;
      kernel_params.batch_ = images_per_alloc;

      device.memset(transform, 0, actual_transform_size);
      const Index in_transform_items = images_per_alloc * params.in_rows_ *
                                       params.in_cols_ * params.channels_;
      sycl_conv::launch_transform<InputTransform>(
          device, input, transform, in_transform_items, kernel_params,
          in_offset, tile_size);
      if (i == 0) {
        sycl_conv::launch_matmul<false, false>(
            device, transform, filter + out_offset, output, static_cast<T>(0),
            n_tiles, tile_size, params.features_);
      } else {
        sycl_conv::launch_matmul<false, false>(
            device, transform, filter + out_offset, output, static_cast<T>(1),
            n_tiles, tile_size, params.features_);
      }
      device.synchronize();
    }
    // At the moment we have to explicitly wait here to ensure that the device
    // queue is cleared before enqueuing the kernels which use huge buffers so
    // that we do not hit any memory allocation failures.
    // TODO(jwlawson): Remove wait when SYCL queue handles allocation waiting.
    device.synchronize();
    device.deallocate(transform);
    return true;
  }
};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::im2col, CType> final
    : public LaunchIm2Col<T, CType> {};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_
