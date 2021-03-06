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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(UnaryOp, CPU, "Inv", functor::inverse, float, Eigen::half, double,
          complex64, complex128);
#if GOOGLE_CUDA
REGISTER4(UnaryOp, GPU, "Inv", functor::inverse, float, Eigen::half, double,
          int64);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER(UnaryOp, SYCL, "Inv", functor::inverse, int64);
#define REGISTER_SYCL(type) \
  REGISTER(UnaryOp, SYCL, "Inv", functor::inverse, type)
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL);
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL

REGISTER5(SimpleBinaryOp, CPU, "InvGrad", functor::inverse_grad, float,
          Eigen::half, double, complex64, complex128);
#if GOOGLE_CUDA
REGISTER3(SimpleBinaryOp, GPU, "InvGrad", functor::inverse_grad, float,
          Eigen::half, double);
#endif
#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(type) \
  REGISTER(SimpleBinaryOp, SYCL, "InvGrad", functor::inverse_grad, type)
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL);
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL

REGISTER5(UnaryOp, CPU, "Reciprocal", functor::inverse, float, Eigen::half,
          double, complex64, complex128);
#if GOOGLE_CUDA
REGISTER4(UnaryOp, GPU, "Reciprocal", functor::inverse, float, Eigen::half,
          double, int64);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER(UnaryOp, SYCL, "Reciprocal", functor::inverse, int64);
#define REGISTER_SYCL(type) \
  REGISTER(UnaryOp, SYCL, "Reciprocal", functor::inverse, type)
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL);
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL

REGISTER5(SimpleBinaryOp, CPU, "ReciprocalGrad", functor::inverse_grad, float,
          Eigen::half, double, complex64, complex128);
#if GOOGLE_CUDA
REGISTER3(SimpleBinaryOp, GPU, "ReciprocalGrad", functor::inverse_grad, float,
          Eigen::half, double);
#endif
#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(type) \
  REGISTER(SimpleBinaryOp, SYCL, "ReciprocalGrad", functor::inverse_grad, type)
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL);
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
