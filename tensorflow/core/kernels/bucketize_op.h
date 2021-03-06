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

#ifndef TENSORFLOW_BUCKETIZE_OP_H_
#define TENSORFLOW_BUCKETIZE_OP_H_

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct BucketizeFunctor {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& input,
                        const std::vector<float>& boundaries_vector,
                        typename TTypes<int32, 1>::Tensor& output) {
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      auto first_bigger_it = std::upper_bound(
          boundaries_vector.begin(), boundaries_vector.end(), input(i));
      output(i) = first_bigger_it - boundaries_vector.begin();
    }

    return Status::OK();
	}
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_BUCKETIZE_OP_H_
