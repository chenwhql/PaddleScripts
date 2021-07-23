#include <iostream>
#include <vector>

#include "paddle/extension.h"

std::vector<paddle::CustomTensor> relu_cuda_forward(const paddle::CustomTensor& x);
std::vector<paddle::CustomTensor> relu_cuda_backward(const paddle::CustomTensor& grad_out,
                                               const paddle::CustomTensor& out,
                                               const paddle::CustomTensor& x);

std::vector<paddle::CustomTensor> ReluForward(const paddle::CustomTensor& x) {
  // TODO(chenweihang): Check Input
  return relu_cuda_forward(x);
}

std::vector<paddle::CustomTensor> ReluBackward(const paddle::CustomTensor& grad_out,
                                         const paddle::CustomTensor& out,
                                         const paddle::CustomTensor& x) {
  // TODO(chenweihang): Check Input
  return relu_cuda_backward(grad_out, out, x);
}

std::vector<std::vector<int64_t>> ReluInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

BUILD_OPERATOR(relu2,
               OP_INFO(ReluForward),
               PD_KERNEL(ReluForward),
               PD_KERNEL(ReluBackward),
               PD_INFER_SHAPE(ReluInferShape));
