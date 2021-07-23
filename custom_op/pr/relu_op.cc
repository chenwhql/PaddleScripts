#include <iostream>
#include <vector>

#include "paddle/extension.h"

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(
    const paddle::Tensor& grad_out, 
    const paddle::Tensor& out,
    const paddle::Tensor& x);


std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  // Add check for inputs
  return relu_cuda_forward(x);
}

std::vector<paddle::Tensor> ReluBackward(
    const paddle::Tensor& grad_out, 
    const paddle::Tensor& out,
    const paddle::Tensor& x) {
  // Add check for inputs
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
