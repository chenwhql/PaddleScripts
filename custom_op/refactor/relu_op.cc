#include <iostream>
#include <vector>

#include "paddle/extension.h"

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out);

std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  // TODO(chenweihang): Check Input
  return relu_cuda_forward(x);
}

std::vector<paddle::Tensor> ReluBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out) {
  // TODO(chenweihang): Check Input
  return relu_cuda_backward(x, out, grad_out);
}

std::vector<std::vector<int64_t>> ReluInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> ReluInferDType(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OPERATOR("relu2")
  .Inputs({"X"})
  .Outputs({"Out"})
  .SetKernelFn(PD_KERNEL(ReluForward))
  .SetInferShapeFn(PD_INFER_SHAPE(ReluInferShape))
  .SetInferDtypeFn(PD_INFER_DTYPE(ReluInferDType))
  .SetBackwardOp("relu2_grad")
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluBackward));

