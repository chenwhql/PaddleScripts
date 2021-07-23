#include <iostream>
#include <vector>

#include "paddle/extension.h"

template <typename data_t>
void print(const data_t* x_data, int64_t x_numel) {
  std::cout << "Tensor data: ";
  for(int i = 0; i < x_numel; ++i) {
    std::cout << x_data[i] << " ";
  }
  std::cout << std::endl;
}

std::vector<paddle::Tensor> Print(const paddle::Tensor& x) {
  PD_DISPATCH_FLOATING_TYPES_AND(paddle::DataType::INT32, x.type(), "print", ([&] {
    print<data_t>(x.data<data_t>(), x.size());
  }));
  return {};
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> InferDType(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP("print_float_and")
  .Inputs({"X"})
  .SetKernelFn(PD_KERNEL(Print))
  .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
  .SetInferDtypeFn(PD_INFER_DTYPE(InferDType));
