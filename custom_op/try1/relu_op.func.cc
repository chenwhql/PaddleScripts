#include <iostream>

#include "paddle/extension.h"

// template <typename T>
std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  auto out = paddle::Tensor();
  out.Resize(x.dims());
  
  auto* x_data = x.data<float>();
  auto* out_data = out.mutable_data<float>(paddle::CPUPlace());
 
  for (int i = 0; i < x.numel(); ++i) {
    out_data[i] = std::max(static_cast<float>(0.), x_data[i]);
  }

  return {out};
}

// template <typename T>
std::vector<paddle::Tensor> ReluBackward(
    const paddle::Tensor& grad_out, 
    const paddle::Tensor& out,
    const paddle::Tensor& x) {
  auto grad_x = paddle::Tensor();
  grad_x.Resize(x.dims());

  auto* grad_out_data = grad_out.data<float>();
  auto* out_data = out.data<float>();
  auto* x_data = x.data<float>();
  auto* grad_x_data = grad_x.mutable_data<float>(paddle::CPUPlace());
  
  for (int i = 0; i < out.numel(); ++i) {
      grad_x_data[i] = grad_out_data[i] * (out_data[i] > static_cast<float>(0) ? 1. : 0.);
  }
  
  return {grad_x};
}

REGISTER_CUSTOM_OPERATOR(relu2, ReluForward, ReluBackward);

// template std::vector<paddle::Tensor> ReluForward<float>(const paddle::Tensor& x);

// register op
// paddle::RegisterOperator("custom_relu", paddle::FuncParser(ReluForward<float>));
// registr op kernel