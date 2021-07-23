#include <iostream>
#include <vector>

#include "paddle/extension.h"

template <typename DType, typename Place>
struct ReluForward {
  std::vector<paddle::Tensor> operator()(const paddle::Tensor& x) {
    auto out = paddle::Tensor();
    out.Resize(x.dims());
    
    auto* x_data = x.data<DType>();
    auto* out_data = out.mutable_data<DType>(Place());
  
    for (int i = 0; i < x.numel(); ++i) {
      out_data[i] = std::max(static_cast<DType>(0.), x_data[i]);
    }

    return {out};
  }
};

template <typename DType, typename Place>
struct ReluBackward {
  std::vector<paddle::Tensor> operator()(
      const paddle::Tensor& grad_out, 
      const paddle::Tensor& out,
      const paddle::Tensor& x) {
    auto grad_x = paddle::Tensor();
    grad_x.Resize(x.dims());

    auto* grad_out_data = grad_out.data<DType>();
    auto* out_data = out.data<DType>();
    auto* x_data = x.data<DType>();
    auto* grad_x_data = grad_x.mutable_data<DType>(Place());
    
    for (int i = 0; i < out.numel(); ++i) {
        grad_x_data[i] = grad_out_data[i] * (out_data[i] > static_cast<DType>(0) ? 1. : 0.);
    }
    
    return {grad_x};
  }
};

REGISTER_CUSTOM_OPERATOR(
    relu2, 
    ReluForward<float, paddle::CPUPlace>,
    ReluBackward<float, paddle::CPUPlace>);