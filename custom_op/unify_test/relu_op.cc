#include <iostream>
#include <vector>

#include "paddle/extension.h"

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data, data_t* out_data, int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data, const data_t* out_data, data_t* grad_x_data, int64_t out_numel) {
  for (int i = 0; i < out_numel; ++i) {
      grad_x_data[i] = grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cpu_forward(const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PaddlePlace(paddle::PlaceType::kCPU));
  out.Reshape(x.shape());

  PD_DISPATCH_FLOATING_TYPES(x.type(), "relu_cpu_forward", ([&]{
      relu_cpu_forward_kernel<data_t>(
        x.data<data_t>(),
        out.mutable_data<data_t>(x.place()),
        x.size());
    }));

  return {out};
}

std::vector<paddle::Tensor> relu_cpu_backward(
    const paddle::Tensor& grad_out, 
    const paddle::Tensor& out,
    const paddle::Tensor& x) {
  auto grad_x = paddle::Tensor(paddle::PaddlePlace(paddle::PlaceType::kCPU));
  grad_x.Reshape(x.shape());

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward", ([&]{
      relu_cpu_backward_kernel<data_t>(
        grad_out.data<data_t>(),
        out.data<data_t>(),
        grad_x.mutable_data<data_t>(x.place()),
        out.size());
    }));

  return {grad_x};
}

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(
    const paddle::Tensor& grad_out, 
    const paddle::Tensor& out,
    const paddle::Tensor& x);


std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  TODO: Check Input
  if (paddle::platform::is_cpu_place(x.place())) {
    return relu_cpu_forward(x);
  } else if (paddle::platform::is_gpu_place(x.place())) {
    return relu_cuda_forward(x);
  } else {
    throw std::runtime_error("Not implemented.");
  }
  return relu_cpu_forward(x);
}

std::vector<paddle::Tensor> ReluBackward(
    const paddle::Tensor& grad_out, 
    const paddle::Tensor& out,
    const paddle::Tensor& x) {
  TODO: Check Input
  if (paddle::platform::is_cpu_place(x.place())) {
    return relu_cpu_backward(grad_out, out, x);
  } else if (paddle::platform::is_gpu_place(x.place())) {
    return relu_cuda_backward(grad_out, out, x);
  } else {
    throw std::runtime_error("Not implemented.");
  }
  return relu_cpu_backward(grad_out, out, x);
}

std::vector<std::vector<int64_t>> ReluInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

ADD_OPERATOR(relu2, OP_INFO(ReluForward), 
  PD_KERNEL(ReluForward), PD_KERNEL(ReluBackward), 
  PD_INFER_SHAPE(ReluInferShape));

