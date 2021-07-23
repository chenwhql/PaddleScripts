#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> ReluCPUForward(const paddle::Tensor& x) {
  CHECK_INPUT(x);

  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  auto x_numel = x.size();
  auto* x_data = x.data<float>();
  auto* out_data = out.mutable_data<float>(x.place());

  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<float>(0.), x_data[i]);
  }

  return {out};
}

std::vector<paddle::Tensor> ReluCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  auto out_numel = out.size();
  auto* out_data = out.data<float>();
  auto* grad_out_data = grad_out.data<float>();
  auto* grad_x_data = grad_x.mutable_data<float>(x.place());

  for (int i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<float>(0) ? 1. : 0.);
  }

  return {grad_x};
}

PD_BUILD_OP(custom_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ReluCPUForward));

PD_BUILD_GRAD_OP(custom_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluCPUBackward));
