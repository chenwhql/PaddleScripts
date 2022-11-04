#include "paddle/extension.h"

#include <vector>

#define CHECK_CPU_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  for (int i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cpu_forward(const paddle::Tensor& x) {
  CHECK_CPU_INPUT(x);

  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cpu_forward_kernel", ([&] {
        relu_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), x.size());
      }));

  return {out};
}

std::vector<paddle::Tensor> relu_cpu_backward(const paddle::Tensor& x,
                                              const paddle::Tensor& out,
                                              const paddle::Tensor& grad_out) {
  CHECK_CPU_INPUT(x);
  CHECK_CPU_INPUT(out);
  CHECK_CPU_INPUT(grad_out);

  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward_kernel", ([&] {
                               relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(x.place()),
                                   out.size());
                             }));

  return {grad_x};
}

// NOTE: If your custom operator may be compiled in an environment with CUDA,
// or it may be compiled in an environment without CUDA, in order to adapt the
// compilation environment, you can use the PADDLE_WITH_CUDA macro control
// the CUDA related code.
#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out);
#endif

std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x) {
  if (x.place() == paddle::PlaceType::kCPU) {
    return relu_cpu_forward(x);
#ifdef PADDLE_WITH_CUDA
  } else if (x.place() == paddle::PlaceType::kGPU) {
    return relu_cuda_forward(x);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom relu operator.");
  }
}

std::vector<paddle::Tensor> ReluBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out) {
  if (x.place() == paddle::PlaceType::kCPU) {
    return relu_cpu_backward(x, out, grad_out);
#ifdef PADDLE_WITH_CUDA
  } else if (x.place() == paddle::PlaceType::kGPU) {
    return relu_cuda_backward(x, out, grad_out);
#endif
  } else {
    PD_THROW("Unsupported device type for backward function of custom relu operator.");
  }
}


std::vector<std::vector<int64_t>> ReluInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> ReluInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ReluForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ReluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ReluInferDtype));

PD_BUILD_GRAD_OP(custom_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluBackward));
