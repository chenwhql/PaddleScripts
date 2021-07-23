#include "paddle/extension.h"

template <typename data_t>
__global__ void relu_cuda_forward_kernel(const data_t* x, data_t* y, const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<data_t>(0.));
  }
}

template <typename data_t>
__global__ void relu_cuda_backward_kernel(const data_t* dy, const data_t* y, data_t* dx, const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x) {
  auto out = paddle::Tensor();
  out.Resize(x.dims());
  
  int numel = x.numel();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(x.type(), "relu_cuda_forward_kernel", ([&]{
    relu_cuda_forward_kernel<data_t><<<grid, block>>>(
      x.data<data_t>(),
      out.mutable_data<data_t>(x.place()),
      numel);
  }));

  return {out};
}

std::vector<paddle::Tensor> relu_cuda_backward(
    const paddle::Tensor& grad_out, 
    const paddle::Tensor& out,
    const paddle::Tensor& x) {
  auto grad_x = paddle::Tensor();
  grad_x.Resize(x.dims());

  int numel = out.numel();
  int block = 512;
  int grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cuda_backward_kernel", ([&]{
    relu_cuda_backward_kernel<data_t><<<grid, block>>>(
      grad_out.data<data_t>(),
      out.data<data_t>(),
      grad_x.mutable_data<data_t>(x.place()),
      numel);
  }));

  return {grad_x};
}
