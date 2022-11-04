#pragma once

#include <exception>

#include "context.h"

template <typename T, typename ContextT>
void func_impl(const ContextT& ctx, int a, int b) {
  std::cout << ctx.get_id();
  std::cout << a + b;
}

// auto declare func marco

#define DISPATCH_ALL_DEVICE(kernel_name, ...) \
  try { \
    kernel_name<T>(dynamic_cast<const CPUContext&>(ctx), __VA_ARGS__); \
  } catch (std::bad_cast exp) { \
    try { \
      kernel_name<T>(dynamic_cast<const CUDAContext&>(ctx), __VA_ARGS__); \
    } catch (std::bad_cast exp) { \
      throw std::runtime_error("no impl"); \
    } \
  } \
  return

#define PT_KERNEL_DECLARE(kernel_name, ...) \
  using kernel_name##_kernel = void(*)(const DeviceContext& ctx, __VA_ARGS__); \
  template <typename T> \
  void kernel_name(const CPUContext& ctx, __VA_ARGS__); \
  template <typename T> \
  void kernel_name(const CUDAContext& ctx, __VA_ARGS__); \
  template <typename T> \
  void kernel_name(const DeviceContext& ctx, __VA_ARGS__)

// auto declare all func declare and impl

PT_KERNEL_DECLARE(func, int a, int b) {
  DISPATCH_ALL_DEVICE(func, a, b);
}

// template <typename T, typename ContextT>
// fc(const ContextT& dev_ctx, ...) {
//   matmul<T, Context>();
//   add<T, Context>();
// }

// PT_REGISTER_CTX_KERNEL(fc, fc, ALL_BACKEND, )


template <typename T>
fc(const DeviceContext& dev_ctx, ...) {
  try {
    matmul<T>(dev_ctx, );
    add<T>(dev_ctx, );
  }
}