
#include <iostream>
#include "func.h"

template <typename T>
void func(const CUDAContext& ctx, int a, int b) {
  func_impl<T, CUDAContext>(ctx, a, b);
}

template void func<float>(const CUDAContext& ctx, int a, int b);
