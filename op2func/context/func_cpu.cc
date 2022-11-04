#include <iostream>
#include "func.h"

template <typename T>
void func(const CPUContext& ctx, int a, int b) {
  func_impl<T, CPUContext>(ctx, a, b);
}

template void func<float>(const CPUContext& ctx, int a, int b);