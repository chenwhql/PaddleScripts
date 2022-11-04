#include <iostream>

#include "context.h"
#include "func.h"

int main() {
  DeviceContext base;
  CPUContext cpu;
  CUDAContext cuda;

  func<float>(cpu, 1, 2);
  func<float>(cuda, 3, 4);
  // func<float>(base, 5, 6);
}