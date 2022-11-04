#pragma once

class DeviceContext {
 public:
  int get_id() const { return 0; }
  virtual ~DeviceContext() = default;
};

class CPUContext : public DeviceContext {
 public:
  int get_id() const { return 1; }
  virtual ~CPUContext() = default;
};

class CUDAContext : public DeviceContext {
 public:
  int get_id() const { return 2; }
  virtual ~CUDAContext() = default;
};
