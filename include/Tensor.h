#pragma once

#include <vector>

namespace yan {

enum class Device { CPU, Metal, CUDA };

class Tensor {
public:
  Device device = Device::CPU;

  // Support 2D shapes for now
  std::vector<size_t> shape;
  std::vector<float> data;

  // Metal specific pointer
  void *mtl_buffer = nullptr;

  // CUDA specific pointer
  void *cuda_buffer = nullptr;

  // Constructors
  Tensor(std::vector<size_t> shape);
  Tensor(std::vector<size_t> shape, float initial_value);
  Tensor(std::vector<size_t> shape, const std::vector<float> &init_data);

  // Provide some common initializers
  static Tensor random(std::vector<size_t> shape); // Standard normal dist
  static Tensor zeros(std::vector<size_t> shape);
  static Tensor ones(std::vector<size_t> shape);

  // Device Management
  void to(Device target_device);

  // Memory Management
  Tensor(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(const Tensor &other);
  Tensor &operator=(Tensor &&other) noexcept;
  ~Tensor();

  // Math OPs
  Tensor matmul(const Tensor &other) const;
  Tensor add(const Tensor &other) const;
  Tensor subtract(const Tensor &other) const;
  Tensor multiply(float scalar) const;
  Tensor multiply(const Tensor &elementwise_other) const; // Hadamard product
  Tensor transpose() const;

  // Utility
  void print() const;
  bool shape_equals(const Tensor &other) const;
};

} // namespace yan
