#pragma once

#include <cstddef>

namespace yan {
namespace cuda {

void launch_matmul_tensor_core(const float *A, const float *B, float *C,
                               size_t M, size_t K, size_t N);
void launch_matmul(const float *A, const float *B, float *C, size_t M, size_t K,
                   size_t N);
void launch_add(const float *A, const float *B, float *C, size_t size);
void launch_subtract(const float *A, const float *B, float *C, size_t size);
void launch_multiply_scalar(const float *A, float scalar, float *C,
                            size_t size);
void launch_multiply_elementwise(const float *A, const float *B, float *C,
                                 size_t size);
void launch_relu(const float *in, float *out, size_t size);
void launch_relu_backward(const float *grad_output, const float *last_input,
                          float *grad_input, size_t size);
void launch_sigmoid(const float *in, float *out, size_t size);
void launch_sigmoid_backward(const float *grad_output, const float *last_output,
                             float *grad_input, size_t size);
void launch_transpose(const float *in, float *out, size_t rows, size_t cols);

} // namespace cuda
} // namespace yan
