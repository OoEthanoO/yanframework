#include "CudaContext.h"
#include <cmath>

#ifndef __CUDACC__
// Dummy definitions for IDE support on non-CUDA machines
#define __global__
#define __shared__
struct dim3 {
  size_t x, y, z;
  dim3(size_t _x = 1, size_t _y = 1, size_t _z = 1) : x(_x), y(_y), z(_z) {}
};
extern dim3 blockIdx;
extern dim3 blockDim;
extern dim3 threadIdx;
void __syncthreads();
#else
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif
#include <cuda_fp16.h>
#include <mma.h>
#endif

namespace yan {
namespace cuda {

#define TILE_SIZE 16

__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              size_t M, size_t K, size_t N) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t local_row = threadIdx.y;
  size_t local_col = threadIdx.x;

  size_t num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (size_t t = 0; t < num_tiles; ++t) {
    size_t tiled_k_A = t * TILE_SIZE + local_col;
    if (row < M && tiled_k_A < K) {
      tileA[local_row][local_col] = A[row * K + tiled_k_A];
    } else {
      tileA[local_row][local_col] = 0.0f;
    }

    size_t tiled_k_B = t * TILE_SIZE + local_row;
    if (tiled_k_B < K && col < N) {
      tileB[local_row][local_col] = B[tiled_k_B * N + col];
    } else {
      tileB[local_row][local_col] = 0.0f;
    }

    __syncthreads();

    for (size_t k = 0; k < TILE_SIZE; ++k) {
      sum += tileA[local_row][k] * tileB[k][local_col];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

__global__ void add_kernel(const float *A, const float *B, float *C,
                           size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = A[index] + B[index];
  }
}

__global__ void subtract_kernel(const float *A, const float *B, float *C,
                                size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = A[index] - B[index];
  }
}

__global__ void multiply_scalar_kernel(const float *A, float scalar, float *C,
                                       size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = A[index] * scalar;
  }
}

__global__ void multiply_elementwise_kernel(const float *A, const float *B,
                                            float *C, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = A[index] * B[index];
  }
}

__global__ void relu_kernel(const float *in, float *out, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    out[index] = fmaxf(0.0f, in[index]);
  }
}

__global__ void relu_backward_kernel(const float *grad_output,
                                     const float *last_input, float *grad_input,
                                     size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    grad_input[index] = last_input[index] > 0.0f ? grad_output[index] : 0.0f;
  }
}

__global__ void sigmoid_kernel(const float *in, float *out, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    out[index] = 1.0f / (1.0f + expf(-in[index]));
  }
}

__global__ void sigmoid_backward_kernel(const float *grad_output,
                                        const float *last_output,
                                        float *grad_input, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    float sig = last_output[index];
    grad_input[index] = grad_output[index] * sig * (1.0f - sig);
  }
}

__global__ void transpose_kernel(const float *in, float *out, size_t rows,
                                 size_t cols) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x; // col
  size_t y = blockIdx.y * blockDim.y + threadIdx.y; // row

  if (x < cols && y < rows) {
    out[x * rows + y] = in[y * cols + x];
  }
}

#ifdef __CUDACC__
using namespace nvcuda;

__global__ void matmul_tensor_core_kernel(const float *A, const float *B,
                                          float *C, size_t M, size_t K,
                                          size_t N) {
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  __shared__ half A_shmem[WMMA_M][WMMA_K];
  __shared__ half B_shmem[WMMA_K][WMMA_N];

  size_t row_tile = blockIdx.y;
  size_t col_tile = blockIdx.x;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (size_t t = 0; t < (K + WMMA_K - 1) / WMMA_K; ++t) {
    for (int i = 0; i < 8; ++i) {
      int idx = threadIdx.x * 8 + i;
      int r = idx / 16;
      int c = idx % 16;

      size_t a_row = row_tile * WMMA_M + r;
      size_t a_col = t * WMMA_K + c;
      if (a_row < M && a_col < K) {
        A_shmem[r][c] = __float2half(A[a_row * K + a_col]);
      } else {
        A_shmem[r][c] = __float2half(0.0f);
      }

      size_t b_row = t * WMMA_K + r;
      size_t b_col = col_tile * WMMA_N + c;
      if (b_row < K && b_col < N) {
        B_shmem[r][c] = __float2half(B[b_row * N + b_col]);
      } else {
        B_shmem[r][c] = __float2half(0.0f);
      }
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        b_frag;

    wmma::load_matrix_sync(a_frag, &A_shmem[0][0], WMMA_K);
    wmma::load_matrix_sync(b_frag, &B_shmem[0][0], WMMA_N);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
  }

  __shared__ float C_shmem[WMMA_M][WMMA_N];
  wmma::store_matrix_sync(&C_shmem[0][0], c_frag, WMMA_N, wmma::mem_row_major);

  __syncthreads();

  for (int i = 0; i < 8; ++i) {
    int idx = threadIdx.x * 8 + i;
    int r = idx / 16;
    int c = idx % 16;
    size_t c_row = row_tile * WMMA_M + r;
    size_t c_col = col_tile * WMMA_N + c;
    if (c_row < M && c_col < N) {
      C[c_row * N + c_col] = C_shmem[r][c];
    }
  }
}
#endif

// Launcher functions
#ifdef __CUDACC__
void launch_matmul_tensor_core(const float *A, const float *B, float *C,
                               size_t M, size_t K, size_t N) {
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
  dim3 blockDim(32);
  matmul_tensor_core_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
}

void launch_matmul(const float *A, const float *B, float *C, size_t M, size_t K,
                   size_t N) {
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, M, K, N);
}

void launch_add(const float *A, const float *B, float *C, size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  add_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, size);
}

void launch_subtract(const float *A, const float *B, float *C, size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  subtract_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, size);
}

void launch_multiply_scalar(const float *A, float scalar, float *C,
                            size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  multiply_scalar_kernel<<<numBlocks, threadsPerBlock>>>(A, scalar, C, size);
}

void launch_multiply_elementwise(const float *A, const float *B, float *C,
                                 size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  multiply_elementwise_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, size);
}

void launch_relu(const float *in, float *out, size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  relu_kernel<<<numBlocks, threadsPerBlock>>>(in, out, size);
}

void launch_relu_backward(const float *grad_output, const float *last_input,
                          float *grad_input, size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  relu_backward_kernel<<<numBlocks, threadsPerBlock>>>(grad_output, last_input,
                                                       grad_input, size);
}

void launch_sigmoid(const float *in, float *out, size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  sigmoid_kernel<<<numBlocks, threadsPerBlock>>>(in, out, size);
}

void launch_sigmoid_backward(const float *grad_output, const float *last_output,
                             float *grad_input, size_t size) {
  size_t threadsPerBlock = 256;
  size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  sigmoid_backward_kernel<<<numBlocks, threadsPerBlock>>>(
      grad_output, last_output, grad_input, size);
}

void launch_transpose(const float *in, float *out, size_t rows, size_t cols) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  transpose_kernel<<<numBlocks, threadsPerBlock>>>(in, out, rows, cols);
}
#else
// Dummy launcher implementations for IDE to parse headers cleanly
void launch_matmul_tensor_core(const float *A, const float *B, float *C,
                               size_t M, size_t K, size_t N) {}
void launch_matmul(const float *A, const float *B, float *C, size_t M, size_t K,
                   size_t N) {}
void launch_add(const float *A, const float *B, float *C, size_t size) {}
void launch_subtract(const float *A, const float *B, float *C, size_t size) {}
void launch_multiply_scalar(const float *A, float scalar, float *C,
                            size_t size) {}
void launch_multiply_elementwise(const float *A, const float *B, float *C,
                                 size_t size) {}
void launch_relu(const float *in, float *out, size_t size) {}
void launch_relu_backward(const float *grad_output, const float *last_input,
                          float *grad_input, size_t size) {}
void launch_sigmoid(const float *in, float *out, size_t size) {}
void launch_sigmoid_backward(const float *grad_output, const float *last_output,
                             float *grad_input, size_t size) {}
void launch_transpose(const float *in, float *out, size_t rows, size_t cols) {}
#endif

} // namespace cuda
} // namespace yan
