#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // A simple element-wise addition kernel
    result[index] = inA[index] + inB[index];
}

kernel void subtract_kernel(device const float* inA,
                            device const float* inB,
                            device float* result,
                            uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] - inB[index];
}

#define TILE_SIZE 16

kernel void matmul_kernel(device const float* inA,
                          device const float* inB,
                          device float* result,
                          constant uint& M,
                          constant uint& K,
                          constant uint& N,
                          uint2 thread_position_in_grid [[thread_position_in_grid]],
                          uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                          uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
    // Allocate threadgroup (shared) memory for tiles of A and B
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    // Local registers for accumulation
    float sum = 0.0;
    
    // Global row and col for this thread
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    
    // Local row and col within the tile
    uint local_row = thread_position_in_threadgroup.y;
    uint local_col = thread_position_in_threadgroup.x;

    // Iterate over tiles needed to cover the dimension K
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; ++t) {
        // Load tile A into threadgroup memory
        // Thread (local_row, local_col) loads one element into tileA
        uint tiled_k_A = t * TILE_SIZE + local_col;
        if (row < M && tiled_k_A < K) {
            tileA[local_row][local_col] = inA[row * K + tiled_k_A];
        } else {
            tileA[local_row][local_col] = 0.0;
        }

        // Load tile B into threadgroup memory
        uint tiled_k_B = t * TILE_SIZE + local_row;
        if (tiled_k_B < K && col < N) {
            tileB[local_row][local_col] = inB[tiled_k_B * N + col];
        } else {
            tileB[local_row][local_col] = 0.0;
        }

        // Synchronize all threads in the threadgroup to ensure tiles are fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_row][k] * tileB[k][local_col];
        }

        // Synchronize again before loading the next tile so we don't overwrite memory currently in use
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final result to global memory
    if (row < M && col < N) {
        result[row * N + col] = sum;
    }
}

kernel void relu_kernel(device const float* in,
                        device float* out,
                        uint index [[thread_position_in_grid]])
{
    out[index] = max(0.0f, in[index]);
}

kernel void relu_backward_kernel(device const float* grad_output,
                                 device const float* last_input,
                                 device float* grad_input,
                                 uint index [[thread_position_in_grid]])
{
    grad_input[index] = last_input[index] > 0 ? grad_output[index] : 0.0f;
}

kernel void sigmoid_kernel(device const float* in,
                           device float* out,
                           uint index [[thread_position_in_grid]])
{
    out[index] = 1.0f / (1.0f + exp(-in[index]));
}

kernel void sigmoid_backward_kernel(device const float* grad_output,
                                    device const float* last_output,
                                    device float* grad_input,
                                    uint index [[thread_position_in_grid]])
{
    float sig = last_output[index];
    grad_input[index] = grad_output[index] * sig * (1.0f - sig);
}

kernel void multiply_scalar_kernel(device const float* in,
                                   device float* out,
                                   constant float& scalar,
                                   uint index [[thread_position_in_grid]])
{
    out[index] = in[index] * scalar;
}

kernel void multiply_elementwise_kernel(device const float* inA,
                                        device const float* inB,
                                        device float* result,
                                        uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}

kernel void transpose_kernel(device const float* in,
                             device float* out,
                             constant uint& rows,
                             constant uint& cols,
                             uint2 index [[thread_position_in_grid]])
{
    // index.x is col (0 to cols-1)
    // index.y is row (0 to rows-1)
    if (index.y >= rows || index.x >= cols) return;
    
    // out is shaped [cols, rows]
    out[index.x * rows + index.y] = in[index.y * cols + index.x];
}
