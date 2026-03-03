// mps_benchmark.mm
// Objective-C++ benchmark: Custom Block-Tiled Metal Kernel vs Apple MPS
//
// This file uses Objective-C++ (.mm) because Metal Performance Shaders (MPS)
// is an Objective-C framework with no metal-cpp C++ wrappers.

#include "Tensor.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

using namespace yan;

// ─────────────────────────────────────────────────────────────
// Helper: fill an MTLBuffer with random floats
// ─────────────────────────────────────────────────────────────
static void fill_random(float* ptr, size_t count) {
    std::mt19937 gen(42); // fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = dist(gen);
    }
}

// ─────────────────────────────────────────────────────────────
// Benchmark using Apple's MPSMatrixMultiplication
// ─────────────────────────────────────────────────────────────
static double benchmark_mps(id<MTLDevice> device, id<MTLCommandQueue> queue,
                            float* dataA, float* dataB, float* dataC,
                            size_t N, int num_runs) {

    // MPS requires row bytes aligned to 16-byte boundaries.
    // For float columns, rowBytes = N * sizeof(float). N=1024 → 4096, already aligned.
    size_t rowBytes = N * sizeof(float);
    size_t totalBytes = N * rowBytes;

    // Create MTLBuffers with shared storage (unified memory on Apple Silicon)
    id<MTLBuffer> bufA = [device newBufferWithBytes:dataA length:totalBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithBytes:dataB length:totalBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:totalBytes options:MTLResourceStorageModeShared];

    // Create MPS matrix descriptors
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                      columns:N
                                                                     rowBytes:rowBytes
                                                                     dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                      columns:N
                                                                     rowBytes:rowBytes
                                                                     dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                      columns:N
                                                                     rowBytes:rowBytes
                                                                     dataType:MPSDataTypeFloat32];

    // Wrap buffers in MPSMatrix objects
    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    // Create the MPSMatrixMultiplication kernel
    // C = alpha * A * B + beta * C   (alpha=1, beta=0 → standard matmul)
    MPSMatrixMultiplication* mpsMatMul = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
         transposeLeft:NO
        transposeRight:NO
            resultRows:N
         resultColumns:N
       interiorColumns:N
                 alpha:1.0
                  beta:0.0];

    // Warm-up run
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        [mpsMatMul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            [mpsMatMul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = (end - start) / (double)num_runs;

    // Copy result back for verification
    memcpy(dataC, bufC.contents, totalBytes);

    return duration.count();
}

// ─────────────────────────────────────────────────────────────
// Benchmark using our custom Block-Tiled Metal kernel
// ─────────────────────────────────────────────────────────────
static double benchmark_custom(float* dataA, float* dataB, float* dataC,
                               size_t N, int num_runs) {

    std::vector<float> vecA(dataA, dataA + N * N);
    std::vector<float> vecB(dataB, dataB + N * N);

    Tensor A({N, N}, vecA);
    Tensor B({N, N}, vecB);

    A.to(Device::Metal);
    B.to(Device::Metal);

    // Warm-up
    Tensor C = A.matmul(B);

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        C = A.matmul(B);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = (end - start) / (double)num_runs;

    // Copy result back for verification
    C.to(Device::CPU);
    memcpy(dataC, C.data.data(), N * N * sizeof(float));

    return duration.count();
}

// ─────────────────────────────────────────────────────────────
// CPU baseline
// ─────────────────────────────────────────────────────────────
static double benchmark_cpu(float* dataA, float* dataB, float* dataC, size_t N) {
    std::vector<float> vecA(dataA, dataA + N * N);
    std::vector<float> vecB(dataB, dataB + N * N);
    Tensor A({N, N}, vecA);
    Tensor B({N, N}, vecB);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor C = A.matmul(B);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    memcpy(dataC, C.data.data(), N * N * sizeof(float));
    return duration.count();
}

// ─────────────────────────────────────────────────────────────
// Verify two result arrays match within epsilon
// ─────────────────────────────────────────────────────────────
static bool verify(float* a, float* b, size_t count, float epsilon = 1e-2f) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "  Max element-wise difference: " << max_diff << "\n";
    return max_diff < epsilon;
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main() {
    @autoreleasepool {
        const size_t N = 1024;
        const int NUM_RUNS = 10;
        const size_t total = N * N;

        std::cout << "╔══════════════════════════════════════════════════╗\n";
        std::cout << "║  YanFramework — MatMul Benchmark (" << N << "x" << N << ")  ║\n";
        std::cout << "╚══════════════════════════════════════════════════╝\n\n";

        // Generate shared test data (same inputs for all backends)
        std::vector<float> dataA(total), dataB(total);
        std::vector<float> resultCPU(total), resultCustom(total), resultMPS(total);
        fill_random(dataA.data(), total);
        fill_random(dataB.data(), total);

        // Get Metal device info
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        std::cout << "Metal Device: " << [[device name] UTF8String] << "\n\n";

        // ── CPU ──
        std::cout << "▸ Running CPU baseline...\n";
        double cpu_ms = benchmark_cpu(dataA.data(), dataB.data(), resultCPU.data(), N);
        std::cout << "  CPU Time: " << cpu_ms << " ms\n\n";

        // ── Custom Block-Tiled Metal ──
        std::cout << "▸ Running Custom Block-Tiled Metal kernel...\n";
        double custom_ms = benchmark_custom(dataA.data(), dataB.data(), resultCustom.data(), N, NUM_RUNS);
        std::cout << "  Custom Metal Time: " << custom_ms << " ms (avg of " << NUM_RUNS << " runs)\n";
        std::cout << "  vs CPU Speedup: " << cpu_ms / custom_ms << "x\n";
        bool custom_ok = verify(resultCPU.data(), resultCustom.data(), total);
        std::cout << "  Correctness: " << (custom_ok ? "✅ PASS" : "❌ FAIL") << "\n\n";

        // ── Apple MPS ──
        std::cout << "▸ Running Apple MPS (MPSMatrixMultiplication)...\n";
        double mps_ms = benchmark_mps(device, queue, dataA.data(), dataB.data(), resultMPS.data(), N, NUM_RUNS);
        std::cout << "  MPS Time: " << mps_ms << " ms (avg of " << NUM_RUNS << " runs)\n";
        std::cout << "  vs CPU Speedup: " << cpu_ms / mps_ms << "x\n";
        bool mps_ok = verify(resultCPU.data(), resultMPS.data(), total);
        std::cout << "  Correctness: " << (mps_ok ? "✅ PASS" : "❌ FAIL") << "\n\n";

        // ── Summary Table ──
        std::cout << "┌─────────────────────────┬────────────┬──────────┐\n";
        std::cout << "│ Backend                 │ Time (ms)  │ Speedup  │\n";
        std::cout << "├─────────────────────────┼────────────┼──────────┤\n";
        printf("│ CPU (C++ loops)         │ %10.2f │ %7.1fx │\n", cpu_ms, 1.0);
        printf("│ Custom Metal (Tiled)    │ %10.2f │ %7.1fx │\n", custom_ms, cpu_ms / custom_ms);
        printf("│ Apple MPS               │ %10.2f │ %7.1fx │\n", mps_ms, cpu_ms / mps_ms);
        std::cout << "└─────────────────────────┴────────────┴──────────┘\n\n";

        double ratio = custom_ms / mps_ms;
        if (ratio < 1.5) {
            std::cout << "🏆 Our custom kernel is within " << (ratio * 100.0) << "% of Apple's MPS!\n";
        } else {
            std::cout << "📊 Apple MPS is " << ratio << "x faster. Room for optimization!\n";
        }
    }
    return 0;
}
