#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <chrono>

using namespace yan;

void test_matmul() {
    std::cout << "--- Testing MatMul ---\n";
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B({3, 2}, {7, 8, 9, 10, 11, 12});
    
    A.print();
    B.print();
    
    Tensor C = A.matmul(B);
    std::cout << "CPU MatMul:\n";
    C.print();
    
    A.to(Device::Metal);
    B.to(Device::Metal);
    Tensor C_Metal = A.matmul(B);
    
    // Bring back to CPU for printing
    C_Metal.to(Device::CPU);
    std::cout << "Metal GPU MatMul:\n";
    C_Metal.print();
}

void test_addition() {
    std::cout << "\n--- Testing Addition ---\n";
    Tensor A({2, 2}, {1, 2, 3, 4});
    Tensor B({2, 2}, {10, 20, 30, 40});
    
    Tensor C = A.add(B);
    std::cout << "CPU Addition:\n";
    C.print();
    
    A.to(Device::Metal);
    B.to(Device::Metal);
    Tensor C_Metal = A.add(B);
    
    // Bring back to CPU for printing
    C_Metal.to(Device::CPU);
    std::cout << "Metal GPU Addition:\n";
    C_Metal.print();
}

void benchmark_matmul() {
    std::cout << "\n--- Benchmarking MatMul (1024 x 1024) ---\n";
    size_t N = 1024;
    Tensor A = Tensor::random({N, N});
    Tensor B = Tensor::random({N, N});

    // CPU Benchmark
    auto start_cpu = std::chrono::high_resolution_clock::now();
    Tensor C_CPU = A.matmul(B);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU MatMul Time: " << cpu_duration.count() << " ms\n";

    // GPU Benchmark
    A.to(Device::Metal);
    B.to(Device::Metal);
    
    // Warm-up to compile/cache pipeline states without timing overhead
    Tensor C_Metal = A.matmul(B);
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) { // Run a few times for stable average
        C_Metal = A.matmul(B);
    }
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = (end_gpu - start_gpu) / 5.0;
    
    std::cout << "Metal (Block-Tiled) MatMul Time: " << gpu_duration.count() << " ms\n";
    std::cout << "Speedup: " << cpu_duration.count() / gpu_duration.count() << "x\n";
}

int main() {
    test_matmul();
    test_addition();
    benchmark_matmul();
    return 0;
}
