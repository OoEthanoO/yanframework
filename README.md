# YanFramework

A custom deep learning framework built from scratch in **C++**, with backend support for both **Apple Metal** (Apple Silicon GPUs) and **NVIDIA CUDA** (NVIDIA GPUs), designed to understand GPU-accelerated neural network training at the hardware level.

## What Is This?

This project implements a complete neural network training pipeline — from raw matrix operations to backpropagation — entirely from scratch, without PyTorch or TensorFlow. Every layer of abstraction is hand-written:

| Layer | Implementation |
|-------|---------------|
| **Tensor Math** | C++ matrix operations with GPU dispatch |
| **GPU Kernels** | Raw Apple Metal (`.metal`) and NVIDIA CUDA (`.cu`) compute kernels |
| **Memory Tiling** | Block-tiled matmul with threadgroup/shared memory |
| **Tensor Cores** | WMMA API implementation for hardware-level `sm_70` acceleration |
| **Neural Network** | Linear layers, ReLU, Sigmoid, MLP container |
| **Training Loop** | SGD optimizer with full backpropagation |

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                          YanFramework                           │
├──────────────┬──────────────────────────────────────────────────┤
│  C++ API     │  Tensor, Linear, ReLU, Sigmoid, MLP              │
├──────────────┼──────────────────────────────────────────────────┤
│  Device      │  CPU (loops) ←→ Metal GPU ←→ CUDA GPU            │
│  Abstraction │  tensor.to(Device::Metal) / tensor.to(Device::CUDA) │
├──────────────┼──────────────────────────────────────────────────┤
│  GPU Engine  │  MetalContext (macOS) / CudaContext (NVIDIA)     │
│              │  PCIe memory management (cudaMalloc/cudaMemcpy)  │
├──────────────┼──────────────────────────────────────────────────┤
│  Kernels     │  kernels.metal / kernels.cu                      │
│              │  Block-Tiled MatMul & WMMA Tensor Cores          │
│              │  Add, Sub, Mul, Transpose, ReLU, Sigmoid         │
└──────────────┴──────────────────────────────────────────────────┘
```

## Features

- **Cross-Platform GPU Support:** Compiles for both Apple Silicon (using Metal) and Windows/Linux NVIDIA machines (using CUDA).
- **PCIe Memory Management:** Efficient data transfers via `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`.
- **Shared Memory Tiling (L1 Cache):** Explicitly manages GPU shared memory (`__shared__`) and synchronization (`__syncthreads()`) to reduce global memory bandwidth bottlenecks.
- **Tensor Cores (WMMA):** Utilizes NVIDIA's Warp Matrix Multiply and Accumulate (WMMA) API (`<mma.h>`) to execute 16x16 matrix multiplications in a single clock cycle at the hardware level (requires Volta `sm_70` or newer).

## Build & Run

### Prerequisites
- CMake 3.15+
- **For Apple Silicon:** macOS, Xcode Command Line Tools
- **For NVIDIA GPUs:** NVIDIA GPU (Volta `sm_70` or newer for Tensor Cores), CUDA Toolkit

### Build

```bash
mkdir build && cd build
```

**For Apple Metal:**
```bash
cmake ..
make
```

**For NVIDIA CUDA:**
```bash
cmake -DUSE_CUDA=ON ..
make
```

### Run Examples

```bash
# XOR neural network training on GPU
./xor_example

# Tensor operation tests (CPU vs GPU correctness)
./tensor_test

# Full benchmark (Mac only): CPU vs Custom Metal vs Apple MPS
./mps_benchmark
```

## Project Structure

```text
yanframework/
├── include/
│   ├── Tensor.h          # Tensor class with Device enum
│   └── NN.h              # Layer, Linear, ReLU, Sigmoid, MLP
├── src/
│   ├── Tensor.cpp         # Math ops with Metal/CUDA dispatch
│   ├── NN.cpp             # Neural network forward/backward
│   ├── metal/             # Apple Metal backend
│   │   ├── MetalContext.h
│   │   ├── MetalContext.cpp
│   │   └── kernels.metal  # Block-tiled GPU compute shaders
│   └── cuda/              # NVIDIA CUDA backend
│       ├── CudaContext.h
│       └── kernels.cu     # CUDA kernels featuring WMMA Tensor Cores
├── examples/
│   ├── xor_example.cpp    # End-to-end GPU training demo
│   ├── tensor_test.cpp    # Correctness + performance tests
│   └── mps_benchmark.mm   # MPS comparison (Obj-C++)
├── ext/metal-cpp/         # Apple's metal-cpp headers
└── CMakeLists.txt         # Build system config with CUDA support
```

## Key Learnings

- **Block Tiling** (loading data into shared memory) is the single most impactful generic GPU optimization — it reduces memory bandwidth pressure by orders of magnitude.
- **Hardware Acceleration (Tensor Cores)** using WMMA vastly outperforms generic CUDA cores for specific mathematical operations by tapping into custom silicon.
- **PCIe Bottlenecks:** Understanding the costs of moving data from the CPU host over the PCIe bus to discrete GPU VRAM is essential for profiling AI frameworks.
- **Cross-Platform Extensibility:** Abstracting device memory states allows seamless switching between CPU, integrated GPUs (Metal), and discrete GPUs (CUDA).

## License

MIT
