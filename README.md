# YanFramework

A custom deep learning framework built from scratch in **C++ and Apple Metal**, designed to understand GPU-accelerated neural network training at the hardware level.

## What Is This?

This project implements a complete neural network training pipeline — from raw matrix operations to backpropagation — entirely from scratch, without PyTorch or TensorFlow. Every layer of abstraction is hand-written:

| Layer | Implementation |
|-------|---------------|
| **Tensor Math** | C++ matrix operations with GPU dispatch |
| **GPU Kernels** | Raw Apple Metal `.metal` compute shaders |
| **Memory Tiling** | Block-tiled matmul with threadgroup shared memory |
| **Neural Network** | Linear layers, ReLU, Sigmoid, MLP container |
| **Training Loop** | SGD optimizer with full backpropagation |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    YanFramework                          │
├──────────────┬───────────────────────────────────────────┤
│  C++ API     │  Tensor, Linear, ReLU, Sigmoid, MLP      │
├──────────────┼───────────────────────────────────────────┤
│  Device      │  CPU (loops)  ←→  Metal GPU (kernels)    │
│  Abstraction │  tensor.to(Device::Metal)                │
├──────────────┼───────────────────────────────────────────┤
│  GPU Engine  │  MetalContext singleton                   │
│              │  Pre-compiled .metallib pipeline states   │
├──────────────┼───────────────────────────────────────────┤
│  Shaders     │  kernels.metal                            │
│              │  Block-Tiled MatMul (16×16 threadgroups)  │
│              │  Add, Sub, Mul, Transpose, ReLU, Sigmoid  │
└──────────────┴───────────────────────────────────────────┘
```

## Benchmark Results

**Matrix Multiplication: 1024 × 1024** on Apple M2 Pro

| Backend | Time (ms) | vs CPU |
|---------|-----------|--------|
| CPU (C++ triple loop) | 3,868 | 1.0x |
| **Custom Metal (Block-Tiled)** | **12.76** | **303x** |
| Apple MPS (reference) | 1.10 | 3,502x |

Our hand-written block-tiled kernel achieves a **303× speedup** over CPU and produces bit-exact results compared to Apple's proprietary MPS implementation.

## Build & Run

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Xcode Command Line Tools
- CMake 3.15+

### Build
```bash
mkdir build && cd build
cmake ..
make
```

### Run Examples
```bash
# XOR neural network training on GPU
./xor_example

# Tensor operation tests (CPU vs GPU correctness)
./tensor_test

# Full benchmark: CPU vs Custom Metal vs Apple MPS
./mps_benchmark
```

## Project Structure

```
yanframework/
├── include/
│   ├── Tensor.h          # Tensor class with Device enum
│   └── NN.h              # Layer, Linear, ReLU, Sigmoid, MLP
├── src/
│   ├── Tensor.cpp         # Math ops with Metal dispatch
│   ├── NN.cpp             # Neural network forward/backward
│   └── metal/
│       ├── MetalContext.h  # GPU singleton (device, queue, PSOs)
│       ├── MetalContext.cpp
│       └── kernels.metal   # Block-tiled GPU compute shaders
├── examples/
│   ├── xor_example.cpp    # End-to-end GPU training demo
│   ├── tensor_test.cpp    # Correctness + performance tests
│   └── mps_benchmark.mm   # MPS comparison (Obj-C++)
├── ext/metal-cpp/         # Apple's metal-cpp headers
└── CMakeLists.txt
```

## Key Learnings

- **Block Tiling** (loading data into threadgroup shared memory) is the single most impactful GPU optimization — it reduced memory bandwidth pressure by orders of magnitude
- **Rule of Five** in C++ is critical when wrapping GPU resources (`MTL::Buffer::retain`/`release`) to prevent memory leaks during training
- Apple MPS achieves ~11× more throughput than our kernel, likely through hardware-specific optimizations (SIMD shuffles, memory coalescing, AMX co-processor usage)
- The gap between 303× and 3,502× represents the frontier of GPU kernel engineering

## License

MIT
