#include "Tensor.h"
#include "metal/MetalContext.h"
#include <stdexcept>
#include <random>
#include <numeric>

namespace yan {

void Tensor::to(Device target_device) {
    if (device == target_device) return;
    
    if (target_device == Device::Metal) {
        // Move to Metal
        if (!mtl_buffer) {
            auto& ctx = MetalContext::instance();
            size_t bytes = data.size() * sizeof(float);
            MTL::Buffer* buffer = ctx.device()->newBuffer(bytes, MTL::ResourceStorageModeShared);
            memcpy(buffer->contents(), data.data(), bytes);
            mtl_buffer = buffer;
        } else {
            // Unlikely to hit this if device was strictly CPU, but just in case
            // we update the contents from our local `data` vector.
            MTL::Buffer* buffer = static_cast<MTL::Buffer*>(mtl_buffer);
            memcpy(buffer->contents(), data.data(), data.size() * sizeof(float));
        }
    } else {
        // Move to CPU
        if (mtl_buffer) {
            MTL::Buffer* buffer = static_cast<MTL::Buffer*>(mtl_buffer);
            memcpy(data.data(), buffer->contents(), data.size() * sizeof(float));
            // Instead of releasing the buffer every time we sync to CPU, let's keep it around
            // so we don't have to reallocate memory constantly during training loops.
        }
    }
    device = target_device;
}

Tensor::Tensor(std::vector<size_t> shape) : shape(shape) {
    size_t total_size = 1;
    for (size_t s : shape) total_size *= s;
    data.resize(total_size);
}

Tensor::Tensor(std::vector<size_t> shape, float initial_value) : shape(shape) {
    size_t total_size = 1;
    for (size_t s : shape) total_size *= s;
    data.assign(total_size, initial_value);
}

Tensor::Tensor(std::vector<size_t> shape, const std::vector<float>& init_data) : shape(shape), data(init_data) {
    size_t total_size = 1;
    for (size_t s : shape) total_size *= s;
    if (init_data.size() != total_size) {
        throw std::invalid_argument("Data size does not match shape.");
    }
}

// Memory Management (Rule of 5)
// Copy Constructor
Tensor::Tensor(const Tensor& other) : shape(other.shape), data(other.data), device(other.device) {
    if (other.mtl_buffer) {
        mtl_buffer = static_cast<MTL::Buffer*>(other.mtl_buffer);
        static_cast<MTL::Buffer*>(mtl_buffer)->retain();
    }
}

// Move Constructor
Tensor::Tensor(Tensor&& other) noexcept : shape(std::move(other.shape)), data(std::move(other.data)), device(other.device), mtl_buffer(other.mtl_buffer) {
    other.mtl_buffer = nullptr;
}

// Copy Assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (mtl_buffer) {
            static_cast<MTL::Buffer*>(mtl_buffer)->release();
        }
        shape = other.shape;
        data = other.data;
        device = other.device;
        mtl_buffer = other.mtl_buffer;
        if (mtl_buffer) {
            static_cast<MTL::Buffer*>(mtl_buffer)->retain();
        }
    }
    return *this;
}

// Move Assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (mtl_buffer) {
            static_cast<MTL::Buffer*>(mtl_buffer)->release();
        }
        shape = std::move(other.shape);
        data = std::move(other.data);
        device = other.device;
        mtl_buffer = other.mtl_buffer;
        other.mtl_buffer = nullptr;
    }
    return *this;
}

// Destructor
Tensor::~Tensor() {
    if (mtl_buffer) {
        static_cast<MTL::Buffer*>(mtl_buffer)->release();
    }
}

Tensor Tensor::random(std::vector<size_t> shape) {
    Tensor t(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (float& val : t.data) {
        val = d(gen);
    }
    return t;
}

Tensor Tensor::zeros(std::vector<size_t> shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(std::vector<size_t> shape) {
    return Tensor(shape, 1.0f);
}

bool Tensor::shape_equals(const Tensor& other) const {
    if (shape.size() != other.shape.size()) return false;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != other.shape[i]) return false;
    }
    return true;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matmul only supports 2D tensors for now.");
    }
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Inner dimensions do not match for Matmul.");
    }

    size_t M = shape[0];
    size_t K = shape[1];
    size_t N = other.shape[1];

    Tensor result({M, N}, 0.0f);
    
    if (device == Device::Metal && other.device == Device::Metal) {
        result.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.matmulPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(other.mtl_buffer), 0, 1);
        encoder->setBuffer(static_cast<MTL::Buffer*>(result.mtl_buffer), 0, 2);
        
        uint32_t M32 = M;
        uint32_t K32 = K;
        uint32_t N32 = N;
        encoder->setBytes(&M32, sizeof(uint32_t), 3);
        encoder->setBytes(&K32, sizeof(uint32_t), 4);
        encoder->setBytes(&N32, sizeof(uint32_t), 5);
        
        // Pad the grid to exact multiples of the tile size
        NS::UInteger gridX = ((N + 15) / 16) * 16;
        NS::UInteger gridY = ((M + 15) / 16) * 16;
        MTL::Size gridSize = MTL::Size::Make(gridX, gridY, 1);
        
        // Use a fixed 16x16 block size to match the TILE_SIZE in the shader
        NS::UInteger threadGroupSizeX = 16;
        NS::UInteger threadGroupSizeY = 16;
        
        // Ensure the device supports this block size
        if (ctx.matmulPSO()->maxTotalThreadsPerThreadgroup() < threadGroupSizeX * threadGroupSizeY) {
            std::cerr << "Warning: Device doesn't support 256 threads per threadgroup." << std::endl;
        }
        
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, threadGroupSizeY, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted(); // Synchronous for now for simplicity
        
        // Sync result back to CPU array
        result.to(Device::CPU);
        result.to(Device::Metal); // Need it back on GPU memory state
        
        return result;
    }

    // CPU Fallback
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += data[i * K + k] * other.data[k * N + j];
            }
            result.data[i * N + j] = sum;
        }
    }

    return result;
}

Tensor Tensor::add(const Tensor& other) const {
    if (!shape_equals(other)) {
        throw std::invalid_argument("Shapes must match for element-wise addition.");
    }
    Tensor result(shape);
    
    if (device == Device::Metal && other.device == Device::Metal) {
        result.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.addPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(other.mtl_buffer), 0, 1);
        encoder->setBuffer(static_cast<MTL::Buffer*>(result.mtl_buffer), 0, 2);
        
        MTL::Size gridSize = MTL::Size::Make(data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.addPSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        result.to(Device::CPU); // Sync
        result.to(Device::Metal); // Maintain state
        
        return result;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (!shape_equals(other)) {
        throw std::invalid_argument("Shapes must match for element-wise subtraction.");
    }
    Tensor result(shape);
    
    if (device == Device::Metal && other.device == Device::Metal) {
        result.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.subtractPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(other.mtl_buffer), 0, 1);
        encoder->setBuffer(static_cast<MTL::Buffer*>(result.mtl_buffer), 0, 2);
        
        MTL::Size gridSize = MTL::Size::Make(data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.subtractPSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        result.to(Device::CPU); result.to(Device::Metal);
        
        return result;
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::multiply(float scalar) const {
    Tensor result(shape);
    
    if (device == Device::Metal) {
        result.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.multiplyScalarPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(result.mtl_buffer), 0, 1);
        encoder->setBytes(&scalar, sizeof(float), 2);
        
        MTL::Size gridSize = MTL::Size::Make(data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.multiplyScalarPSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        result.to(Device::CPU); result.to(Device::Metal);
        return result;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::multiply(const Tensor& elementwise_other) const {
    if (!shape_equals(elementwise_other)) {
        throw std::invalid_argument("Shapes must match for element-wise multiplication (Hadamard product).");
    }
    Tensor result(shape);
    
    if (device == Device::Metal && elementwise_other.device == Device::Metal) {
        result.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.multiplyElementwisePSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(elementwise_other.mtl_buffer), 0, 1);
        encoder->setBuffer(static_cast<MTL::Buffer*>(result.mtl_buffer), 0, 2);
        
        MTL::Size gridSize = MTL::Size::Make(data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.multiplyElementwisePSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        result.to(Device::CPU); result.to(Device::Metal);
        
        return result;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * elementwise_other.data[i];
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Transpose only supports 2D tensors.");
    }
    Tensor result({shape[1], shape[0]});
    
    if (device == Device::Metal) {
        result.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.transposePSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(result.mtl_buffer), 0, 1);
        
        uint32_t R = shape[0];
        uint32_t C = shape[1];
        encoder->setBytes(&R, sizeof(uint32_t), 2);
        encoder->setBytes(&C, sizeof(uint32_t), 3);
        
        MTL::Size gridSize = MTL::Size::Make(C, R, 1);
        NS::UInteger threadGroupSizeX = ctx.transposePSO()->maxTotalThreadsPerThreadgroup();
        if (threadGroupSizeX > 32) threadGroupSizeX = 32;
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, threadGroupSizeX, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        result.to(Device::CPU); result.to(Device::Metal);
        return result;
    }
    
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            result.data[j * shape[0] + i] = data[i * shape[1] + j];
        }
    }
    return result;
}

void Tensor::print() const {
    if (shape.size() == 2) {
        std::cout << "Tensor(" << shape[0] << "x" << shape[1] << "):\n[\n";
        for (size_t i = 0; i < shape[0]; ++i) {
            std::cout << "  [ ";
            for (size_t j = 0; j < shape[1]; ++j) {
                std::cout << data[i * shape[1] + j] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "]\n";
    } else {
        std::cout << "Tensor(";
        for (size_t s : shape) std::cout << s << ",";
        std::cout << "): [ ";
        for (float val : data) std::cout << val << " ";
        std::cout << "]\n";
    }
}

} // namespace yan
