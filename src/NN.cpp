#include "NN.h"
#include "metal/MetalContext.h"
#include <cmath>
#include <algorithm>

namespace yan {

// Linear Layer
Linear::Linear(size_t in_features, size_t out_features)
    : weights(Tensor::random({in_features, out_features})),
      biases(Tensor::zeros({1, out_features})),
      last_input({0}) {}

void Linear::to(Device device) {
    weights.to(device);
    biases.to(device);
    last_input.to(device);
}

Tensor Linear::forward(const Tensor& input) {
    last_input = input; // Input is expected to be [batch_size, in_features]
    Tensor out = input.matmul(weights); // [batch_size, out_features]
    
    // Add biases to each row
    for (size_t i = 0; i < out.shape[0]; ++i) {
        for (size_t j = 0; j < out.shape[1]; ++j) {
            out.data[i * out.shape[1] + j] += biases.data[j];
        }
    }
    return out;
}

Tensor Linear::backward(Tensor grad_output, float learning_rate) {
    // grad_output is [batch_size, out_features]
    
    // 1. Calculate grad_input = grad_output * weights.T
    Tensor weights_T = weights.transpose();
    Tensor grad_input = grad_output.matmul(weights_T); // [batch_size, in_features]
    
    // 2. Calculate grad_weights = last_input.T * grad_output
    Tensor input_T = last_input.transpose();
    Tensor grad_weights = input_T.matmul(grad_output); // [in_features, out_features]
    
    // 3. Calculate grad_biases = sum of grad_output over batch
    Tensor grad_biases({1, biases.shape[1]}, 0.0f);
    if (grad_output.device == Device::Metal) {
        grad_biases.to(Device::Metal);
        // We lack a dedicated reduction sum kernel, so we'll pull grad_output to CPU briefly 
        // to compute bias gradients. Bias gradients are small ([1, out_features]).
        grad_output.to(Device::CPU);
        for (size_t i = 0; i < grad_output.shape[0]; ++i) {
            for (size_t j = 0; j < grad_output.shape[1]; ++j) {
                grad_biases.data[j] += grad_output.data[i * grad_output.shape[1] + j];
            }
        }
        // Push bias grad back to GPU 
        grad_output.to(Device::Metal);
        // For grad_biases, since we modified `data`, we must recreate the buffer or `to(Metal)` won't copy if it thinks it's already there
        // Actually, `to(Metal)` only copies if `device != target_device`.
        grad_biases.device = Device::CPU; 
        grad_biases.to(Device::Metal);
    } else {
        for (size_t i = 0; i < grad_output.shape[0]; ++i) {
            for (size_t j = 0; j < grad_output.shape[1]; ++j) {
                grad_biases.data[j] += grad_output.data[i * grad_output.shape[1] + j];
            }
        }
    }
    
    // 4. Update weights and biases (SGD step)
    Tensor lr_grad_weights = grad_weights.multiply(learning_rate);
    Tensor lr_grad_biases = grad_biases.multiply(learning_rate);
    
    if (weights.device == Device::Metal) {
        lr_grad_weights.to(Device::Metal);
        lr_grad_biases.to(Device::Metal);
        weights = weights.subtract(lr_grad_weights);
        biases = biases.subtract(lr_grad_biases);
    } else {
        for (size_t i = 0; i < weights.data.size(); ++i) {
            weights.data[i] -= lr_grad_weights.data[i];
        }
        for (size_t i = 0; i < biases.data.size(); ++i) {
            biases.data[i] -= lr_grad_biases.data[i];
        }
    }
    
    return grad_input;
}

// ReLU Activation
void ReLU::to(Device device) {
    last_input.to(device);
}

Tensor ReLU::forward(const Tensor& input) {
    last_input = input;
    Tensor out(input.shape);
    
    if (input.device == Device::Metal) {
        out.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.reluPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(input.mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(out.mtl_buffer), 0, 1);
        
        MTL::Size gridSize = MTL::Size::Make(input.data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.reluPSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        out.to(Device::CPU); out.to(Device::Metal); // Sync memory
        return out;
    }
    
    for (size_t i = 0; i < input.data.size(); ++i) {
        out.data[i] = std::max(0.0f, input.data[i]);
    }
    return out;
}

Tensor ReLU::backward(Tensor grad_output, float learning_rate) {
    Tensor grad_input(grad_output.shape);
    
    if (grad_output.device == Device::Metal) {
        grad_input.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.reluBackwardPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(grad_output.mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(last_input.mtl_buffer), 0, 1);
        encoder->setBuffer(static_cast<MTL::Buffer*>(grad_input.mtl_buffer), 0, 2);
        
        MTL::Size gridSize = MTL::Size::Make(grad_output.data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.reluBackwardPSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        grad_input.to(Device::CPU); grad_input.to(Device::Metal);
        return grad_input;
    }
    
    for (size_t i = 0; i < grad_output.data.size(); ++i) {
        grad_input.data[i] = last_input.data[i] > 0 ? grad_output.data[i] : 0.0f;
    }
    return grad_input;
}

// Sigmoid Activation
void Sigmoid::to(Device device) {
    last_output.to(device);
}

Tensor Sigmoid::forward(const Tensor& input) {
    Tensor out(input.shape);
    
    if (input.device == Device::Metal) {
        out.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.sigmoidPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(input.mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(out.mtl_buffer), 0, 1);
        
        MTL::Size gridSize = MTL::Size::Make(input.data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.sigmoidPSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        out.to(Device::CPU); out.to(Device::Metal);
        last_output = out;
        return out;
    }
    
    for (size_t i = 0; i < input.data.size(); ++i) {
        out.data[i] = 1.0f / (1.0f + std::exp(-input.data[i]));
    }
    last_output = out;
    return out;
}

Tensor Sigmoid::backward(Tensor grad_output, float learning_rate) {
    Tensor grad_input(grad_output.shape);
    
    if (grad_output.device == Device::Metal) {
        grad_input.to(Device::Metal);
        auto& ctx = MetalContext::instance();
        MTL::CommandBuffer* cmdBuf = ctx.commandQueue()->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = cmdBuf->computeCommandEncoder();
        
        encoder->setComputePipelineState(ctx.sigmoidBackwardPSO());
        encoder->setBuffer(static_cast<MTL::Buffer*>(grad_output.mtl_buffer), 0, 0);
        encoder->setBuffer(static_cast<MTL::Buffer*>(last_output.mtl_buffer), 0, 1);
        encoder->setBuffer(static_cast<MTL::Buffer*>(grad_input.mtl_buffer), 0, 2);
        
        MTL::Size gridSize = MTL::Size::Make(grad_output.data.size(), 1, 1);
        NS::UInteger threadGroupSizeX = ctx.sigmoidBackwardPSO()->maxTotalThreadsPerThreadgroup();
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);
        
        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        
        grad_input.to(Device::CPU); grad_input.to(Device::Metal);
        return grad_input;
    }
    
    for (size_t i = 0; i < grad_output.data.size(); ++i) {
        float sig = last_output.data[i];
        grad_input.data[i] = grad_output.data[i] * sig * (1.0f - sig);
    }
    return grad_input;
}

// MLP
MLP::MLP() {}

void MLP::to(Device device) {
    for (auto& layer : layers) {
        layer->to(device);
    }
}

void MLP::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Tensor MLP::forward(Tensor input) {
    for (auto& layer : layers) {
        input = layer->forward(input);
    }
    return input;
}

void MLP::backward(Tensor grad_output, float learning_rate) {
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad_output = (*it)->backward(grad_output, learning_rate);
    }
}

// Loss Function
float mse_loss(const Tensor& predictions, const Tensor& targets, Tensor& grad_output) {
    if (!predictions.shape_equals(targets)) {
        throw std::invalid_argument("Shapes must match for MSE loss.");
    }
    
    grad_output = Tensor(predictions.shape);
    float loss = 0.0f;
    size_t n = predictions.data.size();
    
    for (size_t i = 0; i < n; ++i) {
        float diff = predictions.data[i] - targets.data[i];
        loss += diff * diff;
        grad_output.data[i] = 2.0f * diff / n; // Derivative of MSE
    }
    
    return loss / n;
}

} // namespace yan
