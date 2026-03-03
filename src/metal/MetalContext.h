#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace yan {

class MetalContext {
private:
    MTL::Device* _device;
    MTL::CommandQueue* _commandQueue;
    MTL::ComputePipelineState* _matmulPSO;
    MTL::ComputePipelineState* _addPSO;
    MTL::ComputePipelineState* _subtractPSO;
    MTL::ComputePipelineState* _reluPSO;
    MTL::ComputePipelineState* _reluBackwardPSO;
    MTL::ComputePipelineState* _sigmoidPSO;
    MTL::ComputePipelineState* _sigmoidBackwardPSO;
    MTL::ComputePipelineState* _multiplyScalarPSO;
    MTL::ComputePipelineState* _multiplyElementwisePSO;
    MTL::ComputePipelineState* _transposePSO;

    MetalContext(); // Singleton
    ~MetalContext();

    void buildShaders();

public:
    static MetalContext& instance();

    MTL::Device* device() const { return _device; }
    MTL::CommandQueue* commandQueue() const { return _commandQueue; }
    
    // Shader Pipelines
    MTL::ComputePipelineState* matmulPSO() const { return _matmulPSO; }
    MTL::ComputePipelineState* addPSO() const { return _addPSO; }
    MTL::ComputePipelineState* subtractPSO() const { return _subtractPSO; }
    MTL::ComputePipelineState* reluPSO() const { return _reluPSO; }
    MTL::ComputePipelineState* reluBackwardPSO() const { return _reluBackwardPSO; }
    MTL::ComputePipelineState* sigmoidPSO() const { return _sigmoidPSO; }
    MTL::ComputePipelineState* sigmoidBackwardPSO() const { return _sigmoidBackwardPSO; }
    MTL::ComputePipelineState* multiplyScalarPSO() const { return _multiplyScalarPSO; }
    MTL::ComputePipelineState* multiplyElementwisePSO() const { return _multiplyElementwisePSO; }
    MTL::ComputePipelineState* transposePSO() const { return _transposePSO; }
};

} // namespace yan
