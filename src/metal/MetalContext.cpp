#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTLEVENT_PRIVATE_IMPLEMENTATION

#include "MetalContext.h"
#include <iostream>

namespace yan {

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() {
    _device = MTL::CreateSystemDefaultDevice();
    if (!_device) {
        std::cerr << "Failed to find the default Metal device." << std::endl;
        exit(-1);
    }
    std::cout << "Using Metal Device: " << _device->name()->utf8String() << std::endl;

    _commandQueue = _device->newCommandQueue();
    buildShaders();
}

MetalContext::~MetalContext() {
    _matmulPSO->release();
    _addPSO->release();
    _subtractPSO->release();
    _reluPSO->release();
    _reluBackwardPSO->release();
    _sigmoidPSO->release();
    _sigmoidBackwardPSO->release();
    _multiplyScalarPSO->release();
    _multiplyElementwisePSO->release();
    _transposePSO->release();
    _commandQueue->release();
    _device->release();
}

void MetalContext::buildShaders() {
    NS::Error* error = nullptr;

    // Load the pre-compiled default.metallib created by CMake
    NS::String* libPath = NS::String::string("./default.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = _device->newLibrary(libPath, &error);

    if (!library) {
        std::cerr << "Failed to find the default library: " << error->localizedDescription()->utf8String() << std::endl;
        exit(-1);
    }

    // Load matmul function
    NS::String* matmulName = NS::String::string("matmul_kernel", NS::UTF8StringEncoding);
    MTL::Function* matmulFunction = library->newFunction(matmulName);
    _matmulPSO = _device->newComputePipelineState(matmulFunction, &error);
    matmulFunction->release();

    if (!_matmulPSO) {
        std::cerr << "Failed to create MatMul pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        exit(-1);
    }

    auto loadFunction = [&](const char* name) -> MTL::ComputePipelineState* {
        NS::String* nsName = NS::String::string(name, NS::UTF8StringEncoding);
        MTL::Function* function = library->newFunction(nsName);
        MTL::ComputePipelineState* pso = _device->newComputePipelineState(function, &error);
        function->release();
        if (!pso) {
            std::cerr << "Failed to load " << name << ": " << error->localizedDescription()->utf8String() << std::endl;
            exit(-1);
        }
        return pso;
    };

    _addPSO = loadFunction("add_kernel");
    _subtractPSO = loadFunction("subtract_kernel");
    _reluPSO = loadFunction("relu_kernel");
    _reluBackwardPSO = loadFunction("relu_backward_kernel");
    _sigmoidPSO = loadFunction("sigmoid_kernel");
    _sigmoidBackwardPSO = loadFunction("sigmoid_backward_kernel");
    _multiplyScalarPSO = loadFunction("multiply_scalar_kernel");
    _multiplyElementwisePSO = loadFunction("multiply_elementwise_kernel");
    _transposePSO = loadFunction("transpose_kernel");

    library->release();
}

} // namespace yan
