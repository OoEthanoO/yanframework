cmake_minimum_required(VERSION 3.15)
project(yanframework VERSION 1.0 DESCRIPTION "Custom Deep Learning Framework")

set(CMAKE_CXX_STANDARD 17)
option(USE_CUDA "Enable CUDA backend" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    add_library(yanframework_core
        src/Tensor.cpp
        src/NN.cpp
        src/cuda/kernels.cu
    )
    
    target_compile_definitions(yanframework_core PUBLIC YAN_USE_CUDA)
    target_include_directories(yanframework_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(yanframework_core CUDA::cudart)
    target_compile_options(yanframework_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>)
else()
    # Metal Shaders
    set(METAL_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/kernels.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER} -o ${METAL_AIR}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_SHADER}
        COMMENT "Compiling Metal shaders"
    )
    add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB})

    # Core library
    add_library(yanframework_core
        src/Tensor.cpp
        src/NN.cpp
        src/metal/MetalContext.cpp
    )
    add_dependencies(yanframework_core metal_shaders)

    target_include_directories(yanframework_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(yanframework_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ext/metal-cpp/metal-cpp)

    # Link Apple frameworks required by metal-cpp
    target_link_libraries(yanframework_core "-framework Metal" "-framework Foundation" "-framework QuartzCore")
endif()

# Executable for Tensor test
add_executable(tensor_test examples/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE yanframework_core)

# Executable for XOR example
add_executable(xor_example examples/xor_example.cpp)
target_link_libraries(xor_example PRIVATE yanframework_core)

# MPS Benchmark (Objective-C++ for MPS framework access)
add_executable(mps_benchmark examples/mps_benchmark.mm)
set_source_files_properties(examples/mps_benchmark.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
)
target_include_directories(mps_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mps_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ext/metal-cpp/metal-cpp)
target_link_libraries(mps_benchmark PRIVATE yanframework_core
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
)
