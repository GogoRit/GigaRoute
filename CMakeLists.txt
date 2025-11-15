cmake_minimum_required(VERSION 3.18)

# Option to build GPU targets (disabled by default on macOS)
option(BUILD_GPU_TARGETS "Build GPU/CUDA targets" ON)

# Detect macOS and disable GPU targets by default
if(APPLE)
    set(BUILD_GPU_TARGETS OFF CACHE BOOL "GPU targets disabled on macOS" FORCE)
    message(STATUS "macOS detected: GPU targets disabled by default")
    project(CUDAGraphRouting LANGUAGES CXX)
else()
    project(CUDAGraphRouting LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA only if building GPU targets
if(BUILD_GPU_TARGETS)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA enabled for GPU targets")
else()
    message(STATUS "CUDA disabled - building CPU-only targets")
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)

# Include directories
include_directories(src)
include_directories(include)

# Common library for shared graph utilities
add_library(common_graph STATIC
    src/common/graph_parser.cpp
)

target_link_libraries(common_graph PRIVATE
    Threads::Threads
)

# Preprocessing: Graph converter (OSM -> CSR binary)
# Only build if osmium is available
find_path(OSMIUM_INCLUDE_DIR osmium/handler.hpp)
if(OSMIUM_INCLUDE_DIR)
    add_executable(graph_converter src/preprocessing/graph_converter.cpp)
    
    target_include_directories(graph_converter PRIVATE 
        ${OSMIUM_INCLUDE_DIR}
        /opt/homebrew/include
        /opt/homebrew/Cellar/libosmium/2.22.0/include
        /opt/homebrew/opt/expat/include
    )
    
    target_link_libraries(graph_converter PRIVATE 
        Threads::Threads
        ZLIB::ZLIB
        BZip2::BZip2
        /opt/homebrew/lib/liblz4.dylib
        /opt/homebrew/opt/expat/lib/libexpat.dylib
    )
    
    # Variable-weight graph converter for delta-stepping testing
    add_executable(graph_converter_variable_weights src/preprocessing/graph_converter_variable_weights.cpp)
    
    target_include_directories(graph_converter_variable_weights PRIVATE 
        ${OSMIUM_INCLUDE_DIR}
        /opt/homebrew/include
        /opt/homebrew/Cellar/libosmium/2.22.0/include
        /opt/homebrew/opt/expat/include
    )
    
    target_link_libraries(graph_converter_variable_weights PRIVATE 
        Threads::Threads
        ZLIB::ZLIB
        BZip2::BZip2
        /opt/homebrew/lib/liblz4.dylib
        /opt/homebrew/opt/expat/lib/libexpat.dylib
    )
else()
    message(WARNING "Osmium library not found. Skipping graph_converter build.")
endif()

# Graph weight modifier (post-processing tool, no osmium required)
add_executable(graph_weight_modifier 
    src/preprocessing/graph_weight_modifier.cpp
    src/common/graph_parser.cpp
)

target_link_libraries(graph_weight_modifier PRIVATE
    Threads::Threads
)

# CPU implementation
add_executable(cpu_dijkstra 
    src/cpu/main_cpu.cpp
    src/cpu/dijkstra_cpu.cpp
)

target_link_libraries(cpu_dijkstra PRIVATE 
    common_graph
    Threads::Threads
)

# GPU implementation (only if CUDA is enabled)
if(BUILD_GPU_TARGETS)
    # GPU Dijkstra (Work-list SSSP) implementation
    add_executable(gpu_dijkstra 
        src/gpu/dijkstra/main_gpu.cu
        src/gpu/dijkstra/sssp_kernel.cu
        src/gpu/common/gpu_graph.cu
    )

    # Set CUDA properties for GPU target
    set_target_properties(gpu_dijkstra PROPERTIES
        CUDA_ARCHITECTURES "60;70;75;80"
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_link_libraries(gpu_dijkstra PRIVATE
        common_graph
        ${CUDA_LIBRARIES}
        cudart
    )
    
    # GPU Delta-stepping implementation
    add_executable(delta_stepping
        src/gpu/delta_stepping/main_delta_stepping.cu
        src/gpu/delta_stepping/delta_stepping.cu
        src/gpu/delta_stepping/delta_stepping_kernel.cu
        src/gpu/common/gpu_graph.cu
    )
    
    set_target_properties(delta_stepping PROPERTIES
        CUDA_ARCHITECTURES "60;70;75;80"
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    target_link_libraries(delta_stepping PRIVATE
        common_graph
        ${CUDA_LIBRARIES}
        cudart
    )
    
    message(STATUS "GPU targets configured: gpu_dijkstra, delta_stepping")
else()
    message(STATUS "Skipping GPU targets (CUDA disabled)")
endif()

# Utilities
add_executable(verify_graph src/utils/verify_graph.cpp)

target_link_libraries(verify_graph PRIVATE 
    common_graph
    Threads::Threads
)

# Set output directories
if(BUILD_GPU_TARGETS)
    set_target_properties(cpu_dijkstra gpu_dijkstra delta_stepping verify_graph
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
else()
    set_target_properties(cpu_dijkstra verify_graph
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

if(TARGET graph_converter)
    set_target_properties(graph_converter
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Print configuration summary
message(STATUS "=== CUDA Graph Routing Project Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")
if(TARGET graph_converter)
    message(STATUS "Graph converter: Enabled")
else()
    message(STATUS "Graph converter: Disabled (osmium not found)")
endif()