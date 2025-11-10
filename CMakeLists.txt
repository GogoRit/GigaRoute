cmake_minimum_required(VERSION 3.18)
project(CUDAGraphRouting LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)

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
else()
    message(WARNING "Osmium library not found. Skipping graph_converter build.")
endif()

# CPU implementation
add_executable(cpu_dijkstra 
    src/cpu/main_cpu.cpp
    src/cpu/dijkstra_cpu.cpp
)

target_link_libraries(cpu_dijkstra PRIVATE 
    common_graph
    Threads::Threads
)

# GPU implementation
add_executable(gpu_dijkstra 
    src/gpu/main_gpu.cu
    src/gpu/gpu_graph.cu
    src/gpu/sssp_kernel.cu
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

# Utilities
add_executable(verify_graph src/utils/verify_graph.cpp)

target_link_libraries(verify_graph PRIVATE 
    common_graph
    Threads::Threads
)

# Set output directories
set_target_properties(cpu_dijkstra gpu_dijkstra verify_graph
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

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