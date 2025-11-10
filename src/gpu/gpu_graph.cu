#include "gpu_graph.h"
#include <fstream>
#include <iostream>
#include <cstring>

GPUGraphLoader::GPUGraphLoader() : is_loaded(false) {
    // Initialize GPU graph structure
    gpu_graph.d_row_pointers = nullptr;
    gpu_graph.d_column_indices = nullptr;
    gpu_graph.d_values = nullptr;
    gpu_graph.num_nodes = 0;
    gpu_graph.num_edges = 0;
}

GPUGraphLoader::~GPUGraphLoader() {
    freeGPUMemory();
}

bool GPUGraphLoader::loadFromFile(const std::string& filename) {
    // Use common graph parser
    GraphParser parser;
    if (!parser.loadFromFile(filename)) {
        return false;
    }
    
    return loadFromGraph(parser.getGraph());
}

bool GPUGraphLoader::loadFromGraph(const Graph& host_graph) {
    gpu_graph.num_nodes = host_graph.num_nodes;
    gpu_graph.num_edges = host_graph.num_edges;
    
    std::cout << "Loading graph to GPU: " << gpu_graph.num_nodes << " nodes, " 
              << gpu_graph.num_edges << " edges" << std::endl;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&gpu_graph.d_row_pointers, 
                         (gpu_graph.num_nodes + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu_graph.d_column_indices, 
                         gpu_graph.num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu_graph.d_values, 
                         gpu_graph.num_edges * sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(gpu_graph.d_row_pointers, host_graph.row_pointers.data(),
                         (gpu_graph.num_nodes + 1) * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_graph.d_column_indices, host_graph.column_indices.data(),
                         gpu_graph.num_edges * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_graph.d_values, host_graph.values.data(),
                         gpu_graph.num_edges * sizeof(float),
                         cudaMemcpyHostToDevice));

    is_loaded = true;
    std::cout << "Graph successfully loaded to GPU memory" << std::endl;
    return true;
}

void GPUGraphLoader::freeGPUMemory() {
    if (gpu_graph.d_row_pointers) {
        cudaFree(gpu_graph.d_row_pointers);
        gpu_graph.d_row_pointers = nullptr;
    }
    if (gpu_graph.d_column_indices) {
        cudaFree(gpu_graph.d_column_indices);
        gpu_graph.d_column_indices = nullptr;
    }
    if (gpu_graph.d_values) {
        cudaFree(gpu_graph.d_values);
        gpu_graph.d_values = nullptr;
    }
    is_loaded = false;
}

void GPUGraphLoader::printGraphStats() const {
    if (!is_loaded) {
        std::cout << "Graph not loaded" << std::endl;
        return;
    }
    
    std::cout << "=== GPU Graph Statistics ===" << std::endl;
    std::cout << "Nodes: " << gpu_graph.num_nodes << std::endl;
    std::cout << "Edges: " << gpu_graph.num_edges << std::endl;
    std::cout << "Average degree: " << (double)gpu_graph.num_edges / gpu_graph.num_nodes << std::endl;
    
    // Calculate memory usage
    size_t memory_mb = ((gpu_graph.num_nodes + 1) * sizeof(uint32_t) +
                       gpu_graph.num_edges * sizeof(uint32_t) +
                       gpu_graph.num_edges * sizeof(float)) / (1024 * 1024);
    std::cout << "GPU memory usage: " << memory_mb << " MB" << std::endl;
}
