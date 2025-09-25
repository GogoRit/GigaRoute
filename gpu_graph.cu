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
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return false;
    }

    // Read header information
    file.read(reinterpret_cast<char*>(&gpu_graph.num_nodes), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&gpu_graph.num_edges), sizeof(uint32_t));
    
    std::cout << "Loading graph: " << gpu_graph.num_nodes << " nodes, " 
              << gpu_graph.num_edges << " edges" << std::endl;

    // Allocate host memory for temporary storage
    std::vector<uint32_t> h_row_pointers(gpu_graph.num_nodes + 1);
    std::vector<uint32_t> h_column_indices(gpu_graph.num_edges);
    std::vector<float> h_values(gpu_graph.num_edges);

    // Read graph data from file
    file.read(reinterpret_cast<char*>(h_row_pointers.data()), 
              (gpu_graph.num_nodes + 1) * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(h_column_indices.data()), 
              gpu_graph.num_edges * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(h_values.data()), 
              gpu_graph.num_edges * sizeof(float));

    file.close();

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&gpu_graph.d_row_pointers, 
                         (gpu_graph.num_nodes + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu_graph.d_column_indices, 
                         gpu_graph.num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu_graph.d_values, 
                         gpu_graph.num_edges * sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(gpu_graph.d_row_pointers, h_row_pointers.data(),
                         (gpu_graph.num_nodes + 1) * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_graph.d_column_indices, h_column_indices.data(),
                         gpu_graph.num_edges * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_graph.d_values, h_values.data(),
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
