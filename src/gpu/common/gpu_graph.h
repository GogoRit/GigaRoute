#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

#include <cuda_runtime.h>
#include "../../common/graph_parser.h"
#include <vector>
#include <cstdint>

// GPU-friendly graph structure using CSR format
struct GPUGraph {
    uint32_t* d_row_pointers;    // Start indices for each node's edges
    uint32_t* d_column_indices;  // Destination nodes for each edge
    float* d_values;             // Edge weights
    uint32_t num_nodes;          // Total number of nodes
    uint32_t num_edges;          // Total number of edges
};

// Host-side graph loader and manager
class GPUGraphLoader {
private:
    GPUGraph gpu_graph;
    bool is_loaded;

public:
    GPUGraphLoader();
    ~GPUGraphLoader();

    // Load graph from binary file to GPU memory
    bool loadFromFile(const std::string& filename);
    
    // Load graph from already parsed Graph structure
    bool loadFromGraph(const Graph& host_graph);
    
    // Get the GPU graph structure
    const GPUGraph& getGPUGraph() const { return gpu_graph; }
    
    // Memory management
    void freeGPUMemory();
    
    // Utility functions
    void printGraphStats() const;
    bool isLoaded() const { return is_loaded; }
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#endif // GPU_GRAPH_H
