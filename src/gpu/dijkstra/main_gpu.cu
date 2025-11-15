#include "../common/gpu_graph.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <float.h>

// Forward declarations of kernel launch functions
extern "C" {
    void launch_sssp_kernel(
        const GPUGraph& gpu_graph,
        float* d_distances,
        const uint32_t* d_worklist,
        uint32_t* d_new_worklist,
        uint32_t num_current_nodes,
        uint32_t* d_new_worklist_size,
        uint32_t* d_worklist_flags,
        float delta,
        int block_size);
    
    void launch_init_distances(
        float* d_distances,
        uint32_t num_nodes,
        uint32_t source_node,
        int block_size);
}

class GPUDijkstra {
private:
    GPUGraphLoader* graph_loader;
    float* d_distances;
    uint32_t* d_worklist_1;
    uint32_t* d_worklist_2;
    uint32_t* d_worklist_size;
    uint32_t* d_worklist_flags;  // Flag array for deduplication
    uint32_t max_nodes;
    bool is_initialized;

public:
    GPUDijkstra(GPUGraphLoader* loader) : graph_loader(loader), is_initialized(false) {
        if (!loader->isLoaded()) {
            std::cerr << "Error: Graph not loaded in GPUGraphLoader" << std::endl;
            return;
        }
        
        const GPUGraph& gpu_graph = loader->getGPUGraph();
        max_nodes = gpu_graph.num_nodes;
        
        // Allocate GPU memory for SSSP computation
        CUDA_CHECK(cudaMalloc(&d_distances, max_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_worklist_1, max_nodes * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_worklist_2, max_nodes * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_worklist_size, sizeof(uint32_t)));
        // Allocate but don't use flags (deduplication overhead was worse than duplicates)
        CUDA_CHECK(cudaMalloc(&d_worklist_flags, max_nodes * sizeof(uint32_t)));
        
        is_initialized = true;
        std::cout << "GPU Dijkstra initialized for " << max_nodes << " nodes" << std::endl;
    }
    
    ~GPUDijkstra() {
        if (d_distances) cudaFree(d_distances);
        if (d_worklist_1) cudaFree(d_worklist_1);
        if (d_worklist_2) cudaFree(d_worklist_2);
        if (d_worklist_size) cudaFree(d_worklist_size);
        if (d_worklist_flags) cudaFree(d_worklist_flags);
    }
    
    float findShortestPath(uint32_t source, uint32_t target) {
        if (!is_initialized) {
            std::cerr << "Error: GPUDijkstra not properly initialized" << std::endl;
            return -1.0f;
        }
        
        const GPUGraph& gpu_graph = graph_loader->getGPUGraph();
        
        // Initialize distances array
        launch_init_distances(d_distances, max_nodes, source, 256);
        
        // Initialize first worklist with source node
        CUDA_CHECK(cudaMemcpy(d_worklist_1, &source, sizeof(uint32_t), cudaMemcpyHostToDevice));
        
        uint32_t current_worklist_size = 1;
        uint32_t zero = 0;
        float delta = 0.0f; // For basic Dijkstra, delta = 0
        
        uint32_t* current_worklist = d_worklist_1;
        uint32_t* next_worklist = d_worklist_2;
        
        int iteration = 0;
        const int max_iterations = 10000; // Increased safety limit
        
        // Check if target is reachable initially
        float target_distance;
        CUDA_CHECK(cudaMemcpy(&target_distance, &d_distances[target], 
                             sizeof(float), cudaMemcpyDeviceToHost));
        
        // Main SSSP loop
        while (current_worklist_size > 0 && iteration < max_iterations) {
            // OPTIMIZATION: Early termination check less frequently for better performance
            // Check more frequently in early iterations, less frequently later
            uint32_t check_interval = (iteration < 50) ? 10 : (iteration < 500) ? 50 : 100;
            if (iteration % check_interval == 0 && iteration > 0) {
                CUDA_CHECK(cudaMemcpy(&target_distance, &d_distances[target], 
                                     sizeof(float), cudaMemcpyDeviceToHost));
                if (target_distance < FLT_MAX) {
                    std::cout << "Target reached at iteration " << iteration 
                              << " with distance " << target_distance << std::endl;
                    break;
                }
            }
            // Reset next worklist size
            CUDA_CHECK(cudaMemcpy(d_worklist_size, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
            
            // OPTIMIZATION: Dynamic block size based on worklist size
            // Smaller worklists benefit from smaller blocks (better occupancy)
            // Larger worklists can use larger blocks (better throughput)
            int block_size = 256;  // Default
            if (current_worklist_size < 1000) {
                block_size = 128;  // Small worklist - use smaller blocks
            } else if (current_worklist_size > 100000) {
                block_size = 512;  // Large worklist - use larger blocks
            }
            
            // Launch SSSP kernel
            launch_sssp_kernel(
                gpu_graph,
                d_distances,
                current_worklist,
                next_worklist,
                current_worklist_size,
                d_worklist_size,
                d_worklist_flags,  // Keep parameter for compatibility, but not used
                delta,
                block_size
            );
            
            // OPTIMIZATION: Use single synchronization point
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Get new worklist size (single memcpy after sync)
            CUDA_CHECK(cudaMemcpy(&current_worklist_size, d_worklist_size, 
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost));
            
            // Swap worklists for next iteration
            std::swap(current_worklist, next_worklist);
            
            iteration++;
            
            if (iteration % 100 == 0) {
                std::cout << "Iteration " << iteration << ", worklist size: " 
                         << current_worklist_size;
                if (iteration > 0) {
                    std::cout << ", target distance: " << target_distance;
                }
                std::cout << std::endl;
            }
        }
        
        if (iteration >= max_iterations) {
            std::cout << "WARNING: Reached maximum iterations (" << max_iterations 
                      << ") without full convergence" << std::endl;
        } else {
            std::cout << "SSSP completed in " << iteration << " iterations" << std::endl;
        }
        
        // Get final result distance for target node
        float result_distance;
        CUDA_CHECK(cudaMemcpy(&result_distance, &d_distances[target], 
                             sizeof(float), cudaMemcpyDeviceToHost));
        
        if (result_distance >= FLT_MAX) {
            std::cout << "No path found to target node " << target << std::endl;
        }
        
        return result_distance;
    }
    
    void printStatistics(uint32_t source, uint32_t target, float distance, double time_ms) {
        std::cout << "\n=== GPU Dijkstra Results ===" << std::endl;
        std::cout << "Source node: " << source << std::endl;
        std::cout << "Target node: " << target << std::endl;
        std::cout << "Shortest distance: " << distance << " meters" << std::endl;
        std::cout << "Distance in kilometers: " << distance / 1000.0f << " km" << std::endl;
        std::cout << "Computation time: " << time_ms << " ms" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== GPU-Accelerated Dijkstra's Algorithm ===" << std::endl;
    
    // Default graph file path (try multiple possible locations)
    std::vector<std::string> possible_paths = {
        "data/processed/nyc_graph.bin",    // New structure
        "nyc_graph.bin",                   // Project root
        "../nyc_graph.bin",                // From build directory
        "../../nyc_graph.bin"              // From build/bin directory
    };
    
    std::string graph_file;
    if (argc > 1) {
        graph_file = argv[1];
    } else {
        // Try to find the file in possible locations
        for (const auto& path : possible_paths) {
            std::ifstream test_file(path);
            if (test_file.good()) {
                graph_file = path;
                std::cout << "Found graph file at: " << path << std::endl;
                break;
            }
        }
        if (graph_file.empty()) {
            graph_file = "nyc_graph.bin";  // Default fallback
        }
    }
    
    // Initialize CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Load graph to GPU
    GPUGraphLoader graph_loader;
    if (!graph_loader.loadFromFile(graph_file)) {
        std::cerr << "Error: Failed to load graph from " << graph_file << std::endl;
        return 1;
    }
    
    graph_loader.printGraphStats();
    
    // Initialize GPU Dijkstra
    GPUDijkstra gpu_dijkstra(&graph_loader);
    
    // Test cases
    std::vector<std::pair<uint32_t, uint32_t>> test_cases;
    
    if (argc >= 4) {
        // Custom source and target from command line
        uint32_t source = std::atoi(argv[2]);
        uint32_t target = std::atoi(argv[3]);
        test_cases.push_back({source, target});
    } else {
        // Default test cases (same as your CPU baseline)
        test_cases.push_back({0, 100});
        test_cases.push_back({2931668, 5863336});
        
        // Add a few random test cases
        srand(time(nullptr));
        const GPUGraph& gpu_graph = graph_loader.getGPUGraph();
        for (int i = 0; i < 3; i++) {
            uint32_t source = rand() % gpu_graph.num_nodes;
            uint32_t target = rand() % gpu_graph.num_nodes;
            test_cases.push_back({source, target});
        }
    }
    
    // Run test cases
    for (size_t i = 0; i < test_cases.size(); i++) {
        uint32_t source = test_cases[i].first;
        uint32_t target = test_cases[i].second;
        
        std::cout << "\n--- Test Case " << (i + 1) << " ---" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        float distance = gpu_dijkstra.findShortestPath(source, target);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (distance >= 0) {
            gpu_dijkstra.printStatistics(source, target, distance, time_ms);
        } else {
            std::cout << "Error: Could not find path from " << source << " to " << target << std::endl;
        }
    }
    
    std::cout << "\nGPU computation completed!" << std::endl;
    return 0;
}
