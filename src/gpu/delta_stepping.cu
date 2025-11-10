#include "delta_stepping.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <float.h>

GPUDeltaStepping::GPUDeltaStepping(GPUGraphLoader* loader) 
    : graph_loader(loader), d_distances(nullptr), d_buckets(nullptr),
      d_bucket_sizes(nullptr), d_bucket_offsets(nullptr), 
      d_current_bucket(nullptr), d_next_bucket(nullptr),
      is_initialized(false) {
    
    if (!loader->isLoaded()) {
        std::cerr << "Error: Graph not loaded in GPUGraphLoader" << std::endl;
        return;
    }
    
    const GPUGraph& gpu_graph = loader->getGPUGraph();
    max_nodes = gpu_graph.num_nodes;
    
    // Set default configuration
    config.delta = calculateOptimalDelta(gpu_graph);
    config.max_iterations = 10000;
    config.convergence_check_interval = 100;
    config.enable_early_termination = true;
    
    // Calculate number of buckets needed
    // Estimate maximum possible distance (rough upper bound)
    float max_edge_weight = 5000.0f; // Assume max 5km edges
    float estimated_max_distance = max_edge_weight * sqrt(max_nodes);
    num_buckets = (uint32_t)(estimated_max_distance / config.delta) + 1000; // Add buffer
    
    if (initializeGPUMemory()) {
        is_initialized = true;
        std::cout << "GPU Delta-stepping initialized:" << std::endl;
        std::cout << "  Nodes: " << max_nodes << std::endl;
        std::cout << "  Delta: " << config.delta << std::endl;
        std::cout << "  Buckets: " << num_buckets << std::endl;
    }
}

GPUDeltaStepping::~GPUDeltaStepping() {
    freeGPUMemory();
}

bool GPUDeltaStepping::initializeGPUMemory() {
    try {
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_distances, max_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_buckets, max_nodes * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_bucket_sizes, num_buckets * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_bucket_offsets, num_buckets * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_current_bucket, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_next_bucket, sizeof(uint32_t)));
        
        // Initialize bucket arrays to zero
        CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_bucket_offsets, 0, num_buckets * sizeof(uint32_t)));
        
        return true;
    } catch (...) {
        std::cerr << "Error: Failed to allocate GPU memory for delta-stepping" << std::endl;
        freeGPUMemory();
        return false;
    }
}

void GPUDeltaStepping::freeGPUMemory() {
    if (d_distances) { cudaFree(d_distances); d_distances = nullptr; }
    if (d_buckets) { cudaFree(d_buckets); d_buckets = nullptr; }
    if (d_bucket_sizes) { cudaFree(d_bucket_sizes); d_bucket_sizes = nullptr; }
    if (d_bucket_offsets) { cudaFree(d_bucket_offsets); d_bucket_offsets = nullptr; }
    if (d_current_bucket) { cudaFree(d_current_bucket); d_current_bucket = nullptr; }
    if (d_next_bucket) { cudaFree(d_next_bucket); d_next_bucket = nullptr; }
}

uint32_t GPUDeltaStepping::calculateOptimalDelta(const GPUGraph& graph) {
    // Heuristic: delta should be roughly the average edge weight
    // For road networks, this is typically 100-500 meters
    
    // Simple heuristic based on graph size
    if (graph.num_nodes > 10000000) {
        return 200.0f; // Large graphs (like NYC) - 200m buckets
    } else if (graph.num_nodes > 1000000) {
        return 100.0f; // Medium graphs - 100m buckets
    } else {
        return 50.0f;  // Small graphs - 50m buckets
    }
}

void GPUDeltaStepping::resetDistances(uint32_t source) {
    // Initialize distances and bucket assignments
    launch_init_delta_distances(d_distances, d_buckets, max_nodes, source, 256);
    
    // Reset bucket sizes
    CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
    
    // Set bucket 0 size to 1 (contains source node)
    uint32_t initial_bucket_size = 1;
    CUDA_CHECK(cudaMemcpy(&d_bucket_sizes[0], &initial_bucket_size, 
                         sizeof(uint32_t), cudaMemcpyHostToDevice));
}

uint32_t GPUDeltaStepping::findNextNonEmptyBucket() {
    // Simple linear search for next non-empty bucket
    // In a production system, this could be optimized with parallel reduction
    
    std::vector<uint32_t> bucket_sizes(num_buckets);
    CUDA_CHECK(cudaMemcpy(bucket_sizes.data(), d_bucket_sizes, 
                         num_buckets * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    for (uint32_t i = 0; i < num_buckets; i++) {
        if (bucket_sizes[i] > 0) {
            return i;
        }
    }
    
    return UINT32_MAX; // No non-empty buckets found
}

float GPUDeltaStepping::findShortestPath(uint32_t source, uint32_t target) {
    if (!is_initialized) {
        std::cerr << "Error: GPUDeltaStepping not properly initialized" << std::endl;
        return -1.0f;
    }
    
    const GPUGraph& gpu_graph = graph_loader->getGPUGraph();
    
    // Initialize distances and buckets
    resetDistances(source);
    
    uint32_t current_bucket = 0;
    uint32_t iteration = 0;
    float target_distance = FLT_MAX;
    
    std::cout << "Starting delta-stepping with delta = " << config.delta << std::endl;
    
    // Main delta-stepping loop
    while (current_bucket != UINT32_MAX && iteration < config.max_iterations) {
        
        // Check for early termination
        if (config.enable_early_termination && 
            iteration % config.convergence_check_interval == 0 && iteration > 0) {
            
            CUDA_CHECK(cudaMemcpy(&target_distance, &d_distances[target], 
                                 sizeof(float), cudaMemcpyDeviceToHost));
            
            if (target_distance < FLT_MAX) {
                std::cout << "Target reached at iteration " << iteration 
                         << " (bucket " << current_bucket << ") with distance " 
                         << target_distance << std::endl;
                break;
            }
        }
        
        // Process current bucket
        launch_delta_stepping_kernel(
            gpu_graph,
            d_distances,
            d_buckets,
            d_bucket_sizes,
            d_bucket_offsets,
            current_bucket,
            config.delta,
            max_nodes,
            256
        );
        
        // Update bucket assignments after relaxation
        launch_bucket_update_kernel(
            d_distances,
            d_buckets,
            d_bucket_sizes,
            d_bucket_offsets,
            config.delta,
            max_nodes,
            num_buckets,
            256
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Find next non-empty bucket
        current_bucket = findNextNonEmptyBucket();
        
        iteration++;
        
        // Progress reporting
        if (iteration % 100 == 0) {
            std::cout << "Iteration " << iteration << ", processing bucket " 
                     << current_bucket;
            if (iteration > 0 && target_distance < FLT_MAX) {
                std::cout << ", target distance: " << target_distance;
            }
            std::cout << std::endl;
        }
    }
    
    if (iteration >= config.max_iterations) {
        std::cout << "WARNING: Reached maximum iterations (" << config.max_iterations 
                  << ") without full convergence" << std::endl;
    } else {
        std::cout << "Delta-stepping completed in " << iteration << " iterations" << std::endl;
    }
    
    // Get final result
    float result_distance;
    CUDA_CHECK(cudaMemcpy(&result_distance, &d_distances[target], 
                         sizeof(float), cudaMemcpyDeviceToHost));
    
    if (result_distance >= FLT_MAX) {
        std::cout << "No path found to target node " << target << std::endl;
    }
    
    return result_distance;
}

std::vector<float> GPUDeltaStepping::findShortestPaths(
    const std::vector<uint32_t>& sources,
    const std::vector<uint32_t>& targets) {
    
    std::vector<float> results;
    results.reserve(sources.size());
    
    // For now, process queries sequentially
    // TODO: Implement true batch processing
    for (size_t i = 0; i < sources.size() && i < targets.size(); i++) {
        float distance = findShortestPath(sources[i], targets[i]);
        results.push_back(distance);
    }
    
    return results;
}

void GPUDeltaStepping::printStatistics(uint32_t source, uint32_t target, 
                                      float distance, double time_ms, 
                                      uint32_t iterations) const {
    std::cout << "\n=== GPU Delta-stepping Results ===" << std::endl;
    std::cout << "Algorithm: Delta-stepping (delta = " << config.delta << ")" << std::endl;
    std::cout << "Source node: " << source << std::endl;
    std::cout << "Target node: " << target << std::endl;
    std::cout << "Shortest distance: " << distance << " meters" << std::endl;
    std::cout << "Distance in kilometers: " << distance / 1000.0f << " km" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Computation time: " << time_ms << " ms" << std::endl;
    std::cout << "Performance: " << (distance / 1000.0f) / (time_ms / 1000.0f) 
              << " km/s processing rate" << std::endl;
}
