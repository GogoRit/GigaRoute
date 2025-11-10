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
    // Use a smaller delta for debugging - larger deltas might skip nodes
    // For road networks, most edges are 10-500 meters
    return 50.0f; // 50m buckets - smaller for better granularity
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
    
    // Allocate flag for detecting updates
    uint32_t* d_updated_flag;
    CUDA_CHECK(cudaMalloc(&d_updated_flag, sizeof(uint32_t)));
    
    uint32_t current_bucket = 0;
    uint32_t iteration = 0;
    float target_distance = FLT_MAX;
    
    std::cout << "Starting correct delta-stepping with delta = " << config.delta << std::endl;
    
    // Main delta-stepping loop - process buckets sequentially
    while (current_bucket != UINT32_MAX && iteration < config.max_iterations) {
        
        // Check for early termination every iteration for debugging
        CUDA_CHECK(cudaMemcpy(&target_distance, &d_distances[target], 
                             sizeof(float), cudaMemcpyDeviceToHost));
        if (target_distance < FLT_MAX) {
            std::cout << "Target reached at iteration " << iteration 
                     << ", bucket " << current_bucket 
                     << " with distance " << target_distance << std::endl;
            break;
        }
        
        // Simplified: Process ALL edges for nodes in current bucket (no light/heavy split)
        uint32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(d_updated_flag, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        
        // Process all edges for current bucket
        launch_delta_stepping_light_kernel(
            gpu_graph,
            d_distances,
            d_buckets,
            current_bucket,
            config.delta,
            max_nodes,
            d_updated_flag,
            256
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Mark nodes in current bucket as processed
        launch_settle_bucket_kernel(d_buckets, current_bucket, max_nodes, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Update ALL bucket assignments based on new distances
        launch_bucket_update_kernel(d_distances, d_buckets, config.delta, max_nodes, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Recount bucket sizes
        CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
        launch_count_bucket_sizes(d_buckets, d_bucket_sizes, max_nodes, num_buckets, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Find next non-empty bucket
        current_bucket = findNextNonEmptyBucket();
        iteration++;
        
        // Debug first few iterations
        if (iteration <= 3) {
            std::cout << "After iteration " << iteration << ", next bucket: " << current_bucket << std::endl;
            
            // Check bucket sizes after this iteration
            std::vector<uint32_t> debug_bucket_sizes(std::min(10u, num_buckets));
            CUDA_CHECK(cudaMemcpy(debug_bucket_sizes.data(), d_bucket_sizes, 
                                 debug_bucket_sizes.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            
            std::cout << "  Bucket sizes [0-9]: ";
            for (size_t i = 0; i < debug_bucket_sizes.size(); i++) {
                std::cout << debug_bucket_sizes[i] << " ";
            }
            std::cout << std::endl;
            
            // Check first few neighbor distances
            if (iteration == 1 && source == 0) {
                std::vector<float> neighbor_distances(3);
                uint32_t neighbors[3] = {1, 1110127, 4412508};
                for (int i = 0; i < 3; i++) {
                    CUDA_CHECK(cudaMemcpy(&neighbor_distances[i], &d_distances[neighbors[i]], 
                                         sizeof(float), cudaMemcpyDeviceToHost));
                    std::cout << "  Neighbor " << neighbors[i] << " distance: " << neighbor_distances[i] << std::endl;
                }
                
                // Check their bucket assignments
                std::vector<uint32_t> neighbor_buckets(3);
                for (int i = 0; i < 3; i++) {
                    CUDA_CHECK(cudaMemcpy(&neighbor_buckets[i], &d_buckets[neighbors[i]], 
                                         sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    std::cout << "  Neighbor " << neighbors[i] << " bucket: " << neighbor_buckets[i] << std::endl;
                }
            }
        }
        
        // Progress reporting with debugging
        if (iteration % 1000 == 0) {
            std::cout << "Processed bucket " << current_bucket << " (iteration " << iteration << ")" << std::endl;
            
            // Debug: Check bucket sizes
            std::vector<uint32_t> debug_bucket_sizes(std::min(10u, num_buckets));
            CUDA_CHECK(cudaMemcpy(debug_bucket_sizes.data(), d_bucket_sizes, 
                                 debug_bucket_sizes.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            
            std::cout << "  Bucket sizes [0-9]: ";
            for (size_t i = 0; i < debug_bucket_sizes.size(); i++) {
                std::cout << debug_bucket_sizes[i] << " ";
            }
            std::cout << std::endl;
            
            // Check target distance and bucket
            float debug_target_distance;
            uint32_t debug_target_bucket;
            CUDA_CHECK(cudaMemcpy(&debug_target_distance, &d_distances[target], 
                                 sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&debug_target_bucket, &d_buckets[target], 
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost));
            std::cout << "  Target distance: " << debug_target_distance << ", bucket: " << debug_target_bucket << std::endl;
        }
    }
    
    cudaFree(d_updated_flag);
    
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
        return -1.0f;
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
