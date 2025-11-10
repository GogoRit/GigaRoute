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
    // For debugging, use a much smaller delta to see if that helps
    // The issue might be that 200m is too large for the algorithm to work correctly
    
    // Try a very small delta first
    if (graph.num_nodes > 10000000) {
        return 10.0f; // Much smaller for debugging - 10m buckets
    } else if (graph.num_nodes > 1000000) {
        return 5.0f; // Very small buckets
    } else {
        return 1.0f;  // Tiny buckets for small graphs
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
    
    // Debug: Check if source node has any neighbors
    const GPUGraph& debug_graph = graph_loader->getGPUGraph();
    std::vector<uint32_t> row_ptrs(3);
    CUDA_CHECK(cudaMemcpy(row_ptrs.data(), debug_graph.d_row_pointers + source, 
                         3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t num_edges = row_ptrs[1] - row_ptrs[0];
    std::cout << "Source node " << source << " has " << num_edges << " edges" << std::endl;
    
    if (num_edges > 0) {
        std::vector<uint32_t> neighbors(std::min(5u, num_edges));
        std::vector<float> weights(std::min(5u, num_edges));
        CUDA_CHECK(cudaMemcpy(neighbors.data(), debug_graph.d_column_indices + row_ptrs[0], 
                             neighbors.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(weights.data(), debug_graph.d_values + row_ptrs[0], 
                             weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::cout << "First few neighbors: ";
        for (size_t i = 0; i < neighbors.size(); i++) {
            std::cout << neighbors[i] << "(" << weights[i] << "m) ";
        }
        std::cout << std::endl;
    }
    
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
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
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
        
        // Clear and recount bucket sizes for accuracy
        CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
        
        launch_count_bucket_sizes(
            d_buckets,
            d_bucket_sizes,
            max_nodes,
            num_buckets,
            256
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // CRITICAL: Update buckets BEFORE finding next bucket
        // This ensures that relaxed neighbors get assigned to buckets
        
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
        
        // Clear and recount bucket sizes for accuracy
        CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
        
        launch_count_bucket_sizes(
            d_buckets,
            d_bucket_sizes,
            max_nodes,
            num_buckets,
            256
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // CRITICAL: Count bucket sizes BEFORE finding next bucket
        // Clear and recount bucket sizes for accuracy
        CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
        
        launch_count_bucket_sizes(
            d_buckets,
            d_bucket_sizes,
            max_nodes,
            num_buckets,
            256
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // NOW find next non-empty bucket (after counting)
        uint32_t old_bucket = current_bucket;
        current_bucket = findNextNonEmptyBucket();
        
        iteration++;
        
        // Debug: Show bucket transition and check neighbor distances
        if (iteration <= 10) {
            std::cout << "Iteration " << iteration << ": " << old_bucket << " -> " << current_bucket << std::endl;
            
            // Check what happened to the first few neighbors
            if (iteration == 1 && source == 0) {
                std::vector<float> neighbor_distances(3);
                uint32_t neighbors[3] = {1, 1110127, 4412508};
                for (int i = 0; i < 3; i++) {
                    CUDA_CHECK(cudaMemcpy(&neighbor_distances[i], &d_distances[neighbors[i]], 
                                         sizeof(float), cudaMemcpyDeviceToHost));
                    std::cout << "  Neighbor " << neighbors[i] << " distance: " << neighbor_distances[i] << std::endl;
                }
            }
        }
        
        // Progress reporting with debugging (show every iteration for debugging)
        if (iteration % 1 == 0 && iteration < 10) {
            // Get bucket sizes for debugging
            std::vector<uint32_t> debug_bucket_sizes(std::min(10u, num_buckets));
            CUDA_CHECK(cudaMemcpy(debug_bucket_sizes.data(), d_bucket_sizes, 
                                 debug_bucket_sizes.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            
            std::cout << "Iteration " << iteration << ", processing bucket " 
                     << current_bucket << ", bucket sizes [0-9]: ";
            for (size_t i = 0; i < debug_bucket_sizes.size(); i++) {
                std::cout << debug_bucket_sizes[i] << " ";
            }
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
