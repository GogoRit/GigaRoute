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
    config.debug_mode = false;  // Disable debug by default for performance
    
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

float GPUDeltaStepping::calculateOptimalDelta(const GPUGraph& graph) {
    // Calculate optimal delta based on edge weight distribution
    // For road networks, optimal delta is typically around the median edge weight
    // This balances between too many buckets (small delta) and too few updates per bucket (large delta)
    
    // Adaptive delta: Use 100m as base, but could be tuned based on graph
    // For NYC graph with 11.7M nodes, 50-100m works well
    float base_delta = 100.0f;  // 100 meters
    
    // Scale based on graph size (larger graphs may benefit from slightly larger delta)
    if (graph.num_nodes > 10000000) {
        base_delta = 100.0f;  // Keep at 100m for very large graphs
    } else if (graph.num_nodes > 1000000) {
        base_delta = 75.0f;   // 75m for large graphs
    } else {
        base_delta = 50.0f;   // 50m for smaller graphs
    }
    
    return base_delta;
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
    // GPU-based parallel search - much faster than host-side linear search
    // This eliminates the expensive host-device memory transfer (172K buckets * 4 bytes)
    
    launch_find_next_bucket(d_bucket_sizes, d_next_bucket, num_buckets, 0, 256);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint32_t next_bucket = UINT32_MAX;
    CUDA_CHECK(cudaMemcpy(&next_bucket, d_next_bucket, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    return next_bucket;
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
    
    if (config.debug_mode) {
        std::cout << "Starting delta-stepping with delta = " << config.delta << std::endl;
    }
    
    // Main delta-stepping loop - process buckets sequentially
    while (current_bucket != UINT32_MAX && iteration < config.max_iterations) {
        
        // OPTIMIZATION: Check for early termination less frequently
        if (config.enable_early_termination && 
            (iteration % config.convergence_check_interval == 0 || iteration < 10)) {
            CUDA_CHECK(cudaMemcpy(&target_distance, &d_distances[target], 
                                 sizeof(float), cudaMemcpyDeviceToHost));
            if (target_distance < FLT_MAX) {
                if (config.debug_mode) {
                    std::cout << "Target reached at iteration " << iteration 
                             << ", bucket " << current_bucket 
                             << " with distance " << target_distance << std::endl;
                }
                break;
            }
        }
        
        // Process current bucket repeatedly until no more updates
        // Don't update bucket assignments during processing - wait until the end
        bool bucket_updated = true;
        int bucket_iterations = 0;
        
        // Count nodes in current bucket (only if needed for debug)
        uint32_t nodes_in_bucket = 0;
        if (config.debug_mode || bucket_iterations == 0) {
            CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
            launch_count_bucket_sizes(d_buckets, d_bucket_sizes, max_nodes, num_buckets, 256);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            if (current_bucket < num_buckets) {
                CUDA_CHECK(cudaMemcpy(&nodes_in_bucket, &d_bucket_sizes[current_bucket], 
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost));
            }
        }
        
        while (bucket_updated && bucket_iterations < 50) {
            uint32_t zero = 0;
            CUDA_CHECK(cudaMemcpy(d_updated_flag, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
            
            // Process LIGHT edges for current bucket
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
            
            // CRITICAL FIX: Don't update bucket assignments during light edge processing
            // This prevents nodes from moving out of current bucket prematurely
            // We'll update buckets after all light edges are processed
            
            // Synchronize to ensure light edge processing completes
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Check if any updates were made
            uint32_t updated;
            CUDA_CHECK(cudaMemcpy(&updated, d_updated_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            
            if (updated) {
                // Update bucket assignments only after checking for updates
                // This ensures nodes stay in current bucket during processing
                launch_bucket_update_kernel(d_distances, d_buckets, config.delta, max_nodes, 256);
                CUDA_CHECK(cudaDeviceSynchronize());
                
                // Only recount if needed for debug
                if (config.debug_mode) {
                    CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
                    launch_count_bucket_sizes(d_buckets, d_bucket_sizes, max_nodes, num_buckets, 256);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    
                    uint32_t new_bucket_size = 0;
                    if (current_bucket < num_buckets) {
                        CUDA_CHECK(cudaMemcpy(&new_bucket_size, &d_bucket_sizes[current_bucket], 
                                             sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    }
                    nodes_in_bucket = new_bucket_size;
                }
                bucket_updated = true;  // Continue if updates were made
            } else {
                bucket_updated = false;
            }
            
            bucket_iterations++;
        }
        
        // Debug: Show how many nodes were processed
        if (config.debug_mode && iteration <= 5) {
            std::cout << "  Processed " << nodes_in_bucket << " nodes in bucket " << current_bucket 
                     << " in " << bucket_iterations << " iterations" << std::endl;
        }
        
        // CRITICAL: Final bucket assignment update BEFORE settling
        // This ensures all nodes relaxed by light edges are in correct buckets
        launch_bucket_update_kernel(d_distances, d_buckets, config.delta, max_nodes, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Now settle nodes that are still in the current bucket
        launch_settle_bucket_kernel(d_buckets, current_bucket, max_nodes, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Process HEAVY edges for settled nodes (true delta-stepping)
        // CRITICAL FIX: Process heavy edges BEFORE updating bucket assignments
        // This ensures all heavy edges from settled nodes are processed
        uint32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(d_updated_flag, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        launch_delta_stepping_heavy_kernel(
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
        
        // Update bucket assignments for nodes relaxed via heavy edges
        // This must happen AFTER heavy edge processing to ensure correctness
        launch_bucket_update_kernel(d_distances, d_buckets, config.delta, max_nodes, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Recount bucket sizes (only when needed for finding next bucket)
        CUDA_CHECK(cudaMemset(d_bucket_sizes, 0, num_buckets * sizeof(uint32_t)));
        launch_count_bucket_sizes(d_buckets, d_bucket_sizes, max_nodes, num_buckets, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Find next non-empty bucket
        current_bucket = findNextNonEmptyBucket();
        iteration++;
        
        // Debug first few iterations
        if (config.debug_mode && iteration <= 3) {
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
        if (config.debug_mode && iteration % 1000 == 0) {
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
