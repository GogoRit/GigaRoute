#ifndef DELTA_STEPPING_H
#define DELTA_STEPPING_H

#include "../common/gpu_graph.h"
#include <vector>
#include <cstdint>

// Delta-stepping algorithm configuration
struct DeltaSteppingConfig {
    float delta;                    // Step size parameter
    uint32_t max_iterations;        // Maximum iterations before timeout
    uint32_t convergence_check_interval; // Check convergence every N iterations
    bool enable_early_termination;  // Stop when target is reached
    bool debug_mode;                // Enable verbose debug output
};

// GPU Delta-stepping implementation
class GPUDeltaStepping {
private:
    GPUGraphLoader* graph_loader;
    
    // GPU memory for delta-stepping
    float* d_distances;
    uint32_t* d_buckets;           // Bucket assignments for each node
    uint32_t* d_bucket_sizes;      // Size of each bucket
    uint32_t* d_bucket_offsets;    // Offset into bucket array for each bucket
    uint32_t* d_current_bucket;    // Current bucket being processed
    uint32_t* d_next_bucket;       // Next non-empty bucket
    
    // Configuration
    DeltaSteppingConfig config;
    uint32_t num_buckets;
    uint32_t max_nodes;
    bool is_initialized;

public:
    explicit GPUDeltaStepping(GPUGraphLoader* loader);
    ~GPUDeltaStepping();
    
    // Configuration methods
    void setDelta(float delta) { config.delta = delta; }
    void setMaxIterations(uint32_t max_iter) { config.max_iterations = max_iter; }
    void enableEarlyTermination(bool enable) { config.enable_early_termination = enable; }
    void setDebugMode(bool enable) { config.debug_mode = enable; }
    
    // Main algorithm
    float findShortestPath(uint32_t source, uint32_t target);
    
    // Batch processing
    std::vector<float> findShortestPaths(const std::vector<uint32_t>& sources,
                                        const std::vector<uint32_t>& targets);
    
    // Statistics and debugging
    void printStatistics(uint32_t source, uint32_t target, float distance, 
                        double time_ms, uint32_t iterations) const;
    bool isInitialized() const { return is_initialized; }
    
private:
    // Internal methods
    bool initializeGPUMemory();
    void freeGPUMemory();
    float calculateOptimalDelta(const GPUGraph& graph);
    void resetDistances(uint32_t source);
    uint32_t findNextNonEmptyBucket();
};

// CUDA kernel function declarations
extern "C" {
    void launch_delta_stepping_light_kernel(
        const GPUGraph& gpu_graph,
        float* d_distances,
        uint32_t* d_buckets,
        uint32_t current_bucket,
        float delta,
        uint32_t max_nodes,
        uint32_t* d_updated_flag,
        int block_size = 256);
        
    void launch_delta_stepping_heavy_kernel(
        const GPUGraph& gpu_graph,
        float* d_distances,
        uint32_t* d_buckets,
        uint32_t current_bucket,
        float delta,
        uint32_t max_nodes,
        uint32_t* d_updated_flag,
        int block_size = 256);
        
    void launch_bucket_update_kernel(
        float* d_distances,
        uint32_t* d_buckets,
        float delta,
        uint32_t max_nodes,
        int block_size = 256);
        
    void launch_settle_bucket_kernel(
        uint32_t* d_buckets,
        uint32_t bucket_to_settle,
        uint32_t max_nodes,
        int block_size = 256);
        
    void launch_init_delta_distances(
        float* d_distances,
        uint32_t* d_buckets,
        uint32_t num_nodes,
        uint32_t source_node,
        int block_size = 256);
        
    void launch_count_bucket_sizes(
        uint32_t* d_buckets,
        uint32_t* d_bucket_sizes,
        uint32_t max_nodes,
        uint32_t num_buckets,
        int block_size = 256);
        
    void launch_find_next_bucket(
        uint32_t* d_bucket_sizes,
        uint32_t* d_next_bucket,
        uint32_t num_buckets,
        uint32_t start_bucket,
        int block_size = 256);
}

#endif // DELTA_STEPPING_H
