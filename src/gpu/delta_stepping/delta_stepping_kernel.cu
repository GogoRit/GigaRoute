#include "delta_stepping.h"
#include <device_launch_parameters.h>
#include <float.h>

// Atomic min function for floats (reuse from sssp_kernel.cu)
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}

// Calculate bucket index for a given distance
__device__ uint32_t getBucketIndex(float distance, float delta) {
    if (distance >= FLT_MAX) return UINT32_MAX;
    return (uint32_t)(distance / delta);
}

// Delta-stepping kernel: process LIGHT edges only (weight <= delta)
__global__ void delta_stepping_light_kernel(
    const GPUGraph d_graph,
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t current_bucket,
    float delta,
    uint32_t max_nodes,
    uint32_t* d_updated_flag)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= max_nodes) return;
    
    // Only process nodes in the current bucket and not settled
    uint32_t node_bucket = d_buckets[tid];
    if (node_bucket != current_bucket || node_bucket >= 1000000) return;
    
    uint32_t current_node = tid;
    float current_distance = d_distances[current_node];
    
    // Skip if this node has infinite distance
    if (current_distance >= FLT_MAX) return;
    
    // Get the range of edges for this node
    uint32_t edge_start = d_graph.d_row_pointers[current_node];
    uint32_t edge_end = d_graph.d_row_pointers[current_node + 1];
    
    // Process only LIGHT edges (weight <= delta)
    for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
        uint32_t neighbor = d_graph.d_column_indices[edge_idx];
        float edge_weight = d_graph.d_values[edge_idx];
        
        // CRITICAL: Only process LIGHT edges (weight <= delta) in this kernel
        if (edge_weight <= delta) {
            float new_distance = current_distance + edge_weight;
            
            // OPTIMIZATION: Pre-check distance before atomic
            float current_neighbor_distance = d_distances[neighbor];
            if (new_distance < current_neighbor_distance) {
                // Try to update neighbor's distance atomically
                float old_distance = atomicMinFloat(&d_distances[neighbor], new_distance);
                
                // If we updated the distance, signal that we made changes
                if (new_distance < old_distance) {
                    atomicExch(d_updated_flag, 1);
                }
            }
        }
    }
}

// Kernel to process HEAVY edges (weight > delta) for settled nodes
__global__ void delta_stepping_heavy_kernel(
    const GPUGraph d_graph,
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t current_bucket,
    float delta,
    uint32_t max_nodes,
    uint32_t* d_updated_flag)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= max_nodes) return;
    
    // Only process nodes that were in the current bucket (now settled)
    uint32_t node_bucket = d_buckets[tid];
    if (node_bucket != current_bucket && node_bucket != (1000000 + current_bucket)) return;
    
    uint32_t current_node = tid;
    float current_distance = d_distances[current_node];
    
    if (current_distance >= FLT_MAX) return;
    
    uint32_t edge_start = d_graph.d_row_pointers[current_node];
    uint32_t edge_end = d_graph.d_row_pointers[current_node + 1];
    
    // Process only HEAVY edges (weight > delta)
    for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
        uint32_t neighbor = d_graph.d_column_indices[edge_idx];
        float edge_weight = d_graph.d_values[edge_idx];
        
        // CRITICAL: Only process heavy edges in this kernel
        if (edge_weight > delta) {
            float new_distance = current_distance + edge_weight;
            
            // OPTIMIZATION: Pre-check distance before atomic
            float current_neighbor_distance = d_distances[neighbor];
            if (new_distance < current_neighbor_distance) {
                float old_distance = atomicMinFloat(&d_distances[neighbor], new_distance);
                
                if (new_distance < old_distance) {
                    atomicExch(d_updated_flag, 1);
                }
            }
        }
    }
}

// Simple kernel to update all bucket assignments based on current distances
__global__ void bucket_update_kernel(
    float* d_distances,
    uint32_t* d_buckets,
    float delta,
    uint32_t max_nodes)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= max_nodes) return;
    
    // Don't update bucket assignments for settled nodes (but allow UINT32_MAX nodes to be updated)
    if (d_buckets[tid] >= 1000000 && d_buckets[tid] != UINT32_MAX) return;
    
    float distance = d_distances[tid];
    uint32_t new_bucket = getBucketIndex(distance, delta);
    d_buckets[tid] = new_bucket;
}

// Kernel to mark nodes in a bucket as settled
__global__ void settle_bucket_kernel(
    uint32_t* d_buckets,
    uint32_t bucket_to_settle,
    uint32_t max_nodes)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= max_nodes) return;
    
    if (d_buckets[tid] == bucket_to_settle) {
        d_buckets[tid] = 1000000 + bucket_to_settle; // Mark as settled
    }
}

// Kernel to count nodes in each bucket (separate pass for accuracy)
__global__ void count_bucket_sizes_kernel(
    uint32_t* d_buckets,
    uint32_t* d_bucket_sizes,
    uint32_t max_nodes,
    uint32_t num_buckets)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= max_nodes) return;
    
    uint32_t bucket = d_buckets[tid];
    // Count only unsettled nodes (bucket < 1000000) and valid buckets
    if (bucket != UINT32_MAX && bucket < num_buckets && bucket < 1000000) {
        atomicAdd(&d_bucket_sizes[bucket], 1);
    }
}

// Initialize distances and buckets for delta-stepping
__global__ void init_delta_distances(
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t num_nodes,
    uint32_t source_node)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < num_nodes) {
        if (tid == source_node) {
            d_distances[tid] = 0.0f;
            d_buckets[tid] = 0;  // Source goes in bucket 0
        } else {
            d_distances[tid] = FLT_MAX;
            d_buckets[tid] = UINT32_MAX;  // Infinite distance = invalid bucket
        }
    }
}

// Kernel to find next non-empty bucket
__global__ void find_next_bucket_kernel(
    uint32_t* d_bucket_sizes,
    uint32_t* d_next_bucket,
    uint32_t num_buckets,
    uint32_t start_bucket)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= num_buckets) return;
    
    uint32_t bucket_idx = start_bucket + tid;
    if (bucket_idx < num_buckets && d_bucket_sizes[bucket_idx] > 0) {
        atomicMin(d_next_bucket, bucket_idx);
    }
}

// Host function implementations
extern "C" {


void launch_delta_stepping_light_kernel(
    const GPUGraph& gpu_graph,
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t current_bucket,
    float delta,
    uint32_t max_nodes,
    uint32_t* d_updated_flag,
    int block_size)
{
    int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    delta_stepping_light_kernel<<<num_blocks, block_size>>>(
        gpu_graph,
        d_distances,
        d_buckets,
        current_bucket,
        delta,
        max_nodes,
        d_updated_flag
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_delta_stepping_heavy_kernel(
    const GPUGraph& gpu_graph,
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t current_bucket,
    float delta,
    uint32_t max_nodes,
    uint32_t* d_updated_flag,
    int block_size)
{
    int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    delta_stepping_heavy_kernel<<<num_blocks, block_size>>>(
        gpu_graph,
        d_distances,
        d_buckets,
        current_bucket,
        delta,
        max_nodes,
        d_updated_flag
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_bucket_update_kernel(
    float* d_distances,
    uint32_t* d_buckets,
    float delta,
    uint32_t max_nodes,
    int block_size)
{
    int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    bucket_update_kernel<<<num_blocks, block_size>>>(
        d_distances,
        d_buckets,
        delta,
        max_nodes
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_settle_bucket_kernel(
    uint32_t* d_buckets,
    uint32_t bucket_to_settle,
    uint32_t max_nodes,
    int block_size)
{
    int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    settle_bucket_kernel<<<num_blocks, block_size>>>(
        d_buckets,
        bucket_to_settle,
        max_nodes
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_init_delta_distances(
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t num_nodes,
    uint32_t source_node,
    int block_size)
{
    int num_blocks = (num_nodes + block_size - 1) / block_size;
    
    init_delta_distances<<<num_blocks, block_size>>>(
        d_distances,
        d_buckets,
        num_nodes,
        source_node
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_count_bucket_sizes(
    uint32_t* d_buckets,
    uint32_t* d_bucket_sizes,
    uint32_t max_nodes,
    uint32_t num_buckets,
    int block_size)
{
    int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    count_bucket_sizes_kernel<<<num_blocks, block_size>>>(
        d_buckets,
        d_bucket_sizes,
        max_nodes,
        num_buckets
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_find_next_bucket(
    uint32_t* d_bucket_sizes,
    uint32_t* d_next_bucket,
    uint32_t num_buckets,
    uint32_t start_bucket,
    int block_size)
{
    // Initialize result to UINT32_MAX
    uint32_t init_value = UINT32_MAX;
    CUDA_CHECK(cudaMemcpy(d_next_bucket, &init_value, sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    int num_blocks = (num_buckets + block_size - 1) / block_size;
    
    find_next_bucket_kernel<<<num_blocks, block_size>>>(
        d_bucket_sizes,
        d_next_bucket,
        num_buckets,
        start_bucket
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
