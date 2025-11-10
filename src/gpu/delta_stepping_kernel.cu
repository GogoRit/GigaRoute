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

// Delta-stepping kernel: process nodes in current bucket
__global__ void delta_stepping_kernel(
    const GPUGraph d_graph,
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t* d_bucket_sizes,
    uint32_t* d_bucket_offsets,
    uint32_t current_bucket,
    float delta,
    uint32_t max_nodes)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= max_nodes) return;
    
    uint32_t current_node = tid;
    float current_distance = d_distances[current_node];
    
    // Skip if this node has infinite distance
    if (current_distance >= FLT_MAX) return;
    
    // Only process nodes that belong to current bucket based on their stored bucket assignment
    if (d_buckets[current_node] != current_bucket) return;
    
    // Verify the node actually belongs to this bucket (double-check)
    uint32_t expected_bucket = getBucketIndex(current_distance, delta);
    if (expected_bucket != current_bucket) return;
    
    // After processing this node in this bucket, it should be "settled"
    // Mark it as processed by setting its bucket to a special "settled" value
    // Use current_bucket + 1000000 to indicate it was processed in this bucket
    d_buckets[current_node] = current_bucket + 1000000;
    
    // Get the range of edges for this node using CSR format
    uint32_t edge_start = d_graph.d_row_pointers[current_node];
    uint32_t edge_end = d_graph.d_row_pointers[current_node + 1];
    
    // Process all outgoing edges from current_node
    for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
        uint32_t neighbor = d_graph.d_column_indices[edge_idx];
        float edge_weight = d_graph.d_values[edge_idx];
        
        // Calculate new potential distance to neighbor
        float new_distance = current_distance + edge_weight;
        
        // Try to update neighbor's distance atomically
        float old_distance = atomicMinFloat(&d_distances[neighbor], new_distance);
        
        // Don't update buckets here - let the separate bucket update handle it
        // This avoids race conditions between distance updates and bucket assignments
    }
}

// Kernel to rebuild bucket assignments and sizes from scratch
__global__ void bucket_update_kernel(
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t* d_bucket_sizes,
    uint32_t* d_bucket_offsets,
    float delta,
    uint32_t max_nodes,
    uint32_t num_buckets)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= max_nodes) return;
    
    // Skip nodes that have been settled (bucket >= 1000000)
    if (d_buckets[tid] >= 1000000) return;
    
    float distance = d_distances[tid];
    uint32_t new_bucket = getBucketIndex(distance, delta);
    
    // Update bucket assignment for unsettled nodes only
    d_buckets[tid] = new_bucket;
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
    // Count only unsettled nodes (bucket < 1000000)
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

void launch_delta_stepping_kernel(
    const GPUGraph& gpu_graph,
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t* d_bucket_sizes,
    uint32_t* d_bucket_offsets,
    uint32_t current_bucket,
    float delta,
    uint32_t max_nodes,
    int block_size)
{
    int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    delta_stepping_kernel<<<num_blocks, block_size>>>(
        gpu_graph,
        d_distances,
        d_buckets,
        d_bucket_sizes,
        d_bucket_offsets,
        current_bucket,
        delta,
        max_nodes
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_bucket_update_kernel(
    float* d_distances,
    uint32_t* d_buckets,
    uint32_t* d_bucket_sizes,
    uint32_t* d_bucket_offsets,
    float delta,
    uint32_t max_nodes,
    uint32_t num_buckets,
    int block_size)
{
    int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    bucket_update_kernel<<<num_blocks, block_size>>>(
        d_distances,
        d_buckets,
        d_bucket_sizes,
        d_bucket_offsets,
        delta,
        max_nodes,
        num_buckets
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

} // extern "C"
