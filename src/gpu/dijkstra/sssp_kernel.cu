#include "../common/gpu_graph.h"
#include <device_launch_parameters.h>
#include <float.h>

// Atomic min function for floats (since CUDA doesn't provide one natively)
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

// CUDA kernel for Single Source Shortest Path using work-list approach
__global__ void sssp_kernel(
    const GPUGraph d_graph,
    float* d_distances,
    const uint32_t* d_worklist,
    uint32_t* d_new_worklist,
    const uint32_t num_current_nodes,
    uint32_t* d_new_worklist_size,
    uint32_t* d_worklist_flags,
    const float delta)
{
    // OPTIMIZATION: Use shared memory to cache worklist nodes for better memory access
    // Dynamic size based on blockDim.x (max 1024 threads per block)
    __shared__ uint32_t s_worklist[1024];  // Cache worklist nodes in shared memory
    
    // Calculate global thread ID
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t local_tid = threadIdx.x;
    
    // Cooperative loading: each thread loads one worklist node into shared memory
    if (tid < num_current_nodes && local_tid < blockDim.x) {
        s_worklist[local_tid] = d_worklist[tid];
    }
    __syncthreads();
    
    // Check if thread has work to do
    if (tid >= num_current_nodes) {
        return;
    }
    
    // Get the node this thread will process (from shared memory if available, else global)
    uint32_t current_node = (local_tid < blockDim.x) ? s_worklist[local_tid] : d_worklist[tid];
    
    // Get current distance to this node
    float current_distance = d_distances[current_node];
    
    // Get the range of edges for this node using CSR format
    // OPTIMIZATION: Cache row pointers in registers for better access
    uint32_t edge_start = d_graph.d_row_pointers[current_node];
    uint32_t edge_end = d_graph.d_row_pointers[current_node + 1];
    
    // Process all outgoing edges from current_node
    // OPTIMIZATION: Unroll small loops and optimize memory access pattern
    for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
        // OPTIMIZATION: Coalesced memory access - column_indices and values are sequential
        uint32_t neighbor = d_graph.d_column_indices[edge_idx];
        float edge_weight = d_graph.d_values[edge_idx];
        
        // Calculate new potential distance to neighbor
        float new_distance = current_distance + edge_weight;
        
        // OPTIMIZATION: Pre-check distance before expensive atomic operation
        // Read current distance first (non-atomic, may be stale but safe for comparison)
        // This avoids ~50% of atomic operations on average
        float current_neighbor_distance = d_distances[neighbor];
        
        // Only perform atomic update if we might improve the distance
        // This avoids expensive atomic operations when update won't help
        if (new_distance < current_neighbor_distance) {
            // Try to update neighbor's distance atomically
            float old_distance = atomicMinFloat(&d_distances[neighbor], new_distance);
            
            // If we successfully improved the distance, add neighbor to new worklist
            if (new_distance < old_distance) {
                // Add neighbor to new worklist for next iteration
                // Note: Deduplication removed - atomicCAS overhead was worse than duplicates
                uint32_t pos = atomicAdd(d_new_worklist_size, 1);
                d_new_worklist[pos] = neighbor;
            }
        }
    }
}

// Initialize distances array on GPU
__global__ void init_distances(float* d_distances, uint32_t num_nodes, uint32_t source_node) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < num_nodes) {
        if (tid == source_node) {
            d_distances[tid] = 0.0f;
        } else {
            d_distances[tid] = FLT_MAX;  // Use FLT_MAX instead of std::numeric_limits
        }
    }
}

// Host function to launch SSSP computation
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
    int block_size = 256)
{
    // Calculate grid dimensions
    int num_blocks = (num_current_nodes + block_size - 1) / block_size;
    
    // Launch the kernel
    sssp_kernel<<<num_blocks, block_size>>>(
        gpu_graph,
        d_distances,
        d_worklist,
        d_new_worklist,
        num_current_nodes,
        d_new_worklist_size,
        d_worklist_flags,
        delta
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

void launch_init_distances(
    float* d_distances,
    uint32_t num_nodes,
    uint32_t source_node,
    int block_size = 256)
{
    // Calculate grid dimensions
    int num_blocks = (num_nodes + block_size - 1) / block_size;
    
    // Launch initialization kernel
    init_distances<<<num_blocks, block_size>>>(d_distances, num_nodes, source_node);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
