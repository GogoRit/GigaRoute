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
    uint8_t* d_worklist_flags,
    const float delta)
{
    // Calculate global thread ID
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Check if thread has work to do
    if (tid >= num_current_nodes) {
        return;
    }
    
    // Get the node this thread will process
    uint32_t current_node = d_worklist[tid];
    
    // Get current distance to this node
    float current_distance = d_distances[current_node];
    
    // Get the range of edges for this node using CSR format
    uint32_t edge_start = d_graph.d_row_pointers[current_node];
    uint32_t edge_end = d_graph.d_row_pointers[current_node + 1];
    
    // Process all outgoing edges from current_node
    for (uint32_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
        // Get neighbor node and edge weight
        uint32_t neighbor = d_graph.d_column_indices[edge_idx];
        float edge_weight = d_graph.d_values[edge_idx];
        
        // Calculate new potential distance to neighbor
        float new_distance = current_distance + edge_weight;
        
        // OPTIMIZATION: Pre-check distance before expensive atomic operation
        // Read current distance first (non-atomic, may be stale but safe for comparison)
        float current_neighbor_distance = d_distances[neighbor];
        
        // Only perform atomic update if we might improve the distance
        // This avoids expensive atomic operations when update won't help
        if (new_distance < current_neighbor_distance) {
            // Try to update neighbor's distance atomically
            float old_distance = atomicMinFloat(&d_distances[neighbor], new_distance);
            
            // If we successfully improved the distance, add neighbor to new worklist
            if (new_distance < old_distance) {
                // OPTIMIZATION: Deduplication - check if already in worklist
                // Use atomic exchange: if flag is 0, set to 1 and get 0 back (success)
                // If flag is already 1, get 1 back (already in worklist, skip)
                uint8_t old_flag = atomicExch(&d_worklist_flags[neighbor], 1);
                if (old_flag == 0) {
                    // Successfully set flag, node not in worklist yet - add it
                    uint32_t pos = atomicAdd(d_new_worklist_size, 1);
                    d_new_worklist[pos] = neighbor;
                }
                // If old_flag was 1, node is already in worklist - skip
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
    uint8_t* d_worklist_flags,
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
