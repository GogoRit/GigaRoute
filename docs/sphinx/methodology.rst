Methodology
===========

This section details the algorithmic approach, data structures, and implementation strategy for the GPU-accelerated shortest path system.

Algorithm Overview
-----------------

Single Source Shortest Path (SSSP) Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Single Source Shortest Path problem seeks to find the shortest paths from a given source node to all other nodes in a weighted graph. Our implementation uses a work-list based parallel approach optimized for GPU execution.

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

Given a graph :math:`G = (V, E)` with vertices :math:`V` and edges :math:`E`, and a weight function :math:`w: E \rightarrow \mathbb{R}^+`, the SSSP problem finds the minimum distance :math:`d(s, v)` from source :math:`s` to each vertex :math:`v \in V`.

The distance is defined as:

.. math::

   d(s, v) = \min_{p \in \text{paths}(s,v)} \sum_{(u,w) \in p} w(u,w)

where :math:`\text{paths}(s,v)` represents all possible paths from :math:`s` to :math:`v`.

Work-list Based Parallel Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our GPU implementation uses a work-list (frontier) based approach:

1. **Initialization**: Set :math:`d[s] = 0` and :math:`d[v] = \infty` for all :math:`v \neq s`
2. **Frontier Expansion**: For each iteration, process all nodes in the current work-list
3. **Distance Updates**: Use atomic operations to update neighbor distances
4. **Convergence**: Continue until work-list is empty or target is reached

.. code-block:: cuda

   __global__ void sssp_kernel(
       const GPUGraph d_graph,
       float* d_distances,
       const uint32_t* d_worklist,
       uint32_t* d_new_worklist,
       const uint32_t num_current_nodes,
       uint32_t* d_new_worklist_size,
       const float delta)

Data Structures
--------------

Compressed Sparse Row (CSR) Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The graph is stored in CSR format for optimal GPU memory access:

* **Row Pointers**: Array of size :math:`|V| + 1` indicating edge start positions
* **Column Indices**: Array of size :math:`|E|` containing destination vertices  
* **Values**: Array of size :math:`|E|` containing edge weights

.. code-block:: cpp

   struct GPUGraph {
       uint32_t* d_row_pointers;    // Start indices for each node's edges
       uint32_t* d_column_indices;  // Destination nodes for each edge
       float* d_values;             // Edge weights
       uint32_t num_nodes;          // Total number of nodes
       uint32_t num_edges;          // Total number of edges
   };

Memory Layout Benefits
^^^^^^^^^^^^^^^^^^^^^

The CSR format provides several advantages for GPU computation:

1. **Coalesced Memory Access**: Sequential edge processing enables efficient memory throughput
2. **Compact Storage**: Eliminates sparse matrix overhead
3. **Cache Efficiency**: Spatial locality improves GPU cache utilization

Work-list Management
^^^^^^^^^^^^^^^^^^

The algorithm maintains two work-lists that alternate between iterations:

* **Current Work-list**: Nodes to process in this iteration
* **Next Work-list**: Nodes discovered for next iteration
* **Atomic Counters**: Thread-safe work-list size management

Atomic Operations
----------------

Custom Atomic Minimum for Floats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since CUDA doesn't provide native ``atomicMin`` for floating-point values, we implement a software routine:

.. code-block:: cuda

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

This ensures thread-safe distance updates when multiple threads attempt to relax the same vertex simultaneously.

Race Condition Handling
^^^^^^^^^^^^^^^^^^^^^^

The atomic operations prevent race conditions that could occur when:

1. Multiple threads process edges leading to the same vertex
2. Distance updates happen simultaneously
3. Work-list modifications occur concurrently

Convergence Detection
--------------------

Early Termination Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^

The algorithm implements intelligent convergence detection:

.. code-block:: cpp

   // Check target distance every 100 iterations
   if (iteration % 100 == 0 && iteration > 0) {
       CUDA_CHECK(cudaMemcpy(&target_distance, &d_distances[target], 
                            sizeof(float), cudaMemcpyDeviceToHost));
       if (target_distance < FLT_MAX) {
           std::cout << "Target reached at iteration " << iteration 
                     << " with distance " << target_distance << std::endl;
           break;
       }
   }

Performance Optimization
-----------------------

Thread Block Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Block Size**: 256 threads per block (optimal for GTX 1080)
* **Grid Size**: Dynamically calculated based on work-list size
* **Memory Coalescing**: Ensures 32-thread warp efficiency

Memory Access Patterns
^^^^^^^^^^^^^^^^^^^^^

1. **Sequential Edge Traversal**: Improves cache locality
2. **Atomic Distance Updates**: Minimizes memory conflicts  
3. **Work-list Compaction**: Reduces memory overhead

Scalability Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^

The algorithm scales with:

* **Graph Size**: Linear memory usage with number of edges
* **Path Length**: Iteration count proportional to graph diameter
* **Parallelism**: Work-list size determines GPU utilization

Complexity Analysis
------------------

Time Complexity
^^^^^^^^^^^^^^

* **Best Case**: :math:`O(\log V)` for short paths with high parallelism
* **Average Case**: :math:`O(V + E)` similar to sequential Dijkstra
* **Worst Case**: :math:`O(V^2)` for dense graphs with poor work-list distribution

Space Complexity
^^^^^^^^^^^^^^^

* **Graph Storage**: :math:`O(V + E)` for CSR format
* **Distance Array**: :math:`O(V)` for shortest distances
* **Work-lists**: :math:`O(V)` for frontier management
* **Total GPU Memory**: :math:`O(V + E)` linear scaling

Comparison with Traditional Algorithms
-------------------------------------

vs. Sequential Dijkstra
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Algorithm Comparison
   :widths: 30 35 35
   :header-rows: 1

   * - Aspect
     - Sequential Dijkstra
     - GPU Work-list SSSP
   * - Time Complexity
     - :math:`O((V + E) \log V)`
     - :math:`O(V + E)` average
   * - Space Complexity
     - :math:`O(V)`
     - :math:`O(V + E)`
   * - Parallelism
     - None
     - Massive (thousands of threads)
   * - Memory Pattern
     - Random access
     - Coalesced access
   * - Scalability
     - Poor for large graphs
     - Excellent with GPU memory

The GPU implementation trades some space complexity for massive parallel speedup, making it highly effective for large-scale graph problems.
