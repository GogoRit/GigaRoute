# CUDA Graph Routing Project

A high-performance GPU-accelerated implementation of shortest path algorithms for large-scale road networks using CUDA.

## Project Overview

This project implements and compares CPU and GPU versions of shortest path algorithms on the New York City road network (11.7M nodes, 25.3M edges). The implementation includes multiple advanced algorithms:

1. **Data Preprocessing**: Convert OpenStreetMap data to GPU-friendly CSR format
2. **CPU Baseline**: Single-threaded Dijkstra implementation for performance baseline  
3. **GPU Work-list SSSP**: CUDA-accelerated parallel shortest path computation
4. **GPU Delta-stepping**: Advanced bucketed algorithm for improved convergence

## Project Structure

```
├── src/
│   ├── common/              # Shared graph utilities
│   ├── preprocessing/       # OSM data conversion (Phase 1)
│   ├── cpu/                  # CPU baseline implementation (Phase 2)
│   │   ├── dijkstra_cpu.cpp
│   │   ├── dijkstra_cpu.h
│   │   └── main_cpu.cpp
│   ├── gpu/                  # GPU CUDA implementations (Phase 3+)
│   │   ├── common/           # Shared GPU utilities
│   │   │   ├── gpu_graph.cu
│   │   │   └── gpu_graph.h
│   │   ├── dijkstra/         # GPU Work-list SSSP algorithm
│   │   │   ├── main_gpu.cu
│   │   │   └── sssp_kernel.cu
│   │   └── delta_stepping/   # GPU Delta-stepping algorithm
│   │       ├── delta_stepping.cu
│   │       ├── delta_stepping.h
│   │       ├── delta_stepping_kernel.cu
│   │       └── main_delta_stepping.cu
│   └── utils/                # Verification and benchmarking tools
├── data/
│   ├── raw/                  # Original OSM data files (gitignored)
│   └── processed/            # Binary graph files (gitignored)
├── docs/                     # Documentation and progress reports
├── build/                    # Build artifacts (gitignored)
└── scripts/                  # Build and execution scripts
```

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 6.0+ (GTX 1080 or newer)
- Minimum 8GB GPU memory recommended
- 16GB+ system RAM

### Software
- CUDA Toolkit 11.0+ (tested with CUDA 12.2)
- CMake 3.18+
- C++17 compatible compiler
- Optional: libosmium for data preprocessing

## Building

### macOS Development (CPU-only)

For development on Mac with remote GPU testing:

```bash
# Setup Mac environment
./scripts/mac_dev_setup.sh

# Build CPU components locally
mkdir build-mac && cd build-mac
cmake .. -DBUILD_GPU_TARGETS=OFF
make cpu_dijkstra verify_graph graph_converter
```

### Linux/Windows GPU Build

```bash
mkdir build && cd build
cmake ..
make gpu_dijkstra delta_stepping cpu_dijkstra verify_graph
```

### Full Build (Including Preprocessing)

If you have libosmium installed:

```bash
mkdir build && cd build
cmake ..
make  # Builds all targets including graph_converter
```

## Usage

### 1. Data Preprocessing (if needed)

```bash
./bin/graph_converter input.osm.pbf output.bin
```

### 2. CPU Baseline

```bash
./bin/cpu_dijkstra ../../data/processed/nyc_graph.bin [source] [target]
```

### 3. GPU Work-list SSSP

```bash
./bin/gpu_dijkstra ../../data/processed/nyc_graph.bin [source] [target]
```

### 4. GPU Delta-stepping

```bash
./bin/delta_stepping ../../data/processed/nyc_graph.bin [source] [target] [delta]
```

### 5. Remote GPU Testing (from Mac)

```bash
./scripts/remote_test.sh username@gpu-server.edu
```

### 6. Graph Verification

```bash
./bin/verify_graph ../../data/processed/nyc_graph.bin
```

## Performance Results

### Test Environment
- **Graph**: NYC road network (11.7M nodes, 25.3M edges)
- **Hardware**: NVIDIA GTX 1080 (Compute Capability 6.1, 8GB VRAM)
- **Test Case**: Node 0 → Node 1000 (434.224 km)

### Algorithm Performance (November 2025)

| Algorithm | Performance | Accuracy | Status |
|-----------|-------------|----------|--------|
| **CPU Dijkstra** | 3.6s | 100% | Production Ready |
| **GPU Dijkstra** | 17.4s | 100% | Production Ready |
| **Delta-Stepping** | 26.8s (δ=2000) | 99.99% | Configurable Ready |

### Performance Characteristics
- **GPU Memory Usage**: ~237MB for NYC graph
- **GPU Optimizations**: Pre-check atomic operations, bounds checking, memory coalescing
- **Convergence Detection**: 1000-iteration stability window
- **GPU Advantage**: Scales with graph size and query batching

## Algorithm Details

### Data Structure
- **Format**: Compressed Sparse Row (CSR) for memory efficiency
- **Weights**: Haversine distance in meters
- **Graph**: Bidirectional road network

### CPU Implementation
- Standard Dijkstra's algorithm with priority queue
- Early termination when target is reached
- Path reconstruction with predecessors

### GPU Work-list SSSP
- **Status**: Optimized implementation complete and verified
- Work-list based parallel SSSP kernel with atomic operations
- Pre-check optimization to reduce atomic contention by ~50%
- Memory-coalesced graph traversal with bounds checking
- Smart convergence detection (1000-iteration stability window)
- Production-ready with consistent results matching CPU baseline

### GPU Delta-stepping
- **Status**: Configurable implementation complete and verified
- Bucketed priority queue approach for parallel processing
- Adaptive delta parameter (tested: 100-2000 range)
- Successfully finds paths matching CPU Dijkstra results (434.272 km with δ=2000)
- **Performance**: Configurable trade-off between speed and accuracy
- Configurable delta values for different graph characteristics

## Project Milestone (November 2025)

### Completed Achievements
- **Algorithm Implementation**: All three algorithms (CPU Dijkstra, GPU Dijkstra, Delta-Stepping) implemented and verified
- **Correctness Verification**: All algorithms produce consistent results (CPU/GPU accuracy within 0.01%)
- **Performance Optimization**: GPU Dijkstra optimized with 6% performance improvement
- **Stability**: Robust convergence detection and error handling
- **Documentation**: Comprehensive code documentation and performance analysis

### Current Capabilities
- **Single Query Processing**: All algorithms handle individual shortest path queries
- **Multi-Algorithm Comparison**: Direct performance comparison between CPU/GPU approaches
- **Configurable Parameters**: Delta-stepping supports variable delta values for different use cases
- **Memory Optimization**: Efficient GPU memory usage (~237MB for large graphs)
- **Cross-Platform**: CPU-only development on macOS, GPU testing on Linux servers

### Next Steps & Future Work
- **Large Graph Scaling**: Test on larger graphs (100M+ nodes) where GPU advantage becomes significant
- **Batch Query Processing**: Implement parallel processing of multiple simultaneous queries
- **ML Integration**: PyTorch CUDA extensions for graph neural network feature computation
- **Advanced Optimizations**: Shared memory utilization, warp-level primitives, kernel fusion
- **Real-time Applications**: Streaming query processing for routing systems

## Development

### Adding New Algorithms

1. Create new directory under `src/gpu/` for your algorithm
2. Implement kernel in `.cu` file with `__global__` functions
3. Add launcher functions and host interface
4. Update `CMakeLists.txt` to include new target
5. Add tests and benchmarks

### Testing

```bash
# Run all default test cases
./bin/cpu_dijkstra
./bin/gpu_dijkstra

# Custom test case
./bin/gpu_dijkstra ../../data/processed/nyc_graph.bin 0 100000
```

## Contributing

1. Follow the established directory structure
2. Use the common graph parser for consistency
3. Add appropriate error checking and CUDA_CHECK macros
4. Document performance results in `docs/Progress.md`

## References

- [ADDS Paper](docs/paper1.pdf): Accelerated Dijkstra's Shortest Path
- [PD3 Paper](docs/paper2.pdf): Parallel Delta-Stepping Implementation
- [Project Progress](docs/Progress.md): Detailed development log

## License

Academic/Educational use - RIT CUDA Project 2025
