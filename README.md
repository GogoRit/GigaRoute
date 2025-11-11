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
│   ├── common/          # Shared graph utilities
│   ├── preprocessing/   # OSM data conversion (Phase 1)
│   ├── cpu/            # CPU baseline implementation (Phase 2)  
│   ├── gpu/            # GPU CUDA implementation (Phase 3)
│   └── utils/          # Verification and benchmarking tools
├── data/
│   ├── raw/            # Original OSM data files (gitignored)
│   └── processed/      # Binary graph files (gitignored)
├── docs/               # Documentation and progress reports
├── build/              # Build artifacts (gitignored)
└── scripts/            # Build and execution scripts
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

### CPU Baseline (Single-threaded)
- **Hardware**: Standard CPU
- **Performance**: ~1.5-2.4 seconds per query
- **Memory**: ~350MB for NYC graph

### GPU Implementation (CUDA)
- **Hardware**: NVIDIA GTX 1080
- **GPU Work-list SSSP**: ~3.8 seconds per query (2600 iterations)
- **GPU Delta-stepping**: ~84 seconds per query (7607 iterations) - baseline implementation
- **Memory**: GPU memory optimized with CSR format (~237MB for NYC graph)

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
- Work-list based parallel SSSP kernel
- Atomic operations for distance updates
- Multi-iteration frontier expansion
- Memory-coalesced graph traversal

### GPU Delta-stepping
- **Status**: Baseline implementation complete and verified
- Bucketed priority queue approach for parallel processing
- Configurable delta parameter (currently 50m for NYC graph)
- Successfully finds paths matching CPU Dijkstra results (verified: 380.361 km)
- **Performance**: Currently slower than GPU work-list SSSP - optimization phase next
- Batch processing capabilities (framework ready)

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
