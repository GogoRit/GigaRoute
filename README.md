# CUDA Graph Routing Project

A high-performance GPU-accelerated implementation of shortest path algorithms for large-scale road networks using CUDA.

## Project Overview

This project implements and compares CPU and GPU versions of Dijkstra's shortest path algorithm on the New York City road network (11.7M nodes, 25.3M edges). The implementation progresses through three phases:

1. **Data Preprocessing**: Convert OpenStreetMap data to GPU-friendly CSR format
2. **CPU Baseline**: Single-threaded Dijkstra implementation for performance baseline  
3. **GPU Implementation**: CUDA-accelerated parallel shortest path computation

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

### Quick Build (GPU Implementation)

```bash
mkdir build && cd build
cmake ..
make gpu_dijkstra cpu_dijkstra verify_graph
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

### 3. GPU Implementation

```bash
./bin/gpu_dijkstra ../../data/processed/nyc_graph.bin [source] [target]
```

### 4. Graph Verification

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
- **Performance**: [To be measured]
- **Memory**: GPU memory optimized with CSR format

## Algorithm Details

### Data Structure
- **Format**: Compressed Sparse Row (CSR) for memory efficiency
- **Weights**: Haversine distance in meters
- **Graph**: Bidirectional road network

### CPU Implementation
- Standard Dijkstra's algorithm with priority queue
- Early termination when target is reached
- Path reconstruction with predecessors

### GPU Implementation  
- Work-list based parallel SSSP kernel
- Atomic operations for distance updates
- Multi-iteration frontier expansion
- Memory-coalesced graph traversal

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
