#!/bin/bash
# Comprehensive test script to compare delta-stepping vs work-list SSSP
# Tests different scenarios where delta-stepping might excel

echo "=========================================="
echo "Delta-Stepping vs Work-List SSSP Comparison"
echo "=========================================="
echo ""
echo "When Delta-Stepping is Better:"
echo "1. Graphs with highly variable edge weights (highways vs local roads)"
echo "2. Very long paths (>400km) where bucket organization helps"
echo "3. Multi-source SSSP (batch queries)"
echo "4. Graphs with wide edge weight distribution"
echo ""
echo "When Work-List SSSP is Better:"
echo "1. Road networks with uniform edge weights (like NYC: 0-3km)"
echo "2. Short to medium paths (<400km)"
echo "3. Single-source queries"
echo "4. When low overhead is critical"
echo ""
echo "=========================================="
echo ""

# Find graph file (try multiple locations)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_FILE=""

# Try different possible locations
for path in "$SCRIPT_DIR/nyc_graph.bin" "$SCRIPT_DIR/../nyc_graph.bin" "$SCRIPT_DIR/build/nyc_graph.bin" "$SCRIPT_DIR/../build/nyc_graph.bin"; do
    if [ -f "$path" ]; then
        GRAPH_FILE="$path"
        break
    fi
done

if [ -z "$GRAPH_FILE" ]; then
    echo "Error: Could not find nyc_graph.bin"
    echo "Searched in:"
    echo "  $SCRIPT_DIR/nyc_graph.bin"
    echo "  $SCRIPT_DIR/../nyc_graph.bin"
    echo "  $SCRIPT_DIR/build/nyc_graph.bin"
    echo "  $SCRIPT_DIR/../build/nyc_graph.bin"
    exit 1
fi

echo "Using graph file: $GRAPH_FILE"
echo ""

BUILD_DIR="$SCRIPT_DIR/build/bin"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR" || exit 1

echo "Test 1: Short Path (0 -> 100, ~380km)"
echo "--------------------------------------"
echo "GPU Work-List SSSP:"
./gpu_dijkstra "$GRAPH_FILE" 0 100 | grep "Computation time"
echo ""
echo "GPU Delta-Stepping (delta=50m):"
./delta_stepping "$GRAPH_FILE" 0 100 50 | grep "Computation time"
echo ""
echo "GPU Delta-Stepping (delta=100m):"
./delta_stepping "$GRAPH_FILE" 0 100 100 | grep "Computation time"
echo ""
echo "GPU Delta-Stepping (delta=200m):"
./delta_stepping "$GRAPH_FILE" 0 100 200 | grep "Computation time"
echo ""
echo "CPU Baseline:"
./cpu_dijkstra "$GRAPH_FILE" 0 100 | grep "Computation time"
echo ""

echo "Test 2: Medium Path (2931668 -> 5863336, ~96km)"
echo "------------------------------------------------"
echo "GPU Work-List SSSP:"
./gpu_dijkstra "$GRAPH_FILE" 2931668 5863336 | grep "Computation time"
echo ""
echo "GPU Delta-Stepping (delta=100m):"
./delta_stepping "$GRAPH_FILE" 2931668 5863336 100 | grep "Computation time"
echo ""
echo "CPU Baseline:"
./cpu_dijkstra "$GRAPH_FILE" 2931668 5863336 | grep "Computation time"
echo ""

echo "Test 3: Long Path (0 -> 5000000, ~410km)"
echo "-----------------------------------------"
echo "GPU Work-List SSSP:"
./gpu_dijkstra "$GRAPH_FILE" 0 5000000 | grep "Computation time"
echo ""
echo "GPU Delta-Stepping (delta=100m):"
./delta_stepping "$GRAPH_FILE" 0 5000000 100 | grep "Computation time"
echo ""
echo "GPU Delta-Stepping (delta=500m):"
./delta_stepping "$GRAPH_FILE" 0 5000000 500 | grep "Computation time"
echo ""
echo "CPU Baseline:"
./cpu_dijkstra "$GRAPH_FILE" 0 5000000 | grep "Computation time"
echo ""

echo "Test 4: Very Long Path (0 -> 10000000, if exists)"
echo "---------------------------------------------------"
echo "GPU Work-List SSSP:"
./gpu_dijkstra "$GRAPH_FILE" 0 10000000 2>/dev/null | grep "Computation time" || echo "Path not found or node doesn't exist"
echo ""
echo "GPU Delta-Stepping (delta=100m):"
./delta_stepping "$GRAPH_FILE" 0 10000000 100 2>/dev/null | grep "Computation time" || echo "Path not found or node doesn't exist"
echo ""

echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "For NYC road network (uniform edge weights 0-3km):"
echo "- Work-List SSSP is typically 10-15x faster"
echo "- Delta-stepping overhead (bucket management) > benefits"
echo ""
echo "Delta-stepping would excel on graphs with:"
echo "- Highway networks (edges: 100m-100km range)"
echo "- Mixed transportation (walking + driving)"
echo "- Social networks with variable connection strengths"
echo "- Knowledge graphs with diverse relationship weights"
echo "=========================================="

