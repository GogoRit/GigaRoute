#!/bin/bash
# Professional Testing Script for Variable-Weight Graph Analysis
# Tests delta-stepping vs work-list SSSP on variable-weight graphs

set -e  # Exit on error

echo "================================================================"
echo "Variable-Weight Graph Testing Suite"
echo "Delta-Stepping Algorithm Performance Analysis"
echo "================================================================"
echo ""

# Configuration
UNIFORM_GRAPH="nyc_graph.bin"
VARIABLE_GRAPH="nyc_graph_variable_weights.bin"
BUILD_DIR="build"
BIN_DIR="${BUILD_DIR}/bin"

# Test cases (source, target pairs)
declare -a TEST_CASES=(
    "0 100"           # Short path
    "2931668 5863336"  # Medium path
    "0 5000000"        # Long path
)

# Delta values to test
declare -a DELTA_VALUES=(50 100 200 500 1000)

# Check if graphs exist
if [ ! -f "${UNIFORM_GRAPH}" ]; then
    echo "ERROR: Uniform-weight graph not found: ${UNIFORM_GRAPH}"
    echo "Please generate it first using graph_converter"
    exit 1
fi

if [ ! -f "${VARIABLE_GRAPH}" ]; then
    echo "ERROR: Variable-weight graph not found: ${VARIABLE_GRAPH}"
    echo "Please generate it first using:"
    echo "  ./${BIN_DIR}/graph_converter_variable_weights <osm_file> ${VARIABLE_GRAPH} variable"
    exit 1
fi

# Check if binaries exist
if [ ! -f "${BIN_DIR}/gpu_dijkstra" ] || [ ! -f "${BIN_DIR}/delta_stepping" ]; then
    echo "ERROR: Binaries not found. Please build the project first:"
    echo "  cd ${BUILD_DIR} && make -j4"
    exit 1
fi

echo "Test Configuration:"
echo "  Uniform graph: ${UNIFORM_GRAPH}"
echo "  Variable graph: ${VARIABLE_GRAPH}"
echo "  Test cases: ${#TEST_CASES[@]}"
echo "  Delta values: ${DELTA_VALUES[@]}"
echo ""

# Create results directory
RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Function to extract computation time from output
extract_time() {
    grep "Computation time:" | awk '{print $3}' | sed 's/ms//'
}

# Function to extract distance from output
extract_distance() {
    grep "Shortest distance:" | awk '{print $3}' | sed 's/meters//'
}

# Test uniform-weight graph (baseline)
echo "================================================================"
echo "TEST 1: Uniform-Weight Graph (Baseline)"
echo "================================================================"
echo ""

for test_case in "${TEST_CASES[@]}"; do
    read -r source target <<< "${test_case}"
    echo "Test Case: ${source} -> ${target}"
    echo "-----------------------------------"
    
    # Work-List SSSP
    echo -n "  Work-List SSSP: "
    WORKLIST_TIME=$(cd "${BIN_DIR}" && ./gpu_dijkstra "../../${UNIFORM_GRAPH}" ${source} ${target} 2>/dev/null | extract_time)
    echo "${WORKLIST_TIME} ms"
    
    # Delta-Stepping (best delta)
    echo -n "  Delta-Stepping (delta=100m): "
    DELTA_TIME=$(cd "${BIN_DIR}" && ./delta_stepping "../../${UNIFORM_GRAPH}" ${source} ${target} 100 2>/dev/null | extract_time)
    echo "${DELTA_TIME} ms"
    
    RATIO=$(echo "scale=2; ${DELTA_TIME} / ${WORKLIST_TIME}" | bc)
    echo "  Ratio (Delta/Work-List): ${RATIO}x"
    echo ""
done

# Test variable-weight graph
echo "================================================================"
echo "TEST 2: Variable-Weight Graph (Highway Network)"
echo "================================================================"
echo ""

for test_case in "${TEST_CASES[@]}"; do
    read -r source target <<< "${test_case}"
    echo "Test Case: ${source} -> ${target}"
    echo "-----------------------------------"
    
    # Work-List SSSP
    echo -n "  Work-List SSSP: "
    WORKLIST_TIME=$(cd "${BIN_DIR}" && ./gpu_dijkstra "../../${VARIABLE_GRAPH}" ${source} ${target} 2>/dev/null | extract_time)
    echo "${WORKLIST_TIME} ms"
    
    # Delta-Stepping with different delta values
    BEST_DELTA=""
    BEST_TIME="999999"
    
    for delta in "${DELTA_VALUES[@]}"; do
        echo -n "  Delta-Stepping (delta=${delta}m): "
        DELTA_TIME=$(cd "${BIN_DIR}" && ./delta_stepping "../../${VARIABLE_GRAPH}" ${source} ${target} ${delta} 2>/dev/null | extract_time)
        echo "${DELTA_TIME} ms"
        
        # Track best delta
        if (( $(echo "${DELTA_TIME} < ${BEST_TIME}" | bc -l) )); then
            BEST_TIME="${DELTA_TIME}"
            BEST_DELTA="${delta}"
        fi
    done
    
    RATIO=$(echo "scale=2; ${BEST_TIME} / ${WORKLIST_TIME}" | bc)
    echo "  Best Delta: ${BEST_DELTA}m (${BEST_TIME} ms)"
    echo "  Ratio (Best Delta/Work-List): ${RATIO}x"
    echo ""
done

# Comparative analysis
echo "================================================================"
echo "COMPARATIVE ANALYSIS"
echo "================================================================"
echo ""
echo "Key Metrics:"
echo "  - Weight range ratio: Check graph generation output"
echo "  - Performance ratio: Delta-Stepping / Work-List SSSP"
echo "  - Optimal delta: Delta value with best performance"
echo ""
echo "Expected Results for Variable-Weight Graph:"
echo "  - If weight_range_ratio > 100x: Delta-stepping should be < 5x slower"
echo "  - If weight_range_ratio > 10x: Delta-stepping should be < 10x slower"
echo "  - Optimal delta should match highway segment lengths (500-2000m)"
echo ""

echo "================================================================"
echo "Testing Complete"
echo "Results saved to: ${RESULTS_DIR}"
echo "================================================================"

