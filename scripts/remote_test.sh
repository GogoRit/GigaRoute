#!/bin/bash
# Remote GPU Testing Script for University GTX 1080
# Usage: ./remote_test.sh [username@hostname] [optional-test-case]

set -e

# Configuration
REMOTE_HOST="${1:-your_username@gpu_server.university.edu}"
REMOTE_DIR="/tmp/cuda_project_$(date +%s)"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "Remote GPU Testing Pipeline"
echo "=========================="
echo "Local project: $LOCAL_PROJECT_DIR"
echo "Remote host: $REMOTE_HOST"
echo "Remote directory: $REMOTE_DIR"
echo ""

# Function to run remote commands
run_remote() {
    echo -e "${BLUE}Remote: $1${NC}"
    ssh "$REMOTE_HOST" "$1"
}

# Function to copy files to remote
copy_to_remote() {
    echo -e "${BLUE}Copying $1 to remote...${NC}"
    scp -r "$1" "$REMOTE_HOST:$2"
}

# Step 1: Create remote directory
echo -e "${YELLOW}Step 1: Setting up remote environment${NC}"
run_remote "mkdir -p $REMOTE_DIR"

# Step 2: Copy source code (excluding build directories and data)
echo -e "${YELLOW}Step 2: Copying source code${NC}"
rsync -av --exclude='build*' --exclude='data/' --exclude='.git/' \
    "$LOCAL_PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

# Step 3: Build on remote GPU server
echo -e "${YELLOW}Step 3: Building on remote GPU server${NC}"
run_remote "cd $REMOTE_DIR && mkdir -p build && cd build && cmake .. && make -j4"

# Step 4: Check CUDA environment
echo -e "${YELLOW}Step 4: Checking CUDA environment${NC}"
run_remote "nvidia-smi"
run_remote "nvcc --version"

# Step 5: Run tests
echo -e "${YELLOW}Step 5: Running GPU tests${NC}"

# Check if data file exists, if not provide instructions
run_remote "cd $REMOTE_DIR && ls -la data/processed/ || echo 'Data file not found'"

# Test cases
if [ -n "$2" ]; then
    # Custom test case
    echo "Running custom test case: $2"
    run_remote "cd $REMOTE_DIR/build && ./bin/gpu_dijkstra $2"
else
    # Default test cases
    echo "Running default test cases..."
    run_remote "cd $REMOTE_DIR/build && ./bin/gpu_dijkstra || echo 'Note: Requires nyc_graph.bin in data/processed/'"
fi

# Step 6: Copy results back (if any)
echo -e "${YELLOW}Step 6: Copying results back${NC}"
mkdir -p "$LOCAL_PROJECT_DIR/results/remote_$(date +%Y%m%d_%H%M%S)"
scp -r "$REMOTE_HOST:$REMOTE_DIR/build/*.log" "$LOCAL_PROJECT_DIR/results/" 2>/dev/null || echo "No log files to copy"

# Step 7: Cleanup remote directory
echo -e "${YELLOW}Step 7: Cleaning up remote directory${NC}"
run_remote "rm -rf $REMOTE_DIR"

echo -e "${GREEN}Remote testing completed${NC}"
echo "Results saved to: $LOCAL_PROJECT_DIR/results/"
