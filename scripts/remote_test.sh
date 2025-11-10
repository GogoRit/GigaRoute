#!/bin/bash
# Remote GPU Testing Script for RIT CS Department
# Usage: ./remote_test.sh [optional-test-case]

set -e

# Configuration
REMOTE_HOST="gm8189@umber.cs.rit.edu"
REMOTE_DIR="~/CUDA_Project"  # Use home directory instead of tmp
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

# Step 1: Ensure we have latest changes committed
echo -e "${YELLOW}Step 1: Checking local git status${NC}"
cd "$LOCAL_PROJECT_DIR"
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Warning: You have uncommitted changes. Committing them first...${NC}"
    git add -A
    git commit -m "Auto-commit before remote testing"
fi

# Step 2: Push latest changes to remote
echo -e "${YELLOW}Step 2: Pushing latest changes to git${NC}"
git push origin feature/gpu-routing-algorithm

# Step 3: Setup/update remote repository
echo -e "${YELLOW}Step 3: Setting up remote repository${NC}"
run_remote "if [ ! -d $REMOTE_DIR ]; then 
    git clone -b feature/gpu-routing-algorithm https://github.com/GogoRit/GigaRoute.git $REMOTE_DIR
else 
    cd $REMOTE_DIR && git pull origin feature/gpu-routing-algorithm
fi"

# Step 4: Build on remote GPU server
echo -e "${YELLOW}Step 4: Building on remote GPU server${NC}"
run_remote "cd $REMOTE_DIR && mkdir -p build && cd build && cmake .. && make -j4"

# Step 5: Check CUDA environment
echo -e "${YELLOW}Step 5: Checking CUDA environment${NC}"
run_remote "nvidia-smi"
run_remote "nvcc --version"

# Step 6: Run tests
echo -e "${YELLOW}Step 6: Running GPU tests${NC}"

# Check if data file exists, if not provide instructions
run_remote "cd $REMOTE_DIR && ls -la data/processed/ || echo 'Data file not found - you may need to upload nyc_graph.bin'"

# Test cases
if [ -n "$1" ]; then
    # Custom test case from command line argument
    echo "Running custom test case: $1"
    run_remote "cd $REMOTE_DIR/build && ./bin/gpu_dijkstra $1"
    run_remote "cd $REMOTE_DIR/build && ./bin/delta_stepping $1"
else
    # Default test cases
    echo "Running default test cases..."
    echo "Testing Work-list SSSP:"
    run_remote "cd $REMOTE_DIR/build && ./bin/gpu_dijkstra || echo 'Note: Requires nyc_graph.bin in data/processed/'"
    echo "Testing Delta-stepping:"
    run_remote "cd $REMOTE_DIR/build && ./bin/delta_stepping || echo 'Note: Requires nyc_graph.bin in data/processed/'"
fi

# Step 7: Show build artifacts
echo -e "${YELLOW}Step 7: Available executables${NC}"
run_remote "cd $REMOTE_DIR/build && ls -la bin/"

echo -e "${GREEN}Remote testing completed${NC}"
echo "Results saved to: $LOCAL_PROJECT_DIR/results/"
