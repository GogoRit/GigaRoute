#!/bin/bash
# Mac Development Setup for CUDA Project
# This script sets up local development environment and remote GPU testing

set -e

echo "Setting up Mac development environment for CUDA Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is designed for macOS${NC}"
    exit 1
fi

echo -e "${BLUE}Checking Mac development dependencies...${NC}"

# Check Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${YELLOW}CMake not found. Installing...${NC}"
    brew install cmake
fi

# Check required libraries for CPU-only builds
echo -e "${BLUE}Installing development dependencies...${NC}"
brew install zlib bzip2 lz4 expat

# Optional: Install libosmium for preprocessing (if needed)
if ! brew list libosmium &> /dev/null; then
    echo -e "${YELLOW}Installing libosmium for OSM preprocessing...${NC}"
    brew install libosmium
fi

echo -e "${GREEN}Mac development environment ready${NC}"

# Create local build directory for CPU-only testing
echo -e "${BLUE}Setting up local build environment...${NC}"
mkdir -p build-mac
cd build-mac

# Configure for CPU-only build (no CUDA on Mac)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_GPU_TARGETS=OFF \
         -DCMAKE_CXX_STANDARD=17

echo -e "${GREEN}Local build configured (CPU-only for Mac)${NC}"
echo -e "${BLUE}To build CPU components: cd build-mac && make${NC}"
echo -e "${BLUE}To test on remote GPU: use scripts/remote_test.sh${NC}"
