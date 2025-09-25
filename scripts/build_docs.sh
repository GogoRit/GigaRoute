#!/bin/bash
# Documentation Build Script for CUDA Graph Routing Project

set -e  # Exit on any error

echo "=== CUDA Graph Routing Documentation Builder ==="
echo "Building comprehensive documentation with Doxygen + Sphinx"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the project root
if [ ! -f "Doxyfile" ]; then
    print_error "Doxyfile not found. Please run this script from the project root."
    exit 1
fi

# Step 1: Generate Doxygen API Documentation
print_status "Step 1: Generating API documentation with Doxygen..."

if command -v doxygen &> /dev/null; then
    # Clean previous output
    rm -rf docs/api
    
    # Generate Doxygen documentation
    print_status "Running Doxygen..."
    doxygen Doxyfile
    
    if [ -d "docs/api/html" ]; then
        print_success "Doxygen HTML documentation generated successfully"
    else
        print_error "Doxygen failed to generate HTML documentation"
        exit 1
    fi
    
    if [ -d "docs/api/xml" ]; then
        print_success "Doxygen XML output generated for Sphinx integration"
    else
        print_warning "Doxygen XML output not found (needed for Sphinx integration)"
    fi
else
    print_warning "Doxygen not found. Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen graphviz"
    echo "  macOS: brew install doxygen graphviz"
    echo "  Windows: Download from https://www.doxygen.nl/download.html"
fi

# Step 2: Build Sphinx Documentation
print_status "Step 2: Building comprehensive documentation with Sphinx..."

cd docs/sphinx

if command -v sphinx-build &> /dev/null; then
    # Clean previous build
    rm -rf _build
    
    # Build HTML documentation
    print_status "Building Sphinx HTML documentation..."
    sphinx-build -b html . _build/html
    
    if [ -d "_build/html" ]; then
        print_success "Sphinx HTML documentation built successfully"
        
        # Create convenient access link
        if [ ! -L "../../docs/index.html" ]; then
            ln -s sphinx/_build/html/index.html ../../docs/index.html
            print_status "Created convenient access link: docs/index.html"
        fi
    else
        print_error "Sphinx failed to build HTML documentation"
        exit 1
    fi
    
    # Build PDF documentation (if LaTeX is available)
    if command -v pdflatex &> /dev/null; then
        print_status "Building PDF documentation..."
        sphinx-build -b latex . _build/latex
        cd _build/latex
        make
        if [ -f "CUDAGraphRouting.pdf" ]; then
            cp CUDAGraphRouting.pdf ../../../CUDAGraphRouting.pdf
            print_success "PDF documentation generated: CUDAGraphRouting.pdf"
        fi
        cd ../../
    else
        print_warning "LaTeX not found. PDF generation skipped."
        echo "  Ubuntu/Debian: sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra"
        echo "  macOS: brew install --cask mactex"
    fi
else
    print_warning "Sphinx not found. Install with:"
    echo "  pip install sphinx sphinx_rtd_theme breathe myst-parser"
fi

cd ../..

# Step 3: Generate Documentation Summary
print_status "Step 3: Generating documentation summary..."

cat > docs/README.md << 'EOF'
# CUDA Graph Routing Documentation

This directory contains comprehensive documentation for the GPU-accelerated shortest path system.

## Documentation Structure

### 📚 **Comprehensive Guide** (`sphinx/`)
- **Main Documentation**: `docs/index.html` → Professional Sphinx documentation
- **Methodology**: Detailed algorithm descriptions and mathematical analysis
- **Implementation**: Complete system architecture and code organization  
- **Results**: Comprehensive performance analysis and benchmarking
- **API Reference**: Auto-generated from code with Doxygen integration
- **Future Work**: Research directions and optimization roadmap

### 🔧 **API Reference** (`api/`)
- **Code Documentation**: Auto-generated with Doxygen
- **Class Hierarchy**: Complete inheritance diagrams
- **Function Reference**: Detailed parameter descriptions
- **Source Browser**: Searchable code navigation

### 📊 **Project Progress** 
- **Progress.md**: Detailed development log with all phases
- **Performance Charts**: Visual analysis and comparisons
- **Technical Achievements**: Implementation milestones

## Quick Access

- **[Main Documentation](index.html)** - Start here for complete guide
- **[API Reference](api/html/index.html)** - Code documentation
- **[Project Progress](Progress.md)** - Development timeline

## Building Documentation

Run the documentation builder:
```bash
./scripts/build_docs.sh
```

This generates:
- Doxygen API documentation
- Sphinx comprehensive guide  
- PDF report (if LaTeX available)
- Convenient access links

## Academic Quality

The documentation is designed for:
- **Academic Submission**: Conference/journal ready
- **Professional Portfolio**: Industry presentation quality  
- **Open Source**: Community contribution standard
- **Education**: Teaching and learning resource

Generated with professional tools: Doxygen + Sphinx + LaTeX
EOF

print_success "Documentation summary created: docs/README.md"

# Step 4: Final Report
echo ""
echo "=== Documentation Build Complete ==="
echo ""
print_success "✅ Professional documentation system ready!"
echo ""
echo "📁 Generated Documentation:"
if [ -d "docs/api/html" ]; then
    echo "   🔧 API Reference:     docs/api/html/index.html"
fi
if [ -d "docs/sphinx/_build/html" ]; then
    echo "   📚 Main Guide:        docs/index.html (→ sphinx build)"
fi
if [ -f "CUDAGraphRouting.pdf" ]; then
    echo "   📄 PDF Report:        CUDAGraphRouting.pdf"
fi
echo ""
echo "🌐 Quick Start:"
echo "   Open: docs/index.html"
echo "   Or:   docs/api/html/index.html"
echo ""
echo "🎯 Ready for:"
echo "   ✓ Academic presentation"
echo "   ✓ Professional portfolio"  
echo "   ✓ Conference submission"
echo "   ✓ Industry showcase"
echo ""
print_status "Your GPU routing project documentation is production-ready! 🚀"
