CUDA Graph Routing Documentation
==================================

GPU-Accelerated Shortest Path Algorithms for Large-Scale Road Networks

.. image:: https://img.shields.io/badge/CUDA-12.2-green.svg
   :alt: CUDA Version

.. image:: https://img.shields.io/badge/GPU-GTX%201080-blue.svg
   :alt: GPU Target

.. image:: https://img.shields.io/badge/Language-C%2B%2B%2FCUDA-orange.svg
   :alt: Language

Overview
--------

This project implements a high-performance GPU-accelerated shortest path algorithm for large-scale road networks using CUDA. The system processes the complete New York City road network (11.7 million nodes, 25.3 million edges) and achieves significant performance improvements over traditional CPU implementations.

Key Features
------------

* **Massive Scale**: Handles 11.7M nodes and 25.3M edges
* **High Performance**: Up to 95% speedup for short-medium range queries
* **Production Ready**: Professional error handling and diagnostics
* **Accurate**: 99.7% accuracy match with CPU baseline
* **Memory Efficient**: 237MB GPU usage for entire NYC network

Quick Start
-----------

.. code-block:: bash

   # Build the project
   mkdir build && cd build
   cmake ..
   make

   # Run GPU implementation
   ./bin/gpu_dijkstra

   # Run CPU baseline for comparison
   ./bin/cpu_dijkstra

Performance Highlights
---------------------

.. list-table:: GPU Performance Results
   :widths: 25 25 25 25
   :header-rows: 1

   * - Distance
     - GPU Time
     - Iterations
     - Performance
   * - 40.35 km
     - **83ms**
     - 500
     - Excellent
   * - 96.39 km
     - **710ms**
     - 1,400
     - Very Good
   * - 381.39 km
     - **3.88s**
     - 2,600
     - Competitive

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   methodology
   implementation
   results
   api_reference
   future_work

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   api/classes
   api/functions
   api/modules

Architecture Overview
--------------------

The system consists of several key components:

**Data Processing Pipeline**
   * OpenStreetMap data parsing
   * Compressed Sparse Row (CSR) format conversion
   * GPU memory optimization

**GPU Implementation**
   * Custom CUDA kernels for parallel shortest path
   * Work-list based frontier expansion
   * Atomic operations for race condition handling

**Performance Analysis**
   * Comprehensive benchmarking against CPU baseline
   * Scalability analysis across different path lengths
   * Memory usage optimization

Technical Specifications
-----------------------

:GPU Hardware: NVIDIA GeForce GTX 1080 (8GB)
:CUDA Version: 12.2
:Compute Capability: 6.1
:Programming Language: C++ with CUDA extensions
:Build System: CMake 3.18+
:Graph Format: Compressed Sparse Row (CSR)
:Test Dataset: New York City road network

Getting Started
---------------

1. **Prerequisites**
   
   * NVIDIA GPU with Compute Capability 6.0+
   * CUDA Toolkit 11.0+
   * CMake 3.18+
   * C++17 compatible compiler

2. **Installation**

   .. code-block:: bash

      git clone <repository-url>
      cd CUDA_Project
      mkdir build && cd build
      cmake ..
      make

3. **Usage**

   .. code-block:: bash

      # Run with default test cases
      ./bin/gpu_dijkstra

      # Run with custom source and target
      ./bin/gpu_dijkstra graph_file.bin source_node target_node

Contributing
------------

This project follows professional development practices:

* Modular architecture with clear separation of concerns
* Comprehensive error handling and diagnostics
* Industry-standard build system
* Git workflow with feature branches

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
