#include "delta_stepping.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>

int main(int argc, char* argv[]) {
    std::cout << "=== GPU Delta-stepping Algorithm Test ===" << std::endl;
    
    // Default graph file path
    std::vector<std::string> possible_paths = {
        "data/processed/nyc_graph.bin",
        "nyc_graph.bin",
        "../nyc_graph.bin",
        "../../nyc_graph.bin"
    };
    
    std::string graph_file;
    if (argc > 1) {
        graph_file = argv[1];
    } else {
        for (const auto& path : possible_paths) {
            std::ifstream test_file(path);
            if (test_file.good()) {
                graph_file = path;
                std::cout << "Found graph file at: " << path << std::endl;
                break;
            }
        }
        if (graph_file.empty()) {
            graph_file = "nyc_graph.bin";
        }
    }
    
    // Initialize CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Load graph to GPU
    GPUGraphLoader graph_loader;
    if (!graph_loader.loadFromFile(graph_file)) {
        std::cerr << "Error: Failed to load graph from " << graph_file << std::endl;
        return 1;
    }
    
    graph_loader.printGraphStats();
    
    // Initialize GPU Delta-stepping
    GPUDeltaStepping delta_stepping(&graph_loader);
    
    if (!delta_stepping.isInitialized()) {
        std::cerr << "Error: Failed to initialize delta-stepping algorithm" << std::endl;
        return 1;
    }
    
    // Configure algorithm
    if (argc >= 5) {
        float delta = std::atof(argv[4]);
        delta_stepping.setDelta(delta);
        std::cout << "Using custom delta: " << delta << std::endl;
    }
    
    // Test cases
    std::vector<std::pair<uint32_t, uint32_t>> test_cases;
    
    if (argc >= 4) {
        // Custom source and target from command line
        uint32_t source = std::atoi(argv[2]);
        uint32_t target = std::atoi(argv[3]);
        test_cases.push_back({source, target});
    } else {
        // Default test cases (same as baseline for comparison)
        test_cases.push_back({0, 100});
        test_cases.push_back({2931668, 5863336});
        
        // Add some random test cases
        srand(time(nullptr));
        const GPUGraph& gpu_graph = graph_loader.getGPUGraph();
        for (int i = 0; i < 3; i++) {
            uint32_t source = rand() % gpu_graph.num_nodes;
            uint32_t target = rand() % gpu_graph.num_nodes;
            test_cases.push_back({source, target});
        }
    }
    
    // Run test cases
    for (size_t i = 0; i < test_cases.size(); i++) {
        uint32_t source = test_cases[i].first;
        uint32_t target = test_cases[i].second;
        
        std::cout << "\n--- Delta-stepping Test Case " << (i + 1) << " ---" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        float distance = delta_stepping.findShortestPath(source, target);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (distance >= 0) {
            delta_stepping.printStatistics(source, target, distance, time_ms, 0);
        } else {
            std::cout << "Error: Could not find path from " << source << " to " << target << std::endl;
        }
    }
    
    // Batch processing test
    if (test_cases.size() > 1) {
        std::cout << "\n--- Batch Processing Test ---" << std::endl;
        
        std::vector<uint32_t> sources, targets;
        for (const auto& test_case : test_cases) {
            sources.push_back(test_case.first);
            targets.push_back(test_case.second);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> batch_results = delta_stepping.findShortestPaths(sources, targets);
        auto end = std::chrono::high_resolution_clock::now();
        
        double batch_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "Batch processing completed:" << std::endl;
        std::cout << "  Queries: " << batch_results.size() << std::endl;
        std::cout << "  Total time: " << batch_time_ms << " ms" << std::endl;
        std::cout << "  Average time per query: " << batch_time_ms / batch_results.size() << " ms" << std::endl;
        std::cout << "  Throughput: " << (batch_results.size() * 1000.0) / batch_time_ms << " queries/second" << std::endl;
    }
    
    std::cout << "\nDelta-stepping testing completed!" << std::endl;
    return 0;
}
