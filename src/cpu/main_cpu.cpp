#include "../common/graph_parser.h"
#include "dijkstra_cpu.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
    std::cout << "=== CPU Dijkstra's Algorithm Baseline ===" << std::endl;
    
    // Default graph file path
    std::string graph_file = "../../data/processed/nyc_graph.bin";
    if (argc > 1) {
        graph_file = argv[1];
    }
    
    // Load graph
    GraphParser parser;
    if (!parser.loadFromFile(graph_file)) {
        std::cerr << "Error: Failed to load graph from " << graph_file << std::endl;
        return 1;
    }
    
    // Print graph statistics
    parser.printStatistics();
    
    // Validate graph
    if (!parser.validateGraph()) {
        std::cerr << "Error: Graph validation failed" << std::endl;
        return 1;
    }
    
    // Initialize Dijkstra pathfinder
    DijkstraPathfinder pathfinder(parser.getGraph());
    
    // Test cases
    std::vector<std::pair<uint32_t, uint32_t>> test_cases;
    
    if (argc >= 4) {
        // Custom source and target from command line
        uint32_t source = std::atoi(argv[2]);
        uint32_t target = std::atoi(argv[3]);
        test_cases.push_back({source, target});
    } else {
        // Default test cases from your baseline results
        test_cases.push_back({0, 100});          // Test case from Progress.md
        test_cases.push_back({2931668, 5863336}); // Test case from Progress.md
        
        // Add a few random test cases
        srand(time(nullptr));
        const Graph& graph = parser.getGraph();
        for (int i = 0; i < 3; i++) {
            uint32_t source = rand() % graph.num_nodes;
            uint32_t target = rand() % graph.num_nodes;
            test_cases.push_back({source, target});
        }
    }
    
    // Run test cases
    for (size_t i = 0; i < test_cases.size(); i++) {
        uint32_t source = test_cases[i].first;
        uint32_t target = test_cases[i].second;
        
        std::cout << "\n--- Test Case " << (i + 1) << " ---" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        float distance = pathfinder.findShortestPath(source, target);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::vector<uint32_t> path = pathfinder.getPath(source, target);
        pathfinder.printResult(source, target, distance, time_ms, path);
    }
    
    std::cout << "\nCPU baseline computation completed!" << std::endl;
    return 0;
}
