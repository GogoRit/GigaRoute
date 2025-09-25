#include "graph_parser.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>

GraphParser::GraphParser() : is_loaded(false) {
    graph.num_nodes = 0;
    graph.num_edges = 0;
}

bool GraphParser::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return false;
    }

    // Read header
    file.read(reinterpret_cast<char*>(&graph.num_nodes), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&graph.num_edges), sizeof(uint32_t));
    
    std::cout << "Loading graph: " << graph.num_nodes << " nodes, " 
              << graph.num_edges << " edges" << std::endl;

    // Resize vectors
    graph.row_pointers.resize(graph.num_nodes + 1);
    graph.column_indices.resize(graph.num_edges);
    graph.values.resize(graph.num_edges);

    // Read graph data
    file.read(reinterpret_cast<char*>(graph.row_pointers.data()), 
              (graph.num_nodes + 1) * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(graph.column_indices.data()), 
              graph.num_edges * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(graph.values.data()), 
              graph.num_edges * sizeof(float));

    file.close();
    is_loaded = true;
    
    std::cout << "Graph loaded successfully" << std::endl;
    return true;
}

void GraphParser::printStatistics() const {
    if (!is_loaded) {
        std::cout << "Graph not loaded" << std::endl;
        return;
    }
    
    std::cout << "\n=== Graph Statistics ===" << std::endl;
    std::cout << "Nodes: " << graph.num_nodes << std::endl;
    std::cout << "Edges: " << graph.num_edges << std::endl;
    std::cout << "Average degree: " << std::fixed << std::setprecision(5) 
              << (double)graph.num_edges / graph.num_nodes << std::endl;
    
    // Calculate degree statistics
    std::vector<uint32_t> degrees;
    for (uint32_t i = 0; i < graph.num_nodes; i++) {
        degrees.push_back(graph.row_pointers[i + 1] - graph.row_pointers[i]);
    }
    
    auto min_max = std::minmax_element(degrees.begin(), degrees.end());
    std::cout << "Min degree: " << *min_max.first << std::endl;
    std::cout << "Max degree: " << *min_max.second << std::endl;
    
    // Edge weight statistics
    auto weight_min_max = std::minmax_element(graph.values.begin(), graph.values.end());
    std::cout << "Min edge weight: " << *weight_min_max.first << " meters" << std::endl;
    std::cout << "Max edge weight: " << std::fixed << std::setprecision(2) 
              << *weight_min_max.second << " meters" << std::endl;
}

bool GraphParser::validateGraph() const {
    if (!is_loaded) {
        std::cerr << "Graph not loaded" << std::endl;
        return false;
    }
    
    // Check row pointers
    if (graph.row_pointers.size() != graph.num_nodes + 1) {
        std::cerr << "Invalid row_pointers size" << std::endl;
        return false;
    }
    
    // Check that row pointers are sorted
    for (uint32_t i = 0; i < graph.num_nodes; i++) {
        if (graph.row_pointers[i] > graph.row_pointers[i + 1]) {
            std::cerr << "Row pointers not sorted at node " << i << std::endl;
            return false;
        }
    }
    
    // Check final row pointer
    if (graph.row_pointers[graph.num_nodes] != graph.num_edges) {
        std::cerr << "Final row pointer doesn't match edge count" << std::endl;
        return false;
    }
    
    // Check column indices
    for (uint32_t i = 0; i < graph.num_edges; i++) {
        if (graph.column_indices[i] >= graph.num_nodes) {
            std::cerr << "Invalid column index " << graph.column_indices[i] 
                      << " at edge " << i << std::endl;
            return false;
        }
    }
    
    std::cout << "Graph validation passed" << std::endl;
    return true;
}

void GraphParser::printSampleEdges(int count) const {
    if (!is_loaded) {
        std::cout << "Graph not loaded" << std::endl;
        return;
    }
    
    std::cout << "\n=== Sample Edges ===" << std::endl;
    std::cout << "Showing first " << count << " edges:" << std::endl;
    
    int shown = 0;
    for (uint32_t node = 0; node < graph.num_nodes && shown < count; node++) {
        uint32_t start = graph.row_pointers[node];
        uint32_t end = graph.row_pointers[node + 1];
        
        for (uint32_t edge_idx = start; edge_idx < end && shown < count; edge_idx++) {
            uint32_t neighbor = graph.column_indices[edge_idx];
            float weight = graph.values[edge_idx];
            
            std::cout << "Edge " << shown << ": " << node << " -> " << neighbor 
                      << " (weight: " << std::fixed << std::setprecision(2) 
                      << weight << " meters)" << std::endl;
            shown++;
        }
    }
}
