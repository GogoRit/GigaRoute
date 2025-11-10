#ifndef GRAPH_PARSER_H
#define GRAPH_PARSER_H

#include <vector>
#include <string>
#include <cstdint>

// Common graph structure for both CPU and GPU implementations
struct Graph {
    std::vector<uint32_t> row_pointers;    // CSR row pointers
    std::vector<uint32_t> column_indices;  // CSR column indices
    std::vector<float> values;             // Edge weights
    uint32_t num_nodes;                    // Number of nodes
    uint32_t num_edges;                    // Number of edges
};

// Graph parser class for loading from binary files
class GraphParser {
private:
    Graph graph;
    bool is_loaded;
    
public:
    GraphParser();
    ~GraphParser() = default;
    
    // Load graph from binary file
    bool loadFromFile(const std::string& filename);
    
    // Get the loaded graph
    const Graph& getGraph() const { return graph; }
    
    // Utility functions
    void printStatistics() const;
    bool isLoaded() const { return is_loaded; }
    
    // Graph validation
    bool validateGraph() const;
    void printSampleEdges(int count = 10) const;
};

#endif // GRAPH_PARSER_H
