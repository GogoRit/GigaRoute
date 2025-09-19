#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>

int main() {
    std::ifstream infile("nyc_graph.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Could not open nyc_graph.bin" << std::endl;
        return 1;
    }
    
    uint32_t num_nodes, num_edges;
    infile.read(reinterpret_cast<char*>(&num_nodes), sizeof(uint32_t));
    infile.read(reinterpret_cast<char*>(&num_edges), sizeof(uint32_t));
    
    std::cout << "Graph has " << num_nodes << " nodes and " << num_edges << " edges" << std::endl;
    
    std::vector<uint32_t> row_pointers(num_nodes + 1);
    std::vector<uint32_t> column_indices(num_edges);
    std::vector<double> values(num_edges);
    
    infile.read(reinterpret_cast<char*>(row_pointers.data()), 
               row_pointers.size() * sizeof(uint32_t));
    infile.read(reinterpret_cast<char*>(column_indices.data()), 
               column_indices.size() * sizeof(uint32_t));
    infile.read(reinterpret_cast<char*>(values.data()), 
               values.size() * sizeof(double));
    
    infile.close();
    
    std::cout << "Data loaded successfully!" << std::endl;
    
    // Verification 1: Check if row_pointers are non-decreasing
    bool valid_row_pointers = true;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (row_pointers[i] > row_pointers[i + 1]) {
            valid_row_pointers = false;
            break;
        }
    }
    std::cout << "Row pointers valid: " << (valid_row_pointers ? "Yes" : "No") << std::endl;
    
    // Verification 2: Check if all column indices are within bounds
    bool valid_column_indices = true;
    for (uint32_t i = 0; i < num_edges; ++i) {
        if (column_indices[i] >= num_nodes) {
            valid_column_indices = false;
            std::cout << "Invalid column index: " << column_indices[i] << " at position " << i << std::endl;
            break;
        }
    }
    std::cout << "Column indices valid: " << (valid_column_indices ? "Yes" : "No") << std::endl;
    
    // Verification 3: Check edge weights (should be non-negative)
    bool valid_weights = true;
    double min_weight = values[0], max_weight = values[0];
    uint32_t zero_weights = 0;
    for (uint32_t i = 0; i < num_edges; ++i) {
        if (values[i] < 0) {
            valid_weights = false;
            std::cout << "Negative weight: " << values[i] << " at position " << i << std::endl;
            break;
        }
        if (values[i] == 0.0) zero_weights++;
        min_weight = std::min(min_weight, values[i]);
        max_weight = std::max(max_weight, values[i]);
    }
    std::cout << "Edge weights valid: " << (valid_weights ? "Yes" : "No") << std::endl;
    std::cout << "Weight range: " << min_weight << " to " << max_weight << " meters" << std::endl;
    std::cout << "Zero-weight edges: " << zero_weights << std::endl;
    
    // Verification 4: Check last row pointer
    bool valid_last_pointer = (row_pointers[num_nodes] == num_edges);
    std::cout << "Last row pointer valid: " << (valid_last_pointer ? "Yes" : "No") << std::endl;
    
    // Verification 5: Sample some adjacency lists
    std::cout << "\nSample adjacency lists:" << std::endl;
    for (uint32_t node = 0; node < std::min(5u, num_nodes); ++node) {
        uint32_t start = row_pointers[node];
        uint32_t end = row_pointers[node + 1];
        uint32_t degree = end - start;
        
        std::cout << "Node " << node << " (degree " << degree << "): ";
        for (uint32_t i = start; i < std::min(start + 5, end); ++i) {
            std::cout << column_indices[i] << "(" << values[i] << "m) ";
        }
        if (end - start > 5) std::cout << "...";
        std::cout << std::endl;
    }
    
    // Verification 6: Check for self-loops
    uint32_t self_loops = 0;
    for (uint32_t node = 0; node < num_nodes; ++node) {
        uint32_t start = row_pointers[node];
        uint32_t end = row_pointers[node + 1];
        for (uint32_t i = start; i < end; ++i) {
            if (column_indices[i] == node) {
                self_loops++;
            }
        }
    }
    std::cout << "\nSelf-loops found: " << self_loops << std::endl;
    
    std::cout << "\nGraph verification complete!" << std::endl;
    return 0;
}
