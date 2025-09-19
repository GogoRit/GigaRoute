#include <iostream>
#include <vector>
#include <fstream>
#include <queue>
#include <limits>
#include <chrono>
#include <algorithm>

class GraphParser {
public:
    std::vector<uint32_t> row_pointers;
    std::vector<uint32_t> column_indices;
    std::vector<double> values;
    uint32_t num_nodes;
    uint32_t num_edges;
    
    bool load_graph(const std::string& filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }
        
        // Read the header information
        infile.read(reinterpret_cast<char*>(&num_nodes), sizeof(uint32_t));
        infile.read(reinterpret_cast<char*>(&num_edges), sizeof(uint32_t));
        
        std::cout << "Loading graph with " << num_nodes << " nodes and " << num_edges << " edges..." << std::endl;
        
        // Resize vectors
        row_pointers.resize(num_nodes + 1);
        column_indices.resize(num_edges);
        values.resize(num_edges);
        
        // Read row_pointers
        infile.read(reinterpret_cast<char*>(row_pointers.data()), 
                   row_pointers.size() * sizeof(uint32_t));
        
        // Read column_indices
        infile.read(reinterpret_cast<char*>(column_indices.data()), 
                   column_indices.size() * sizeof(uint32_t));
        
        // Read values (edge weights)
        infile.read(reinterpret_cast<char*>(values.data()), 
                   values.size() * sizeof(double));
        
        infile.close();
        
        // Verify data integrity
        if (row_pointers[num_nodes] != num_edges) {
            std::cerr << "Error: Data integrity check failed. Expected " << num_edges 
                      << " edges but row_pointers indicates " << row_pointers[num_nodes] << std::endl;
            return false;
        }
        
        std::cout << "Graph loaded successfully!" << std::endl;
        return true;
    }
    
    void print_graph_stats() {
        if (num_nodes == 0) {
            std::cout << "No graph loaded." << std::endl;
            return;
        }
        
        std::cout << "\n=== Graph Statistics ===" << std::endl;
        std::cout << "Nodes: " << num_nodes << std::endl;
        std::cout << "Edges: " << num_edges << std::endl;
        
        // Calculate average degree
        double avg_degree = static_cast<double>(num_edges) / num_nodes;
        std::cout << "Average degree: " << avg_degree << std::endl;
        
        // Find min/max degree
        uint32_t min_degree = std::numeric_limits<uint32_t>::max();
        uint32_t max_degree = 0;
        
        for (uint32_t i = 0; i < num_nodes; ++i) {
            uint32_t degree = row_pointers[i + 1] - row_pointers[i];
            min_degree = std::min(min_degree, degree);
            max_degree = std::max(max_degree, degree);
        }
        
        std::cout << "Min degree: " << min_degree << std::endl;
        std::cout << "Max degree: " << max_degree << std::endl;
        
        // Sample edge weights statistics
        if (!values.empty()) {
            auto minmax_weights = std::minmax_element(values.begin(), values.end());
            std::cout << "Min edge weight: " << *minmax_weights.first << " meters" << std::endl;
            std::cout << "Max edge weight: " << *minmax_weights.second << " meters" << std::endl;
        }
    }
    
    // Get neighbors of a node
    std::vector<std::pair<uint32_t, double>> get_neighbors(uint32_t node) const {
        std::vector<std::pair<uint32_t, double>> neighbors;
        if (node >= num_nodes) return neighbors;
        
        uint32_t start = row_pointers[node];
        uint32_t end = row_pointers[node + 1];
        
        for (uint32_t i = start; i < end; ++i) {
            neighbors.emplace_back(column_indices[i], values[i]);
        }
        
        return neighbors;
    }
};

class DijkstraPathfinder {
private:
    const GraphParser& graph;
    
public:
    DijkstraPathfinder(const GraphParser& g) : graph(g) {}
    
    struct PathResult {
        std::vector<uint32_t> path;
        double total_distance;
        bool path_found;
        int nodes_visited;
        double computation_time_ms;
    };
    
    PathResult find_shortest_path(uint32_t start, uint32_t end) {
        PathResult result;
        result.path_found = false;
        result.total_distance = std::numeric_limits<double>::infinity();
        result.nodes_visited = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (start >= graph.num_nodes || end >= graph.num_nodes) {
            std::cerr << "Error: Invalid start or end node." << std::endl;
            return result;
        }
        
        if (start == end) {
            result.path = {start};
            result.total_distance = 0.0;
            result.path_found = true;
            return result;
        }
        
        // Distance array
        std::vector<double> dist(graph.num_nodes, std::numeric_limits<double>::infinity());
        std::vector<uint32_t> previous(graph.num_nodes, std::numeric_limits<uint32_t>::max());
        std::vector<bool> visited(graph.num_nodes, false);
        
        // Priority queue: (distance, node)
        std::priority_queue<std::pair<double, uint32_t>, 
                           std::vector<std::pair<double, uint32_t>>,
                           std::greater<>> pq;
        
        // Initialize
        dist[start] = 0.0;
        pq.emplace(0.0, start);
        
        while (!pq.empty()) {
            auto [current_dist, current_node] = pq.top();
            pq.pop();
            
            if (visited[current_node]) continue;
            visited[current_node] = true;
            result.nodes_visited++;
            
            // Early termination if we reached the target
            if (current_node == end) {
                result.path_found = true;
                result.total_distance = dist[end];
                break;
            }
            
            // Explore neighbors
            uint32_t start_idx = graph.row_pointers[current_node];
            uint32_t end_idx = graph.row_pointers[current_node + 1];
            
            for (uint32_t i = start_idx; i < end_idx; ++i) {
                uint32_t neighbor = graph.column_indices[i];
                double edge_weight = graph.values[i];
                
                if (!visited[neighbor]) {
                    double new_dist = dist[current_node] + edge_weight;
                    
                    if (new_dist < dist[neighbor]) {
                        dist[neighbor] = new_dist;
                        previous[neighbor] = current_node;
                        pq.emplace(new_dist, neighbor);
                    }
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // Reconstruct path if found
        if (result.path_found) {
            std::vector<uint32_t> path;
            uint32_t current = end;
            
            while (current != std::numeric_limits<uint32_t>::max()) {
                path.push_back(current);
                current = previous[current];
            }
            
            std::reverse(path.begin(), path.end());
            result.path = std::move(path);
        }
        
        return result;
    }
    
    void print_path_result(const PathResult& result, uint32_t start, uint32_t end) {
        std::cout << "\n=== Path Finding Result ===" << std::endl;
        std::cout << "Start node: " << start << std::endl;
        std::cout << "End node: " << end << std::endl;
        std::cout << "Path found: " << (result.path_found ? "Yes" : "No") << std::endl;
        
        if (result.path_found) {
            std::cout << "Total distance: " << result.total_distance << " meters" << std::endl;
            std::cout << "Distance in kilometers: " << result.total_distance / 1000.0 << " km" << std::endl;
            std::cout << "Path length (nodes): " << result.path.size() << std::endl;
            std::cout << "Nodes visited during search: " << result.nodes_visited << std::endl;
            std::cout << "Computation time: " << result.computation_time_ms << " ms" << std::endl;
            
            if (result.path.size() <= 20) {
                std::cout << "Path: ";
                for (size_t i = 0; i < result.path.size(); ++i) {
                    if (i > 0) std::cout << " -> ";
                    std::cout << result.path[i];
                }
                std::cout << std::endl;
            } else {
                std::cout << "Path (first 10 and last 10 nodes): ";
                for (int i = 0; i < 10; ++i) {
                    if (i > 0) std::cout << " -> ";
                    std::cout << result.path[i];
                }
                std::cout << " ... ";
                for (size_t i = result.path.size() - 10; i < result.path.size(); ++i) {
                    std::cout << " -> " << result.path[i];
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "No path exists between these nodes." << std::endl;
            std::cout << "Nodes visited during search: " << result.nodes_visited << std::endl;
            std::cout << "Computation time: " << result.computation_time_ms << " ms" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    std::string graph_file = "nyc_graph.bin";
    if (argc > 1) {
        graph_file = argv[1];
    }
    
    // Load the graph
    GraphParser parser;
    if (!parser.load_graph(graph_file)) {
        return 1;
    }
    
    parser.print_graph_stats();
    
    // Create pathfinder
    DijkstraPathfinder pathfinder(parser);
    
    // Interactive mode or command line arguments
    if (argc >= 4) {
        // Command line mode: ./dijkstra graph_file start_node end_node
        uint32_t start = std::stoul(argv[2]);
        uint32_t end = std::stoul(argv[3]);
        
        auto result = pathfinder.find_shortest_path(start, end);
        pathfinder.print_path_result(result, start, end);
    } else {
        // Interactive mode
        std::cout << "\n=== Interactive Path Finding ===" << std::endl;
        std::cout << "Enter start and end node IDs (0 to " << (parser.num_nodes - 1) << ")" << std::endl;
        std::cout << "Enter -1 for either value to quit." << std::endl;
        
        // Test with some sample nodes first
        std::cout << "\nRunning test queries..." << std::endl;
        
        // Test 1: Close nodes
        std::cout << "\nTest 1: Close nodes (0 -> 100)" << std::endl;
        auto result1 = pathfinder.find_shortest_path(0, 100);
        pathfinder.print_path_result(result1, 0, 100);
        
        // Test 2: Random nodes
        uint32_t random_start = parser.num_nodes / 4;
        uint32_t random_end = parser.num_nodes / 2;
        std::cout << "\nTest 2: Random nodes (" << random_start << " -> " << random_end << ")" << std::endl;
        auto result2 = pathfinder.find_shortest_path(random_start, random_end);
        pathfinder.print_path_result(result2, random_start, random_end);
        
        // Interactive loop
        while (true) {
            std::cout << "\nEnter start node ID: ";
            int start;
            std::cin >> start;
            
            if (start == -1) break;
            
            std::cout << "Enter end node ID: ";
            int end;
            std::cin >> end;
            
            if (end == -1) break;
            
            if (start < 0 || end < 0 || 
                static_cast<uint32_t>(start) >= parser.num_nodes || 
                static_cast<uint32_t>(end) >= parser.num_nodes) {
                std::cout << "Invalid node IDs. Please enter values between 0 and " 
                          << (parser.num_nodes - 1) << std::endl;
                continue;
            }
            
            auto result = pathfinder.find_shortest_path(static_cast<uint32_t>(start), 
                                                       static_cast<uint32_t>(end));
            pathfinder.print_path_result(result, static_cast<uint32_t>(start), 
                                        static_cast<uint32_t>(end));
        }
    }
    
    std::cout << "\nProgram finished." << std::endl;
    return 0;
}
