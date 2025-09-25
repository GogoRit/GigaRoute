#include "dijkstra_cpu.h"
#include <iostream>
#include <algorithm>
#include <iomanip>

DijkstraPathfinder::DijkstraPathfinder(const Graph& g) : graph(g) {
    distances.resize(graph.num_nodes);
    predecessors.resize(graph.num_nodes);
    nodes_visited = 0;
}

float DijkstraPathfinder::findShortestPath(uint32_t source, uint32_t target) {
    if (source >= graph.num_nodes || target >= graph.num_nodes) {
        std::cerr << "Error: Invalid source or target node" << std::endl;
        return -1.0f;
    }
    
    // Initialize distances and predecessors
    std::fill(distances.begin(), distances.end(), std::numeric_limits<float>::infinity());
    std::fill(predecessors.begin(), predecessors.end(), UINT32_MAX);
    
    distances[source] = 0.0f;
    nodes_visited = 0;
    
    // Priority queue: (distance, node)
    std::priority_queue<std::pair<float, uint32_t>, 
                       std::vector<std::pair<float, uint32_t>>,
                       std::greater<std::pair<float, uint32_t>>> pq;
    
    pq.push({0.0f, source});
    
    while (!pq.empty()) {
        auto [current_dist, current_node] = pq.top();
        pq.pop();
        
        // Skip if we've already found a better path
        if (current_dist > distances[current_node]) {
            continue;
        }
        
        nodes_visited++;
        
        // Early termination if we reached the target
        if (current_node == target) {
            break;
        }
        
        // Explore neighbors
        uint32_t start_edge = graph.row_pointers[current_node];
        uint32_t end_edge = graph.row_pointers[current_node + 1];
        
        for (uint32_t edge_idx = start_edge; edge_idx < end_edge; edge_idx++) {
            uint32_t neighbor = graph.column_indices[edge_idx];
            float edge_weight = graph.values[edge_idx];
            float new_distance = current_dist + edge_weight;
            
            if (new_distance < distances[neighbor]) {
                distances[neighbor] = new_distance;
                predecessors[neighbor] = current_node;
                pq.push({new_distance, neighbor});
            }
        }
    }
    
    return distances[target];
}

std::vector<uint32_t> DijkstraPathfinder::getPath(uint32_t source, uint32_t target) {
    std::vector<uint32_t> path;
    
    if (distances[target] == std::numeric_limits<float>::infinity()) {
        return path; // No path found
    }
    
    // Reconstruct path
    uint32_t current = target;
    while (current != UINT32_MAX) {
        path.push_back(current);
        current = predecessors[current];
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

void DijkstraPathfinder::printResult(uint32_t source, uint32_t target, float distance, 
                                   double time_ms, const std::vector<uint32_t>& path) const {
    std::cout << "\n=== Path Finding Result ===" << std::endl;
    std::cout << "Start node: " << source << std::endl;
    std::cout << "End node: " << target << std::endl;
    
    if (distance == std::numeric_limits<float>::infinity()) {
        std::cout << "Path found: No" << std::endl;
        return;
    }
    
    std::cout << "Path found: Yes" << std::endl;
    std::cout << "Total distance: " << std::fixed << std::setprecision(1) 
              << distance << " meters" << std::endl;
    std::cout << "Distance in kilometers: " << std::fixed << std::setprecision(4) 
              << distance / 1000.0f << " km" << std::endl;
    std::cout << "Path length (nodes): " << path.size() << std::endl;
    std::cout << "Nodes visited during search: " << nodes_visited << std::endl;
    std::cout << "Computation time: " << std::fixed << std::setprecision(2) 
              << time_ms << " ms" << std::endl;
    
    // Print path preview (first 10 and last 10 nodes)
    if (!path.empty()) {
        std::cout << "Path (first 10 and last 10 nodes): ";
        
        size_t preview_count = std::min(size_t(10), path.size());
        for (size_t i = 0; i < preview_count; i++) {
            std::cout << path[i];
            if (i < preview_count - 1) std::cout << " -> ";
        }
        
        if (path.size() > 20) {
            std::cout << " ... ";
            for (size_t i = path.size() - 10; i < path.size(); i++) {
                std::cout << " -> " << path[i];
            }
        } else if (path.size() > 10) {
            for (size_t i = 10; i < path.size(); i++) {
                std::cout << " -> " << path[i];
            }
        }
        
        std::cout << std::endl;
    }
}
