#ifndef DIJKSTRA_CPU_H
#define DIJKSTRA_CPU_H

#include "../common/graph_parser.h"
#include <vector>
#include <queue>
#include <limits>

class DijkstraPathfinder {
private:
    const Graph& graph;
    std::vector<float> distances;
    std::vector<uint32_t> predecessors;
    uint32_t nodes_visited;
    
public:
    explicit DijkstraPathfinder(const Graph& g);
    
    // Find shortest path between source and target
    float findShortestPath(uint32_t source, uint32_t target);
    
    // Get path reconstruction
    std::vector<uint32_t> getPath(uint32_t source, uint32_t target);
    
    // Statistics
    uint32_t getNodesVisited() const { return nodes_visited; }
    
    // Print results
    void printResult(uint32_t source, uint32_t target, float distance, 
                    double time_ms, const std::vector<uint32_t>& path) const;
};

#endif // DIJKSTRA_CPU_H
