#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <osmium/handler.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/visitor.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/osm/types.hpp>
#include <osmium/geom/haversine.hpp>

// Pass 1: Handler to build the ID map
class IDMapHandler : public osmium::handler::Handler {
public:
    std::unordered_map<osmium::object_id_type, uint32_t> id_map;
    uint32_t current_id = 0;

    void way(const osmium::Way& way) {
        // Check if this is a road (look for the "highway" tag)
        if (way.tags().has_key("highway")) {
            for (const auto& node_ref : way.nodes()) {
                if (id_map.find(node_ref.ref()) == id_map.end()) {
                    id_map[node_ref.ref()] = current_id++;
                }
            }
        }
    }
};

// Helper function to calculate haversine distance
double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    const double R = 6371000.0; // Earth's radius in meters
    double dlat = (lat2 - lat1) * M_PI / 180.0;
    double dlon = (lon2 - lon1) * M_PI / 180.0;
    double a = sin(dlat/2) * sin(dlat/2) + cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0) * sin(dlon/2) * sin(dlon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return R * c;
}

// Pass 2: Handler to build the graph
class GraphBuilderHandler : public osmium::handler::Handler {
private:
    std::unordered_map<osmium::object_id_type, uint32_t>& id_map;
    std::vector<std::vector<std::pair<uint32_t, double>>> adjacency_list;

public:
    std::vector<uint32_t> row_pointers;
    std::vector<uint32_t> column_indices;
    std::vector<double> values;

    GraphBuilderHandler(std::unordered_map<osmium::object_id_type, uint32_t>& map_ref)
        : id_map(map_ref) {
        adjacency_list.resize(id_map.size());
    }
    
    void way(const osmium::Way& way) {
        if (way.tags().has_key("highway")) {
            const auto& nodes = way.nodes();
            
            // Process consecutive pairs of nodes
            for (size_t i = 0; i < nodes.size() - 1; ++i) {
                const auto& current_node = nodes[i];
                const auto& next_node = nodes[i + 1];
                
                // Check if both nodes have locations
                if (!current_node.location().valid() || !next_node.location().valid()) {
                    continue;
                }
                
                // Get compact IDs
                auto current_it = id_map.find(current_node.ref());
                auto next_it = id_map.find(next_node.ref());
                
                if (current_it == id_map.end() || next_it == id_map.end()) {
                    continue;
                }
                
                uint32_t current_id = current_it->second;
                uint32_t next_id = next_it->second;
                
                // Calculate haversine distance
                double lat1 = current_node.location().lat();
                double lon1 = current_node.location().lon();
                double lat2 = next_node.location().lat();
                double lon2 = next_node.location().lon();
                
                double distance = haversine_distance(lat1, lon1, lat2, lon2);
                
                // Add bidirectional edges (roads are typically bidirectional)
                adjacency_list[current_id].emplace_back(next_id, distance);
                adjacency_list[next_id].emplace_back(current_id, distance);
            }
        }
    }
    
    void build_csr() {
        row_pointers.resize(adjacency_list.size() + 1);
        row_pointers[0] = 0;
        
        // Count edges for each node
        for (size_t i = 0; i < adjacency_list.size(); ++i) {
            row_pointers[i + 1] = row_pointers[i] + adjacency_list[i].size();
        }
        
        // Reserve space
        column_indices.reserve(row_pointers.back());
        values.reserve(row_pointers.back());
        
        // Fill CSR arrays
        for (size_t i = 0; i < adjacency_list.size(); ++i) {
            for (const auto& edge : adjacency_list[i]) {
                column_indices.push_back(edge.first);
                values.push_back(edge.second);
            }
        }
    }
};

int main(int argc, char* argv[]) {
    // Use the provided filename or default to the dataset file
    std::string filename = "Dataset/new-york-250916.osm.pbf";
    if (argc > 1) {
        filename = argv[1];
    }

    // PASS 1: Create ID map
    std::cout << "Pass 1: Creating node ID map..." << std::endl;
    IDMapHandler id_map_handler;
    osmium::io::Reader reader1(filename);
    osmium::apply(reader1, id_map_handler);
    reader1.close();
    std::cout << "Done. Found " << id_map_handler.id_map.size() << " unique road nodes." << std::endl;

    // PASS 2: Build graph
    std::cout << "Pass 2: Building graph data structures..." << std::endl;
    
    // Create an instance of the graph builder
    GraphBuilderHandler graph_builder(id_map_handler.id_map);
    osmium::io::Reader reader2(filename);
    
    // This is a special handler needed to get node locations
    using storage_type = osmium::index::map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location>;
    using node_location_handler_type = osmium::handler::NodeLocationsForWays<storage_type>;
    storage_type storage;
    node_location_handler_type location_handler(storage);
    osmium::apply(reader2, location_handler, graph_builder);
    
    reader2.close();
    
    // Build CSR representation
    std::cout << "Building CSR representation..." << std::endl;
    graph_builder.build_csr();
    std::cout << "Done. Graph has " << graph_builder.column_indices.size() << " edges." << std::endl;

    // SAVE THE CSR ARRAYS TO A BINARY FILE
    // File format: [num_nodes][num_edges][row_pointers][column_indices][values]
    std::cout << "Writing graph to nyc_graph.bin..." << std::endl;
    std::ofstream outfile("nyc_graph.bin", std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not open output file nyc_graph.bin" << std::endl;
        return 1;
    }
    
    // Write the sizes first
    uint32_t num_nodes = graph_builder.row_pointers.size() - 1;
    uint32_t num_edges = graph_builder.column_indices.size();
    outfile.write(reinterpret_cast<const char*>(&num_nodes), sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(&num_edges), sizeof(uint32_t));
    
    // Write row_pointers
    outfile.write(reinterpret_cast<const char*>(graph_builder.row_pointers.data()), 
                  graph_builder.row_pointers.size() * sizeof(uint32_t));
    
    // Write column_indices
    outfile.write(reinterpret_cast<const char*>(graph_builder.column_indices.data()), 
                  graph_builder.column_indices.size() * sizeof(uint32_t));
    
    // Write values (edge weights)
    outfile.write(reinterpret_cast<const char*>(graph_builder.values.data()), 
                  graph_builder.values.size() * sizeof(double));
    
    outfile.close();
    
    std::cout << "Graph successfully saved to nyc_graph.bin" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Nodes: " << num_nodes << std::endl;
    std::cout << "  Edges: " << num_edges << std::endl;
    std::cout << "  File size: " << (2 * sizeof(uint32_t) + 
                                     graph_builder.row_pointers.size() * sizeof(uint32_t) +
                                     graph_builder.column_indices.size() * sizeof(uint32_t) +
                                     graph_builder.values.size() * sizeof(double)) << " bytes" << std::endl;

    return 0;
}