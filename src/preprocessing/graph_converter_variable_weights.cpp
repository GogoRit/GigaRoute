#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>
#include <osmium/handler.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/visitor.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/osm/types.hpp>
#include <osmium/geom/haversine.hpp>

/**
 * Professional Variable-Weight Graph Generator for Delta-Stepping Algorithm Testing
 * 
 * This tool creates graphs with variable edge weights to test delta-stepping algorithm
 * performance on graphs where it theoretically excels (highway networks with mixed
 * road types).
 * 
 * Research Purpose:
 * - Validate delta-stepping performance on variable-weight graphs
 * - Compare with uniform-weight graphs (NYC baseline)
 * - Identify graph characteristics where delta-stepping outperforms work-list SSSP
 */

// Highway type classification and weight multipliers
struct HighwayConfig {
    std::string highway_type;
    double weight_multiplier;  // Multiplier for edge weight
    std::string description;
};

// Professional highway classification based on OSM highway tags
// Multipliers simulate "effective distance" - highways are faster but represent longer segments
const std::vector<HighwayConfig> HIGHWAY_CONFIGS = {
    // Major highways (long segments, high speed)
    {"motorway", 50.0, "Freeway/Motorway - Long segments, high speed"},
    {"motorway_link", 30.0, "Motorway ramp - Medium segments"},
    {"trunk", 40.0, "Trunk road - Major highway"},
    {"trunk_link", 25.0, "Trunk road ramp"},
    
    // Primary roads (medium segments)
    {"primary", 10.0, "Primary road - Major arterial"},
    {"primary_link", 8.0, "Primary road ramp"},
    {"secondary", 5.0, "Secondary road - Regional"},
    {"secondary_link", 4.0, "Secondary road ramp"},
    
    // Local roads (short segments, original weight)
    {"tertiary", 2.0, "Tertiary road - Local arterial"},
    {"tertiary_link", 1.5, "Tertiary road ramp"},
    {"residential", 1.0, "Residential street - Original weight"},
    {"service", 1.0, "Service road - Original weight"},
    {"unclassified", 1.0, "Unclassified - Original weight"},
    {"living_street", 1.0, "Living street - Original weight"},
    {"pedestrian", 0.5, "Pedestrian path - Shorter effective distance"}
};

// Weight distribution statistics
struct WeightStatistics {
    double min_weight;
    double max_weight;
    double mean_weight;
    double median_weight;
    double std_dev;
    double weight_range_ratio;  // max/min ratio
    std::vector<size_t> weight_distribution;  // Histogram bins
};

class WeightAnalyzer {
private:
    std::vector<double> weights;
    
public:
    void addWeight(double weight) {
        weights.push_back(weight);
    }
    
    WeightStatistics computeStatistics() const {
        WeightStatistics stats;
        if (weights.empty()) {
            return stats;
        }
        
        std::vector<double> sorted_weights = weights;
        std::sort(sorted_weights.begin(), sorted_weights.end());
        
        stats.min_weight = sorted_weights.front();
        stats.max_weight = sorted_weights.back();
        stats.weight_range_ratio = stats.max_weight / stats.min_weight;
        
        // Mean
        double sum = 0.0;
        for (double w : weights) {
            sum += w;
        }
        stats.mean_weight = sum / weights.size();
        
        // Median
        size_t n = sorted_weights.size();
        if (n % 2 == 0) {
            stats.median_weight = (sorted_weights[n/2 - 1] + sorted_weights[n/2]) / 2.0;
        } else {
            stats.median_weight = sorted_weights[n/2];
        }
        
        // Standard deviation
        double variance = 0.0;
        for (double w : weights) {
            variance += (w - stats.mean_weight) * (w - stats.mean_weight);
        }
        stats.std_dev = sqrt(variance / weights.size());
        
        // Histogram (10 bins)
        stats.weight_distribution.resize(10, 0);
        double bin_width = (stats.max_weight - stats.min_weight) / 10.0;
        for (double w : weights) {
            int bin = std::min(9, static_cast<int>((w - stats.min_weight) / bin_width));
            stats.weight_distribution[bin]++;
        }
        
        return stats;
    }
    
    size_t getCount() const { return weights.size(); }
    
private:
    // Helper to access weights for histogram
    const std::vector<double>& getWeights() const { return weights; }
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

// Get weight multiplier for highway type
double getHighwayMultiplier(const std::string& highway_type) {
    for (const auto& config : HIGHWAY_CONFIGS) {
        if (config.highway_type == highway_type) {
            return config.weight_multiplier;
        }
    }
    // Default: treat as local road
    return 1.0;
}

// Pass 1: Handler to build the ID map
class IDMapHandler : public osmium::handler::Handler {
public:
    std::unordered_map<osmium::object_id_type, uint32_t> id_map;
    uint32_t current_id = 0;

    void way(const osmium::Way& way) {
        if (way.tags().has_key("highway")) {
            for (const auto& node_ref : way.nodes()) {
                if (id_map.find(node_ref.ref()) == id_map.end()) {
                    id_map[node_ref.ref()] = current_id++;
                }
            }
        }
    }
};

// Pass 2: Handler to build the graph with variable weights
class VariableWeightGraphBuilder : public osmium::handler::Handler {
private:
    std::unordered_map<osmium::object_id_type, uint32_t>& id_map;
    std::vector<std::vector<std::pair<uint32_t, double>>> adjacency_list;
    bool apply_variable_weights;
    WeightAnalyzer weight_analyzer;
    std::unordered_map<std::string, size_t> highway_type_counts;

public:
    std::vector<uint32_t> row_pointers;
    std::vector<uint32_t> column_indices;
    std::vector<double> values;

    VariableWeightGraphBuilder(std::unordered_map<osmium::object_id_type, uint32_t>& map_ref,
                               bool variable_weights = true)
        : id_map(map_ref), apply_variable_weights(variable_weights) {
        adjacency_list.resize(id_map.size());
    }
    
    void way(const osmium::Way& way) {
        if (!way.tags().has_key("highway")) {
            return;
        }
        
        const std::string highway_type = way.tags().get_value_by_key("highway");
        double multiplier = apply_variable_weights ? getHighwayMultiplier(highway_type) : 1.0;
        
        // Track highway type distribution
        highway_type_counts[highway_type]++;
        
        const auto& nodes = way.nodes();
        
        for (size_t i = 0; i < nodes.size() - 1; ++i) {
            const auto& current_node = nodes[i];
            const auto& next_node = nodes[i + 1];
            
            if (!current_node.location().valid() || !next_node.location().valid()) {
                continue;
            }
            
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
            
            // Apply highway type multiplier for variable-weight graphs
            double final_weight = distance * multiplier;
            
            // Track weight for analysis
            weight_analyzer.addWeight(final_weight);
            
            // Add bidirectional edges
            adjacency_list[current_id].emplace_back(next_id, final_weight);
            adjacency_list[next_id].emplace_back(current_id, final_weight);
        }
    }
    
    void build_csr() {
        row_pointers.resize(adjacency_list.size() + 1);
        row_pointers[0] = 0;
        
        for (size_t i = 0; i < adjacency_list.size(); ++i) {
            row_pointers[i + 1] = row_pointers[i] + adjacency_list[i].size();
        }
        
        column_indices.reserve(row_pointers.back());
        values.reserve(row_pointers.back());
        
        for (size_t i = 0; i < adjacency_list.size(); ++i) {
            for (const auto& edge : adjacency_list[i]) {
                column_indices.push_back(edge.first);
                values.push_back(edge.second);
            }
        }
    }
    
    WeightStatistics getWeightStatistics() const {
        return weight_analyzer.computeStatistics();
    }
    
    std::unordered_map<std::string, size_t> getHighwayTypeCounts() const {
        return highway_type_counts;
    }
};

void printWeightStatistics(const WeightStatistics& stats, size_t total_edges, const std::string& graph_name) {
    std::cout << "\n=== Weight Distribution Analysis: " << graph_name << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Min weight: " << stats.min_weight << " meters" << std::endl;
    std::cout << "Max weight: " << stats.max_weight << " meters" << std::endl;
    std::cout << "Mean weight: " << stats.mean_weight << " meters" << std::endl;
    std::cout << "Median weight: " << stats.median_weight << " meters" << std::endl;
    std::cout << "Std deviation: " << stats.std_dev << " meters" << std::endl;
    std::cout << "Weight range ratio (max/min): " << stats.weight_range_ratio << "x" << std::endl;
    
    std::cout << "\nWeight Distribution Histogram:" << std::endl;
    double bin_width = (stats.max_weight - stats.min_weight) / 10.0;
    for (size_t i = 0; i < stats.weight_distribution.size(); ++i) {
        double bin_start = stats.min_weight + i * bin_width;
        double bin_end = stats.min_weight + (i + 1) * bin_width;
        double percentage = 100.0 * stats.weight_distribution[i] / total_edges;
        std::cout << "  [" << std::setw(10) << bin_start << " - " << std::setw(10) << bin_end 
                  << "]: " << std::setw(8) << stats.weight_distribution[i] 
                  << " edges (" << std::setprecision(1) << percentage << "%)" << std::endl;
    }
    
    // Research interpretation
    std::cout << "\n=== Research Interpretation ===" << std::endl;
    if (stats.weight_range_ratio > 100.0) {
        std::cout << "HIGH VARIABILITY: Weight range > 100x - Delta-stepping should excel" << std::endl;
    } else if (stats.weight_range_ratio > 10.0) {
        std::cout << "MODERATE VARIABILITY: Weight range 10-100x - Delta-stepping may help" << std::endl;
    } else {
        std::cout << "LOW VARIABILITY: Weight range < 10x - Work-list SSSP likely optimal" << std::endl;
    }
}

void printHighwayTypeDistribution(const std::unordered_map<std::string, size_t>& counts) {
    std::cout << "\n=== Highway Type Distribution ===" << std::endl;
    std::vector<std::pair<std::string, size_t>> sorted_counts(counts.begin(), counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (const auto& pair : sorted_counts) {
        double multiplier = getHighwayMultiplier(pair.first);
        std::cout << "  " << std::setw(20) << pair.first << ": " << std::setw(10) << pair.second 
                  << " segments (multiplier: " << multiplier << "x)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Variable-Weight Graph Generator" << std::endl;
    std::cout << "For Delta-Stepping Algorithm Testing" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Parse command line arguments
    std::string input_file = "Dataset/new-york-250916.osm.pbf";
    std::string output_file = "nyc_graph_variable_weights.bin";
    bool variable_weights = true;
    
    if (argc > 1) {
        input_file = argv[1];
    }
    if (argc > 2) {
        output_file = argv[2];
    }
    if (argc > 3) {
        std::string mode = argv[3];
        variable_weights = (mode == "variable" || mode == "var" || mode == "1");
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input file: " << input_file << std::endl;
    std::cout << "  Output file: " << output_file << std::endl;
    std::cout << "  Weight mode: " << (variable_weights ? "VARIABLE (highway-based)" : "UNIFORM (baseline)") << std::endl;
    std::cout << std::endl;
    
    // PASS 1: Create ID map
    std::cout << "Pass 1: Creating node ID map..." << std::endl;
    IDMapHandler id_map_handler;
    osmium::io::Reader reader1(input_file);
    osmium::apply(reader1, id_map_handler);
    reader1.close();
    std::cout << "Done. Found " << id_map_handler.id_map.size() << " unique road nodes." << std::endl;

    // PASS 2: Build graph with variable weights
    std::cout << "\nPass 2: Building graph with " << (variable_weights ? "variable" : "uniform") << " weights..." << std::endl;
    
    VariableWeightGraphBuilder graph_builder(id_map_handler.id_map, variable_weights);
    osmium::io::Reader reader2(input_file);
    
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

    // Analyze weight distribution
    WeightStatistics stats = graph_builder.getWeightStatistics();
    uint32_t num_edges = graph_builder.column_indices.size();
    printWeightStatistics(stats, num_edges, output_file);
    printHighwayTypeDistribution(graph_builder.getHighwayTypeCounts());

    // Save the graph
    std::cout << "\nWriting graph to " << output_file << "..." << std::endl;
    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }
    
    uint32_t num_nodes = graph_builder.row_pointers.size() - 1;
    uint32_t num_edges = graph_builder.column_indices.size();
    outfile.write(reinterpret_cast<const char*>(&num_nodes), sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(&num_edges), sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(graph_builder.row_pointers.data()), 
                  graph_builder.row_pointers.size() * sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(graph_builder.column_indices.data()), 
                  graph_builder.column_indices.size() * sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(graph_builder.values.data()), 
                  graph_builder.values.size() * sizeof(double));
    outfile.close();
    
    std::cout << "Graph successfully saved to " << output_file << std::endl;
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Nodes: " << num_nodes << std::endl;
    std::cout << "  Edges: " << num_edges << std::endl;
    std::cout << "  File size: " << (2 * sizeof(uint32_t) + 
                                     graph_builder.row_pointers.size() * sizeof(uint32_t) +
                                     graph_builder.column_indices.size() * sizeof(uint32_t) +
                                     graph_builder.values.size() * sizeof(double)) << " bytes" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Next Steps:" << std::endl;
    std::cout << "1. Test algorithms on this graph:" << std::endl;
    std::cout << "   ./bin/gpu_dijkstra " << output_file << " 0 1000" << std::endl;
    std::cout << "   ./bin/delta_stepping " << output_file << " 0 1000 100" << std::endl;
    std::cout << "2. Compare with uniform-weight baseline" << std::endl;
    std::cout << "3. Analyze performance differences" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

