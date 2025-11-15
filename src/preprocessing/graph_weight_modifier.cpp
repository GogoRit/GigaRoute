#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "../common/graph_parser.h"

/**
 * Graph Weight Modifier - Post-processing tool
 * 
 * Modifies edge weights in an existing binary graph to create variable-weight
 * distribution for delta-stepping algorithm testing.
 * 
 * This tool reads an existing graph.bin file and applies weight multipliers
 * to create a variable-weight graph without requiring OSM data.
 * 
 * Strategy: Randomly assign edges as "highways" (long segments) vs "local roads"
 */

struct WeightStatistics {
    double min_weight;
    double max_weight;
    double mean_weight;
    double median_weight;
    double std_dev;
    double weight_range_ratio;
    std::vector<size_t> histogram;
};

WeightStatistics computeStatistics(const std::vector<double>& weights) {
    WeightStatistics stats;
    if (weights.empty()) return stats;
    
    std::vector<double> sorted = weights;
    std::sort(sorted.begin(), sorted.end());
    
    stats.min_weight = sorted.front();
    stats.max_weight = sorted.back();
    stats.weight_range_ratio = stats.max_weight / stats.min_weight;
    
    // Mean
    double sum = 0.0;
    for (double w : weights) sum += w;
    stats.mean_weight = sum / weights.size();
    
    // Median
    size_t n = sorted.size();
    stats.median_weight = (n % 2 == 0) 
        ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 
        : sorted[n/2];
    
    // Standard deviation
    double variance = 0.0;
    for (double w : weights) {
        variance += (w - stats.mean_weight) * (w - stats.mean_weight);
    }
    stats.std_dev = sqrt(variance / weights.size());
    
    // Histogram (10 bins)
    stats.histogram.resize(10, 0);
    double bin_width = (stats.max_weight - stats.min_weight) / 10.0;
    if (bin_width > 0) {
        for (double w : weights) {
            int bin = std::min(9, static_cast<int>((w - stats.min_weight) / bin_width));
            stats.histogram[bin]++;
        }
    }
    
    return stats;
}

void printStatistics(const WeightStatistics& stats, size_t total_edges, const std::string& name) {
    std::cout << "\n=== Weight Distribution: " << name << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Min weight: " << stats.min_weight << " meters" << std::endl;
    std::cout << "Max weight: " << stats.max_weight << " meters" << std::endl;
    std::cout << "Mean weight: " << stats.mean_weight << " meters" << std::endl;
    std::cout << "Median weight: " << stats.median_weight << " meters" << std::endl;
    std::cout << "Std deviation: " << stats.std_dev << " meters" << std::endl;
    std::cout << "Weight range ratio (max/min): " << stats.weight_range_ratio << "x" << std::endl;
    
    std::cout << "\nWeight Distribution Histogram:" << std::endl;
    double bin_width = (stats.max_weight - stats.min_weight) / 10.0;
    for (size_t i = 0; i < stats.histogram.size(); ++i) {
        double bin_start = stats.min_weight + i * bin_width;
        double bin_end = stats.min_weight + (i + 1) * bin_width;
        double percentage = 100.0 * stats.histogram[i] / total_edges;
        std::cout << "  [" << std::setw(10) << bin_start << " - " << std::setw(10) << bin_end 
                  << "]: " << std::setw(8) << stats.histogram[i] 
                  << " edges (" << std::setprecision(1) << percentage << "%)" << std::endl;
    }
    
    std::cout << "\n=== Research Interpretation ===" << std::endl;
    if (stats.weight_range_ratio > 100.0) {
        std::cout << "HIGH VARIABILITY: Weight range > 100x - Delta-stepping should excel" << std::endl;
    } else if (stats.weight_range_ratio > 10.0) {
        std::cout << "MODERATE VARIABILITY: Weight range 10-100x - Delta-stepping may help" << std::endl;
    } else {
        std::cout << "LOW VARIABILITY: Weight range < 10x - Work-list SSSP likely optimal" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Graph Weight Modifier" << std::endl;
    std::cout << "Create Variable-Weight Graph for Testing" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_graph.bin> <output_graph.bin> [highway_percentage] [highway_multiplier]" << std::endl;
        std::cerr << "  input_graph.bin: Existing graph file to modify" << std::endl;
        std::cerr << "  output_graph.bin: Output file for variable-weight graph" << std::endl;
        std::cerr << "  highway_percentage: Percentage of edges to treat as highways (default: 10%)" << std::endl;
        std::cerr << "  highway_multiplier: Weight multiplier for highways (default: 50.0)" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " nyc_graph.bin nyc_graph_variable_weights.bin 10 50.0" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    double highway_percentage = (argc > 3) ? std::stod(argv[3]) : 10.0;
    double highway_multiplier = (argc > 4) ? std::stod(argv[4]) : 50.0;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input: " << input_file << std::endl;
    std::cout << "  Output: " << output_file << std::endl;
    std::cout << "  Highway percentage: " << highway_percentage << "%" << std::endl;
    std::cout << "  Highway multiplier: " << highway_multiplier << "x" << std::endl;
    std::cout << std::endl;
    
    // Load existing graph
    std::cout << "Loading graph from " << input_file << "..." << std::endl;
    GraphParser parser;
    if (!parser.loadFromFile(input_file)) {
        std::cerr << "Error: Failed to load graph" << std::endl;
        return 1;
    }
    
    const Graph& graph = parser.getGraph();
    std::cout << "Loaded: " << graph.num_nodes << " nodes, " << graph.num_edges << " edges" << std::endl;
    
    // Compute original statistics
    std::vector<double> original_weights(graph.values.begin(), graph.values.end());
    WeightStatistics original_stats = computeStatistics(original_weights);
    printStatistics(original_stats, graph.num_edges, "Original Graph");
    
    // Create modified weights
    std::cout << "\nApplying weight modifications..." << std::endl;
    std::vector<double> modified_weights(graph.values.begin(), graph.values.end());
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    size_t highway_count = 0;
    size_t local_count = 0;
    
    // Apply multipliers: randomly select edges as "highways"
    for (size_t i = 0; i < modified_weights.size(); ++i) {
        if (dis(gen) < highway_percentage) {
            // This edge is a "highway" - apply multiplier
            modified_weights[i] *= highway_multiplier;
            highway_count++;
        } else {
            local_count++;
        }
    }
    
    std::cout << "Modified " << highway_count << " edges as highways (" 
              << (100.0 * highway_count / modified_weights.size()) << "%)" << std::endl;
    std::cout << "Kept " << local_count << " edges as local roads" << std::endl;
    
    // Compute modified statistics
    WeightStatistics modified_stats = computeStatistics(modified_weights);
    printStatistics(modified_stats, graph.num_edges, "Modified Graph (Variable-Weight)");
    
    // Save modified graph
    std::cout << "\nSaving modified graph to " << output_file << "..." << std::endl;
    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not open output file" << std::endl;
        return 1;
    }
    
    // Write header
    outfile.write(reinterpret_cast<const char*>(&graph.num_nodes), sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(&graph.num_edges), sizeof(uint32_t));
    
    // Write row_pointers and column_indices (unchanged)
    outfile.write(reinterpret_cast<const char*>(graph.row_pointers.data()), 
                  (graph.num_nodes + 1) * sizeof(uint32_t));
    outfile.write(reinterpret_cast<const char*>(graph.column_indices.data()), 
                  graph.num_edges * sizeof(uint32_t));
    
    // Write modified values (as double, matching original format)
    std::vector<double> double_values(modified_weights.begin(), modified_weights.end());
    outfile.write(reinterpret_cast<const char*>(double_values.data()), 
                  graph.num_edges * sizeof(double));
    
    outfile.close();
    
    std::cout << "Graph successfully saved!" << std::endl;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Next Steps:" << std::endl;
    std::cout << "1. Test algorithms on variable-weight graph:" << std::endl;
    std::cout << "   ./bin/gpu_dijkstra " << output_file << " 0 1000" << std::endl;
    std::cout << "   ./bin/delta_stepping " << output_file << " 0 1000 500" << std::endl;
    std::cout << "2. Compare with original uniform-weight graph" << std::endl;
    std::cout << "3. Analyze performance differences" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

